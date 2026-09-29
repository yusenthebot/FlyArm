from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from flyarm.graph import Graph
from flyarm.interfaces import NeuralInterface
from flyarm.live import (
    LiveRuntime,
    _approved_ui_dist,
    _validate_server_scope,
    build_graph_payload,
    create_app,
    summarize_evidence,
)


def small_graph() -> Graph:
    return Graph(
        ids=np.array([10, 20, 30], dtype=np.int64),
        pre=np.array([0, 1], dtype=np.int64),
        post=np.array([1, 2], dtype=np.int64),
        contacts=np.array([4, 7], dtype=np.float32),
        signs=np.array([1, -1, 0], dtype=np.float32),
        metadata={"dataset": "MaleCNS v1.0"},
    )


def test_graph_payload_uses_measured_soma_locations(tmp_path: Path) -> None:
    graph = small_graph()
    interface = NeuralInterface.bind(
        graph,
        np.array([10], dtype=np.int64),
        np.array([30], dtype=np.int64),
    )
    annotations = tmp_path / "annotations.feather"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [10, 20, 30],
                "type": ["AN001", "IN001", "DN001"],
                "superclass": ["ascending_neuron", "cb_intrinsic", "descending_neuron"],
                "somaLocation": [[1, 2, 3], [2, 4, 6], [3, 6, 9]],
            }
        ),
        annotations,
    )

    payload = build_graph_payload(graph, interface, annotations)

    assert payload["layout"] == "measured_soma_locations"
    assert [node["role"] for node in payload["nodes"]] == ["input", "internal", "output"]
    assert payload["nodes"][0]["soma_location"] == [1.0, 2.0, 3.0]
    assert payload["edges"][1]["source"] == 20
    assert payload["edges"][1]["target"] == 30


def test_evidence_summary_never_promotes_failed_or_missing_runs() -> None:
    assert summarize_evidence(None)["graph_mediated"] == "pending"
    assert summarize_evidence({"status": "failed", "models": []})["graph_mediated"] == "pending"


def test_evidence_summary_separates_mediation_from_topology() -> None:
    results = {
        "status": "complete",
        "models": [
            *[
                {
                    "kind": "restricted_connectome",
                    "seed": seed,
                    "clean": {"success_rate": 1.0},
                    "edges_silenced": {"success_rate": 0.0},
                }
                for seed in range(3)
            ],
            *[
                {
                    "kind": "restricted_shuffled",
                    "seed": seed,
                    "clean": {"success_rate": 1.0},
                }
                for seed in range(3)
            ],
        ],
    }

    summary = summarize_evidence(results)

    assert summary["graph_mediated"] == "supported"
    assert summary["topology_advantage"] == "not_supported"


def test_evidence_summary_requires_three_matched_seeds_for_positive_claim() -> None:
    results = {
        "status": "complete",
        "models": [
            {
                "kind": "restricted_connectome",
                "seed": 0,
                "clean": {"success_rate": 1.0},
                "edges_silenced": {"success_rate": 0.0},
            },
            {"kind": "restricted_shuffled", "seed": 0, "clean": {"success_rate": 0.0}},
        ],
    }

    summary = summarize_evidence(results)

    assert summary["graph_mediated"] == "pending"
    assert summary["topology_advantage"] == "pending"


@pytest.mark.parametrize(
    "value,name,height",
    [
        ((0.1, 0.0, 0.02), "object", 0.02),
        ((0.4, 0.4, 0.002), "goal", 0.002),
        ((0.4, 0.0, 0.2), "object", 0.02),
    ],
)
def test_interactive_workspace_is_bounded(
    value: tuple[float, float, float], name: str, height: float
) -> None:
    with pytest.raises(ValueError, match="bounded tabletop"):
        LiveRuntime._workspace_vector(value, name, height)


class FakeRuntime:
    def __init__(self) -> None:
        self.running = False

    def snapshot(self) -> dict[str, Any]:
        return {"running": self.running}

    def run(self) -> None:
        self.running = True

    def pause(self) -> None:
        self.running = False

    def step_once(self) -> None:
        return

    def reset(self, request: object) -> None:
        return

    def set_mode(self, mode: object) -> None:
        return

    def close(self) -> None:
        return


def test_live_api_requires_session_and_same_origin(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(
        '<!doctype html><meta name="flyarm-session" content="__FLYARM_SESSION__">'
    )
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime, {}, tmp_path))  # type: ignore[arg-type]

    assert client.post("/api/run").status_code == 401
    root = client.get("/")
    assert root.status_code == 200
    token = root.text.split('content="', 1)[1].split('"', 1)[0]
    assert client.post("/api/run", headers={"origin": "http://evil.invalid"}).status_code == 403
    response = client.post("/api/run", headers={"origin": "http://testserver"})

    assert response.status_code == 200
    assert runtime.running
    client.cookies.clear()
    assert (
        client.post(
            "/api/pause",
            headers={
                "origin": "http://testserver",
                "x-flyarm-session": token,
            },
        ).status_code
        == 200
    )


def test_live_websocket_requires_same_origin(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(
        '<!doctype html><meta name="flyarm-session" content="__FLYARM_SESSION__">'
    )
    client = TestClient(create_app(FakeRuntime(), {}, tmp_path))  # type: ignore[arg-type]
    root = client.get("/")
    token = root.text.split('content="', 1)[1].split('"', 1)[0]

    with pytest.raises(WebSocketDisconnect) as rejected:
        with client.websocket_connect(
            "/ws/state",
            headers={"origin": "http://evil.invalid"},
            subprotocols=["flyarm", token],
        ):
            pass
    assert rejected.value.code == 1008
    with pytest.raises(WebSocketDisconnect) as missing_protocol:
        with client.websocket_connect("/ws/state", headers={"origin": "http://testserver"}):
            pass
    assert missing_protocol.value.code == 1008
    with client.websocket_connect(
        "/ws/state",
        headers={"origin": "http://testserver"},
        subprotocols=["flyarm", token],
    ) as websocket:
        assert websocket.receive_json() == {"running": False}


def test_live_server_rejects_remote_hosts_and_arbitrary_static_roots(tmp_path: Path) -> None:
    approved = _approved_ui_dist()
    assert _validate_server_scope(approved, "127.0.0.1") == approved.resolve()
    with pytest.raises(ValueError, match="loopback"):
        _validate_server_scope(approved, "0.0.0.0")
    with pytest.raises(ValueError, match="project build"):
        _validate_server_scope(tmp_path, "127.0.0.1")


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (
            {
                "is_success": False,
                "ever_lifted": False,
                "ever_grasped": False,
                "contact_left": False,
                "contact_right": False,
            },
            "free",
        ),
        (
            {
                "is_success": False,
                "ever_lifted": False,
                "ever_grasped": True,
                "contact_left": True,
                "contact_right": True,
            },
            "grasped",
        ),
        (
            {
                "is_success": True,
                "ever_lifted": True,
                "ever_grasped": True,
                "contact_left": False,
                "contact_right": False,
            },
            "placed",
        ),
    ],
)
def test_physical_stage_is_not_the_teacher_state(info: dict[str, bool], expected: str) -> None:
    assert LiveRuntime._physical_stage(info) == expected


def test_activity_code_keeps_four_decades_and_signs() -> None:
    from flyarm.live import decode_activity, encode_activity

    values = np.array([0.0, 5e-5, 1e-4, -1e-3, 3.3e-3, -0.02, 0.5, -1.0, 1.7])
    decoded = decode_activity(encode_activity(values))
    assert decoded[0] == 0.0 and decoded[1] == 0.0
    assert np.all(np.sign(decoded[2:]) == np.sign(values[2:]))
    np.testing.assert_allclose(decoded[2:8], values[2:8], rtol=0.04)
    assert decoded[8] == 1.0
