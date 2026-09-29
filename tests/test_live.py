from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from flyarm.graph import Graph
from flyarm.interfaces import NeuralInterface
from flyarm.live import build_graph_payload, summarize_evidence


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
