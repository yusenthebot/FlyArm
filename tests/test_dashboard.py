from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flyarm.dashboard.catalog import Root, family, find_runs, resolve_media, summarize


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def runs(tmp_path: Path) -> Root:
    root = tmp_path / "runs"
    _write(root / "kitchen-a" / "config.json", {"split": "complete"})
    _write(
        root / "kitchen-a" / "results.json",
        {"status": "running", "summary": {"flyleg": {"clean": 25.0, "seeds": [0, 1]}}},
    )
    _write(
        root / "kitchen-a" / "flyleg-0" / "learning.json",
        [
            {"epoch": 1, "train_loss": 0.3, "validation_loss": 0.2, "selection_score": 1.0},
            {"epoch": 2, "train_loss": 0.2, "validation_loss": 0.15},
        ],
    )
    (root / "kitchen-a" / "clip.mp4").write_bytes(b"\x00\x00")
    _write(root / "ppo-b" / "config.json", {"base_run": "runs/x"})
    _write(
        root / "ppo-b" / "results.json",
        {"status": "running", "base": {"nominal": {"successes": 4, "lifts": 18}}},
    )
    _write(
        root / "ppo-b" / "evaluations.json",
        [
            {
                "iteration": 20,
                "env_steps": 100,
                "variants": {"nominal": {"successes": 6, "lifts": 20}},
            }
        ],
    )
    _write(root / "sweep" / "results.json", [{"score": 1}])
    (root / "not-a-run").mkdir()
    (root / "secret.txt").write_text("no")
    return Root("main", root.resolve())


def test_runs_are_found_and_summarized_by_family(runs: Root) -> None:
    names = [path.name for path in find_runs(runs)]
    assert names == ["kitchen-a", "ppo-b", "sweep"]
    kitchen = summarize(runs, runs.path / "kitchen-a")
    assert kitchen["family"] == "kitchen" and kitchen["media"] == 1
    assert kitchen["headline"] == [{"label": "flyleg", "value": "25.0 (2 seeds)"}]
    ppo = summarize(runs, runs.path / "ppo-b")
    assert ppo["headline"][1]["value"].startswith("place 6 lift 20")
    assert family(None, [1]) == "sweep" and family({"task": "pick-place"}, {}) == "whole-brain"


def test_media_resolution_refuses_escapes_and_non_media(runs: Root) -> None:
    assert resolve_media([runs], "main", "kitchen-a/clip.mp4").name == "clip.mp4"
    with pytest.raises(PermissionError):
        resolve_media([runs], "main", "../runs/secret.txt")
    with pytest.raises(PermissionError):
        resolve_media([runs], "main", "secret.txt")
    with pytest.raises(PermissionError):
        resolve_media([runs], "main", "../../etc/hosts.mp4")


def test_api_serves_list_detail_and_media(runs: Root, monkeypatch: pytest.MonkeyPatch) -> None:
    from flyarm.dashboard import server

    monkeypatch.setattr(server, "discover_roots", lambda repo: [runs])
    client = TestClient(server.create_dashboard_app(Path(".")))
    listing = client.get("/api/runs").json()
    assert {run["name"] for run in listing["runs"]} == {"kitchen-a", "ppo-b", "sweep"}
    detail = client.get("/api/runs/main/kitchen-a").json()
    assert {s["metric"] for s in detail["curves"]} >= {"train_loss", "selection_score"}
    assert detail["media_files"] == ["kitchen-a/clip.mp4"]
    assert client.get("/media/main/kitchen-a/clip.mp4").status_code == 200
    assert client.get("/media/main/secret.txt").status_code == 403
    assert client.get("/api/runs/main/%2E%2E").status_code == 404  # encoded, not normalized
    assert "FlyArm runs" in client.get("/").text
