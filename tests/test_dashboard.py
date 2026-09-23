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
    assert family({}, {}, "hand-dexterous-001") == "dexterous"
    assert family({"train_episodes": 96}, {}) == "prototype"


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
    assert detail["media_files"] == [{"path": "kitchen-a/clip.mp4", "title": "clip"}]
    gallery = client.get("/api/gallery").json()
    assert [v["path"] for v in gallery["videos"]] == ["kitchen-a/clip.mp4"]
    assert gallery["videos"][0]["area"] == "Kitchen" and gallery["featured"] == []
    assert client.get("/media/main/kitchen-a/clip.mp4").status_code == 200
    assert client.get("/media/main/secret.txt").status_code == 403
    assert client.get("/api/runs/main/%2E%2E").status_code == 404  # encoded, not normalized
    assert "FlyArm runs" in client.get("/").text


def test_gallery_skips_archives_hides_smoke_and_titles_files(tmp_path: Path) -> None:
    from flyarm.dashboard.catalog import gallery, readable

    root = tmp_path / "runs"
    for relative in (
        "ppo-a/videos/heavier_6_10x-before-after.mp4",
        "_archive/old/clip.mp4",
        "overnight/site/videos/copy.mp4",
        "multitask-smoke-001/videos/connectome-0-push.mp4",
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(b"\x00")
    featured = tmp_path / "featured.json"
    featured.write_text(
        json.dumps(
            [
                {
                    "root": "main",
                    "path": "ppo-a/videos/heavier_6_10x-before-after.mp4",
                    "title": "t",
                },
                {"root": "main", "path": "missing.mp4", "title": "gone"},
            ]
        )
    )
    result = gallery([Root("main", root.resolve())], featured)
    paths = {v["path"]: v for v in result["videos"]}
    assert set(paths) == {
        "ppo-a/videos/heavier_6_10x-before-after.mp4",
        "multitask-smoke-001/videos/connectome-0-push.mp4",
    }
    assert paths["multitask-smoke-001/videos/connectome-0-push.mp4"]["hidden"]
    assert paths["ppo-a/videos/heavier_6_10x-before-after.mp4"]["area"] == "Reinforcement learning"
    assert [f["title"] for f in result["featured"]] == ["t"]
    assert readable("heavier_6_10x-before-after") == "heavier 6-10x · before vs after PPO"
    assert (
        readable("pick-place-shuffled-seed5-r1") == "pick place shuffled CNS · seed 5 · shuffle #2"
    )


def test_live_endpoint_lists_the_newest_progress_clip_of_each_run(tmp_path: Path) -> None:
    from flyarm.dashboard.catalog import Root, live

    run = tmp_path / "runs" / "ppo-demo"
    (run / "progress").mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({"status": "running"}))
    (run / "progress" / "latest.mp4").write_bytes(b"not really a video")
    (run / "progress" / "latest.json").write_text(
        json.dumps(
            {
                "iteration": 120,
                "outcomes": ["placed", "lifted"],
                "recorded_at": "2026-09-22 15:00:00",
                "curve": {"mean_reward": 1.5, "success_rate": 0.25},
            }
        )
    )
    quiet = tmp_path / "runs" / "no-clip"
    quiet.mkdir()
    clips = live([Root(label="main", path=tmp_path / "runs")])["clips"]
    assert [clip["run"] for clip in clips] == ["ppo-demo"]
    assert clips[0]["path"] == "ppo-demo/progress/latest.mp4"
    assert clips[0]["iteration"] == 120 and clips[0]["status"] == "running"
    assert clips[0]["curve"]["mean_reward"] == 1.5
    (run / "results.json").write_text(json.dumps({"status": "stopped"}))
    assert live([Root(label="main", path=tmp_path / "runs")])["clips"] == []
    kept = live([Root(label="main", path=tmp_path / "runs")], include_inactive=True)["clips"]
    assert [clip["status"] for clip in kept] == ["stopped"]
