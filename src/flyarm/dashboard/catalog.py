"""Discover experiment runs across the project's git worktrees and summarize them.

A run is a directory under some worktree's runs/ that holds results.json, config.json or
curves.json. Summaries are read-only views of those files: status, timestamps, a headline
per experiment family, learning curves, evaluations, the log tail and rollout media.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_MARKERS = ("results.json", "config.json", "curves.json")
MEDIA_SUFFIXES = {".mp4", ".png", ".jpg", ".gif"}
STALLED_AFTER_SECONDS = 45 * 60
MAX_LOG_LINES = 80


@dataclass(frozen=True)
class Root:
    label: str
    path: Path  # a runs/ directory


def discover_roots(repo: Path) -> list[Root]:
    """The runs/ directory of every worktree of ``repo``, labelled by branch."""
    try:
        listing = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        listing = f"worktree {repo}\n"
    roots: list[Root] = []
    path: Path | None = None
    for line in [*listing.splitlines(), ""]:
        if line.startswith("worktree "):
            path = Path(line.split(" ", 1)[1])
        elif line.startswith("branch ") and path is not None:
            branch = line.split(" ", 1)[1].removeprefix("refs/heads/")
            if (path / "runs").is_dir():
                label = "main" if path.resolve() == repo.resolve() else branch.split("/")[-1]
                roots.append(Root(label, (path / "runs").resolve()))
            path = None
    if (
        not any(root.path == (repo / "runs").resolve() for root in roots)
        and (repo / "runs").is_dir()
    ):
        roots.insert(0, Root("main", (repo / "runs").resolve()))
    return roots


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def find_runs(root: Root) -> list[Path]:
    return sorted(
        child
        for child in root.path.iterdir()
        if child.is_dir() and any((child / marker).is_file() for marker in RUN_MARKERS)
    )


def family(config: Any, results: Any) -> str:
    config = config if isinstance(config, dict) else {}
    if isinstance(results, list):
        return "sweep"
    if "base_run" in config:
        return "rl-ppo"
    if "split" in config:
        return "kitchen"
    if "skills" in config or "skill_weights" in config:
        return "multitask"
    if config.get("task") in ("reach", "pick-place"):
        return "whole-brain"
    return "other"


def _updated(run: Path) -> float:
    stamps = [run.stat().st_mtime]
    stamps += [p.stat().st_mtime for p in run.iterdir() if p.is_file()]
    log = run.with_suffix(".log")
    if log.is_file():
        stamps.append(log.stat().st_mtime)
    return max(stamps)


def _started(run: Path) -> float:
    config = run / "config.json"
    return (config if config.is_file() else run).stat().st_mtime


def _count(models: list[dict[str, Any]], field: str) -> str:
    episodes = [len(m["clean"].get("episodes", [])) for m in models]
    hits = sum(round(m["clean"][field] * n) for m, n in zip(models, episodes, strict=True))
    return f"{hits}/{sum(episodes)}"


def headline(kind: str, results: Any) -> list[dict[str, str]]:
    """A few label/value pairs that say how the run is doing."""
    if not isinstance(results, (dict, list)):
        return []
    if kind == "sweep" and isinstance(results, list):
        return [{"label": "settings run", "value": str(len(results))}]
    assert isinstance(results, dict)
    items: list[dict[str, str]] = []
    if kind == "kitchen":
        for name, entry in (results.get("summary") or {}).items():
            seeds = len(entry.get("seeds", []))
            items.append({"label": name, "value": f"{entry['clean']:.1f} ({seeds} seeds)"})
    elif kind == "multitask":
        for name, entry in (results.get("summary") or {}).items():
            splits = entry.get("evaluation", {})
            value = " · ".join(
                f"{split} {100 * cell['success_rate']:.0f}%" for split, cell in splits.items()
            )
            items.append({"label": name, "value": value})
    elif kind == "whole-brain":
        models = [m for m in results.get("models", []) if "clean" in m]
        for name in dict.fromkeys(m["kind"] for m in models):
            group = [m for m in models if m["kind"] == name]
            fields = [f for f in ("lift_rate", "success_rate") if f in group[0]["clean"]]
            value = " · ".join(
                f"{'lift' if f == 'lift_rate' else 'success'} {_count(group, f)}" for f in fields
            )
            items.append({"label": name, "value": value})
    elif kind == "rl-ppo":
        base, best = results.get("base"), results.get("best")
        if isinstance(base, dict):
            first = next(iter(base.values())) if "successes" not in base else base
            items.append(
                {"label": "base", "value": f"place {first['successes']} lift {first['lifts']}"}
            )
        if isinstance(best, dict) and "successes" in best:
            items.append(
                {
                    "label": "best",
                    "value": f"place {best['successes']} lift {best['lifts']} "
                    f"(it {best.get('iteration', '?')})",
                }
            )
    return items


def safe_headline(kind: str, results: Any) -> list[dict[str, str]]:
    """Headline extraction never breaks the listing: a new result layout shows nothing."""
    try:
        return headline(kind, results)
    except (KeyError, TypeError, ValueError, StopIteration, AttributeError):
        return []


def summarize(root: Root, run: Path) -> dict[str, Any]:
    config, results = _load(run / "config.json"), _load(run / "results.json")
    kind = family(config, results)
    status = results.get("status", "unknown") if isinstance(results, dict) else "unknown"
    if kind == "sweep":
        status = "results"
    updated = _updated(run)
    if status == "running" and time.time() - updated > STALLED_AFTER_SECONDS:
        status = "stalled"
    if kind == "rl-ppo" and isinstance(results, dict) and "best" not in results:
        results = {**results, "best": _ppo_best(run)}
    return {
        "id": f"{root.label}/{run.name}",
        "root": root.label,
        "name": run.name,
        "family": kind,
        "status": status,
        "started": _started(run),
        "updated": updated,
        "headline": safe_headline(kind, results),
        "media": sum(1 for _ in _media(run)),
    }


def _ppo_best(run: Path) -> dict[str, Any] | None:
    evaluations = _load(run / "evaluations.json") or []
    best: dict[str, Any] | None = None
    for entry in evaluations:
        scored = entry.get("variants", {"nominal": entry})
        chosen = entry.get("selection", scored)
        first, key = next(iter(scored.values())), next(iter(chosen.values()))
        if "successes" in first and (best is None or key["successes"] > best["criterion"]):
            best = {**first, "iteration": entry.get("iteration"), "criterion": key["successes"]}
    return best


def _media(run: Path, depth: int = 3) -> list[Path]:
    found: list[Path] = []
    for path in sorted(run.rglob("*")):
        if path.suffix in MEDIA_SUFFIXES and len(path.relative_to(run).parts) <= depth:
            found.append(path)
    return found


def _strip_episodes(value: Any) -> Any:
    """Drop per-episode arrays so run details stay small."""
    if isinstance(value, dict):
        return {k: _strip_episodes(v) for k, v in value.items() if k not in ("episodes", "seeds")}
    if isinstance(value, list):
        return [_strip_episodes(v) for v in value]
    return value


def curves(run: Path) -> list[dict[str, Any]]:
    """Learning-curve series: per-model learning.json files and PPO curves/evaluations."""
    series: list[dict[str, Any]] = []
    for learning in sorted(run.glob("*/learning.json")):
        points = _load(learning) or []
        if not isinstance(points, list) or not points:
            continue
        x_key = "epoch" if "epoch" in points[0] else "step"
        # Phases restart their epoch counter; plot against the running index.
        index = list(range(1, len(points) + 1))
        for metric in ("train_loss", "validation_loss", "train_mse", "validation_mse"):
            values = [p.get(metric) for p in points]
            if any(v is not None for v in values):
                series.append(
                    {
                        "group": learning.parent.name,
                        "metric": metric,
                        "x": index if x_key == "epoch" else [p[x_key] for p in points],
                        "y": values,
                    }
                )
        scores = [p.get("selection_score") for p in points]
        if any(s is not None for s in scores):
            series.append(
                {
                    "group": learning.parent.name,
                    "metric": "selection_score",
                    "x": index,
                    "y": scores,
                }
            )
    ppo = _load(run / "curves.json")
    if isinstance(ppo, list) and ppo:
        x = [p["env_steps"] for p in ppo]
        for metric in ("mean_reward", "success_rate", "lift_rate"):
            series.append(
                {"group": "ppo rollouts", "metric": metric, "x": x, "y": [p[metric] for p in ppo]}
            )
    evaluations = _load(run / "evaluations.json")
    if isinstance(evaluations, list) and evaluations:
        x = [e["env_steps"] for e in evaluations]
        variants = evaluations[0].get("variants", {"nominal": evaluations[0]})
        for name in variants:
            for metric in ("successes", "lifts"):
                values = [e.get("variants", {"nominal": e})[name].get(metric) for e in evaluations]
                series.append({"group": f"eval {name}", "metric": metric, "x": x, "y": values})
    return series


def log_tail(run: Path) -> list[str]:
    for log in (run.with_suffix(".log"), *sorted(run.glob("*.log"))):
        if log.is_file():
            lines = log.read_text(errors="replace").splitlines()
            keep = [line for line in lines if "AdroitHand" not in line]
            return keep[-MAX_LOG_LINES:]
    return []


def sweep_table(results: Any) -> list[dict[str, Any]]:
    """Mean and per-seed score of every (setting, kind) group of a protocol sweep."""
    if not isinstance(results, list):
        return []
    groups: dict[tuple[str, str], list[float]] = {}
    for row in results:
        if isinstance(row, dict) and "score" in row:
            key = (str(row.get("setting", row.get("chunk", "?"))), str(row.get("kind", "?")))
            groups.setdefault(key, []).append(float(row["score"]))
    return [
        {
            "setting": setting,
            "kind": kind,
            "mean": sum(scores) / len(scores),
            "scores": scores,
        }
        for (setting, kind), scores in groups.items()
    ]


def detail(root: Root, run: Path) -> dict[str, Any]:
    results = _load(run / "results.json")
    return {
        **summarize(root, run),
        "table": sweep_table(results),
        "config": _load(run / "config.json"),
        "results": _strip_episodes(results),
        "curves": curves(run),
        "log": log_tail(run),
        "media_files": [str(path.relative_to(root.path)) for path in _media(run)],
    }


def gallery(roots: list[Root], limit: int = 200) -> list[dict[str, Any]]:
    """Every rollout video under the roots, newest first."""
    videos: list[dict[str, Any]] = []
    for root in roots:
        for path in root.path.rglob("*.mp4"):
            if len(path.relative_to(root.path).parts) > 5:
                continue
            videos.append(
                {
                    "root": root.label,
                    "path": str(path.relative_to(root.path)),
                    "name": path.stem,
                    "updated": path.stat().st_mtime,
                }
            )
    return sorted(videos, key=lambda item: item["updated"], reverse=True)[:limit]


def resolve_media(roots: list[Root], label: str, relative: str) -> Path:
    """The media file ``relative`` under root ``label``; refuses anything outside it."""
    root = next((r for r in roots if r.label == label), None)
    if root is None:
        raise FileNotFoundError(label)
    path = (root.path / relative).resolve()
    if not path.is_relative_to(root.path) or path.suffix not in MEDIA_SUFFIXES:
        raise PermissionError(relative)
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path
