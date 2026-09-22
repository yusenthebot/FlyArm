"""Live progress clips: re-record one short rollout whenever a run saves a new checkpoint.

A watcher polls a PPO run directory, and every time a newer ``policy-XXXX.safetensors``
appears it renders a couple of labelled episodes with that checkpoint and replaces
``progress/latest.mp4`` and ``progress/latest.json`` in the run. Exactly one clip per run ever
exists: the new file is written beside it and moved over the old one, so the Live view always
shows what the controller does right now and the run directory never grows with clips.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from flyarm.config import PPOConfig
from flyarm.experiment import save_json
from flyarm.rl.record import _episodes
from flyarm.video import write_video
from flyarm.whole_brain.experiment import load_trained_policy
from flyarm.whole_brain.policy import BrainPolicy

FPS = 20


def checkpoints(run: Path) -> list[Path]:
    return sorted(run.glob("policy-*.safetensors"))


def _iteration(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else 0


def _curve(run: Path, iteration: int) -> dict[str, Any]:
    """The training curve row at that iteration, if the run has written one."""
    path = run / "curves.json"
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    matching = [row for row in rows if row.get("iteration") == iteration]
    row = matching[-1] if matching else (rows[-1] if rows else {})
    keys = ("iteration", "mean_reward", "success_rate", "lift_rate", "env_steps", "encoder_loss")
    return {key: row[key] for key in keys if key in row}


def record_progress(
    run: Path,
    pack_root: Path,
    model_path: Path,
    *,
    checkpoint: Path,
    seeds: list[int],
    variant: str = "nominal",
) -> dict[str, Any]:
    """Render ``seeds`` episodes of one checkpoint into run/progress/latest.mp4."""
    config = PPOConfig.model_validate_json((run / "config.json").read_text())
    task, policy = load_trained_policy(
        Path(config.base_run), config.base_kind, config.base_seed, pack_root, model_path
    )
    task.close()
    if not isinstance(policy, BrainPolicy):
        raise ValueError("Progress clips are defined for brain policies")
    policy.load(checkpoint)
    iteration = _iteration(checkpoint)
    title = f"{run.name} · iteration {iteration}"
    clips, outcomes = _episodes(policy, model_path, seeds, config, variant, title)
    frames = [
        np.concatenate([clip[min(index, len(clip) - 1)] for clip in clips], axis=0)
        for index in range(max(len(clip) for clip in clips))
    ]
    directory = run / "progress"
    directory.mkdir(exist_ok=True)
    temporary = directory / "writing.mp4"
    temporary.unlink(missing_ok=True)  # a previous recording may have been killed mid-write
    write_video(temporary, frames, FPS)
    os.replace(temporary, directory / "latest.mp4")
    kept = directory / "latest.mp4"
    manifest = {
        "run": str(run),
        "megabytes": round(kept.stat().st_size / 1e6, 2),
        "iteration": iteration,
        "checkpoint": checkpoint.name,
        "variant": variant,
        "seeds": seeds,
        "outcomes": outcomes,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "curve": _curve(run, iteration),
    }
    save_json(directory / "latest.json", manifest)
    return manifest


def watch(
    run: Path,
    pack_root: Path,
    model_path: Path,
    *,
    episodes: int = 2,
    poll_seconds: float = 60.0,
    variant: str = "nominal",
    first_seed: int = 60_000,
    once: bool = False,
) -> None:
    """Re-record the newest checkpoint of ``run`` until the run is finished."""
    seeds = list(range(first_seed, first_seed + episodes))
    recorded: str | None = None
    while True:
        latest = checkpoints(run)[-1] if checkpoints(run) else None
        if latest is not None and latest.name != recorded:
            manifest = record_progress(
                run, pack_root, model_path, checkpoint=latest, seeds=seeds, variant=variant
            )
            recorded = latest.name
            stages = ", ".join(
                f"{seed} {stage}" for seed, stage in zip(seeds, manifest["outcomes"], strict=True)
            )
            print(
                f"{manifest['recorded_at']} {run.name} iteration {manifest['iteration']}: {stages}",
                flush=True,
            )
        if once:
            return
        results = run / "results.json"
        if results.is_file():
            try:
                status = json.loads(results.read_text()).get("status")
            except json.JSONDecodeError:
                status = "running"
            if status in {"complete", "failed", "stopped"} and latest is not None:
                if latest.name == recorded:
                    return
        time.sleep(poll_seconds)
