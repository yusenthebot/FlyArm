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
from flyarm.rl.batched_pick_place import ACTION_DIM, OBS_DIM
from flyarm.rl.record import _episodes
from flyarm.video import write_video
from flyarm.whole_brain.experiment import load_trained_policy
from flyarm.whole_brain.policy import BrainPolicy, DirectPolicy

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


def _kitchen_frames(
    run: Path, pack_root: Path, checkpoint: Path, seeds: list[int], title: str
) -> tuple[list[list[np.ndarray]], list[str]]:
    """Labelled kitchen episodes of one PPO checkpoint, rendered in the benchmark env."""
    from flyarm.benchmarks import kitchen
    from flyarm.config import KitchenPPOConfig
    from flyarm.flyleg.record import rollout_frames
    from flyarm.rl.ppo_kitchen import scratch_kitchen_policy

    config = KitchenPPOConfig.model_validate_json((run / "config.json").read_text())
    policy = scratch_kitchen_policy(config, pack_root)
    policy.load(checkpoint)
    env = (
        kitchen._minari()
        .load_dataset(kitchen.DATASETS["complete"])
        .recover_environment(render_mode="rgb_array")
    )
    clips, outcomes = [], []
    try:
        for seed in seeds:
            frames, completed = rollout_frames(env, policy, seed, title)
            clips.append(frames)
            outcomes.append(", ".join(completed) if completed else "no task")
    finally:
        env.close()
    return clips, outcomes


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
    settings = json.loads((run / "config.json").read_text())
    iteration = _iteration(checkpoint)
    if "completion_bonus" in settings:  # a kitchen run (KitchenPPOConfig)
        title = f"{run.name} · iteration {iteration}"
        clips, outcomes = _kitchen_frames(run, pack_root, checkpoint, seeds, title)
        return _write_clip(run, checkpoint, seeds, clips, outcomes, iteration, variant="kitchen")
    config = PPOConfig.model_validate_json((run / "config.json").read_text())
    policy: BrainPolicy | DirectPolicy
    if config.controller == "mlp":
        policy = DirectPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    else:
        task, loaded = load_trained_policy(
            Path(config.base_run), config.base_kind, config.base_seed, pack_root, model_path
        )
        task.close()
        if not isinstance(loaded, BrainPolicy):
            raise ValueError("Progress clips are defined for brain policies")
        policy = loaded
    policy.load(checkpoint)
    title = f"{run.name} · iteration {iteration}"
    clips, outcomes = _episodes(policy, model_path, seeds, config, variant, title)
    return _write_clip(run, checkpoint, seeds, clips, outcomes, iteration, variant=variant)


def _write_clip(
    run: Path,
    checkpoint: Path,
    seeds: list[int],
    clips: list[list[np.ndarray]],
    outcomes: list[str],
    iteration: int,
    *,
    variant: str,
) -> dict[str, Any]:
    """Stack the episodes into one clip and replace the run's single progress file."""
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
