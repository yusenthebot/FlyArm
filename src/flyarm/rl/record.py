"""Before-and-after rollout videos of PPO-tuned connectome controllers.

Each row is one test episode; the left column is the imitation checkpoint PPO started from,
the right column the PPO checkpoint, both on the same seed and task variant. Physics runs in
BatchedPickPlace (so physics variants such as heavier cubes apply); frames are rendered by
copying each simulation's state into one MjData.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from flyarm.config import PPOConfig
from flyarm.dashboard.catalog import readable
from flyarm.rl.batched_pick_place import ACTION_DIM, BatchedPickPlace
from flyarm.rl.ppo import Controller, MotorHead, rollout_for, task_variant
from flyarm.video import annotate, tile, write_video
from flyarm.whole_brain.experiment import load_trained_policy

FPS = 20


def _stage(env: BatchedPickPlace, row: int, placed: bool) -> str:
    if placed:
        return "placed"
    if env.ever_lifted[row]:
        return "lifted"
    if env.ever_grasped[row]:
        return "grasped"
    left, right = env.contacts()
    return "contact" if left[row] or right[row] else "free"


def _episodes(
    policy: Controller,
    model_path: Path,
    seeds: list[int],
    config: PPOConfig,
    variant_name: str,
    title: str,
) -> tuple[list[list[np.ndarray]], list[str]]:
    """Frames per episode and the final stage of each, deterministic mean actions."""
    variant = task_variant(config.eval_variants[variant_name])
    env = BatchedPickPlace(model_path, len(seeds), horizon=config.horizon, variant=variant)
    obs = env.reset(seeds=np.array(seeds))
    brain = rollout_for(policy, len(seeds))
    head = MotorHead(policy.decoder, config.log_std, ACTION_DIM)
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    data = mujoco.MjData(env.model)
    frames: list[list[np.ndarray]] = [[] for _ in seeds]
    placed = np.zeros(len(seeds), dtype=bool)
    active = np.ones(len(seeds), dtype=bool)

    def render(row: int, step: int) -> np.ndarray:
        data.qpos[:] = env.qpos[row]
        data.qvel[:] = env.qvel[row]
        data.mocap_pos[:] = env.mocap_pos[row]
        mujoco.mj_forward(env.model, data)
        renderer.update_scene(data)
        detail = (
            f"episode {seeds[row]} · step {step}/{config.horizon} · cube mass "
            f"x{env.mass_scale[row]:.1f}\n{_stage(env, row, placed[row])}"
        )
        return annotate(renderer.render(), title, detail, success=bool(placed[row]))

    for row in range(len(seeds)):
        frames[row].append(render(row, 0))
    try:
        for step in range(1, config.horizon + 1):
            action = np.asarray(head.mean(brain.features(obs)), dtype=np.float64)
            result = env.step(action, auto_reset=False)
            placed |= active & result.success
            for running in np.flatnonzero(active).tolist():
                frames[running].append(render(running, step))
            active &= ~(result.success | result.truncated)
            obs = result.obs
            if not active.any():
                break
    finally:
        renderer.close()
    return frames, [_stage(env, row, placed[row]) for row in range(len(seeds))]


def record_before_after(
    run: Path,
    pack_root: Path,
    model_path: Path,
    *,
    variant: str,
    seeds: list[int],
    iteration: int | None = None,
) -> Path:
    """Write run/videos/<variant>-before-after.mp4 and a JSON manifest next to it."""
    config = PPOConfig.model_validate_json((run / "config.json").read_text())
    if variant not in config.eval_variants:
        raise ValueError(f"{variant} is not one of the run's variants {list(config.eval_variants)}")
    checkpoints = sorted(run.glob("policy-*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"{run} has no PPO checkpoints")
    tuned_path = run / f"policy-{iteration:04d}.safetensors" if iteration else checkpoints[-1]
    columns: list[tuple[str, Path | None]] = [
        ("Fly brain before PPO (imitation only)", None),
        ("Fly brain after PPO (decoder only)", tuned_path),
    ]
    shown = readable(variant)
    clips: dict[str, list[list[np.ndarray]]] = {}
    outcomes: dict[str, list[str]] = {}
    for label, weights in columns:
        task, policy = load_trained_policy(
            Path(config.base_run), config.base_kind, config.base_seed, pack_root, model_path
        )
        task.close()
        if weights is not None:
            policy.load(weights)
        clips[label], outcomes[label] = _episodes(
            policy, model_path, seeds, config, variant, f"{label} · {shown}"
        )
    rows = [
        tile([clips[label][row] for label, _ in columns], columns=2) for row in range(len(seeds))
    ]
    length = max(len(row) for row in rows)
    frames = [
        np.concatenate([row[min(index, len(row) - 1)] for row in rows], axis=0)
        for index in range(length)
    ]
    output = run / "videos" / f"{variant}-before-after.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    write_video(output, frames, FPS)
    manifest: dict[str, Any] = {
        "title": f"PPO before and after · {shown}",
        "run": str(run),
        "variant": variant,
        "seeds": seeds,
        "checkpoint": str(tuned_path),
        "outcomes": outcomes,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return output
