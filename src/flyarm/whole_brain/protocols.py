"""Teacher data, loss weights and closed-loop evaluation of the reach and pick-place tasks.

These are the robot-side protocols the B1a experiment shares with the original subgraph
runs: the same seeds, teachers, recorded arrays and evaluation fields, so results stay
directly comparable. Controllers are anything with ``reset()`` and ``act()``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from flyarm.env import PandaReachEnv
from flyarm.pick_place_env import PandaPickPlaceEnv, physical_stage

PICK_PLACE_STAGES = {
    "approach": 0,
    "descend": 1,
    "close": 2,
    "lift": 3,
    "transport": 4,
    "lower": 5,
    "open": 6,
    "retreat": 7,
}


class ActionController(Protocol):
    """Anything that can drive one closed-loop episode: reset() once, then act() per step."""

    def reset(self) -> None: ...

    def act(self, observation: np.ndarray) -> np.ndarray: ...


def collect_reach(env: PandaReachEnv, seeds: list[int], path: Path) -> dict:
    observations = np.zeros((len(seeds), env.horizon, 20), dtype=np.float32)
    actions = np.zeros((len(seeds), env.horizon, 3), dtype=np.float32)
    mask = np.zeros((len(seeds), env.horizon), dtype=np.float32)
    targets, success = [], []
    for row, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        targets.append(info["target"])
        for _step in range(env.horizon):
            action = env.teacher_action()
            observations[row, _step], actions[row, _step], mask[row, _step] = obs, action, 1
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        success.append(bool(info["is_success"]))
    data = {
        "obs": observations,
        "actions": actions,
        "mask": mask,
        "seeds": np.array(seeds),
        "targets": np.array(targets),
        "teacher_success": np.array(success),
    }
    np.savez_compressed(
        path,
        obs=observations,
        actions=actions,
        mask=mask,
        seeds=data["seeds"],
        targets=data["targets"],
        teacher_success=data["teacher_success"],
    )
    return data


def evaluate_reach(
    env: PandaReachEnv,
    controller: ActionController | None,
    seeds: list[int],
    mode: str = "learned",
    noise_std: float = 0.0,
) -> dict:
    episodes = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 987654)
        if controller:
            controller.reset()
        timings, changes = [], []
        previous = np.zeros(3)
        for _step in range(env.horizon):
            noisy_obs = obs.copy()
            # Noise applies to xyz observations (metres), not velocities or joint radians.
            noisy_obs[14:] += rng.normal(0, noise_std, 6).astype(np.float32)
            start = time.perf_counter()
            if controller is not None:
                action = controller.act(noisy_obs)
            elif mode == "teacher":
                action = env.teacher_action()
            elif mode == "zero":
                action = np.zeros(3, dtype=np.float32)
            else:
                raise ValueError(mode)
            timings.append(time.perf_counter() - start)
            changes.append(float(np.linalg.norm(action - previous)))
            previous = action.copy()
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": seed,
                "success": bool(info["is_success"]),
                "steps": _step + 1,
                "final_distance_m": float(info["distance"]),
                "target": np.asarray(info["target"]).tolist(),
                "action_delta_mean": float(np.mean(changes)),
                "inference_ms_median": float(np.median(timings) * 1000),
                "inference_ms_p95": float(np.quantile(timings, 0.95) * 1000),
            }
        )
    return {
        "mode": mode,
        "xyz_noise_std_m": noise_std,
        "episodes": episodes,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_final_distance_m": float(np.mean([e["final_distance_m"] for e in episodes])),
    }


def collect_pick_place(
    env: PandaPickPlaceEnv, seeds: list[int], path: Path
) -> dict[str, np.ndarray]:
    obs_dim = env.observation_dim
    observations = np.zeros((len(seeds), env.horizon, obs_dim), dtype=np.float32)
    actions = np.zeros((len(seeds), env.horizon, 4), dtype=np.float32)
    mask = np.zeros((len(seeds), env.horizon), dtype=np.float32)
    stages = np.full((len(seeds), env.horizon), -1, dtype=np.int16)
    success = np.zeros(len(seeds), dtype=bool)
    ever_grasped = np.zeros(len(seeds), dtype=bool)
    ever_lifted = np.zeros(len(seeds), dtype=bool)
    final_goal_error = np.zeros(len(seeds), dtype=np.float32)
    for row, seed in enumerate(seeds):
        observation, info = env.reset(seed=seed)
        for step in range(env.horizon):
            action = env.teacher_action()
            observations[row, step] = observation
            actions[row, step] = action
            mask[row, step] = 1.0
            stages[row, step] = PICK_PLACE_STAGES[env.teacher_stage]
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        success[row] = bool(info["is_success"])
        ever_grasped[row] = bool(info["ever_grasped"])
        ever_lifted[row] = bool(info["ever_lifted"])
        final_goal_error[row] = float(info["goal_xy_error"])
    data: dict[str, np.ndarray] = {
        "obs": observations,
        "actions": actions,
        "mask": mask,
        "stages": stages,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "teacher_success": success,
        "teacher_ever_grasped": ever_grasped,
        "teacher_ever_lifted": ever_lifted,
        "teacher_goal_xy_error": final_goal_error,
    }
    np.savez_compressed(
        path,
        obs=observations,
        actions=actions,
        mask=mask,
        stages=stages,
        seeds=data["seeds"],
        teacher_success=success,
        teacher_ever_grasped=ever_grasped,
        teacher_ever_lifted=ever_lifted,
        teacher_goal_xy_error=final_goal_error,
    )
    return data


def collect_dagger_queries(
    env: PandaPickPlaceEnv,
    controller: ActionController,
    seeds: list[int],
    path: Path,
) -> dict[str, np.ndarray]:
    """Label policy-visited states with the same-API teacher; never correct actions online."""
    obs_dim = env.observation_dim
    observations = np.zeros((len(seeds), env.horizon, obs_dim), dtype=np.float32)
    actions = np.zeros((len(seeds), env.horizon, 4), dtype=np.float32)
    mask = np.zeros((len(seeds), env.horizon), dtype=np.float32)
    stages = np.full((len(seeds), env.horizon), -1, dtype=np.int16)
    for row, seed in enumerate(seeds):
        observation, _ = env.reset(seed=seed)
        controller.reset()
        for step in range(env.horizon):
            expert_action = env.teacher_action()
            observations[row, step] = observation
            actions[row, step] = expert_action
            mask[row, step] = 1.0
            stages[row, step] = PICK_PLACE_STAGES[env.teacher_stage]
            policy_action = controller.act(observation)
            observation, _, terminated, truncated, _ = env.step(policy_action)
            if terminated or truncated:
                break
    data: dict[str, np.ndarray] = {
        "obs": observations,
        "actions": actions,
        "mask": mask,
        "stages": stages,
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    np.savez_compressed(
        path,
        obs=observations,
        actions=actions,
        mask=mask,
        stages=stages,
        seeds=data["seeds"],
    )
    return data


def concatenate_data(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    keys = {"obs", "actions", "mask", "stages", "seeds"}
    return {key: np.concatenate((first[key], second[key]), axis=0) for key in keys}


def stage_balanced_weights(data: dict[str, np.ndarray]) -> np.ndarray:
    """Per-step loss weights that equalize the eight teacher phases (clipped to 0.4-4)."""
    mask = data["mask"]
    stages = data["stages"]
    valid_stages = stages[mask.astype(bool)]
    counts = np.bincount(valid_stages, minlength=len(PICK_PLACE_STAGES)).astype(np.float32)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights = np.clip(weights, 0.4, 4.0)
    return mask * weights[np.maximum(stages, 0)]


def evaluate_pick_place(
    env: PandaPickPlaceEnv,
    controller: ActionController | None,
    seeds: list[int],
    *,
    mode: str = "learned",
    reset_state_every_step: bool = False,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        observation, info = env.reset(seed=seed)
        if controller is not None:
            controller.reset()
        timings: list[float] = []
        action_changes: list[float] = []
        previous = np.zeros(4, dtype=np.float32)
        steps = 0
        for _step in range(env.horizon):
            steps += 1
            if controller is not None:
                if reset_state_every_step:
                    controller.reset()
                started = time.perf_counter()
                action = controller.act(observation)
                timings.append(time.perf_counter() - started)
            elif mode == "teacher":
                action = env.teacher_action()
            elif mode == "zero":
                action = np.zeros(4, dtype=np.float32)
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")
            action_changes.append(float(np.linalg.norm(action - previous)))
            previous = action.copy()
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": seed,
                "success": bool(info["is_success"]),
                "steps": steps,
                "ever_grasped": bool(info["ever_grasped"]),
                "ever_lifted": bool(info["ever_lifted"]),
                "goal_xy_error_m": float(info["goal_xy_error"]),
                "object_height_m": float(info["object_height"]),
                "object_speed_mps": float(info["object_speed"]),
                # The teacher state machine only advances when the teacher acts, so a
                # learned policy is described by its physical outcome instead.
                "final_physical_stage": physical_stage(info),
                "final_teacher_stage": str(info["stage"]) if mode == "teacher" else None,
                "action_delta_mean": float(np.mean(action_changes)),
                "inference_ms_median": (float(np.median(timings) * 1000) if timings else None),
            }
        )
    return {
        "mode": mode,
        "reset_state_every_step": reset_state_every_step,
        "episodes": episodes,
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "grasp_rate": float(np.mean([episode["ever_grasped"] for episode in episodes])),
        "lift_rate": float(np.mean([episode["ever_lifted"] for episode in episodes])),
        "mean_goal_xy_error_m": float(
            np.mean([episode["goal_xy_error_m"] for episode in episodes])
        ),
    }
