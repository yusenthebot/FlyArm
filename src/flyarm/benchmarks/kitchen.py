"""D4RL FrankaKitchen (Minari v2 datasets, Gymnasium-Robotics FrankaKitchen-v1).

The benchmark is used unmodified: its datasets, its environment as recorded in each dataset,
its 280-step limit and its task-completion signal. The score of an episode is the number of
target tasks completed (0-4); the D4RL normalized score is 25 x that number, averaged.
Out-of-distribution variants only raise the environment's own observation-noise ratios or
offset the robot's initial joint angles, and are always labelled as such.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from flyarm.benchmarks import _robotics_compat

DATASETS = {
    "complete": "D4RL/kitchen/complete-v2",
    "partial": "D4RL/kitchen/partial-v2",
    "mixed": "D4RL/kitchen/mixed-v2",
}
# Published D4RL behavior cloning (original v0 environment), for approximate reference only.
PUBLISHED_BC = {"complete": 65.0, "partial": 38.0, "mixed": 51.5}
OBS_DIM, ACTION_DIM = 59, 9
# Controllers see positions only: the 9 robot joint positions and the 21 object joint positions.
# Robot joint velocities are almost exactly the previous velocity command, so behavior cloning
# with them learns to copy its last action ("copycat" causal confusion) and never starts the
# task: measured on kitchen-complete, the same MLP scores 0 with velocities and 45 without.
# Recurrent controllers can still infer velocity from their own history.
POLICY_FEATURES = np.r_[0:9, 18:39]
FEATURE_DIM = len(POLICY_FEATURES)
PROPRIOCEPTION = slice(0, 9)  # robot joint positions within the policy features
EXTEROCEPTION = slice(9, 30)  # object joint positions within the policy features


class Controller(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: np.ndarray) -> np.ndarray: ...


class PositionFeatures:
    """Feeds a controller the position-only policy features of the benchmark observation."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller

    def reset(self) -> None:
        self.controller.reset()

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.controller.act(observation[POLICY_FEATURES])


def _minari() -> Any:
    import gymnasium as gym
    import gymnasium_robotics
    import minari

    _robotics_compat.install()
    gym.register_envs(gymnasium_robotics)
    return minari


@dataclass(frozen=True)
class KitchenData:
    split: str
    dataset_id: str
    obs: np.ndarray  # [episodes, steps, 59]
    actions: np.ndarray  # [episodes, steps, 9]
    mask: np.ndarray  # [episodes, steps]
    episode_ids: np.ndarray
    provenance: dict[str, Any]

    def subset(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Training arrays for the given episodes, restricted to the policy features."""
        return {
            "obs": self.obs[rows][..., POLICY_FEATURES],
            "actions": self.actions[rows],
            "mask": self.mask[rows],
            "seeds": self.episode_ids[rows],
        }


def load(split: str, *, download: bool = True) -> KitchenData:
    """All episodes of one split, padded to the longest episode with a validity mask."""
    if split not in DATASETS:
        raise ValueError(f"Unknown kitchen split {split}; choose from {sorted(DATASETS)}")
    minari = _minari()
    dataset = minari.load_dataset(DATASETS[split], download=download)
    episodes = [episode for episode in dataset.iterate_episodes() if len(episode.actions) > 1]
    horizon = max(len(episode.actions) for episode in episodes)
    obs = np.zeros((len(episodes), horizon, OBS_DIM), dtype=np.float32)
    actions = np.zeros((len(episodes), horizon, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((len(episodes), horizon), dtype=np.float32)
    for row, episode in enumerate(episodes):
        steps = len(episode.actions)
        obs[row, :steps] = episode.observations["observation"][:steps]
        actions[row, :steps] = episode.actions
        mask[row, :steps] = 1.0
    if not np.all(np.isfinite(obs)) or np.abs(actions).max() > 1.0 + 1e-6:
        raise ValueError(f"{DATASETS[split]} contains non-finite observations or bad actions")
    root = Path(dataset.spec.data_path)
    return KitchenData(
        split=split,
        dataset_id=DATASETS[split],
        obs=obs,
        actions=np.clip(actions, -1.0, 1.0),
        mask=mask,
        episode_ids=np.array([int(episode.id) for episode in episodes], dtype=np.int64),
        provenance={
            "dataset_id": DATASETS[split],
            "minari_version": minari.__version__,
            "episodes": len(episodes),
            "dropped_short_episodes": dataset.total_episodes - len(episodes),
            "transitions": int(mask.sum()),
            "tasks": list(recover_env(split).unwrapped.goal),
            "data_files": sorted(str(path.relative_to(root)) for path in root.rglob("*.hdf5")),
        },
    )


def recover_env(
    split: str, *, robot_noise_ratio: float | None = None, object_noise_ratio: float | None = None
) -> Any:
    """The benchmark environment exactly as recorded in the dataset, optionally noisier."""
    minari = _minari()
    dataset = minari.load_dataset(DATASETS[split])
    overrides: dict[str, float] = {}
    if robot_noise_ratio is not None:
        overrides["robot_noise_ratio"] = robot_noise_ratio
    if object_noise_ratio is not None:
        overrides["object_noise_ratio"] = object_noise_ratio
    return dataset.recover_environment(**overrides)


def evaluate(
    env: Any,
    controller: Controller | None,
    seeds: list[int],
    *,
    initial_joint_offset: float = 0.0,
    replay_actions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Closed-loop episodes; score = target tasks completed per episode (0-4)."""
    episodes: list[dict[str, Any]] = []
    tasks = list(env.unwrapped.goal)
    for seed in seeds:
        observation, info = env.reset(seed=seed)
        if initial_joint_offset:
            observation = _offset_initial_joints(env, seed, initial_joint_offset)
        if controller is not None:
            controller.reset()
        completed: list[str] = []
        timings: list[float] = []
        for step in range(env.spec.max_episode_steps):
            if controller is not None:
                started = time.perf_counter()
                action = controller.act(np.asarray(observation["observation"], dtype=np.float32))
                timings.append(time.perf_counter() - started)
            elif replay_actions is not None:
                if step >= len(replay_actions):
                    break
                action = replay_actions[step]
            else:
                action = np.zeros(ACTION_DIM, dtype=np.float32)
            observation, _, terminated, truncated, info = env.step(np.asarray(action, np.float64))
            completed = list(info["episode_task_completions"])
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": seed,
                "tasks_completed": len(completed),
                "completed": completed,
                "steps": step + 1,
                "inference_ms_median": float(np.median(timings) * 1000) if timings else None,
            }
        )
    counts = np.array([episode["tasks_completed"] for episode in episodes], dtype=np.float64)
    return {
        "tasks": tasks,
        "episodes": episodes,
        "mean_tasks": float(counts.mean()),
        "normalized_score": float(25.0 * counts.mean()),
        "normalized_score_sem": float(25.0 * counts.std(ddof=1) / np.sqrt(len(counts)))
        if len(counts) > 1
        else 0.0,
        "per_task_success": {
            task: float(np.mean([task in episode["completed"] for episode in episodes]))
            for task in tasks
        },
        "initial_joint_offset_rad": initial_joint_offset,
    }


def _offset_initial_joints(env: Any, seed: int, magnitude: float) -> dict[str, Any]:
    """OOD start: offset the 7 arm joints uniformly in [-magnitude, magnitude] (seeded)."""
    import mujoco

    base = env.unwrapped
    robot = base.robot_env
    offset = np.random.default_rng(seed + 424242).uniform(-magnitude, magnitude, 7)
    robot.data.qpos[:7] = robot.init_qpos[:7] + offset
    robot.data.qvel[:] = 0.0
    mujoco.mj_forward(robot.model, robot.data)
    # robot._get_obs() also resets the controller's reference joint positions.
    return base._get_obs(robot._get_obs())
