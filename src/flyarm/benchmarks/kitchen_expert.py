"""A FrankaKitchen teacher built only from the benchmark's own demonstrations.

The tracker finds the demonstration state closest to the current one (robot and object
joint positions) and returns that demonstration's action plus a proportional correction
that pulls the arm back onto the demonstrated joint path. The benchmark commands joint
velocity as ``2 * action`` and targets ``observed q + velocity * dt``, so a joint error e is
removed in one step by ``e / (2 dt)``; ``gain`` removes that fraction of it per step.

It is a teacher for interactive imitation (DAgger), not a controller under test.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flyarm.benchmarks import kitchen

CONTROL_DT = 0.08  # FrankaKitchen-v1: frame_skip 40 x 2 ms
ACTION_TO_VELOCITY = 2.0  # FrankaKitchenEnv.act_rng


class DemonstrationTracker:
    def __init__(
        self, obs: np.ndarray, actions: np.ndarray, mask: np.ndarray, *, gain: float = 0.5
    ) -> None:
        """``obs`` are full 59-D benchmark observations [episodes, steps, 59]."""
        if obs.shape[:2] != actions.shape[:2] or obs.shape[:2] != mask.shape:
            raise ValueError("Demonstration arrays must share [episodes, steps]")
        if not 0 <= gain <= 1:
            raise ValueError("gain must be in [0, 1]")
        rows, steps = np.nonzero(mask > 0)
        self.gain = gain
        self._features = obs[rows, steps][:, kitchen.POLICY_FEATURES].astype(np.float64)
        self._joints = obs[rows, steps, :9].astype(np.float64)
        self._actions = actions[rows, steps].astype(np.float64)

    @classmethod
    def from_data(cls, data: kitchen.KitchenData, *, gain: float = 0.5) -> DemonstrationTracker:
        return cls(data.obs, data.actions, data.mask, gain=gain)

    def label(self, observation: np.ndarray) -> np.ndarray:
        """Teacher action for one full 59-D benchmark observation."""
        features = np.asarray(observation, dtype=np.float64)[kitchen.POLICY_FEATURES]
        nearest = int(np.argmin(((self._features - features) ** 2).sum(1)))
        error = self._joints[nearest] - features[:9]
        action = self._actions[nearest] + self.gain * error / (ACTION_TO_VELOCITY * CONTROL_DT)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        """Stateless; present so the tracker can be evaluated as a controller."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.label(observation)


def noisy_teacher_episodes(
    tracker: DemonstrationTracker, env: Any, episodes: int, noise: float, first_seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """DART data: the tracker acts with Gaussian action noise; labels are its clean actions."""
    if episodes < 1 or noise <= 0:
        raise ValueError("DART needs at least one episode and positive noise")
    horizon = env.spec.max_episode_steps
    obs = np.zeros((episodes, horizon, kitchen.FEATURE_DIM), np.float32)
    actions = np.zeros((episodes, horizon, kitchen.ACTION_DIM), np.float32)
    mask = np.zeros((episodes, horizon), np.float32)
    generator = np.random.default_rng(first_seed)
    completed: list[int] = []
    for episode in range(episodes):
        observation, info = env.reset(seed=first_seed + episode)
        for step in range(horizon):
            full = np.asarray(observation["observation"], dtype=np.float32)
            label = tracker.label(full)
            obs[episode, step] = full[kitchen.POLICY_FEATURES]
            actions[episode, step] = label
            mask[episode, step] = 1.0
            executed = np.clip(label + noise * generator.standard_normal(label.shape), -1, 1)
            observation, _, terminated, truncated, info = env.step(executed.astype(np.float64))
            if terminated or truncated:
                break
        completed.append(len(info.get("episode_task_completions", [])))
    stats = {
        "episodes": episodes,
        "noise": noise,
        "first_seed": first_seed,
        "teacher_mean_tasks_under_noise": float(np.mean(completed)),
        "labelled_states": int(mask.sum()),
    }
    return {"obs": obs, "actions": actions, "mask": mask}, stats
