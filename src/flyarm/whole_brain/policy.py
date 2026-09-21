"""Trainable adapters around the frozen connectome, and a parameter-matched GRU (MLX).

Only ``encoder`` (observation -> ascending-neuron current) and ``decoder`` (pooled
descending/motor activity -> action) are trainable. The connectome lives in RateDynamics,
which is a plain attribute rather than a module parameter, so no optimizer can see it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from flyarm.whole_brain.backend_mlx import RateDynamics


class _Normalized(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_mean = mx.zeros((obs_dim,))
        self.obs_scale = mx.ones((obs_dim,))
        self.freeze(keys=["obs_mean", "obs_scale"], recurse=False)

    def set_normalization(self, mean: np.ndarray, scale: np.ndarray) -> None:
        if mean.shape != (self.obs_dim,) or scale.shape != (self.obs_dim,):
            raise ValueError("Normalization shape does not match obs_dim")
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("Normalization scale must be finite and positive")
        self.obs_mean = mx.array(mean, dtype=mx.float32)
        self.obs_scale = mx.array(scale, dtype=mx.float32)
        self.freeze(keys=["obs_mean", "obs_scale"], recurse=False)

    def normalize(self, obs: mx.array) -> mx.array:
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(f"Expected observations shaped [batch, {self.obs_dim}]")
        return (obs - self.obs_mean) / self.obs_scale

    def trainable_parameter_count(self) -> int:
        leaves = cast(list[tuple[str, mx.array]], tree_flatten(self.trainable_parameters()))
        return sum(value.size for _, value in leaves)

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        self.save_weights(str(path))

    def load(self, path: Path) -> None:
        self.load_weights(str(path), strict=True)
        self.freeze(keys=["obs_mean", "obs_scale"], recurse=False)


class BrainPolicy(_Normalized):
    """obs -> encoder -> frozen connectome (ascending in, descending/motor out) -> action."""

    def __init__(
        self,
        kind: str,
        dynamics: RateDynamics,
        *,
        obs_dim: int,
        action_dim: int,
        neural_steps: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__(obs_dim)
        if neural_steps < 1 or action_dim < 1:
            raise ValueError("neural_steps and action_dim must be positive")
        self.kind = kind
        self.seed = seed
        self.action_dim = action_dim
        self.neural_steps = neural_steps
        self.dynamics = dynamics
        mx.random.seed(seed)
        self.encoder = nn.Linear(obs_dim, dynamics.input_count)
        self.decoder = nn.Linear(dynamics.output_count, action_dim)

    @property
    def state_size(self) -> int:
        return self.dynamics.neurons

    def initial_state(self, batch: int) -> mx.array:
        return self.dynamics.zeros(batch)

    def step(self, obs: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        current = self.encoder(self.normalize(obs))
        state, pooled = self.dynamics.advance(state, current, self.neural_steps)
        return mx.tanh(self.decoder(pooled)), state

    def with_dynamics(self, kind: str, dynamics: RateDynamics) -> BrainPolicy:
        """Same learned adapters over another frozen graph (post-training ablations)."""
        same_io = all(
            np.array_equal(
                np.asarray(getattr(dynamics, name)), np.asarray(getattr(self.dynamics, name))
            )
            for name in ("body_ids", "input_indices", "output_indices")
        )
        if not same_io:
            raise ValueError("Ablation dynamics must keep the same neurons and interface")
        clone = BrainPolicy(
            kind,
            dynamics,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            neural_steps=self.neural_steps,
            seed=self.seed,
        )
        clone.update(self.parameters())
        clone.freeze(keys=["obs_mean", "obs_scale"], recurse=False)
        return clone


def gru_hidden_for_budget(obs_dim: int, action_dim: int, budget: int) -> int:
    """Hidden width whose MLX GRU + linear readout is closest to ``budget`` parameters."""

    def count(hidden: int) -> int:
        return 3 * hidden * (obs_dim + hidden + 1) + hidden + (hidden + 1) * action_dim

    return min(range(1, 2048), key=lambda hidden: abs(count(hidden) - budget))


class GRUPolicy(_Normalized):
    kind = "gru"

    def __init__(self, *, obs_dim: int, action_dim: int, hidden: int, seed: int = 0) -> None:
        super().__init__(obs_dim)
        self.action_dim = action_dim
        self.hidden = hidden
        mx.random.seed(seed)
        self.cell = nn.GRU(obs_dim, hidden)
        self.readout = nn.Linear(hidden, action_dim)

    @property
    def state_size(self) -> int:
        return self.hidden

    def initial_state(self, batch: int) -> mx.array:
        return mx.zeros((batch, self.hidden))

    def step(self, obs: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self.cell(self.normalize(obs)[:, None, :], hidden=state)[:, -1]
        return mx.tanh(self.readout(hidden)), hidden


SequencePolicy = BrainPolicy | GRUPolicy


class MlxController:
    """Single-episode closed-loop inference; only the action returns to the host."""

    def __init__(self, policy: SequencePolicy) -> None:
        self.policy = policy
        self.state = policy.initial_state(1)

    def reset(self) -> None:
        self.state = self.policy.initial_state(1)

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = mx.array(np.asarray(observation, dtype=np.float32)[None])
        action, self.state = self.policy.step(obs, self.state)
        mx.eval(action, self.state)
        return np.asarray(action[0], dtype=np.float32)

    def output_activity(self) -> Any:
        """Current state of the declared output neurons (brain policies only)."""
        if not isinstance(self.policy, BrainPolicy):
            return None
        dynamics = self.policy.dynamics
        return np.asarray(mx.take(self.state, dynamics.output_indices, axis=0))[:, 0]
