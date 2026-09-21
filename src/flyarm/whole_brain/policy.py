"""Trainable adapters around the frozen connectome, and a parameter-matched GRU (MLX).

Only ``encoder`` (observation -> ascending-neuron current) and ``decoder`` (pooled
descending/motor activity -> action) are trainable. The connectome lives in RateDynamics,
which is a plain attribute rather than a module parameter, so no optimizer can see it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from flyarm.whole_brain.backend_mlx import RateDynamics

# ACT's temporal-ensembling constant: w_i = exp(-m * i), i = 0 for the oldest prediction.
ENSEMBLE_DECAY = 0.01


class _Normalized(nn.Module):
    """Observation normalization plus the action-chunk layout shared by every policy.

    A policy with ``chunk`` k emits, at every control step, the next k actions flattened to
    ``[batch, k * action_dim]`` (ACT-style action chunking); k = 1 is ordinary control.
    """

    #: Buffers saved with the checkpoint but never trained.
    frozen_keys: tuple[str, ...] = ("obs_mean", "obs_scale")

    def __init__(self, obs_dim: int, action_dim: int, chunk: int) -> None:
        super().__init__()
        if action_dim < 1 or chunk < 1:
            raise ValueError("action_dim and chunk must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk = chunk
        self.obs_mean = mx.zeros((obs_dim,))
        self.obs_scale = mx.ones((obs_dim,))
        self._freeze_buffers()

    @property
    def output_dim(self) -> int:
        return self.action_dim * self.chunk

    def _freeze_buffers(self) -> None:
        present = [key for key in self.frozen_keys if key in self]
        self.freeze(keys=present, recurse=False)

    def set_normalization(self, mean: np.ndarray, scale: np.ndarray) -> None:
        if mean.shape != (self.obs_dim,) or scale.shape != (self.obs_dim,):
            raise ValueError("Normalization shape does not match obs_dim")
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise ValueError("Normalization scale must be finite and positive")
        self.obs_mean = mx.array(mean, dtype=mx.float32)
        self.obs_scale = mx.array(scale, dtype=mx.float32)
        self._freeze_buffers()

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
        weights = cast(dict[str, mx.array], mx.load(str(path)))
        # Checkpoints written before a buffer existed keep that buffer's neutral default.
        for key in self.frozen_keys:
            if key not in weights and key in self:
                weights[key] = self[key]
        self.load_weights(list(weights.items()), strict=True)
        self._freeze_buffers()


Channel = tuple[int, int, int]  # (first observation index, stop index, input neurons)


class BrainPolicy(_Normalized):
    """obs -> linear encoder(s) -> frozen connectome -> readout scale -> linear decoder -> action.

    With ``channels``, each observation slice is written only into its own consecutive block
    of input neurons (for example joint state into leg proprioceptors and scene state into
    head sensory neurons); without it one encoder writes every observation into every input.

    ``readout_scale`` is a frozen per-output-neuron gain (default 1). calibrate_readout() sets
    it to the inverse RMS activity on training data, so weakly driven motor neurons reach the
    decoder at unit scale; it rescales the declared outputs only and adds no pathway.
    """

    frozen_keys = ("obs_mean", "obs_scale", "readout_scale")

    def __init__(
        self,
        kind: str,
        dynamics: RateDynamics,
        *,
        obs_dim: int,
        action_dim: int,
        neural_steps: int = 3,
        seed: int = 0,
        channels: Sequence[Channel] | None = None,
        chunk: int = 1,
    ) -> None:
        super().__init__(obs_dim, action_dim, chunk)
        if neural_steps < 1:
            raise ValueError("neural_steps must be positive")
        self.kind = kind
        self.seed = seed
        self.neural_steps = neural_steps
        self.dynamics = dynamics
        self.channels = tuple(channels) if channels is not None else None
        mx.random.seed(seed)
        if self.channels is None:
            self.encoder = nn.Linear(obs_dim, dynamics.input_count)
        else:
            if sum(size for _, _, size in self.channels) != dynamics.input_count:
                raise ValueError("Channel input blocks must cover every declared input neuron")
            if any(not 0 <= start < stop <= obs_dim for start, stop, _ in self.channels):
                raise ValueError("Channel observation slices must lie inside the observation")
            self.encoders = [nn.Linear(stop - start, size) for start, stop, size in self.channels]
        self.decoder = nn.Linear(dynamics.output_count, self.output_dim)
        self.readout_scale = mx.ones((dynamics.output_count,))
        self._freeze_buffers()

    def calibrate_readout(
        self, obs: np.ndarray, mask: np.ndarray, *, floor_fraction: float = 0.01
    ) -> dict[str, float]:
        """Run training episodes through the current policy; scale outputs to unit RMS."""
        state = self.initial_state(obs.shape[0])
        total = np.zeros(self.dynamics.output_count)
        count = 0.0
        for step in range(obs.shape[1]):
            current = self.encode(self.normalize(mx.array(obs[:, step])))
            state, pooled = self.dynamics.advance(state, current, self.neural_steps)
            mx.eval(state, pooled)
            weights = mask[:, step][:, None]
            total += (np.asarray(pooled, dtype=np.float64) ** 2 * weights).sum(0)
            count += float(weights.sum())
        rms = np.sqrt(total / max(count, 1.0))
        floor = max(float(np.median(rms)) * floor_fraction, 1e-12)
        self.readout_scale = mx.array((1.0 / np.maximum(rms, floor)).astype(np.float32))
        self._freeze_buffers()
        return {
            "median_rms": float(np.median(rms)),
            "min_rms": float(rms.min()),
            "max_rms": float(rms.max()),
            "floored_outputs": int(np.sum(rms < floor)),
        }

    def encode(self, x: mx.array) -> mx.array:
        if self.channels is None:
            return self.encoder(x)
        blocks = [
            encoder(x[:, start:stop])
            for encoder, (start, stop, _) in zip(self.encoders, self.channels, strict=True)
        ]
        return mx.concatenate(blocks, axis=1)

    def input_modules(self) -> list[nn.Module]:
        return [self.encoder] if self.channels is None else list(self.encoders)

    @property
    def state_size(self) -> int:
        return self.dynamics.neurons

    def initial_state(self, batch: int) -> mx.array:
        return self.dynamics.zeros(batch)

    def step(self, obs: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        current = self.encode(self.normalize(obs))
        state, pooled = self.dynamics.advance(state, current, self.neural_steps)
        return mx.tanh(self.decoder(pooled * self.readout_scale)), state

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
            channels=self.channels,
            chunk=self.chunk,
        )
        clone.update(self.parameters())
        clone._freeze_buffers()
        return clone

    def silence_channel(self, kind: str, channel: int) -> BrainPolicy:
        """Lesion: the same trained policy with one sensory channel receiving no current."""
        if self.channels is None or not 0 <= channel < len(self.channels):
            raise ValueError("Policy has no such sensory channel")
        clone = self.with_dynamics(kind, self.dynamics)
        encoder = clone.encoders[channel]
        encoder.weight = mx.zeros_like(encoder.weight)
        encoder.bias = mx.zeros_like(encoder.bias)
        return clone


def gru_hidden_for_budget(obs_dim: int, output_dim: int, budget: int) -> int:
    """Hidden width whose MLX GRU + linear readout is closest to ``budget`` parameters.

    ``output_dim`` is the readout width: action_dim times the action chunk.
    """

    def count(hidden: int) -> int:
        return 3 * hidden * (obs_dim + hidden + 1) + hidden + (hidden + 1) * output_dim

    return min(range(1, 2048), key=lambda hidden: abs(count(hidden) - budget))


class GRUPolicy(_Normalized):
    kind = "gru"

    def input_modules(self) -> list[nn.Module]:
        return [self.cell]

    def __init__(
        self, *, obs_dim: int, action_dim: int, hidden: int, seed: int = 0, chunk: int = 1
    ) -> None:
        super().__init__(obs_dim, action_dim, chunk)
        self.hidden = hidden
        mx.random.seed(seed)
        self.cell = nn.GRU(obs_dim, hidden)
        self.readout = nn.Linear(hidden, self.output_dim)

    @property
    def state_size(self) -> int:
        return self.hidden

    def initial_state(self, batch: int) -> mx.array:
        return mx.zeros((batch, self.hidden))

    def step(self, obs: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self.cell(self.normalize(obs)[:, None, :], hidden=state)[:, -1]
        return mx.tanh(self.readout(hidden)), hidden


class MLPPolicy(_Normalized):
    """Memoryless behavior-cloning MLP (two ReLU layers), the standard D4RL BC architecture."""

    kind = "mlp"

    def __init__(
        self, *, obs_dim: int, action_dim: int, hidden: int = 256, seed: int = 0, chunk: int = 1
    ) -> None:
        super().__init__(obs_dim, action_dim, chunk)
        self.hidden = hidden
        mx.random.seed(seed)
        self.layers = [
            nn.Linear(obs_dim, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, self.output_dim),
        ]

    @property
    def state_size(self) -> int:
        return 0

    def initial_state(self, batch: int) -> mx.array:
        return mx.zeros((batch, 1))

    def step(self, obs: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        x = self.normalize(obs)
        x = nn.relu(self.layers[0](x))
        x = nn.relu(self.layers[1](x))
        return mx.tanh(self.layers[2](x)), state

    def input_modules(self) -> list[nn.Module]:
        return [self.layers[0]]


Slice = tuple[int, int]  # (first observation index, stop index)


class ACTPolicy(_Normalized):
    """ACT (Zhao et al., 2023) on state observations: a reference controller, not a fly model.

    Training adds a CVAE encoder (a transformer over [CLS, joint state, demonstrated chunk])
    whose latent z explains which demonstrated mode the chunk follows. The policy itself is a
    transformer encoder over [z, one token per observation slice] and a transformer decoder
    whose ``chunk`` learned queries emit the action chunk. At test time z = 0, as in ACT. It
    conditions on the current observation only (no recurrent state).
    """

    kind = "act"

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        chunk: int,
        token_slices: Sequence[Slice],
        dims: int = 256,
        heads: int = 8,
        mlp_dims: int = 1024,
        encoder_layers: int = 4,
        decoder_layers: int = 1,
        latent_dim: int = 32,
        dropout: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__(obs_dim, action_dim, chunk)
        slices = tuple((int(start), int(stop)) for start, stop in token_slices)
        if not slices or any(not 0 <= start < stop <= obs_dim for start, stop in slices):
            raise ValueError("ACT token slices must be non-empty ranges inside the observation")
        mx.random.seed(seed)
        self.token_slices = slices
        self.latent_dim = latent_dim
        joint_start, joint_stop = slices[0]

        def embedding(rows: int) -> mx.array:
            return 0.02 * mx.random.normal((rows, dims))

        self.posterior_cls = embedding(1)
        self.posterior_joints = nn.Linear(joint_stop - joint_start, dims)
        self.posterior_actions = nn.Linear(action_dim, dims)
        self.posterior_positions = embedding(chunk + 2)
        self.posterior_encoder = nn.TransformerEncoder(
            encoder_layers, dims, heads, mlp_dims, dropout
        )
        self.posterior_head = nn.Linear(dims, 2 * latent_dim)
        self.latent_in = nn.Linear(latent_dim, dims)
        self.tokens_in = [nn.Linear(stop - start, dims) for start, stop in slices]
        self.token_positions = embedding(len(slices) + 1)
        self.encoder = nn.TransformerEncoder(encoder_layers, dims, heads, mlp_dims, dropout)
        self.queries = embedding(chunk)
        self.decoder = nn.TransformerDecoder(decoder_layers, dims, heads, mlp_dims, dropout)
        self.head = nn.Linear(dims, action_dim)
        self.eval()

    @property
    def state_size(self) -> int:
        return 0

    def initial_state(self, batch: int) -> mx.array:
        return mx.zeros((batch, 1))

    def input_modules(self) -> list[nn.Module]:
        return list(self.tokens_in)

    def posterior(
        self, x: mx.array, chunk_actions: mx.array, valid: mx.array
    ) -> tuple[mx.array, mx.array]:
        """CVAE encoder: (mu, logvar) of z from normalized obs and the [B, chunk, A] targets."""
        batch = x.shape[0]
        start, stop = self.token_slices[0]
        tokens = mx.concatenate(
            [
                mx.broadcast_to(self.posterior_cls[None], (batch, 1, self.posterior_cls.shape[1])),
                self.posterior_joints(x[:, start:stop])[:, None],
                self.posterior_actions(chunk_actions),
            ],
            axis=1,
        )
        keep = mx.concatenate([mx.ones((batch, 2), dtype=mx.bool_), valid.astype(mx.bool_)], 1)
        encoded = self.posterior_encoder(tokens + self.posterior_positions, keep[:, None, None])
        mu, logvar = mx.split(self.posterior_head(encoded[:, 0]), 2, axis=-1)
        return mu, logvar

    def decode(self, x: mx.array, z: mx.array) -> mx.array:
        """Normalized obs and latent -> flattened action chunk in [-1, 1]."""
        tokens = [self.latent_in(z)] + [
            embed(x[:, start:stop])
            for embed, (start, stop) in zip(self.tokens_in, self.token_slices, strict=True)
        ]
        memory = self.encoder(mx.stack(tokens, axis=1) + self.token_positions, None)
        queries = mx.broadcast_to(self.queries[None], (x.shape[0], *self.queries.shape))
        decoded = self.decoder(queries, memory, None, None)
        return mx.tanh(self.head(decoded)).reshape(x.shape[0], self.output_dim)

    def step(self, obs: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        x = self.normalize(obs)
        return self.decode(x, mx.zeros((x.shape[0], self.latent_dim))), state


SequencePolicy = BrainPolicy | GRUPolicy | MLPPolicy | ACTPolicy


class MlxController:
    """Single-episode closed-loop inference; only the action returns to the host.

    With an action chunk k > 1 the executed action is ACT's temporal ensemble: the average of
    the (up to k) predictions made for this step by the last k chunks, oldest weighted
    highest. It is a fixed, parameter-free output filter; for k = 1 it is the prediction.
    """

    def __init__(self, policy: SequencePolicy, *, ensemble_decay: float = ENSEMBLE_DECAY) -> None:
        if ensemble_decay < 0:
            raise ValueError("ensemble_decay must be nonnegative")
        self.policy = policy
        self.ensemble_decay = ensemble_decay
        self.state = policy.initial_state(1)
        self._plans: deque[np.ndarray] = deque(maxlen=policy.chunk)

    @property
    def plan(self) -> np.ndarray | None:
        """The latest predicted chunk, shaped [chunk, action_dim]."""
        return self._plans[-1] if self._plans else None

    def reset(self) -> None:
        self.reset_state()
        self._plans.clear()

    def reset_state(self) -> None:
        """Clear the recurrent state only; queued chunk predictions are kept."""
        self.state = self.policy.initial_state(1)

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = mx.array(np.asarray(observation, dtype=np.float32)[None])
        output, self.state = self.policy.step(obs, self.state)
        mx.eval(output, self.state)
        plan = np.asarray(output[0], dtype=np.float32)
        self._plans.append(plan.reshape(self.policy.chunk, self.policy.action_dim))
        # _plans[i] was predicted len - 1 - i steps ago, so its row for now is len - 1 - i.
        count = len(self._plans)
        rows = np.stack([chunk[count - 1 - i] for i, chunk in enumerate(self._plans)])
        weights = np.exp(-self.ensemble_decay * np.arange(count, dtype=np.float32))
        return (weights @ rows / weights.sum()).astype(np.float32)

    def output_activity(self) -> Any:
        """Current state of the declared output neurons (brain policies only)."""
        if not isinstance(self.policy, BrainPolicy):
            return None
        dynamics = self.policy.dynamics
        return np.asarray(mx.take(self.state, dynamics.output_indices, axis=0))[:, 0]
