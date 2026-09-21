"""Masked sequence behavior cloning in MLX with truncated BPTT.

Mirrors the torch trainers: normalization from training frames, Adam, gradient-norm clip
1.0, state carried across chunks but detached between them, best-validation checkpoint.
Two B1a-specific choices: short BPTT windows (the adjoint runs through the full 166k-neuron
state), and an optional decoder-only warmup before the encoder joins training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from flyarm.whole_brain.policy import SequencePolicy


@dataclass(frozen=True)
class Budget:
    epochs: int
    decoder_warmup_epochs: int
    batch_size: int
    bptt_steps: int
    learning_rate: float
    deadline: float


def _set_input_frozen(policy: SequencePolicy, frozen: bool) -> None:
    for module in policy.input_modules():
        module.freeze() if frozen else module.unfreeze()


def _chunk_loss(
    policy: SequencePolicy,
    obs: mx.array,
    targets: mx.array,
    weights: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    total = mx.array(0.0)
    for step in range(obs.shape[1]):
        action, state = policy.step(obs[:, step], state)
        error = ((action - targets[:, step]) ** 2).mean(-1)
        total = total + (error * weights[:, step]).sum()
    return total / weights.sum(), state


def sequence_loss(
    policy: SequencePolicy, data: dict[str, np.ndarray], weights: np.ndarray
) -> float:
    """Weighted per-step MSE over complete episodes, no gradient."""
    obs = mx.array(data["obs"])
    targets = mx.array(data["actions"])
    step_weights = mx.array(weights.astype(np.float32))
    state = policy.initial_state(obs.shape[0])
    total = mx.array(0.0)
    for step in range(obs.shape[1]):
        action, state = policy.step(obs[:, step], state)
        error = ((action - targets[:, step]) ** 2).mean(-1)
        total = total + (error * step_weights[:, step]).sum()
        mx.eval(total, state)
    return float(total) / float(weights.sum())


def train_sequence_policy(
    policy: SequencePolicy,
    train_data: dict[str, np.ndarray],
    train_weights: np.ndarray,
    validation_data: dict[str, np.ndarray],
    validation_weights: np.ndarray,
    budget: Budget,
    seed: int,
    *,
    phase: str = "behavior_cloning",
    set_normalization: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set_normalization:
        samples = train_data["obs"][train_data["mask"].astype(bool)]
        policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
    observations = mx.array(train_data["obs"])
    targets = mx.array(train_data["actions"])
    weights = mx.array(train_weights.astype(np.float32))
    episodes, horizon = train_data["obs"].shape[:2]
    generator = np.random.default_rng(seed)
    gradient_fn = nn.value_and_grad(policy, _chunk_loss)
    best_loss, best_epoch = float("inf"), -1
    best_parameters = policy.parameters()
    curves: list[dict[str, Any]] = []
    started = time.monotonic()
    optimizer: optim.Adam | None = None
    for epoch in range(budget.epochs):
        if time.monotonic() >= budget.deadline:
            raise TimeoutError("Whole-brain budget exhausted; partial artifacts retained")
        warmup = epoch < budget.decoder_warmup_epochs
        _set_input_frozen(policy, warmup)
        if optimizer is None or epoch == budget.decoder_warmup_epochs:
            # MLX optimizers cannot absorb parameters unfrozen after initialization.
            optimizer = optim.Adam(learning_rate=budget.learning_rate)
        losses: list[float] = []
        order = generator.permutation(episodes)
        for start in range(0, episodes, budget.batch_size):
            batch = mx.array(order[start : start + budget.batch_size])
            batch_obs, batch_targets = observations[batch], targets[batch]
            batch_weights = weights[batch]
            state = policy.initial_state(batch.size)
            for time_start in range(0, horizon, budget.bptt_steps):
                time_stop = min(time_start + budget.bptt_steps, horizon)
                chunk_weights = batch_weights[:, time_start:time_stop]
                if float(chunk_weights.sum()) == 0.0:
                    break
                (loss, state), gradients = gradient_fn(
                    policy,
                    batch_obs[:, time_start:time_stop],
                    batch_targets[:, time_start:time_stop],
                    chunk_weights,
                    state,
                )
                gradients, _ = optim.clip_grad_norm(gradients, 1.0)
                optimizer.update(policy, gradients)
                state = mx.stop_gradient(state)
                mx.eval(policy.parameters(), optimizer.state, state, loss)
                losses.append(float(loss))
        validation_loss = sequence_loss(policy, validation_data, validation_weights)
        curves.append(
            {
                "epoch": epoch + 1,
                "phase": phase,
                "decoder_only": warmup,
                "train_mse": float(np.mean(losses)),
                "validation_mse": validation_loss,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(
            f"  {phase} epoch {epoch + 1}/{budget.epochs} "
            f"train={curves[-1]['train_mse']:.5f} val={validation_loss:.5f} "
            f"({curves[-1]['elapsed_seconds']:.0f}s)",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_parameters = policy.parameters()
    _set_input_frozen(policy, False)
    policy.update(best_parameters)
    return curves, {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "training_seconds": time.monotonic() - started,
    }
