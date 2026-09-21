"""Masked sequence behavior cloning in MLX with truncated BPTT.

Mirrors the torch trainers: normalization from training frames, Adam, gradient-norm clip
1.0, state carried across chunks but detached between them, best-validation checkpoint.
Two B1a-specific choices: short BPTT windows (the adjoint runs through the full 166k-neuron
state), and an optional decoder-only warmup before the encoder joins training.

Chunked policies (ACT-style) are trained on the next ``chunk`` demonstrated actions at every
step; targets past the end of an episode are masked, as ACT masks its padding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from flyarm.whole_brain.policy import ACTPolicy, SequencePolicy


@dataclass(frozen=True)
class Budget:
    epochs: int
    decoder_warmup_epochs: int
    batch_size: int
    bptt_steps: int
    learning_rate: float
    deadline: float
    loss: Literal["mse", "l1"] = "mse"


def _set_input_frozen(policy: SequencePolicy, frozen: bool) -> None:
    for module in policy.input_modules():
        module.freeze() if frozen else module.unfreeze()


def chunk_targets(
    actions: np.ndarray, mask: np.ndarray, weights: np.ndarray, chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    """Targets [E, T, chunk, A] = actions[t + j]; weight = weights[t] while step t + j exists."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    episodes, horizon, dim = actions.shape
    padded = np.concatenate((actions, np.zeros((episodes, chunk - 1, dim), actions.dtype)), 1)
    valid = np.concatenate((mask > 0, np.zeros((episodes, chunk - 1), bool)), 1)
    targets = np.stack([padded[:, j : j + horizon] for j in range(chunk)], axis=2)
    exists = np.stack([valid[:, j : j + horizon] for j in range(chunk)], axis=2)
    return targets.astype(np.float32), (weights[..., None] * exists).astype(np.float32)


def _step_error(policy: SequencePolicy, output: mx.array, targets: mx.array, loss: str) -> mx.array:
    """Per-chunk-position error [batch, chunk] of one control step."""
    predicted = output.reshape(output.shape[0], policy.chunk, policy.action_dim)
    difference = predicted - targets
    return (mx.abs(difference) if loss == "l1" else difference**2).mean(-1)


def _chunk_loss(
    policy: SequencePolicy,
    obs: mx.array,
    targets: mx.array,
    weights: mx.array,
    state: mx.array,
    *,
    loss: str,
) -> tuple[mx.array, mx.array]:
    total = mx.array(0.0)
    for step in range(obs.shape[1]):
        output, state = policy.step(obs[:, step], state)
        error = _step_error(policy, output, targets[:, step], loss)
        total = total + (error * weights[:, step]).sum()
    return total / weights.sum(), state


def sequence_loss(
    policy: SequencePolicy,
    data: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    loss: str = "mse",
) -> float:
    """Weighted per-step chunk error over complete episodes, no gradient."""
    targets_np, weights_np = chunk_targets(data["actions"], data["mask"], weights, policy.chunk)
    obs = mx.array(data["obs"])
    targets = mx.array(targets_np)
    step_weights = mx.array(weights_np)
    state = policy.initial_state(obs.shape[0])
    total = mx.array(0.0)
    for step in range(obs.shape[1]):
        output, state = policy.step(obs[:, step], state)
        error = _step_error(policy, output, targets[:, step], loss)
        total = total + (error * step_weights[:, step]).sum()
        mx.eval(total, state)
    return float(total) / float(weights_np.sum())


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
    targets_np, weights_np = chunk_targets(
        train_data["actions"], train_data["mask"], train_weights, policy.chunk
    )
    targets, weights = mx.array(targets_np), mx.array(weights_np)
    episodes, horizon = train_data["obs"].shape[:2]
    generator = np.random.default_rng(seed)
    gradient_fn = nn.value_and_grad(policy, partial(_chunk_loss, loss=budget.loss))
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
        validation_loss = sequence_loss(
            policy, validation_data, validation_weights, loss=budget.loss
        )
        curves.append(
            {
                "epoch": epoch + 1,
                "phase": phase,
                "decoder_only": warmup,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(
            f"  {phase} epoch {epoch + 1}/{budget.epochs} "
            f"{budget.loss} train={curves[-1]['train_loss']:.5f} val={validation_loss:.5f} "
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
        "loss": budget.loss,
        "best_validation_loss": best_loss,
        "training_seconds": time.monotonic() - started,
    }


@dataclass(frozen=True)
class ACTBudget:
    steps: int
    batch_size: int
    learning_rate: float
    kl_weight: float
    eval_every: int
    deadline: float
    weight_decay: float = 1e-4


def _act_frames(
    data: dict[str, np.ndarray], chunk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets, weights = chunk_targets(data["actions"], data["mask"], data["mask"], chunk)
    rows, steps = np.nonzero(data["mask"] > 0)
    return data["obs"][rows, steps], targets[rows, steps], weights[rows, steps]


def _act_loss(
    policy: ACTPolicy, obs: mx.array, targets: mx.array, valid: mx.array, *, kl_weight: float
) -> tuple[mx.array, mx.array, mx.array]:
    x = policy.normalize(obs)
    mu, logvar = policy.posterior(x, targets, valid)
    z = mu + mx.exp(0.5 * logvar) * mx.random.normal(mu.shape)
    predicted = policy.decode(x, z).reshape(targets.shape)
    l1 = (mx.abs(predicted - targets).mean(-1) * valid).sum() / valid.sum()
    kl = (-0.5 * (1 + logvar - mu**2 - mx.exp(logvar)).sum(-1)).mean()
    return l1 + kl_weight * kl, l1, kl


def _act_validation(
    policy: ACTPolicy, obs: np.ndarray, targets: np.ndarray, valid: np.ndarray
) -> float:
    """L1 of the test-time policy (z = 0, no dropout) over validation frames."""
    total, count = 0.0, 0.0
    for start in range(0, len(obs), 1024):
        rows = slice(start, start + 1024)
        output, _ = policy.step(mx.array(obs[rows]), policy.initial_state(len(obs[rows])))
        predicted = np.asarray(output).reshape(targets[rows].shape)
        error = np.abs(predicted - targets[rows]).mean(-1)
        total += float((error * valid[rows]).sum())
        count += float(valid[rows].sum())
    return total / count


def train_act_policy(
    policy: ACTPolicy,
    train_data: dict[str, np.ndarray],
    validation_data: dict[str, np.ndarray],
    budget: ACTBudget,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """ACT's recipe: frame-level minibatches, L1 + beta * KL, AdamW, best-validation weights."""
    samples = train_data["obs"][train_data["mask"].astype(bool)]
    policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
    obs, targets, valid = _act_frames(train_data, policy.chunk)
    val_obs, val_targets, val_valid = _act_frames(validation_data, policy.chunk)
    generator = np.random.default_rng(seed)
    mx.random.seed(seed)
    optimizer = optim.AdamW(learning_rate=budget.learning_rate, weight_decay=budget.weight_decay)
    gradient_fn = nn.value_and_grad(policy, partial(_act_loss, kl_weight=budget.kl_weight))
    best_loss, best_step = float("inf"), 0
    best_parameters = policy.parameters()
    curves: list[dict[str, Any]] = []
    started = time.monotonic()
    window: list[tuple[float, float]] = []
    for step in range(1, budget.steps + 1):
        if time.monotonic() >= budget.deadline:
            raise TimeoutError("ACT budget exhausted; partial artifacts retained")
        policy.train()
        batch = generator.integers(0, len(obs), budget.batch_size)
        (_, l1, kl), gradients = gradient_fn(
            policy, mx.array(obs[batch]), mx.array(targets[batch]), mx.array(valid[batch])
        )
        gradients, _ = optim.clip_grad_norm(gradients, 10.0)
        optimizer.update(policy, gradients)
        mx.eval(policy.parameters(), optimizer.state, l1, kl)
        window.append((float(l1), float(kl)))
        if step % budget.eval_every and step != budget.steps:
            continue
        policy.eval()
        validation_loss = _act_validation(policy, val_obs, val_targets, val_valid)
        train_l1, train_kl = np.mean(window, axis=0)
        window.clear()
        curves.append(
            {
                "step": step,
                "phase": "act",
                "train_loss": float(train_l1),
                "train_kl": float(train_kl),
                "validation_loss": validation_loss,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(
            f"  act step {step}/{budget.steps} l1 train={train_l1:.5f} kl={train_kl:.3f} "
            f"val={validation_loss:.5f} ({curves[-1]['elapsed_seconds']:.0f}s)",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss, best_step = validation_loss, step
            best_parameters = policy.parameters()
    policy.update(best_parameters)
    policy.eval()
    return curves, {
        "best_step": best_step,
        "loss": "l1",
        "best_validation_loss": best_loss,
        "training_seconds": time.monotonic() - started,
    }
