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
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from flyarm.whole_brain.policy import ACTPolicy, BrainPolicy, SequencePolicy


@dataclass(frozen=True)
class Budget:
    epochs: int
    decoder_warmup_epochs: int
    batch_size: int
    bptt_steps: int
    learning_rate: float
    deadline: float
    loss: Literal["mse", "l1"] = "mse"
    # Learning rate of the input modules (the encoder into the ascending neurons); None trains
    # them at ``learning_rate``, as every run before the manipulation benchmark did. Adam moves
    # each weight by about the learning rate per early step, so with 217 correlated inputs the
    # current into the ascending neurons grew from RMS 0.48 to 2.2 (tanh saturation) in one
    # epoch at 1e-3; a tenth of it kept 0.5 and fit better (docs/MANIPULATION_ENV.md).
    input_learning_rate: float | None = None
    # Window sampling: each update fits ``window_batch`` windows of ``bptt_steps`` steps drawn
    # uniformly from all demonstrated steps of all episodes, each entered after ``burn_in``
    # steps of the episode run without gradient from the zero state (from the episode's start
    # when it has fewer). None keeps the episode sweep of every run before the manipulation
    # benchmark: batches of whole episodes walked window by window, whose consecutive updates
    # are strongly correlated (docs/MANIPULATION_ENV.md, "Imitation failure analysis").
    window_batch: int | None = None
    burn_in: int = 0


def _optimizer(policy: SequencePolicy, budget: Budget, warmup: bool = False) -> optim.Optimizer:
    """Adam, with the input modules at their own rate when the budget sets one.

    During the decoder-only warmup the input modules are frozen and have no gradients, so one
    Adam serves (a MultiOptimizer cannot initialize a group whose gradient tree is empty).
    """
    if budget.input_learning_rate is None or warmup:
        return optim.Adam(learning_rate=budget.learning_rate)
    inputs = [id(module) for module in policy.input_modules()]
    prefixes = tuple(
        f"{name}." for name, module in policy.named_modules() if id(module) in inputs and name
    )
    return optim.MultiOptimizer(
        [
            optim.Adam(learning_rate=budget.input_learning_rate),
            optim.Adam(learning_rate=budget.learning_rate),
        ],
        [lambda path, _: path.startswith(prefixes)],
    )


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
    # Steps after every episode has ended carry no weight; they are not run.
    steps = _episode_lengths(data["mask"]).max(initial=0)
    obs = mx.array(data["obs"][:, :steps])
    targets = mx.array(targets_np[:, :steps])
    step_weights = mx.array(weights_np[:, :steps])
    state = policy.initial_state(obs.shape[0])
    total = mx.array(0.0)
    for step in range(steps):
        output, state = policy.step(obs[:, step], state)
        error = _step_error(policy, output, targets[:, step], loss)
        total = total + (error * step_weights[:, step]).sum()
        mx.eval(total, state)
    return float(total) / float(weights_np.sum())


def _episode_lengths(mask: np.ndarray) -> np.ndarray:
    """[E] index after each episode's last valid step (0 for an empty episode)."""
    valid = np.asarray(mask) > 0
    return np.where(valid.any(1), valid.shape[1] - np.argmax(valid[:, ::-1], axis=1), 0)


def _batches(
    order: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    generator: np.random.Generator,
    length_buckets: bool,
) -> list[np.ndarray]:
    """Episode batches of one epoch; with ``length_buckets`` similar lengths share a batch.

    A batch runs until its longest episode ends, so mixing a 600-step and a 2,500-step
    episode computes 1,900 masked steps for nothing. Bucketing sorts the shuffled episodes by
    length, cuts consecutive batches and shuffles the batch order.
    """
    if not length_buckets:
        return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    ranked = order[np.argsort(lengths[order], kind="stable")]
    batches = [ranked[start : start + batch_size] for start in range(0, len(ranked), batch_size)]
    return [batches[index] for index in generator.permutation(len(batches))]


def _keep_state(policy: SequencePolicy, state: mx.array, started: np.ndarray) -> mx.array:
    """Zero the state of windows whose episode has not started yet (left padding)."""
    keep = mx.array(started.astype(np.float32))
    if isinstance(policy, BrainPolicy):
        return state * keep[None, :]  # [neurons, batch]
    return state * keep[:, None]


def _window_epoch(
    policy: SequencePolicy,
    observations: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    budget: Budget,
    generator: np.random.Generator,
    gradient_fn: Callable[..., Any],
    optimizer: optim.Optimizer,
) -> list[float]:
    """One epoch of window sampling: as many updates as cover every demonstrated step once."""
    assert budget.window_batch is not None
    starts = np.argwhere(weights.reshape(weights.shape[0], weights.shape[1], -1).sum(-1) > 0)
    horizon = observations.shape[1]
    width, burn = budget.bptt_steps, budget.burn_in
    updates = max(1, int(np.ceil(len(starts) / (budget.window_batch * width))))
    offsets = np.arange(-burn, width)
    losses: list[float] = []
    for _ in range(updates):
        picked = starts[generator.integers(0, len(starts), budget.window_batch)]
        episode, start = picked[:, 0], picked[:, 1]
        steps = start[:, None] + offsets[None, :]  # [W, burn + width]
        inside = (steps >= 0) & (steps < horizon)
        clipped = np.clip(steps, 0, horizon - 1)
        obs = observations[episode[:, None], clipped] * inside[..., None]
        state = policy.initial_state(len(picked))
        for t in range(burn):
            _, state = policy.step(mx.array(obs[:, t]), state)
            state = _keep_state(policy, state, steps[:, t] >= 0)
        state = mx.stop_gradient(state)
        mx.eval(state)
        window = slice(burn, burn + width)
        window_weights = weights[episode[:, None], clipped[:, window]] * inside[:, window, None]
        if float(window_weights.sum()) == 0.0:
            continue
        (loss, _), gradients = gradient_fn(
            policy,
            mx.array(obs[:, window]),
            mx.array(targets[episode[:, None], clipped[:, window]]),
            mx.array(window_weights),
            state,
        )
        gradients, _ = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(policy, gradients)
        mx.eval(policy.parameters(), optimizer.state, loss)
        losses.append(float(loss))
    return losses


@dataclass(frozen=True)
class StepData:
    """Episodes stored end to end on the host (no padding): step arrays and, per episode, the
    index of its first step and its length. DAgger data sets of millions of mostly short
    episodes would not fit padded to the longest one."""

    obs: np.ndarray  # [S, obs_dim]
    actions: np.ndarray  # [S, action_dim]
    weights: np.ndarray  # [S]
    starts: np.ndarray  # [E]
    lengths: np.ndarray  # [E]

    def __post_init__(self) -> None:
        steps = int(self.lengths.sum())
        if len(self.obs) != steps or len(self.actions) != steps or len(self.weights) != steps:
            raise ValueError("step arrays must hold exactly the episodes' steps")
        if not np.array_equal(self.starts, np.concatenate(([0], np.cumsum(self.lengths)[:-1]))):
            raise ValueError("episodes must be stored end to end in order")


def window_indices(
    data: StepData, picked: np.ndarray, burn_in: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flat step indices [W, burn_in + width] of windows ending ``width`` steps after each
    picked step's burn-in, and whether each index lies inside its own episode (steps before
    the episode's start are padding; windows never cross into another episode)."""
    episode = np.searchsorted(data.starts, picked, side="right") - 1
    position = picked - data.starts[episode]
    steps = position[:, None] + np.arange(-burn_in, width)[None, :]
    inside = (steps >= 0) & (steps < data.lengths[episode][:, None])
    clipped = np.clip(steps, 0, data.lengths[episode][:, None] - 1)
    return data.starts[episode][:, None] + clipped, inside


def train_windows(
    policy: SequencePolicy,
    data: StepData,
    budget: Budget,
    updates: int,
    generator: np.random.Generator,
    *,
    warmup_updates: int = 0,
) -> list[float]:
    """``updates`` Adam steps of window sampling on ragged episodes (one action per step).

    Each update fits ``budget.window_batch`` windows of ``budget.bptt_steps`` steps starting at
    uniformly drawn weighted steps, entered after ``budget.burn_in`` steps without gradient;
    the first ``warmup_updates`` train the decoder only. Returns the per-update losses.
    """
    if policy.chunk != 1:
        raise ValueError("train_windows drives one action per step")
    if budget.window_batch is None:
        raise ValueError("train_windows needs budget.window_batch")
    candidates = np.flatnonzero(data.weights > 0)
    if not candidates.size:
        raise ValueError("no weighted steps to train on")
    gradient_fn = nn.value_and_grad(policy, partial(_chunk_loss, loss=budget.loss))
    width, burn = budget.bptt_steps, budget.burn_in
    optimizer: optim.Optimizer | None = None
    losses: list[float] = []
    for update in range(updates):
        if update % 50 == 0 and time.monotonic() >= budget.deadline:
            raise TimeoutError("training budget exhausted; completed rounds are kept")
        warmup = update < warmup_updates
        if optimizer is None or update == warmup_updates:
            _set_input_frozen(policy, warmup)
            optimizer = _optimizer(policy, budget, warmup)
        picked = candidates[generator.integers(0, len(candidates), budget.window_batch)]
        flat, inside = window_indices(data, picked, burn, width)
        obs = data.obs[flat] * inside[..., None]
        state = policy.initial_state(len(picked))
        for t in range(burn):
            _, state = policy.step(mx.array(obs[:, t]), state)
            state = _keep_state(policy, state, inside[:, t])
        state = mx.stop_gradient(state)
        window = slice(burn, burn + width)
        weights = (data.weights[flat[:, window]] * inside[:, window])[..., None]
        if float(weights.sum()) == 0.0:
            continue
        (loss, _), gradients = gradient_fn(
            policy,
            mx.array(obs[:, window]),
            mx.array(data.actions[flat[:, window]][:, :, None, :]),
            mx.array(weights.astype(np.float32)),
            state,
        )
        gradients, _ = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(policy, gradients)
        mx.eval(policy.parameters(), optimizer.state, loss)
        losses.append(float(loss))
    _set_input_frozen(policy, False)
    return losses


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
    selector: Callable[[SequencePolicy], float] | None = None,
    select_every: int = 1,
    length_buckets: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns learning curves and the selected checkpoint's summary.

    Without ``selector`` the kept weights are those with the lowest validation loss. With it,
    every ``select_every`` epochs (and at the last epoch) ``selector(policy)`` scores the
    current weights, for example by closed-loop success on held-out validation episodes;
    the highest score wins and validation loss breaks ties.

    The episodes stay in host memory and each batch is copied to the device when it is used,
    cut at its longest episode, so long-horizon data sets never sit on the GPU whole.
    """
    if select_every < 1:
        raise ValueError("select_every must be positive")
    if set_normalization:
        samples = train_data["obs"][train_data["mask"].astype(bool)]
        policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
    observations = train_data["obs"]
    targets, weights = chunk_targets(
        train_data["actions"], train_data["mask"], train_weights, policy.chunk
    )
    lengths = _episode_lengths(train_data["mask"])
    episodes = train_data["obs"].shape[0]
    generator = np.random.default_rng(seed)
    gradient_fn = nn.value_and_grad(policy, partial(_chunk_loss, loss=budget.loss))
    best_key: tuple[float, float] = (float("inf"), float("inf"))
    best_loss, best_epoch, best_score = float("inf"), -1, None
    best_parameters = policy.parameters()
    curves: list[dict[str, Any]] = []
    started = time.monotonic()
    optimizer: optim.Optimizer | None = None
    for epoch in range(budget.epochs):
        if time.monotonic() >= budget.deadline:
            raise TimeoutError("Whole-brain budget exhausted; partial artifacts retained")
        warmup = epoch < budget.decoder_warmup_epochs
        _set_input_frozen(policy, warmup)
        if optimizer is None or epoch == budget.decoder_warmup_epochs:
            # MLX optimizers cannot absorb parameters unfrozen after initialization.
            optimizer = _optimizer(policy, budget, warmup)
        if budget.window_batch is not None:
            losses = _window_epoch(
                policy, observations, targets, weights, budget, generator, gradient_fn, optimizer
            )
        else:
            losses = []
            order = generator.permutation(episodes)
            for rows in _batches(order, lengths, budget.batch_size, generator, length_buckets):
                horizon = int(lengths[rows].max())
                batch_obs = mx.array(observations[rows, :horizon])
                batch_targets = mx.array(targets[rows, :horizon])
                batch_weights = mx.array(weights[rows, :horizon])
                state = policy.initial_state(len(rows))
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
        score = None
        if selector is not None:
            if (epoch + 1) % select_every and epoch + 1 != budget.epochs:
                continue
            score = float(selector(policy))
            curves[-1]["selection_score"] = score
            print(f"    selection score {score:.3f}", flush=True)
        key = (-score if score is not None else 0.0, validation_loss)
        if key < best_key:
            best_key, best_parameters = key, policy.parameters()
            best_loss, best_epoch, best_score = validation_loss, epoch + 1, score
    _set_input_frozen(policy, False)
    policy.update(best_parameters)
    return curves, {
        "best_epoch": best_epoch,
        "loss": budget.loss,
        "best_validation_loss": best_loss,
        "selection": "closed_loop_validation" if selector is not None else "validation_loss",
        "selection_score": best_score,
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
    *,
    phase: str = "act",
    set_normalization: bool = True,
    selector: Callable[[SequencePolicy], float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """ACT's recipe: frame-level minibatches, L1 + beta * KL, AdamW, best-validation weights.

    ``selector`` works as in train_sequence_policy, scored at every validation point.
    """
    if set_normalization:
        samples = train_data["obs"][train_data["mask"].astype(bool)]
        policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
    obs, targets, valid = _act_frames(train_data, policy.chunk)
    val_obs, val_targets, val_valid = _act_frames(validation_data, policy.chunk)
    generator = np.random.default_rng(seed)
    mx.random.seed(seed)
    optimizer = optim.AdamW(learning_rate=budget.learning_rate, weight_decay=budget.weight_decay)
    gradient_fn = nn.value_and_grad(policy, partial(_act_loss, kl_weight=budget.kl_weight))
    best_key: tuple[float, float] = (float("inf"), float("inf"))
    best_loss, best_step, best_score = float("inf"), 0, None
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
                "phase": phase,
                "train_loss": float(train_l1),
                "train_kl": float(train_kl),
                "validation_loss": validation_loss,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        print(
            f"  {phase} step {step}/{budget.steps} l1 train={train_l1:.5f} kl={train_kl:.3f} "
            f"val={validation_loss:.5f} ({curves[-1]['elapsed_seconds']:.0f}s)",
            flush=True,
        )
        score = float(selector(policy)) if selector is not None else None
        if score is not None:
            curves[-1]["selection_score"] = score
            print(f"    selection score {score:.3f}", flush=True)
        key = (-score if score is not None else 0.0, validation_loss)
        if key < best_key:
            best_key, best_parameters = key, policy.parameters()
            best_loss, best_step, best_score = validation_loss, step, score
    policy.update(best_parameters)
    policy.eval()
    return curves, {
        "best_step": best_step,
        "loss": "l1",
        "best_validation_loss": best_loss,
        "selection": "closed_loop_validation" if selector is not None else "validation_loss",
        "selection_score": best_score,
        "training_seconds": time.monotonic() - started,
    }
