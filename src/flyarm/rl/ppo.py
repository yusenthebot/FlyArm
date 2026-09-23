"""PPO fine-tuning of the motor interface of a trained connectome controller (MLX).

The controller stays the FlyArm policy: frozen linear encoder, frozen complete connectome,
linear decoder. PPO trains only the decoder (motor neurons -> action mean) and a per-action
exploration scale, so no gradient passes through the brain and the connectome is never
changed. The critic reads privileged simulator state and exists only during training; the
exploration noise is training-only too, and evaluation runs the deterministic mean action.

The trainer itself is task-agnostic: a ``TaskAdapter`` supplies the batched environment, the
observation and action dimensions and the benchmark's own evaluation, so the same loop runs
against BatchedPickPlace (flyarm.rl.ppo.PickPlaceTask) and BatchedKitchen
(flyarm.rl.ppo_kitchen.KitchenTask).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from flyarm.config import KitchenPPOConfig, PPOConfig, TaskVariantConfig
from flyarm.rl.batched_pick_place import ACTION_DIM, OBS_DIM, BatchedPickPlace, TaskVariant
from flyarm.whole_brain.policy import BrainPolicy, DirectPolicy

TRAIN_SEED = 1_000_000  # environment seeds for PPO rollouts, disjoint from every B1a split
PPOSettings = PPOConfig | KitchenPPOConfig


class MotorHead(nn.Module):
    """The trainable part: the decoder over motor-neuron activity plus exploration scale."""

    def __init__(self, decoder: nn.Linear, log_std: float, action_dim: int | None = None) -> None:
        super().__init__()
        self.decoder = decoder
        self.action_dim = int(decoder.weight.shape[0]) if action_dim is None else action_dim
        self.log_std = mx.full((self.action_dim,), log_std)

    def mean(self, features: mx.array) -> mx.array:
        return mx.tanh(self.decoder(features))


class Critic(nn.Module):
    def __init__(self, input_dim: int = OBS_DIM + 1, hidden: int = 256) -> None:
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden), nn.Linear(hidden, hidden)]
        self.value = nn.Linear(hidden, 1)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.tanh(layer(x))
        return self.value(x)[..., 0]


def gaussian_log_prob(actions: mx.array, mean: mx.array, log_std: mx.array) -> mx.array:
    variance = mx.exp(2 * log_std)
    return (-((actions - mean) ** 2) / (2 * variance) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


def normalized_advantages(advantages: np.ndarray, clip: float = 0.0) -> np.ndarray:
    """Standardize advantages, optionally clipping the tail to ``clip`` standard deviations.

    A sparse terminal bonus that is rare and far larger than the per-step terms leaves a few
    samples tens of standard deviations out, and the first policy update after critic warmup
    then moves the policy far outside the trust region and lands it on a dead fixed point
    (research log E39: one update with a mean ratio deviation of 0.94 against a clip of 0.2).
    Clipping the standardized advantages bounds that update; 0 leaves them unclipped, as in
    every run before E39.
    """
    if clip < 0:
        raise ValueError("advantage_clip must be non-negative")
    standardized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return np.clip(standardized, -clip, clip) if clip else standardized


def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation over [T, N] arrays; dones end the episode after t."""
    advantages = np.zeros_like(rewards)
    carry = np.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        following = last_value if t == len(rewards) - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * following * alive - values[t]
        carry = delta + gamma * lam * alive * carry
        advantages[t] = carry
    return advantages, advantages + values


class BrainRollout:
    """Frozen encoder + connectome for a batch of environments; returns motor features."""

    def __init__(self, policy: BrainPolicy, num_envs: int) -> None:
        self.policy = policy
        self.state = policy.initial_state(num_envs)
        self.feature_dim = policy.dynamics.output_count

    def features(self, obs: np.ndarray) -> mx.array:
        current = self.policy.encode(self.policy.normalize(mx.array(obs)))
        self.state, pooled = self.policy.dynamics.advance(
            self.state, current, self.policy.neural_steps
        )
        features = self.policy.readout(pooled)
        mx.eval(self.state, features)
        return features

    def reset(self, done: np.ndarray) -> None:
        if done.any():
            self.state = self.state * mx.array((~done).astype(np.float32))[None, :]


class DirectRollout:
    """The deep-RL control's features: the normalized observation itself, no memory."""

    def __init__(self, policy: DirectPolicy, num_envs: int) -> None:
        self.policy = policy
        self.state = policy.initial_state(num_envs)
        self.feature_dim = policy.obs_dim

    def features(self, obs: np.ndarray) -> mx.array:
        features = self.policy.normalize(mx.array(np.asarray(obs, dtype=np.float32)))
        mx.eval(features)
        return features

    def reset(self, done: np.ndarray) -> None:
        pass


Controller = BrainPolicy | DirectPolicy


def rollout_for(policy: Controller, num_envs: int) -> BrainRollout | DirectRollout:
    if isinstance(policy, DirectPolicy):
        return DirectRollout(policy, num_envs)
    return BrainRollout(policy, num_envs)


class RunningNorm:
    """Exponential running mean and scale of the critic input (training only)."""

    def __init__(self, momentum: float = 0.99) -> None:
        self.momentum = momentum
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def update(self, samples: np.ndarray) -> None:
        mean, scale = samples.mean(0), samples.std(0)
        if self.mean is None or self.scale is None:
            self.mean, self.scale = mean, scale
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
            self.scale = self.momentum * self.scale + (1 - self.momentum) * scale
        self.scale = np.maximum(self.scale, 1e-3)

    def __call__(self, states: np.ndarray) -> mx.array:
        if self.mean is None or self.scale is None:
            raise RuntimeError("RunningNorm used before update")
        return mx.array((states - self.mean) / self.scale)


def _critic_input(obs: np.ndarray, steps: np.ndarray, horizon: int) -> np.ndarray:
    return np.concatenate((obs, (steps / horizon)[:, None]), axis=1).astype(np.float32)


def task_variant(config: TaskVariantConfig) -> TaskVariant:
    return TaskVariant(**config.model_dump())


class TaskAdapter(Protocol):
    """What PPO needs to know about one batched benchmark.

    ``score`` returns one entry per evaluation variant; every entry carries "success_rate",
    which is what the checkpoint selection in train_ppo compares.
    """

    obs_dim: int
    privileged_dim: int
    action_dim: int
    extra_key: str  # curve field for the task's secondary per-episode counter
    extra_label: str  # its name in the progress line
    # Optional: a per-batch peak worth recording, such as the most tasks any one episode earned.
    batch_peak_key: str | None

    def make_env(self, num_envs: int, first_seed: int) -> Any: ...

    def score(
        self, policy: BrainPolicy, head: MotorHead, seeds: list[int]
    ) -> dict[str, dict[str, Any]]: ...

    def extra(self, result: Any, done: np.ndarray) -> int: ...

    def batch_peak(self, result: Any, done: np.ndarray) -> float: ...

    def describe(self, name: str, scored: dict[str, Any], episodes: int) -> str: ...


def evaluate(
    policy: Controller,
    head: MotorHead,
    model_path: Path,
    seeds: list[int],
    horizon: int,
    variant: TaskVariant | None = None,
) -> dict[str, Any]:
    """Deterministic mean actions on fixed seeds; the benchmark's own success rule."""
    env = BatchedPickPlace(model_path, len(seeds), horizon=horizon, variant=variant)
    obs = env.reset(seeds=np.array(seeds))
    brain = rollout_for(policy, len(seeds))
    active = np.ones(len(seeds), dtype=bool)
    success = np.zeros(len(seeds), dtype=bool)
    grasped = np.zeros(len(seeds), dtype=bool)
    lifted = np.zeros(len(seeds), dtype=bool)
    for _ in range(horizon):
        action = np.asarray(head.mean(brain.features(obs)), dtype=np.float64)
        result = env.step(action, auto_reset=False)
        grasped |= active & env.ever_grasped
        lifted |= active & result.lifted
        success |= active & result.success
        active &= ~(result.success | result.truncated)
        obs = result.obs
        if not active.any():
            break
    return {
        "seeds": seeds,
        "success_rate": float(success.mean()),
        "grasp_rate": float(grasped.mean()),
        "lift_rate": float(lifted.mean()),
        "successes": int(success.sum()),
        "lifts": int(lifted.sum()),
        "grasps": int(grasped.sum()),
    }


class PickPlaceTask:
    """The B1a pick-and-place adapter: BatchedPickPlace with the config's task variants."""

    obs_dim = OBS_DIM
    privileged_dim = OBS_DIM
    action_dim = ACTION_DIM
    extra_key = "lift_rate"
    extra_label = "lift"
    batch_peak_key: str | None = None

    def __init__(self, model_path: Path, settings: PPOConfig) -> None:
        self.model_path = Path(model_path)
        self.settings = settings

    def make_env(self, num_envs: int, first_seed: int) -> BatchedPickPlace:
        return BatchedPickPlace(
            self.model_path,
            num_envs,
            horizon=self.settings.horizon,
            first_seed=first_seed,
            variant=task_variant(self.settings.train_variant),
            success_bonus=self.settings.success_bonus,
            reach_slope=self.settings.reach_slope,
        )

    def score(
        self, policy: BrainPolicy, head: MotorHead, seeds: list[int]
    ) -> dict[str, dict[str, Any]]:
        return {
            name: evaluate(
                policy, head, self.model_path, seeds, self.settings.horizon, task_variant(variant)
            )
            for name, variant in self.settings.eval_variants.items()
        }

    def extra(self, result: Any, done: np.ndarray) -> int:
        return int((done & result.lifted).sum())

    def batch_peak(self, result: Any, done: np.ndarray) -> float:
        return 0.0  # pick-and-place has no per-episode count to peak over

    def describe(self, name: str, scored: dict[str, Any], episodes: int) -> str:
        return f"{name} place {scored['successes']}/{episodes} lift {scored['lifts']}"


def encoder_pass(
    policy: BrainPolicy,
    head: MotorHead,
    optimizer: optim.Optimizer,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    done: np.ndarray,
    state: mx.array,
    max_grad_norm: float,
) -> float:
    """One on-policy REINFORCE-with-baseline pass over the rollout for the encoder.

    Gradients through the recurrent connectome are truncated to the current control step: the
    incoming state is a constant, so only that step's 3 neural updates are differentiated and
    just one step's graph is alive at a time. The rollout's observations replay the same states
    (the dynamics are deterministic), so nothing extra has to be stored during the rollout.
    """
    if policy.channels is not None:
        raise ValueError("Encoder training expects the single-encoder B1a interface")

    def loss_fn(
        encoder: nn.Module, state: mx.array, obs: mx.array, action: mx.array, weight: mx.array
    ) -> tuple[mx.array, mx.array]:
        current = encoder(policy.normalize(obs))
        state, pooled = policy.dynamics.advance(state, current, policy.neural_steps)
        log_prob = gaussian_log_prob(action, head.mean(policy.readout(pooled)), head.log_std)
        return -(weight * log_prob).mean(), state

    gradient_fn = nn.value_and_grad(policy.encoder, loss_fn)
    total = 0.0
    for step in range(len(observations)):
        (loss, state), grads = gradient_fn(
            policy.encoder,
            mx.stop_gradient(state),
            mx.array(observations[step]),
            mx.array(actions[step]),
            mx.array(advantages[step]),
        )
        grads, _ = optim.clip_grad_norm(grads, max_grad_norm)
        optimizer.update(policy.encoder, grads)
        state = state * mx.array(1.0 - done[step])[None, :]
        mx.eval(policy.encoder.parameters(), optimizer.state, state)
        total += float(loss)
    return total / max(len(observations), 1)


def train_ppo(
    policy: Controller,
    task: TaskAdapter,
    output: Path,
    settings: PPOSettings,
    eval_seeds: list[int],
    select_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Fine-tune ``policy``'s decoder in place with PPO; returns curves and evaluations.

    At every evaluation point the variants are scored on ``eval_seeds`` (reported) and, when
    given, on ``select_seeds`` (disjoint validation seeds); the kept "best" checkpoint is
    chosen on the validation score only, and its test score is what gets reported.
    """
    output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(settings.seed)
    generator = np.random.default_rng(settings.seed)
    head = MotorHead(policy.decoder, settings.log_std, task.action_dim)
    critic = Critic(task.privileged_dim + 1)
    head_optimizer = optim.Adam(learning_rate=settings.decoder_lr)
    critic_optimizer = optim.Adam(learning_rate=settings.critic_lr)
    encoder_optimizer = optim.Adam(learning_rate=settings.encoder_lr)
    env = task.make_env(settings.num_envs, TRAIN_SEED + 100_000 * settings.seed)
    obs = env.reset()
    privileged = env.observation(privileged=True)
    brain = rollout_for(policy, settings.num_envs)
    n, horizon = settings.num_envs, settings.rollout_steps
    critic_norm = RunningNorm()

    def losses(
        head: MotorHead,
        critic: Critic,
        feats: mx.array,
        states: mx.array,
        actions: mx.array,
        old_log_prob: mx.array,
        advantages: mx.array,
        returns: mx.array,
        train_policy: bool,
    ) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array]]:
        log_prob = gaussian_log_prob(actions, head.mean(feats), head.log_std)
        ratio = mx.exp(log_prob - old_log_prob)
        clipped = mx.clip(ratio, 1 - settings.clip, 1 + settings.clip)
        policy_loss = -mx.minimum(ratio * advantages, clipped * advantages).mean()
        value_loss = ((critic(states) - returns) ** 2).mean()
        entropy = (head.log_std + 0.5 * np.log(2 * np.pi * np.e)).sum()
        total = settings.value_coef * value_loss - settings.entropy_coef * entropy
        if train_policy:
            total = total + policy_loss
        return total, (policy_loss, value_loss, ratio)

    class Pair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head, self.critic = head, critic

    pair = Pair()
    grad_fn = nn.value_and_grad(pair, lambda p, *args: losses(p.head, p.critic, *args))

    curves: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    started = time.monotonic()
    env_steps = 0
    best: dict[str, Any] = {"success_rate": -1.0}
    for iteration in range(settings.iterations):
        feats_buf = np.zeros((horizon, n, brain.feature_dim), np.float32)
        obs_buf = np.zeros((horizon, n, task.obs_dim), np.float32)
        rollout_state = brain.state
        critic_buf = np.zeros((horizon, n, task.privileged_dim + 1), np.float32)
        action_buf = np.zeros((horizon, n, task.action_dim), np.float32)
        logp_buf = np.zeros((horizon, n), np.float32)
        reward_buf = np.zeros((horizon, n), np.float32)
        done_buf = np.zeros((horizon, n), np.float32)
        finished = successes = extras = 0
        peak = 0.0
        rollout_start = time.monotonic()
        for t in range(horizon):
            feats = brain.features(obs)
            mean = head.mean(feats)
            std = mx.exp(head.log_std)
            actions = mx.clip(mean + std * mx.random.normal(mean.shape), -1.0, 1.0)
            log_prob = gaussian_log_prob(actions, mean, head.log_std)
            mx.eval(actions, log_prob)
            feats_buf[t] = np.asarray(feats)
            obs_buf[t] = obs
            critic_buf[t] = _critic_input(privileged, env.steps, settings.horizon)
            action_buf[t] = np.asarray(actions)
            logp_buf[t] = np.asarray(log_prob)
            result = env.step(action_buf[t].astype(np.float64))
            done = result.terminated | result.truncated
            reward_buf[t], done_buf[t] = result.reward, done
            finished += int(done.sum())
            successes += int(result.terminated.sum())
            extras += task.extra(result, done)
            if getattr(task, "batch_peak_key", None):
                peak = max(peak, task.batch_peak(result, done))
            brain.reset(done)
            obs = result.obs
            privileged = env.observation(privileged=True)
        env_steps += n * horizon
        rollout_seconds = time.monotonic() - rollout_start

        flat_states = critic_buf.reshape(-1, task.privileged_dim + 1)
        critic_norm.update(flat_states)
        normalized = critic_norm

        values = np.asarray(critic(normalized(flat_states))).reshape(horizon, n)
        last_value = np.asarray(
            critic(normalized(_critic_input(privileged, env.steps, settings.horizon)))
        )
        advantages, returns = gae(
            reward_buf, values, done_buf, last_value, settings.gamma, settings.lam
        )
        samples = horizon * n
        data = {
            "feats": mx.array(feats_buf.reshape(samples, -1)),
            "states": normalized(flat_states),
            "actions": mx.array(action_buf.reshape(samples, task.action_dim)),
            "logp": mx.array(logp_buf.reshape(samples)),
            "adv": mx.array(
                normalized_advantages(advantages, settings.advantage_clip).reshape(samples)
            ),
            "ret": mx.array(returns.reshape(samples)),
        }
        train_policy = iteration >= settings.critic_warmup
        encoder_loss = None
        if settings.encoder_lr > 0 and train_policy and isinstance(policy, BrainPolicy):
            encoder_loss = encoder_pass(
                policy,
                head,
                encoder_optimizer,
                obs_buf,
                action_buf,
                np.asarray(data["adv"]).reshape(horizon, n),
                done_buf,
                rollout_state,
                settings.max_grad_norm,
            )
        stats: list[tuple[float, float, float]] = []
        for _ in range(settings.epochs):
            order = generator.permutation(samples)
            for start in range(0, samples, settings.minibatch):
                rows = mx.array(order[start : start + settings.minibatch])
                (_, (policy_loss, value_loss, ratio)), grads = grad_fn(
                    pair,
                    data["feats"][rows],
                    data["states"][rows],
                    data["actions"][rows],
                    data["logp"][rows],
                    data["adv"][rows],
                    data["ret"][rows],
                    train_policy,
                )
                grads, _ = optim.clip_grad_norm(grads, settings.max_grad_norm)
                critic_optimizer.update(critic, grads["critic"])
                if train_policy:
                    head_optimizer.update(head, grads["head"])
                mx.eval(head.parameters(), critic.parameters(), policy_loss, value_loss)
                stats.append(
                    (float(policy_loss), float(value_loss), float(mx.abs(ratio - 1).mean()))
                )
        curve = {
            "iteration": iteration + 1,
            "env_steps": env_steps,
            "mean_reward": float(reward_buf.mean()),
            "episodes": finished,
            "success_rate": successes / finished if finished else None,
            task.extra_key: extras / finished if finished else None,
            "policy_loss": float(np.mean([s[0] for s in stats])),
            "value_loss": float(np.mean([s[1] for s in stats])),
            "ratio_deviation": float(np.mean([s[2] for s in stats])),
            "encoder_loss": encoder_loss,
            "action_std": np.exp(np.asarray(head.log_std)).round(4).tolist(),
            "policy_trained": train_policy,
            "steps_per_second": n * horizon / rollout_seconds,
            "elapsed_seconds": time.monotonic() - started,
        }
        if getattr(task, "batch_peak_key", None):
            curve[str(task.batch_peak_key)] = peak
        curves.append(curve)
        print(
            f"it {iteration + 1:4d} steps {env_steps:9d} reward {curve['mean_reward']:.3f} "
            f"episodes {finished:3d} success {successes:3d} {task.extra_label} {extras:4d} "
            f"peak {peak:.0f} "
            f"vloss {curve['value_loss']:.3f} sps {curve['steps_per_second']:.0f}",
            flush=True,
        )
        if (iteration + 1) % settings.eval_every == 0 or iteration + 1 == settings.iterations:
            scored = task.score(policy, head, eval_seeds)
            chosen = task.score(policy, head, select_seeds) if select_seeds else None
            entry: dict[str, Any] = {
                "iteration": iteration + 1,
                "env_steps": env_steps,
                "variants": scored,
            }
            if chosen is not None:
                entry["selection"] = chosen
            evaluations.append(entry)
            print(
                "  eval: "
                + "; ".join(task.describe(name, r, len(eval_seeds)) for name, r in scored.items()),
                flush=True,
            )
            policy.save(output / f"policy-{iteration + 1:04d}.safetensors")
            criterion = (chosen or scored)[settings.best_on]["success_rate"]
            if criterion > best.get("criterion", -1.0):
                best = {
                    **scored[settings.best_on],
                    "iteration": iteration + 1,
                    "criterion": criterion,
                    "selected_on": "validation seeds" if chosen else "test seeds (biased)",
                }
        (output / "curves.json").write_text(json.dumps(curves, indent=1))
        (output / "evaluations.json").write_text(json.dumps(evaluations, indent=1))
    return {"curves": curves, "evaluations": evaluations, "best": best}


def trainable_count(module: nn.Module) -> int:
    leaves = cast(list[tuple[str, mx.array]], tree_flatten(module.trainable_parameters()))
    return sum(value.size for _, value in leaves)


SCRATCH_SEED = 2_000_000  # environment seeds for the random rollouts that calibrate a fresh policy


def scratch_policy(
    config: PPOConfig, pack_root: Path, model_path: Path, obs_dim: int
) -> Controller:
    """A fresh brain policy for reward-only training: no demonstrations anywhere.

    The interface and rate model are the base run's, the encoder and decoder are random, and
    the two frozen normalizations (observation statistics and readout scale) are measured from
    random-action rollouts, which carry no task information.
    """
    from flyarm.interfaces import NeuralInterface
    from flyarm.whole_brain.backend_mlx import RateDynamics
    from flyarm.whole_brain.compiler import ConnectomePack

    env = BatchedPickPlace(
        model_path,
        config.scratch_envs,
        horizon=config.horizon,
        first_seed=SCRATCH_SEED + 10_000 * config.seed,
        variant=task_variant(config.train_variant),
    )
    generator = np.random.default_rng(config.seed)
    rows = [env.reset()]
    for _ in range(config.scratch_steps - 1):
        action = generator.uniform(-1.0, 1.0, (config.scratch_envs, ACTION_DIM))
        rows.append(env.step(action, auto_reset=True).obs)
    observations = np.stack(rows, axis=1).astype(np.float32)
    flat = observations.reshape(-1, obs_dim)
    mean, scale = flat.mean(0), np.maximum(flat.std(0), 0.05)
    if config.controller == "mlp":
        direct = DirectPolicy(obs_dim=obs_dim, action_dim=ACTION_DIM, seed=config.seed)
        direct.set_normalization(mean, scale)
        return direct
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    interface = NeuralInterface.load(Path(config.base_run) / "interface.json")
    policy = BrainPolicy(
        "connectome",
        RateDynamics(pack, interface),
        obs_dim=obs_dim,
        action_dim=ACTION_DIM,
        neural_steps=config.neural_steps,
        seed=config.seed,
    )
    policy.set_normalization(mean, scale)
    # unit_norm, not scale: with 2,022 readout outputs one PPO iteration shifts the decoder's
    # pre-activation by about 2, which saturates the tanh on the first trained iteration and
    # freezes the policy (research log E26, E32, E39).
    calibration = policy.calibrate_readout(
        observations, np.ones(observations.shape[:2], dtype=np.float32), unit_norm=True
    )
    print(
        f"fresh policy calibrated on {observations.shape[0]} random rollouts of "
        f"{observations.shape[1]} steps: median readout spread {calibration['median_spread']:.2e}",
        flush=True,
    )
    return policy


def run_ppo(config: PPOConfig, pack_root: Path, model_path: Path, output: Path) -> dict[str, Any]:
    """Load the base checkpoint, score it, fine-tune its decoder with PPO, save everything."""
    from flyarm.experiment import save_json
    from flyarm.whole_brain.experiment import load_trained_policy

    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose a new output: {output}")
    task, loaded = load_trained_policy(
        Path(config.base_run), config.base_kind, config.base_seed, pack_root, model_path
    )
    eval_seeds = task.seeds("test", config.eval_episodes)
    select_seeds = task.seeds("validation", config.eval_episodes)
    obs_dim = task.obs_dim
    task.close()
    if not isinstance(loaded, BrainPolicy):
        raise ValueError("PPO fine-tuning is defined for brain policies (connectome, shuffled)")
    output.mkdir(parents=True)
    save_json(output / "config.json", config.model_dump())
    policy = (
        scratch_policy(config, pack_root, model_path, obs_dim) if config.from_scratch else loaded
    )
    task_adapter = PickPlaceTask(model_path, config)
    head = MotorHead(policy.decoder, config.log_std, task_adapter.action_dim)
    before = task_adapter.score(policy, head, eval_seeds)
    print(
        "base checkpoint: "
        + "; ".join(f"{n} place {r['successes']} lift {r['lifts']}" for n, r in before.items()),
        flush=True,
    )
    results: dict[str, Any] = {
        "status": "running",
        "base": before,
        "trainable_parameters": trainable_count(head),
        "claim": (
            "reward only: a tanh MLP trained end to end by the same PPO (deep-RL control)"
            if config.controller == "mlp"
            else "reward only: random encoder, frozen connectome, PPO trains the motor decoder"
            if config.from_scratch
            else "PPO tunes only the linear motor decoder; encoder and connectome frozen"
        ),
        "from_scratch": config.from_scratch,
        "controller": config.controller,
    }
    save_json(output / "results.json", results)
    try:
        run = train_ppo(policy, task_adapter, output, config, eval_seeds, select_seeds)
    except (Exception, KeyboardInterrupt) as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "results.json", results)
        raise
    results.update(status="complete", best=run["best"], final=run["evaluations"][-1])
    save_json(output / "results.json", results)
    return results
