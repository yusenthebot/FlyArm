"""PPO against the batched FrankaKitchen benchmark, reusing the pick-and-place trainer.

flyarm.rl.ppo.train_ppo is task-agnostic: it takes a TaskAdapter that supplies the batched
environment, the observation and action dimensions and the benchmark's own evaluation.
KitchenTask is that adapter for flyarm.rl.batched_kitchen, so the same loop, the same critic,
the same clipped updates and the same optional encoder training (research log E38) run on the
kitchen with no change to the trainer.

Evaluation always reports the benchmark's score: the number of the "complete" split's four
tasks an episode completes, and 25 x that number as the D4RL normalized score. The kept
checkpoint is chosen on validation seeds and reported on disjoint test seeds. Note that the
benchmark's episodes all start from the same physical state (research log E29), so under a
deterministic policy every nominal seed gives the same episode; seeds separate episodes only
under a variant that randomizes, such as the perturbed starts of research log E30, which is
why a run that wants an honest selection evaluates one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flyarm.config import FlyLegConfig, KitchenPPOConfig, KitchenVariantConfig
from flyarm.rl.batched_kitchen import ACTION_DIM, OBS_DIM, BatchedKitchen, KitchenVariant
from flyarm.rl.kitchen_quality import STRICT_THRESHOLD, QualityWeights
from flyarm.rl.ppo import Controller, MotorHead, rollout_for, train_ppo, trainable_count
from flyarm.whole_brain.policy import BrainPolicy, DirectPolicy

TEST_SEED = 0  # the kitchen benchmark's evaluation seeds (flyarm.flyleg.experiment)
SELECTION_SEED = 50_000  # disjoint validation seeds, also from flyarm.flyleg.experiment
SCRATCH_SEED = 2_000_000  # random-action rollouts that calibrate a fresh policy


def kitchen_variant(config: KitchenVariantConfig) -> KitchenVariant:
    return KitchenVariant(**config.model_dump())


def quality_weights(settings: KitchenPPOConfig) -> QualityWeights:
    return QualityWeights(
        depth=settings.depth_weight,
        disturbance=settings.disturbance_weight,
        collision=settings.collision_weight,
        action=settings.action_weight,
        smoothness=settings.smoothness_weight,
    )


def evaluate_kitchen(
    policy: Controller,
    head: MotorHead,
    seeds: list[int],
    settings: KitchenPPOConfig,
    variant: KitchenVariant | None = None,
) -> dict[str, Any]:
    """Deterministic mean actions on fixed seeds; the benchmark's own completion count."""
    env = BatchedKitchen(
        len(seeds),
        horizon=settings.horizon,
        variant=variant,
        completion_bonus=settings.completion_bonus,
        approach_slope=settings.approach_slope,
        gamma=settings.gamma,
        tracking_weight=settings.tracking_weight,
        tracking_form=settings.tracking_form,
        target_rule=settings.target_rule,
        shaping_scope=settings.shaping_scope,
        task_shaping_weight=settings.task_shaping_weight,
        task_shaping_form=settings.task_shaping_form,
        completion_order=settings.completion_order,
        quality=quality_weights(settings),
        terminate_on_all_tasks=settings.terminate_on_all_tasks,
        strict_bonus_fraction=settings.strict_bonus_fraction,
        completion_threshold=settings.completion_threshold,
        final_strict_bonus=settings.final_strict_bonus,
        tracking_sigma=settings.tracking_sigma,
        reference_episode=settings.reference_episode,
    )
    obs = env.reset(seeds=np.array(seeds))
    brain = rollout_for(policy, len(seeds))
    completed = np.zeros((len(seeds), len(env.tasks)), dtype=bool)
    reward = np.zeros(len(seeds))
    active = np.ones(len(seeds), dtype=bool)
    stray_steps = np.zeros(len(seeds))
    steps = np.zeros(len(seeds))
    magnitude, saturated, change = np.zeros(len(seeds)), np.zeros(len(seeds)), np.zeros(len(seeds))
    previous = np.zeros((len(seeds), ACTION_DIM))
    final_distance = np.zeros((len(seeds), len(env.tasks)))
    disturbance = np.zeros(len(seeds))
    for _ in range(settings.horizon):
        action = np.asarray(head.mean(brain.features(obs)), dtype=np.float64)
        result = env.step(action, auto_reset=False)
        completed |= active[:, None] & result.completed
        reward += active * result.reward
        stray_steps += active * result.stray_contact
        steps += active
        magnitude += active * np.abs(action).mean(1)
        saturated += active * (np.abs(action) > 0.95).mean(1)
        change += active * np.abs(action - previous).mean(1)
        previous = action
        final_distance[active] = result.goal_distance[active]
        disturbance[active] = -env.disturbance_potential()[active]
        active &= ~(result.terminated | result.truncated)
        obs = result.obs
        if not active.any():
            break
    counts = completed.sum(1)
    strict = final_distance < STRICT_THRESHOLD
    return {
        "seeds": seeds,
        "tasks": list(env.tasks),
        # The fraction of the split completed; train_ppo selects checkpoints on this key.
        "success_rate": float(counts.mean() / len(env.tasks)),
        "mean_tasks": float(counts.mean()),
        "normalized_score": float(25.0 * counts.mean()),
        "completed_all": int((counts == len(env.tasks)).sum()),
        "mean_episode_reward": float(reward.mean()),
        "per_task_success": {
            task: float(completed[:, index].mean()) for index, task in enumerate(env.tasks)
        },
        # Motion quality (research log E50): the strict score counts a task only if its element
        # ends the episode within STRICT_THRESHOLD of its goal.
        "strict_score": float(25.0 * strict.sum(1).mean()),
        "strict_per_task": {
            task: float(strict[:, index].mean()) for index, task in enumerate(env.tasks)
        },
        "final_goal_distance": {
            task: float(np.median(final_distance[:, index])) for index, task in enumerate(env.tasks)
        },
        "stray_contact_fraction": float((stray_steps / np.maximum(steps, 1)).mean()),
        "mean_abs_action": float((magnitude / np.maximum(steps, 1)).mean()),
        "saturated_fraction": float((saturated / np.maximum(steps, 1)).mean()),
        "mean_abs_action_change": float((change / np.maximum(steps, 1)).mean()),
        "disturbance_rad": float(disturbance.mean()),
    }


class KitchenTask:
    """The FrankaKitchen adapter for flyarm.rl.ppo.train_ppo."""

    obs_dim = OBS_DIM
    action_dim = ACTION_DIM
    extra_key = "tasks_per_episode"
    extra_label = "tasks"
    batch_peak_key: str | None = "max_tasks_in_one_episode"

    def __init__(self, settings: KitchenPPOConfig) -> None:
        self.settings = settings
        self.privileged_dim = self.make_env(1, 0).privileged_dim

    def make_env(self, num_envs: int, first_seed: int) -> BatchedKitchen:
        return BatchedKitchen(
            num_envs,
            horizon=self.settings.horizon,
            first_seed=first_seed,
            variant=kitchen_variant(self.settings.train_variant),
            completion_bonus=self.settings.completion_bonus,
            approach_slope=self.settings.approach_slope,
            gamma=self.settings.gamma,
            tracking_weight=self.settings.tracking_weight,
            tracking_form=self.settings.tracking_form,
            target_rule=self.settings.target_rule,
            shaping_scope=self.settings.shaping_scope,
            task_shaping_weight=self.settings.task_shaping_weight,
            task_shaping_form=self.settings.task_shaping_form,
            completion_order=self.settings.completion_order,
            quality=quality_weights(self.settings),
            terminate_on_all_tasks=self.settings.terminate_on_all_tasks,
            strict_bonus_fraction=self.settings.strict_bonus_fraction,
            completion_threshold=self.settings.completion_threshold,
            final_strict_bonus=self.settings.final_strict_bonus,
            tracking_sigma=self.settings.tracking_sigma,
            reference_episode=self.settings.reference_episode,
        )

    def score(
        self, policy: BrainPolicy, head: MotorHead, seeds: list[int]
    ) -> dict[str, dict[str, Any]]:
        return {
            name: evaluate_kitchen(policy, head, seeds, self.settings, kitchen_variant(variant))
            for name, variant in self.settings.eval_variants.items()
        }

    def extra(self, result: Any, done: np.ndarray) -> int:
        return int(result.tasks_completed[done].sum())

    def batch_peak(self, result: Any, done: np.ndarray) -> float:
        """Most tasks any single finished episode earned; the number the kitchen goal is about."""
        return float(result.tasks_completed[done].max()) if done.any() else 0.0

    def describe(self, name: str, scored: dict[str, Any], episodes: int) -> str:
        return (
            f"{name} score {scored['normalized_score']:.1f} "
            f"({scored['mean_tasks']:.2f} tasks, all four in {scored['completed_all']}/{episodes}) "
            f"strict {scored['strict_score']:.1f} stray {scored['stray_contact_fraction']:.2f} "
            f"|a| {scored['mean_abs_action']:.2f} sat {scored['saturated_fraction']:.2f}"
        )


def scratch_kitchen_policy(config: KitchenPPOConfig, pack_root: Path) -> Controller:
    """A fresh policy for reward-only kitchen training: no demonstrations anywhere.

    For the connectome, the interface is the base run's frozen one; the encoder and decoder are
    random, and both frozen normalizations (observation statistics and readout scale) are
    measured from random-action rollouts, which carry no task information (research log E36).
    The MLP control gets the same observation normalization from the same rollouts.
    """
    env = BatchedKitchen(
        config.scratch_envs,
        horizon=config.horizon,
        first_seed=SCRATCH_SEED + 10_000 * config.seed,
        variant=kitchen_variant(config.train_variant),
        completion_bonus=config.completion_bonus,
        approach_slope=config.approach_slope,
        gamma=config.gamma,
    )
    generator = np.random.default_rng(config.seed)
    rows = [env.reset()]
    for _ in range(config.scratch_steps - 1):
        action = generator.uniform(-1.0, 1.0, (config.scratch_envs, ACTION_DIM))
        rows.append(env.step(action, auto_reset=True).obs)
    observations = np.stack(rows, axis=1).astype(np.float32)
    flat = observations.reshape(-1, OBS_DIM)
    mean, scale = flat.mean(0), np.maximum(flat.std(0), 0.05)
    if config.controller == "mlp":
        direct = DirectPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=config.seed)
        direct.set_normalization(mean, scale)
        return direct
    return _calibrated_brain(config, pack_root, observations, mean, scale)


def _calibrated_brain(
    config: KitchenPPOConfig,
    pack_root: Path,
    observations: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> BrainPolicy:
    from flyarm.interfaces import NeuralInterface
    from flyarm.whole_brain.backend_mlx import RateDynamics
    from flyarm.whole_brain.compiler import ConnectomePack

    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    interface = NeuralInterface.load(Path(config.base_run) / "interface.json")
    policy = BrainPolicy(
        "connectome",
        RateDynamics(pack, interface),
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        neural_steps=config.neural_steps,
        seed=config.seed,
    )
    policy.set_normalization(mean, scale)
    calibration = policy.calibrate_readout(
        observations,
        np.ones(observations.shape[:2], dtype=np.float32),
        center=config.readout_calibration != "scale",
        unit_norm=config.readout_calibration == "unit_norm",
    )
    print(
        f"fresh policy calibrated on {observations.shape[0]} random rollouts of "
        f"{observations.shape[1]} steps: median readout spread {calibration['median_spread']:.2e}",
        flush=True,
    )
    return policy


def kitchen_base_config(run_root: Path) -> FlyLegConfig:
    """The imitation run's own config, checked for the three properties PPO depends on.

    A warm start only makes sense from a run that speaks the same interface, the same action
    layout and the same split as the batched environment, so each is refused by name rather
    than failing later inside the trainer.
    """
    path = run_root / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run_root} is not a kitchen imitation run: no config.json")
    config = FlyLegConfig.model_validate_json(path.read_text())
    if config.interface != "whole_body":
        raise ValueError(
            f"{run_root} trained the '{config.interface}' interface; kitchen PPO warm starts "
            "from the whole-body B1a interface, whose ascending encoder is the only path the "
            "scene has into the connectome (research log E32)"
        )
    if config.action_chunk != 1:
        raise ValueError(
            f"{run_root} emits chunks of {config.action_chunk} actions; the PPO head maps the "
            f"readout straight to the {ACTION_DIM} joint velocities, so the base run needs "
            "action_chunk 1"
        )
    if config.split != "complete":
        raise ValueError(
            f"{run_root} was trained on the '{config.split}' split; the batched environment "
            "scores the four tasks of 'complete'"
        )
    return config


def load_kitchen_policy(config: KitchenPPOConfig, pack_root: Path) -> BrainPolicy:
    """Rebuild a trained B2 kitchen checkpoint as the PPO starting point."""
    from flyarm.flyleg.record import load_flyleg_policy

    run_root = Path(config.base_run)
    kitchen_base_config(run_root)
    checkpoint = run_root / f"{config.base_kind}-{config.base_seed}" / "policy.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No {config.base_kind} seed-{config.base_seed} checkpoint: {checkpoint}"
        )
    policy = load_flyleg_policy(
        run_root, config.base_kind, config.base_seed, pack_root, Path(config.annotations)
    )
    # Backstops: the config above already rules these out, but the checkpoint decides.
    if not isinstance(policy, BrainPolicy):
        raise ValueError("Kitchen PPO fine-tuning is defined for brain policies")
    if policy.chunk != 1:
        raise ValueError(
            f"The base checkpoint emits chunks of {policy.chunk} actions; PPO drives one "
            "action per step, so retrain the base run with action_chunk 1"
        )
    if policy.obs_dim != OBS_DIM or policy.action_dim != ACTION_DIM:
        raise ValueError(
            f"The base checkpoint is {policy.obs_dim} -> {policy.action_dim}, "
            f"expected {OBS_DIM} -> {ACTION_DIM}"
        )
    return policy


def demonstration_features(
    policy: BrainPolicy, base_run: Path, batch: int = 64
) -> tuple[mx.array, mx.array]:
    """Connectome features and actions of the base run's own training demonstrations.

    These are exactly the episodes the imitation run trained on: the dataset episodes listed
    as training in its splits.json plus its DART episodes (dart.npz). Each episode is replayed
    through the frozen encoder and connectome from a zero state, as in imitation, and only the
    valid steps are kept. The encoder must stay frozen during PPO for these to remain exact.
    """
    from flyarm.benchmarks import kitchen

    config = kitchen_base_config(base_run)
    splits = json.loads((base_run / "splits.json").read_text())
    data = kitchen.load(config.split)
    rows = np.flatnonzero(np.isin(data.episode_ids, splits["train_episode_ids"]))
    parts = [data.subset(rows)]
    dart_path = base_run / "dart.npz"
    if dart_path.is_file():
        dart = np.load(dart_path)
        parts.append({key: dart[key] for key in ("obs", "actions", "mask")})
    features, actions = [], []
    for part in parts:
        for start in range(0, len(part["obs"]), batch):
            obs = part["obs"][start : start + batch]
            mask = part["mask"][start : start + batch] > 0
            rollout = rollout_for(policy, len(obs))
            steps = [np.asarray(rollout.features(obs[:, t])) for t in range(obs.shape[1])]
            stacked = np.stack(steps, axis=1)  # [episodes, steps, features]
            features.append(stacked[mask])
            actions.append(part["actions"][start : start + batch][mask])
    return mx.array(np.concatenate(features)), mx.array(np.concatenate(actions))


def run_kitchen_ppo(config: KitchenPPOConfig, pack_root: Path, output: Path) -> dict[str, Any]:
    """Build the starting policy, score it, train it with PPO and save everything."""
    from flyarm.experiment import save_json

    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose a new output: {output}")
    output.mkdir(parents=True)
    save_json(output / "config.json", config.model_dump())
    policy = (
        scratch_kitchen_policy(config, pack_root)
        if config.from_scratch
        else load_kitchen_policy(config, pack_root)
    )
    if config.init_checkpoint is not None:
        checkpoint = Path(config.init_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {checkpoint}")
        policy.load(checkpoint)
        print(f"continuing from {checkpoint}", flush=True)
    task = KitchenTask(config)
    head = MotorHead(policy.decoder, config.log_std, ACTION_DIM)
    eval_seeds = list(range(TEST_SEED, TEST_SEED + config.eval_episodes))
    select_seeds = list(range(SELECTION_SEED, SELECTION_SEED + config.eval_episodes))
    before = task.score(policy, head, eval_seeds)
    summary = "; ".join(task.describe(n, r, len(eval_seeds)) for n, r in before.items())
    print(f"base checkpoint: {summary}", flush=True)
    results: dict[str, Any] = {
        "status": "running",
        "benchmark": "D4RL/kitchen/complete-v2 (FrankaKitchen-v1)",
        "base": before,
        "trainable_parameters": trainable_count(head),
        "claim": (
            "reward only: a tanh MLP trained end to end by the same PPO (deep-RL control)"
            if config.controller == "mlp"
            else "reward only: random encoder, frozen connectome, PPO trains the motor decoder"
            + (" and the encoder" if config.encoder_lr > 0 else "")
            if config.from_scratch
            else "PPO tunes the trained kitchen checkpoint against the shaped kitchen reward"
        ),
        "from_scratch": config.from_scratch,
        "controller": config.controller,
        "init_checkpoint": config.init_checkpoint,
        "encoder_trained": config.encoder_lr > 0,
        "base_run": config.base_run,
        "base_kind": None if config.from_scratch else config.base_kind,
    }
    save_json(output / "results.json", results)
    demonstrations = None
    if config.bc_weight > 0:
        if not isinstance(policy, BrainPolicy):
            raise ValueError("The DAPG term is defined for a warm-started brain policy")
        demonstrations = demonstration_features(policy, Path(config.base_run))
        results["demonstration_steps"] = int(demonstrations[0].shape[0])
        print(f"DAPG term on {demonstrations[0].shape[0]} demonstration steps", flush=True)
    try:
        run = train_ppo(
            policy, task, output, config, eval_seeds, select_seeds, demonstrations=demonstrations
        )
    except (Exception, KeyboardInterrupt) as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "results.json", results)
        raise
    results.update(status="complete", best=run["best"], final=run["evaluations"][-1])
    save_json(output / "results.json", results)
    return results


__all__ = [
    "KitchenTask",
    "evaluate_kitchen",
    "kitchen_base_config",
    "kitchen_variant",
    "load_kitchen_policy",
    "run_kitchen_ppo",
    "scratch_kitchen_policy",
]
