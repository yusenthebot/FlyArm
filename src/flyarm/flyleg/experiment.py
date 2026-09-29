"""B2 experiment: the fly's left front leg drives a Franka on D4RL FrankaKitchen.

Controllers, all trained by the same masked sequence behavior cloning on the same episodes:

- ``flyleg``: complete MaleCNS; joint state -> left front-leg proprioceptors, scene state ->
  head sensory neurons, joint velocities <- left front-leg motor neurons.
- ``flyleg_shuffled``: the same interface on a degree-preserving shuffle of the whole CNS.
- ``mlp``: the standard D4RL BC architecture (two 256-unit ReLU layers, no memory).
- ``gru``: a recurrent control with the fly policy's trainable-parameter budget.
- ``act``: ACT (transformer + CVAE), a reference for what the demonstrations support; not a
  fly model and not parameter-matched.

With ``action_chunk`` k > 1 every controller predicts the next k actions at each step and
executes their temporal ensemble (ACT's action chunking), so all rows share one protocol.

Post-training checks of every ``flyleg`` checkpoint: edges off, direct sensory-to-motor
synapses only, deafferented leg (proprioceptors silenced), head-sensory deprivation and
state reset every step. OOD evaluations of every checkpoint: 10x robot or object observation
noise and seeded initial arm-joint offsets.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flyarm.assets import digest_file
from flyarm.benchmarks import kitchen
from flyarm.benchmarks.kitchen_expert import DemonstrationTracker, noisy_teacher_episodes
from flyarm.config import FlyLegConfig
from flyarm.flyleg.interface import front_leg_interface, front_leg_report, leg_channels
from flyarm.interfaces import NeuralInterface
from flyarm.io import save_json
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.diagnostics import direct_only_weights
from flyarm.whole_brain.experiment import ResetEveryStep
from flyarm.whole_brain.interface import annotation_interface, interface_report
from flyarm.whole_brain.policy import (
    ACTPolicy,
    BrainPolicy,
    Channel,
    GRUPolicy,
    MLPPolicy,
    MlxController,
    SequencePolicy,
    gru_hidden_for_budget,
)
from flyarm.whole_brain.shuffle import shuffle_pack
from flyarm.whole_brain.training import (
    ACTBudget,
    Budget,
    train_act_policy,
    train_sequence_policy,
)

VALIDATION_SEED = 90_000
# Environment seeds; evaluation uses 0..eval_episodes-1, so every other range is disjoint.
SELECTION_SEED = 50_000
DAGGER_SEED = 200_000
DART_SEED = 300_000


def split_episodes(count: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic episode-level split; at least one validation episode."""
    order = np.random.default_rng(seed).permutation(count)
    held_out = max(1, int(round(count * fraction)))
    return np.sort(order[held_out:]), np.sort(order[:held_out])


def ood_suite(config: FlyLegConfig) -> dict[str, dict[str, float]]:
    return {
        "robot_noise_x10": {"robot_noise_ratio": 0.1},
        "object_noise_x10": {"object_noise_ratio": 0.005},
        **{
            f"joint_offset_{offset:g}rad": {"initial_joint_offset": offset}
            for offset in config.ood_joint_offsets
        },
    }


def _evaluate(
    config: FlyLegConfig, controller: Any, seeds: list[int], **variant: float
) -> dict[str, Any]:
    features = kitchen.PositionFeatures(controller) if controller is not None else None
    return _evaluate_raw(config, features, seeds, **variant)


def _evaluate_raw(
    config: FlyLegConfig, controller: Any, seeds: list[int], **variant: float
) -> dict[str, Any]:
    """Evaluate a controller that reads the full benchmark observation (e.g. the teacher)."""
    offset = float(variant.pop("initial_joint_offset", 0.0))
    env = kitchen.recover_env(config.split, **variant)
    try:
        result = kitchen.evaluate(env, controller, seeds, initial_joint_offset=offset)
    finally:
        env.close()
    return {**result, "environment_overrides": variant}


def leg_dynamics(
    config: FlyLegConfig, pack: ConnectomePack, interface: NeuralInterface, **options: Any
) -> RateDynamics:
    """The connectome dynamics of a run, in the rate regime its config names."""
    if config.weight_norm_power != 1.0 and options.get("edges", True):
        options.setdefault("weights", pack.normalized_weights(config.weight_norm_power))
    return RateDynamics(pack, interface, recurrent_gain=config.recurrent_gain, **options)


def fly_interface(
    config: FlyLegConfig, pack: ConnectomePack, annotations: Path
) -> tuple[NeuralInterface, list[Channel] | None, dict[str, Any]]:
    """The run's fly interface, its sensory channels (None: one encoder) and its report.

    "front_leg": the Franka as the left front leg (leg proprioceptors and head senses in,
    leg motor neurons out). "whole_body": the B1a interface, every observation feature into
    the 1,846 ascending neurons and the 1,314 descending plus 708 VNC motor neurons read out.
    """
    if config.interface == "whole_body":
        body = annotation_interface(pack)
        return body, None, interface_report(pack, body)
    leg = front_leg_interface(
        pack,
        annotations,
        include_head=config.sensory_channels == "proprioception+head",
        include_descending=config.readout == "leg_motor+descending",
    )
    return leg.interface, leg_channels(leg), front_leg_report(pack, leg, annotations)


def make_policy(
    kind: str,
    config: FlyLegConfig,
    seed: int,
    *,
    dynamics: RateDynamics | None,
    channels: list[Channel] | None,
    fly_budget: int,
) -> SequencePolicy:
    """Untrained controller of one kind; the same constructor serves training and replay."""
    obs_dim, action_dim, chunk = kitchen.FEATURE_DIM, kitchen.ACTION_DIM, config.action_chunk
    if kind == "mlp":
        return MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, chunk=chunk, seed=seed)
    if kind == "gru":
        output_dim = kitchen.ACTION_DIM * config.action_chunk
        hidden = gru_hidden_for_budget(kitchen.FEATURE_DIM, output_dim, fly_budget)
        return GRUPolicy(
            obs_dim=obs_dim, action_dim=action_dim, chunk=chunk, hidden=hidden, seed=seed
        )
    if kind == "act":
        slices = [
            (kitchen.PROPRIOCEPTION.start, kitchen.PROPRIOCEPTION.stop),
            (kitchen.EXTEROCEPTION.start, kitchen.EXTEROCEPTION.stop),
        ]
        return ACTPolicy(
            obs_dim=obs_dim, action_dim=action_dim, chunk=chunk, token_slices=slices, seed=seed
        )
    if dynamics is None:
        raise ValueError(f"{kind} needs connectome dynamics")
    return BrainPolicy(
        kind,
        dynamics,
        obs_dim=obs_dim,
        action_dim=action_dim,
        chunk=chunk,
        neural_steps=config.neural_steps,
        seed=seed,
        channels=channels,
    )


def _pad(data: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    extra = horizon - data["obs"].shape[1]
    if extra < 0:
        raise ValueError("Episodes are longer than the benchmark horizon")
    return {
        key: np.pad(value, [(0, 0), (0, extra)] + [(0, 0)] * (value.ndim - 2))
        if key in ("obs", "actions", "mask")
        else value
        for key, value in data.items()
    }


def _dagger_rollouts(
    policy: SequencePolicy,
    expert: DemonstrationTracker,
    env: Any,
    episodes: int,
    seed: int,
    iteration: int,
    beta: float = 0.0,
    start_offset: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Episodes from clean starts, every visited state labelled by the tracker.

    Each step executes the teacher's action with probability ``beta`` and the learner's
    otherwise; the learner sees every observation either way, so its state stays in step.
    With ``start_offset`` the arm starts perturbed as in the joint-offset evaluation.
    """
    horizon = env.spec.max_episode_steps
    obs = np.zeros((episodes, horizon, kitchen.FEATURE_DIM), np.float32)
    actions = np.zeros((episodes, horizon, kitchen.ACTION_DIM), np.float32)
    mask = np.zeros((episodes, horizon), np.float32)
    controller = MlxController(policy)
    generator = np.random.default_rng([DAGGER_SEED, seed, iteration])
    completed: list[int] = []
    teacher_steps = 0
    for episode in range(episodes):
        reset_seed = DAGGER_SEED + 10_000 * seed + 100 * iteration + episode
        observation, info = env.reset(seed=reset_seed)
        if start_offset:
            observation = kitchen._offset_initial_joints(env, reset_seed, start_offset)
        controller.reset()
        info = {}
        for step in range(horizon):
            full = np.asarray(observation["observation"], dtype=np.float32)
            obs[episode, step] = full[kitchen.POLICY_FEATURES]
            actions[episode, step] = expert.label(full)
            mask[episode, step] = 1.0
            action = controller.act(obs[episode, step])
            if beta and generator.random() < beta:
                action = actions[episode, step]
                teacher_steps += 1
            observation, _, terminated, truncated, info = env.step(action.astype(np.float64))
            if terminated or truncated:
                break
        completed.append(len(info.get("episode_task_completions", [])))
    stats = {
        "rollout_episodes": episodes,
        "beta": beta,
        "start_offset_rad": start_offset,
        "teacher_step_fraction": teacher_steps / max(int(mask.sum()), 1),
        "rollout_mean_tasks": float(np.mean(completed)),
        "labelled_states": int(mask.sum()),
    }
    return {"obs": obs, "actions": actions, "mask": mask}, stats


Selector = Callable[[SequencePolicy], float]


def _fit(
    policy: SequencePolicy,
    config: FlyLegConfig,
    data: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    seed: int,
    deadline: float,
    *,
    phase: str,
    selector: Selector | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One training phase: behavior cloning on the training set, or a DAgger aggregate."""
    first = phase == "behavior_cloning"
    if isinstance(policy, ACTPolicy):
        steps = config.act_steps if first else config.act_dagger_steps
        budget = ACTBudget(
            steps=steps,
            batch_size=config.act_batch_size,
            learning_rate=config.act_learning_rate,
            kl_weight=config.act_kl_weight,
            eval_every=max(1, steps // 50),
            deadline=deadline,
        )
        return train_act_policy(
            policy,
            data,
            validation,
            budget,
            seed,
            phase=phase,
            set_normalization=first,
            selector=selector,
        )
    brain = isinstance(policy, BrainPolicy)
    return train_sequence_policy(
        policy,
        data,
        data["mask"],
        validation,
        validation["mask"],
        Budget(
            epochs=config.epochs if first else config.dagger_epochs,
            decoder_warmup_epochs=config.decoder_warmup_epochs if brain and first else 0,
            batch_size=config.batch_size,
            bptt_steps=config.bptt_steps,
            learning_rate=config.learning_rate,
            deadline=deadline,
            loss=config.loss,
        ),
        seed,
        phase=phase,
        set_normalization=first and not brain,
        selector=selector,
        select_every=config.select_every,
    )


def _train(
    policy: SequencePolicy,
    config: FlyLegConfig,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    seed: int,
    deadline: float,
    expert: DemonstrationTracker | None,
    initial: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float] | None]:
    calibration = None
    if initial is not None:
        # Encoder, decoder, normalization and readout scale all come from the checkpoint.
        policy.load(initial)
    elif isinstance(policy, BrainPolicy):
        samples = train["obs"][train["mask"].astype(bool)]
        policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
        calibration = policy.calibrate_readout(
            train["obs"],
            train["mask"],
            center=config.readout_calibration == "standardize",
            unit_norm=config.readout_calibration == "unit_norm",
        )
    env = kitchen.recover_env(config.split)
    selection_seeds = list(range(SELECTION_SEED, SELECTION_SEED + config.selection_episodes))

    def closed_loop(candidate: SequencePolicy) -> float:
        controller = kitchen.PositionFeatures(MlxController(candidate))
        result = kitchen.evaluate(
            env, controller, selection_seeds, initial_joint_offset=config.selection_joint_offset
        )
        return float(result["mean_tasks"])

    selector = closed_loop if config.selection == "closed_loop" else None
    try:
        curves: list[dict[str, Any]] = []
        phases: list[dict[str, Any]] = []
        if initial is None:
            curves, summary = _fit(
                policy,
                config,
                train,
                validation,
                seed,
                deadline,
                phase="behavior_cloning",
                selector=selector,
            )
            phases.append({"phase": "behavior_cloning", **summary})
        else:
            score = selector(policy) if selector is not None else None
            phases.append(
                {
                    "phase": "initialized",
                    "source": str(initial),
                    "selection_score": score,
                    "training_seconds": 0.0,
                }
            )
        # With closed-loop selection the kept weights are the best over every phase, not the
        # last phase's best: a DAgger round may make the controller worse.
        best_score = phases[-1].get("selection_score")
        best_phase, best_parameters = phases[-1]["phase"], policy.parameters()
        aggregate = _pad(train, env.spec.max_episode_steps)
        for iteration in range(config.dagger_iterations if expert is not None else 0):
            assert expert is not None
            beta = config.dagger_beta * config.dagger_beta_decay**iteration
            rollouts, stats = _dagger_rollouts(
                policy,
                expert,
                env,
                config.dagger_episodes,
                seed,
                iteration,
                beta,
                start_offset=config.dagger_start_offset,
            )
            aggregate = {
                key: np.concatenate((aggregate[key], rollouts[key]))
                for key in ("obs", "actions", "mask")
            }
            phase = f"dagger_{iteration + 1}"
            more, summary = _fit(
                policy,
                config,
                aggregate,
                validation,
                seed + iteration + 1,
                deadline,
                phase=phase,
                selector=selector,
            )
            curves += more
            phases.append(
                {"phase": phase, **summary, **stats, "aggregate_episodes": len(aggregate["obs"])}
            )
            score = summary.get("selection_score")
            if score is not None and (best_score is None or score > best_score):
                best_score, best_phase, best_parameters = score, phase, policy.parameters()
            print(
                f"  {phase}: beta {beta:.2f}, rollout mean tasks {stats['rollout_mean_tasks']:.2f}",
                flush=True,
            )
        if selector is not None:
            policy.update(best_parameters)
    finally:
        env.close()
    training = {
        "selected_phase": best_phase,
        "phases": phases,
        "loss": phases[-1]["loss"],
        "selection": phases[-1]["selection"],
        "selection_score": best_score,
        "best_validation_loss": phases[-1]["best_validation_loss"],
        "training_seconds": float(sum(phase["training_seconds"] for phase in phases)),
    }
    return curves, training, calibration


def _short(result: dict[str, Any]) -> str:
    return f"{result['normalized_score']:.1f}±{result['normalized_score_sem']:.1f}"


def run_flyleg_experiment(
    pack_root: Path, annotations: Path, output: Path, config: FlyLegConfig
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose a new output: {output}")
    started = time.monotonic()
    try:
        return _run(pack_root, annotations, output, config)
    except (Exception, KeyboardInterrupt) as error:
        if output.is_dir():
            path = output / "results.json"
            partial = json.loads(path.read_text()) if path.exists() else {"models": []}
            partial.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                elapsed_seconds=time.monotonic() - started,
            )
            save_json(path, partial)
        raise


def _run(pack_root: Path, annotations: Path, output: Path, config: FlyLegConfig) -> dict[str, Any]:
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    interface, channels, report = fly_interface(config, pack, annotations)
    data = kitchen.load(config.split)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + config.max_seconds
    interface.save(output / "interface.json")
    save_json(output / "interface_report.json", report)
    save_json(output / "config.json", config.model_dump())
    package = Path(__file__).resolve().parent.parent
    save_json(
        output / "provenance.json",
        {
            "pack_fingerprint": pack.fingerprint(),
            "interface_fingerprint": interface.fingerprint,
            "annotations_sha256": digest_file(annotations),
            "dataset": data.provenance,
            "source_files_sha256": {
                str(path.relative_to(package)): digest_file(path)
                for path in sorted(package.rglob("*.py"))
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                name: version(name)
                for name in ["mlx", "mujoco", "numpy", "gymnasium", "gymnasium-robotics", "minari"]
            },
            "device": str(mx.default_device()),
            "observation_channels": {
                "proprioception": "obs[0:9] robot joint positions",
                "exteroception": "obs[18:39] kitchen object joint positions",
                "excluded": "all velocities (copycat causal confusion in behavior cloning)",
            },
            "action": "9 joint velocities in [-1, 1] (7 arm, 2 fingers), benchmark units",
        },
    )
    train_rows, validation_rows = split_episodes(
        len(data.episode_ids), config.validation_fraction, VALIDATION_SEED
    )
    train, validation = data.subset(train_rows), data.subset(validation_rows)
    save_json(
        output / "splits.json",
        {
            "train_episode_ids": data.episode_ids[train_rows].tolist(),
            "validation_episode_ids": data.episode_ids[validation_rows].tolist(),
            "evaluation_seeds": list(range(config.eval_episodes)),
        },
    )
    seeds = list(range(config.eval_episodes))
    results: dict[str, Any] = {
        "status": "running",
        "split": config.split,
        "published_bc_reference": kitchen.PUBLISHED_BC[config.split],
        "claim": "fly front-leg interface on FrankaKitchen; no topology advantage assumed",
        "zero_action": _evaluate(config, None, seeds[:5]),
        "models": [],
    }
    expert = None
    if config.dagger_iterations or config.dart_episodes:
        expert = DemonstrationTracker.from_data(data, gain=config.tracker_gain)
        results["teacher"] = {
            "name": "demonstration tracker (nearest demo state + joint correction)",
            "gain": config.tracker_gain,
            "clean": _evaluate_raw(config, expert, seeds),
        }
    if config.dart_episodes and expert is not None:
        env = kitchen.recover_env(config.split)
        try:
            dart, dart_stats = noisy_teacher_episodes(
                expert,
                env,
                config.dart_episodes,
                config.dart_noise,
                DART_SEED,
                start_offset=config.dart_start_offset,
            )
        finally:
            env.close()
        np.savez_compressed(
            output / "dart.npz", obs=dart["obs"], actions=dart["actions"], mask=dart["mask"]
        )
        results["teacher"]["dart"] = dart_stats
        padded = _pad(train, dart["obs"].shape[1])
        train = {
            key: np.concatenate((padded[key], dart[key])) for key in ("obs", "actions", "mask")
        }
    save_json(output / "results.json", results)
    if config.init_from is not None:
        source = FlyLegConfig.model_validate_json(
            (Path(config.init_from) / "config.json").read_text()
        )
        if (source.action_chunk, source.sensory_channels, source.split) != (
            config.action_chunk,
            config.sensory_channels,
            config.split,
        ):
            raise ValueError("init_from must share the action chunk, senses and split")

    measured = leg_dynamics(config, pack, interface)
    fly_budget = make_policy(
        "flyleg", config, 0, dynamics=measured, channels=channels, fly_budget=0
    ).trainable_parameter_count()

    for seed in config.seeds:
        dynamics = {"flyleg": measured}
        if "flyleg_shuffled" in config.policies:
            shuffled = shuffle_pack(pack, seed + 17000)
            rebound = NeuralInterface.bind(
                shuffled,
                interface.input_body_ids,
                interface.output_body_ids,
                label=interface.label,
            )
            dynamics["flyleg_shuffled"] = leg_dynamics(config, shuffled, rebound)
            save_json(
                output / f"shuffled-{seed}.json",
                {
                    "fingerprint": shuffled.fingerprint(),
                    **{k: v for k, v in shuffled.manifest.items() if k != "sources"},
                },
            )
        for kind in config.policies:
            print(f"{config.split} {kind} seed={seed}", flush=True)
            run = output / f"{kind}-{seed}"
            run.mkdir()
            policy = make_policy(
                kind,
                config,
                seed,
                dynamics=dynamics.get(kind),
                channels=channels,
                fly_budget=fly_budget,
            )
            initial = None
            if config.init_from is not None:
                initial = Path(config.init_from) / f"{kind}-{seed}" / "policy.safetensors"
                if not initial.is_file():
                    raise FileNotFoundError(f"init_from has no checkpoint {initial}")
            curves, training, calibration = _train(
                policy, config, train, validation, seed, deadline, expert, initial
            )
            save_json(run / "learning.json", curves)
            policy.save(run / "policy.safetensors")
            item: dict[str, Any] = {
                "kind": kind,
                "seed": seed,
                "trainable_parameters": policy.trainable_parameter_count(),
                "state_size": policy.state_size,
                "action_chunk": policy.chunk,
                "readout_calibration": calibration,
                **training,
                "clean": _evaluate(config, MlxController(policy), seeds),
                "ood": {
                    name: _evaluate(config, MlxController(policy), seeds, **variant)
                    for name, variant in ood_suite(config).items()
                },
            }
            if isinstance(policy, BrainPolicy) and kind == "flyleg":
                lesions: dict[str, SequencePolicy] = {
                    "edges_off": policy.with_dynamics(
                        "edges_off", leg_dynamics(config, pack, interface, edges=False)
                    ),
                    "direct_only": policy.with_dynamics(
                        "direct_only",
                        leg_dynamics(
                            config,
                            pack,
                            interface,
                            weights=direct_only_weights(pack, interface, config.weight_norm_power),
                        ),
                    ),
                }
                if channels is not None:
                    lesions["deafferented_leg"] = policy.silence_channel("deafferented_leg", 0)
                if channels is not None and len(channels) > 1:
                    lesions["head_sensory_deprived"] = policy.silence_channel(
                        "head_sensory_deprived", 1
                    )
                item["lesions"] = {
                    name: _evaluate(config, MlxController(lesioned), seeds)
                    for name, lesioned in lesions.items()
                }
                item["lesions"]["state_reset_every_step"] = _evaluate(
                    config, ResetEveryStep(MlxController(policy)), seeds
                )
            save_json(run / "evaluation.json", item)
            results["models"].append(item)
            results["summary"] = summarize(results["models"])
            save_json(output / "results.json", results)
            lesion_text = {k: _short(v) for k, v in item.get("lesions", {}).items()}
            ood_text = {k: _short(v) for k, v in item["ood"].items()}
            print(
                f"  clean={_short(item['clean'])} ood={ood_text} lesions={lesion_text}", flush=True
            )
    results.update(status="complete", elapsed_seconds=time.monotonic() - started)
    save_json(output / "results.json", results)
    return results


def summarize(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Seed-averaged normalized scores for every model, OOD variant and lesion."""
    summary: dict[str, Any] = {}
    for kind in sorted({model["kind"] for model in models}):
        group = [model for model in models if model["kind"] == kind]
        entry: dict[str, Any] = {
            "seeds": [model["seed"] for model in group],
            "clean": float(np.mean([model["clean"]["normalized_score"] for model in group])),
            "ood": {
                name: float(np.mean([model["ood"][name]["normalized_score"] for model in group]))
                for name in group[0]["ood"]
            },
        }
        if "lesions" in group[0]:
            entry["lesions"] = {
                name: float(
                    np.mean([model["lesions"][name]["normalized_score"] for model in group])
                )
                for name in group[0]["lesions"]
            }
        summary[kind] = entry
    return summary
