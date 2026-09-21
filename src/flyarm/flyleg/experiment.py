"""B2 experiment: the fly's left front leg drives a Franka on D4RL FrankaKitchen.

Controllers, all trained by the same masked sequence behavior cloning on the same episodes:

- ``flyleg``: complete MaleCNS; joint state -> left front-leg proprioceptors, scene state ->
  head sensory neurons, joint velocities <- left front-leg motor neurons.
- ``flyleg_shuffled``: the same interface on a degree-preserving shuffle of the whole CNS.
- ``mlp``: the standard D4RL BC architecture (two 256-unit ReLU layers, no memory).
- ``gru``: a recurrent control with the fly policy's trainable-parameter budget.

Post-training checks of every ``flyleg`` checkpoint: edges off, direct sensory-to-motor
synapses only, deafferented leg (proprioceptors silenced), head-sensory deprivation and
state reset every step. OOD evaluations of every checkpoint: 10x robot or object observation
noise and seeded initial arm-joint offsets.
"""

from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flyarm.assets import digest_file
from flyarm.benchmarks import kitchen
from flyarm.config import FlyLegConfig
from flyarm.experiment import save_json
from flyarm.flyleg.interface import front_leg_interface, front_leg_report
from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.diagnostics import direct_only_weights
from flyarm.whole_brain.experiment import ResetEveryStep
from flyarm.whole_brain.policy import (
    BrainPolicy,
    GRUPolicy,
    MLPPolicy,
    MlxController,
    SequencePolicy,
    gru_hidden_for_budget,
)
from flyarm.whole_brain.shuffle import shuffle_pack
from flyarm.whole_brain.training import Budget, train_sequence_policy

VALIDATION_SEED = 90_000


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
    offset = float(variant.pop("initial_joint_offset", 0.0))
    env = kitchen.recover_env(config.split, **variant)
    features = kitchen.PositionFeatures(controller) if controller is not None else None
    try:
        result = kitchen.evaluate(env, features, seeds, initial_joint_offset=offset)
    finally:
        env.close()
    return {**result, "environment_overrides": variant}


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
    leg = front_leg_interface(pack, annotations)
    data = kitchen.load(config.split)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + config.max_seconds
    leg.interface.save(output / "interface.json")
    save_json(output / "interface_report.json", front_leg_report(pack, leg, annotations))
    save_json(output / "config.json", config.model_dump())
    package = Path(__file__).resolve().parent.parent
    save_json(
        output / "provenance.json",
        {
            "pack_fingerprint": pack.fingerprint(),
            "interface_fingerprint": leg.interface.fingerprint,
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
    save_json(output / "results.json", results)

    channels = [
        (kitchen.PROPRIOCEPTION.start, kitchen.PROPRIOCEPTION.stop, len(leg.proprioceptors)),
        (kitchen.EXTEROCEPTION.start, kitchen.EXTEROCEPTION.stop, len(leg.exteroceptors)),
    ]
    measured = RateDynamics(pack, leg.interface)
    budget_policy = BrainPolicy(
        "flyleg",
        measured,
        obs_dim=kitchen.FEATURE_DIM,
        action_dim=kitchen.ACTION_DIM,
        channels=channels,
    )
    fly_budget = budget_policy.trainable_parameter_count()
    weights_train, weights_validation = train["mask"], validation["mask"]

    for seed in config.seeds:
        dynamics = {"flyleg": measured}
        if "flyleg_shuffled" in config.policies:
            shuffled = shuffle_pack(pack, seed + 17000)
            rebound = NeuralInterface.bind(
                shuffled,
                leg.interface.input_body_ids,
                leg.interface.output_body_ids,
                label=leg.interface.label,
            )
            dynamics["flyleg_shuffled"] = RateDynamics(shuffled, rebound)
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
            policy: SequencePolicy
            if kind == "mlp":
                policy = MLPPolicy(
                    obs_dim=kitchen.FEATURE_DIM, action_dim=kitchen.ACTION_DIM, seed=seed
                )
            elif kind == "gru":
                hidden = gru_hidden_for_budget(kitchen.FEATURE_DIM, kitchen.ACTION_DIM, fly_budget)
                policy = GRUPolicy(
                    obs_dim=kitchen.FEATURE_DIM,
                    action_dim=kitchen.ACTION_DIM,
                    hidden=hidden,
                    seed=seed,
                )
            else:
                policy = BrainPolicy(
                    kind,
                    dynamics[kind],
                    obs_dim=kitchen.FEATURE_DIM,
                    action_dim=kitchen.ACTION_DIM,
                    neural_steps=config.neural_steps,
                    seed=seed,
                    channels=channels,
                )
            calibration = None
            if isinstance(policy, BrainPolicy):
                samples = train["obs"][train["mask"].astype(bool)]
                policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
                calibration = policy.calibrate_readout(train["obs"], train["mask"])
            curves, training = train_sequence_policy(
                policy,
                train,
                weights_train,
                validation,
                weights_validation,
                Budget(
                    epochs=config.epochs,
                    decoder_warmup_epochs=(
                        config.decoder_warmup_epochs if isinstance(policy, BrainPolicy) else 0
                    ),
                    batch_size=config.batch_size,
                    bptt_steps=config.bptt_steps,
                    learning_rate=config.learning_rate,
                    deadline=deadline,
                ),
                seed,
                set_normalization=not isinstance(policy, BrainPolicy),
            )
            save_json(run / "learning.json", curves)
            policy.save(run / "policy.safetensors")
            item: dict[str, Any] = {
                "kind": kind,
                "seed": seed,
                "trainable_parameters": policy.trainable_parameter_count(),
                "state_size": policy.state_size,
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
                        "edges_off", RateDynamics(pack, leg.interface, edges=False)
                    ),
                    "direct_only": policy.with_dynamics(
                        "direct_only",
                        RateDynamics(
                            pack, leg.interface, weights=direct_only_weights(pack, leg.interface)
                        ),
                    ),
                    "deafferented_leg": policy.silence_channel("deafferented_leg", 0),
                    "head_sensory_deprived": policy.silence_channel("head_sensory_deprived", 1),
                }
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
