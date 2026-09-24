"""Imitation of the scripted teacher on the manipulation benchmark, on the frozen connectome.

The controller is the B1a whole-body interface of flyarm.whole_brain: the 220-feature
observation is written through a trainable linear encoder into the 1,846 ascending neurons of
the frozen MaleCNS, the 1,314 descending and 708 VNC motor neurons are read out through a
frozen unit-norm calibration (research log E26) and a trainable linear decoder gives the 5
actions, one per control step. Training is the same masked sequence behavior cloning with
truncated BPTT through the frozen connectome as every other B1a and B2 run
(flyarm.whole_brain.training.train_sequence_policy), followed by DAgger rounds that roll out
the learner and label every visited state with the ManipulationTeacher, and a closed-loop
choice among the phases on validation episodes (research log E33).

Why a sibling of flyarm.whole_brain.experiment and flyarm.flyleg.experiment rather than a
branch of them: both are built around one Gymnasium environment stepped one episode at a time
with a single-episode controller, which is right for 100 to 400-step episodes. Here episodes
are 1,100 to 2,600 steps and every number is a per-split, per-template rate, so demonstration
collection, DAgger, selection and evaluation all run batched (flyarm.manipulation.rollout),
and the data layer adds what the long episodes need (per-skill loss weights, length-bucketed
batches, host-side episode storage). The policy classes, the trainer, the interface, the
shuffle and the phase-selection rule are the existing ones.

Controls through the same code path: ``shuffled`` (a degree-preserving shuffle of the whole
CNS, the same interface), ``gru`` (parameter-matched) and ``mlp`` (the D4RL BC architecture).
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
from flyarm.config import ManipulationImitationConfig
from flyarm.experiment import save_json
from flyarm.interfaces import NeuralInterface
from flyarm.manipulation import rollout
from flyarm.manipulation.env import DEFAULT_ASSET_ROOT, BatchedManipulation
from flyarm.manipulation.sim import OBS_DIM
from flyarm.manipulation.splits import record_path
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.interface import annotation_interface, interface_report
from flyarm.whole_brain.policy import (
    BrainPolicy,
    GRUPolicy,
    MLPPolicy,
    SequencePolicy,
    gru_hidden_for_budget,
)
from flyarm.whole_brain.shuffle import shuffle_pack
from flyarm.whole_brain.training import Budget, train_sequence_policy

ACTION_DIM = rollout.ACTION_DIM
BRAIN_KINDS = ("connectome", "shuffled")


class PolicyActor:
    """A sequence policy driving N episodes at once; its output is the executed action."""

    def __init__(self, policy: SequencePolicy, num_envs: int) -> None:
        if policy.chunk != 1:
            raise ValueError("the manipulation benchmark drives one action per step")
        self.policy = policy
        self.state = policy.initial_state(num_envs)

    def act(self, obs: np.ndarray) -> np.ndarray:
        output, self.state = self.policy.step(mx.array(obs.astype(np.float32)), self.state)
        mx.eval(output, self.state)
        return np.asarray(output, dtype=np.float64)


def shuffle_seed(seed: int) -> int:
    return seed + 17000


def dynamics_for(
    kind: str, seed: int, pack: ConnectomePack, interface: NeuralInterface
) -> tuple[RateDynamics | None, dict[str, Any] | None]:
    """The frozen graph of a controller kind, and the shuffle's record when there is one."""
    if kind == "connectome":
        return RateDynamics(pack, interface), None
    if kind != "shuffled":
        return None, None
    shuffled = shuffle_pack(pack, shuffle_seed(seed))
    rebound = NeuralInterface.bind(
        shuffled, interface.input_body_ids, interface.output_body_ids, label=interface.label
    )
    record = {
        "fingerprint": shuffled.fingerprint(),
        **{key: value for key, value in shuffled.manifest.items() if key != "sources"},
        "interface_fingerprint": rebound.fingerprint,
        "shuffle_seed": shuffle_seed(seed),
    }
    return RateDynamics(shuffled, rebound), record


def build_policy(
    kind: str,
    config: ManipulationImitationConfig,
    seed: int,
    dynamics: RateDynamics | None,
    brain_budget: int,
) -> SequencePolicy:
    """An untrained controller of one kind; the same constructor serves training and replay."""
    dims = {"obs_dim": OBS_DIM, "action_dim": ACTION_DIM, "chunk": config.action_chunk}
    if kind == "mlp":
        return MLPPolicy(**dims, seed=seed)
    if kind == "gru":
        hidden = gru_hidden_for_budget(OBS_DIM, ACTION_DIM * config.action_chunk, brain_budget)
        return GRUPolicy(**dims, hidden=hidden, seed=seed)
    if dynamics is None:
        raise ValueError(f"{kind} needs connectome dynamics")
    return BrainPolicy(kind, dynamics, neural_steps=config.neural_steps, seed=seed, **dims)


def brain_budget(pack: ConnectomePack, interface: NeuralInterface, neural_steps: int) -> int:
    """Trainable parameters of the connectome policy, the GRU control's budget."""
    policy = BrainPolicy(
        "connectome",
        RateDynamics(pack, interface),
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        neural_steps=neural_steps,
    )
    return policy.trainable_parameter_count()


class Workbench:
    """The batched environments of one run, built once and reused (a compile takes seconds)."""

    def __init__(
        self, model_path: Path, asset_root: Path, cue: bool, velocities: bool = True
    ) -> None:
        self.model_path, self.asset_root, self.cue = Path(model_path), Path(asset_root), cue
        self.velocities = velocities
        self._envs: dict[tuple[str, int], BatchedManipulation] = {}

    def env(self, episodes: rollout.EpisodePlan) -> BatchedManipulation:
        key = (episodes.split, len(episodes))
        if key not in self._envs:
            self._envs[key] = rollout.make_env(
                self.model_path,
                episodes,
                asset_root=self.asset_root,
                cue=self.cue,
                velocities=self.velocities,
            )
        return self._envs[key]

    def run(
        self,
        plans: list[rollout.EpisodePlan],
        actor_for: Any,
        **options: Any,
    ) -> list[rollout.EpisodeLog]:
        """Run ``plans`` in lockstep; ``actor_for(n)`` builds the actor for n episodes or None."""
        envs = [self.env(episodes) for episodes in plans]
        actor = actor_for(sum(len(episodes) for episodes in plans))
        return rollout.run_episodes(envs, plans, actor, **options)

    def evaluate(
        self, policy: SequencePolicy | None, plans: list[rollout.EpisodePlan]
    ) -> dict[str, dict[str, Any]]:
        """Per-split summaries of the policy (or of the teacher when ``policy`` is None)."""
        logs = self.run(plans, lambda n: None if policy is None else PolicyActor(policy, n))
        return {log.plan.split: rollout.summarize(log) for log in logs}

    def collect(self, episodes: rollout.EpisodePlan) -> tuple[dict[str, np.ndarray], dict]:
        """Teacher demonstrations of ``episodes``: recorded steps and their summary."""
        (log,) = self.run([episodes], lambda n: None, record=True)
        return log.data(), rollout.summarize(log)


def _weights(config: ManipulationImitationConfig, data: dict[str, np.ndarray]) -> np.ndarray:
    if config.sample_weights == "uniform":
        return data["mask"].astype(np.float32)
    return rollout.skill_weights(data, config.max_skill_weight)


def _save_data(path: Path, data: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **data)


def load_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {key: stored[key] for key in ("obs", "actions", "mask", "skill")}


def evaluation_plans(
    config: ManipulationImitationConfig, per_template: int
) -> list[rollout.EpisodePlan]:
    return [rollout.plan(split, per_template, rollout.TEST_OFFSET) for split in config.eval_splits]


def _train(
    policy: SequencePolicy,
    config: ManipulationImitationConfig,
    bench: Workbench,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    seed: int,
    deadline: float,
    run: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    """Behavior cloning, DAgger rounds and the closed-loop choice among the phases."""
    calibration = None
    brain = isinstance(policy, BrainPolicy)
    if brain:
        samples = train["obs"][train["mask"] > 0]
        policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
        calibration = policy.calibrate_readout(train["obs"], train["mask"], unit_norm=True)
    validation_plan = rollout.plan(
        "train", config.val_episodes_per_template, rollout.VALIDATION_OFFSET
    )
    scored: list[dict[str, Any]] = []  # every closed-loop validation of the current phase

    def closed_loop(candidate: SequencePolicy) -> float:
        summary = bench.evaluate(candidate, [validation_plan])["train"]
        scored.append(summary)
        print(
            f"    validation: success {summary['successes']}/{summary['episodes']} "
            f"subgoals {summary['subgoal_fraction']:.3f}",
            flush=True,
        )
        return float(summary["selection_score"])

    selector = closed_loop if config.selection == "closed_loop" else None
    validation_weights = _weights(config, validation)

    def fit(data: dict[str, np.ndarray], phase: str, first: bool, phase_seed: int) -> tuple:
        return train_sequence_policy(
            policy,
            data,
            _weights(config, data),
            validation,
            validation_weights,
            Budget(
                epochs=config.epochs if first else config.dagger_epochs,
                decoder_warmup_epochs=config.decoder_warmup_epochs if brain and first else 0,
                batch_size=config.batch_size,
                bptt_steps=config.bptt_steps,
                learning_rate=config.learning_rate,
                deadline=deadline,
                loss=config.loss,
                # The scale is about the connectome's ascending neurons; the controls' input
                # modules (a GRU cell, an MLP's first layer) train at the full rate.
                window_batch=config.window_batch if config.sampling == "windows" else None,
                burn_in=config.burn_in,
                input_learning_rate=(
                    config.learning_rate * config.encoder_learning_rate_scale if brain else None
                ),
            ),
            phase_seed,
            phase=phase,
            set_normalization=first and not brain,
            selector=selector,
            select_every=config.select_every,
            length_buckets=config.length_buckets,
        )

    phases: list[dict[str, Any]] = []
    best: tuple[float, str] | None = None

    def close_phase(phase: str, summary: dict[str, Any], extra: dict[str, Any]) -> None:
        nonlocal best
        score = summary.get("selection_score")
        if score is None:  # selected on imitation loss: score the kept weights closed loop
            score = closed_loop(policy)
        # The kept weights are those of the best-scored epoch; evaluation is deterministic.
        validation = next(item for item in scored if item["selection_score"] == score)
        scored.clear()
        entry = {"phase": phase, **summary, **extra, "validation": validation}
        entry["phase_score"] = score
        phases.append(entry)
        policy.save(run / f"policy-{phase}.safetensors")
        if best is None or score > best[0]:
            best = (score, phase)

    curves, summary = fit(train, "behavior_cloning", True, seed)
    close_phase("behavior_cloning", summary, {"episodes": int(train["obs"].shape[0])})
    aggregate = train
    for iteration in range(config.dagger_iterations):
        beta = config.dagger_beta * config.dagger_beta_decay**iteration
        offset = rollout.DAGGER_OFFSET + rollout.DAGGER_ROUND_STRIDE * iteration
        episodes = rollout.plan("train", config.dagger_episodes_per_template, offset)
        generator = np.random.default_rng([seed, iteration, 7])
        (log,) = bench.run(
            [episodes],
            lambda n: PolicyActor(policy, n),
            record=True,
            beta=beta,
            generator=generator,
        )
        rollouts = log.data()
        _save_data(run / f"dagger-{iteration + 1}.npz", rollouts)
        stats = rollout.summarize(log)
        aggregate = rollout.concatenate([aggregate, rollouts])
        phase = f"dagger_{iteration + 1}"
        print(
            f"  {phase}: beta {beta:.2f}, rollout success {stats['successes']}/"
            f"{stats['episodes']}, subgoals {stats['subgoal_fraction']:.3f}",
            flush=True,
        )
        more, summary = fit(aggregate, phase, False, seed + 1000 * (iteration + 1))
        curves += more
        close_phase(
            phase,
            summary,
            {
                "beta": beta,
                "rollout": {k: stats[k] for k in ("success_rate", "subgoal_fraction")},
                "teacher_step_fraction": stats["teacher_step_fraction"],
                "labelled_steps": int(rollouts["mask"].sum()),
                "aggregate_episodes": int(aggregate["obs"].shape[0]),
            },
        )
    assert best is not None
    selected = best[1] if config.phase_selection == "validation_success" else phases[-1]["phase"]
    policy.load(run / f"policy-{selected}.safetensors")
    for entry in phases:
        entry["selected"] = entry["phase"] == selected
    print(f"  selected phase {selected}", flush=True)
    training = {
        "selected_phase": selected,
        "phases": phases,
        "training_seconds": float(sum(entry["training_seconds"] for entry in phases)),
    }
    return curves, training, calibration


def _provenance(
    pack: ConnectomePack, interface: NeuralInterface, config: ManipulationImitationConfig
) -> dict[str, Any]:
    package = Path(__file__).resolve().parent.parent
    return {
        "pack_root": str(pack.root),
        "pack_fingerprint": pack.fingerprint(),
        "interface_fingerprint": interface.fingerprint,
        "splits_record_sha256": digest_file(record_path()),
        "source_files_sha256": {
            str(path.relative_to(package)): digest_file(path)
            for path in sorted(package.rglob("*.py"))
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ["mlx", "mujoco", "numpy"]},
        "device": str(mx.default_device()),
        "dynamics": (
            f"h <- 0.5 h + 0.5 tanh(I + 0.8 W h), {config.neural_steps} neural steps per "
            "20 Hz control step, state kept across steps and cleared at episode reset"
        ),
        "trainable": "encoder obs->ascending current, decoder pooled outputs->action",
        "seeds": {
            "demonstrations": "train split, offset 0",
            "validation": f"train split, offset {rollout.VALIDATION_OFFSET}",
            "dagger": f"train split, offset {rollout.DAGGER_OFFSET} + "
            f"{rollout.DAGGER_ROUND_STRIDE} x round",
            "test": "each held-out split's own seed block, offset 0",
            "layout": f"seed_start + offset + {rollout.TEMPLATE_STRIDE} x template + episode",
        },
    }


def run_manipulation_imitation(
    pack_root: Path,
    model_path: Path,
    output: Path,
    config: ManipulationImitationConfig,
    asset_root: Path = DEFAULT_ASSET_ROOT,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose a new output: {output}")
    started = time.monotonic()
    try:
        return _run(pack_root, model_path, output, config, asset_root)
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


def _run(
    pack_root: Path,
    model_path: Path,
    output: Path,
    config: ManipulationImitationConfig,
    asset_root: Path,
) -> dict[str, Any]:
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    interface = annotation_interface(pack)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + config.max_seconds
    interface.save(output / "interface.json")
    save_json(output / "interface_report.json", interface_report(pack, interface))
    save_json(output / "config.json", config.model_dump())
    save_json(output / "provenance.json", _provenance(pack, interface, config))
    bench = Workbench(model_path, asset_root, config.cue, config.velocities)
    demonstrations = rollout.plan(
        "train", config.train_episodes_per_template, rollout.DEMONSTRATION_OFFSET
    )
    validation_plan = rollout.plan(
        "train", config.val_episodes_per_template, rollout.VALIDATION_OFFSET
    )
    train, train_summary = bench.collect(demonstrations)
    validation, validation_summary = bench.collect(validation_plan)
    _save_data(output / "train.npz", train)
    _save_data(output / "validation.npz", validation)
    plans = evaluation_plans(config, config.eval_episodes_per_template)
    save_json(
        output / "splits.json",
        {
            "train": {"seeds": list(demonstrations.seeds), "templates": demonstrations.templates},
            "validation": {
                "seeds": list(validation_plan.seeds),
                "templates": validation_plan.templates,
            },
            "test": {p.split: {"seeds": list(p.seeds), "templates": p.templates} for p in plans},
        },
    )
    results: dict[str, Any] = {
        "status": "running",
        "benchmark": "articulated multi-step manipulation (flyarm.manipulation)",
        "claim": "B1a whole-body connectome controller; no topology advantage assumed",
        "teacher_demonstrations": train_summary,
        "teacher_validation": validation_summary,
        "training_steps": int(train["mask"].sum()),
        "models": [],
    }
    print(
        f"demonstrations: {train_summary['successes']}/{train_summary['episodes']} teacher "
        f"successes, {results['training_steps']} steps",
        flush=True,
    )
    if config.evaluate_teacher:
        results["teacher"] = bench.evaluate(None, plans)
    save_json(output / "results.json", results)

    budget = brain_budget(pack, interface, config.neural_steps)
    for seed in config.seeds:
        for kind in config.policies:
            print(f"manipulation {kind} seed={seed}", flush=True)
            dynamics, shuffle_record = dynamics_for(kind, seed, pack, interface)
            if shuffle_record is not None:
                save_json(output / f"shuffled-{seed}.json", shuffle_record)
            run = output / f"{kind}-{seed}"
            run.mkdir()
            policy = build_policy(kind, config, seed, dynamics, budget)
            curves, training, calibration = _train(
                policy, config, bench, train, validation, seed, deadline, run
            )
            save_json(run / "learning.json", curves)
            policy.save(run / "policy.safetensors")
            evaluation = bench.evaluate(policy, plans)
            item: dict[str, Any] = {
                "kind": kind,
                "seed": seed,
                "trainable_parameters": policy.trainable_parameter_count(),
                "state_size": policy.state_size,
                "readout_calibration": calibration,
                **training,
                "evaluation": evaluation,
            }
            save_json(run / "evaluation.json", item)
            results["models"].append(item)
            save_json(output / "results.json", results)
            print(
                "  "
                + "; ".join(
                    f"{split} success {r['successes']}/{r['episodes']} "
                    f"subgoals {r['subgoal_fraction']:.3f}"
                    for split, r in evaluation.items()
                ),
                flush=True,
            )
    results.update(status="complete", elapsed_seconds=time.monotonic() - started)
    save_json(output / "results.json", results)
    return results


def load_manipulation_policy(
    run_root: Path, kind: str, seed: int, pack_root: Path, checkpoint: Path | None = None
) -> tuple[ManipulationImitationConfig, SequencePolicy]:
    """Rebuild a trained checkpoint over its exact frozen graph (a shuffle is regenerated)."""
    config = ManipulationImitationConfig.model_validate_json((run_root / "config.json").read_text())
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    interface = NeuralInterface.load(run_root / "interface.json")
    dynamics, shuffle_record = dynamics_for(kind, seed, pack, interface)
    if shuffle_record is not None:
        recorded = json.loads((run_root / f"shuffled-{seed}.json").read_text())
        if recorded["fingerprint"] != shuffle_record["fingerprint"]:
            raise ValueError("Regenerated shuffle differs from the one used in training")
    budget = brain_budget(pack, interface, config.neural_steps)
    policy = build_policy(kind, config, seed, dynamics, budget)
    path = checkpoint or run_root / f"{kind}-{seed}" / "policy.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"No {kind} seed-{seed} checkpoint: {path}")
    policy.load(path)
    return config, policy


__all__ = [
    "PolicyActor",
    "Workbench",
    "build_policy",
    "load_data",
    "load_manipulation_policy",
    "run_manipulation_imitation",
]
