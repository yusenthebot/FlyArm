"""B1a experiment: full MaleCNS vs full shuffled vs edges-off vs parameter-matched GRU.

The robot side is unchanged from the subgraph protocols: the same Menagerie Panda, the
same teachers, the same train/validation/test episode seeds and the same evaluation
code. Only the controller changes. Post-training ablations of the measured-graph policy:

- ``edges_off``: every connectome edge removed. Input and output sets are disjoint, so
  outputs are exactly zero and the action is constant; retraining this control is
  therefore structurally pointless and it is evaluated with the learned adapters.
- ``direct_only``: only the 38,772 direct ascending -> output synapses kept. Measures how
  much of the policy runs through 1-hop shortcuts rather than the rest of the CNS.
- ``state_reset_every_step``: edges kept, memory across control steps removed.
"""

from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import mlx.core as mx
import numpy as np

from flyarm import experiment as reach
from flyarm import pick_place_experiment as pick
from flyarm.assets import MENAGERIE_SHA, digest_file, verify_arm
from flyarm.config import WholeBrainConfig
from flyarm.env import PandaReachEnv
from flyarm.interfaces import NeuralInterface
from flyarm.io import save_json
from flyarm.pick_place_env import PandaPickPlaceEnv, physical_stage
from flyarm.video import annotate
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.diagnostics import direct_only_weights
from flyarm.whole_brain.interface import annotation_interface, interface_report
from flyarm.whole_brain.policy import (
    BrainPolicy,
    GRUPolicy,
    MlxController,
    SequencePolicy,
    gru_hidden_for_budget,
)
from flyarm.whole_brain.shuffle import shuffle_pack
from flyarm.whole_brain.training import Budget, train_sequence_policy

SPLITS = {
    # Identical to the subgraph protocols so results are directly comparable.
    "reach": {"train": 0, "validation": 10000, "test": 30000},
    "pick-place": {"train": 40000, "validation": 50000, "test": 60000, "dagger": 70000},
}


class ResetEveryStep:
    """Controller wrapper that clears neural state before each action.

    Only the recurrent state is cleared; a chunked controller keeps its fixed output
    ensemble, so the lesion removes the brain's memory and nothing else.
    """

    def __init__(self, controller: MlxController) -> None:
        self.controller = controller

    def reset(self) -> None:
        self.controller.reset()

    def act(self, observation: np.ndarray) -> np.ndarray:
        self.controller.reset_state()
        return self.controller.act(observation)


class Task:
    """The unchanged robot task: env, teacher data, loss weights and evaluation."""

    def __init__(self, config: WholeBrainConfig, model_path: Path) -> None:
        self.name = config.task
        self.config = config
        self.env: PandaReachEnv | PandaPickPlaceEnv
        if config.task == "reach":
            self.env = PandaReachEnv(model_path, horizon=config.horizon)
            self.obs_dim, self.action_dim = 20, 3
        else:
            self.env = PandaPickPlaceEnv(
                model_path, horizon=config.horizon, teacher_resync=config.teacher == "resync"
            )
            self.obs_dim, self.action_dim = self.env.observation_dim, 4

    def seeds(self, split: str, count: int) -> list[int]:
        start = SPLITS[self.name][split]
        return list(range(start, start + count))

    def collect(self, seeds: list[int], path: Path) -> dict[str, np.ndarray]:
        if isinstance(self.env, PandaReachEnv):
            return reach.collect(self.env, seeds, path)
        return pick.collect_demonstrations(self.env, seeds, path)

    def weights(self, data: dict[str, np.ndarray]) -> np.ndarray:
        if self.name == "reach":
            return data["mask"].astype(np.float32)
        return pick.stage_balanced_weights(data).astype(np.float32)

    def evaluate(self, controller: Any, seeds: list[int], **kwargs: Any) -> dict[str, Any]:
        if isinstance(self.env, PandaReachEnv):
            return reach.evaluate(self.env, controller, seeds, **kwargs)
        return pick.evaluate(self.env, controller, seeds, **kwargs)

    def baseline(self, mode: str, seeds: list[int]) -> dict[str, Any]:
        return self.evaluate(None, seeds, mode=mode)

    def close(self) -> None:
        self.env.close()


def _scalars(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
        for key, value in info.items()
        if isinstance(value, (bool, int, float, np.bool_, np.floating, np.integer))
    }


def _status(info: dict[str, Any]) -> tuple[str, bool]:
    if "goal_xy_error" in info:
        stage = physical_stage(info)
        return f"{stage} · goal error {100 * info['goal_xy_error']:.1f} cm", stage == "placed"
    return f"distance {1000 * info['distance']:.1f} mm", bool(info["is_success"])


def record_rollouts(
    task: Task,
    policy: SequencePolicy,
    seeds: list[int],
    output: Path,
    *,
    title: str | None = None,
) -> None:
    """Real MuJoCo frames plus a same-frame trace; output-neuron activity goes to an NPZ.

    With ``title`` every frame carries the controller name, episode, step and physical state.
    """
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    controller = MlxController(policy)
    trace: list[dict[str, Any]] = []
    activity: list[np.ndarray] = []

    def frame(seed: int, step: int, info: dict[str, Any]) -> np.ndarray:
        image = task.env.render()
        if title is None:
            return image
        status, success = _status(info)
        return annotate(image, title, f"episode {seed} · step {step} · {status}", success=success)

    with cast(Any, imageio.get_writer(output, fps=20, codec="libx264", quality=7)) as writer:
        for seed in seeds:
            observation, info = task.env.reset(seed=seed)
            controller.reset()
            writer.append_data(frame(seed, 0, info))
            for step in range(task.env.horizon):
                action = controller.act(observation)
                observation, _, terminated, truncated, info = task.env.step(action)
                outputs = controller.output_activity()
                if outputs is not None:
                    activity.append(outputs.astype(np.float16))
                trace.append(
                    {
                        "seed": seed,
                        "step": step,
                        "action": action.tolist(),
                        "output_rms": (
                            float(np.sqrt(np.mean(outputs**2))) if outputs is not None else None
                        ),
                        **_scalars(info),
                    }
                )
                writer.append_data(frame(seed, step + 1, info))
                if terminated or truncated:
                    break
    save_json(
        output.with_suffix(".json"),
        {"kind": policy.kind, "fps": 20, "seeds": seeds, "trajectory": trace},
    )
    if activity:
        np.savez_compressed(output.with_name(output.stem + "-outputs.npz"), outputs=activity)


def shuffle_seed(seed: int, replicate: int) -> int:
    return seed + 17000 + 1000 * replicate


def run_name(kind: str, seed: int, replicate: int = 0) -> str:
    """Checkpoint directory name; replicate 0 keeps the name used before replicates existed."""
    return f"{kind}-{seed}" if replicate == 0 else f"{kind}-{seed}-r{replicate}"


def load_trained_policy(
    run_root: Path,
    kind: str,
    seed: int,
    pack_root: Path,
    model_path: Path,
    *,
    replicate: int = 0,
) -> tuple[Task, Any]:
    """Rebuild a saved checkpoint over its exact frozen graph (shuffles are regenerated)."""
    config = WholeBrainConfig.model_validate_json((run_root / "config.json").read_text())
    verify_arm(model_path.resolve().parent.parent)
    task = Task(config, model_path)
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    interface = NeuralInterface.load(run_root / "interface.json")
    dynamics = {"connectome": RateDynamics(pack, interface)}
    if kind == "shuffled":
        shuffled = shuffle_pack(pack, shuffle_seed(seed, replicate))
        recorded = json.loads(
            (run_root / f"{run_name('shuffled', seed, replicate)}.json").read_text()
        )
        if shuffled.fingerprint() != recorded["fingerprint"]:
            raise ValueError("Regenerated shuffle differs from the one used in training")
        rebound = NeuralInterface.bind(
            shuffled, interface.input_body_ids, interface.output_body_ids, label=interface.label
        )
        dynamics["shuffled"] = RateDynamics(shuffled, rebound)
    budget = BrainPolicy(
        "connectome", dynamics["connectome"], obs_dim=task.obs_dim, action_dim=task.action_dim
    ).trainable_parameter_count()
    policy = _build(kind, seed, task, config, dynamics, budget)
    policy.load(run_root / run_name(kind, seed, replicate) / "policy.safetensors")
    return task, policy


def _summary(evaluation: dict[str, Any]) -> dict[str, float]:
    keys = ("success_rate", "grasp_rate", "lift_rate", "mean_final_distance_m")
    return {key: evaluation[key] for key in keys if key in evaluation}


def evidence(models: list[dict[str, Any]]) -> dict[str, Any]:
    """The preregistered B1a reading rules, applied to held-out clean success."""

    def rate(kind: str, field: str = "clean") -> float | None:
        values = [m[field]["success_rate"] for m in models if m["kind"] == kind and field in m]
        return float(np.mean(values)) if values else None

    brain, shuffled, gru = rate("connectome"), rate("shuffled"), rate("gru")
    edges_off = rate("connectome", "edges_off")
    direct = rate("connectome", "direct_only")
    seeds = len({m["seed"] for m in models if m["kind"] == "connectome"})
    readings: list[str] = []
    if brain is not None and edges_off is not None:
        if brain > 0.7 and edges_off < 0.1:
            readings.append("graph-mediated control supported (MaleCNS > 70%, edges-off < 10%)")
        else:
            readings.append("graph-mediated control not established")
    if brain is not None and direct is not None and brain > 0.7:
        if direct >= brain - 0.1:
            readings.append("direct ascending->output synapses alone reproduce the policy")
        else:
            readings.append("policy needs connectome paths beyond direct I/O synapses")
    if brain is not None and shuffled is not None and brain > 0.7:
        if brain - shuffled >= 0.2 and seeds >= 3:
            readings.append("candidate topology advantage; needs a significance test")
        elif abs(brain - shuffled) < 0.2:
            readings.append("measured and shuffled graphs equivalent: graph is a usable medium")
    if gru is not None and brain is not None and gru > brain + 0.1:
        readings.append("GRU better: task is solvable, no connectome advantage")
    return {
        "connectome_success": brain,
        "shuffled_success": shuffled,
        "gru_success": gru,
        "edges_off_success": edges_off,
        "direct_only_success": direct,
        "connectome_seeds": seeds,
        "readings": readings or ["pending"],
    }


def _build(
    kind: str,
    seed: int,
    task: Task,
    config: WholeBrainConfig,
    dynamics: dict[str, RateDynamics],
    brain_budget: int,
) -> SequencePolicy:
    if kind == "gru":
        hidden = gru_hidden_for_budget(task.obs_dim, task.action_dim, brain_budget)
        return GRUPolicy(obs_dim=task.obs_dim, action_dim=task.action_dim, hidden=hidden, seed=seed)
    return BrainPolicy(
        kind,
        dynamics[kind],
        obs_dim=task.obs_dim,
        action_dim=task.action_dim,
        neural_steps=config.neural_steps,
        seed=seed,
    )


def _train(
    policy: SequencePolicy,
    task: Task,
    config: WholeBrainConfig,
    data: dict[str, dict[str, np.ndarray]],
    seed: int,
    deadline: float,
    run: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[int]]]:
    budget = Budget(
        epochs=config.epochs,
        decoder_warmup_epochs=config.decoder_warmup_epochs,
        batch_size=config.batch_size,
        bptt_steps=config.bptt_steps,
        learning_rate=config.learning_rate,
        deadline=deadline,
    )
    train, validation = data["train"], data["validation"]
    validation_weights = task.weights(validation)
    curves, summary = train_sequence_policy(
        policy, train, task.weights(train), validation, validation_weights, budget, seed
    )
    phases = [{"phase": "behavior_cloning", **summary}]
    selected = _PhaseSelector(policy, task, config, run)
    selected.score(phases[-1])
    dagger_sets: list[list[int]] = []
    aggregate = train
    for iteration in range(config.dagger_iterations):
        if not isinstance(task.env, PandaPickPlaceEnv):
            raise ValueError("DAgger is defined for the pick-place teacher only")
        seeds = task.seeds("dagger", config.dagger_episodes)
        seeds = [value + iteration * 1000 for value in seeds]
        dagger_sets.append(seeds)
        queries = pick.collect_dagger_queries(
            task.env, MlxController(policy), seeds, run / f"dagger-{iteration + 1}.npz"
        )
        aggregate = pick.concatenate_data(aggregate, queries)
        phase = f"dagger_{iteration + 1}"
        more, summary = train_sequence_policy(
            policy,
            aggregate,
            task.weights(aggregate),
            validation,
            validation_weights,
            Budget(
                epochs=config.dagger_epochs,
                decoder_warmup_epochs=0,
                batch_size=config.batch_size,
                bptt_steps=config.bptt_steps,
                learning_rate=config.learning_rate,
                deadline=deadline,
            ),
            seed + iteration + 1000,
            phase=phase,
            set_normalization=False,
        )
        curves.extend(more)
        phases.append({"phase": phase, **summary})
        selected.score(phases[-1])
    selected.restore(phases)
    return curves, phases, dagger_sets


class _PhaseSelector:
    """Closed-loop choice among training phases (config.phase_selection).

    With "validation_success" every phase's weights are saved and scored by stable-place
    success (lift as tie-break) on the validation seeds, and the best phase is restored at the
    end: DAgger rounds can collapse the controller while lowering the imitation loss (research
    log E33). With "last" the final phase is kept, as in every run before E33.
    """

    def __init__(self, policy: SequencePolicy, task: Task, config: WholeBrainConfig, run: Path):
        self.policy, self.task, self.run = policy, task, run
        self.active = config.phase_selection == "validation_success"
        self.seeds = task.seeds("validation", config.val_episodes)
        self.best: tuple[float, float] | None = None
        self.best_phase: str | None = None

    def score(self, phase: dict[str, Any]) -> None:
        if not self.active:
            return
        result = self.task.evaluate(MlxController(self.policy), self.seeds)
        key = (float(result["success_rate"]), float(result["lift_rate"]))
        phase["validation_success_rate"], phase["validation_lift_rate"] = key
        path = self.run / f"policy-{phase['phase']}.safetensors"
        self.policy.save(path)
        print(f"  {phase['phase']}: validation success {key[0]:.3f} lift {key[1]:.3f}", flush=True)
        if self.best is None or key > self.best:
            self.best, self.best_phase = key, str(phase["phase"])

    def restore(self, phases: list[dict[str, Any]]) -> None:
        if not self.active or self.best_phase is None:
            return
        self.policy.load(self.run / f"policy-{self.best_phase}.safetensors")
        for phase in phases:
            phase["selected"] = phase["phase"] == self.best_phase
        print(f"  selected phase {self.best_phase}", flush=True)


def _provenance(
    pack: ConnectomePack,
    interface: NeuralInterface,
    config: WholeBrainConfig,
    model_path: Path,
) -> dict[str, Any]:
    package = Path(__file__).resolve().parent.parent
    return {
        "pack_root": str(pack.root),
        "pack_fingerprint": pack.fingerprint(),
        "pack_manifest": pack.manifest,
        "interface_fingerprint": interface.fingerprint,
        "source_files_sha256": {
            str(path.relative_to(package)): digest_file(path)
            for path in sorted(package.rglob("*.py"))
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: version(name) for name in ["mlx", "mujoco", "numpy", "gymnasium"]},
        "device": str(mx.default_device()),
        "menagerie_revision": MENAGERIE_SHA,
        "model": str(model_path),
        "dynamics": (
            f"h <- 0.5 h + 0.5 tanh(I + 0.8 W h), {config.neural_steps} neural steps per "
            "20 Hz control step, state kept across steps and cleared at episode reset"
        ),
        "trainable": "encoder obs->ascending current, decoder pooled outputs->action",
        "rng_plan": (
            "mx.random.seed(model seed); numpy Generator(training seed); "
            "shuffle seed+17000; Gym episode seeds from SPLITS"
        ),
        "reproducibility_scope": "seed replay on the recorded Apple GPU, not cross-platform",
    }


def run_whole_brain_experiment(
    pack_root: Path, model_path: Path, output: Path, config: WholeBrainConfig
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose a new output: {output}")
    started = time.monotonic()
    try:
        return _run(pack_root, model_path, output, config)
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
    pack_root: Path, model_path: Path, output: Path, config: WholeBrainConfig
) -> dict[str, Any]:
    verified_scene = verify_arm(model_path.resolve().parent.parent)
    if verified_scene.resolve() != model_path.resolve():
        raise ValueError("Model must be the pinned, unmodified Menagerie Panda scene.xml")
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    interface = annotation_interface(pack)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + config.max_seconds
    interface.save(output / "interface.json")
    save_json(output / "interface_report.json", interface_report(pack, interface))
    save_json(output / "config.json", config.model_dump())
    save_json(output / "provenance.json", _provenance(pack, interface, config, model_path))
    task = Task(config, model_path)
    try:
        test_seeds = task.seeds("test", config.test_episodes)
        data = {
            "train": task.collect(task.seeds("train", config.train_episodes), output / "train.npz"),
            "validation": task.collect(
                task.seeds("validation", config.val_episodes), output / "validation.npz"
            ),
        }
        save_json(
            output / "splits.json",
            {
                "train": task.seeds("train", config.train_episodes),
                "validation": task.seeds("validation", config.val_episodes),
                "test": test_seeds,
                "protocol": "identical seeds to the 256-node subgraph protocol for this task",
            },
        )
        results: dict[str, Any] = {
            "status": "running",
            "task": config.task,
            "claim": "B1a whole-connectome rate controller; no topology advantage assumed",
            "teacher_train_success": float(data["train"]["teacher_success"].mean()),
            "teacher": task.baseline("teacher", test_seeds),
            "zero": task.baseline("zero", test_seeds),
            "models": [],
        }
        save_json(output / "results.json", results)
        if results["teacher"]["success_rate"] < 0.9:
            raise RuntimeError("Teacher success below 90%; model comparison is invalid")

        measured = RateDynamics(pack, interface)
        brain_budget = BrainPolicy(
            "connectome", measured, obs_dim=task.obs_dim, action_dim=task.action_dim
        ).trainable_parameter_count()
        for seed in config.seeds:
            jobs: list[tuple[str, int, dict[str, RateDynamics]]] = []
            for policy_kind in config.policies:
                if policy_kind != "shuffled":
                    jobs.append((policy_kind, 0, {"connectome": measured}))
                    continue
                for replicate in config.shuffle_replicates:
                    shuffled = shuffle_pack(pack, shuffle_seed(seed, replicate))
                    shuffled_interface = NeuralInterface.bind(
                        shuffled,
                        interface.input_body_ids,
                        interface.output_body_ids,
                        label=interface.label,
                    )
                    save_json(
                        output / f"{run_name('shuffled', seed, replicate)}.json",
                        {
                            "fingerprint": shuffled.fingerprint(),
                            **{k: v for k, v in shuffled.manifest.items() if k != "sources"},
                            "interface_fingerprint": shuffled_interface.fingerprint,
                            "shuffle_seed": shuffle_seed(seed, replicate),
                        },
                    )
                    jobs.append(
                        (
                            policy_kind,
                            replicate,
                            {"shuffled": RateDynamics(shuffled, shuffled_interface)},
                        )
                    )
            for kind, replicate, dynamics in jobs:
                print(f"{config.task} {kind} seed={seed} replicate={replicate}", flush=True)
                run = output / run_name(kind, seed, replicate)
                run.mkdir()
                policy = _build(kind, seed, task, config, dynamics, brain_budget)
                curves, phases, dagger_sets = _train(
                    policy, task, config, data, seed, deadline, run
                )
                save_json(run / "learning.json", curves)
                policy.save(run / "policy.safetensors")
                item: dict[str, Any] = {
                    "kind": kind,
                    "seed": seed,
                    "shuffle_replicate": replicate if kind == "shuffled" else None,
                    "trainable_parameters": policy.trainable_parameter_count(),
                    "state_size": policy.state_size,
                    "training_phases": phases,
                    "training_seconds": float(sum(p["training_seconds"] for p in phases)),
                    "dagger_seed_sets": dagger_sets,
                    "clean": task.evaluate(MlxController(policy), test_seeds),
                }
                if config.task == "reach":
                    item["noisy"] = task.evaluate(
                        MlxController(policy), test_seeds, noise_std=config.noise_std_m
                    )
                if isinstance(policy, BrainPolicy) and kind == "connectome":
                    ablations = {
                        "edges_off": RateDynamics(pack, interface, edges=False),
                        "direct_only": RateDynamics(
                            pack, interface, weights=direct_only_weights(pack, interface)
                        ),
                    }
                    for name, ablated in ablations.items():
                        clone = policy.with_dynamics(name, ablated)
                        item[name] = task.evaluate(MlxController(clone), test_seeds)
                    item["state_reset_every_step"] = task.evaluate(
                        ResetEveryStep(MlxController(policy)), test_seeds
                    )
                save_json(run / "evaluation.json", item)
                results["models"].append(item)
                results["evidence"] = evidence(results["models"])
                save_json(output / "results.json", results)
                extra = {
                    key: _summary(item[key])
                    for key in ("edges_off", "direct_only", "state_reset_every_step")
                    if key in item
                }
                print(f"  clean={_summary(item['clean'])} {extra}", flush=True)
        results.update(status="complete", elapsed_seconds=time.monotonic() - started)
        save_json(output / "results.json", results)
        return results
    finally:
        task.close()
