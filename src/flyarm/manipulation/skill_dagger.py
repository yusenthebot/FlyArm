"""Skill-level DAgger at scale: imitation from subgoal resets with many on-policy episodes.

Diagnosis it answers (research log): the imitation runs fit the teacher well and still fail
closed loop, the signature of covariate shift, with far too few on-policy episodes for
1,100 to 2,600-step tasks. The teacher is Markov and cheap and the batched environment is fast,
so this pipeline scales DAgger by an order of magnitude and starts it on short horizons.

- Round 0 records ``teacher_episodes`` teacher episodes; round r >= 1 rolls out the current
  learner on ``episodes_per_round`` episodes, executing the teacher's action instead with
  probability ``betas[r - 1]`` (0 once the list ends). Every visited state is labelled with the
  teacher's action and added to the aggregate, which lives on the host with episodes stored end
  to end (flyarm.whole_brain.training.StepData).
- Episodes follow the round's stage: a share of true starts (full template), the rest start from
  recorded teacher states at the start of a subgoal (flyarm.manipulation.curriculum's bank, with
  its fingerprint and bit-exact restore), skills balanced, and run a budget of subgoals.
- Each round trains the warm-started controller for a fixed number of window-sampled updates
  (fresh Adam each round) with the same burn-in, learning rates and per-skill weights as the
  imitation pipeline.
- Per round it saves the round's new data, the weights and metrics (training loss, single-
  subgoal success per skill from the validation bank, full-task success and subgoal fraction from
  true starts on the train split's validation seeds); a run stopped at any point resumes from
  the last completed round. The best round on validation is evaluated on every split.

The run directory has the layout of an imitation run (config.json is the controller's
ManipulationImitationConfig, interface.json, train.npz, {kind}-{seed}/policy.safetensors), so
curriculum PPO can warm-start from it unchanged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from flyarm.config import DaggerStage, SkillDaggerConfig
from flyarm.io import save_json
from flyarm.manipulation import curriculum as cu
from flyarm.manipulation import rollout
from flyarm.manipulation.detour import sample_detours
from flyarm.manipulation.env import DEFAULT_ASSET_ROOT
from flyarm.manipulation.imitation import (
    PolicyActor,
    Workbench,
    brain_budget,
    build_policy,
    dynamics_for,
)
from flyarm.manipulation.splits import SPLITS
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.interface import annotation_interface, interface_report
from flyarm.whole_brain.policy import BrainPolicy, SequencePolicy
from flyarm.whole_brain.training import Budget, StepData, train_windows

DAGGER_OFFSET = 600_000  # train-split seeds of the rollouts' true starts, round r at + 10,000 r
ROUND_STRIDE = 10_000
CALIBRATION_EPISODES = 64


# ----------------------------------------------------------------------------------- schedule
def stage_for_round(config: SkillDaggerConfig, index: int) -> tuple[int, DaggerStage]:
    """The stage that round ``index`` (0-based, round 0 included) belongs to."""
    bounds = np.cumsum([stage.rounds for stage in config.stages])
    position = int(np.searchsorted(bounds, index, side="right"))
    if position >= len(config.stages):
        raise ValueError(f"round {index} is past the last stage")
    return position, config.stages[position]


def beta_for_round(config: SkillDaggerConfig, index: int) -> float:
    """1 in round 0 (the teacher acts), then the configured betas, then 0."""
    if index == 0:
        return 1.0
    return config.betas[index - 1] if index - 1 < len(config.betas) else 0.0


def round_plans(
    stage: DaggerStage,
    index: int,
    episodes: int,
    bank: cu.SubgoalBank,
    generator: np.random.Generator,
    weights: dict[str, float] | None = None,
    detour: tuple[float, tuple[int, int]] = (0.0, (20, 60)),
) -> list[rollout.EpisodePlan]:
    """The round's episodes: true starts (templates in turn) and skill-balanced subgoal starts,
    a ``detour[0]`` share of the latter first moved to a random hand pose (detour.py)."""
    true = min(episodes, max(1, int(round(stage.true_start_share * episodes))))
    templates = SPLITS["train"].templates
    names = [templates[i % len(templates)] for i in range(true)]
    if true > rollout.TEMPLATE_STRIDE * len(templates):
        raise ValueError("too many true starts for one round's seed block")
    offset = DAGGER_OFFSET + ROUND_STRIDE * index
    seeds = [
        SPLITS["train"].seed_start
        + offset
        + rollout.TEMPLATE_STRIDE * templates.index(name)
        + i // len(templates)
        for i, name in enumerate(names)
    ]
    plans = [rollout.EpisodePlan("train", tuple(seeds), tuple(names), label="dagger-true-starts")]
    resets = episodes - true
    if resets:
        picks = generator.choice(len(bank), size=resets, p=cu.group_weights(bank, weights or {}))
        budgets = generator.integers(stage.min_subgoals, stage.max_subgoals + 1, resets)
        plans.append(
            rollout.EpisodePlan(
                "train",
                tuple(int(seed) for seed in bank.seeds[picks]),
                tuple(str(name) for name in bank.templates[picks]),
                starts=tuple(int(pick) for pick in picks),
                budgets=tuple(int(b) for b in budgets),
                bank=bank,
                label="dagger-resets",
                # Drawn last and only when on, so runs without detours keep their random stream.
                detour=sample_detours(resets, detour[0], detour[1], generator)
                if detour[0] > 0
                else None,
            )
        )
    return plans


def round_updates(config: SkillDaggerConfig, index: int, aggregate_steps: int) -> int:
    """Updates of round ``index``: the fixed count, or ``epochs_per_round`` passes over the
    aggregate when that is more, capped at ``max_updates_per_round``."""
    fixed = config.first_round_updates if index == 0 else config.updates_per_round
    per_update = config.model.window_batch * config.model.bptt_steps
    passes = int(np.ceil(config.epochs_per_round * aggregate_steps / per_update))
    return min(max(fixed, passes), max(fixed, config.max_updates_per_round))


# ----------------------------------------------------------------------------------- data
def ragged(logs: list[rollout.EpisodeLog]) -> dict[str, np.ndarray]:
    """The recorded steps of ``logs`` stored end to end: obs, labels, skill, episode lengths."""
    obs, actions, skill, lengths = [], [], [], []
    for log in logs:
        data = log.data()
        valid = data["mask"] > 0
        for row in range(valid.shape[0]):
            length = int(valid[row].sum())
            if not length:
                continue
            if not valid[row, :length].all():
                raise ValueError("recorded steps must be contiguous from the episode's start")
            obs.append(data["obs"][row, :length])
            actions.append(data["actions"][row, :length])
            skill.append(data["skill"][row, :length])
            lengths.append(length)
    return {
        "obs": np.concatenate(obs).astype(np.float32),
        "actions": np.concatenate(actions).astype(np.float32),
        "skill": np.concatenate(skill).astype(np.int64),
        "lengths": np.array(lengths, dtype=np.int64),
    }


def aggregate(parts: list[dict[str, np.ndarray]], max_weight: float) -> StepData:
    """All rounds' steps, with per-step weights that give every skill the same total weight."""
    lengths = np.concatenate([part["lengths"] for part in parts])
    skill = np.concatenate([part["skill"] for part in parts])
    return StepData(
        obs=np.concatenate([part["obs"] for part in parts]),
        actions=np.concatenate([part["actions"] for part in parts]),
        weights=flat_skill_weights(skill, max_weight),
        starts=np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64),
        lengths=lengths,
    )


def flat_skill_weights(skill: np.ndarray, max_weight: float) -> np.ndarray:
    """flyarm.manipulation.rollout.skill_weights on steps stored end to end."""
    present, inverse, counts = np.unique(skill, return_inverse=True, return_counts=True)
    share = skill.size / (len(present) * counts)
    weights = np.minimum(share[inverse], max_weight)
    return (weights / weights.mean()).astype(np.float32)


def padded(part: dict[str, np.ndarray], limit: int) -> dict[str, np.ndarray]:
    """Up to ``limit`` episodes of a ragged part as padded [E, T] arrays (with a mask)."""
    lengths = part["lengths"]
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    rows = np.linspace(0, len(lengths) - 1, min(limit, len(lengths))).round().astype(int)
    rows = np.unique(rows)
    horizon = int(lengths[rows].max())
    out: dict[str, np.ndarray] = {
        "obs": np.zeros((len(rows), horizon, part["obs"].shape[1]), np.float32),
        "actions": np.zeros((len(rows), horizon, part["actions"].shape[1]), np.float32),
        "mask": np.zeros((len(rows), horizon), np.float32),
        "skill": np.zeros((len(rows), horizon), np.int64),
    }
    for k, row in enumerate(rows):
        span = slice(starts[row], starts[row] + lengths[row])
        out["obs"][k, : lengths[row]] = part["obs"][span]
        out["actions"][k, : lengths[row]] = part["actions"][span]
        out["skill"][k, : lengths[row]] = part["skill"][span]
        out["mask"][k, : lengths[row]] = 1.0
    return out


# ----------------------------------------------------------------------------------- rounds
def collect_round(
    bench: Workbench,
    policy: SequencePolicy | None,
    plans: list[rollout.EpisodePlan],
    beta: float,
    generator: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Roll out ``policy`` (the teacher when None) on ``plans``, every visited state labelled."""
    started = time.monotonic()
    logs = bench.run(
        plans,
        lambda n: None if policy is None else PolicyActor(policy, n),
        record=True,
        beta=beta if policy is not None else 0.0,
        generator=generator,
    )
    part = ragged(logs)
    seconds = time.monotonic() - started
    stats = {
        log.plan.label: {
            key: rollout.summarize(log)[key]
            for key in ("episodes", "success_rate", "subgoal_fraction", "teacher_step_fraction")
        }
        for log in logs
    }
    return part, {
        "rollouts": stats,
        "labelled_steps": int(part["lengths"].sum()),
        "rollout_seconds": round(seconds, 1),
        "labelled_steps_per_second": round(float(part["lengths"].sum()) / seconds, 1),
    }


def evaluate(
    bench: Workbench, policy: SequencePolicy, plans: list[rollout.EpisodePlan]
) -> dict[str, dict[str, Any]]:
    logs = bench.run(plans, lambda n: PolicyActor(policy, n))
    return {log.plan.label or log.plan.split: rollout.summarize(log) for log in logs}


def selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    """Validation from true starts first, the mean single-subgoal skill success second."""
    return (float(metrics["validation"]["selection_score"]), float(metrics["mean_skill_success"]))


# ----------------------------------------------------------------------------------- run
def run_skill_dagger(
    pack_root: Path,
    model_path: Path,
    output: Path,
    config: SkillDaggerConfig,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    *,
    resume: bool = False,
    stop_after_round: int | None = None,
) -> dict[str, Any]:
    """Run (or resume) skill-level DAgger for every policy and seed of ``config.model``."""
    started = time.monotonic()
    deadline = started + config.max_seconds
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    interface = annotation_interface(pack)
    if output.exists() and not resume:
        raise FileExistsError(f"Run directory already exists; resume or choose a new one: {output}")
    if resume:
        saved = SkillDaggerConfig.model_validate_json((output / "skill-dagger.json").read_text())
        if saved != config:
            raise ValueError("resume needs the configuration the run started with")
    else:
        output.mkdir(parents=True)
        interface.save(output / "interface.json")
        save_json(output / "interface_report.json", interface_report(pack, interface))
        save_json(output / "skill-dagger.json", config.model_dump())
        save_json(output / "config.json", config.model.model_dump())
    model = config.model
    bench = Workbench(model_path, asset_root, model.cue, model.velocities, model.phase_cue)
    bank, validation_bank = _banks(config, bench, model_path, asset_root, output)
    budget = brain_budget(pack, interface, model)
    results: dict[str, Any] = {
        "status": "running",
        "benchmark": "articulated multi-step manipulation (flyarm.manipulation)",
        "method": "skill-level DAgger from subgoal resets (flyarm.manipulation.skill_dagger)",
        "banks": {"train": bank.skill_counts(), "validation": validation_bank.skill_counts()},
        "models": [],
    }
    for seed in model.seeds:
        for kind in model.policies:
            item = _run_one(
                kind,
                seed,
                config,
                pack,
                interface,
                budget,
                bench,
                bank,
                validation_bank,
                output,
                deadline,
                stop_after_round,
            )
            if item is None:
                results["status"] = "stopped"
                save_json(output / "results.json", results)
                return results
            results["models"].append(item)
            save_json(output / "results.json", results)
    results.update(status="complete", elapsed_seconds=time.monotonic() - started)
    save_json(output / "results.json", results)
    return results


def _banks(
    config: SkillDaggerConfig,
    bench: Workbench,
    model_path: Path,
    asset_root: Path,
    output: Path,
) -> tuple[cu.SubgoalBank, cu.SubgoalBank]:
    banks = []
    for name, per_template, offset in (
        ("bank", config.bank_episodes_per_template, cu.BANK_OFFSET),
        (
            "validation-bank",
            config.validation_bank_episodes_per_template,
            cu.VALIDATION_BANK_OFFSET,
        ),
    ):
        path = output / f"{name}.npz"
        if path.is_file():
            banks.append(cu.SubgoalBank.load(path))
            continue
        episodes = cu.bank_plan(per_template, offset)
        env = rollout.make_env(
            model_path,
            episodes,
            asset_root=asset_root,
            cue=config.model.cue,
            velocities=config.model.velocities,
            phase_cue=config.model.phase_cue,
        )
        bank = cu.record_bank(env, episodes)
        bank.save(path)
        print(f"{name}: {len(bank)} subgoal starts {bank.skill_counts()}", flush=True)
        banks.append(bank)
    return banks[0], banks[1]


def _completed_rounds(run: Path) -> int:
    count = 0
    while (run / f"round-{count:02d}" / "metrics.json").is_file():
        count += 1
    return count


def _run_one(
    kind: str,
    seed: int,
    config: SkillDaggerConfig,
    pack: ConnectomePack,
    interface: Any,
    budget: int,
    bench: Workbench,
    bank: cu.SubgoalBank,
    validation_bank: cu.SubgoalBank,
    output: Path,
    deadline: float,
    stop_after_round: int | None,
) -> dict[str, Any] | None:
    run = output / f"{kind}-{seed}"
    if (run / "evaluation.json").is_file():  # finished before a resume
        return json.loads((run / "evaluation.json").read_text())
    dynamics, shuffle_record = dynamics_for(kind, seed, pack, interface)
    if shuffle_record is not None:
        save_json(output / f"shuffled-{seed}.json", shuffle_record)
    policy = build_policy(kind, config.model, seed, dynamics, budget)
    history = train_rounds(
        policy, run, config, bench, bank, validation_bank, seed, deadline, stop_after_round
    )
    if history is None:
        return None
    return finish(policy, run, kind, seed, config, bench, validation_bank, history)


def train_rounds(
    policy: SequencePolicy,
    run: Path,
    config: SkillDaggerConfig,
    bench: Workbench,
    bank: cu.SubgoalBank,
    validation_bank: cu.SubgoalBank,
    seed: int,
    deadline: float,
    stop_after_round: int | None = None,
) -> list[dict[str, Any]] | None:
    """Every round of one controller, resumed after the last completed round in ``run``.

    Returns the per-round metrics, or None when stopped after ``stop_after_round``. Round 0
    also writes the run's train.npz (the teacher's round-0 episodes) next to ``run``.
    """
    model = config.model
    brain = isinstance(policy, BrainPolicy)
    run.mkdir(parents=True, exist_ok=True)
    label = run.name
    done = _completed_rounds(run)
    parts = [dict(np.load(run / f"round-{r:02d}" / "data.npz")) for r in range(done)]
    history = [
        json.loads((run / f"round-{r:02d}" / "metrics.json").read_text()) for r in range(done)
    ]
    if done:
        policy.load(run / f"round-{done - 1:02d}" / "policy.safetensors")
        print(f"{label}: resuming after round {done - 1}", flush=True)
    eval_plans = [
        rollout.plan("train", config.val_episodes_per_template, rollout.VALIDATION_OFFSET),
        *cu.skill_plans(validation_bank, config.skill_eval_episodes),
    ]
    for index in range(done, config.rounds):
        if time.monotonic() >= deadline:
            raise TimeoutError("skill-DAgger budget exhausted; completed rounds are kept")
        stage_index, stage = stage_for_round(config, index)
        beta = beta_for_round(config, index)
        generator = np.random.default_rng([seed, index, 13])
        episodes = config.teacher_episodes if index == 0 else config.episodes_per_round
        plans = round_plans(
            stage,
            index,
            episodes,
            bank,
            generator,
            config.reset_group_weights,
            (config.detour_share, config.detour_steps),
        )
        part, collected = collect_round(
            bench, None if index == 0 else policy, plans, beta, generator
        )
        directory = run / f"round-{index:02d}"
        directory.mkdir(exist_ok=True)
        # numpy's stub types savez_compressed's **kwds against its own allow_pickle: bool
        # keyword too, so a dict[str, ndarray] never satisfies it without this.
        np.savez_compressed(directory / "data.npz", **cast(dict[str, Any], part))
        if index == 0:
            _start(policy, part, run.parent)
        parts.append(part)
        data = aggregate(parts, model.max_skill_weight)
        trained = time.monotonic()
        losses = train_windows(
            policy,
            data,
            Budget(
                epochs=1,
                decoder_warmup_epochs=0,
                batch_size=model.batch_size,
                bptt_steps=model.bptt_steps,
                learning_rate=model.learning_rate,
                deadline=deadline,
                loss=model.loss,
                input_learning_rate=(
                    model.learning_rate * model.encoder_learning_rate_scale if brain else None
                ),
                window_batch=model.window_batch,
                burn_in=model.burn_in,
            ),
            round_updates(config, index, int(data.lengths.sum())),
            np.random.default_rng([seed, index, 17]),
            warmup_updates=config.warmup_updates if brain and index == 0 else 0,
        )
        train_seconds = time.monotonic() - trained
        scored = evaluate(bench, policy, eval_plans)
        skills = {
            name.split(":", 1)[1]: summary["success_rate"]
            for name, summary in scored.items()
            if name.startswith("skill:")
        }
        tail = losses[-max(1, len(losses) // 10) :]
        metrics = {
            "round": index,
            "stage": stage.name,
            "stage_index": stage_index,
            "beta": beta,
            **collected,
            "aggregate_steps": int(data.lengths.sum()),
            "aggregate_episodes": int(len(data.lengths)),
            "updates": len(losses),
            "train_loss_first": float(np.mean(losses[: max(1, len(losses) // 10)])),
            "train_loss_last": float(np.mean(tail)),
            "train_seconds": round(train_seconds, 1),
            "validation": scored["train"],
            "skills": skills,
            "mean_skill_success": float(np.mean(list(skills.values()))),
        }
        (directory / "policy.safetensors").unlink(missing_ok=True)  # a round cut short
        policy.save(directory / "policy.safetensors")
        save_json(directory / "metrics.json", metrics)  # written last: the round is complete
        history.append(metrics)
        print(
            f"{label} round {index} ({stage.name}, beta {beta:.2f}): "
            f"{collected['labelled_steps']} labelled steps "
            f"({collected['labelled_steps_per_second']:.0f}/s), "
            f"aggregate {metrics['aggregate_steps']}, "
            f"L1 {metrics['train_loss_last']:.4f}, validation "
            f"{scored['train']['successes']}/{scored['train']['episodes']} "
            f"subgoals {scored['train']['subgoal_fraction']:.3f}, skills "
            + " ".join(f"{name} {rate:.2f}" for name, rate in skills.items()),
            flush=True,
        )
        if stop_after_round is not None and index >= stop_after_round:
            return None
    return history


def finish(
    policy: SequencePolicy,
    run: Path,
    kind: str,
    seed: int,
    config: SkillDaggerConfig,
    bench: Workbench,
    validation_bank: cu.SubgoalBank,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep the best round on validation and evaluate it on every split and skill."""
    # Ties go to the later round, which has trained on more of the learner's own states.
    best = max(range(len(history)), key=lambda r: (*selection_key(history[r]), r))
    policy.load(run / f"round-{best:02d}" / "policy.safetensors")
    final = run / "policy.safetensors"
    final.unlink(missing_ok=True)
    policy.save(final)
    splits = [
        rollout.plan(split, config.eval_episodes_per_template, rollout.TEST_OFFSET)
        for split in config.eval_splits
    ]
    item = {
        "kind": kind,
        "seed": seed,
        "trainable_parameters": policy.trainable_parameter_count(),
        "selected_round": best,
        "rounds": [
            {
                key: metrics[key]
                for key in (
                    "round",
                    "stage",
                    "beta",
                    "labelled_steps",
                    "aggregate_steps",
                    "train_loss_last",
                    "mean_skill_success",
                    "skills",
                )
            }
            | {
                "validation_success": metrics["validation"]["success_rate"],
                "validation_subgoal_fraction": metrics["validation"]["subgoal_fraction"],
            }
            for metrics in history
        ],
        "evaluation": evaluate(
            bench, policy, [*splits, *cu.skill_plans(validation_bank, config.skill_eval_episodes)]
        ),
    }
    save_json(run / "evaluation.json", item)
    return item


def _start(policy: SequencePolicy, part: dict[str, np.ndarray], output: Path) -> None:
    """Round 0: the frozen normalizations from the teacher's data, and train.npz for DAPG."""
    sample = padded(part, CALIBRATION_EPISODES)
    steps = part["obs"]
    policy.set_normalization(steps.mean(0), np.maximum(steps.std(0), 0.05))
    if isinstance(policy, BrainPolicy):
        policy.scale_encoder(steps[:: max(1, len(steps) // 20_000)])
        policy.calibrate_readout(sample["obs"], sample["mask"], unit_norm=True)
    demonstrations = output / "train.npz"
    if not demonstrations.is_file():  # the teacher's round-0 episodes, for curriculum PPO's DAPG
        data = cast(dict[str, Any], padded(part, len(part["lengths"])))
        np.savez_compressed(demonstrations, **data)


__all__ = [
    "aggregate",
    "beta_for_round",
    "flat_skill_weights",
    "ragged",
    "round_plans",
    "run_skill_dagger",
    "stage_for_round",
]
