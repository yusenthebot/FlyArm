"""Batched closed-loop episodes on the manipulation benchmark, for any controller.

``run_episodes`` runs whole episodes, each until success or its own horizon, in one or several
BatchedManipulation environments in lockstep, so that a controller that is expensive per call
(the connectome) is called once per control step for every episode of every split. It serves
demonstration collection (the teacher acts), DAgger (the learner acts, the teacher labels every
visited state, and with probability beta the teacher's action is executed instead) and
evaluation (task success, subgoals completed and motion quality per split).

Episode seeds. A plan draws ``per_template`` episodes of every template of a split, with seed
``split.seed_start + offset + TEMPLATE_STRIDE * t + i`` for the split's template t and episode
i, so adding episodes never changes existing ones and every block below is disjoint:

- train split, offset 0: teacher demonstrations;
- train split, offset 100,000: validation (imitation loss, phase and checkpoint selection);
- train split, offset 200,000 + 10,000 r: DAgger round r;
- train split, 1,000,000 onward: PPO training episodes (flyarm.rl.ppo.TRAIN_SEED);
- every held-out split, offset 0: test episodes (their seed blocks start at 3,000,000).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from flyarm.manipulation.env import DEFAULT_ASSET_ROOT, BatchedManipulation
from flyarm.manipulation.sim import OBS_DIM, RewardConfig
from flyarm.manipulation.splits import SPLITS
from flyarm.manipulation.teacher import ManipulationTeacher

TEMPLATE_STRIDE = 1000
DEMONSTRATION_OFFSET = 0
VALIDATION_OFFSET = 100_000
DAGGER_OFFSET = 200_000
DAGGER_ROUND_STRIDE = 10_000
TEST_OFFSET = 0
SATURATED = 0.95  # |a| above this counts as a saturated command
ACTION_DIM = 5


@dataclass(frozen=True)
class EpisodePlan:
    """Which episodes one environment runs: a split, a seed and a template per row."""

    split: str
    seeds: tuple[int, ...]
    templates: tuple[str, ...]
    # Subgoal resets (flyarm.manipulation.curriculum): entries of ``bank`` to start from, each
    # episode ending after ``budget`` subgoals; empty for true starts.
    starts: tuple[int, ...] = ()
    budget: int = 0
    budgets: tuple[int, ...] = ()  # per-episode budgets; empty uses ``budget`` for all
    bank: Any = field(default=None, compare=False, repr=False)
    label: str = ""  # the summary's name ("" names it by the split)

    def __len__(self) -> int:
        return len(self.seeds)


def plan(
    split: str,
    per_template: int,
    offset: int = 0,
    templates: Sequence[str] | None = None,
) -> EpisodePlan:
    """``per_template`` episodes of every template of ``split`` (or of ``templates``)."""
    chosen = SPLITS[split]
    names = tuple(templates) if templates is not None else chosen.templates
    if not 1 <= per_template <= TEMPLATE_STRIDE:
        raise ValueError(f"per_template must be in [1, {TEMPLATE_STRIDE}]")
    unknown = set(names) - set(chosen.templates)
    if unknown:
        raise ValueError(f"templates {sorted(unknown)} are not in split {split}")
    seeds, rows = [], []
    for name in names:
        t = chosen.templates.index(name)
        for i in range(per_template):
            seeds.append(chosen.seed_start + offset + TEMPLATE_STRIDE * t + i)
            rows.append(name)
    return EpisodePlan(split, tuple(seeds), tuple(rows))


def make_env(
    model_path: Path,
    episodes: EpisodePlan,
    *,
    asset_root: Path = DEFAULT_ASSET_ROOT,
    reward: RewardConfig | None = None,
    cue: bool = True,
    velocities: bool = True,
    phase_cue: bool = False,
) -> BatchedManipulation:
    return BatchedManipulation(
        model_path,
        len(episodes),
        split=episodes.split,
        asset_root=asset_root,
        reward=reward,
        cue=cue,
        velocities=velocities,
        phase_cue=phase_cue,
    )


class Actor(Protocol):
    """A batched deterministic controller: observations [N, OBS_DIM] to actions [N, 5]."""

    def act(self, obs: np.ndarray) -> np.ndarray: ...


@dataclass
class EpisodeLog:
    """Per-episode outcome of one environment's plan, and the recorded steps if asked for."""

    plan: EpisodePlan
    success: np.ndarray  # [N] every subgoal done
    steps: np.ndarray  # [N] control steps until success or the horizon
    subgoals: np.ndarray  # [N] most subgoals ever done in task order
    subgoals_total: np.ndarray
    stray_steps: np.ndarray
    abs_action: np.ndarray  # [N] sums over the episode's steps
    saturated: np.ndarray
    abs_change: np.ndarray
    disturbance: np.ndarray  # [N] final displacement of what the task does not involve
    teacher_steps: np.ndarray  # [N] steps on which the teacher's action was executed
    obs: np.ndarray | None = None  # [N, T, OBS_DIM]
    labels: np.ndarray | None = None  # [N, T, 5] teacher actions
    mask: np.ndarray | None = None  # [N, T]
    skill: np.ndarray | None = None  # [N, T] skill of the current subgoal

    def data(self) -> dict[str, np.ndarray]:
        """The recorded steps as a training set (obs, actions, mask, skill)."""
        if self.obs is None or self.labels is None or self.mask is None or self.skill is None:
            raise ValueError("these episodes were run without record=True")
        return {"obs": self.obs, "actions": self.labels, "mask": self.mask, "skill": self.skill}


def run_episodes(
    envs: Sequence[BatchedManipulation],
    plans: Sequence[EpisodePlan],
    actor: Actor | None,
    *,
    record: bool = False,
    beta: float = 0.0,
    generator: np.random.Generator | None = None,
) -> list[EpisodeLog]:
    """Run every plan's episodes to completion, all environments in lockstep.

    ``actor`` None: the scripted teacher acts. Otherwise the actor acts on the concatenated
    observations of all environments. With ``record`` every visited state is stored with the
    teacher's label for it, which is DAgger's query; ``beta`` executes the label instead of
    the actor's action with that probability per step and episode.
    """
    if len(envs) != len(plans):
        raise ValueError("one plan per environment")
    if beta > 0 and generator is None:
        raise ValueError("beta needs a generator")
    for env, episodes in zip(envs, plans, strict=True):
        if env.num_envs != len(episodes):
            raise ValueError("each environment needs exactly one row per planned episode")
        if env.split.name != episodes.split:
            raise ValueError(f"environment is split {env.split.name}, plan is {episodes.split}")
    obs = [
        p.bank.reset(
            env,
            env.rows,
            np.array(p.starts),
            np.array(p.budgets) if p.budgets else np.full(len(p), p.budget),
        )
        if p.starts
        else env.reset(seeds=np.array(p.seeds), templates=list(p.templates))
        for env, p in zip(envs, plans, strict=True)
    ]
    teachers = []
    for env in envs:
        teacher = ManipulationTeacher(env) if actor is None or record or beta > 0 else None
        if teacher is not None:
            teacher.reset()
        teachers.append(teacher)
    logs = [_empty_log(env, p, record) for env, p in zip(envs, plans, strict=True)]
    active = [np.ones(env.num_envs, dtype=bool) for env in envs]
    skills = [env.current_skill() for env in envs]
    previous = [np.zeros((env.num_envs, ACTION_DIM)) for env in envs]
    bounds = np.cumsum([0] + [env.num_envs for env in envs])
    longest = max(int(env.horizons.max()) for env in envs)
    for t in range(longest):
        if not any(rows.any() for rows in active):
            break
        chosen = None if actor is None else actor.act(np.concatenate(obs).astype(np.float32))
        for k, env in enumerate(envs):
            if not active[k].any():
                continue
            log, live, teacher = logs[k], active[k], teachers[k]
            labels = teacher.act().astype(np.float64) if teacher is not None else None
            if chosen is None:
                assert labels is not None
                action = labels
            else:
                action = np.clip(np.asarray(chosen[bounds[k] : bounds[k + 1]], np.float64), -1, 1)
            if beta > 0 and labels is not None and generator is not None:
                use_teacher = generator.random(env.num_envs) < beta
                action = np.where(use_teacher[:, None], labels, action)
                log.teacher_steps += live & use_teacher
            elif actor is None:
                log.teacher_steps += live
            if record:
                assert log.obs is not None and log.labels is not None and labels is not None
                assert log.mask is not None and log.skill is not None
                log.obs[live, t] = obs[k][live]
                log.labels[live, t] = labels[live]
                log.mask[live, t] = 1.0
                log.skill[live, t] = skills[k][live]
            result = env.step(action, auto_reset=False)
            log.steps += live
            log.subgoals = np.where(live, np.maximum(log.subgoals, result.high_water), log.subgoals)
            log.stray_steps += live & result.stray_contact
            log.abs_action += live * np.abs(action).mean(1)
            log.saturated += live * (np.abs(action) > SATURATED).mean(1)
            log.abs_change += live * np.abs(action - previous[k]).mean(1)
            previous[k] = action
            log.success |= live & result.success
            finished = live & (result.success | result.truncated)
            if finished.any():
                log.disturbance[finished] = -env.disturbance()[finished]
            active[k] = live & ~finished
            obs[k] = result.obs
            skills[k] = result.current_skill
    for log in logs:
        if record:
            _trim(log)
    return logs


def _empty_log(env: BatchedManipulation, episodes: EpisodePlan, record: bool) -> EpisodeLog:
    n = env.num_envs
    horizon = int(env.horizons.max())
    log = EpisodeLog(
        plan=episodes,
        success=np.zeros(n, dtype=bool),
        steps=np.zeros(n, dtype=np.int64),
        subgoals=np.zeros(n, dtype=np.int64),
        subgoals_total=env.sub_count.copy(),
        stray_steps=np.zeros(n, dtype=np.int64),
        abs_action=np.zeros(n),
        saturated=np.zeros(n),
        abs_change=np.zeros(n),
        disturbance=np.zeros(n),
        teacher_steps=np.zeros(n, dtype=np.int64),
    )
    if record:
        log.obs = np.zeros((n, horizon, OBS_DIM), np.float32)
        log.labels = np.zeros((n, horizon, ACTION_DIM), np.float32)
        log.mask = np.zeros((n, horizon), np.float32)
        log.skill = np.zeros((n, horizon), np.int64)
    return log


def _trim(log: EpisodeLog) -> None:
    """Cut the recorded arrays at the longest episode."""
    assert log.mask is not None
    length = int(log.steps.max(initial=0))
    log.obs = None if log.obs is None else log.obs[:, :length]
    log.labels = None if log.labels is None else log.labels[:, :length]
    log.mask = log.mask[:, :length]
    log.skill = None if log.skill is None else log.skill[:, :length]


def selection_score(success_rate: float, subgoal_fraction: float, episodes: int) -> float:
    """Success rate first, the mean fraction of subgoals done only to break ties.

    A difference of one success is 1 / episodes, more than the largest possible tie-break
    difference 1 / (episodes + 1), so the order is lexicographic.
    """
    return success_rate + subgoal_fraction / (episodes + 1)


def summarize(log: EpisodeLog) -> dict[str, Any]:
    """Task success, subgoals completed and motion quality of one split's episodes."""
    steps = np.maximum(log.steps, 1)
    fraction = log.subgoals / np.maximum(log.subgoals_total, 1)
    names = log.plan.templates
    per_template = {}
    for name in dict.fromkeys(names):
        rows = np.array(names) == name
        per_template[name] = {
            "episodes": int(rows.sum()),
            "success_rate": float(log.success[rows].mean()),
            "subgoal_fraction": float(fraction[rows].mean()),
            "mean_subgoals": float(log.subgoals[rows].mean()),
            "subgoals": int(log.subgoals_total[rows][0]),
        }
    successful = log.steps[log.success]
    success_rate = float(log.success.mean())
    subgoal_fraction = float(fraction.mean())
    return {
        "split": log.plan.label or log.plan.split,
        "episodes": len(names),
        "seeds": [int(seed) for seed in log.plan.seeds],
        "successes": int(log.success.sum()),
        "success_rate": success_rate,
        "subgoal_fraction": subgoal_fraction,
        "mean_subgoals": float(log.subgoals.mean()),
        "selection_score": selection_score(success_rate, subgoal_fraction, len(names)),
        "mean_steps_to_success": float(successful.mean()) if successful.size else None,
        "per_template": per_template,
        "stray_contact_fraction": float((log.stray_steps / steps).mean()),
        "mean_abs_action": float((log.abs_action / steps).mean()),
        "saturated_fraction": float((log.saturated / steps).mean()),
        "mean_abs_action_change": float((log.abs_change / steps).mean()),
        "disturbance": float(log.disturbance.mean()),
        "teacher_step_fraction": float(log.teacher_steps.sum() / steps.sum()),
    }


def skill_weights(data: dict[str, np.ndarray], max_weight: float = 5.0) -> np.ndarray:
    """Per-step weights that give every skill the same total weight, mean 1 over valid steps.

    A skill's steps share an equal part of the total; weights are then clipped at
    ``max_weight`` times the mean (a rare skill's few steps must not dominate a batch) and
    renormalized to mean 1.
    """
    valid = data["mask"] > 0
    skills = data["skill"][valid]
    weights = np.zeros(data["mask"].shape, np.float32)
    if not skills.size:
        return weights
    present, counts = np.unique(skills, return_counts=True)
    share = skills.size / (len(present) * counts)
    per_step = share[np.searchsorted(present, skills)]
    per_step = np.minimum(per_step, max_weight)
    weights[valid] = per_step / per_step.mean()
    return weights


def concatenate(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack recorded sets of different lengths along episodes, padding with masked steps."""
    length = max(part["obs"].shape[1] for part in parts)

    def pad(value: np.ndarray) -> np.ndarray:
        extra = length - value.shape[1]
        return np.pad(value, [(0, 0), (0, extra)] + [(0, 0)] * (value.ndim - 2))

    return {
        key: np.concatenate([pad(part[key]) for part in parts])
        for key in ("obs", "actions", "mask", "skill")
    }


__all__ = [
    "DAGGER_OFFSET",
    "DAGGER_ROUND_STRIDE",
    "EpisodeLog",
    "EpisodePlan",
    "TEST_OFFSET",
    "VALIDATION_OFFSET",
    "concatenate",
    "make_env",
    "plan",
    "run_episodes",
    "skill_weights",
    "summarize",
]
