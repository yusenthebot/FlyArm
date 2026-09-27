"""Subgoal resets and a skill curriculum for PPO on the manipulation benchmark.

Modelled on what solved the kitchen (research log E53, resets to demonstration states): the
scripted teacher runs episodes of the train split, and at the start of every subgoal k of its
template the full state is stored (``SubgoalBank``). A training episode can then start from such
a state with the first k subgoals counted done (they pay no bonus), end after at most m more
subgoals and last 300 steps per subgoal it has to do (``ManipulationSim.reset_to_subgoal``).
Only states are stored, never actions. The cue, the ordered bonus, the shaping target and the
stray-contact allowances all follow from the scene's subgoal flags with the preset included, so
they refer to subgoal k from the first step.

``CurriculumManipulation`` is the batched training environment: every reset draws a true start
with the current stage's share, otherwise a recorded subgoal start with skills balanced and m
drawn from the stage's range. Evaluation never uses it; per-skill evaluation (single-subgoal
episodes from a separate validation bank) is reported next to the headline per-split success
from true starts.

A stored state is replayed on the environment that recorded it (the train split's objects and
furniture ranges): the seed and template rebuild the episode and a fingerprint of the episode
guards against any drift in how episodes are drawn.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flyarm.manipulation import tasks as tk
from flyarm.manipulation.env import BatchedManipulation
from flyarm.manipulation.rollout import TEMPLATE_STRIDE, EpisodePlan, plan
from flyarm.manipulation.teacher import ManipulationTeacher

BANK_OFFSET = 400_000  # train-split seeds of the training bank (DAgger ends below 300,000)
VALIDATION_BANK_OFFSET = 500_000  # the per-skill evaluation bank


def episode_key(episode: tk.Episode) -> str:
    return hashlib.sha1(repr(episode).encode()).hexdigest()[:16]


@dataclass
class SubgoalBank:
    """States at the start of subgoals of recorded teacher episodes, one entry per start."""

    seeds: np.ndarray  # [K] episode seed
    templates: np.ndarray  # [K] template name
    subgoal: np.ndarray  # [K] index k of the subgoal the state starts
    skill: np.ndarray  # [K] skill of that subgoal
    remaining: np.ndarray  # [K] subgoals from k to the template's end
    keys: np.ndarray  # [K] fingerprint of the episode
    states: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.seeds)

    def select(self, rows: np.ndarray) -> SubgoalBank:
        rows = np.asarray(rows, dtype=np.int64)
        return SubgoalBank(
            self.seeds[rows],
            self.templates[rows],
            self.subgoal[rows],
            self.skill[rows],
            self.remaining[rows],
            self.keys[rows],
            {name: value[rows] for name, value in self.states.items()},
        )

    def save(self, path: Path) -> None:
        arrays = {
            "seeds": self.seeds,
            "templates": self.templates.astype(str),
            "subgoal": self.subgoal,
            "skill": self.skill,
            "remaining": self.remaining,
            "keys": self.keys.astype(str),
            **{f"state_{name}": value for name, value in self.states.items()},
        }
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> SubgoalBank:
        with np.load(path) as stored:
            states = {
                name[len("state_") :]: stored[name]
                for name in stored.files
                if name.startswith("state_")
            }
            return cls(
                stored["seeds"],
                stored["templates"],
                stored["subgoal"],
                stored["skill"],
                stored["remaining"],
                stored["keys"],
                states,
            )

    def reset(
        self, env: BatchedManipulation, ids: np.ndarray, picks: np.ndarray, budget: np.ndarray
    ) -> np.ndarray:
        """Reset ``ids`` of ``env`` from entries ``picks``, each with its subgoal budget."""
        chosen = self.select(picks)
        obs = env.reset_to_subgoal(
            ids, chosen.seeds, list(chosen.templates), chosen.states, chosen.subgoal, budget
        )
        for row, key in zip(ids, chosen.keys, strict=True):
            episode = env.episodes[row]
            if episode is None or episode_key(episode) != str(key):
                raise ValueError(
                    "a stored subgoal state no longer rebuilds its episode; re-record the bank"
                )
        return obs

    def skill_counts(self) -> dict[str, int]:
        return {tk.SKILLS[k]: int((self.skill == k).sum()) for k in np.unique(self.skill)}


def record_bank(env: BatchedManipulation, episodes: EpisodePlan) -> SubgoalBank:
    """Run the teacher on ``episodes`` and store the state at the start of every subgoal.

    A subgoal starts at the reset (k = 0) and whenever the number of subgoals done in order
    grows; only states the teacher actually reached are stored, successful episode or not.
    """
    if env.num_envs != len(episodes) or env.split.name != episodes.split:
        raise ValueError("the environment must match the plan (split and one row per episode)")
    env.reset(seeds=np.array(episodes.seeds), templates=list(episodes.templates))
    teacher = ManipulationTeacher(env)
    teacher.reset()
    entries: list[tuple[int, int]] = []  # (row, k)
    parts: list[dict[str, np.ndarray]] = []
    reached = np.full(env.num_envs, -1)
    active = np.ones(env.num_envs, dtype=bool)
    leading = np.zeros(env.num_envs, dtype=np.int64)
    keys = np.array([episode_key(e) for e in env.episodes if e is not None])

    def store(rows: np.ndarray) -> None:
        if rows.size:
            parts.append(env.snapshot(rows))
            entries.extend((int(row), int(leading[row])) for row in rows)
            reached[rows] = leading[rows]

    store(env.rows)
    for _ in range(int(env.horizons.max())):
        if not active.any():
            break
        result = env.step(teacher.act().astype(np.float64), auto_reset=False)
        leading = result.subgoals_done
        new = active & (leading > reached) & (leading < env.sub_count)
        store(np.flatnonzero(new))
        active &= ~(result.success | result.truncated)
    rows = np.array([row for row, _ in entries])
    subgoal = np.array([k for _, k in entries])
    states = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    return SubgoalBank(
        seeds=np.array(episodes.seeds)[rows],
        templates=np.array(episodes.templates)[rows],
        subgoal=subgoal,
        skill=env.sub_kind[rows, subgoal],
        remaining=env.sub_count[rows] - subgoal,
        keys=keys[rows],
        states=states,
    )


def skill_weights(bank: SubgoalBank) -> np.ndarray:
    """Sampling weights that make every skill equally likely."""
    _, inverse, counts = np.unique(bank.skill, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def subgoal_group(template: str, subgoal: int) -> str:
    """The skill of a subgoal, with the receptacle for placements ("place:shelf")."""
    skill, _, target = tk.TEMPLATES[template].steps[subgoal]
    if skill == "place":
        return f"place:{'drawer' if target == 'D' else target}"
    return skill


def group_weights(bank: SubgoalBank, weights: dict[str, float]) -> np.ndarray:
    """Sampling weights over bank entries: each group (skill, or placement receptacle) gets
    its configured weight (a skill's weight when its group is not listed, 1 otherwise),
    shared evenly among the group's entries. Empty ``weights`` gives skill_weights."""
    if not weights:
        return skill_weights(bank)
    groups = np.array(
        [subgoal_group(str(t), int(k)) for t, k in zip(bank.templates, bank.subgoal, strict=True)]
    )
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    base = np.array([weights.get(name, weights.get(name.split(":")[0], 1.0)) for name in names])
    if np.any(base < 0) or not np.any(base > 0):
        raise ValueError("group weights must be non-negative with at least one positive")
    entry = base[inverse] / counts[inverse]
    return entry / entry.sum()


@dataclass(frozen=True)
class Stage:
    """One curriculum stage: true-start share and the range of subgoals per reset episode."""

    name: str
    iterations: int
    min_subgoals: int
    max_subgoals: int
    true_start_share: float


class CurriculumManipulation(BatchedManipulation):
    """The training environment: resets draw true starts or recorded subgoal starts.

    Explicit resets (seeds, templates or episodes given) are plain resets, so the same object
    can still be evaluated; a reset without them (the first one, and every automatic reset
    after an episode ends) follows the current stage.
    """

    def __init__(
        self,
        *args: Any,
        bank: SubgoalBank,
        stages: Sequence[Stage],
        curriculum_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.bank = bank
        self.stages = tuple(stages)
        self.stage_index = 0
        self._weights = skill_weights(bank)
        self._draws = np.random.default_rng([curriculum_seed, 91])
        self.subgoal_starts = 0
        self.true_starts = 0
        self._started = False
        super().__init__(*args, **kwargs)

    @property
    def stage(self) -> Stage:
        return self.stages[self.stage_index]

    def set_stage(self, index: int) -> None:
        if not 0 <= index < len(self.stages):
            raise ValueError(f"no curriculum stage {index}")
        self.stage_index = index

    def reset(
        self,
        ids: np.ndarray | None = None,
        seeds: np.ndarray | None = None,
        templates: Sequence[str] | None = None,
        episodes: Sequence[tk.Episode] | None = None,
    ) -> np.ndarray:
        if seeds is not None or templates is not None or episodes is not None:
            return super().reset(ids, seeds, templates, episodes)
        if not self._started:
            # Every row gets a plain episode first, so no row is ever observed uninitialized
            # while the others are drawn (an unset lid frame divides by zero in the cue).
            super().reset()
            self._started = True
        ids = self.rows.copy() if ids is None else np.asarray(ids, dtype=np.int64)
        stage = self.stage
        true = self._draws.random(len(ids)) < stage.true_start_share
        obs = None
        if true.any():
            obs = super().reset(ids[true])
            self.true_starts += int(true.sum())
        resets = ids[~true]
        if resets.size:
            picks = self._draws.choice(len(self.bank), size=resets.size, p=self._weights)
            budget = self._draws.integers(stage.min_subgoals, stage.max_subgoals + 1, resets.size)
            obs = self.bank.reset(self, resets, picks, budget)
            self.subgoal_starts += int(resets.size)
        assert obs is not None
        return obs


def skill_plans(bank: SubgoalBank, per_skill: int) -> list[EpisodePlan]:
    """Single-subgoal evaluation episodes: the first ``per_skill`` bank entries of every skill."""
    plans = []
    for skill in np.unique(bank.skill):
        rows = np.flatnonzero(bank.skill == skill)[:per_skill]
        plans.append(
            EpisodePlan(
                split="train",
                seeds=tuple(int(seed) for seed in bank.seeds[rows]),
                templates=tuple(str(name) for name in bank.templates[rows]),
                starts=tuple(int(row) for row in rows),
                budget=1,
                bank=bank,
                label=f"skill:{tk.SKILLS[int(skill)]}",
            )
        )
    return plans


def bank_plan(per_template: int, offset: int) -> EpisodePlan:
    if per_template > TEMPLATE_STRIDE:
        raise ValueError("too many bank episodes per template")
    return plan("train", per_template, offset)


__all__ = [
    "BANK_OFFSET",
    "CurriculumManipulation",
    "Stage",
    "SubgoalBank",
    "VALIDATION_BANK_OFFSET",
    "bank_plan",
    "record_bank",
    "skill_plans",
]
