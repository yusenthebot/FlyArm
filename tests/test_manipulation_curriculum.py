from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from flyarm.config import CurriculumStage, ManipulationPPOConfig
from flyarm.manipulation import rollout
from flyarm.manipulation import tasks as tk

MODEL = os.environ.get("FLYARM_MODEL")
OBJECTS = Path(os.environ.get("FLYARM_OBJECTS", "assets/objects"))
needs_env = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file() or not OBJECTS.is_dir(),
    reason="set FLYARM_MODEL to the Panda scene.xml and fetch the grasp objects",
)
CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_curriculum_config_checks_its_budget_and_stages() -> None:
    stages = [
        {"name": "single", "iterations": 2, "true_start_share": 0.2},
        {"name": "full", "iterations": 3, "max_subgoals": 8, "true_start_share": 1.0},
    ]
    config = ManipulationPPOConfig(iterations=5, curriculum=stages)
    assert [stage.name for stage in config.curriculum] == ["single", "full"]
    with pytest.raises(ValidationError, match="add up"):
        ManipulationPPOConfig(iterations=6, curriculum=stages)
    with pytest.raises(ValidationError):
        CurriculumStage(name="x", iterations=1, true_start_share=0.0)  # true starts in every stage
    with pytest.raises(ValidationError, match="at least"):
        CurriculumStage(name="x", iterations=1, min_subgoals=3, max_subgoals=2, true_start_share=1)
    shipped = ManipulationPPOConfig.model_validate_json(
        (CONFIGS / "ppo-manipulation-curriculum.json").read_text()
    )
    assert len(shipped.curriculum) == 3 and shipped.curriculum[0].max_subgoals == 1


@pytest.fixture(scope="module")
def recorded():
    """A teacher episode of tidy with its per-step snapshots, observations and actions."""
    if not MODEL:
        pytest.skip("needs FLYARM_MODEL")
    from flyarm.manipulation.env import BatchedManipulation
    from flyarm.manipulation.teacher import ManipulationTeacher

    episodes = rollout.plan("train", 1, 400_000, templates=["tidy"])
    env = BatchedManipulation(Path(MODEL), 1, asset_root=OBJECTS)
    obs = env.reset(seeds=np.array(episodes.seeds), templates=list(episodes.templates))
    teacher = ManipulationTeacher(env)
    teacher.reset()
    trace = []
    for _ in range(700):
        leading = int(env.leading(env.subgoal_done(env.effects(env.scene_state())))[0])
        action = teacher.act().astype(np.float64)
        trace.append((obs[0].copy(), env.snapshot(np.array([0])), leading, action))
        obs = env.step(action, auto_reset=False).obs
    return episodes, trace


@needs_env
def test_a_subgoal_reset_restores_the_state_exactly(recorded) -> None:
    from flyarm.manipulation.env import BatchedManipulation

    episodes, trace = recorded
    start = next(t for t, (_, _, leading, _) in enumerate(trace) if leading == 2)
    obs, state, leading, _ = trace[start]
    env = BatchedManipulation(Path(MODEL), 1, asset_root=OBJECTS)
    restored = env.reset_to_subgoal(
        np.array([0]), np.array(episodes.seeds), ["tidy"], state, np.array([leading]), np.array([8])
    )
    np.testing.assert_array_equal(restored[0], obs)
    worst = 0.0
    for t in range(start, start + 60):  # the recorded actions replayed from the restored state
        result = env.step(trace[t][3], auto_reset=False)
        worst = max(worst, float(np.abs(result.obs[0] - trace[t + 1][0]).max()))
    assert worst == 0.0, worst  # bit for bit


@needs_env
def test_preset_subgoals_are_done_unpaid_and_the_cue_names_subgoal_k(recorded) -> None:
    from flyarm.manipulation.env import BatchedManipulation
    from flyarm.manipulation.sim import cue_slices

    episodes, trace = recorded
    start = next(t for t, (_, _, leading, _) in enumerate(trace) if leading == 2)
    _, state, _, _ = trace[start]
    env = BatchedManipulation(Path(MODEL), 1, asset_root=OBJECTS)
    # Start from the state at subgoal 2 but declare 3 done: the reset index decides, not the scene.
    obs = env.reset_to_subgoal(
        np.array([0]), np.array(episodes.seeds), ["tidy"], state, np.array([3]), np.array([1])
    )
    cue = cue_slices()
    assert obs[0, cue["skill"]].argmax() + 1 == tk.OPEN_DOOR  # tidy's subgoal 3
    assert env.high_water[0] == 3 and env.goal_count[0] == 4
    assert env.horizons[0] == tk.HORIZON_PER_SUBGOAL
    result = env.step(np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]), auto_reset=False)
    assert result.reward[0] < 1.0  # no bonus for the preset subgoals
    assert result.subgoals_done[0] == 3 and not result.success[0]


@needs_env
def test_a_single_subgoal_episode_ends_when_that_subgoal_is_done(recorded) -> None:
    from flyarm.manipulation.env import BatchedManipulation

    episodes, trace = recorded
    start = next(t for t, (_, _, leading, _) in enumerate(trace) if leading == 1)
    _, state, leading, _ = trace[start]
    env = BatchedManipulation(Path(MODEL), 1, asset_root=OBJECTS)
    env.reset_to_subgoal(
        np.array([0]), np.array(episodes.seeds), ["tidy"], state, np.array([leading]), np.array([1])
    )
    for t in range(start, len(trace) - 1):
        result = env.step(trace[t][3], auto_reset=False)
        if result.success[0]:
            break
    assert result.success[0] and result.subgoals_done[0] == 2 < env.sub_count[0]
    assert result.reward[0] > 40.0  # the bonus of subgoal 1, the one this episode was about


@needs_env
def test_the_bank_and_the_curriculum_environment() -> None:
    from flyarm.manipulation import curriculum as cu

    episodes = cu.bank_plan(1, cu.BANK_OFFSET)
    env = rollout.make_env(Path(MODEL), episodes, asset_root=OBJECTS)
    bank = cu.record_bank(env, episodes)
    counts = bank.skill_counts()
    assert set(counts) == set(tk.SKILLS[1:])  # every skill has starting states
    assert np.all(bank.subgoal < tk.MAX_SUBGOALS) and np.all(bank.remaining >= 1)
    weights = cu.skill_weights(bank)
    per_skill = {k: weights[bank.skill == k].sum() for k in np.unique(bank.skill)}
    assert np.allclose(list(per_skill.values()), 1 / len(per_skill))
    stages = [cu.Stage("single", 1, 1, 1, 0.25), cu.Stage("full", 1, 8, 8, 1.0)]
    training = cu.CurriculumManipulation(
        Path(MODEL), 16, asset_root=OBJECTS, bank=bank, stages=stages
    )
    training.reset()
    single = training.goal_count - training.preset
    reset_rows = training.preset > 0
    assert np.all(single[reset_rows] == 1) and training.subgoal_starts > 0
    assert np.all(training.horizons[training.subgoal_starts > 0] > 0)
    training.set_stage(1)
    training.reset()
    assert np.all(training.preset == 0) and np.all(training.goal_count == training.sub_count)
    plans = cu.skill_plans(bank, 2)
    assert {p.label for p in plans} == {f"skill:{name}" for name in tk.SKILLS[1:]}
    logs = rollout.run_episodes(
        [rollout.make_env(Path(MODEL), p, asset_root=OBJECTS) for p in plans[:2]],
        plans[:2],
        None,
    )
    for log in logs:  # the teacher finishes single-subgoal episodes from bank states
        assert log.success.mean() >= 0.5 and np.all(log.steps <= tk.HORIZON_PER_SUBGOAL)
