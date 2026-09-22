from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flyarm.benchmarks import _robotics_compat, kitchen
from flyarm.rl.batched_kitchen import (
    APPROACH_WEIGHT,
    COMPLETE_TASKS,
    MAX_STEP_REWARD,
    PROGRESS_WEIGHT,
    BatchedKitchen,
    KitchenVariant,
    minimum_completion_bonus,
    rollout,
    shaped_reward,
)

BONUS_THRESH = 0.3  # gymnasium_robotics.envs.franka_kitchen.kitchen_env.BONUS_THRESH
# mjbatch and MjData sum contact forces in a different order, so the two integrations differ
# in the last bits and the contact-rich kitchen amplifies that: the gap stays below 5e-7 rad
# for 90 steps of flailing random actions and passes 1e-5 around step 100.
EQUIVALENCE_STEPS = 90
EQUIVALENCE_TOLERANCE = 1e-5


def _dataset_available() -> bool:
    return (Path.home() / ".minari" / "datasets" / "D4RL" / "kitchen" / "complete-v2").is_dir()


def _single_env(**kwargs: float):
    _robotics_compat.install()
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv

    return KitchenEnv(
        tasks_to_complete=list(COMPLETE_TASKS),
        remove_task_when_completed=False,
        terminate_on_tasks_completed=False,
        robot_noise_ratio=0.0,
        object_noise_ratio=0.0,
        **kwargs,
    )


def test_batched_env_reproduces_the_single_gymnasium_env() -> None:
    """The benchmark's own control law, stepped by mjbatch, on one shared action sequence."""
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (EQUIVALENCE_STEPS, 9))
    env = _single_env()
    observation, _ = env.reset(seed=0)
    reference = [np.asarray(observation["observation"], dtype=np.float64)]
    joints = []
    for action in actions:
        observation, _, _, _, info = env.step(action)
        reference.append(np.asarray(observation["observation"], dtype=np.float64))
        joints.append(env.data.qpos.copy())
    assert not info["episode_task_completions"]  # random actions complete nothing

    batched = BatchedKitchen(2)
    first = batched.reset(seeds=np.array([0, 1]))
    for row in range(2):
        assert np.allclose(first[row], reference[0][kitchen.POLICY_FEATURES], atol=1e-6)
    worst_joint = worst_obs = 0.0
    for step, action in enumerate(actions):
        result = batched.step(np.repeat(action[None], 2, axis=0), auto_reset=False)
        expected = reference[step + 1][kitchen.POLICY_FEATURES]
        for row in range(2):
            worst_obs = max(worst_obs, float(np.abs(result.obs[row] - expected).max()))
            worst_joint = max(worst_joint, float(np.abs(batched.qpos[row] - joints[step]).max()))
        assert not result.completed.any()
    assert worst_joint < EQUIVALENCE_TOLERANCE, worst_joint
    assert worst_obs < EQUIVALENCE_TOLERANCE, worst_obs


def test_completion_uses_the_benchmark_goals_threshold_and_task_order() -> None:
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import (
        OBS_ELEMENT_GOALS,
        OBS_ELEMENT_INDICES,
    )

    env = BatchedKitchen(3)
    env.reset(seeds=np.array([0, 1, 2]))
    assert env.tasks == ("microwave", "kettle", "light switch", "slide cabinet")
    assert not env.completed.any() and (env.target() == 0).all()

    # Row 0 lands the microwave exactly on its goal, row 1 just inside the threshold and row 2
    # just outside it; the third element of the split is put at its goal in every row.
    indices, goal = OBS_ELEMENT_INDICES["microwave"], OBS_ELEMENT_GOALS["microwave"]
    for row, error in enumerate((0.0, 0.9 * BONUS_THRESH, 1.1 * BONUS_THRESH)):
        env.qpos[row, indices] = goal + error
    light = OBS_ELEMENT_INDICES["light switch"]
    env.qpos[:, light] = OBS_ELEMENT_GOALS["light switch"]
    result = env.step(np.zeros((3, 9)), auto_reset=False)
    assert result.completed[:, 0].tolist() == [True, True, False]
    assert result.completed[:, 2].tolist() == [True, True, True]  # order is the split's
    assert result.tasks_completed.tolist() == [2, 2, 1]
    # The shaping target is the first task of the split that is still open.
    assert result.target.tolist() == [0, 0, 0]
    assert env.target().tolist() == [1, 1, 0]
    # A completed element that drifts back out still counts for the episode, as in KitchenEnv
    # with remove_task_when_completed False.
    env.qpos[:, light] = OBS_ELEMENT_GOALS["light switch"] + 10.0
    again = env.step(np.zeros((3, 9)), auto_reset=False)
    assert again.completed[:, 2].all()


def test_completion_bonus_must_outweigh_stalling_on_the_per_step_maximum() -> None:
    gamma = 0.99
    assert minimum_completion_bonus(gamma) == pytest.approx(MAX_STEP_REWARD / (1 - gamma))
    # The per-step terms are bounded even when the gripper sits on a finished element.
    at_best = shaped_reward(
        np.zeros(1), np.full(1, BONUS_THRESH), np.ones(1), np.zeros(1, dtype=int), np.ones(1, bool)
    )
    assert at_best.max() <= MAX_STEP_REWARD + 1e-6
    completed = shaped_reward(
        np.zeros(1), np.full(1, BONUS_THRESH), np.ones(1), np.ones(1, dtype=int), np.ones(1, bool)
    )
    assert completed.min() > at_best.max() / (1 - gamma)
    with pytest.raises(ValueError, match="research log E34"):
        BatchedKitchen(1, completion_bonus=minimum_completion_bonus(gamma))


def test_shaped_reward_pays_for_approach_and_for_moving_the_element() -> None:
    handle = np.array([1.0, 0.02])  # gripper-to-handle distance, far and touching
    start = np.full(2, 1.0)  # element joint distance to its goal at the episode start
    zeros, active = np.zeros(2, dtype=int), np.ones(2, bool)
    approach_only = shaped_reward(handle, start, start, zeros, active)
    assert approach_only[1] > approach_only[0]  # closer to the handle pays more
    assert approach_only.max() < APPROACH_WEIGHT + 1e-6  # no progress yet

    # Progress runs from 0 at the initial joint distance to 1 at the completion threshold; the
    # approach distance is held fixed, so the difference is the progress term alone.
    at_threshold = shaped_reward(handle, np.full(2, BONUS_THRESH), start, zeros, active)
    halfway = shaped_reward(handle, np.full(2, 0.5 * (1.0 + BONUS_THRESH)), start, zeros, active)
    assert at_threshold - approach_only == pytest.approx([PROGRESS_WEIGHT] * 2, abs=1e-5)
    assert halfway - approach_only == pytest.approx([0.5 * PROGRESS_WEIGHT] * 2, abs=1e-5)
    # Moving the element away from its goal is worth zero, never negative.
    backwards = shaped_reward(handle, np.full(2, 2.0), start, zeros, active)
    assert backwards == pytest.approx(approach_only, abs=1e-5)
    # With every task completed there is no target left and only bonuses are paid.
    assert shaped_reward(np.zeros(1), np.ones(1), np.ones(1), np.zeros(1, int), np.zeros(1, bool))[
        0
    ] == pytest.approx(0.0)


def test_auto_reset_draws_fresh_seeds_and_clears_completions() -> None:
    env = BatchedKitchen(3, horizon=20, first_seed=100)
    env.reset()
    assert env.episode_seed.tolist() == [100, 101, 102]
    for _ in range(20):
        result = env.step(np.zeros((3, 9)))
    assert result.truncated.all() and not result.terminated.any()
    assert env.episode_seed.tolist() == [103, 104, 105]
    assert (env.steps == 0).all() and not env.completed.any()
    assert result.obs.shape == (3, kitchen.FEATURE_DIM)


def test_privileged_observation_adds_velocities_and_task_state() -> None:
    env = BatchedKitchen(2)
    env.reset(seeds=np.array([0, 1]))
    obs, privileged = env.observation(), env.observation(privileged=True)
    assert obs.shape == (2, kitchen.FEATURE_DIM)
    assert privileged.shape == (2, env.privileged_dim)
    assert np.array_equal(privileged[:, : kitchen.FEATURE_DIM], obs)
    assert np.allclose(privileged[:, -4:], 0.0)  # no task completed yet
    assert np.allclose(privileged[:, 30:39], 0.0)  # the robot starts at rest


def test_variant_perturbs_the_start_and_leaves_the_nominal_episode_alone() -> None:
    nominal = BatchedKitchen(2)
    nominal.reset(seeds=np.array([7, 8]))
    perturbed = BatchedKitchen(2, variant=KitchenVariant(initial_joint_offset=0.3))
    perturbed.reset(seeds=np.array([7, 8]))
    delta = perturbed.qpos[:, :7] - nominal.qpos[:, :7]
    assert np.abs(delta).max() <= 0.3 and np.abs(delta).max() > 0.05
    assert np.allclose(perturbed.qpos[:, 9:], nominal.qpos[:, 9:])  # objects untouched
    again = BatchedKitchen(2, variant=KitchenVariant(initial_joint_offset=0.3))
    again.reset(seeds=np.array([7, 8]))
    assert np.array_equal(again.qpos[:, :7], perturbed.qpos[:, :7])  # seeded

    heavy = BatchedKitchen(2, variant=KitchenVariant(kettle_mass_scale=(2.0, 5.0)))
    heavy.reset(seeds=np.array([7, 8]))
    assert np.all((heavy.mass_scale >= 2.0) & (heavy.mass_scale <= 5.0))
    body = heavy._kettle_body
    assert np.allclose(heavy.body_mass[:, body], nominal.model.body_mass[body] * heavy.mass_scale)
    # The mass draw has its own stream, so the nominal episode still starts identically.
    assert np.allclose(heavy.qpos[:, :9], nominal.qpos[:, :9])

    with pytest.raises(ValueError, match="initial_joint_offset"):
        KitchenVariant(initial_joint_offset=1.0)
    with pytest.raises(ValueError, match="kettle_mass_scale"):
        KitchenVariant(kettle_mass_scale=(2.0, 1.0))


def test_a_random_policy_completes_nothing_and_earns_little() -> None:
    generator = np.random.default_rng(0)
    env = BatchedKitchen(4, horizon=120)
    report = rollout(env, lambda _: generator.uniform(-1.0, 1.0, (4, 9)), 120, seeds=np.arange(4))
    assert report["mean_tasks"] == 0.0 and report["normalized_score"] == 0.0
    assert 0.0 < report["mean_step_reward"] < 0.2 * MAX_STEP_REWARD


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_the_scripted_teacher_completes_the_four_tasks_in_the_batched_env() -> None:
    """The tracker scores 100 on the benchmark, so it must score 100 here (control semantics)."""
    from flyarm.benchmarks.kitchen_expert import DemonstrationTracker

    data = kitchen.load("complete", download=False)
    tracker = DemonstrationTracker.from_data(data)
    rows = 2
    full = np.zeros((rows, kitchen.OBS_DIM), dtype=np.float32)

    def teacher(observation: np.ndarray) -> np.ndarray:
        full[:, kitchen.POLICY_FEATURES] = observation
        return np.stack([tracker.label(full[row]) for row in range(rows)])

    env = BatchedKitchen(rows)
    report = rollout(env, teacher, 280, seeds=np.arange(rows))
    assert report["mean_tasks"] == 4.0 and report["normalized_score"] == 100.0
    assert all(rate == 1.0 for rate in report["per_task_success"].values())
    # Four completions alone are worth four bonuses; anything less means a miscounted task.
    assert report["mean_episode_reward"] > 4 * env.completion_bonus
