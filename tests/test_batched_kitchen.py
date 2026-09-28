from __future__ import annotations

from pathlib import Path
from typing import Any

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


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_the_reference_term_raises_the_stalling_floor_and_the_check_fires() -> None:
    """The tracking term joins the per-step budget, so the E34 invariant moves with it."""
    from flyarm.rl.batched_kitchen import TRACKING_SIGMA

    assert minimum_completion_bonus(0.99, MAX_STEP_REWARD + 0.5) == pytest.approx(150.0)
    # A weight of 0.5 puts the floor at 150, which 200 clears; 1.5 puts it at 250, which 200
    # does not, and the constructor refuses it.
    env = BatchedKitchen(1, tracking_weight=0.5, tracking_form="gaussian", completion_bonus=200.0)
    assert env.max_step_reward == pytest.approx(1.5)
    with pytest.raises(ValueError, match="research log E34"):
        BatchedKitchen(1, tracking_weight=1.5, tracking_form="gaussian", completion_bonus=200.0)
    with pytest.raises(ValueError, match="tracking_weight"):
        BatchedKitchen(1, tracking_weight=-0.1)
    with pytest.raises(ValueError, match="tracking_sigma"):
        BatchedKitchen(1, tracking_weight=0.5, tracking_sigma=0.0)
    with pytest.raises(ValueError, match="tracking_form"):
        BatchedKitchen(1, tracking_weight=0.5, tracking_form="cosine")

    # Off by default: no reference is loaded and the term contributes nothing.
    plain = BatchedKitchen(1)
    assert plain.reference is None and plain.max_step_reward == pytest.approx(MAX_STEP_REWARD)
    assert plain.tracking_reward().tolist() == [0.0]

    # The term is exp(-||q - q_ref||^2 / sigma^2) over the 9 robot joints, 1 on the reference.
    env.reset(seeds=np.array([0]))
    reference = env.reference
    assert reference is not None
    env.qpos[0, :9] = reference[0]
    assert env.tracking_reward()[0] == pytest.approx(1.0)
    env.qpos[0, :9] = reference[0] + np.r_[TRACKING_SIGMA, np.zeros(8)]
    assert env.tracking_reward()[0] == pytest.approx(np.exp(-1.0), rel=1e-5)


def test_shaped_reward_adds_the_tracking_term_on_top_of_the_task_terms() -> None:
    handle, start = np.full(2, 1.0), np.full(2, 1.0)
    zeros, active = np.zeros(2, dtype=int), np.ones(2, bool)
    without = shaped_reward(handle, start, start, zeros, active)
    with_term = shaped_reward(handle, start, start, zeros, active, tracking=np.array([1.0, 0.25]))
    assert with_term == pytest.approx(without, abs=1e-6)  # weight 0 changes nothing
    weighted = shaped_reward(
        handle, start, start, zeros, active, tracking=np.array([1.0, 0.25]), tracking_weight=0.5
    )
    assert weighted - without == pytest.approx([0.5, 0.125], abs=1e-5)
    # It is paid even when every task is done, because it is about where the arm is.
    done = shaped_reward(
        handle,
        start,
        start,
        zeros,
        np.zeros(2, bool),
        tracking=np.ones(2),
        tracking_weight=0.5,
    )
    assert done == pytest.approx([0.5, 0.5], abs=1e-5)


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_the_reference_is_states_only_and_the_tracker_beats_a_random_policy_on_it() -> None:
    from flyarm.rl.batched_kitchen import reference_joints

    reference = reference_joints(0)
    assert reference.ndim == 2 and reference.shape[1] == 9 and 100 < len(reference) < 280
    data = kitchen.load("complete", download=False)
    # States only: the reference is exactly the demonstration's joint positions, and the
    # demonstration's actions never enter it.
    steps = int(data.mask[0].sum())
    assert np.allclose(reference, data.obs[0, :steps, :9])

    from flyarm.benchmarks.kitchen_expert import DemonstrationTracker

    tracker = DemonstrationTracker.from_data(data)
    rows = 2
    full = np.zeros((rows, kitchen.OBS_DIM), dtype=np.float32)

    def teacher(observation: np.ndarray) -> np.ndarray:
        full[:, kitchen.POLICY_FEATURES] = observation
        return np.stack([tracker.label(full[row]) for row in range(rows)])

    def measure(controller: Any) -> float:
        env = BatchedKitchen(rows, tracking_weight=0.5, tracking_form="gaussian", horizon=150)
        obs = env.reset(seeds=np.arange(rows))
        total = 0.0
        for _ in range(150):
            result = env.step(np.asarray(controller(obs), dtype=np.float64), auto_reset=False)
            total += float(env.tracking_reward().mean())
            obs = result.obs
            if (result.terminated | result.truncated).all():
                break
        return total / 150

    generator = np.random.default_rng(0)
    tracked = measure(teacher)
    random = measure(lambda _: generator.uniform(-1.0, 1.0, (rows, 9)))
    # From the benchmark's own start the tracker reproduces the reference almost exactly.
    assert tracked > 0.9 and random < 0.1 and tracked > 10 * random


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_the_potential_form_leaves_the_stalling_floor_untouched() -> None:
    """Potential-based shaping telescopes, so it adds nothing to a sustained trajectory (E41)."""
    env = BatchedKitchen(1, tracking_weight=0.5, completion_bonus=200.0)
    assert env.tracking_form == "potential"
    # The Gaussian form at the same weight would put the floor at 150; this one leaves it at 100,
    # so the completion bonus keeps its full 2.0x headroom.
    assert env.max_step_reward == pytest.approx(MAX_STEP_REWARD)
    assert minimum_completion_bonus(0.99, env.max_step_reward) == pytest.approx(100.0)
    # A weight that the Gaussian form could not afford is fine here.
    assert BatchedKitchen(1, tracking_weight=5.0).max_step_reward == pytest.approx(MAX_STEP_REWARD)


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_the_potential_term_telescopes_over_an_episode_whatever_path_is_taken() -> None:
    """The default form sums undiscounted to phi(s_T) - phi(s_0), so only the endpoints matter."""
    length = 40
    generator = np.random.default_rng(0)
    for actions in (generator.uniform(-1.0, 1.0, (length, 9)), np.zeros((length, 9))):
        env = BatchedKitchen(1, tracking_weight=1.0, horizon=length + 5, approach_slope=1.0)
        env.reset(seeds=np.array([0]))
        start = float(env.potential()[0])
        paid = 0.0
        for action in actions:
            paid += float(env.step(action[None], auto_reset=False).tracking[0])
        assert paid == pytest.approx(float(env.potential()[0]) - start, abs=1e-9)


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_the_discounted_potential_form_telescopes_only_under_the_discount() -> None:
    """Kept for the record: its undiscounted residual is what made it the wrong choice (E41)."""
    gamma, length = 0.99, 40
    actions = np.random.default_rng(1).uniform(-1.0, 1.0, (length, 9))
    env = BatchedKitchen(
        1,
        tracking_weight=1.0,
        tracking_form="potential_discounted",
        gamma=gamma,
        horizon=length + 5,
        approach_slope=1.0,
    )
    env.reset(seeds=np.array([0]))
    start = float(env.potential()[0])
    discounted = plain = 0.0
    for step, action in enumerate(actions):
        paid = float(env.step(action[None], auto_reset=False).tracking[0])
        discounted += gamma**step * paid
        plain += paid
    end = float(env.potential()[0])
    assert discounted == pytest.approx(gamma**length * end - start, abs=1e-9)
    # Undiscounted it does not telescope: the gap is the (1 - gamma) * distance residual, which
    # is positive and therefore pays for sitting far from the reference.
    assert plain > end - start


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_standing_still_pays_zero_and_closing_the_distance_pays_the_metres_closed() -> None:
    env = BatchedKitchen(3, tracking_weight=1.0, approach_slope=1.0)
    env.reset(seeds=np.array([0, 1, 2]))
    reference = env.reference
    assert reference is not None
    target = reference[min(1, len(reference) - 1)]
    env.previous_potential[:] = -1.0  # all three were 1.0 rad from the reference
    env.steps[:] = 1
    env.qpos[0, :9] = target  # closed the whole 1.0 rad
    env.qpos[1, :9] = target + 2.0  # opened to ||2 * ones(9)|| = 6.0 rad
    env.qpos[2, :9] = target + 1.0 / 3.0  # ||ones(9) / 3|| = 1.0 rad, unchanged
    term = env.tracking_reward()
    assert term[0] == pytest.approx(1.0, abs=1e-6)
    assert term[1] == pytest.approx(-5.0, abs=1e-6)
    assert term[2] == pytest.approx(0.0, abs=1e-6)  # standing still is worth exactly nothing


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_the_potential_reference_is_still_states_only() -> None:
    from flyarm.rl.batched_kitchen import reference_joints

    data = kitchen.load("complete", download=False)
    for episode in (0, 3):
        reference = reference_joints(episode)
        steps = int(data.mask[episode].sum())
        assert np.allclose(reference, data.obs[episode, :steps, :9])
        # The demonstration's actions are 9-D too, so check they are not what was loaded.
        assert not np.allclose(reference, data.actions[episode, :steps])
    env = BatchedKitchen(1, tracking_weight=0.5, reference_episode=3)
    assert env.reference is not None and np.allclose(env.reference, reference_joints(3))


def test_the_target_rule_defaults_to_the_split_order_and_the_alternatives_pick_by_score() -> None:
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import (
        OBS_ELEMENT_GOALS,
        OBS_ELEMENT_INDICES,
    )

    from flyarm.rl.batched_kitchen import TARGET_RULES

    env = BatchedKitchen(2)
    assert env.target_rule == "split_order"
    env.reset(seeds=np.array([0, 1]))
    assert env.target().tolist() == [0, 0]  # microwave first, whatever the geometry
    with pytest.raises(ValueError, match="target_rule"):
        BatchedKitchen(1, target_rule="whatever")
    assert "split_order" in TARGET_RULES and "nearest" in TARGET_RULES

    # "nearest" follows the geometry: the handle closest to the gripper wins.
    near = BatchedKitchen(2, target_rule="nearest")
    near.reset(seeds=np.array([0, 1]))
    distances = near.handle_distances()
    assert distances.shape == (2, 4)
    assert near.target().tolist() == distances.argmin(1).tolist()

    # "moved" follows absolute element travel, and falls back to the split's order on a tie.
    moved = BatchedKitchen(2, target_rule="moved")
    moved.reset(seeds=np.array([0, 1]))
    assert np.allclose(moved.element_travel(), 0.0)
    assert moved.target().tolist() == [0, 0]  # nothing has moved yet, so the tie breaks in order
    light = OBS_ELEMENT_INDICES["light switch"]
    moved.qpos[:, light] = OBS_ELEMENT_GOALS["light switch"]  # element 2 travels furthest
    assert moved.target().tolist() == [2, 2]
    assert moved.element_travel()[:, 2].min() > 0.0

    # A completed task is never the target again.
    moved.step(np.zeros((2, 9)), auto_reset=False)
    assert moved.completed[:, 2].all()
    assert (moved.target() != 2).all()


def test_the_progress_rule_is_skewed_by_the_span_and_moved_is_not() -> None:
    """Normalised progress saturates fastest for the smallest span to threshold (E43)."""
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import OBS_ELEMENT_INDICES

    env = BatchedKitchen(1, target_rule="progress")
    env.reset(seeds=np.array([0]))
    spans = env.initial_goal_distance[0] - BONUS_THRESH
    # The slide cabinet starts far closer to its threshold than the microwave does.
    assert spans[3] < 0.15 and spans[0] > 0.4
    # The same absolute travel is far more "progress" on the narrow-span element.
    travel = 0.05
    env.qpos[0, OBS_ELEMENT_INDICES["microwave"]] -= travel
    env.qpos[0, OBS_ELEMENT_INDICES["slide cabinet"]] += travel
    progress = env.element_progress()[0]
    assert progress[3] > 2 * progress[0]
    assert env.target()[0] == 3  # so "progress" chases the narrow element

    absolute = BatchedKitchen(1, target_rule="moved")
    absolute.reset(seeds=np.array([0]))
    absolute.qpos[0, OBS_ELEMENT_INDICES["microwave"]] -= 2 * travel
    absolute.qpos[0, OBS_ELEMENT_INDICES["slide cabinet"]] += travel
    assert absolute.target()[0] == 0  # "moved" follows the larger real motion


def test_the_prefix_curriculum_presets_elements_without_paying_their_bonus() -> None:
    from flyarm.rl.batched_kitchen import assemble_reward, averaged_task_shaping

    plain = BatchedKitchen(2)
    plain.reset(seeds=np.array([0, 1]))
    assert not plain.variant.uses_curriculum and (plain.prefix == 0).all()
    assert not plain.preset.any()

    env = BatchedKitchen(
        16, variant=KitchenVariant(curriculum_prefix=3, curriculum_true_start_share=0.25)
    )
    env.reset(seeds=np.arange(16))
    # Drawn per environment, so one batch mixes true starts with advanced ones.
    assert env.prefix.min() == 0 and env.prefix.max() >= 1 and len(set(env.prefix.tolist())) > 1
    assert (env.prefix <= 3).all()
    for row in range(16):
        k = int(env.prefix[row])
        assert env.preset[row, :k].all() and not env.preset[row, k:].any()
        assert env.completed[row, :k].all()  # marked done, so never the shaping target
        for index in range(k):
            # Placed at its goal joint positions, at rest.
            assert np.allclose(
                env.qpos[row, env._element_indices[index]], env._element_goals[index]
            )
            assert np.allclose(env.qvel[row, env._element_dofs[index]], 0.0)
        if k < 4:
            assert env.target()[row] == k

    # A preset element pays no completion bonus and does not count as earned.
    result = env.step(np.zeros((16, 9)), auto_reset=False)
    assert (result.newly_completed == 0).all()
    assert (result.tasks_completed == 0).all()
    assert np.array_equal(result.prefix, env.prefix)
    # With the curriculum off, earned tasks are exactly the benchmark's count.
    assert (plain.step(np.zeros((2, 9)), auto_reset=False).tasks_completed == 0).all()

    with pytest.raises(ValueError, match="curriculum_prefix"):
        KitchenVariant(curriculum_prefix=-1)
    with pytest.raises(ValueError, match="curriculum_true_start_share"):
        KitchenVariant(curriculum_true_start_share=1.5)

    # The averaged scope keeps the per-step maximum, so the E34 floor is untouched.
    perfect = averaged_task_shaping(np.zeros((1, 4)), np.ones((1, 4)), np.ones((1, 4), bool))
    assert perfect[0] == pytest.approx(MAX_STEP_REWARD, abs=1e-6)
    none_left = averaged_task_shaping(np.zeros((1, 4)), np.ones((1, 4)), np.zeros((1, 4), bool))
    assert none_left[0] == pytest.approx(0.0)
    assert BatchedKitchen(1, shaping_scope="sum").max_step_reward == pytest.approx(MAX_STEP_REWARD)
    with pytest.raises(ValueError, match="shaping_scope"):
        BatchedKitchen(1, shaping_scope="product")
    # assemble_reward is the single place the bonus and the reference term are added.
    assert assemble_reward(np.array([0.5]), np.array([2]), 200.0)[0] == pytest.approx(400.5)


def test_the_potential_task_form_pays_nothing_for_idleness_and_rebases_on_a_completion() -> None:
    """The defect of the level form was that idleness out-earned the expert (E44)."""
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import (
        OBS_ELEMENT_GOALS,
        OBS_ELEMENT_INDICES,
    )

    level = BatchedKitchen(2, approach_slope=1.0)
    level.reset(seeds=np.array([0, 1]))
    assert level.task_shaping_form == "level" and level.task_shaping_weight == 1.0
    # The level form pays every step for standing where it stands.
    assert level.step(np.zeros((2, 9)), auto_reset=False).reward.min() > 0.1

    env = BatchedKitchen(2, approach_slope=1.0, task_shaping_form="potential")
    env.reset(seeds=np.array([0, 1]))
    # Frozen arm and frozen scene: the potential does not change, so the term is exactly 0.
    env.previous_task_potential[:] = env.task_potential(env.target())
    frozen = env.tracking_reward()  # no reference term configured, so this is zero too
    assert np.allclose(frozen, 0.0)
    held = env.task_potential(env.target()) - env.previous_task_potential
    assert np.allclose(held, 0.0)

    # A completion moves the target, and the rebase must absorb that jump rather than charge it.
    env.qpos[:, OBS_ELEMENT_INDICES["microwave"]] = OBS_ELEMENT_GOALS["microwave"]
    result = env.step(np.zeros((2, 9)), auto_reset=False)
    assert (result.newly_completed == 1).all()
    # The reward is the bonus plus a bounded step term, never a large negative spike.
    assert (result.reward > env.completion_bonus - 1.0).all()
    # After the step the potential is rebased onto the new target, so the next idle step is 0.
    assert np.allclose(env.previous_task_potential, env.task_potential(env.target()))
    idle = env.step(np.zeros((2, 9)), auto_reset=False)
    assert np.abs(idle.reward).max() < 0.05


def test_task_shaping_weight_scales_the_terms_and_zero_leaves_the_bonus_alone() -> None:
    off = BatchedKitchen(2, approach_slope=1.0, task_shaping_weight=0.0)
    off.reset(seeds=np.array([0, 1]))
    assert off.max_step_reward == pytest.approx(0.0)
    for _ in range(3):
        assert np.allclose(off.step(np.zeros((2, 9)), auto_reset=False).reward, 0.0)

    half = BatchedKitchen(2, approach_slope=1.0, task_shaping_weight=0.5)
    full = BatchedKitchen(2, approach_slope=1.0)
    for env in (half, full):
        env.reset(seeds=np.array([0, 1]))
    assert half.max_step_reward == pytest.approx(0.5 * MAX_STEP_REWARD)
    assert half.step(np.zeros((2, 9)), auto_reset=False).reward == pytest.approx(
        0.5 * full.step(np.zeros((2, 9)), auto_reset=False).reward, rel=1e-5
    )

    # The potential form telescopes, so it never enters the per-step maximum whatever the weight.
    potential = BatchedKitchen(1, task_shaping_form="potential", task_shaping_weight=50.0)
    assert potential.max_step_reward == pytest.approx(0.0)
    with pytest.raises(ValueError, match="task_shaping_weight"):
        BatchedKitchen(1, task_shaping_weight=-1.0)
    with pytest.raises(ValueError, match="task_shaping_form"):
        BatchedKitchen(1, task_shaping_form="derivative")
    with pytest.raises(ValueError, match="shaping_scope 'target'"):
        BatchedKitchen(1, task_shaping_form="potential", shaping_scope="sum")


def test_split_order_pays_a_task_only_once_the_earlier_ones_are_done() -> None:
    def place(env: BatchedKitchen, index: int) -> None:
        env.qpos[0, env._element_indices[index]] = env._element_goals[index]
        env.qvel[0, env._element_dofs[index]] = 0.0
        env.batch.forward(np.array([0]))

    bonus = 200.0
    for order, kettle_pays in (("any", True), ("split_order", False)):
        env = BatchedKitchen(
            1, completion_order=order, completion_bonus=bonus, task_shaping_weight=0.0
        )
        env.reset(seeds=np.array([0]))
        kettle = env.tasks.index("kettle")
        place(env, kettle)
        first = env.step(np.zeros((1, 9)))
        assert first.newly_completed.tolist() == [1]  # the benchmark counts it either way
        assert (first.reward[0] >= bonus) == kettle_pays
        place(env, 0)  # the microwave, first of the split
        second = env.step(np.zeros((1, 9)))
        paid = 2 if order == "split_order" else 1  # the kettle's deferred bonus arrives now
        assert second.reward[0] == pytest.approx(paid * bonus, abs=1.0)
    with pytest.raises(ValueError, match="completion_order"):
        BatchedKitchen(1, completion_order="whatever")


def test_strict_bonus_share_is_paid_only_close_to_the_goal() -> None:
    env = BatchedKitchen(
        1, strict_bonus_fraction=0.5, completion_bonus=200.0, task_shaping_weight=0.0
    )
    env.reset(seeds=np.array([0]))
    index = env.tasks.index("slide cabinet")
    goal = env._element_goals[index]
    start = env.qpos[0, env._element_indices[index]].copy()
    # Within the benchmark's 0.3 but not the strict 0.1: half the bonus.
    env.qpos[0, env._element_indices[index]] = goal + 0.2 * np.sign(start - goal)
    env.batch.forward(np.array([0]))
    first = env.step(np.zeros((1, 9)))
    assert first.newly_completed.tolist() == [1]
    assert first.reward[0] == pytest.approx(100.0, abs=1.0)
    # Now all the way: the other half.
    env.qpos[0, env._element_indices[index]] = goal
    env.qvel[0, env._element_dofs[index]] = 0.0
    env.batch.forward(np.array([0]))
    second = env.step(np.zeros((1, 9)))
    assert second.reward[0] == pytest.approx(100.0, abs=1.0)
    with pytest.raises(ValueError, match="strict_bonus_fraction"):
        BatchedKitchen(1, strict_bonus_fraction=1.0)


def test_a_stricter_completion_threshold_withholds_the_bonus_until_close() -> None:
    env = BatchedKitchen(
        1, completion_threshold=0.1, completion_bonus=200.0, task_shaping_form="potential"
    )
    env.reset(seeds=np.array([0]))
    index = env.tasks.index("slide cabinet")
    goal = env._element_goals[index]
    start = env.qpos[0, env._element_indices[index]].copy()
    env.qpos[0, env._element_indices[index]] = goal + 0.2 * np.sign(start - goal)
    env.batch.forward(np.array([0]))
    first = env.step(np.zeros((1, 9)))
    assert first.newly_completed.tolist() == [1]  # the benchmark counts it at 0.3
    assert first.reward[0] < 50.0  # but the reward does not pay the bonus yet
    env.qpos[0, env._element_indices[index]] = goal
    env.qvel[0, env._element_dofs[index]] = 0.0
    env.batch.forward(np.array([0]))
    assert env.step(np.zeros((1, 9))).reward[0] > 150.0
    with pytest.raises(ValueError, match="potential"):
        BatchedKitchen(1, completion_threshold=0.1)


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_demonstration_resets_load_demo_states_and_their_finished_tasks() -> None:
    from flyarm.rl.batched_kitchen import KitchenVariant, demonstration_states

    qpos, qvel, time = demonstration_states()
    assert qpos.shape[1] == 30 and qvel.shape[1] == 29 and len(time) == len(qpos)
    env = BatchedKitchen(64, variant=KitchenVariant(demo_reset_fraction=0.5))
    env.reset(seeds=np.arange(64))
    demo = env._demo_row
    assert 16 < demo.sum() < 48  # about half
    assert (env.steps[~demo] == 0).all() and (env.steps[demo] > 0).any()
    # Every demo-started row sits on a recorded state, and tasks done there are preset.
    done = env.goal_distance() < 0.3
    assert (env.preset[demo] == done[demo]).all()
    assert not env.preset[~demo].any()
    # A task the demonstration had already done is not earned again.
    result = env.step(np.zeros((64, 9)))
    assert not (result.completed & env.preset & (result.newly_completed[:, None] > 0)).any()


def test_final_strict_bonus_pays_for_elements_left_at_their_goal() -> None:
    env = BatchedKitchen(
        1,
        horizon=20,
        final_strict_bonus=50.0,
        task_shaping_weight=0.0,
        terminate_on_all_tasks=False,
    )
    env.reset(seeds=np.array([0]))
    index = env.tasks.index("slide cabinet")
    env.qpos[0, env._element_indices[index]] = env._element_goals[index]
    env.qvel[0, env._element_dofs[index]] = 0.0
    env.batch.forward(np.array([0]))
    rewards = [env.step(np.zeros((1, 9)), auto_reset=False).reward[0] for _ in range(20)]
    assert rewards[0] == pytest.approx(200.0, abs=1.0)  # the completion bonus
    assert all(abs(r) < 1.0 for r in rewards[1:-1])
    assert rewards[-1] == pytest.approx(50.0, abs=1.0)  # held at the end
