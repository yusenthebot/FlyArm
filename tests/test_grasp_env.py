from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from flyarm.grasp import task

MODEL = os.environ.get("FLYARM_MODEL")
OBJECTS = Path(os.environ.get("FLYARM_OBJECTS", "assets/objects"))
needs_assets = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file() or not OBJECTS.is_dir(),
    reason="set FLYARM_MODEL to the Panda scene.xml and fetch the grasp objects",
)
# A few objects of different families and both splits keep these tests quick.
NAMES = ("classic_blue_mug", "hammer", "crayon_box", "school_bus")


def _model_path() -> Path:
    if MODEL is None:
        pytest.skip("set FLYARM_MODEL to the Panda scene.xml")
    return Path(MODEL)


def _objects() -> list:
    from flyarm.grasp.objects import load_manifest

    manifest = load_manifest()
    return [manifest[name] for name in NAMES]


# Rules that need no simulator ----------------------------------------------------------------


def test_hold_counts_consecutive_lifted_grasps_and_success_needs_twenty() -> None:
    hold = np.zeros(3, dtype=np.int64)
    lifted = np.array([task.LIFT_HEIGHT, task.LIFT_HEIGHT, task.LIFT_HEIGHT - 1e-3])
    grasped = np.array([True, True, True])
    for _ in range(task.HOLD_STEPS - 1):
        hold = task.update_hold(hold, grasped, lifted)
    assert not task.is_success(hold).any()
    hold = task.update_hold(hold, grasped, lifted)
    assert task.is_success(hold).tolist() == [True, True, False]
    # Losing either finger or dropping below the target restarts the count.
    hold = task.update_hold(hold, np.array([False, True, True]), lifted)
    assert hold.tolist() == [0, task.HOLD_STEPS + 1, 0]


def test_success_bonus_exceeds_the_stalling_floor() -> None:
    floor = task.minimum_success_bonus(task.DEFAULT_GAMMA)
    assert (
        task.DEFAULT_SUCCESS_BONUS
        > floor
        == pytest.approx(task.STEP_REWARD_MAX / (1 - task.DEFAULT_GAMMA))
    )
    with pytest.raises(ValueError, match="stalling floor"):
        task.RewardConfig(success_bonus=floor)
    with pytest.raises(ValueError, match="stalling floor"):
        task.RewardConfig(success_bonus=task.DEFAULT_SUCCESS_BONUS, gamma=0.995)
    task.RewardConfig(success_bonus=2.0 * task.minimum_success_bonus(0.995), gamma=0.995)


def test_per_step_reward_is_bounded_and_the_action_cost_only_subtracts() -> None:
    generator = np.random.default_rng(0)
    n = 4096
    config = task.RewardConfig()
    ee = generator.uniform(-0.1, 0.1, (n, 3))
    target = ee + generator.normal(0, 0.02, (n, 3)) * generator.integers(0, 2, (n, 1))
    arguments = (
        ee,
        target,
        generator.uniform(-np.pi / 2, np.pi / 2, n),
        generator.integers(0, 2, n).astype(bool),
        generator.uniform(-0.05, 0.3, n),
        np.zeros(n, dtype=bool),
    )
    still = task.shaped_reward(*arguments, np.zeros((n, 5)), config)
    moving = task.shaped_reward(*arguments, generator.uniform(-1, 1, (n, 5)), config)
    assert still.min() >= 0.0
    assert still.max() <= task.STEP_REWARD_MAX + 1e-6
    assert np.all(moving <= still + 1e-6)
    assert np.all(still - moving <= 5 * config.action_cost + 1e-6)
    # The best non-terminal step is worth less, forever, than finishing once.
    finished = task.shaped_reward(*arguments[:5], np.ones(n, dtype=bool), np.zeros((n, 5)), config)
    assert finished.min() > task.STEP_REWARD_MAX / (1 - config.gamma)


def test_rotation_error_matches_mujoco_sub_quat() -> None:
    generator = np.random.default_rng(1)
    for _ in range(50):
        current_q, desired_q = generator.normal(size=4), generator.normal(size=4)
        current_q /= np.linalg.norm(current_q)
        desired_q = current_q + 0.3 * desired_q
        desired_q /= np.linalg.norm(desired_q)
        current, desired = np.empty(9), np.empty(9)
        mujoco.mju_quat2Mat(current, current_q)
        mujoco.mju_quat2Mat(desired, desired_q)
        local = np.empty(3)
        mujoco.mju_subQuat(local, desired_q, current_q)
        expected = current.reshape(3, 3) @ local
        got = task.rotation_error(desired.reshape(1, 3, 3), current.reshape(1, 3, 3))[0]
        assert np.allclose(got, expected, atol=1e-9)
        assert np.allclose(task.quat_to_mat(current_q[None])[0], current.reshape(3, 3))


def test_posture_pull_stays_in_the_null_space() -> None:
    generator = np.random.default_rng(2)
    jacobian = generator.normal(size=(8, 6, 7))
    joints = generator.uniform(-1, 1, (8, 7))
    limits = np.tile([[-3.0, 3.0]], (7, 1))
    zero = np.zeros((8, 3))
    moved = task.ik_step(jacobian, joints, limits, zero, zero, joints + 0.2)
    weighted = np.concatenate((jacobian[:, :3], 0.25 * jacobian[:, 3:]), axis=1)
    assert np.abs(moved - joints).max() > 1e-3
    assert np.abs(np.einsum("nij,nj->ni", weighted, moved - joints)).max() < 1e-3


# Simulated environments ----------------------------------------------------------------------


@needs_assets
def test_batched_env_matches_the_single_env_step_for_step() -> None:
    from flyarm.grasp.env import BatchedGrasp, PandaGraspEnv
    from flyarm.grasp.teacher import GraspTeacher

    objects, seeds, steps = _objects(), [123, 456, 789, 1011], 140
    # A fixed action sequence per episode: the teacher's actions, recorded once in the single
    # environment, then replayed open loop in the batched one (one object per environment).
    single = PandaGraspEnv(_model_path(), objects=objects, asset_root=OBJECTS)
    teacher = GraspTeacher(single.sim)
    recorded, trajectories = [], []
    for row, seed in enumerate(seeds):
        obs, _ = single.reset(seed=seed, options={"object": row})
        teacher.reset()
        actions, observations = [], [obs]
        for _ in range(steps):
            actions.append(teacher.act()[0])
            observations.append(single.step(actions[-1])[0])
        recorded.append(actions)
        trajectories.append(observations)
    action_array, expected = np.array(recorded), np.array(trajectories)

    batched = BatchedGrasp(_model_path(), len(seeds), objects=objects, asset_root=OBJECTS)
    obs = batched.reset(seeds=np.array(seeds), objects=np.arange(len(seeds)))
    worst = float(np.abs(obs - expected[:, 0]).max())
    for step in range(steps):
        result = batched.step(action_array[:, step], auto_reset=False)
        worst = max(worst, float(np.abs(result.obs - expected[:, step + 1]).max()))
    assert worst < 1e-6, worst
    assert batched.ever_grasped.all()  # the replay exercised contact, not just free motion


@needs_assets
def test_hinge_jacobian_matches_mj_jacsite() -> None:
    from flyarm.grasp.env import PandaGraspEnv

    env = PandaGraspEnv(_model_path(), objects=_objects()[:1], asset_root=OBJECTS)
    generator = np.random.default_rng(3)
    env.reset(seed=5)
    for _ in range(12):
        env.step(generator.uniform(-1, 1, 5))
        model, data, sim = env.model, env.data, env.sim
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sim._ee_site)
        expected = np.concatenate((jacp[:, sim._dadr], jacr[:, sim._dadr]))
        assert np.allclose(sim.jacobian()[0], expected, atol=1e-12)


@needs_assets
def test_one_batch_holds_different_objects_and_parks_the_rest() -> None:
    from flyarm.grasp.env import BatchedGrasp

    objects = _objects()
    env = BatchedGrasp(_model_path(), 4, objects=objects, asset_root=OBJECTS)
    env.reset(seeds=np.arange(4), objects=np.array([3, 2, 1, 0]))
    assert env.object_index.tolist() == [3, 2, 1, 0]
    parked_before = env.qpos.copy()
    for _ in range(30):
        env.step(np.zeros((4, 5)), auto_reset=False)
    for row in range(4):
        active = env.object_index[row]
        on_table = env.object_pos()[row]
        assert task.WORKSPACE_LOW[0] - 0.01 <= on_table[0] <= task.WORKSPACE_HIGH[0] + 0.01
        assert abs(env.height_gain()[row]) < 0.01  # resting where it was placed
        for index, start in enumerate(env._obj_qadr):
            if index == active:
                continue
            # Parked objects float motionless, far from the robot and from each other.
            assert (
                np.abs(
                    env.qpos[row, start : start + 7] - parked_before[row, start : start + 7]
                ).max()
                < 1e-6
            )
            assert env.qpos[row, start] > 2.0


@needs_assets
def test_reset_pose_depends_on_the_seed_not_on_the_object_set() -> None:
    from flyarm.grasp.env import BatchedGrasp

    objects = _objects()
    many = BatchedGrasp(_model_path(), 2, objects=objects, asset_root=OBJECTS)
    one = BatchedGrasp(_model_path(), 2, objects=objects[1:2], asset_root=OBJECTS)
    many.reset(seeds=np.array([5, 6]), objects=np.array([1, 1]))
    one.reset(seeds=np.array([5, 6]))
    assert np.allclose(many.object_pos(), one.object_pos())
    assert np.allclose(many.object_quat(), one.object_quat())


@needs_assets
def test_observation_layout_and_descriptor() -> None:
    from flyarm.grasp.env import BatchedGrasp

    objects = _objects()
    env = BatchedGrasp(_model_path(), 4, objects=objects, asset_root=OBJECTS)
    obs = env.reset(seeds=np.arange(4), objects=np.arange(4))
    privileged = env.observation(privileged=True)
    assert obs.shape == (4, task.OBS_DIM) and privileged.shape == (4, task.PRIVILEGED_DIM)
    assert np.array_equal(privileged[:, : task.OBS_DIM], obs)
    fields = task.field_slices()
    assert np.allclose(obs[:, fields["object_descriptor"]], [item.descriptor for item in objects])
    assert np.allclose(obs[:, fields["lift_target"]][:, 0], [item.rest_z + 0.1 for item in objects])
    assert np.all(np.isfinite(obs))


@needs_assets
def test_teacher_grasps_lifts_and_holds_objects_of_several_families() -> None:
    from flyarm.grasp.env import BatchedGrasp
    from flyarm.grasp.teacher import GraspTeacher

    objects = _objects()
    episodes = 4
    env = BatchedGrasp(_model_path(), len(objects) * episodes, objects=objects, asset_root=OBJECTS)
    teacher = GraspTeacher(env)
    index = np.repeat(np.arange(len(objects)), episodes)
    env.reset(seeds=np.arange(len(index)) + 777, objects=index)
    teacher.reset()
    success = np.zeros(len(index), dtype=bool)
    active = np.ones(len(index), dtype=bool)
    for _ in range(env.horizon):
        before = env.hold.copy()
        result = env.step(teacher.act(), auto_reset=False)
        finished = active & result.success
        # A terminated episode really met the rule: 20 lifted steps with both fingers on.
        assert np.all(before[finished] == task.HOLD_STEPS - 1)
        assert np.all(result.grasped[finished] & (result.height_gain[finished] >= task.LIFT_HEIGHT))
        success |= finished
        active &= ~(result.success | result.truncated)
    assert success.mean() >= 0.9, success.reshape(len(objects), episodes)
