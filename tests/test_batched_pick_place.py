from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

MODEL = os.environ.get("FLYARM_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file(), reason="set FLYARM_MODEL to Panda scene.xml"
)


def _teacher_episode(seed: int, steps: int) -> tuple[list[np.ndarray], list[np.ndarray], list]:
    from flyarm.pick_place_env import PandaPickPlaceEnv

    env = PandaPickPlaceEnv(Path(MODEL))
    obs, _ = env.reset(seed=seed)
    observations, actions, infos = [obs], [], []
    for _ in range(steps):
        action = env.teacher_action()
        obs, _, terminated, _, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        infos.append(info)
        if terminated:
            break
    return observations, actions, infos


def test_batched_env_reproduces_the_single_env_under_the_teacher() -> None:
    from flyarm.rl.batched_pick_place import BatchedPickPlace

    seeds = [60000, 60001]
    episodes = [_teacher_episode(seed, 400) for seed in seeds]
    batched = BatchedPickPlace(Path(MODEL), len(seeds))
    first = batched.reset(seeds=np.array(seeds))
    for row, (observations, _, _) in enumerate(episodes):
        assert np.allclose(first[row], observations[0], atol=1e-6)
    length = min(len(actions) for _, actions, _ in episodes)
    worst = 0.0
    for step in range(length):
        actions = np.stack([actions[step] for _, actions, _ in episodes])
        result = batched.step(actions, auto_reset=False)
        for row, (observations, _, infos) in enumerate(episodes):
            worst = max(worst, float(np.abs(result.obs[row] - observations[step + 1]).max()))
            assert result.grasped[row] == infos[step]["grasped"]
            assert result.success[row] == infos[step]["is_success"]
    assert worst < 1e-5, worst
    # The teacher places both cubes, so the success rule fired in the batched env as well.
    assert all(infos[-1]["is_success"] for _, _, infos in episodes)


def test_hinge_jacobian_matches_mj_jacsite() -> None:
    from flyarm.rl.batched_pick_place import BatchedPickPlace

    batched = BatchedPickPlace(Path(MODEL), 1)
    batched.reset(seeds=np.array([3]))
    model = batched.model
    data = mujoco.MjData(model)
    data.qpos[:] = batched.qpos[0]
    mujoco.mj_forward(model, data)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, batched._ee_site)
    axes, anchors = data.xaxis[batched._jnt], data.xanchor[batched._jnt]
    site = data.site_xpos[batched._ee_site]
    assert np.allclose(np.cross(axes, site - anchors).T, jacp[:, batched._dadr], atol=1e-12)
    assert np.allclose(axes.T, jacr[:, batched._dadr], atol=1e-12)


def test_auto_reset_draws_fresh_seeds_and_clears_episode_flags() -> None:
    from flyarm.rl.batched_pick_place import BatchedPickPlace

    batched = BatchedPickPlace(Path(MODEL), 3, horizon=20, first_seed=100)
    batched.reset()
    assert batched.episode_seed.tolist() == [100, 101, 102]
    for _ in range(20):
        result = batched.step(np.zeros((3, 4)))
    assert result.truncated.all() and not result.terminated.any()
    assert batched.episode_seed.tolist() == [103, 104, 105]
    assert (batched.steps == 0).all() and not batched.ever_lifted.any()


def test_memory_variant_blanks_the_goal_for_the_controller_only() -> None:
    from flyarm.rl.batched_pick_place import GOAL_FIELDS, BatchedPickPlace, TaskVariant

    env = BatchedPickPlace(Path(MODEL), 2, variant=TaskVariant(goal_visible_steps=2))
    first = env.reset(seeds=np.array([5, 6]))
    assert np.abs(first[:, GOAL_FIELDS]).sum() > 0  # visible at the start
    for _ in range(2):
        result = env.step(np.zeros((2, 4)))
    assert np.all(result.obs[:, GOAL_FIELDS] == 0.0)
    privileged = env.observation(privileged=True)
    assert np.allclose(privileged[:, 20:23], env.goal)
    other = [i for i in range(37) if i not in set(GOAL_FIELDS)]
    assert np.array_equal(result.obs[:, other], privileged[:, other])


def test_physics_variant_is_seeded_bounded_and_leaves_the_nominal_task_alone() -> None:
    from flyarm.rl.batched_pick_place import BASE_FRICTION, BatchedPickPlace, TaskVariant

    variant = TaskVariant(mass_scale=(2.0, 8.0), friction_scale=(0.2, 0.5))
    env = BatchedPickPlace(Path(MODEL), 3, variant=variant)
    env.reset(seeds=np.array([7, 8, 9]))
    cube, geom = env._cube_body, env._cube_geom
    assert np.all((env.mass_scale >= 2.0) & (env.mass_scale <= 8.0))
    assert np.allclose(env.body_mass[:, cube], 0.025 * env.mass_scale)
    assert np.allclose(env.geom_friction[:, geom, 0], BASE_FRICTION * env.friction_scale)
    assert np.allclose(env.geom_friction[:, env._pads[0], 0], BASE_FRICTION * env.friction_scale)
    again = BatchedPickPlace(Path(MODEL), 3, variant=variant)
    again.reset(seeds=np.array([7, 8, 9]))
    assert np.array_equal(again.mass_scale, env.mass_scale)
    # Randomization uses its own stream: the cube and goal start where the nominal task does.
    nominal = BatchedPickPlace(Path(MODEL), 3)
    nominal.reset(seeds=np.array([7, 8, 9]))
    assert np.allclose(nominal.goal, env.goal) and np.allclose(nominal.cube(), env.cube())
