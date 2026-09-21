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
