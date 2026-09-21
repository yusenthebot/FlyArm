from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from flyarm.env import PandaReachEnv

MODEL = os.environ.get("FLYARM_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file(), reason="set FLYARM_MODEL to Panda scene.xml"
)


@pytest.fixture
def env() -> Iterator[PandaReachEnv]:
    assert MODEL is not None
    instance = PandaReachEnv(Path(MODEL), horizon=160)
    yield instance
    instance.close()


def test_reset_is_seeded_and_observation_is_finite(env: PandaReachEnv) -> None:
    first, _ = env.reset(seed=7)
    target = env.target.copy()
    second, _ = env.reset(seed=7)
    assert first.shape == (20,)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.allclose(first, second)
    assert np.allclose(target, env.target)
    marker = env.model.body("flyarm_target")
    assert marker is not None
    assert np.allclose(env.data.xpos[marker.id], env.target)


@pytest.mark.parametrize("action", [np.zeros(2), np.array([np.nan, 0, 0]), np.array([1.1, 0, 0])])
def test_invalid_actions_are_rejected(env: PandaReachEnv, action: np.ndarray) -> None:
    with pytest.raises(ValueError):
        env.step(action)


def test_commands_are_bounded_and_step_is_dynamic(env: PandaReachEnv) -> None:
    env.reset(seed=1)
    qpos_before = env.data.qpos.copy()
    env.step(np.ones(3, dtype=np.float32))
    limits = env.model.jnt_range[env._joint_ids]
    assert np.all(env.last_joint_command >= limits[:, 0])
    assert np.all(env.last_joint_command <= limits[:, 1])
    assert env.data.time == pytest.approx(0.05)
    assert not np.array_equal(qpos_before, env.data.qpos)


def test_teacher_reaches_real_mujoco_target(env: PandaReachEnv) -> None:
    env.reset(seed=4)
    success = False
    for _ in range(env.horizon):
        _, _, terminated, truncated, info = env.step(env.teacher_action())
        success |= terminated or info["is_success"]
        if terminated or truncated:
            break
    assert success, f"teacher failed: distance={info['distance']:.3f}"


def test_teacher_reliably_reaches_development_regression_seeds(env: PandaReachEnv) -> None:
    failures = []
    for seed in range(20_000, 20_024):
        env.reset(seed=seed)
        for _ in range(env.horizon):
            _, _, terminated, truncated, info = env.step(env.teacher_action())
            if terminated or truncated:
                break
        if not info["is_success"]:
            failures.append((seed, info["distance"]))
    assert not failures, f"teacher failures: {failures}"
