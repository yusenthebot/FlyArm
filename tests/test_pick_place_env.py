from __future__ import annotations

import inspect
import os
import platform
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flyarm.pick_place_env import PandaPickPlaceEnv

MODEL = os.environ.get("FLYARM_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file(), reason="set FLYARM_MODEL to Panda scene.xml"
)


@pytest.fixture
def env() -> Iterator[PandaPickPlaceEnv]:
    assert MODEL is not None
    instance = PandaPickPlaceEnv(Path(MODEL), horizon=400)
    yield instance
    instance.close()


def _run_teacher(env: PandaPickPlaceEnv, seed: int) -> dict[str, Any]:
    env.reset(seed=seed)
    for _ in range(env.horizon):
        _, _, terminated, truncated, info = env.step(env.teacher_action())
        if terminated or truncated:
            return info
    raise AssertionError("environment did not terminate or truncate")


def test_seeded_reset_and_observation_contract(env: PandaPickPlaceEnv) -> None:
    first, _ = env.reset(seed=71)
    first_object, first_goal = env.object_position.copy(), env.goal.copy()
    second, _ = env.reset(seed=71)
    assert first.shape == (37,)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.allclose(first, second)
    assert np.allclose(first_object, env.object_position)
    assert np.allclose(first_goal, env.goal)


@pytest.mark.parametrize(
    "action", [np.zeros(3), np.array([np.nan, 0, 0, 0]), np.array([0, 0, 0, 1.1])]
)
def test_invalid_actions_are_rejected(env: PandaPickPlaceEnv, action: np.ndarray) -> None:
    with pytest.raises(ValueError):
        env.step(action)


def test_step_uses_dynamics_and_does_not_teleport_object(env: PandaPickPlaceEnv) -> None:
    env.reset(seed=3)
    arm_before = env.data.qpos[env._qadr].copy()
    cube_before = env.data.qpos[env._cube_qadr : env._cube_qadr + 7].copy()
    env.step(np.array([0.4, -0.2, 0.3, 1.0], dtype=np.float32))
    assert env.data.time == pytest.approx(0.05)
    assert not np.array_equal(arm_before, env.data.qpos[env._qadr])
    # The cube may settle vertically under gravity, but an open, spatially
    # separated gripper cannot translate it horizontally in one step.  This
    # catches an accidental object-state assignment hidden in step().
    cube_after = env.data.qpos[env._cube_qadr : env._cube_qadr + 7]
    assert np.allclose(cube_before[:2], cube_after[:2])
    source = inspect.getsource(PandaPickPlaceEnv.step)
    assert "qpos[" not in source
    assert "mocap_pos" not in source


def test_teacher_performs_physical_pick_place_and_release(env: PandaPickPlaceEnv) -> None:
    info = _run_teacher(env, 4)
    assert info["is_success"]
    assert info["ever_grasped"]
    assert info["ever_lifted"]
    assert not info["contact_left"] and not info["contact_right"]
    assert float(info["object_height"]) < 0.032
    assert float(info["goal_xy_error"]) < 0.03


def test_teacher_reliably_solves_development_seeds(env: PandaPickPlaceEnv) -> None:
    results = {seed: _run_teacher(env, seed) for seed in range(12)}
    successes = [seed for seed, info in results.items() if bool(info["is_success"])]
    assert len(successes) >= 11, f"teacher success={successes}; results={results}"


def test_zero_action_cannot_complete_contact_task(env: PandaPickPlaceEnv) -> None:
    env.reset(seed=4)
    info: dict[str, object] = {}
    for _ in range(env.horizon):
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        if terminated or truncated:
            break
    assert not info["is_success"]
    assert not info["ever_lifted"]


@pytest.mark.skipif(
    os.environ.get("CI") == "true" and platform.system() == "Darwin",
    reason="GitHub macOS runners do not expose a CGL pixel format",
)
def test_render_shows_rgb_scene(env: PandaPickPlaceEnv) -> None:
    env.reset(seed=6)
    image = env.render()
    assert image.shape == (480, 640, 3)
    assert image.dtype == np.uint8
    assert int(image.max()) > int(image.min())


def test_robot_payload_exports_the_compiled_panda_geometry(env: PandaPickPlaceEnv) -> None:
    import base64

    from flyarm.live import ROBOT_BODIES, build_robot_payload, robot_body_poses

    payload = build_robot_payload(env.model)
    meshes = [geom for geom in payload["geoms"] if geom["kind"] == "mesh"]
    assert payload["bodies"] == list(ROBOT_BODIES)
    assert len(meshes) >= 50 and any(geom["kind"] == "box" for geom in payload["geoms"])
    for geom in meshes:
        vertices = len(base64.b64decode(geom["positions"])) // 12
        width = "<u2" if geom["index_width"] == 2 else "<u4"
        index = np.frombuffer(base64.b64decode(geom["index"]), dtype=width)
        assert len(index) % 3 == 0 and index.max() < vertices
        assert len(base64.b64decode(geom["normals"])) == vertices * 3
    env.reset(seed=3)
    poses = np.asarray(robot_body_poses(env.model, env.data))
    assert poses.shape == (len(ROBOT_BODIES), 7)
    np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-5)


def _teacher_trace(teacher_resync: bool, seed: int) -> list[np.ndarray]:
    env = PandaPickPlaceEnv(Path(MODEL), teacher_resync=teacher_resync)
    obs, _ = env.reset(seed=seed)
    trace = [obs]
    for _ in range(env.horizon):
        obs, _, terminated, _, _ = env.step(env.teacher_action())
        trace.append(obs)
        if terminated:
            break
    env.close()
    return trace


def test_stage_resync_leaves_the_teacher_demonstrations_unchanged() -> None:
    for seed in (40000, 40001, 40002, 40003, 40004, 40005):
        plain, resync = _teacher_trace(False, seed), _teacher_trace(True, seed)
        assert len(plain) == len(resync)
        assert all(np.array_equal(a, b) for a, b in zip(plain, resync, strict=True))


def test_stage_resync_keeps_holding_a_cube_the_learner_lifted() -> None:
    """A learner that grasps and lifts on its own must not be told to open the gripper."""
    reference = PandaPickPlaceEnv(Path(MODEL))
    reference.reset(seed=60000)
    actions = []
    for _ in range(reference.horizon):
        action = reference.teacher_action()
        actions.append(action)
        _, _, _, _, info = reference.step(action)
        if info["ever_lifted"] and info["grasped"]:
            break
    assert info["ever_lifted"] and info["grasped"]
    labels = {}
    for resync in (False, True):
        learner = PandaPickPlaceEnv(Path(MODEL), teacher_resync=resync)
        learner.reset(seed=60000)
        for action in actions:  # the "learner" acts; its teacher is never consulted
            learner.step(action)
        labels[resync] = learner.teacher_action()
        learner.close()
    assert labels[False][3] > 0  # the stateful teacher is still approaching: "open"
    assert labels[True][3] < 0  # the resynchronized teacher keeps squeezing
