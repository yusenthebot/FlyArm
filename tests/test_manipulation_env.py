from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from flyarm.manipulation import furniture as fu
from flyarm.manipulation import tasks as tk

MODEL = os.environ.get("FLYARM_MODEL")
OBJECTS = Path(os.environ.get("FLYARM_OBJECTS", "assets/objects"))
pytestmark = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file() or not OBJECTS.is_dir(),
    reason="set FLYARM_MODEL to the Panda scene.xml and fetch the grasp objects",
)
HOLD = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # stay still with the hand open


@pytest.fixture(scope="module")
def env():
    from flyarm.manipulation.env import PandaManipulationEnv

    environment = PandaManipulationEnv(Path(MODEL), split="train", asset_root=OBJECTS)
    yield environment
    environment.close()


def _settle(env, steps: int) -> None:
    for _ in range(steps):
        env.step(HOLD)


def _put(env, slot: int, xyz: np.ndarray, yaw: float) -> None:
    """Teleport the object in ``slot`` to rest at ``xyz`` (its origin) with ``yaw``."""
    sim = env.sim
    start = sim._obj_qadr[sim.slot_object[0, slot]]
    sim.qpos[0, start : start + 7] = [*xyz, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    dof = sim._obj_dadr[sim.slot_object[0, slot]]
    sim.qvel[0, dof : dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    sim.remember_poses()


def _set_joint(env, index: int, value: float) -> None:
    env.sim.qpos[0, env.sim._art_qadr[index]] = value
    env.sim.qvel[0, env.sim._art_dadr[index]] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.sim.remember_poses()


def _effects(env) -> np.ndarray:
    sim = env.sim
    return sim.effects(sim.scene_state())[0, : sim.sub_count[0]]


def test_drawer_and_lid_success_checks(env) -> None:
    env.reset(seed=11, options={"template": "tidy"})
    goals = env.sim.episodes[0].subgoals
    drawer = goals[0].target
    travel = env.sim.travel[0, drawer]
    _set_joint(env, drawer, 0.8 * travel)
    effects = _effects(env)
    assert effects[0] and not effects[2]  # open, not closed
    _set_joint(env, drawer, 0.7 * travel)
    assert not _effects(env)[0]
    _set_joint(env, drawer, 0.003)
    assert _effects(env)[2]  # within 4 mm of the stop counts as closed
    _set_joint(env, drawer, 0.006)
    assert not _effects(env)[2]
    _set_joint(env, 2, 1.6)
    assert _effects(env)[3] and not _effects(env)[5]
    _set_joint(env, 2, 1.5)
    assert not _effects(env)[3]
    _set_joint(env, 2, 0.015)
    assert _effects(env)[5]


def test_place_needs_the_whole_object_inside_and_at_rest(env) -> None:
    env.reset(seed=12, options={"template": "put_away"})
    sim = env.sim
    goals = sim.episodes[0].subgoals
    drawer, slot = goals[0].target, goals[1].obj
    _set_joint(env, drawer, sim.travel[0, drawer])
    origin, rotation = (array[0, drawer] for array in sim.receptacle_frames())
    item = sim.objects[sim.slot_object[0, slot]]
    point = origin + rotation @ (sim.place_point[0, 1] + np.array([0.0, 0.0, item.rest_z + 0.001]))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    _put(env, slot, point, yaw)
    _settle(env, tk.PLACE_STEPS - 2)
    assert not _effects(env)[1]  # not yet at rest for long enough
    _settle(env, 4)
    effects = _effects(env)
    assert effects[1] and effects[0]  # placed, and the opening it needed stays done
    done = sim.subgoal_done(sim.effects(sim.scene_state()))
    assert sim.leading(done)[0] == 2  # the drawer is still open: close_drawer is next
    # The same object at rest on the table beside the cabinet is not in the drawer.
    region = sim.region_pos[0] + np.array([0.0, 0.0, item.rest_z + 0.001])
    _put(env, slot, region, 0.0)
    _settle(env, tk.PLACE_STEPS + 2)
    assert not _effects(env)[1]


def test_stack_needs_twenty_stable_steps_on_top(env) -> None:
    env.reset(seed=13, options={"template": "tower"})
    sim = env.sim
    goals = sim.episodes[0].subgoals
    base, top = goals[1].target, goals[1].obj
    region = sim.region_pos[0]
    base_item = sim.objects[sim.slot_object[0, base]]
    top_item = sim.objects[sim.slot_object[0, top]]
    _put(env, base, region + np.array([0.0, 0.0, base_item.rest_z + 0.001]), 0.0)
    height = base_item.size[2] + top_item.rest_z + 0.002
    _put(env, top, region + np.array([0.0, 0.0, height]), 0.0)
    _settle(env, tk.STACK_STEPS - 5)
    assert not _effects(env)[1]  # on top, but not yet stable for long enough
    _settle(env, 7)
    effects = _effects(env)
    assert effects[0] and effects[1]
    # Beside the base instead of on it: not a stack.
    clear = max(base_item.size[:2]) / 2 + max(top_item.size[:2]) / 2 + 0.03
    beside = region + np.array([0.0, clear, top_item.rest_z + 0.001])
    _put(env, top, beside, 0.0)
    _settle(env, tk.STACK_STEPS + 2)
    assert not _effects(env)[1]


def test_bonus_is_paid_once_in_order_and_undoing_pays_nothing(env) -> None:
    env.reset(seed=14, options={"template": "put_away"})
    sim = env.sim
    drawer = sim.episodes[0].subgoals[0].target
    config = sim.reward_config
    _set_joint(env, drawer, sim.travel[0, drawer])
    _, reward, *_ = env.step(HOLD)
    assert reward > config.subgoal_bonus - config.shaping  # the first completion pays
    _set_joint(env, drawer, 0.0)
    _, undo, *_ = env.step(HOLD)
    _set_joint(env, drawer, sim.travel[0, drawer])
    _, redo, *_ = env.step(HOLD)
    assert redo < config.subgoal_bonus / 2  # the high-water mark stops a second bonus
    assert redo + undo == pytest.approx(0.0, abs=0.5)  # shaping telescopes


def test_cue_names_the_current_subgoal_and_the_no_cue_control_is_blank(env) -> None:
    from flyarm.manipulation.env import PandaManipulationEnv
    from flyarm.manipulation.sim import cue_slices

    obs, _ = env.reset(seed=15, options={"template": "shelve"})
    cue = cue_slices()
    assert obs[cue["skill"]].argmax() + 1 == tk.OPEN_DOOR
    assert obs[cue["articulation"]].tolist() == [0.0, 0.0, 1.0]
    _set_joint(env, 2, 1.62)
    obs, *_ = env.step(HOLD)
    assert obs[cue["skill"]].argmax() + 1 == tk.PLACE
    assert obs[cue["receptacle"]].argmax() == tk.SHELF
    blank = PandaManipulationEnv(Path(MODEL), split="train", asset_root=OBJECTS, cue=False)
    obs, _ = blank.reset(seed=15, options={"template": "shelve"})
    assert not obs[slice(cue["skill"].start, None)].any()
    blank.close()


def test_batched_env_matches_the_single_env_step_for_step() -> None:
    from flyarm.manipulation.env import BatchedManipulation, PandaManipulationEnv
    from flyarm.manipulation.teacher import ManipulationTeacher

    plan, steps = [(21, "put_away"), (22, "shelve")], 160
    single = PandaManipulationEnv(Path(MODEL), split="train", asset_root=OBJECTS)
    recorded, observed = [], []
    for seed, template in plan:
        obs, _ = single.reset(seed=seed, options={"template": template})
        teacher = ManipulationTeacher(single.sim)
        teacher.reset()
        actions, observations = [], [obs]
        for _ in range(steps):
            actions.append(teacher.act()[0])
            observations.append(single.step(actions[-1])[0])
        recorded.append(actions)
        observed.append(observations)
    actions, expected = np.array(recorded), np.array(observed)
    batched = BatchedManipulation(Path(MODEL), len(plan), split="train", asset_root=OBJECTS)
    obs = batched.reset(
        seeds=np.array([seed for seed, _ in plan]), templates=[name for _, name in plan]
    )
    worst = float(np.abs(obs - expected[:, 0]).max())
    for step in range(steps):
        result = batched.step(actions[:, step], auto_reset=False)
        worst = max(worst, float(np.abs(result.obs - expected[:, step + 1]).max()))
    assert worst < 1e-6, worst
    assert np.abs(batched.joints()).max() > 0.02  # the replay moved a drawer or the lid


def _rotation(quat: np.ndarray) -> np.ndarray:
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, quat)
    return out.reshape(3, 3)


def test_resized_furniture_stays_inside_the_compiled_bounding_boxes() -> None:
    """Every configuration's geoms lie inside the bounds MuJoCo computed at compile time."""
    from flyarm.grasp.objects import split_objects
    from flyarm.manipulation.scene import build_manipulation_model

    model = build_manipulation_model(Path(MODEL), split_objects("train")[:2], OBJECTS)
    fields = fu.FurnitureFields(model)
    names = {bid: name for name, bid in fields.body_ids.items()}
    generator = np.random.default_rng(0)
    for draw in range(60):
        ranges = fu.TRAIN_RANGES if draw % 2 else fu.HELD_OUT_RANGES
        plan = fu.layout(fu.sample_furniture(generator, ranges))
        for name, geom in plan.geoms.items():
            gid = fields.geom_ids[name]
            if not geom.enabled or not model.geom_contype[gid]:
                continue
            radius, half = geom.size[0], geom.size[1]
            kind = mujoco.mjtGeom(int(model.geom_type[gid]))
            local, sphere = {
                mujoco.mjtGeom.mjGEOM_BOX: (np.array(geom.size), np.linalg.norm(geom.size)),
                mujoco.mjtGeom.mjGEOM_CYLINDER: (
                    np.array([radius, radius, half]),
                    math.hypot(radius, half),
                ),
                mujoco.mjtGeom.mjGEOM_SPHERE: (np.array([radius] * 3), radius),
            }[kind]
            # Geom level: the bounding sphere and the geom-frame box.
            assert sphere <= model.geom_rbound[gid] + 1e-9, (draw, name)
            assert np.all(local <= model.geom_aabb[gid, 3:] + 1e-9), (draw, name)
            # Body level: the root of the body's bounding tree, in its inertial frame.
            body = int(model.geom_bodyid[gid])
            ipos = plan.body_ipos.get(names[body], model.body_ipos[body])
            inertial = _rotation(model.body_iquat[body]).T
            centre = inertial @ (np.array(geom.pos) - ipos)
            extent = np.abs(inertial @ _rotation(model.geom_quat[gid])) @ local
            root = model.bvh_aabb[model.body_bvhadr[body]]
            assert np.all(np.abs(centre - root[:3]) + extent <= root[3:] + 1e-9), (draw, name)


def test_teacher_completes_drawer_tasks() -> None:
    from flyarm.manipulation.env import BatchedManipulation
    from flyarm.manipulation.teacher import ManipulationTeacher

    names = ["put_away"] * 3 + ["retrieve"] * 3
    env = BatchedManipulation(Path(MODEL), len(names), split="train", asset_root=OBJECTS)
    teacher = ManipulationTeacher(env)
    env.reset(seeds=np.arange(len(names)) + 500, templates=names)
    teacher.reset()
    success = np.zeros(len(names), dtype=bool)
    active = np.ones(len(names), dtype=bool)
    for _ in range(env.horizon):
        result = env.step(teacher.act(), auto_reset=False)
        success |= active & result.success
        active &= ~(result.success | result.truncated)
        if not active.any():
            break
    assert success.sum() >= 5, success


def test_containment_uses_the_hull_not_the_bounding_box_corners(env) -> None:
    """A round object's box corners overshoot its hull; the hull points span the box faces."""
    sim = env.sim
    for index, item in enumerate(sim.objects):
        points = sim.hull_points_all[index]
        half = np.array(item.size) / 2
        assert np.all(np.abs(points.max(0) - half) < 0.002), item.name
        assert np.all(np.abs(points.min(0) + half) < 0.002), item.name
    radius = [np.linalg.norm(points[:, :2], axis=1).max() for points in sim.hull_points_all]
    corners = [np.linalg.norm(item.size[:2]) / 2 for item in sim.objects]
    assert min(r / c for r, c in zip(radius, corners, strict=True)) < 0.8  # round objects exist
