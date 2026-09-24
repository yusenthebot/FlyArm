"""The articulated tabletop scene: Panda, scanned objects, drawer cabinet, lidded cabinet, bin.

One compiled model holds the robot, every object of an object split (parked unless used, as in
:mod:`flyarm.grasp.scene`) and one instance of each furniture piece compiled at its largest
configuration (:mod:`flyarm.manipulation.furniture`). Mid-phase collision is disabled so that
per-episode resizing stays exact.

Contact sensors: two per object for the fingers (as in the grasp task), one counting every
contact of the robot, and one per object and per moving furniture body counting the robot's
contacts with it, so a stray contact is the robot total minus the allowed ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from flyarm.grasp.objects import GraspObject
from flyarm.grasp.scene import add_object, body_name, check_compiled, look_at
from flyarm.manipulation.furniture import add_furniture, compile_config
from flyarm.pick_place_env import add_flyarm_gripper, compile_pick_place_model

ROBOT_ROOT = "link0"
ARTICULATED = ("drawer_0", "drawer_1", "lid", "lid_handle")
CAMERA = "flyarm_manipulation_view"
CAMERA_POS = np.array([1.35, 0.0, 0.95])
CAMERA_TARGET = np.array([0.42, 0.0, 0.08])
NOSLIP_ITERATIONS = 4
_FOUND = 1  # contact sensor data: number of matching contacts
_REDUCE_MAXFORCE, _ONE = 2, 1
# Contact pairs that carry no information because a joint limit already enforces them, and that
# dominated the physics cost: a closed lid rests on the four cabinet walls with 20 contacts in
# every episode (the cabinet is welded to the world, so MuJoCo's parent-child filter does not
# apply), although the hinge's lower limit already holds it at 0; and fingers closed on nothing
# made up to 44 finger-to-finger contacts, although the finger joints' lower limit already stops
# them at touching. Excluding them cut a 128-environment step of an undertrained policy from
# 264 to 176 ms (docs/MANIPULATION_ENV.md, "Excluded contact pairs").
EXCLUDED_PAIRS = (("cabinet", "lid"), ("left_finger", "right_finger"))


def robot_sensor_name(target: str) -> str:
    return f"robot_contact_{target}"


def finger_sensor_name(side: str, body: str) -> str:
    return f"{side}_finger_contact_{body}"


def _robot_contact(spec: mujoco.MjSpec, name: str, body: str | None, subtree: bool = False) -> None:
    kwargs = {}
    if body is not None:
        kind = mujoco.mjtObj.mjOBJ_XBODY if subtree else mujoco.mjtObj.mjOBJ_BODY
        kwargs = {"reftype": kind, "refname": body}
    spec.add_sensor(
        name=name,
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,  # the robot's whole subtree
        objname=ROBOT_ROOT,
        intprm=[_FOUND, _REDUCE_MAXFORCE, _ONE],
        **kwargs,
    )


def build_manipulation_spec(
    model_path: Path, objects: Sequence[GraspObject], asset_root: Path
) -> mujoco.MjSpec:
    if not objects:
        raise ValueError("the manipulation scene needs at least one object")
    spec = mujoco.MjSpec.from_file(str(model_path))
    add_flyarm_gripper(spec)
    for index, item in enumerate(objects):
        add_object(spec, item, index, asset_root)
    add_furniture(spec, compile_config())
    for first, second in EXCLUDED_PAIRS:
        spec.add_exclude(bodyname1=first, bodyname2=second)
    _robot_contact(spec, robot_sensor_name("any"), None)
    _robot_contact(spec, robot_sensor_name("self"), ROBOT_ROOT, subtree=True)
    for item in objects:
        _robot_contact(spec, robot_sensor_name(item.name), body_name(item))
    for body in ARTICULATED:
        _robot_contact(spec, robot_sensor_name(body), body)
        for side in ("left", "right"):
            spec.add_sensor(
                name=finger_sensor_name(side, body),
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                objname=f"{side}_finger",
                reftype=mujoco.mjtObj.mjOBJ_BODY,
                refname=body,
                intprm=[1, 1, 1],
            )
    spec.worldbody.add_camera(
        name=CAMERA, pos=CAMERA_POS.tolist(), xyaxes=look_at(CAMERA_POS, CAMERA_TARGET), fovy=50
    )
    return spec


def build_manipulation_model(
    model_path: Path, objects: Sequence[GraspObject], asset_root: Path
) -> mujoco.MjModel:
    model = compile_pick_place_model(build_manipulation_spec(model_path, objects, asset_root))
    # Geometry is resized per episode inside compile-time bounds; the mid phase would use
    # per-geom bounding trees built at compile time, the broad phase only whole-body bounds.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    # MuJoCo's no-slip post-solver: objects carried for a hundred steps otherwise creep out of
    # the pinch (a 47 g toy basket tilted 29 degrees and slipped out during one carry with the
    # defaults). Elliptic cones with a high impedance ratio also stop the creep but make a pinch
    # on a resting object chatter between the two pads, so the default pyramidal cone stays.
    model.opt.noslip_iterations = NOSLIP_ITERATIONS
    for item in objects:
        check_compiled(model, item)
    return model
