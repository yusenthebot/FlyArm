"""Compatibility shim for gymnasium-robotics 1.4.2 on MuJoCo >= 3.13.

Its joint accessors check ``model.jnt_type[i] in (mjJNT_HINGE, mjJNT_SLIDE)``. With newer
MuJoCo bindings that membership test is False for the numpy integer the model returns
(``==`` still works, ``in`` compares with the enum's own ``__eq__``), so FrankaKitchen fails
to construct. Downgrading MuJoCo would change FlyArm's verified physics, so the four
accessors are replaced with versions that compare integer joint types and are otherwise
identical to upstream (including upstream's dimensions per joint type).
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_SCALAR = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}


def _span(model: Any, name: str, free: int, ball: int, address: Any) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        raise ValueError(f"Joint with name '{name}' is not part of the model!")
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == _FREE:
        ndim = free
    elif joint_type == _BALL:
        ndim = ball
    elif joint_type in _SCALAR:
        ndim = 1
    else:
        raise ValueError(f"Unsupported joint type {joint_type} for '{name}'")
    start = int(address[joint_id])
    return start, start + ndim


def _set(target: np.ndarray, span: tuple[int, int], name: str, value: Any) -> None:
    array = np.array(value)
    if span[1] - span[0] > 1 and array.shape != (span[1] - span[0],):
        raise ValueError(f"Value has incorrect shape {name}: {value}")
    target[span[0] : span[1]] = array


def get_joint_qpos(model: Any, data: Any, name: str) -> np.ndarray:
    start, stop = _span(model, name, 7, 4, model.jnt_qposadr)
    return data.qpos[start:stop].copy()


def get_joint_qvel(model: Any, data: Any, name: str) -> np.ndarray:
    start, stop = _span(model, name, 6, 4, model.jnt_dofadr)
    return data.qvel[start:stop].copy()


def set_joint_qpos(model: Any, data: Any, name: str, value: Any) -> None:
    _set(data.qpos, _span(model, name, 7, 4, model.jnt_qposadr), name, value)


def set_joint_qvel(model: Any, data: Any, name: str, value: Any) -> None:
    _set(data.qvel, _span(model, name, 6, 3, model.jnt_dofadr), name, value)


def install() -> None:
    """Idempotently replace the upstream accessors (module globals are looked up per call)."""
    from gymnasium_robotics.utils import mujoco_utils

    for function in (get_joint_qpos, get_joint_qvel, set_joint_qpos, set_joint_qvel):
        setattr(mujoco_utils, function.__name__, function)
