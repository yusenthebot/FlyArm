"""The grasp scene: the FlyArm Panda plus every object of a split, most of them parked.

mjbatch copies one model for all of its simulations, so a batch that holds a different object
in each environment needs every candidate object in that one model. Each object gets its own
free body. The episode's object is placed on the table; every other object is parked in the
air far behind the robot, each at its own spot, with an external force equal to its weight
(``xfrc_applied``, part of the simulation state, so it is per environment) so it floats
motionless and touches nothing. Nothing else about the model differs between environments.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from flyarm.grasp.objects import GEOMETRY_TOLERANCE, GraspObject
from flyarm.pick_place_env import add_flyarm_gripper, compile_pick_place_model

OBJECT_FRICTION = (1.0, 0.005, 0.0001)  # the rubber pads bring their own friction of 4
# Torsional friction on object contacts. A contact takes the larger condim and friction of its
# two geoms, so pad contacts get the pads' torsional coefficient (0.08): a rubber pad grips over
# an area and resists the object turning about the jaw axis, which point contacts (condim 3)
# cannot, and an object held off its centre of mass would otherwise swing out of the jaws.
OBJECT_CONDIM = 4
# Stiff object contacts. MuJoCo's soft contacts act at the acceleration level, so a light
# object squeezed by the gripper sinks into the jaws: with the default time constant (0.02 s)
# a 30 g mug under the Menagerie gripper's 2-3 N penetrated the fingers by 6-10 mm, the
# contact normals tilted and the squeeze pushed it sideways out of the jaws like wet soap.
# 0.004 s (twice the timestep, the stable limit) cuts that to about a millimetre; the large
# solmix makes the object's time constant win the average with the finger's default.
OBJECT_SOLREF = (0.004, 1.0)
OBJECT_SOLMIX = 1000.0
# Parked objects float behind the video camera, so neither they nor their reflections show.
PARK_ORIGIN = np.array([2.6, -1.05, 0.8])
PARK_SPACING = 0.3  # larger than any object's diagonal, so parked objects never touch
PARK_COLUMNS = 8
CAMERA = "flyarm_grasp_view"
CAMERA_POS = np.array([0.98, 0.42, 0.46])
CAMERA_TARGET = np.array([0.44, 0.0, 0.16])


def park_position(index: int) -> np.ndarray:
    row, column = divmod(index, PARK_COLUMNS)
    return PARK_ORIGIN + np.array([PARK_SPACING * row, PARK_SPACING * column, 0.0])


def look_at(position: np.ndarray, target: np.ndarray) -> list[float]:
    """MJCF ``xyaxes`` of a level camera at ``position`` looking at ``target``."""
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right.tolist(), *up.tolist()]


def body_name(item: GraspObject) -> str:
    return f"grasp_{item.name}"


def geom_name(item: GraspObject) -> str:
    return f"grasp_{item.name}_geom"


def contact_sensor_name(item: GraspObject, side: str) -> str:
    return f"grasp_{item.name}_{side}_contact"


def build_grasp_spec(
    model_path: Path, objects: Sequence[GraspObject], asset_root: Path
) -> mujoco.MjSpec:
    """Panda scene, FlyArm pads and EE site, every object parked, two contact sensors each."""
    if not objects:
        raise ValueError("the grasp scene needs at least one object")
    spec = mujoco.MjSpec.from_file(str(model_path))
    add_flyarm_gripper(spec)
    for index, item in enumerate(objects):
        _add_object(spec, item, index, asset_root)
    spec.worldbody.add_camera(
        name=CAMERA, pos=CAMERA_POS.tolist(), xyaxes=look_at(CAMERA_POS, CAMERA_TARGET), fovy=45
    )
    return spec


def _add_object(spec: mujoco.MjSpec, item: GraspObject, index: int, asset_root: Path) -> None:
    mesh_path, texture_path = item.mesh_path(asset_root), item.texture_path(asset_root)
    for path in (mesh_path, texture_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run scripts/fetch_grasp_objects.py to download the objects"
            )
    mesh = spec.add_mesh()
    mesh.name = f"grasp_{item.name}_mesh"
    mesh.file = str(mesh_path.resolve())
    mesh.scale = [item.scale] * 3
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
    texture = spec.add_texture()
    texture.name = f"grasp_{item.name}_texture"
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.file = str(texture_path.resolve())
    material = spec.add_material()
    material.name = f"grasp_{item.name}_material"
    textures = [""] * len(material.textures)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture.name
    material.textures = textures
    material.specular, material.shininess = 0.3, 0.3

    body = spec.worldbody.add_body(name=body_name(item), pos=park_position(index).tolist())
    body.add_freejoint(name=f"grasp_{item.name}_free")
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), item.yaw)
    body.add_geom(
        name=geom_name(item),
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        material=material.name,
        pos=list(item.offset),
        quat=quat.tolist(),
        mass=item.mass,
        friction=list(OBJECT_FRICTION),
        condim=OBJECT_CONDIM,
        solref=list(OBJECT_SOLREF),
        solmix=OBJECT_SOLMIX,
    )
    for side in ("left", "right"):
        spec.add_sensor(
            name=contact_sensor_name(item, side),
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=geom_name(item),
            reftype=mujoco.mjtObj.mjOBJ_BODY,
            refname=f"{side}_finger",
            intprm=[1, 1, 1],  # data="found", reduce="mindist", num=1
        )


def compile_grasp_model(spec: mujoco.MjSpec, objects: Sequence[GraspObject]) -> mujoco.MjModel:
    """Compile with the pick-place solver settings and check every object's compiled geometry."""
    model = compile_pick_place_model(spec)
    for item in objects:
        check_compiled(model, item)
    return model


def check_compiled(model: mujoco.MjModel, item: GraspObject) -> None:
    """Fail loudly when the compiled hull does not have the manifest's bounding box."""
    geom = model.geom(geom_name(item))
    mesh = int(geom.dataid[0])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, geom.quat)
    vertices = model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T + geom.pos
    low, high = vertices.min(0), vertices.max(0)
    error = max(
        float(np.abs(high - low - np.asarray(item.size)).max()), float(np.abs(low + high).max())
    )
    if error > GEOMETRY_TOLERANCE:
        raise ValueError(
            f"{item.name}: compiled bounding box {np.round(high - low, 4).tolist()} centred at "
            f"{np.round((low + high) / 2, 4).tolist()} does not match the manifest {item.size}; "
            "re-run scripts/fetch_grasp_objects.py --measure after changing a mesh"
        )
