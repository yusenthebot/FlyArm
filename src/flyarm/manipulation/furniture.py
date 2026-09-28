"""Parametric tabletop furniture built from MuJoCo primitives, resized per episode.

Four pieces share every episode: a drawer cabinet with one or two drawers side by side on
slide joints, a lidded cabinet whose door is a hinged lid over an interior shelf, an open bin
and a marked table region. Each piece has its own frame on the floor: the origin is the centre
of its footprint and local +x points out of its front, toward the robot.

Why the cabinet door is a lid. The 5-D action keeps the gripper pointing down. With the hand
over a point 0.62 m in front of the base at shelf height, links 5 and 6 reach past the hand
and up to 0.63 m high (measured on the FlyArm Panda), so reaching into a front-opening cabinet
would need a 0.65 m tall opening. A lid hinged along the back edge keeps the door physically
necessary (the shelf cannot be reached while it is closed) and every motion top-down.

mjbatch copies one model for all simulations, so sizes, handle types and placement change per
episode through per-simulation model fields (``geom_size``, ``geom_pos``, ``body_pos``,
``body_quat``, ``body_ipos``, ``jnt_range``, ``geom_contype``, ``geom_conaffinity``,
``geom_rgba``). MuJoCo's bounding volumes are compile-time, so the model is compiled at the
largest configuration of every range and mid-phase collision is disabled: the compile-time
body bounds then contain every smaller configuration and the broad phase stays exact.
``tests/test_manipulation_env.py`` checks that containment over random configurations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import mujoco
import numpy as np

from flyarm.grasp.scene import OBJECT_SOLMIX as STIFF_SOLMIX
from flyarm.grasp.scene import OBJECT_SOLREF as STIFF_SOLREF

WALL = 0.012  # carcass wall and cabinet wall thickness
PANEL = 0.012  # drawer front panel thickness
GAP = 0.002  # clearance between a drawer and its bay
TRAY_FLOOR = 0.006
TRAY_WALL = 0.006
TRAY_WALL_HEIGHT = 0.035
TRAY_CLEARANCE = 0.006  # between a closed drawer's back and the carcass back wall
TRAVEL_FRACTION = 0.85  # how far a drawer slides out, as a fraction of its tray depth
LID_THICKNESS = 0.01
LID_OPEN_LIMIT = 1.66  # 95 degrees: past vertical, so an opened lid rests on its stop
BIN_FLOOR = 0.008
BIN_WALL = 0.008
BIN_WALL_HEIGHT = 0.035
REGION_THICKNESS = 0.001
# The Panda hand is 21 cm wide along its jaw axis, so a drawer handle is pinched with the jaws
# closing parallel to the drawer front (closing along the pull would put the palm over the
# carcass): the drawer bar is a vertical D-pull. The lid handle sits at the lid's front edge and
# must let the lid turn under a hand that stays vertical: its bar runs along the hinge on a
# swivel and is pinched front to back; its knob is a sphere pinched along the hinge direction,
# which turns freely between point contacts (handle geoms keep condim 3).
DRAWER_STANDOFF = 0.045
LID_STANDOFF = 0.05
BAR_RADIUS = 0.008
BAR_LENGTH = 0.04  # between the posts
LID_BAR_LENGTH = 0.07
STEM_RADIUS = 0.006
KNOB_RADIUS = 0.019  # large enough that the gripper, whose force grows with its opening, holds it
DRAWER_MASS = 0.25
LID_MASS = 0.12
HANDLE_TYPES = ("bar", "knob")
PIECES = ("drawer_cabinet", "cabinet", "bin", "region")

WOOD = (0.72, 0.55, 0.38, 1.0)
DRAWER_WOOD = (0.84, 0.70, 0.52, 1.0)
CABINET_PAINT = (0.36, 0.52, 0.62, 1.0)
LID_PAINT = (0.45, 0.62, 0.72, 1.0)
METAL = (0.78, 0.78, 0.80, 1.0)
BIN_PLASTIC = (0.30, 0.34, 0.40, 1.0)
REGION_COLOUR = (0.20, 0.75, 0.35, 0.55)


@dataclass(frozen=True)
class FurnitureConfig:
    """One episode's furniture. Lengths in metres, angles in radians, poses as (x, y, yaw)."""

    drawer_count: int
    drawer_width: float  # interior width of one drawer bay
    drawer_depth: float  # outer depth of the drawer cabinet
    drawer_height: float  # interior height of a drawer bay
    drawer_handle: int  # index into HANDLE_TYPES
    drawer_handle_height: float  # handle height as a fraction of the front panel
    cabinet_width: float
    cabinet_depth: float
    shelf_height: float  # height of the interior shelf above the floor
    interior_height: float  # from the shelf to the underside of the closed lid
    lid_handle: int
    lid_handle_offset: float  # how far the lid's grip sticks out beyond the lid's front edge
    bin_width: float
    bin_depth: float
    region_size: float
    drawer_pose: tuple[float, float, float]
    cabinet_pose: tuple[float, float, float]
    bin_pose: tuple[float, float, float]
    region_pose: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.drawer_count not in (1, 2):
            raise ValueError("drawer_count must be 1 or 2")
        if self.drawer_handle not in (0, 1) or self.lid_handle not in (0, 1):
            raise ValueError("handle types are 0 (bar) or 1 (knob)")

    # Derived geometry ------------------------------------------------------------------------
    @property
    def drawer_cabinet_width(self) -> float:
        return self.drawer_count * self.drawer_width + (self.drawer_count + 1) * WALL

    @property
    def drawer_cabinet_height(self) -> float:
        return self.drawer_height + 2 * WALL

    @property
    def tray_depth(self) -> float:
        return self.drawer_depth - WALL - PANEL - TRAY_CLEARANCE

    @property
    def drawer_travel(self) -> float:
        return TRAVEL_FRACTION * self.tray_depth

    @property
    def cabinet_height(self) -> float:
        return self.shelf_height + self.interior_height

    def drawer_y(self, index: int) -> float:
        """Centre of drawer bay ``index`` along the cabinet's local y."""
        width = self.drawer_cabinet_width
        return -width / 2 + WALL + self.drawer_width / 2 + index * (self.drawer_width + WALL)

    def as_vector(self) -> np.ndarray:
        values: list[float] = []
        for field in fields(self):
            value = getattr(self, field.name)
            values.extend(value if isinstance(value, tuple) else [value])
        return np.asarray(values, dtype=np.float64)


def largest(configs: list[FurnitureConfig]) -> FurnitureConfig:
    """Field-wise maximum (two drawers, both handle types) for compiling the model."""
    base = configs[0]
    updates: dict[str, Any] = {}
    for field in fields(base):
        values = [getattr(config, field.name) for config in configs]
        if isinstance(values[0], tuple):
            updates[field.name] = values[0]
        else:
            updates[field.name] = max(values)
    updates["drawer_count"] = 2
    return replace(base, **updates)


# ----------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class GeomLayout:
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    enabled: bool = True


@dataclass(frozen=True)
class Layout:
    """Everything a configuration sets on the model, by element name."""

    geoms: dict[str, GeomLayout]
    bodies: dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (pos, quat)
    body_ipos: dict[str, np.ndarray]
    joint_ranges: dict[str, tuple[float, float]]
    handles: dict[str, np.ndarray]  # grasp point of each handle in its moving body's frame


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _box(pos: tuple[float, float, float], half: tuple[float, float, float]) -> GeomLayout:
    return GeomLayout(pos=pos, size=half)


def layout(config: FurnitureConfig) -> Layout:
    """Geom positions and sizes, body poses and joint ranges of one configuration."""
    geoms: dict[str, GeomLayout] = {}
    bodies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ipos: dict[str, np.ndarray] = {}
    ranges: dict[str, tuple[float, float]] = {}
    handles: dict[str, np.ndarray] = {}

    # Drawer cabinet --------------------------------------------------------------------------
    depth, bay, width = config.drawer_depth, config.drawer_height, config.drawer_cabinet_width
    height = config.drawer_cabinet_height
    half_depth = depth / 2
    geoms["dc_bottom"] = _box((0.0, 0.0, WALL / 2), (half_depth, width / 2, WALL / 2))
    geoms["dc_top"] = _box((0.0, 0.0, height - WALL / 2), (half_depth, width / 2, WALL / 2))
    for side, sign in (("left", 1.0), ("right", -1.0)):
        geoms[f"dc_{side}"] = _box(
            (0.0, sign * (width / 2 - WALL / 2), WALL + bay / 2), (half_depth, WALL / 2, bay / 2)
        )
    geoms["dc_back"] = _box(
        (-half_depth + WALL / 2, 0.0, WALL + bay / 2), (WALL / 2, width / 2 - WALL, bay / 2)
    )
    geoms["dc_divider"] = GeomLayout(
        pos=(0.0, config.drawer_y(0) + config.drawer_width / 2 + WALL / 2, WALL + bay / 2),
        size=(half_depth - WALL, WALL / 2, bay / 2),
        enabled=config.drawer_count == 2,
    )
    x, y, yaw = config.drawer_pose
    bodies["drawer_cabinet"] = (np.array([x, y, 0.0]), yaw_quat(yaw))
    tray = config.tray_depth
    handle_z = config.drawer_handle_height * bay
    bar = config.drawer_handle == 0
    for index in range(2):
        enabled = index < config.drawer_count
        name = f"drawer_{index}"
        # The drawer hangs on its slide GAP above the carcass floor, so it touches nothing.
        bodies[name] = (
            np.array([half_depth, config.drawer_y(index), WALL + GAP]),
            yaw_quat(0.0),
        )
        ranges[f"{name}_slide"] = (0.0, config.drawer_travel)
        ipos[name] = np.array([-PANEL - tray / 2, 0.0, bay / 4])
        inner = config.drawer_width / 2 - GAP
        panel_half = bay / 2 - GAP
        parts = {
            "panel": _box((-PANEL / 2, 0.0, panel_half), (PANEL / 2, inner, panel_half)),
            "floor": _box(
                (-PANEL - tray / 2, 0.0, TRAY_FLOOR / 2), (tray / 2, inner, TRAY_FLOOR / 2)
            ),
            "back": _box(
                (-PANEL - tray + TRAY_WALL / 2, 0.0, TRAY_FLOOR + TRAY_WALL_HEIGHT / 2),
                (TRAY_WALL / 2, inner, TRAY_WALL_HEIGHT / 2),
            ),
        }
        for side, sign in (("left", 1.0), ("right", -1.0)):
            parts[f"wall_{side}"] = _box(
                (
                    -PANEL - tray / 2,
                    sign * (inner - TRAY_WALL / 2),
                    TRAY_FLOOR + TRAY_WALL_HEIGHT / 2,
                ),
                (tray / 2, TRAY_WALL / 2, TRAY_WALL_HEIGHT / 2),
            )
        grip = (DRAWER_STANDOFF, 0.0, handle_z)
        handles[name] = np.array(grip)
        for side, sign in (("low", -1.0), ("high", 1.0)):
            parts[f"post_{side}"] = GeomLayout(
                (DRAWER_STANDOFF / 2, 0.0, handle_z + sign * BAR_LENGTH / 2),
                (STEM_RADIUS, DRAWER_STANDOFF / 2, 0.0),
                bar,
            )
        parts["bar"] = GeomLayout(grip, (BAR_RADIUS, BAR_LENGTH / 2 + STEM_RADIUS, 0.0), bar)
        parts["stem"] = GeomLayout(
            (DRAWER_STANDOFF / 2, 0.0, handle_z), (STEM_RADIUS, DRAWER_STANDOFF / 2, 0.0), not bar
        )
        parts["knob"] = GeomLayout(grip, (KNOB_RADIUS, 0.0, 0.0), not bar)
        for part, geom in parts.items():
            geoms[f"{name}_{part}"] = replace(geom, enabled=geom.enabled and enabled)

    # Lidded cabinet --------------------------------------------------------------------------
    cab_w, cab_d = config.cabinet_width, config.cabinet_depth
    shelf, cab_h = config.shelf_height, config.cabinet_height
    inner_h = config.interior_height
    geoms["cab_shelf"] = _box((0.0, 0.0, shelf / 2), (cab_d / 2, cab_w / 2, shelf / 2))
    for side, sign in (("front", 1.0), ("back", -1.0)):
        geoms[f"cab_{side}"] = _box(
            (sign * (cab_d / 2 - WALL / 2), 0.0, shelf + inner_h / 2),
            (WALL / 2, cab_w / 2, inner_h / 2),
        )
    for side, sign in (("left", 1.0), ("right", -1.0)):
        geoms[f"cab_{side}"] = _box(
            (0.0, sign * (cab_w / 2 - WALL / 2), shelf + inner_h / 2),
            (cab_d / 2 - WALL, WALL / 2, inner_h / 2),
        )
    x, y, yaw = config.cabinet_pose
    bodies["cabinet"] = (np.array([x, y, 0.0]), yaw_quat(yaw))
    bodies["lid"] = (np.array([-cab_d / 2, 0.0, cab_h]), yaw_quat(0.0))
    ranges["lid_hinge"] = (0.0, LID_OPEN_LIMIT)
    ipos["lid"] = np.array([cab_d / 2, 0.0, LID_THICKNESS / 2])
    geoms["lid_plate"] = _box(
        (cab_d / 2, 0.0, LID_THICKNESS / 2), (cab_d / 2, cab_w / 2, LID_THICKNESS / 2)
    )
    lid_bar = config.lid_handle == 0
    # The grip sticks out forward from the lid's front edge, so once the lid stands open the
    # grip points up above its top edge and the hand holding it is above the lid, not behind it.
    reach = config.lid_handle_offset
    grip = np.array([cab_d + reach, 0.0, LID_THICKNESS / 2])
    handles["lid"] = grip
    for side, sign in (("left", 1.0), ("right", -1.0)):
        geoms[f"lid_post_{side}"] = GeomLayout(
            (cab_d + reach / 2, sign * LID_BAR_LENGTH / 2, LID_THICKNESS / 2),
            (STEM_RADIUS, reach / 2, 0.0),
            lid_bar,
        )
    geoms["lid_stem"] = GeomLayout(
        (cab_d + reach / 2 - KNOB_RADIUS / 2, 0.0, LID_THICKNESS / 2),
        (STEM_RADIUS, reach / 2 - KNOB_RADIUS / 2, 0.0),
        not lid_bar,
    )
    # The grip swivels about the hinge's direction, like a bar on bearings, so the fingers keep
    # hold of it while the lid turns under the hand.
    bodies["lid_handle"] = (grip, yaw_quat(0.0))
    geoms["lid_handle_bar"] = GeomLayout(
        (0.0, 0.0, 0.0), (BAR_RADIUS, LID_BAR_LENGTH / 2 + STEM_RADIUS, 0.0), lid_bar
    )
    geoms["lid_handle_knob"] = GeomLayout((0.0, 0.0, 0.0), (KNOB_RADIUS, 0.0, 0.0), not lid_bar)

    # Bin and region ---------------------------------------------------------------------------
    bin_w, bin_d = config.bin_width, config.bin_depth
    geoms["bin_floor"] = _box((0.0, 0.0, BIN_FLOOR / 2), (bin_d / 2, bin_w / 2, BIN_FLOOR / 2))
    wall_z = BIN_FLOOR + BIN_WALL_HEIGHT / 2
    for side, sign in (("front", 1.0), ("back", -1.0)):
        geoms[f"bin_{side}"] = _box(
            (sign * (bin_d / 2 - BIN_WALL / 2), 0.0, wall_z),
            (BIN_WALL / 2, bin_w / 2, BIN_WALL_HEIGHT / 2),
        )
    for side, sign in (("left", 1.0), ("right", -1.0)):
        geoms[f"bin_{side}"] = _box(
            (0.0, sign * (bin_w / 2 - BIN_WALL / 2), wall_z),
            (bin_d / 2 - BIN_WALL, BIN_WALL / 2, BIN_WALL_HEIGHT / 2),
        )
    x, y, yaw = config.bin_pose
    bodies["bin"] = (np.array([x, y, 0.0]), yaw_quat(yaw))
    size = config.region_size
    geoms["region_mark"] = _box(
        (0.0, 0.0, REGION_THICKNESS / 2), (size / 2, size / 2, REGION_THICKNESS / 2)
    )
    x, y, yaw = config.region_pose
    bodies["region"] = (np.array([x, y, 0.0]), yaw_quat(yaw))
    return Layout(geoms=geoms, bodies=bodies, body_ipos=ipos, joint_ranges=ranges, handles=handles)


# ----------------------------------------------------------------------------- spec building

_CYLINDER_ALONG = {
    "x": (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
    "y": (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0),
    "z": (1.0, 0.0, 0.0, 0.0),
}


def _geom_kind(name: str) -> tuple[Any, str, tuple[float, float, float, float]]:
    # Rods (handle bars, posts, stems) are capsules: MuJoCo 3.13's convex collider segfaulted
    # in EPA on a finger pad pressed 8 mm into a cylinder bar, at the same state at 35 and at
    # 50 iterations (research log E60), while capsule against box has an analytic collider.
    # rod_size() keeps a rod's end-to-end length, so only its end faces become round.
    box, cylinder, sphere = (
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_SPHERE,
    )
    if name.startswith("drawer_"):
        if name.endswith(("post_low", "post_high", "stem")):
            return cylinder, "x", METAL
        if name.endswith("bar"):
            return cylinder, "z", METAL
        if name.endswith("knob"):
            return sphere, "z", METAL
        return box, "z", DRAWER_WOOD
    if name.startswith("lid_"):
        if name in ("lid_post_left", "lid_post_right", "lid_stem"):
            return cylinder, "x", METAL
        if name == "lid_handle_bar":
            return cylinder, "y", METAL
        if name == "lid_handle_knob":
            return sphere, "z", METAL
        return box, "z", LID_PAINT
    if name.startswith("dc_"):
        return box, "z", WOOD
    if name.startswith("cab_"):
        return box, "z", CABINET_PAINT
    if name.startswith("bin_"):
        return box, "z", BIN_PLASTIC
    return box, "z", REGION_COLOUR


def rod_size(name: str, size: Sequence[float]) -> list[float]:
    """The MuJoCo size of a planned geom: a rod's half-length loses its end caps' radius."""
    if _geom_kind(name)[0] != mujoco.mjtGeom.mjGEOM_CAPSULE:
        return list(size)
    radius, half = float(size[0]), float(size[1])
    return [radius, max(half - radius, 0.1 * half), *map(float, size[2:])]


def _geom_body(name: str) -> str:
    if name.startswith("dc_"):
        return "drawer_cabinet"
    if name.startswith("drawer_"):
        return name[: len("drawer_0")]
    if name.startswith("lid_handle_"):
        return "lid_handle"
    if name.startswith("lid_"):
        return "lid"
    if name.startswith("cab_"):
        return "cabinet"
    if name.startswith("bin_"):
        return "bin"
    return "region"


def add_furniture(spec: mujoco.MjSpec, compile_config: FurnitureConfig) -> None:
    """Add every furniture body, joint and geom, sized by ``compile_config`` (the largest)."""
    plan = layout(compile_config)
    world = spec.worldbody
    made: dict[str, Any] = {}
    for root in ("drawer_cabinet", "cabinet", "bin", "region"):
        pos, quat = plan.bodies[root]
        made[root] = world.add_body(name=root, pos=pos.tolist(), quat=quat.tolist())
    for index in range(2):
        name = f"drawer_{index}"
        pos, quat = plan.bodies[name]
        body = made["drawer_cabinet"].add_body(name=name, pos=pos.tolist(), quat=quat.tolist())
        body.explicitinertial = True
        body.mass = DRAWER_MASS
        body.ipos = plan.body_ipos[name].tolist()
        body.inertia = [2e-3, 2e-3, 3e-3]
        body.add_joint(
            name=f"{name}_slide",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1.0, 0.0, 0.0],
            range=list(plan.joint_ranges[f"{name}_slide"]),
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            damping=4.0,
            frictionloss=0.3,
            armature=0.05,
        )
        made[name] = body
    pos, quat = plan.bodies["lid"]
    lid = made["cabinet"].add_body(name="lid", pos=pos.tolist(), quat=quat.tolist())
    lid.explicitinertial = True
    lid.mass = LID_MASS
    lid.ipos = plan.body_ipos["lid"].tolist()
    lid.inertia = [1e-3, 5e-4, 1.4e-3]
    lid.add_joint(
        name="lid_hinge",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, -1.0, 0.0],
        range=list(plan.joint_ranges["lid_hinge"]),
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        damping=0.5,  # slow enough that a lid released past vertical settles on its stop
        armature=0.002,
    )
    made["lid"] = lid
    pos, quat = plan.bodies["lid_handle"]
    handle = lid.add_body(name="lid_handle", pos=pos.tolist(), quat=quat.tolist())
    handle.explicitinertial = True
    handle.mass = 0.03
    handle.inertia = [1e-5, 1e-5, 1e-5]
    handle.add_joint(
        name="lid_handle_swivel",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 1.0, 0.0],
        damping=0.0005,
    )
    made["lid_handle"] = handle
    for name, geom in plan.geoms.items():
        kind, axis, rgba = _geom_kind(name)
        visual_only = name == "region_mark"
        made[_geom_body(name)].add_geom(
            name=name,
            type=kind,
            size=rod_size(name, geom.size),
            pos=list(geom.pos),
            quat=list(_CYLINDER_ALONG[axis]),
            rgba=list(rgba),
            contype=0 if visual_only else 1,
            conaffinity=0 if visual_only else 1,
            density=0.0,
            group=2 if visual_only else 0,
            # Stiff contacts, as for the objects (flyarm.grasp.scene): a light drawer or lid
            # handle pinched by the gripper would otherwise sink into the pads and slip.
            solref=list(STIFF_SOLREF),
            solmix=STIFF_SOLMIX,
        )


# ----------------------------------------------------------------------------- runtime binding


class FurnitureFields:
    """Writes a configuration into per-environment model fields (and reads them back)."""

    def __init__(self, model: mujoco.MjModel) -> None:
        plan_names = layout(_REFERENCE).geoms
        self.geom_ids = {name: model.geom(name).id for name in plan_names}
        self.body_ids = {
            name: model.body(name).id
            for name in (*PIECES, "drawer_0", "drawer_1", "lid", "lid_handle")
        }
        self.joint_ids = {
            name: model.joint(name).id for name in ("drawer_0_slide", "drawer_1_slide", "lid_hinge")
        }
        self.visual = {name: model.geom_rgba[gid].copy() for name, gid in self.geom_ids.items()}
        self.solid = {
            name: (int(model.geom_contype[gid]), int(model.geom_conaffinity[gid]))
            for name, gid in self.geom_ids.items()
        }

    def apply(self, fields_: dict[str, np.ndarray], row: int, config: FurnitureConfig) -> None:
        plan = layout(config)
        for name, geom in plan.geoms.items():
            gid = self.geom_ids[name]
            fields_["geom_pos"][row, gid] = geom.pos
            fields_["geom_size"][row, gid] = rod_size(name, geom.size)
            contype, conaffinity = self.solid[name] if geom.enabled else (0, 0)
            fields_["geom_contype"][row, gid] = contype
            fields_["geom_conaffinity"][row, gid] = conaffinity
            rgba = self.visual[name].copy()
            rgba[3] = rgba[3] if geom.enabled else 0.0
            fields_["geom_rgba"][row, gid] = rgba
        for name, (pos, quat) in plan.bodies.items():
            bid = self.body_ids[name]
            fields_["body_pos"][row, bid] = pos
            fields_["body_quat"][row, bid] = quat
        for name, position in plan.body_ipos.items():
            fields_["body_ipos"][row, self.body_ids[name]] = position
        for name, bounds in plan.joint_ranges.items():
            fields_["jnt_range"][row, self.joint_ids[name]] = bounds


MODEL_FIELDS = (
    "geom_pos",
    "geom_size",
    "geom_contype",
    "geom_conaffinity",
    "geom_rgba",
    "body_pos",
    "body_quat",
    "body_ipos",
    "jnt_range",
)

_REFERENCE = FurnitureConfig(
    drawer_count=2,
    drawer_width=0.15,
    drawer_depth=0.2,
    drawer_height=0.1,
    drawer_handle=0,
    drawer_handle_height=0.5,
    cabinet_width=0.28,
    cabinet_depth=0.2,
    shelf_height=0.05,
    interior_height=0.12,
    lid_handle=0,
    lid_handle_offset=0.06,
    bin_width=0.18,
    bin_depth=0.18,
    region_size=0.12,
    drawer_pose=(0.2, 0.45, -2.0),
    cabinet_pose=(0.2, -0.5, 2.0),
    bin_pose=(0.65, 0.0, math.pi),
    region_pose=(0.45, 0.15, 0.0),
)


# ----------------------------------------------------------------------------- sampling


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def draw(self, generator: np.random.Generator) -> float:
        return float(generator.uniform(self.low, self.high))


@dataclass(frozen=True)
class FurnitureRanges:
    """Where every furniture factor is drawn from; train and held-out ranges do not overlap."""

    drawer_width: Range
    drawer_depth: Range
    drawer_height: Range
    drawer_handles: tuple[int, ...]
    drawer_handle_height: Range
    cabinet_width: Range
    cabinet_depth: Range
    shelf_height: Range
    interior_height: Range
    lid_handles: tuple[int, ...]
    side_azimuth: Range  # degrees off the robot's forward axis, for both cabinets
    drawer_front_radius: Range  # distance from the robot base to the drawer cabinet's front
    cabinet_front_radius: Range
    facing_jitter: Range  # |degrees| the cabinets turn away from facing the robot
    bin_size: Range


# Held-out ranges sit beside the training ones, on whichever side keeps the task reachable: an
# opened lid's grip rises to about shelf + interior + depth + 6 cm, and the arm cannot hold a
# vertical hand much above 0.47 m at 0.6 m out, so held-out cabinets are shallower and nearer.
TRAIN_RANGES = FurnitureRanges(
    drawer_width=Range(0.13, 0.16),
    drawer_depth=Range(0.18, 0.22),
    drawer_height=Range(0.09, 0.11),
    drawer_handles=(0, 1),
    drawer_handle_height=Range(0.35, 0.6),
    cabinet_width=Range(0.26, 0.30),
    cabinet_depth=Range(0.17, 0.20),
    shelf_height=Range(0.025, 0.045),
    interior_height=Range(0.105, 0.12),
    lid_handles=(0,),
    side_azimuth=Range(50.0, 62.0),
    drawer_front_radius=Range(0.55, 0.60),
    cabinet_front_radius=Range(0.40, 0.44),
    facing_jitter=Range(0.0, 10.0),
    bin_size=Range(0.16, 0.20),
)
HELD_OUT_RANGES = FurnitureRanges(
    drawer_width=Range(0.16, 0.18),
    drawer_depth=Range(0.22, 0.24),
    drawer_height=Range(0.11, 0.12),
    drawer_handles=(0, 1),
    drawer_handle_height=Range(0.6, 0.7),
    cabinet_width=Range(0.30, 0.32),
    cabinet_depth=Range(0.16, 0.17),
    shelf_height=Range(0.015, 0.025),
    interior_height=Range(0.12, 0.13),
    lid_handles=(1,),
    side_azimuth=Range(62.0, 68.0),
    drawer_front_radius=Range(0.60, 0.63),
    cabinet_front_radius=Range(0.37, 0.40),
    facing_jitter=Range(10.0, 16.0),
    bin_size=Range(0.20, 0.22),
)
REGION_SIZE = Range(0.11, 0.13)


def sample_furniture(generator: np.random.Generator, ranges: FurnitureRanges) -> FurnitureConfig:
    """One episode's furniture: sizes, handles and a placement facing the robot."""
    count = int(generator.integers(1, 3))
    drawer_width = ranges.drawer_width.draw(generator)
    drawer_depth = ranges.drawer_depth.draw(generator)
    drawer_height = ranges.drawer_height.draw(generator)
    drawer_handle = int(generator.choice(ranges.drawer_handles))
    handle_height = ranges.drawer_handle_height.draw(generator)
    cabinet_width = ranges.cabinet_width.draw(generator)
    cabinet_depth = ranges.cabinet_depth.draw(generator)
    shelf = ranges.shelf_height.draw(generator)
    interior = ranges.interior_height.draw(generator)
    lid_handle = int(generator.choice(ranges.lid_handles))
    lid_offset = float(generator.uniform(0.055, 0.065))
    side = 1.0 if generator.random() < 0.5 else -1.0

    def cabinet_pose(
        azimuth_sign: float, radius: Range, depth: float
    ) -> tuple[float, float, float]:
        azimuth = math.radians(azimuth_sign * ranges.side_azimuth.draw(generator))
        centre = radius.draw(generator) + depth / 2
        jitter = math.radians(ranges.facing_jitter.draw(generator))
        jitter *= 1.0 if generator.random() < 0.5 else -1.0
        return (
            centre * math.cos(azimuth),
            centre * math.sin(azimuth),
            azimuth + math.pi + jitter,
        )

    drawer_pose = cabinet_pose(side, ranges.drawer_front_radius, drawer_depth)
    cab_pose = cabinet_pose(-side, ranges.cabinet_front_radius, cabinet_depth)
    bin_w, bin_d = ranges.bin_size.draw(generator), ranges.bin_size.draw(generator)
    bin_azimuth = math.radians(generator.uniform(-8.0, 8.0))
    bin_radius = float(generator.uniform(0.60, 0.64)) + bin_d / 2
    bin_pose = (
        bin_radius * math.cos(bin_azimuth),
        bin_radius * math.sin(bin_azimuth),
        math.pi + math.radians(generator.uniform(-15.0, 15.0)),
    )
    region = REGION_SIZE.draw(generator)
    region_azimuth = math.radians(side * generator.uniform(18.0, 30.0))
    region_radius = float(generator.uniform(0.46, 0.52))
    region_pose = (
        region_radius * math.cos(region_azimuth),
        region_radius * math.sin(region_azimuth),
        region_azimuth,
    )
    return FurnitureConfig(
        drawer_count=count,
        drawer_width=drawer_width,
        drawer_depth=drawer_depth,
        drawer_height=drawer_height,
        drawer_handle=drawer_handle,
        drawer_handle_height=handle_height,
        cabinet_width=cabinet_width,
        cabinet_depth=cabinet_depth,
        shelf_height=shelf,
        interior_height=interior,
        lid_handle=lid_handle,
        lid_handle_offset=lid_offset,
        bin_width=bin_w,
        bin_depth=bin_d,
        region_size=region,
        drawer_pose=drawer_pose,
        cabinet_pose=cab_pose,
        bin_pose=bin_pose,
        region_pose=region_pose,
    )


def compile_config() -> FurnitureConfig:
    """The largest configuration over train and held-out ranges; the model is built at it."""
    corners = []
    for ranges in (TRAIN_RANGES, HELD_OUT_RANGES):
        corners.append(
            replace(
                _REFERENCE,
                drawer_width=ranges.drawer_width.high,
                drawer_depth=ranges.drawer_depth.high,
                drawer_height=ranges.drawer_height.high,
                drawer_handle_height=ranges.drawer_handle_height.high,
                cabinet_width=ranges.cabinet_width.high,
                cabinet_depth=ranges.cabinet_depth.high,
                shelf_height=ranges.shelf_height.high,
                interior_height=ranges.interior_height.high,
                lid_handle_offset=0.065,
                bin_width=ranges.bin_size.high,
                bin_depth=ranges.bin_size.high,
                region_size=REGION_SIZE.high,
            )
        )
    return largest(corners)
