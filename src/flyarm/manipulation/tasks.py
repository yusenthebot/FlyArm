"""Skills, subgoals, task templates and episode sampling for the articulated scene.

A task is a list of subgoals, each one skill with its arguments:

- ``open_drawer(d)`` / ``close_drawer(d)``: drawer ``d`` slid out past 75% of its travel /
  within 4 mm of its stop;
- ``open_door`` / ``close_door``: the cabinet lid past vertical and resting on its stop /
  within 0.02 rad (about 4 mm at the handle) of closed;
- ``pick(a)``: object ``a`` held by both fingers and out of the receptacle it started in (or
  3 cm up when it started on the table);
- ``place(a, r)``: object ``a`` wholly inside receptacle ``r`` (a drawer's interior, the
  cabinet shelf, the bin, the table region), at rest relative to it and untouched, for 10 steps;
- ``stack(a, b)``: object ``a`` resting on top of object ``b``, both at rest, ``a`` untouched,
  for 20 steps.

Progress is a pure function of the scene, so a controller without memory can be told what to
do next. Subgoal ``k`` counts as done when its effect holds now, or when every later subgoal
that consumed it is done: an ``open`` is consumed by the picks and places that needed the
opening, up to the matching ``close``, and a ``pick`` by the place or stack of the same object.
Closing a drawer after putting an object in it therefore keeps the ``open`` done. The current
subgoal is the first one not done; the cue names it.

Task templates are written with roles (objects A, B, C, E and a drawer D) and bound per episode
to concrete slots, a drawer and objects whose geometry fits the receptacles of that episode's
furniture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from flyarm.grasp.objects import GraspObject
from flyarm.manipulation import furniture as fu

SKILLS = (
    "none",
    "open_drawer",
    "close_drawer",
    "open_door",
    "close_door",
    "pick",
    "place",
    "stack",
)
NONE, OPEN_DRAWER, CLOSE_DRAWER, OPEN_DOOR, CLOSE_DOOR, PICK, PLACE, STACK = range(len(SKILLS))
RECEPTACLES = ("drawer_0", "drawer_1", "shelf", "bin", "region")
DRAWER_0, DRAWER_1, SHELF, BIN, REGION = range(len(RECEPTACLES))
NO_RECEPTACLE = -1
MAX_OBJECTS = 4
MIN_OBJECTS = 2  # a scene always has a second object, a distractor if the task needs one
MAX_SUBGOALS = 8
STACKABLE_FAMILIES = ("box", "can", "bowl")
STACK_ASPECT = 1.5  # a stacked object's height over its narrow width, at most

# Success thresholds (see the module docstring).
DRAWER_OPEN_FRACTION = 0.75
DRAWER_CLOSED = 0.004
LID_OPEN = 1.58
LID_CLOSED = 0.02
PICK_LIFT = 0.03
PLACE_STEPS = 10
STACK_STEPS = 20
REST_MOVE = 0.002  # m per control step (4 cm/s) relative to the receptacle
REST_TURN = 0.03  # rad per control step
INSIDE_TOLERANCE = 0.004
STACK_GAP = 0.012  # the top object's lowest point within this of the base object's highest
HORIZON_BASE = 200
HORIZON_PER_SUBGOAL = 300


@dataclass(frozen=True)
class Subgoal:
    kind: int
    obj: int = -1  # object slot
    target: int = -1  # drawer index, receptacle id or base object slot
    spot: int = 0  # place position within a receptacle holding several objects

    def describe(self, names: Sequence[str] | None = None) -> str:
        def slot(index: int) -> str:
            return names[index] if names is not None else f"slot {index}"

        skill = SKILLS[self.kind]
        if self.kind in (OPEN_DRAWER, CLOSE_DRAWER):
            return f"{skill}({self.target})"
        if self.kind in (OPEN_DOOR, CLOSE_DOOR):
            return skill
        if self.kind == PICK:
            return f"pick({slot(self.obj)})"
        if self.kind == PLACE:
            return f"place({slot(self.obj)}, {RECEPTACLES[self.target]})"
        return f"stack({slot(self.obj)} on {slot(self.target)})"


# ----------------------------------------------------------------------------- templates

# A template step: (skill, object role, target). Targets: "D" (the drawer), a receptacle name,
# or an object role for stacks. Object roles start on the table unless listed in ``in_drawer``.
Step = tuple[str, str, str]


@dataclass(frozen=True)
class Template:
    name: str
    steps: tuple[Step, ...]
    in_drawer: tuple[str, ...] = ()
    distractors: tuple[int, int] = (0, 1)  # extra objects on the table, inclusive range

    @property
    def roles(self) -> tuple[str, ...]:
        seen: list[str] = []
        for _, role, target in self.steps:
            for name in (role, target):
                if name in ("A", "B", "C", "E") and name not in seen:
                    seen.append(name)
        return tuple(seen)

    def signature(self) -> tuple[str, ...]:
        """The skill composition: skill and receptacle kind of every step, roles erased."""
        out = []
        for skill, _, target in self.steps:
            kind = "drawer" if target == "D" else ("object" if target in "ABCE" else target)
            out.append(f"{skill}:{kind}" if target else skill)
        return tuple(out)


def _t(name: str, steps: list[Step], **kwargs) -> Template:
    return Template(name=name, steps=tuple(steps), **kwargs)


TEMPLATES: dict[str, Template] = {
    template.name: template
    for template in (
        _t("put_away", [("open_drawer", "", "D"), ("place", "A", "D"), ("close_drawer", "", "D")]),
        _t(
            "retrieve",
            [
                ("open_drawer", "", "D"),
                ("pick", "A", ""),
                ("place", "A", "bin"),
                ("close_drawer", "", "D"),
            ],
            in_drawer=("A",),
        ),
        _t(
            "shelve",
            [
                ("open_door", "", ""),
                ("place", "A", "shelf"),
                ("place", "B", "shelf"),
                ("close_door", "", ""),
            ],
        ),
        _t(
            "tower",
            [("place", "A", "region"), ("stack", "B", "A"), ("stack", "C", "B")],
            distractors=(0, 0),
        ),
        _t(
            "sort",
            [("place", "A", "bin"), ("place", "B", "region"), ("stack", "C", "B")],
            distractors=(0, 0),
        ),
        _t(
            "tidy",
            [
                ("open_drawer", "", "D"),
                ("place", "A", "D"),
                ("close_drawer", "", "D"),
                ("open_door", "", ""),
                ("place", "B", "shelf"),
                ("close_door", "", ""),
            ],
        ),
        _t(
            "unpack",
            [
                ("open_drawer", "", "D"),
                ("pick", "A", ""),
                ("place", "A", "region"),
                ("close_drawer", "", "D"),
                ("open_door", "", ""),
                ("place", "B", "shelf"),
                ("close_door", "", ""),
            ],
            in_drawer=("A",),
        ),
        # Held-out compositions: every one contains an adjacent pair of steps that no training
        # template has (tests/test_manipulation_tasks.py checks it).
        _t(
            "shelve_then_put_away",
            [
                ("open_door", "", ""),
                ("place", "A", "shelf"),
                ("close_door", "", ""),
                ("open_drawer", "", "D"),
                ("place", "B", "D"),
                ("close_drawer", "", "D"),
            ],
        ),
        _t(
            "retrieve_to_shelf",
            [
                ("open_door", "", ""),
                ("open_drawer", "", "D"),
                ("pick", "A", ""),
                ("place", "A", "shelf"),
                ("close_drawer", "", "D"),
                ("close_door", "", ""),
            ],
            in_drawer=("A",),
        ),
        _t(
            "tower_then_put_away",
            [
                ("place", "A", "region"),
                ("stack", "B", "A"),
                ("open_drawer", "", "D"),
                ("place", "C", "D"),
                ("close_drawer", "", "D"),
            ],
            distractors=(0, 0),
        ),
        _t(
            "full_cleanup",
            [
                ("open_drawer", "", "D"),
                ("place", "A", "D"),
                ("close_drawer", "", "D"),
                ("place", "B", "region"),
                ("stack", "C", "B"),
                ("open_door", "", ""),
                ("place", "E", "shelf"),
                ("close_door", "", ""),
            ],
            distractors=(0, 0),
        ),
    )
}
TRAIN_TEMPLATES = ("put_away", "retrieve", "shelve", "tower", "sort", "tidy", "unpack")
HELD_OUT_TEMPLATES = (
    "shelve_then_put_away",
    "retrieve_to_shelf",
    "tower_then_put_away",
    "full_cleanup",
)


def horizon(steps: int) -> int:
    return HORIZON_BASE + HORIZON_PER_SUBGOAL * steps


# ----------------------------------------------------------------------------- receptacle fit

INTERIOR_MARGIN = 0.012  # an object must clear every wall of a receptacle by this much


def drawer_place_x(config: fu.FurnitureConfig) -> float:
    """Place point along the drawer's axis (drawer frame): the middle of the part a drawer
    opened to the success threshold leaves outside the carcass."""
    exposed = DRAWER_OPEN_FRACTION * config.drawer_travel
    return -fu.PANEL - (exposed - fu.PANEL) / 2


def drawer_exposed_length(config: fu.FurnitureConfig) -> float:
    return DRAWER_OPEN_FRACTION * config.drawer_travel - fu.PANEL


def fits_drawer(item: GraspObject, config: fu.FurnitureConfig) -> bool:
    long, narrow, height = item.size
    inner_width = config.drawer_width - 2 * fu.GAP - 2 * fu.TRAY_WALL
    clearance = config.drawer_height - 2 * fu.GAP - fu.TRAY_FLOOR
    return (
        long <= drawer_exposed_length(config) - 2 * INTERIOR_MARGIN
        and narrow <= inner_width - 2 * INTERIOR_MARGIN
        and height <= clearance - INTERIOR_MARGIN
    )


def shelf_spots(config: fu.FurnitureConfig, count: int) -> list[np.ndarray]:
    """Place points on the shelf (cabinet frame), spread along the cabinet's depth."""
    depth = config.cabinet_depth - 2 * fu.WALL
    offsets = [0.0] if count == 1 else [depth / 4, -depth / 4]
    return [np.array([offset, 0.0, config.shelf_height]) for offset in offsets]


def fits_shelf(item: GraspObject, config: fu.FurnitureConfig, count: int) -> bool:
    long, narrow, height = item.size
    depth = config.cabinet_depth - 2 * fu.WALL
    width = config.cabinet_width - 2 * fu.WALL
    return (
        long <= depth / count - 2 * INTERIOR_MARGIN
        and narrow <= width - 2 * INTERIOR_MARGIN
        and height <= config.interior_height - INTERIOR_MARGIN
    )


def fits_bin(item: GraspObject, config: fu.FurnitureConfig) -> bool:
    inner = min(config.bin_width, config.bin_depth) - 2 * fu.BIN_WALL
    return item.size[0] <= inner - 2 * INTERIOR_MARGIN


def fits_region(item: GraspObject, config: fu.FurnitureConfig) -> bool:
    return item.size[0] <= config.region_size - 0.01


def stackable(item: GraspObject) -> bool:
    return item.family in STACKABLE_FAMILIES and item.size[1] >= 0.04 and item.size[2] <= 0.09


def _footprint(item: GraspObject) -> float:
    return item.size[0] * item.size[1]


def stable_on(top: GraspObject, base: GraspObject) -> bool:
    """A stack the scripted teacher can build: the top no bigger, longer or heavier than allows,
    and not tall for its width.

    Measured: a 101 g, 7.9 cm box set on two 30 g bowls toppled the tower, and a 57 g box 7.9 cm
    tall on a 4.3 cm side tipped off a 3 cm bowl as the fingers let go.
    """
    return (
        _footprint(top) <= 1.2 * _footprint(base)
        and top.size[0] <= base.size[0] + 0.015
        and top.mass <= 1.5 * base.mass + 0.02
        and top.size[2] <= STACK_ASPECT * top.size[1]
    )


# ----------------------------------------------------------------------------- episodes


@dataclass(frozen=True)
class Episode:
    """Everything that defines one episode; drawn by :func:`sample_episode`."""

    template: str
    furniture: fu.FurnitureConfig
    subgoals: tuple[Subgoal, ...]
    objects: tuple[int, ...]  # object index (into the scene's object list) per slot, -1 empty
    # Initial pose per slot: (x, y, yaw) on the table in the world frame, or in the drawer's
    # frame when the slot's entry of ``in_drawer`` holds that drawer's index.
    poses: tuple[tuple[float, float, float], ...]
    in_drawer: tuple[int, ...]  # drawer index per slot, -1 when the object starts on the table
    drawer: int  # the drawer the task uses, -1 if none

    @property
    def horizon(self) -> int:
        return horizon(len(self.subgoals))


class NoEpisode(RuntimeError):
    """The sampler could not bind a template to this furniture and object set."""


def _furniture_obstacles(config: fu.FurnitureConfig) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """(centre, yaw, half extents) of every footprint the table objects must avoid."""
    drawer_reach = config.drawer_depth + config.drawer_travel + fu.DRAWER_STANDOFF
    out = []
    x, y, yaw = config.drawer_pose
    # The drawer cabinet's footprint grows forward by a fully opened drawer and its handle.
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    centre = np.array([x, y]) + forward * (drawer_reach - config.drawer_depth) / 2
    out.append((centre, yaw, np.array([drawer_reach / 2, config.drawer_cabinet_width / 2])))
    x, y, yaw = config.cabinet_pose
    out.append(
        (
            np.array([x, y]),
            yaw,
            np.array([config.cabinet_depth / 2 + 0.03, config.cabinet_width / 2]),
        )
    )
    x, y, yaw = config.bin_pose
    out.append((np.array([x, y]), yaw, np.array([config.bin_depth / 2, config.bin_width / 2])))
    x, y, yaw = config.region_pose
    half = config.region_size / 2
    out.append((np.array([x, y]), yaw, np.array([half, half])))
    return out


def _circle_clear(
    point: np.ndarray, radius: float, box: tuple[np.ndarray, float, np.ndarray]
) -> bool:
    centre, yaw, half = box
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[c, s], [-s, c]]) @ (point - centre)
    nearest = np.clip(local, -half, half)
    return float(np.linalg.norm(local - nearest)) > radius


TABLE_RADIUS = (0.36, 0.60)  # nearer than 0.36 m the folded arm cannot turn the hand
TABLE_AZIMUTH = math.radians(48.0)
OBJECT_MARGIN = 0.035
# The open hand reaches about 10 cm either side of the jaws' centre: an object nearer than that
# to a cabinet, the bin or the region (where towers stand) can be ungraspable at the heading its
# shape needs, or the hand knocks what stands there on the way down.
HAND_CLEARANCE = 0.1


def _table_poses(
    generator: np.random.Generator,
    items: list[GraspObject],
    config: fu.FurnitureConfig,
) -> list[tuple[float, float, float]]:
    obstacles = _furniture_obstacles(config)
    placed: list[tuple[np.ndarray, float]] = []
    poses = []
    for item in items:
        radius = math.hypot(item.size[0], item.size[1]) / 2
        for _ in range(400):
            distance = generator.uniform(*TABLE_RADIUS)
            azimuth = generator.uniform(-TABLE_AZIMUTH, TABLE_AZIMUTH)
            point = distance * np.array([math.cos(azimuth), math.sin(azimuth)])
            clearance = max(radius + OBJECT_MARGIN, HAND_CLEARANCE)
            if not all(_circle_clear(point, clearance, box) for box in obstacles):
                continue
            if any(
                np.linalg.norm(point - other) < radius + r + OBJECT_MARGIN for other, r in placed
            ):
                continue
            placed.append((point, radius))
            poses.append(
                (float(point[0]), float(point[1]), float(generator.uniform(-math.pi, math.pi)))
            )
            break
        else:
            raise NoEpisode("no free table spot")
    return poses


def _role_ok(
    role: str,
    item: GraspObject,
    template: Template,
    config: fu.FurnitureConfig,
    shelf_count: int,
) -> bool:
    """Whether ``item`` can play ``role`` everywhere the template sends it."""
    if role in template.in_drawer and not fits_drawer(item, config):
        return False
    for skill, obj, target in template.steps:
        if skill == "stack" and role in (obj, target) and not stackable(item):
            return False
        if obj != role:
            continue
        if skill == "place":
            check = {
                "D": lambda: fits_drawer(item, config),
                "shelf": lambda: fits_shelf(item, config, shelf_count),
                "bin": lambda: fits_bin(item, config),
                "region": lambda: fits_region(item, config),
            }[target]
            if not check():
                return False
    return True


def bind(
    template: Template,
    furniture: fu.FurnitureConfig,
    objects: Sequence[GraspObject],
    generator: np.random.Generator,
) -> Episode:
    """Bind ``template`` to objects that fit this furniture; raises :class:`NoEpisode`."""
    shelf_count = sum(
        1 for skill, _, target in template.steps if skill == "place" and target == "shelf"
    )
    drawer = int(generator.integers(furniture.drawer_count))
    roles = template.roles
    chosen: dict[str, int] = {}
    order = list(generator.permutation(len(objects)))
    for role in roles:
        for index in order:
            if index in chosen.values():
                continue
            if _role_ok(role, objects[index], template, furniture, shelf_count):
                chosen[role] = int(index)
                break
        else:
            raise NoEpisode(f"no object fits role {role} of {template.name}")
    # Towers stand the object with the largest footprint at the bottom.
    if template.name == "tower":
        ranked = sorted(("A", "B", "C"), key=lambda r: -_footprint(objects[chosen[r]]))
        chosen = dict(zip(("A", "B", "C"), (chosen[r] for r in ranked), strict=True))
    for skill, top, base in template.steps:
        if skill == "stack" and not stable_on(objects[chosen[top]], objects[chosen[base]]):
            raise NoEpisode(
                f"{objects[chosen[top]].name} would not stand on {objects[chosen[base]].name}"
            )
    extra = int(generator.integers(template.distractors[0], template.distractors[1] + 1))
    extra = min(max(extra, MIN_OBJECTS - len(roles)), MAX_OBJECTS - len(roles))
    for _ in range(extra):
        free = [i for i in order if i not in chosen.values()]
        chosen[f"x{_}"] = int(free[0])
    slot_of = {role: slot for slot, role in enumerate(chosen)}
    table = [role for role in chosen if role not in template.in_drawer]
    table_poses = dict(
        zip(
            table,
            _table_poses(generator, [objects[chosen[r]] for r in table], furniture),
            strict=True,
        )
    )
    poses, in_drawer = [], []
    for role in chosen:
        if role in template.in_drawer:
            poses.append((drawer_place_x(furniture), 0.0, 0.0))
            in_drawer.append(drawer)
        else:
            poses.append(table_poses[role])
            in_drawer.append(-1)
    spots: dict[str, int] = {}
    subgoals = []
    for skill, role, target in template.steps:
        kind = SKILLS.index(skill)
        if kind in (OPEN_DRAWER, CLOSE_DRAWER):
            subgoals.append(Subgoal(kind, target=drawer))
        elif kind in (OPEN_DOOR, CLOSE_DOOR):
            subgoals.append(Subgoal(kind))
        elif kind == PICK:
            subgoals.append(Subgoal(kind, obj=slot_of[role]))
        elif kind == STACK:
            subgoals.append(Subgoal(kind, obj=slot_of[role], target=slot_of[target]))
        else:
            receptacle = drawer if target == "D" else RECEPTACLES.index(target)
            spot = spots.get(target, 0)
            spots[target] = spot + 1
            subgoals.append(Subgoal(kind, obj=slot_of[role], target=receptacle, spot=spot))
    return Episode(
        template=template.name,
        furniture=furniture,
        subgoals=tuple(subgoals),
        objects=tuple(chosen.values()) + (-1,) * (MAX_OBJECTS - len(chosen)),
        poses=tuple(poses) + ((0.0, 0.0, 0.0),) * (MAX_OBJECTS - len(chosen)),
        in_drawer=tuple(in_drawer) + (-1,) * (MAX_OBJECTS - len(chosen)),
        drawer=drawer if any(target == "D" for _, _, target in template.steps) else -1,
    )


def sample_episode(
    generator: np.random.Generator,
    templates: Sequence[str],
    ranges: fu.FurnitureRanges,
    objects: Sequence[GraspObject],
    attempts: int = 50,
) -> Episode:
    """Draw a template, furniture and fitting objects; retries until everything fits."""
    template = TEMPLATES[templates[int(generator.integers(len(templates)))]]
    for _ in range(attempts):
        furniture = fu.sample_furniture(generator, ranges)
        try:
            return bind(template, furniture, objects, generator)
        except NoEpisode:
            continue
    raise NoEpisode(f"could not bind {template.name} in {attempts} attempts")


# ----------------------------------------------------------------------------- compiled tasks


def consumers(subgoals: Sequence[Subgoal]) -> list[list[int]]:
    """For every subgoal, the later subgoals whose completion also completes it."""
    out: list[list[int]] = [[] for _ in subgoals]
    for k, goal in enumerate(subgoals):
        if goal.kind in (OPEN_DRAWER, OPEN_DOOR):
            receptacle = goal.target if goal.kind == OPEN_DRAWER else SHELF
            close = CLOSE_DRAWER if goal.kind == OPEN_DRAWER else CLOSE_DOOR
            for j in range(k + 1, len(subgoals)):
                later = subgoals[j]
                if later.kind == close and (goal.kind == OPEN_DOOR or later.target == goal.target):
                    break
                if later.kind == PLACE and later.target == receptacle:
                    out[k].append(j)
                if later.kind == PICK and goal.kind == OPEN_DRAWER:
                    out[k].append(j)  # every pick in a template takes an object out of D
        elif goal.kind == PICK:
            for j in range(k + 1, len(subgoals)):
                if subgoals[j].kind in (PLACE, STACK) and subgoals[j].obj == goal.obj:
                    out[k].append(j)
                    break
    return out
