"""N multi-step manipulation episodes over one model, on the grasp task's arm and backends.

:class:`ManipulationSim` adds to :class:`flyarm.grasp.arm.ArmSim` everything the articulated
tasks need: per-episode furniture written into per-simulation model fields, up to four objects
in fixed slots, the subgoal predicates of :mod:`flyarm.manipulation.tasks` evaluated from the
scene every step, the observation with its memoryless sub-task cue, and the reward.

Everything that moves is read from ``qpos``/``qvel`` (current after a step) and every furniture
frame is computed from the episode's configuration and joint positions, so the batched and the
single environment evaluate the same numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from flyarm.grasp import task as arm_task
from flyarm.grasp.arm import ArmSim, Physics
from flyarm.grasp.objects import GraspObject
from flyarm.grasp.scene import body_name, contact_sensor_name
from flyarm.manipulation import furniture as fu
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.scene import ARTICULATED, finger_sensor_name, robot_sensor_name

S, R, M = tk.MAX_OBJECTS, len(tk.RECEPTACLES), tk.MAX_SUBGOALS
DROP_GAP = 0.001
OPEN_TOP_HEIGHT = 0.3  # the bin and the region count everything above them up to this
ARTICULATIONS = ("drawer_0", "drawer_1", "lid")
_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)

# Observation layout ----------------------------------------------------------------------------
ROBOT_FIELDS = (
    ("joint_pos", 7),
    ("joint_vel", 7),
    ("ee_pos", 3),
    ("gripper_yaw", 2),
    ("gripper_opening", 1),
)
SLOT_FIELDS = (
    ("present", 1),
    ("position", 3),
    ("minus_ee", 3),
    ("rot6d", 6),
    ("descriptor", 5),
    ("velocity", 3),
    ("grasped", 1),
)
FURNITURE_FIELDS = (
    ("drawer_cabinet_xy", 2),
    ("drawer_cabinet_facing", 2),  # cos, sin of the direction its front faces
    ("drawer_present", 2),
    ("drawer_open_fraction", 2),  # joint position over the full travel
    ("drawer_handle", 6),  # both handles, world position
    ("drawer_handle_minus_ee", 6),
    ("drawer_handle_knob", 1),
    ("cabinet_xy", 2),
    ("cabinet_facing", 2),
    ("lid_angle", 1),
    ("lid_handle", 3),
    ("lid_handle_minus_ee", 3),
    ("lid_handle_knob", 1),
    ("shelf_height", 1),
    ("cabinet_interior", 3),  # half depth, half width, height
    ("bin_xy", 2),
    ("bin_facing", 2),
    ("bin_half", 2),
    ("region_xy", 2),
    ("region_half", 1),
)
CUE_FIELDS = (
    ("skill", len(tk.SKILLS) - 1),  # one-hot over the seven skills
    ("articulation", len(ARTICULATIONS)),  # drawer 0, drawer 1, lid
    ("object_slot", S),
    ("object_position", 3),
    ("object_minus_ee", 3),
    ("object_rot6d", 6),
    ("object_descriptor", 5),
    ("receptacle", R + 1),  # the five receptacles, then "on top of an object"
    ("target_point", 3),  # where the object's bottom centre should go
    ("target_minus_ee", 3),
    ("target_minus_object", 3),
    ("receptacle_half", 3),
    ("receptacle_facing", 2),  # cos, sin of the receptacle's x axis
    ("handle", 3),
    ("handle_minus_ee", 3),
    ("joint_fraction", 1),
    ("pull_direction", 3),  # unit direction the handle must move
    ("progress", 2),  # subgoals done / total, subgoals left / MAX_SUBGOALS
)
PRIVILEGED_FIELDS = (("object_mass", S), ("subgoal_done", M), ("yaw_command", 1))


def _width(fields: tuple[tuple[str, int], ...]) -> int:
    return sum(size for _, size in fields)


ROBOT_DIM, SLOT_DIM = _width(ROBOT_FIELDS), _width(SLOT_FIELDS)
FURNITURE_DIM, CUE_DIM = _width(FURNITURE_FIELDS), _width(CUE_FIELDS)
CUE_START = ROBOT_DIM + S * SLOT_DIM + FURNITURE_DIM
OBS_DIM = CUE_START + CUE_DIM
PRIVILEGED_DIM = OBS_DIM + _width(PRIVILEGED_FIELDS)


def cue_slices() -> dict[str, slice]:
    out, start = {}, CUE_START
    for name, size in CUE_FIELDS:
        out[name] = slice(start, start + size)
        start += size
    return out


# Reward --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardConfig:
    """Kitchen-style reward; see docs/MANIPULATION_ENV.md for the invariants.

    ``subgoal_bonus`` is paid once per subgoal, only when the leading run of done subgoals
    grows past its high-water mark, so subgoals pay in task order and undoing and redoing one
    pays nothing. ``shaping`` weighs the potential ``Phi = leading + phi``, where ``phi`` in
    [0, 1) is the current subgoal's approach and progress; it telescopes over the episode. The
    other terms are penalties: stray robot contacts, displacement of objects and articulations
    the task does not involve (as a potential), and the action magnitude.
    """

    subgoal_bonus: float = 50.0
    shaping: float = 10.0
    stray_contact: float = 0.05
    disturbance: float = 5.0
    action_cost: float = 0.01
    gamma: float = 0.99

    def __post_init__(self) -> None:
        for name in ("shaping", "stray_contact", "disturbance", "action_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if self.subgoal_bonus <= stalling_floor(self.gamma):
            raise ValueError("subgoal_bonus must exceed the stalling floor")
        if self.subgoal_bonus <= self.shaping:
            raise ValueError(
                "subgoal_bonus must exceed the shaping weight, the most shaping can pay per subgoal"
            )


MAX_LEVEL_REWARD = 0.0  # every per-step level term is a penalty


def stalling_floor(gamma: float) -> float:
    """Discounted value of the best per-step level reward held forever (research log E34)."""
    return MAX_LEVEL_REWARD / (1.0 - gamma)


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray  # every subgoal done
    truncated: np.ndarray
    success: np.ndarray
    subgoals_done: np.ndarray  # [N] leading done subgoals this step (before reset)
    subgoals_total: np.ndarray
    high_water: np.ndarray  # [N] most subgoals ever done in order this episode
    stray_contact: np.ndarray
    current_skill: np.ndarray


def _yaw_matrices(yaw: np.ndarray) -> np.ndarray:
    return arm_task.yaw_matrix(np.asarray(yaw, dtype=np.float64))


def _lid_rotation(angle: np.ndarray) -> np.ndarray:
    """Rotation of the lid about the cabinet's -y axis by ``angle``, [N, 3, 3]."""
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(angle), np.ones_like(angle)
    return np.stack(
        (np.stack((c, zero, -s), -1), np.stack((zero, one, zero), -1), np.stack((s, zero, c), -1)),
        axis=1,
    )


class ManipulationSim(ArmSim):
    """Task state and rules for N environments; episodes are drawn by ``sampler``."""

    def __init__(
        self,
        model: mujoco.MjModel,
        objects: Sequence[GraspObject],
        physics: Physics,
        *,
        templates: Sequence[str],
        ranges: fu.FurnitureRanges,
        first_seed: int = 0,
        reward: RewardConfig | None = None,
        cue: bool = True,
        horizon: int | None = None,
    ) -> None:
        longest = max(len(tk.TEMPLATES[name].steps) for name in templates)
        super().__init__(
            model,
            physics,
            horizon=horizon or tk.horizon(longest),
            first_seed=first_seed,
        )
        self.fixed_horizon = horizon
        self.objects = list(objects)
        self.templates = tuple(templates)
        self.ranges = ranges
        self.reward_config = reward or RewardConfig()
        self.cue_input = cue
        self.furniture_fields = fu.FurnitureFields(model)
        self.model_fields = {name: physics.model_field(name) for name in fu.MODEL_FIELDS}
        bodies = [model.body(body_name(item)) for item in self.objects]
        self._body = np.array([body.id for body in bodies])
        free = [model.joint(int(body.jntadr[0])) for body in bodies]
        self._obj_qadr = np.array([int(joint.qposadr[0]) for joint in free])
        self._obj_dadr = np.array([int(joint.dofadr[0]) for joint in free])
        self._park_qpos = np.array([model.qpos0[a : a + 7] for a in self._obj_qadr])
        self._weight = -model.opt.gravity[2] * model.body_mass[self._body]
        self._finger_adr = np.array(
            [
                [
                    int(model.sensor(contact_sensor_name(item, side)).adr[0])
                    for side in ("left", "right")
                ]
                for item in self.objects
            ]
        )
        self._robot_object_adr = np.array(
            [int(model.sensor(robot_sensor_name(item.name)).adr[0]) for item in self.objects]
        )
        self._robot_any_adr = int(model.sensor(robot_sensor_name("any")).adr[0])
        self._robot_self_adr = int(model.sensor(robot_sensor_name("self")).adr[0])
        self._robot_body_adr = np.array(
            [int(model.sensor(robot_sensor_name(name)).adr[0]) for name in ARTICULATED]
        )
        # Finger contacts with each handle-bearing body: drawer 0, drawer 1, the lid handle.
        self._handle_finger_adr = np.array(
            [
                [
                    int(model.sensor(finger_sensor_name(side, body)).adr[0])
                    for side in ("left", "right")
                ]
                for body in ("drawer_0", "drawer_1", "lid_handle")
            ]
        )
        joints = [model.joint(name) for name in ("drawer_0_slide", "drawer_1_slide", "lid_hinge")]
        self._art_qadr = np.array([int(joint.qposadr[0]) for joint in joints])
        self._art_dadr = np.array([int(joint.dofadr[0]) for joint in joints])
        self._swivel_qadr = int(model.joint("lid_handle_swivel").qposadr[0])
        self.descriptors_all = np.array([item.descriptor for item in self.objects])
        self.sizes_all = np.array([item.size for item in self.objects])
        self.masses_all = np.array([item.mass for item in self.objects])
        self._allocate()

    # ------------------------------------------------------------------------------ state
    def _allocate(self) -> None:
        n = self.num_envs
        self.episodes: list[tk.Episode | None] = [None] * n
        self.horizons = np.full(n, self.horizon, dtype=np.int64)
        self.slot_object = np.full((n, S), -1, dtype=np.int64)
        self.present = np.zeros((n, S), dtype=bool)
        self.source = np.full((n, S), -1, dtype=np.int64)
        self.start_pos = np.zeros((n, S, 3))
        self.sub_kind = np.zeros((n, M), dtype=np.int64)
        self.sub_obj = np.zeros((n, M), dtype=np.int64)
        self.sub_target = np.zeros((n, M), dtype=np.int64)
        self.sub_count = np.zeros(n, dtype=np.int64)
        self.consumer = np.zeros((n, M, M), dtype=bool)
        self.place_point = np.zeros((n, M, 3))  # receptacle frame, object bottom centre
        self.involved = np.zeros((n, S), dtype=bool)
        self.articulation_involved = np.zeros((n, 3), dtype=bool)
        self.dc_pos, self.dc_rot = np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1))
        self.cab_pos, self.cab_rot = np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1))
        self.bin_pos, self.bin_rot = np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1))
        self.region_pos, self.region_rot = np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1))
        self.drawer_body = np.zeros((n, 2, 3))
        self.drawer_present = np.zeros((n, 2), dtype=bool)
        self.travel = np.ones((n, 2))
        self.drawer_handle_local = np.zeros((n, 3))
        self.drawer_knob = np.zeros(n)
        self.lid_hinge = np.zeros((n, 3))
        self.lid_handle_local = np.zeros((n, 3))
        self.lid_knob = np.zeros(n)
        self.box_centre = np.zeros((n, R, 3))
        self.box_half = np.zeros((n, R, 3))
        self.place_count = np.zeros((n, S, R), dtype=np.int64)
        self.stack_count = np.zeros((n, S, S), dtype=np.int64)
        self.high_water = np.zeros(n, dtype=np.int64)
        self.previous_potential = np.zeros(n)
        self.previous_disturbance = np.zeros(n)
        self.start_joints = np.zeros((n, 3))
        self.previous_position = np.zeros((n, S, 3))
        self.previous_rotation = np.tile(np.eye(3), (n, S, 1, 1))
        self.previous_origins = np.zeros((n, R, 3))

    obs_dim = OBS_DIM
    privileged_dim = PRIVILEGED_DIM

    # ------------------------------------------------------------------------------ readers
    def _slot_index(self) -> np.ndarray:
        return np.maximum(self.slot_object, 0)

    def object_pos(self) -> np.ndarray:
        """[N, S, 3] object origins (the centre of each oriented bounding box)."""
        address = self._obj_qadr[self._slot_index()]
        return self.qpos[self.rows[:, None, None], address[..., None] + np.arange(3)]

    def object_rotation(self) -> np.ndarray:
        address = self._obj_qadr[self._slot_index()]
        quat = self.qpos[self.rows[:, None, None], address[..., None] + 3 + np.arange(4)]
        return arm_task.quat_to_mat(quat.reshape(-1, 4)).reshape(self.num_envs, S, 3, 3)

    def object_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        address = self._obj_dadr[self._slot_index()]
        velocity = self.qvel[self.rows[:, None, None], address[..., None] + np.arange(6)]
        return velocity[..., :3], velocity[..., 3:]

    def object_size(self) -> np.ndarray:
        return self.sizes_all[self._slot_index()] * self.present[..., None]

    def object_descriptor(self) -> np.ndarray:
        return self.descriptors_all[self._slot_index()] * self.present[..., None]

    def object_corners(self) -> np.ndarray:
        """[N, S, 8, 3] world corners of every object's oriented bounding box."""
        half = self.object_size()[:, :, None, :] / 2 * _SIGNS
        return self.object_pos()[:, :, None, :] + np.einsum(
            "nsij,nskj->nski", self.object_rotation(), half
        )

    def finger_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        address = self._finger_adr[self._slot_index()]
        found = self.sensordata[self.rows[:, None, None], address] > 0
        found &= self.present[..., None]
        return found[..., 0], found[..., 1]

    def handle_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        """[N, 3] left and right finger touching drawer 0, drawer 1 and the lid handle."""
        found = self.sensordata[:, self._handle_finger_adr] > 0
        return found[..., 0], found[..., 1]

    def grasped(self) -> np.ndarray:
        left, right = self.finger_contacts()
        return left & right

    def joints(self) -> np.ndarray:
        """[N, 3] drawer 0 and drawer 1 slide positions (m) and the lid angle (rad)."""
        return self.qpos[:, self._art_qadr].copy()

    def joint_velocity(self) -> np.ndarray:
        return self.qvel[:, self._art_dadr].copy()

    def drawer_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """[N, 2, 3] world origins of both drawers; they share the cabinet's rotation."""
        slide = self.joints()[:, :2, None] * np.array([1.0, 0.0, 0.0])
        local = self.drawer_body + slide
        return self.dc_pos[:, None, :] + np.einsum("nij,nkj->nki", self.dc_rot, local), self.dc_rot

    def drawer_handles(self) -> np.ndarray:
        """[N, 2, 3] world position of both drawer handles' grasp points."""
        origins, rotation = self.drawer_frames()
        return origins + np.einsum("nij,nj->ni", rotation, self.drawer_handle_local)[:, None, :]

    def lid_frame(self) -> tuple[np.ndarray, np.ndarray]:
        rotation = self.cab_rot @ _lid_rotation(self.joints()[:, 2])
        origin = self.cab_pos + np.einsum("nij,nj->ni", self.cab_rot, self.lid_hinge)
        return origin, rotation

    def lid_handle(self) -> np.ndarray:
        origin, rotation = self.lid_frame()
        return origin + np.einsum("nij,nj->ni", rotation, self.lid_handle_local)

    def lid_handle_tangent(self) -> np.ndarray:
        """[N, 3] unit direction the lid handle moves as the lid opens."""
        angle = self.joints()[:, 2]
        derivative = np.stack(
            (
                -np.sin(angle) * self.lid_handle_local[:, 0]
                - np.cos(angle) * self.lid_handle_local[:, 2],
                np.zeros_like(angle),
                np.cos(angle) * self.lid_handle_local[:, 0]
                - np.sin(angle) * self.lid_handle_local[:, 2],
            ),
            -1,
        )
        world = np.einsum("nij,nj->ni", self.cab_rot, derivative)
        return world / np.linalg.norm(world, axis=1, keepdims=True)

    def receptacle_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """[N, R, 3] origins and [N, R, 3, 3] rotations of the five receptacles."""
        drawers, drawer_rot = self.drawer_frames()
        origins = np.concatenate(
            (drawers, self.cab_pos[:, None], self.bin_pos[:, None], self.region_pos[:, None]), 1
        )
        rotations = np.stack(
            (drawer_rot, drawer_rot, self.cab_rot, self.bin_rot, self.region_rot), axis=1
        )
        return origins, rotations

    def receptacle_velocity(self) -> np.ndarray:
        velocity = np.zeros((self.num_envs, R, 3))
        axis = self.dc_rot[:, :, 0]
        velocity[:, :2] = self.joint_velocity()[:, :2, None] * axis[:, None, :]
        return velocity

    def grasp_points(self) -> np.ndarray:
        """[N, S, 3] where the EE site goes to close the pads on each object (grasp task rule)."""
        descriptor = self.object_descriptor()
        local = np.stack(
            (
                descriptor[..., 3],
                np.zeros(descriptor.shape[:2]),
                descriptor[..., 4] - descriptor[..., 2] / 2 + arm_task.PAD_BELOW_SITE,
            ),
            -1,
        )
        points = self.object_pos() + np.einsum("nsij,nsj->nsi", self.object_rotation(), local)
        points[..., 2] = np.maximum(points[..., 2], arm_task.SITE_FLOOR)
        return points

    def handle_sites(self) -> np.ndarray:
        """[N, 3, 3] where the EE site goes to grasp each articulation's handle."""
        handles = np.concatenate((self.drawer_handles(), self.lid_handle()[:, None]), 1)
        return handles + np.array([0.0, 0.0, arm_task.PAD_BELOW_SITE])

    # ------------------------------------------------------------------------------ predicates
    def inside(self, corners: np.ndarray) -> np.ndarray:
        """[N, S, R] every corner of the object inside the receptacle's volume."""
        origins, rotations = self.receptacle_frames()
        offset = corners[:, :, None] - origins[:, None, :, None, :]
        local = np.einsum("nrji,nsrkj->nsrki", rotations, offset)
        low = self.box_centre - self.box_half - tk.INSIDE_TOLERANCE
        high = self.box_centre + self.box_half + tk.INSIDE_TOLERANCE
        ok = (local >= low[:, None, :, None, :]) & (local <= high[:, None, :, None, :])
        return ok.all(axis=(3, 4))

    def scene_state(self) -> dict[str, np.ndarray]:
        """Every quantity the predicates, reward and observation share, for this step."""
        corners = self.object_corners()
        inside = self.inside(corners) & self.present[..., None]
        # At rest: over the last control step the object moved under REST_MOVE relative to the
        # receptacle and turned under REST_TURN. Displacements, not instantaneous velocities: a
        # light object on a drawer floor chatters at a few rad/s with sub-degree amplitude.
        position = self.object_pos()
        rotation = self.object_rotation()
        origins, _ = self.receptacle_frames()
        moved = position - self.previous_position
        carried = origins - self.previous_origins
        relative = moved[:, :, None, :] - carried[:, None, :, :]
        turn = np.einsum("nsji,nsjk->nsik", self.previous_rotation, rotation)
        angle = np.arccos(np.clip((np.trace(turn, axis1=2, axis2=3) - 1) / 2, -1.0, 1.0))
        turned_little = angle < tk.REST_TURN
        still = (np.linalg.norm(relative, axis=3) < tk.REST_MOVE) & turned_little[..., None]
        left, right = self.finger_contacts()
        robot = self.sensordata[self.rows[:, None], self._robot_object_adr[self._slot_index()]] > 0
        touched = (left | right | robot) & self.present
        grasped = left & right
        joints = self.joints()
        bottom = corners[..., 2].min(2)
        top = corners[..., 2].max(2)
        # Stacking: a's lowest point on b's highest, a's centre over b's footprint, a upright.
        offset = position[:, :, None, :] - position[:, None, :, :]
        over = np.einsum("nbji,nabj->nabi", rotation, offset)
        size = self.object_size()
        footprint = (np.abs(over[..., 0]) <= size[:, None, :, 0] / 2) & (
            np.abs(over[..., 1]) <= size[:, None, :, 1] / 2
        )
        touching = np.abs(bottom[:, :, None] - top[:, None, :]) <= tk.STACK_GAP
        upright = rotation[:, :, 2, 2] > 0.9
        speed = (np.linalg.norm(moved, axis=2) < tk.REST_MOVE) & turned_little
        pair = self.present[:, :, None] & self.present[:, None, :] & ~np.eye(S, dtype=bool)
        geometry = pair & footprint & touching & (upright & ~touched)[:, :, None]
        stacked = geometry & speed[:, :, None] & speed[:, None, :]
        untouched_inside = inside & ~touched[..., None]
        placed = untouched_inside & still
        return {
            "placed_still_or_moving": untouched_inside,
            "stacked_geometry": geometry,
            "corners": corners,
            "inside": inside,
            "placed_now": placed,
            "stacked_now": stacked,
            "grasped": grasped,
            "touched": touched,
            "joints": joints,
            "bottom": bottom,
            "top": top,
        }

    def remember_poses(self, ids: np.ndarray | None = None) -> None:
        """Store object poses and receptacle origins for the next step's rest test."""
        ids = self.rows if ids is None else ids
        self.previous_position[ids] = self.object_pos()[ids]
        self.previous_rotation[ids] = self.object_rotation()[ids]
        self.previous_origins[ids] = self.receptacle_frames()[0][ids]

    def update_counters(self, state: dict[str, np.ndarray]) -> None:
        """Consecutive steps each placement and stack has held.

        A placement must first rest (relative to its receptacle) for PLACE_STEPS; after that it
        keeps counting while the object stays inside and untouched, so a drawer carrying it shut
        does not undo it. Stacks likewise keep counting once stable while the geometry holds.
        """
        placed = state["placed_now"] | (
            state["placed_still_or_moving"] & (self.place_count >= tk.PLACE_STEPS)
        )
        stacked = state["stacked_now"] | (
            state["stacked_geometry"] & (self.stack_count >= tk.STACK_STEPS)
        )
        self.place_count = np.where(placed, self.place_count + 1, 0)
        self.stack_count = np.where(stacked, self.stack_count + 1, 0)

    def effects(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """[N, M] whether each subgoal's own effect holds in the current scene."""
        joints = state["joints"]
        drawer_open = joints[:, :2] >= tk.DRAWER_OPEN_FRACTION * self.travel
        drawer_closed = joints[:, :2] <= tk.DRAWER_CLOSED
        lid_open = joints[:, 2] >= tk.LID_OPEN
        lid_closed = joints[:, 2] <= tk.LID_CLOSED
        placed = self.place_count >= tk.PLACE_STEPS
        stacked = self.stack_count >= tk.STACK_STEPS
        rows = self.rows[:, None]
        obj, target = self.sub_obj, self.sub_target
        drawer = np.clip(target, 0, 1)
        source = self.source[rows, obj]
        out_of_source = ~state["inside"][rows, obj, np.maximum(source, 0)]
        lifted = self.object_pos()[rows, obj, 2] - self.start_pos[rows, obj, 2] >= tk.PICK_LIFT
        picked = state["grasped"][rows, obj] & np.where(source >= 0, out_of_source, lifted)
        kind = self.sub_kind
        effect = np.ones_like(kind, dtype=bool)
        effect = np.where(kind == tk.OPEN_DRAWER, drawer_open[rows, drawer], effect)
        effect = np.where(kind == tk.CLOSE_DRAWER, drawer_closed[rows, drawer], effect)
        effect = np.where(kind == tk.OPEN_DOOR, lid_open[:, None], effect)
        effect = np.where(kind == tk.CLOSE_DOOR, lid_closed[:, None], effect)
        effect = np.where(kind == tk.PICK, picked, effect)
        effect = np.where(kind == tk.PLACE, placed[rows, obj, np.clip(target, 0, R - 1)], effect)
        effect = np.where(kind == tk.STACK, stacked[rows, obj, np.clip(target, 0, S - 1)], effect)
        return effect

    def subgoal_done(self, effect: np.ndarray) -> np.ndarray:
        """[N, M] done = effect, or every consumer done; padding subgoals count as done."""
        done = np.ones_like(effect)
        for k in range(M - 1, -1, -1):
            consumers = self.consumer[:, k]
            has = consumers.any(1)
            consumed = (done | ~consumers).all(1) & has
            done[:, k] = effect[:, k] | consumed
        return done | (np.arange(M) >= self.sub_count[:, None])

    def leading(self, done: np.ndarray) -> np.ndarray:
        """[N] how many subgoals are done in task order from the first."""
        return np.minimum(np.cumprod(done, axis=1).sum(1), self.sub_count)

    # ------------------------------------------------------------------------------ targets
    def current(self, leading: np.ndarray) -> np.ndarray:
        """[N] index of the current subgoal (clamped to the last one when all are done)."""
        return np.minimum(leading, np.maximum(self.sub_count - 1, 0))

    def place_targets(self) -> np.ndarray:
        """[N, M, 3] world point where each place or stack subgoal wants the object's bottom."""
        origins, rotations = self.receptacle_frames()
        rows = self.rows[:, None]
        receptacle = np.clip(self.sub_target, 0, R - 1)
        world = origins[rows, receptacle] + np.einsum(
            "nmij,nmj->nmi", rotations[rows, receptacle], self.place_point
        )
        # Stacks: on top of the base object's highest point, over its centre.
        base = np.clip(self.sub_target, 0, S - 1)
        corners = self.object_corners()
        top = corners[..., 2].max(2)
        base_pos = self.object_pos()[rows, base]
        stack = np.concatenate((base_pos[..., :2], top[rows, base][..., None]), -1)
        return np.where((self.sub_kind == tk.STACK)[..., None], stack, world)

    def potential(self, state: dict[str, np.ndarray], leading: np.ndarray) -> np.ndarray:
        """Phi = leading + phi(current subgoal), phi in [0, 1)."""
        index = self.current(leading)
        rows = self.rows
        kind = self.sub_kind[rows, index]
        obj = self.sub_obj[rows, index]
        target = self.sub_target[rows, index]
        ee = self.ee()
        joints = state["joints"]
        handles = self.handle_sites()
        articulation = np.where(
            np.isin(kind, (tk.OPEN_DOOR, tk.CLOSE_DOOR)), 2, np.clip(target, 0, 1)
        )
        handle_reach = 1.0 - np.tanh(5.0 * np.linalg.norm(ee - handles[rows, articulation], axis=1))
        fraction = np.where(
            articulation == 2,
            joints[:, 2] / tk.LID_OPEN,
            joints[rows, np.clip(articulation, 0, 1)]
            / (tk.DRAWER_OPEN_FRACTION * self.travel[rows, np.clip(articulation, 0, 1)]),
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        grasp = self.grasp_points()[rows, obj]
        object_reach = 1.0 - np.tanh(5.0 * np.linalg.norm(ee - grasp, axis=1))
        grasped = state["grasped"][rows, obj].astype(float)
        bottom_centre = self.object_pos()[rows, obj].copy()
        bottom_centre[:, 2] = state["bottom"][rows, obj]
        goal = self.place_targets()[rows, index]
        closeness = 1.0 - np.tanh(5.0 * np.linalg.norm(bottom_centre - goal, axis=1))
        effect_now = self.effects(state)[rows, index].astype(float)
        phi = np.zeros(self.num_envs)
        phi = np.where(
            np.isin(kind, (tk.OPEN_DRAWER, tk.OPEN_DOOR)), 0.3 * handle_reach + 0.7 * fraction, phi
        )
        phi = np.where(
            np.isin(kind, (tk.CLOSE_DRAWER, tk.CLOSE_DOOR)),
            0.3 * handle_reach + 0.7 * (1 - fraction),
            phi,
        )
        phi = np.where(kind == tk.PICK, 0.4 * object_reach + 0.3 * grasped + 0.3 * effect_now, phi)
        carry = 0.3 * np.maximum(object_reach, grasped) + 0.2 * grasped + 0.5 * closeness
        phi = np.where(np.isin(kind, (tk.PLACE, tk.STACK)), carry, phi)
        phi = 0.99 * np.clip(phi, 0.0, 1.0)
        complete = leading >= self.sub_count
        return np.where(complete, self.sub_count, leading + phi)

    def disturbance(self) -> np.ndarray:
        """[N] minus the displacement of objects and articulations the task does not involve."""
        moved = np.linalg.norm(self.object_pos() - self.start_pos, axis=2)
        idle = self.present & ~self.involved
        joints = np.abs(self.joints() - self.start_joints)
        joints[:, :2] *= self.drawer_present
        return -((moved * idle).sum(1) + (joints * ~self.articulation_involved).sum(1))

    def stray_contact(self, index: np.ndarray) -> np.ndarray:
        """[N] the robot touches something the current subgoal does not need it to touch."""
        rows = self.rows
        data = self.sensordata
        total = data[:, self._robot_any_adr] - data[:, self._robot_self_adr]
        kind = self.sub_kind[rows, index]
        obj = self.sub_obj[rows, index]
        allowed = np.zeros(self.num_envs)
        uses_object = np.isin(kind, (tk.PICK, tk.PLACE, tk.STACK))
        object_adr = self._robot_object_adr[self.slot_object[rows, obj].clip(0)]
        allowed += np.where(uses_object, data[rows, object_adr], 0.0)
        drawer = np.clip(self.sub_target[rows, index], 0, 1)
        uses_drawer = np.isin(kind, (tk.OPEN_DRAWER, tk.CLOSE_DRAWER))
        allowed += np.where(uses_drawer, data[rows, self._robot_body_adr[drawer]], 0.0)
        uses_lid = np.isin(kind, (tk.OPEN_DOOR, tk.CLOSE_DOOR))
        lid = data[:, self._robot_body_adr[2]] + data[:, self._robot_body_adr[3]]
        allowed += np.where(uses_lid, lid, 0.0)
        return (total - allowed) > 0

    # ------------------------------------------------------------------------------ observation
    def observation(self, *, privileged: bool = False) -> np.ndarray:
        state = self.scene_state()
        done = self.subgoal_done(self.effects(state))
        leading = self.leading(done)
        return self._observe(state, done, leading, privileged)

    def _observe(
        self,
        state: dict[str, np.ndarray],
        done: np.ndarray,
        leading: np.ndarray,
        privileged: bool,
    ) -> np.ndarray:
        n = self.num_envs
        ee = self.ee()
        present = self.present[..., None].astype(float)
        position = self.object_pos() * present
        rotation = self.object_rotation()
        rot6d = rotation[..., :2].transpose(0, 1, 3, 2).reshape(n, S, 6) * present
        linear, _ = self.object_velocity()
        slots = np.concatenate(
            (
                present,
                position,
                (position - ee[:, None]) * present,
                rot6d,
                self.object_descriptor(),
                linear * present,
                state["grasped"][..., None].astype(float),
            ),
            axis=2,
        ).reshape(n, -1)
        joints = state["joints"]
        drawer_handles = self.drawer_handles() * self.drawer_present[..., None]
        lid_handle = self.lid_handle()
        facing = lambda rot: rot[:, :2, 0]  # noqa: E731  (cos, sin of the local x axis)
        furniture = np.concatenate(
            (
                self.dc_pos[:, :2],
                facing(self.dc_rot),
                self.drawer_present.astype(float),
                joints[:, :2] / self.travel,
                drawer_handles.reshape(n, 6),
                ((self.drawer_handles() - ee[:, None]) * self.drawer_present[..., None]).reshape(
                    n, 6
                ),
                self.drawer_knob[:, None],
                self.cab_pos[:, :2],
                facing(self.cab_rot),
                joints[:, 2:3],
                lid_handle,
                lid_handle - ee,
                self.lid_knob[:, None],
                self.box_centre[:, tk.SHELF, 2:3] - self.box_half[:, tk.SHELF, 2:3],
                np.concatenate(
                    (self.box_half[:, tk.SHELF, :2], 2 * self.box_half[:, tk.SHELF, 2:3]), 1
                ),
                self.bin_pos[:, :2],
                facing(self.bin_rot),
                self.box_half[:, tk.BIN, :2],
                self.region_pos[:, :2],
                self.box_half[:, tk.REGION, :1],
            ),
            axis=1,
        )
        parts = [*self.arm_state(), slots, furniture, self._cue(state, done, leading)]
        if privileged:
            parts += [
                self.masses_all[self._slot_index()] * self.present,
                done.astype(float),
                self.yaw_command[:, None],
            ]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _cue(
        self, state: dict[str, np.ndarray], done: np.ndarray, leading: np.ndarray
    ) -> np.ndarray:
        """The memoryless sub-task cue, recomputed from the scene; zero in the no-cue control."""
        n, rows = self.num_envs, self.rows
        cue = np.zeros((n, CUE_DIM))
        if not self.cue_input:
            return cue
        index = self.current(leading)
        active = leading < self.sub_count
        kind = self.sub_kind[rows, index]
        obj = self.sub_obj[rows, index]
        target = self.sub_target[rows, index]
        ee = self.ee()
        fields: dict[str, np.ndarray] = {}
        fields["skill"] = np.eye(len(tk.SKILLS))[kind][:, 1:]
        articulation_skill = np.isin(
            kind, (tk.OPEN_DRAWER, tk.CLOSE_DRAWER, tk.OPEN_DOOR, tk.CLOSE_DOOR)
        )
        articulation = np.where(
            np.isin(kind, (tk.OPEN_DOOR, tk.CLOSE_DOOR)), 2, np.clip(target, 0, 1)
        )
        fields["articulation"] = np.eye(3)[articulation] * articulation_skill[:, None]
        object_skill = np.isin(kind, (tk.PICK, tk.PLACE, tk.STACK))
        mask = object_skill[:, None].astype(float)
        fields["object_slot"] = np.eye(S)[obj] * mask
        position = self.object_pos()[rows, obj]
        fields["object_position"] = position * mask
        fields["object_minus_ee"] = (position - ee) * mask
        rotation = self.object_rotation()[rows, obj]
        fields["object_rot6d"] = rotation[:, :, :2].transpose(0, 2, 1).reshape(n, 6) * mask
        fields["object_descriptor"] = self.object_descriptor()[rows, obj] * mask
        placing = np.isin(kind, (tk.PLACE, tk.STACK))
        receptacle = np.where(kind == tk.STACK, R, np.clip(target, 0, R - 1))
        fields["receptacle"] = np.eye(R + 1)[receptacle] * placing[:, None]
        goal = self.place_targets()[rows, index]
        bottom = position.copy()
        bottom[:, 2] = state["bottom"][rows, obj]
        pmask = placing[:, None].astype(float)
        fields["target_point"] = goal * pmask
        fields["target_minus_ee"] = (goal - ee) * pmask
        fields["target_minus_object"] = (goal - bottom) * pmask
        base = np.clip(target, 0, S - 1)
        stack_half = self.object_size()[rows, base] / 2
        half = np.where(
            (kind == tk.STACK)[:, None], stack_half, self.box_half[rows, np.clip(target, 0, R - 1)]
        )
        fields["receptacle_half"] = half * pmask
        _, rotations = self.receptacle_frames()
        receptacle_rot = rotations[rows, np.clip(target, 0, R - 1)]
        base_rot = self.object_rotation()[rows, base]
        axis = np.where((kind == tk.STACK)[:, None], base_rot[:, :2, 0], receptacle_rot[:, :2, 0])
        fields["receptacle_facing"] = axis * pmask
        handles = np.concatenate((self.drawer_handles(), self.lid_handle()[:, None]), 1)
        handle = handles[rows, articulation]
        amask = articulation_skill[:, None].astype(float)
        fields["handle"] = handle * amask
        fields["handle_minus_ee"] = (handle - ee) * amask
        joints = state["joints"]
        span = np.where(
            articulation == 2, fu.LID_OPEN_LIMIT, self.travel[rows, np.clip(articulation, 0, 1)]
        )
        fields["joint_fraction"] = (joints[rows, articulation] / span)[:, None] * amask
        opening = np.isin(kind, (tk.OPEN_DRAWER, tk.OPEN_DOOR))
        direction = np.where(
            (articulation == 2)[:, None], self.lid_handle_tangent(), self.dc_rot[:, :, 0]
        )
        direction = direction * np.where(opening, 1.0, -1.0)[:, None]
        fields["pull_direction"] = direction * amask
        total = np.maximum(self.sub_count, 1)
        fields["progress"] = np.stack(
            (np.minimum(leading, total) / total, (self.sub_count - np.minimum(leading, total)) / M),
            1,
        )
        out = np.concatenate([fields[name] for name, _ in CUE_FIELDS], axis=1)
        out[~active, : CUE_DIM - 2] = 0.0
        return out

    # ------------------------------------------------------------------------------ reset
    def reset(
        self,
        ids: np.ndarray | None = None,
        seeds: np.ndarray | None = None,
        templates: Sequence[str] | None = None,
        episodes: Sequence[tk.Episode] | None = None,
    ) -> np.ndarray:
        """Reset envs ``ids`` with explicit or consecutive seeds.

        Each seed draws the arm jitter, then the episode (template, furniture, objects, poses)
        from this environment's templates and furniture ranges. ``templates`` fixes the template
        per reset env; ``episodes`` fixes the whole episode (the seed then only jitters the arm).
        """
        ids, seeds = self._reset_seeds(ids, seeds)
        for k, (row, seed) in enumerate(zip(ids, seeds, strict=True)):
            generator = np.random.default_rng(int(seed))
            self._home(row, generator)
            if episodes is not None:
                episode = episodes[k]
            else:
                names = self.templates if templates is None else (templates[k],)
                episode = tk.sample_episode(generator, names, self.ranges, self.objects)
            self._load_episode(row, episode)
        # The furniture's model fields are written; the reset below applies them.
        arm = self.qpos[np.ix_(ids, self._qadr)].copy()
        self.physics.reset(ids)
        self.qpos[np.ix_(ids, self._qadr)] = arm
        for row in ids:
            self._place_objects(row)
        self.physics.forward(ids)
        self._finish_arm_reset(ids, seeds)
        self.place_count[ids] = 0
        self.stack_count[ids] = 0
        self.high_water[ids] = 0
        self.start_pos[ids] = self.object_pos()[ids]
        self.start_joints[ids] = self.joints()[ids]
        self.remember_poses(ids)
        state = self.scene_state()
        done = self.subgoal_done(self.effects(state))
        leading = self.leading(done)
        self.high_water[ids] = leading[ids]
        self.previous_potential[ids] = self.potential(state, leading)[ids]
        self.previous_disturbance[ids] = self.disturbance()[ids]
        return self._observe(state, done, leading, False)

    def _load_episode(self, row: int, episode: tk.Episode) -> None:
        config = episode.furniture
        self.episodes[row] = episode
        self.furniture_fields.apply(self.model_fields, row, config)
        self.horizons[row] = self.fixed_horizon or episode.horizon
        self.slot_object[row] = episode.objects
        self.present[row] = np.asarray(episode.objects) >= 0
        self.source[row] = episode.in_drawer
        self.dc_pos[row] = [config.drawer_pose[0], config.drawer_pose[1], 0.0]
        self.dc_rot[row] = _yaw_matrices(np.array([config.drawer_pose[2]]))[0]
        self.cab_pos[row] = [config.cabinet_pose[0], config.cabinet_pose[1], 0.0]
        self.cab_rot[row] = _yaw_matrices(np.array([config.cabinet_pose[2]]))[0]
        self.bin_pos[row] = [config.bin_pose[0], config.bin_pose[1], 0.0]
        self.bin_rot[row] = _yaw_matrices(np.array([config.bin_pose[2]]))[0]
        self.region_pos[row] = [config.region_pose[0], config.region_pose[1], 0.0]
        self.region_rot[row] = _yaw_matrices(np.array([config.region_pose[2]]))[0]
        plan = fu.layout(config)
        for index in range(2):
            self.drawer_body[row, index] = plan.bodies[f"drawer_{index}"][0]
        self.drawer_present[row] = [True, config.drawer_count == 2]
        self.travel[row] = config.drawer_travel
        self.drawer_handle_local[row] = plan.handles["drawer_0"]
        self.drawer_knob[row] = float(config.drawer_handle)
        self.lid_hinge[row] = plan.bodies["lid"][0]
        self.lid_handle_local[row] = plan.handles["lid"]
        self.lid_knob[row] = float(config.lid_handle)
        # Receptacle volumes, each in its own frame: (centre, half extents).
        inner = config.drawer_width / 2 - fu.GAP - fu.TRAY_WALL
        tray = config.tray_depth
        floor = fu.TRAY_FLOOR
        top = config.drawer_height - 2 * fu.GAP  # the carcass top, in the drawer's frame
        drawer_box = (
            np.array([-fu.PANEL - tray / 2, 0.0, (floor + top) / 2]),
            np.array([tray / 2 - fu.TRAY_WALL / 2, inner, (top - floor) / 2]),
        )
        cab_half = np.array(
            [config.cabinet_depth / 2 - fu.WALL, config.cabinet_width / 2 - fu.WALL]
        )
        shelf_box = (
            np.array([0.0, 0.0, config.shelf_height + config.interior_height / 2]),
            np.array([cab_half[0], cab_half[1], config.interior_height / 2]),
        )
        bin_half = np.array(
            [config.bin_depth / 2 - fu.BIN_WALL, config.bin_width / 2 - fu.BIN_WALL]
        )
        bin_box = (
            np.array([0.0, 0.0, fu.BIN_FLOOR + OPEN_TOP_HEIGHT / 2]),
            np.array([bin_half[0], bin_half[1], OPEN_TOP_HEIGHT / 2]),
        )
        region = config.region_size / 2
        region_box = (
            np.array([0.0, 0.0, OPEN_TOP_HEIGHT / 2]),
            np.array([region, region, OPEN_TOP_HEIGHT / 2]),
        )
        for receptacle, (centre, half) in enumerate(
            (drawer_box, drawer_box, shelf_box, bin_box, region_box)
        ):
            self.box_centre[row, receptacle] = centre
            self.box_half[row, receptacle] = half
        # Subgoals.
        goals = episode.subgoals
        self.sub_count[row] = len(goals)
        self.sub_kind[row] = 0
        self.sub_obj[row] = 0
        self.sub_target[row] = 0
        self.consumer[row] = False
        self.place_point[row] = 0.0
        shelf_total = sum(1 for g in goals if g.kind == tk.PLACE and g.target == tk.SHELF)
        spots = tk.shelf_spots(config, max(shelf_total, 1))
        for k, goal in enumerate(goals):
            self.sub_kind[row, k] = goal.kind
            self.sub_obj[row, k] = max(goal.obj, 0)
            self.sub_target[row, k] = max(goal.target, 0)
            if goal.kind == tk.PLACE:
                if goal.target in (tk.DRAWER_0, tk.DRAWER_1):
                    point = np.array([tk.drawer_place_x(config), 0.0, fu.TRAY_FLOOR])
                elif goal.target == tk.SHELF:
                    point = spots[min(goal.spot, len(spots) - 1)]
                elif goal.target == tk.BIN:
                    point = np.array([0.0, 0.0, fu.BIN_FLOOR])
                else:
                    point = np.zeros(3)
                self.place_point[row, k] = point
        for k, later in enumerate(tk.consumers(goals)):
            self.consumer[row, k, later] = True
        involved = np.zeros(S, dtype=bool)
        articulations = np.zeros(3, dtype=bool)
        for goal in goals:
            if goal.kind in (tk.PICK, tk.PLACE, tk.STACK):
                involved[goal.obj] = True
            if goal.kind == tk.STACK:
                involved[goal.target] = True
            if goal.kind in (tk.OPEN_DRAWER, tk.CLOSE_DRAWER):
                articulations[goal.target] = True
            if goal.kind in (tk.OPEN_DOOR, tk.CLOSE_DOOR):
                articulations[2] = True
        self.involved[row] = involved
        self.articulation_involved[row] = articulations

    def _place_objects(self, row: int) -> None:
        episode = self.episodes[row]
        assert episode is not None
        config = episode.furniture
        self.qvel[row] = 0.0
        self.xfrc[row] = 0.0
        used = set()
        for other, start in enumerate(self._obj_qadr):
            self.qpos[row, start : start + 7] = self._park_qpos[other]
        for slot, index in enumerate(episode.objects):
            if index < 0:
                continue
            used.add(index)
            item = self.objects[index]
            x, y, yaw = episode.poses[slot]
            drawer = episode.in_drawer[slot]
            if drawer >= 0:
                local = self.drawer_body[row, drawer] + np.array(
                    [x, y, fu.TRAY_FLOOR + item.rest_z + DROP_GAP]
                )
                world = self.dc_pos[row] + self.dc_rot[row] @ local
                yaw = config.drawer_pose[2] + yaw
                z = world[2]
                x, y = world[0], world[1]
            else:
                z = item.rest_z + DROP_GAP
            start = self._obj_qadr[index]
            self.qpos[row, start : start + 7] = [
                x,
                y,
                z,
                np.cos(yaw / 2),
                0.0,
                0.0,
                np.sin(yaw / 2),
            ]
        parked = np.array([index not in used for index in range(len(self.objects))])
        self.xfrc[row, self._body[parked], 2] = self._weight[parked]
        self.qpos[row, self._art_qadr] = 0.0
        self.qpos[row, self._swivel_qadr] = 0.0

    # ------------------------------------------------------------------------------ step
    def step(self, action: np.ndarray, *, auto_reset: bool = True) -> StepResult:
        self.apply_action(action)
        self.physics.step(arm_task.SUBSTEPS)
        self.steps += 1
        state = self.scene_state()
        self.update_counters(state)
        self.remember_poses()
        done = self.subgoal_done(self.effects(state))
        leading = self.leading(done)
        config = self.reward_config
        potential = self.potential(state, leading)
        disturbance = self.disturbance()
        index = self.current(leading)
        stray = self.stray_contact(index)
        newly = np.maximum(leading - self.high_water, 0)
        self.high_water = np.maximum(self.high_water, leading)
        reward = (
            config.subgoal_bonus * newly
            + config.shaping * (potential - self.previous_potential)
            + config.disturbance * (disturbance - self.previous_disturbance)
            - config.stray_contact * stray
            - config.action_cost * np.mean(self.last_action**2, axis=1)
        ).astype(np.float32)
        self.previous_potential = potential
        self.previous_disturbance = disturbance
        success = leading >= self.sub_count
        truncated = (self.steps >= self.horizons) & ~success
        result = StepResult(
            obs=self._observe(state, done, leading, False),
            reward=reward,
            terminated=success,
            truncated=truncated,
            success=success,
            subgoals_done=leading.copy(),
            subgoals_total=self.sub_count.copy(),
            high_water=self.high_water.copy(),
            stray_contact=stray,
            current_skill=self.sub_kind[self.rows, index].copy(),
        )
        finished = success | truncated
        if auto_reset and finished.any():
            fresh = self.reset(np.flatnonzero(finished))
            obs = result.obs.copy()
            obs[finished] = fresh[finished]
            result = StepResult(**{**result.__dict__, "obs": obs})
        return result
