"""The Panda arm shared by the grasp and manipulation tasks: control, readers, physics backends.

:class:`ArmSim` holds the arm half of every task: joint and actuator indices, the bound
simulator fields, the 5-D action (Cartesian step, yaw turn, gripper) through the damped
least-squares IK with a null-space pull toward home, and the arm's part of a reset. Task
classes subclass it and add their objects, rules and observations. Physics runs in a backend:
:class:`MjbatchPhysics` (N simulations in mjbatch) or :class:`MjDataPhysics` (one ``MjData``).
Both also expose per-simulation model fields, which the manipulation task resizes per episode.
"""

from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np
from mjbatch import Batch

from flyarm.grasp import task
from flyarm.pick_place_env import PandaPickPlaceEnv
from flyarm.rl.batched_pick_place import simulation_threads

HOME = PandaPickPlaceEnv._HOME
HOME_JITTER = 0.012
BOUND_FIELDS = (
    "qpos",
    "qvel",
    "ctrl",
    "xfrc_applied",
    "site_xpos",
    "site_xmat",
    "xanchor",
    "xaxis",
    "sensordata",
    "qacc_warmstart",  # solver warm start: part of the state a snapshot must restore exactly
)


class Physics(Protocol):
    num_envs: int

    def field(self, name: str) -> np.ndarray: ...

    def reset(self, ids: np.ndarray) -> None: ...

    def forward(self, ids: np.ndarray) -> None: ...

    def step(self, nstep: int) -> None: ...

    def model_field(self, name: str) -> np.ndarray: ...


class MjbatchPhysics:
    """N simulations stepped in parallel by mjbatch."""

    def __init__(self, model: mujoco.MjModel, num_envs: int, num_threads: int = 0) -> None:
        self.num_envs = num_envs
        self.batch = Batch(model, num_envs, simulation_threads(num_threads))
        self._fields = {name: self.batch.bind(name) for name in BOUND_FIELDS}
        self._model_fields: dict[str, np.ndarray] = {}

    def field(self, name: str) -> np.ndarray:
        return self._fields[name]

    def reset(self, ids: np.ndarray) -> None:
        self.batch.reset(ids)

    def forward(self, ids: np.ndarray) -> None:
        self.batch.forward(ids)

    def step(self, nstep: int) -> None:
        self.batch.step(nstep=nstep)

    def model_field(self, name: str) -> np.ndarray:
        """Per-simulation values of an MjModel field, applied before every physics call."""
        if name not in self._model_fields:
            self._model_fields[name] = self.batch.expand(name)
        return self._model_fields[name]


class MjDataPhysics:
    """One simulation in a plain MjData; fields are exposed as [1, ...] views."""

    num_envs = 1

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)

    def field(self, name: str) -> np.ndarray:
        return getattr(self.data, name)[None]

    def reset(self, ids: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def forward(self, ids: np.ndarray) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self, nstep: int) -> None:
        mujoco.mj_step(self.model, self.data, nstep=nstep)

    def model_field(self, name: str) -> np.ndarray:
        """The model's own field as a [1, ...] view: this backend owns its model copy."""
        return getattr(self.model, name)[None]


class ArmSim:
    """Arm state, readers and control for N environments; tasks subclass it."""

    def __init__(
        self, model: mujoco.MjModel, physics: Physics, *, horizon: int, first_seed: int
    ) -> None:
        if horizon < 20:
            raise ValueError("horizon must be at least 20")
        self.model, self.physics = model, physics
        self.num_envs, self.horizon = physics.num_envs, horizon
        joints = [model.joint(f"joint{i}") for i in range(1, 8)]
        self._joints = np.array([joint.id for joint in joints])
        self._qadr = np.array([int(joint.qposadr[0]) for joint in joints])
        self._dadr = np.array([int(joint.dofadr[0]) for joint in joints])
        self._limits = model.jnt_range[self._joints]
        self._servo = np.array(
            [
                next(
                    a
                    for a in range(model.nu)
                    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.actuator_trnid[a, 0] == joint
                )
                for joint in self._joints
            ]
        )
        self._gripper = model.actuator("actuator8").id
        self._finger_qadr = int(model.joint("finger_joint1").qposadr[0])
        self._ee_site = model.site("flyarm_pick_ee").id
        f = physics.field
        self.qpos, self.qvel, self.ctrl = f("qpos"), f("qvel"), f("ctrl")
        self.xfrc, self.sensordata = f("xfrc_applied"), f("sensordata")
        self.site_xpos, self.site_xmat = f("site_xpos"), f("site_xmat")
        self.xaxis, self.xanchor = f("xaxis"), f("xanchor")
        n = self.num_envs
        self.rows = np.arange(n)
        self.base_rotation = np.tile(np.eye(3), (n, 1, 1))
        self.yaw_command = np.zeros(n)
        self.steps = np.zeros(n, dtype=np.int64)
        self.episode_seed = np.zeros(n, dtype=np.int64)
        self.last_action = np.zeros((n, task.ACTION_DIM))
        self._next_seed = first_seed

    action_dim = task.ACTION_DIM

    # Readers -------------------------------------------------------------------------------
    def ee(self) -> np.ndarray:
        return self.site_xpos[:, self._ee_site].copy()

    def ee_rotation(self) -> np.ndarray:
        return self.site_xmat[:, self._ee_site].reshape(-1, 3, 3).copy()

    def gripper_opening(self) -> np.ndarray:
        return np.clip(self.qpos[:, self._finger_qadr] / 0.04, 0.0, 1.0)

    def jacobian(self) -> np.ndarray:
        """EE-site Jacobian over the arm joints, [N, 6, 7] (see task.hinge_jacobian)."""
        return task.hinge_jacobian(
            self.site_xpos[:, self._ee_site],
            self.xaxis[:, self._joints],
            self.xanchor[:, self._joints],
        )

    def closing_yaw(self) -> np.ndarray:
        return task.closing_axis_yaw(self.site_xmat[:, self._ee_site])

    def commanded_closing_yaw(self) -> np.ndarray:
        desired = task.yaw_matrix(self.yaw_command) @ self.base_rotation
        return task.closing_axis_yaw(desired)

    def arm_state(self) -> list[np.ndarray]:
        """Joint positions and velocities, EE position, jaw yaw (sin 2phi, cos 2phi), opening."""
        phi = self.closing_yaw()
        return [
            self.qpos[:, self._qadr],
            self.qvel[:, self._dadr],
            self.ee(),
            np.stack((np.sin(2 * phi), np.cos(2 * phi)), -1),
            self.gripper_opening()[:, None],
        ]

    # Reset helpers -------------------------------------------------------------------------
    def _reset_seeds(
        self, ids: np.ndarray | None, seeds: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = self.rows.copy() if ids is None else np.asarray(ids, dtype=np.int64)
        if seeds is None:
            seeds = np.arange(self._next_seed, self._next_seed + len(ids))
            self._next_seed += len(ids)
        seeds = np.asarray(seeds, dtype=np.int64)
        if len(seeds) != len(ids):
            raise ValueError("one seed per reset environment is required")
        return ids, seeds

    def _home(self, row: int, generator: np.random.Generator) -> None:
        """Seven joint jitters from ``generator``: always the first draws of a reset."""
        self.qpos[row, self._qadr] = HOME + generator.uniform(-HOME_JITTER, HOME_JITTER, 7)

    def _finish_arm_reset(self, ids: np.ndarray, seeds: np.ndarray) -> None:
        """After the forward pass: hold the pose, open the gripper, clear the yaw command."""
        self.base_rotation[ids] = self.site_xmat[ids, self._ee_site].reshape(-1, 3, 3)
        self.ctrl[np.ix_(ids, self._servo)] = self.qpos[np.ix_(ids, self._qadr)]
        self.ctrl[ids, self._gripper] = 255.0
        self.yaw_command[ids] = 0.0
        self.steps[ids] = 0
        self.last_action[ids] = 0.0
        self.episode_seed[ids] = seeds

    # Control -------------------------------------------------------------------------------
    def apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.num_envs, task.ACTION_DIM) or not np.all(np.isfinite(action)):
            raise ValueError(f"actions must be finite with shape ({self.num_envs}, 5)")
        if np.any(np.abs(action) > 1.0):
            raise ValueError("action components must be in [-1, 1]")
        # A copy: the caller's array must not change when a reset later clears this row.
        self.last_action = action.copy()
        self.yaw_command = np.clip(
            self.yaw_command + action[:, 3] * task.YAW_STEP, -task.YAW_LIMIT, task.YAW_LIMIT
        )
        desired = task.yaw_matrix(self.yaw_command) @ self.base_rotation
        rotation = task.rotation_error(desired, self.ee_rotation())
        self.ctrl[:, self._servo] = task.ik_step(
            self.jacobian(),
            self.qpos[:, self._qadr],
            self._limits,
            action[:, :3] * task.STEP_METERS,
            rotation,
            HOME,
        )
        # Official Menagerie mapping: 0 is closed and 255 is open.
        self.ctrl[:, self._gripper] = 127.5 * (action[:, 4] + 1.0)
