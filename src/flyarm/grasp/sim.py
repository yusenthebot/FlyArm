"""N grasp episodes over one MuJoCo model, with a pluggable physics backend.

:class:`GraspSim` holds every task rule that touches simulator state: resets, the 5-D action,
the IK, observations, reward and success. It reads and writes simulator fields as ``[N, ...]``
arrays and delegates only physics to a backend: :class:`MjbatchPhysics` (N simulations in
mjbatch) or :class:`MjDataPhysics` (one plain ``MjData`` and ``mj_step``). The batched and the
single environment are these two backends under the same rules; ``tests/test_grasp_env.py``
checks that they step alike and that the IK's hinge Jacobian equals ``mj_jacSite``.

Reset with seed ``s`` draws, in order: seven joint jitters, the table position, the yaw and
the object index, from ``numpy.random.default_rng(s)``. The pose therefore depends on the seed
alone, not on the object set, and an explicit object choice overrides only the drawn index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from flyarm.grasp import task
from flyarm.grasp.arm import (  # noqa: F401  (re-exported for existing callers)
    HOME,
    HOME_JITTER,
    ArmSim,
    MjbatchPhysics,
    MjDataPhysics,
    Physics,
)
from flyarm.grasp.objects import GraspObject
from flyarm.grasp.scene import (
    body_name,
    build_grasp_spec,
    compile_grasp_model,
    contact_sensor_name,
)

DROP_GAP = 0.001  # objects start this far above the table so no contact starts penetrating


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray  # [N, OBS_DIM], after automatic resets of finished episodes
    reward: np.ndarray  # [N]
    terminated: np.ndarray  # [N] bool: the success rule was met this step
    truncated: np.ndarray  # [N] bool: horizon reached without success
    success: np.ndarray  # [N] bool
    grasped: np.ndarray  # [N] bool: both fingers touching the object
    lifted: np.ndarray  # [N] bool: object at or above the lift target this step
    ever_lifted: np.ndarray  # [N] bool: lifted at any step of the episode (before reset)
    height_gain: np.ndarray  # [N] object height above its resting height
    object_index: np.ndarray  # [N] the episode's object (before reset)


def build_model(
    model_path: Path, objects: Sequence[GraspObject], asset_root: Path
) -> mujoco.MjModel:
    return compile_grasp_model(build_grasp_spec(model_path, objects, asset_root), objects)


class GraspSim(ArmSim):
    """Task state and rules for N environments; see the module docstring."""

    def __init__(
        self,
        model: mujoco.MjModel,
        objects: Sequence[GraspObject],
        physics: Physics,
        *,
        horizon: int = task.DEFAULT_HORIZON,
        first_seed: int = 0,
        reward: task.RewardConfig | None = None,
    ) -> None:
        super().__init__(model, physics, horizon=horizon, first_seed=first_seed)
        self.objects = list(objects)
        self.reward_config = reward or task.RewardConfig()
        bodies = [model.body(body_name(item)) for item in self.objects]
        self._body = np.array([body.id for body in bodies])
        free = [model.joint(int(body.jntadr[0])) for body in bodies]
        self._obj_qadr = np.array([int(joint.qposadr[0]) for joint in free])
        self._obj_dadr = np.array([int(joint.dofadr[0]) for joint in free])
        self._park_qpos = np.array([model.qpos0[a : a + 7] for a in self._obj_qadr])
        self._weight = -model.opt.gravity[2] * model.body_mass[self._body]
        self._contact_adr = np.array(
            [
                [
                    int(model.sensor(contact_sensor_name(item, side)).adr[0])
                    for side in ("left", "right")
                ]
                for item in self.objects
            ]
        )
        self.descriptors = np.array([item.descriptor for item in self.objects])
        self.rest_z = np.array([item.rest_z for item in self.objects])
        self.masses = np.array([item.mass for item in self.objects])

        n = self.num_envs
        self.object_index = np.zeros(n, dtype=np.int64)
        self.hold = np.zeros(n, dtype=np.int64)
        self.ever_grasped = np.zeros(n, dtype=bool)
        self.ever_lifted = np.zeros(n, dtype=bool)

    # Dimensions --------------------------------------------------------------------------
    obs_dim = task.OBS_DIM
    privileged_dim = task.PRIVILEGED_DIM

    @property
    def object_names(self) -> list[str]:
        return [item.name for item in self.objects]

    # State readers -----------------------------------------------------------------------
    def _object_slice(self, width: int, dof: bool = False) -> tuple[np.ndarray, np.ndarray]:
        start = (self._obj_dadr if dof else self._obj_qadr)[self.object_index]
        return self.rows[:, None], start[:, None] + np.arange(width)

    def object_pos(self) -> np.ndarray:
        return self.qpos[self._object_slice(3)]

    def object_quat(self) -> np.ndarray:
        rows, columns = self._object_slice(7)
        return self.qpos[rows, columns[:, 3:]]

    def object_rotation(self) -> np.ndarray:
        return task.quat_to_mat(self.object_quat())

    def object_velocity(self) -> np.ndarray:
        """World-frame linear velocity and body-frame angular velocity, [N, 6]."""
        return self.qvel[self._object_slice(6, dof=True)]

    def contacts(self) -> tuple[np.ndarray, np.ndarray]:
        address = self._contact_adr[self.object_index]
        found = self.sensordata[self.rows[:, None], address] > 0
        return found[:, 0], found[:, 1]

    def height_gain(self) -> np.ndarray:
        return self.object_pos()[:, 2] - self.rest_z[self.object_index]

    def grasp_target(self) -> np.ndarray:
        """Where the EE site should be to close the pads on the object's grasp point, [N, 3]."""
        descriptor = self.descriptors[self.object_index]
        local = np.stack(
            (
                descriptor[:, 3],
                np.zeros(self.num_envs),
                descriptor[:, 4] - descriptor[:, 2] / 2 + task.PAD_BELOW_SITE,
            ),
            -1,
        )
        target = self.object_pos() + np.einsum("nij,nj->ni", self.object_rotation(), local)
        target[:, 2] = np.maximum(target[:, 2], task.SITE_FLOOR)
        return target

    def object_yaw(self) -> np.ndarray:
        return task.object_axis_yaw(self.object_rotation())

    # Observation -------------------------------------------------------------------------
    def observation(self, *, privileged: bool = False) -> np.ndarray:
        """The controller observation; ``privileged`` appends mass, spin and commanded yaw."""
        ee, obj = self.ee(), self.object_pos()
        rotation = self.object_rotation()
        phi, psi = self.closing_yaw(), task.object_axis_yaw(rotation)
        relative = task.half_turn_wrap(psi - phi)
        velocity = self.object_velocity()
        left, right = self.contacts()
        rest = self.rest_z[self.object_index]
        lift_target = rest + task.LIFT_HEIGHT
        parts = [
            *self.arm_state(),
            obj,
            obj - ee,
            rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6),
            np.stack((np.sin(2 * psi), np.cos(2 * psi)), -1),
            np.stack((np.sin(2 * relative), np.cos(2 * relative)), -1),
            velocity[:, :3],
            self.descriptors[self.object_index],
            self.grasp_target() - ee,
            lift_target[:, None],
            (lift_target - obj[:, 2])[:, None],
            np.stack((left, right), -1),
            self.ever_grasped[:, None],
            (obj[:, 2] - rest >= task.LIFT_HEIGHT)[:, None],
            (self.hold / task.HOLD_STEPS)[:, None],
        ]
        if privileged:
            parts += [
                self.masses[self.object_index][:, None],
                velocity[:, 3:],
                self.yaw_command[:, None],
            ]
        return np.concatenate(parts, axis=1).astype(np.float32)

    # Reset -------------------------------------------------------------------------------
    def reset(
        self,
        ids: np.ndarray | None = None,
        seeds: np.ndarray | None = None,
        objects: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reset envs ``ids`` (all by default) with explicit or consecutive seeds.

        ``objects`` optionally fixes each reset env's object (indices into ``self.objects``).
        """
        ids, seeds = self._reset_seeds(ids, seeds)
        if objects is not None:
            objects = np.asarray(objects, dtype=np.int64)
            if (
                objects.shape != ids.shape
                or objects.min() < 0
                or objects.max() >= len(self.objects)
            ):
                raise ValueError("objects must hold one valid object index per reset environment")
        self.physics.reset(ids)
        for k, (row, seed) in enumerate(zip(ids, seeds, strict=True)):
            generator = np.random.default_rng(int(seed))
            self._home(row, generator)
            xy = generator.uniform(task.WORKSPACE_LOW, task.WORKSPACE_HIGH)
            yaw = generator.uniform(-np.pi, np.pi)
            drawn = int(generator.integers(len(self.objects)))
            index = drawn if objects is None else int(objects[k])
            self.object_index[row] = index
            for other, start in enumerate(self._obj_qadr):
                self.qpos[row, start : start + 7] = self._park_qpos[other]
            start = self._obj_qadr[index]
            z = self.rest_z[index] + DROP_GAP
            self.qpos[row, start : start + 7] = [
                xy[0],
                xy[1],
                z,
                np.cos(yaw / 2),
                0.0,
                0.0,
                np.sin(yaw / 2),
            ]
            self.qvel[row] = 0.0
            self.xfrc[row] = 0.0
            parked = np.arange(len(self.objects)) != index
            self.xfrc[row, self._body[parked], 2] = self._weight[parked]
        self.physics.forward(ids)
        self._finish_arm_reset(ids, seeds)
        self.hold[ids] = 0
        self.ever_grasped[ids] = self.ever_lifted[ids] = False
        return self.observation()

    # Step --------------------------------------------------------------------------------
    def step(self, action: np.ndarray, *, auto_reset: bool = True) -> StepResult:
        self.apply_action(action)
        self.physics.step(task.SUBSTEPS)
        self.steps += 1
        left, right = self.contacts()
        grasped = left & right
        gain = self.height_gain()
        lifted = gain >= task.LIFT_HEIGHT
        self.ever_grasped |= grasped
        self.ever_lifted |= lifted
        self.hold = task.update_hold(self.hold, grasped, gain)
        success = task.is_success(self.hold)
        relative = task.half_turn_wrap(self.object_yaw() - self.closing_yaw())
        reward = task.shaped_reward(
            self.ee(),
            self.grasp_target(),
            relative,
            grasped,
            gain,
            success,
            self.last_action,
            self.reward_config,
        )
        truncated = (self.steps >= self.horizon) & ~success
        ever_lifted = self.ever_lifted.copy()
        object_index = self.object_index.copy()
        done = success | truncated
        if auto_reset and done.any():
            self.reset(np.flatnonzero(done))
        return StepResult(
            obs=self.observation(),
            reward=reward,
            terminated=success,
            truncated=truncated,
            success=success,
            grasped=grasped,
            lifted=lifted,
            ever_lifted=ever_lifted,
            height_gain=gain,
            object_index=object_index,
        )
