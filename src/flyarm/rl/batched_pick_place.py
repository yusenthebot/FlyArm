"""N copies of the FlyArm Panda pick-and-place task stepped in parallel with mjbatch.

Same compiled model, observation, action and success rule as PandaPickPlaceEnv; only the
bookkeeping is vectorized. Two read-only contact sensors (cube vs each finger body) replace
the contact-list scan, and the IK Jacobian of the end-effector site is built from each
hinge's world axis and anchor, which equals mj_jacSite for this all-hinge arm. A reset with
seed s draws the same random numbers in the same order as PandaPickPlaceEnv.reset(seed=s),
so both start from the same state.

TaskVariant makes the task harder without changing its observation or action layout:
per-episode cube mass and grip friction (drawn from a separate stream, so the nominal task
keeps its draws), a wider goal range, and a memory condition in which the goal fields of the
observation are blanked after the first steps while the critic still sees them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mjbatch import Batch

from flyarm.pick_place_env import (
    CUBE_HALF,
    PandaPickPlaceEnv,
    build_pick_place_spec,
    compile_pick_place_model,
)

OBS_DIM = PandaPickPlaceEnv.observation_dim
ACTION_DIM = 4
SUBSTEPS = 25
STEP_METERS = 0.014
LIFT_HEIGHT = CUBE_HALF + 0.06
GOAL_FIELDS = np.r_[20:23, 26:29]  # goal and goal - cube inside the 37-D observation
BASE_FRICTION = 4.0


@dataclass(frozen=True)
class TaskVariant:
    """Per-episode task conditions; the defaults are exactly the B1a pick-and-place task."""

    mass_scale: tuple[float, float] = (1.0, 1.0)  # log-uniform factor on the 25 g cube
    friction_scale: tuple[float, float] = (1.0, 1.0)  # log-uniform factor, cube and finger pads
    goal_reach: float = 0.11  # goal xy offset half-width around the initial end effector
    goal_visible_steps: int | None = None  # memory task: goal blanked after this many steps

    def __post_init__(self) -> None:
        for low, high in (self.mass_scale, self.friction_scale):
            if not 0 < low <= high:
                raise ValueError("scale ranges must satisfy 0 < low <= high")
        if not 0.05 <= self.goal_reach <= 0.3:
            raise ValueError("goal_reach must be in [0.05, 0.3] m")
        if self.goal_visible_steps is not None and self.goal_visible_steps < 1:
            raise ValueError("goal_visible_steps must be positive")

    @property
    def randomizes_physics(self) -> bool:
        return self.mass_scale != (1.0, 1.0) or self.friction_scale != (1.0, 1.0)


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray  # [N, 37], after automatic resets for finished episodes
    reward: np.ndarray  # [N]
    terminated: np.ndarray  # [N] bool, stable placement reached
    truncated: np.ndarray  # [N] bool, horizon reached
    success: np.ndarray  # [N] bool, stable placement at this step
    grasped: np.ndarray  # [N] bool, both fingers touching the cube
    lifted: np.ndarray  # [N] bool, ever lifted this episode (before reset)
    goal_xy_error: np.ndarray  # [N]


class BatchedPickPlace:
    def __init__(
        self,
        model_path: Path,
        num_envs: int,
        *,
        horizon: int = 400,
        first_seed: int = 0,
        num_threads: int = 0,
        variant: TaskVariant | None = None,
    ) -> None:
        if num_envs < 1 or horizon < 20:
            raise ValueError("num_envs must be positive and horizon at least 20")
        spec = build_pick_place_spec(Path(model_path))
        for side in ("left", "right"):
            spec.add_sensor(
                name=f"flyarm_{side}_cube_contact",
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_GEOM,
                objname="flyarm_cube_geom",
                reftype=mujoco.mjtObj.mjOBJ_BODY,
                refname=f"{side}_finger",
                intprm=[1, 1, 1],  # data="found", reduce="mindist", num=1
            )
        self.model = model = compile_pick_place_model(spec)
        self.num_envs, self.horizon = num_envs, horizon
        self.variant = variant or TaskVariant()
        self.batch = Batch(model, num_envs, num_threads)
        joints = [model.joint(f"joint{i}") for i in range(1, 8)]
        self._jnt = np.array([joint.id for joint in joints])
        self._qadr = np.array([int(joint.qposadr[0]) for joint in joints])
        self._dadr = np.array([int(joint.dofadr[0]) for joint in joints])
        self._limits = model.jnt_range[self._jnt]
        self._servo = np.array(
            [
                next(
                    a
                    for a in range(model.nu)
                    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.actuator_trnid[a, 0] == joint
                )
                for joint in self._jnt
            ]
        )
        self._gripper = model.actuator("actuator8").id
        self._finger_qadr = int(model.joint("finger_joint1").qposadr[0])
        cube_joint = model.joint("flyarm_cube_free")
        self._cube_qadr, self._cube_dadr = int(cube_joint.qposadr[0]), int(cube_joint.dofadr[0])
        self._cube_body = model.body("flyarm_cube").id
        self._ee_site = model.site("flyarm_pick_ee").id
        self._goal_mocap = int(model.body_mocapid[model.body("flyarm_goal").id])
        contact = [model.sensor(f"flyarm_{side}_cube_contact") for side in ("left", "right")]
        self._contact_adr = np.array([int(sensor.adr[0]) for sensor in contact])

        b = self.batch
        self.qpos, self.qvel, self.ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
        self.xpos, self.mocap_pos = b.bind("xpos"), b.bind("mocap_pos")
        self.site_xpos, self.site_xmat = b.bind("site_xpos"), b.bind("site_xmat")
        self.xanchor, self.xaxis = b.bind("xanchor"), b.bind("xaxis")
        self.sensordata = b.bind("sensordata")
        self._cube_geom = model.geom("flyarm_cube_geom").id
        self._pads = np.array(
            [model.geom(f"flyarm_{side}_finger_grip_pad").id for side in ("left", "right")]
        )
        if self.variant.randomizes_physics:
            self.body_mass = b.expand("body_mass")
            self.body_inertia = b.expand("body_inertia")
            self.geom_friction = b.expand("geom_friction")
        self.mass_scale = np.ones(num_envs)
        self.friction_scale = np.ones(num_envs)

        n = num_envs
        self.goal = np.zeros((n, 3))
        self.desired_quat = np.zeros((n, 4))
        self.steps = np.zeros(n, dtype=np.int64)
        self.stable = np.zeros(n, dtype=np.int64)
        self.ever_left = np.zeros(n, dtype=bool)
        self.ever_right = np.zeros(n, dtype=bool)
        self.ever_grasped = np.zeros(n, dtype=bool)
        self.ever_lifted = np.zeros(n, dtype=bool)
        self.episode_seed = np.zeros(n, dtype=np.int64)
        self._next_seed = first_seed

    # State readers ------------------------------------------------------------------------
    def ee(self) -> np.ndarray:
        return self.site_xpos[:, self._ee_site].copy()

    def cube(self) -> np.ndarray:
        return self.xpos[:, self._cube_body].copy()

    def contacts(self) -> tuple[np.ndarray, np.ndarray]:
        found = self.sensordata[:, self._contact_adr] > 0
        return found[:, 0], found[:, 1]

    def gripper_opening(self) -> np.ndarray:
        return np.clip(self.qpos[:, self._finger_qadr] / 0.04, 0.0, 1.0)

    def observation(self, *, privileged: bool = False) -> np.ndarray:
        """The controller's 37-D observation; ``privileged`` never blanks the goal (critic)."""
        obs = self._full_observation()
        hidden = self.variant.goal_visible_steps
        if hidden is not None and not privileged:
            obs[np.ix_(self.steps >= hidden, GOAL_FIELDS)] = 0.0
        return obs

    def _full_observation(self) -> np.ndarray:
        ee, cube = self.ee(), self.cube()
        left, right = self.contacts()
        cube_velocity = self.qvel[:, self._cube_dadr : self._cube_dadr + 3]
        return np.concatenate(
            (
                self.qpos[:, self._qadr],
                self.qvel[:, self._dadr],
                ee,
                cube,
                self.goal,
                cube - ee,
                self.goal - cube,
                self.gripper_opening()[:, None],
                cube_velocity,
                np.stack((left, right), 1),
                np.stack((self.ever_grasped, self.ever_lifted), 1),
            ),
            axis=1,
        ).astype(np.float32)

    # Reset ---------------------------------------------------------------------------------
    def reset(self, ids: np.ndarray | None = None, seeds: np.ndarray | None = None) -> np.ndarray:
        """Reset the given envs (all by default) with explicit or consecutive seeds."""
        ids = np.arange(self.num_envs) if ids is None else np.asarray(ids, dtype=np.int64)
        if seeds is None:
            seeds = np.arange(self._next_seed, self._next_seed + len(ids))
            self._next_seed += len(ids)
        if len(seeds) != len(ids):
            raise ValueError("One seed per reset environment is required")
        generators = [np.random.default_rng(int(seed)) for seed in seeds]
        self.batch.reset(ids)
        for row, generator in zip(ids, generators, strict=True):
            home = PandaPickPlaceEnv._HOME + generator.uniform(-0.012, 0.012, 7)
            self.qpos[row, self._qadr] = home
        self.batch.forward(ids)
        initial = self.site_xpos[ids, self._ee_site].copy()
        for k, (row, generator) in enumerate(zip(ids, generators, strict=True)):
            object_xy = initial[k, :2] + generator.uniform([-0.055, -0.055], [0.055, 0.055])
            reach = self.variant.goal_reach
            goal_xy = initial[k, :2] + generator.uniform([-reach, -reach], [reach, reach])
            if np.linalg.norm(goal_xy - object_xy) < 0.07:
                goal_xy = object_xy + np.array([0.10, 0.0])
            self.goal[row] = [goal_xy[0], goal_xy[1], 0.002]
            cube = slice(self._cube_qadr, self._cube_qadr + 7)
            self.qpos[row, cube] = [object_xy[0], object_xy[1], CUBE_HALF, 1.0, 0.0, 0.0, 0.0]
            self.qvel[row, self._cube_dadr : self._cube_dadr + 6] = 0.0
            self.mocap_pos[row, self._goal_mocap] = self.goal[row]
        self.batch.forward(ids)
        for row in ids:
            mujoco.mju_mat2Quat(self.desired_quat[row], self.site_xmat[row, self._ee_site])
            self.ctrl[row, self._servo] = self.qpos[row, self._qadr]
            self.ctrl[row, self._gripper] = 255.0
        for flags in (self.ever_left, self.ever_right, self.ever_grasped, self.ever_lifted):
            flags[ids] = False
        self.steps[ids] = self.stable[ids] = 0
        self.episode_seed[ids] = seeds
        if self.variant.randomizes_physics:
            self._randomize_physics(ids, seeds)
        return self.observation()

    def _randomize_physics(self, ids: np.ndarray, seeds: np.ndarray) -> None:
        """Cube mass (inertia scaled with it) and grip friction, from a stream of their own."""
        model, variant = self.model, self.variant
        for row, seed in zip(ids, seeds, strict=True):
            generator = np.random.default_rng([int(seed), 1])
            mass, friction = (
                float(np.exp(generator.uniform(np.log(low), np.log(high))))
                for low, high in (variant.mass_scale, variant.friction_scale)
            )
            self.mass_scale[row], self.friction_scale[row] = mass, friction
            self.body_mass[row, self._cube_body] = model.body_mass[self._cube_body] * mass
            self.body_inertia[row, self._cube_body] = model.body_inertia[self._cube_body] * mass
            for geom in (self._cube_geom, *self._pads):
                self.geom_friction[row, geom, 0] = BASE_FRICTION * friction
        self.batch.set_const(ids)
        self.batch.forward(ids)

    # Step ----------------------------------------------------------------------------------
    def _ik_commands(self, displacement: np.ndarray) -> np.ndarray:
        site = self.site_xpos[:, self._ee_site]  # [N, 3]
        axes = self.xaxis[:, self._jnt]  # [N, 7, 3]
        anchors = self.xanchor[:, self._jnt]
        jacp = np.cross(axes, site[:, None, :] - anchors).transpose(0, 2, 1)  # [N, 3, 7]
        jacr = axes.transpose(0, 2, 1)
        rotation = np.zeros((self.num_envs, 3))
        current = np.empty(4)
        for row in range(self.num_envs):
            matrix = self.site_xmat[row, self._ee_site]
            mujoco.mju_mat2Quat(current, matrix)
            mujoco.mju_subQuat(rotation[row], self.desired_quat[row], current)
            rotation[row] = matrix.reshape(3, 3) @ rotation[row]
        jacobian = np.concatenate((jacp, 0.25 * jacr), axis=1)  # [N, 6, 7]
        error = np.concatenate((displacement, 0.25 * rotation), axis=1)[..., None]
        gram = jacobian @ jacobian.transpose(0, 2, 1) + 1e-3 * np.eye(6)
        delta = (jacobian.transpose(0, 2, 1) @ np.linalg.solve(gram, error))[..., 0]
        command = self.qpos[:, self._qadr] + np.clip(delta, -0.055, 0.055)
        return np.clip(command, self._limits[:, 0] + 0.01, self._limits[:, 1] - 0.01)

    def step(self, action: np.ndarray, *, auto_reset: bool = True) -> StepResult:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.num_envs, ACTION_DIM) or not np.all(np.isfinite(action)):
            raise ValueError(f"actions must be finite with shape ({self.num_envs}, 4)")
        if np.any(np.abs(action) > 1.0):
            raise ValueError("action components must be in [-1, 1]")
        self.ctrl[:, self._servo] = self._ik_commands(action[:, :3] * STEP_METERS)
        self.ctrl[:, self._gripper] = 127.5 * (action[:, 3] + 1.0)
        self.batch.step(nstep=SUBSTEPS)
        self.steps += 1
        cube = self.cube()
        left, right = self.contacts()
        grasped = left & right
        self.ever_left |= left
        self.ever_right |= right
        self.ever_grasped |= grasped
        self.ever_lifted |= cube[:, 2] >= LIFT_HEIGHT
        goal_error = np.linalg.norm(cube[:, :2] - self.goal[:, :2], axis=1)
        speed = np.linalg.norm(self.qvel[:, self._cube_dadr : self._cube_dadr + 3], axis=1)
        release_ready = (
            self.ever_left
            & self.ever_right
            & self.ever_lifted
            & ~left
            & ~right
            & (self.gripper_opening() > 0.75)
            & (goal_error < 0.03)
            & (cube[:, 2] < CUBE_HALF + 0.012)
            & (speed < 0.06)
        )
        self.stable = np.where(release_ready, self.stable + 1, 0)
        success = self.stable >= 10
        reward = shaped_reward(self.ee(), cube, self.goal, grasped, self.ever_lifted, success)
        truncated = self.steps >= self.horizon
        lifted = self.ever_lifted.copy()
        done = success | truncated
        if auto_reset and done.any():
            self.reset(np.flatnonzero(done))
        return StepResult(
            obs=self.observation(),
            reward=reward,
            terminated=success,
            truncated=truncated & ~success,
            success=success,
            grasped=grasped,
            lifted=lifted,
            goal_xy_error=goal_error,
        )


def shaped_reward(
    ee: np.ndarray,
    cube: np.ndarray,
    goal: np.ndarray,
    grasped: np.ndarray,
    ever_lifted: np.ndarray,
    success: np.ndarray,
) -> np.ndarray:
    """Staged dense reward: reach the cube, hold it, carry it over the goal, place it.

    Terms are bounded per step; stable placement pays a large terminal bonus. The reward
    shapes training only; evaluation reports the benchmark's own success rule.
    """
    reach = 1.0 - np.tanh(10.0 * np.linalg.norm(ee - (cube + [0.0, 0.0, 0.045]), axis=1))
    height = np.clip((cube[:, 2] - CUBE_HALF) / 0.06, 0.0, 1.0)
    carry = 1.0 - np.tanh(10.0 * np.linalg.norm(cube[:, :2] - goal[:, :2], axis=1))
    return (
        0.5 * reach
        + 0.5 * grasped
        + 1.0 * grasped * height
        + 2.0 * ever_lifted * carry
        + 50.0 * success
    ).astype(np.float32)
