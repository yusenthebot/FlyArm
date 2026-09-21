"""A small, dynamics-only Cartesian reaching environment for the Panda arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


class PandaReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Panda Cartesian reach task using DLS IK and MuJoCo position servos.

    Actions are normalized XYZ displacements; every call to :meth:`step` advances
    MuJoCo for 25 2-ms substeps (20 Hz). It never writes ``qpos`` in step.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self, model_path: Path, horizon: int = 100, render_mode: str | None = None
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.model_path, self.horizon, self.render_mode = (
            model_path,
            horizon,
            render_mode,
        )
        self._spec = mujoco.MjSpec.from_file(str(model_path))
        hand = self._spec.body("hand")
        if hand is None:
            raise ValueError("expected Menagerie Panda body named 'hand'")
        hand.add_site(name="flyarm_ee", pos=[0.0, 0.0, 0.10], size=[0.008])
        target = self._spec.worldbody.add_body(name="flyarm_target", mocap=True)
        target.add_geom(
            name="flyarm_target_marker",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025, 0.0, 0.0],
            rgba=[0.95, 0.1, 0.1, 0.65],
            contype=0,
            conaffinity=0,
        )
        self.model = self._spec.compile()
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self._joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                for i in range(1, 8)
            ]
        )
        if np.any(self._joint_ids < 0):
            raise ValueError("expected Menagerie Panda joints joint1 through joint7")
        self._qadr = self.model.jnt_qposadr[self._joint_ids]
        self._dadr = self.model.jnt_dofadr[self._joint_ids]
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "flyarm_ee")
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flyarm_target")
        self._target_mocap = int(self.model.body_mocapid[target_body])
        # Menagerie scene.xml supplies one position servo for each Panda arm joint.
        self._servo_ids = np.array(
            [
                next(
                    actuator
                    for actuator in range(self.model.nu)
                    if self.model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
                    and self.model.actuator_trnid[actuator, 0] == joint
                )
                for joint in self._joint_ids
            ]
        )
        self._gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if self._gripper_id < 0:
            raise ValueError("expected Menagerie Panda gripper actuator8")
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self._renderer: mujoco.Renderer | None = None
        self.target = np.zeros(3)
        self._desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._steps = self._success_steps = 0
        self.last_joint_command = np.zeros(7)
        self.reset()

    def _ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site].copy()

    def _observation(self) -> np.ndarray:
        ee = self._ee_position()
        return np.concatenate(
            (
                self.data.qpos[self._qadr],
                self.data.qvel[self._dadr],
                ee,
                self.target - ee,
            )
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        ee = self._ee_position()
        distance = float(np.linalg.norm(self.target - ee))
        return {
            "distance": distance,
            "is_success": distance < 0.02,
            "ee_position": ee.copy(),
            "target": self.target.copy(),
        }

    @staticmethod
    def _vector(value: Any, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector with shape (3,)")
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        home = np.array([0.0, -0.45, 0.0, -2.2, 0.0, 1.75, 0.75])
        self.data.qpos[self._qadr] = home + self.np_random.uniform(-0.025, 0.025, 7)
        mujoco.mj_forward(self.model, self.data)
        self._desired_orientation = np.empty(4)
        mujoco.mju_mat2Quat(self._desired_orientation, self.data.site_xmat[self._ee_site])
        center = self._ee_position()
        options = options or {}
        self.target = (
            self._vector(options["target"], "target")
            if "target" in options
            else center + self.np_random.uniform([-0.12, -0.12, -0.10], [0.12, 0.12, 0.10])
        )
        self.data.mocap_pos[self._target_mocap] = self.target
        self.data.mocap_quat[self._target_mocap] = [1.0, 0.0, 0.0, 0.0]
        # Refresh derived body/site positions so render() shows the reset target.
        mujoco.mj_forward(self.model, self.data)
        self._steps = self._success_steps = 0
        self.last_joint_command = self.data.qpos[self._qadr].copy()
        self.data.ctrl[self._servo_ids] = self.last_joint_command
        self.data.ctrl[self._gripper_id] = 255.0
        return self._observation(), self._info()

    def _ik_command(self, displacement: np.ndarray) -> np.ndarray:
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site)
        current_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self._ee_site])
        rotation_error = np.empty(3)
        mujoco.mju_subQuat(rotation_error, self._desired_orientation, current_quat)
        # mju_subQuat is expressed in the current body frame, whereas mj_jacSite's
        # rotational Jacobian maps qvel to world-frame angular velocity.
        rotation_error = self.data.site_xmat.reshape(3, 3) @ rotation_error
        jacobian = np.vstack((jacp[:, self._dadr], 0.25 * jacr[:, self._dadr]))
        error = np.concatenate((displacement, 0.25 * rotation_error))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-3 * np.eye(6), error)
        command = self.data.qpos[self._qadr] + np.clip(delta, -0.06, 0.06)
        limits = self.model.jnt_range[self._joint_ids]
        return np.clip(command, limits[:, 0] + 0.01, limits[:, 1] - 0.01)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = self._vector(action, "action")
        if np.any(np.abs(action) > 1.0):
            raise ValueError("action components must be in [-1, 1]")
        self.last_joint_command = self._ik_command(action * 0.02)
        self.data.ctrl[self._servo_ids] = self.last_joint_command
        self.data.ctrl[self._gripper_id] = 255.0
        mujoco.mj_step(self.model, self.data, nstep=25)
        self._steps += 1
        info = self._info()
        self._success_steps = self._success_steps + 1 if info["is_success"] else 0
        terminated = self._success_steps >= 5
        info["is_success"] = terminated
        truncated = self._steps >= self.horizon
        return self._observation(), -info["distance"], terminated, truncated, info

    def teacher_action(self) -> np.ndarray:
        """Normalized proportional Cartesian action in the public action interface."""
        return np.clip((self.target - self._ee_position()) / 0.02, -1.0, 1.0).astype(np.float32)

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
