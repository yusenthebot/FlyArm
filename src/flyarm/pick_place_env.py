"""Contact-based Panda pick-and-place task used by FlyArm causal experiments.

The environment deliberately exposes no shortcut that moves the cube: the only
state written in :meth:`reset` is the initial state.  Every :meth:`step` uses
joint servos, a physical gripper, and MuJoCo contact dynamics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


def physical_stage(info: dict[str, Any]) -> str:
    """Furthest physically verified outcome, independent of the teacher state machine."""
    if bool(info["is_success"]):
        return "placed"
    if bool(info["ever_lifted"]):
        return "lifted"
    if bool(info["ever_grasped"]):
        return "grasped"
    if bool(info["contact_left"]) or bool(info["contact_right"]):
        return "contact"
    return "free"


class PandaPickPlaceEnv(gym.Env[np.ndarray, np.ndarray]):
    """A tabletop 4-cm-cube pick-and-place task with real Panda contacts.

    The action is ``[delta_x, delta_y, delta_z, gripper]`` in ``[-1, 1]``.
    XYZ are bounded Cartesian increments and gripper ``-1`` closes while ``+1``
    opens.  Each action advances 25 2-ms MuJoCo steps (20 Hz).
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}
    observation_dim = 37
    _HOME = np.array([0.0, -0.45, 0.0, -2.2, 0.0, 1.75, 0.75])
    _CUBE_HALF = 0.02
    _SETTLED_HEIGHT = _CUBE_HALF
    _LIFT_HEIGHT = _SETTLED_HEIGHT + 0.06

    def __init__(
        self, model_path: Path, horizon: int = 400, render_mode: str | None = None
    ) -> None:
        if horizon < 20:
            raise ValueError("horizon must be at least 20")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.horizon, self.render_mode = horizon, render_mode

        spec = mujoco.MjSpec.from_file(str(self.model_path))
        hand = spec.body("hand")
        if hand is None:
            raise ValueError("expected Menagerie Panda body named 'hand'")
        # This coincides with the physical finger pads.  The reach-only task's
        # site at 10 cm is below the Panda fingers and would let a controller
        # appear to grasp while the real pads were still 4 cm above the cube.
        hand.add_site(name="flyarm_pick_ee", pos=[0.0, 0.0, 0.06], size=[0.008])
        # The Menagerie fingertips are intentionally tiny.  These are explicit
        # high-friction rubber jaw pads, rigidly part of each finger (not an
        # object constraint), so a 4-cm cube can be side-pinched above a table.
        # They make the simulated end effector a practical parallel gripper
        # while preserving ordinary MuJoCo contact, friction, and gravity.
        for name in ("left_finger", "right_finger"):
            finger = spec.body(name)
            if finger is None:
                raise ValueError(f"expected Menagerie Panda body named {name}")
            finger.add_geom(
                name=f"flyarm_{name}_grip_pad",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.0, 0.0, 0.045],
                size=[0.018, 0.006, 0.010],
                friction=[4.0, 0.08, 0.003],
                rgba=[0.12, 0.12, 0.12, 1.0],
            )

        cube = spec.worldbody.add_body(name="flyarm_cube", pos=[0.45, 0.0, 0.10])
        cube.add_freejoint(name="flyarm_cube_free")
        cube.add_geom(
            name="flyarm_cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self._CUBE_HALF] * 3,
            mass=0.025,
            friction=[4.0, 0.08, 0.003],
            rgba=[0.06, 0.42, 0.95, 1.0],
        )
        goal = spec.worldbody.add_body(name="flyarm_goal", mocap=True)
        goal.add_geom(
            name="flyarm_goal_marker",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.032, 0.001, 0.0],
            rgba=[0.95, 0.15, 0.12, 0.65],
            contype=0,
            conaffinity=0,
        )
        self.model = spec.compile()
        self.model.opt.timestep = 0.002
        # More solver iterations help the light cube resist a squeeze without a
        # nonphysical attachment or equality constraint.
        self.model.opt.iterations = 100
        self.data = mujoco.MjData(self.model)

        self._joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                for i in range(1, 8)
            ],
            dtype=np.int32,
        )
        if np.any(self._joint_ids < 0):
            raise ValueError("expected Menagerie Panda joints joint1 through joint7")
        self._qadr = self.model.jnt_qposadr[self._joint_ids]
        self._dadr = self.model.jnt_dofadr[self._joint_ids]
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "flyarm_pick_ee")
        self._cube_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "flyarm_cube_free"
        )
        self._cube_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "flyarm_cube_geom"
        )
        self._cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flyarm_cube")
        goal_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flyarm_goal")
        self._left_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self._right_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        required_ids = {
            "flyarm_pick_ee": self._ee_site,
            "flyarm_cube_free": self._cube_joint,
            "flyarm_cube_geom": self._cube_geom,
            "flyarm_cube": self._cube_body,
            "flyarm_goal": goal_body,
            "left_finger": self._left_finger,
            "right_finger": self._right_finger,
        }
        missing = [name for name, identifier in required_ids.items() if identifier < 0]
        if missing:
            raise ValueError(f"compiled Panda task is missing required objects: {missing}")
        self._cube_qadr = int(self.model.jnt_qposadr[self._cube_joint])
        self._cube_dadr = int(self.model.jnt_dofadr[self._cube_joint])
        self._goal_mocap = int(self.model.body_mocapid[goal_body])
        self._servo_ids = np.array(
            [
                next(
                    actuator
                    for actuator in range(self.model.nu)
                    if self.model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
                    and self.model.actuator_trnid[actuator, 0] == joint
                )
                for joint in self._joint_ids
            ],
            dtype=np.int32,
        )
        self._gripper_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        )
        if self._gripper_id < 0:
            raise ValueError("expected Menagerie Panda gripper actuator8")
        finger_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        if finger_joint < 0:
            raise ValueError("expected Menagerie Panda finger_joint1")
        self._finger_qadr = int(self.model.jnt_qposadr[finger_joint])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.observation_dim,), np.float32
        )
        self._renderer: mujoco.Renderer | None = None
        self._desired_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.object_position = np.zeros(3)
        self.goal = np.zeros(3)
        self._steps = self._stable_success_steps = 0
        self._ever_left_contact = self._ever_right_contact = False
        self._ever_grasped = self._ever_lifted = False
        self._teacher_stage = "approach"
        self._teacher_contact_steps = 0
        self.last_joint_command = np.zeros(7)
        self.reset()

    @staticmethod
    def _vector(value: Any, name: str, length: int) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (length,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector with shape ({length},)")
        return result

    def _ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site].copy()

    def _cube_position(self) -> np.ndarray:
        return self.data.xpos[self._cube_body].copy()

    def _finger_contacts(self) -> tuple[bool, bool]:
        left = right = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 == self._cube_geom:
                other = contact.geom2
            elif contact.geom2 == self._cube_geom:
                other = contact.geom1
            else:
                continue
            body = int(self.model.geom_bodyid[other])
            left |= body == self._left_finger
            right |= body == self._right_finger
        return left, right

    def _gripper_opening(self) -> float:
        # The two Panda finger joints are equality-coupled; expose aperture, not
        # the internal duplicated joint coordinate.
        return float(np.clip(self.data.qpos[self._finger_qadr] / 0.04, 0.0, 1.0))

    @property
    def steps(self) -> int:
        """Number of control actions applied in the current episode."""
        return self._steps

    @property
    def teacher_stage(self) -> str:
        """Current demonstrator phase, exposed only for labeled data collection."""
        return self._teacher_stage

    def _observation(self) -> np.ndarray:
        ee, cube = self._ee_position(), self._cube_position()
        left, right = self._finger_contacts()
        return np.concatenate(
            (
                self.data.qpos[self._qadr],  # 7
                self.data.qvel[self._dadr],  # 7
                ee,  # 3
                cube,  # 3
                self.goal,  # 3
                cube - ee,  # 3
                self.goal - cube,  # 3
                [self._gripper_opening()],  # 1
                self.data.qvel[self._cube_dadr : self._cube_dadr + 3],  # 3
                [left, right],  # 2
                [self._ever_grasped, self._ever_lifted],  # 2
            )
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        cube = self._cube_position()
        left, right = self._finger_contacts()
        grasped = left and right
        opening = self._gripper_opening()
        object_height = float(cube[2])
        goal_xy_error = float(np.linalg.norm(cube[:2] - self.goal[:2]))
        return {
            "stage": self._teacher_stage,
            "contact_left": left,
            "contact_right": right,
            "grasped": grasped,
            "ever_grasped": self._ever_grasped,
            "ever_lifted": self._ever_lifted,
            "object_height": object_height,
            "object_speed": float(
                np.linalg.norm(self.data.qvel[self._cube_dadr : self._cube_dadr + 3])
            ),
            "goal_xy_error": goal_xy_error,
            "gripper_opening": opening,
            "is_success": self._stable_success_steps >= 10,
            "ee_position": self._ee_position(),
            "object_position": cube,
            "goal": self.goal.copy(),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # The table locations remain inside the Panda's comfortable vertical
        # workspace.  Keeping the task compact makes contact regression tests
        # deterministic while still varying all object/goal XY coordinates.
        options = options or {}
        self.data.qpos[self._qadr] = self._HOME + self.np_random.uniform(-0.012, 0.012, 7)
        mujoco.mj_forward(self.model, self.data)
        initial_ee = self._ee_position()
        object_xy = (
            self._vector(options["object"], "object", 3)[:2]
            if "object" in options
            else initial_ee[:2] + self.np_random.uniform([-0.055, -0.055], [0.055, 0.055])
        )
        goal_xy = (
            self._vector(options["goal"], "goal", 3)[:2]
            if "goal" in options
            else initial_ee[:2] + self.np_random.uniform([-0.11, -0.11], [0.11, 0.11])
        )
        # Avoid a degenerate no-transport target in random resets.
        if np.linalg.norm(goal_xy - object_xy) < 0.07:
            goal_xy = object_xy + np.array([0.10, 0.0])
        self.goal = np.array([goal_xy[0], goal_xy[1], 0.002])
        self.data.qpos[self._cube_qadr : self._cube_qadr + 3] = [
            object_xy[0],
            object_xy[1],
            self._SETTLED_HEIGHT,
        ]
        self.data.qpos[self._cube_qadr + 3 : self._cube_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._cube_dadr : self._cube_dadr + 6] = 0.0
        self.data.mocap_pos[self._goal_mocap] = self.goal
        self.data.mocap_quat[self._goal_mocap] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        current_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self._ee_site])
        self._desired_orientation = current_quat
        self.object_position = self._cube_position()
        self._steps = self._stable_success_steps = 0
        self._ever_left_contact = self._ever_right_contact = False
        self._ever_grasped = self._ever_lifted = False
        self._teacher_stage = "approach"
        self._teacher_contact_steps = 0
        self.last_joint_command = self.data.qpos[self._qadr].copy()
        self.data.ctrl[self._servo_ids] = self.last_joint_command
        self.data.ctrl[self._gripper_id] = 255.0
        return self._observation(), self._info()

    def _ik_command(self, displacement: np.ndarray) -> np.ndarray:
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site)
        current_quat, rotation_error = np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self._ee_site])
        mujoco.mju_subQuat(rotation_error, self._desired_orientation, current_quat)
        rotation_error = self.data.site_xmat.reshape(3, 3) @ rotation_error
        jacobian = np.vstack((jacp[:, self._dadr], 0.25 * jacr[:, self._dadr]))
        error = np.concatenate((displacement, 0.25 * rotation_error))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-3 * np.eye(6), error)
        command = self.data.qpos[self._qadr] + np.clip(delta, -0.055, 0.055)
        limits = self.model.jnt_range[self._joint_ids]
        return np.clip(command, limits[:, 0] + 0.01, limits[:, 1] - 0.01)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = self._vector(action, "action", 4)
        if np.any(np.abs(action) > 1.0):
            raise ValueError("action components must be in [-1, 1]")
        self.last_joint_command = self._ik_command(action[:3] * 0.014)
        self.data.ctrl[self._servo_ids] = self.last_joint_command
        # Official Menagerie mapping: 0 is closed and 255 is open.
        self.data.ctrl[self._gripper_id] = 127.5 * (float(action[3]) + 1.0)
        mujoco.mj_step(self.model, self.data, nstep=25)
        self.object_position = self._cube_position()
        self._steps += 1
        left, right = self._finger_contacts()
        self._ever_left_contact |= left
        self._ever_right_contact |= right
        self._ever_grasped |= left and right
        self._ever_lifted |= self.object_position[2] >= self._LIFT_HEIGHT
        info = self._info()
        release_ready = (
            self._ever_left_contact
            and self._ever_right_contact
            and self._ever_lifted
            and not left
            and not right
            and self._gripper_opening() > 0.75
            and info["goal_xy_error"] < 0.03
            and info["object_height"] < self._SETTLED_HEIGHT + 0.012
            and info["object_speed"] < 0.06
        )
        self._stable_success_steps = self._stable_success_steps + 1 if release_ready else 0
        info["is_success"] = self._stable_success_steps >= 10
        reward = -float(info["goal_xy_error"])
        if self._ever_lifted:
            reward += 0.2
        if info["is_success"]:
            reward += 5.0
        return (
            self._observation(),
            reward,
            bool(info["is_success"]),
            self._steps >= self.horizon,
            info,
        )

    def teacher_action(self) -> np.ndarray:
        """A state-machine demonstrator that uses the same four-action API.

        It only selects Cartesian motions and gripper controls; it never changes
        cube coordinates or contact state.  A small dwell period after closing
        lets the physical finger contacts settle before lift.
        """
        cube, ee = self._cube_position(), self._ee_position()
        left, right = self._finger_contacts()
        # The control site is at the finger-body center; the colliding Panda
        # pads sit about 4.3 cm below it.  These setpoints therefore put pads
        # around the cube midplane rather than driving them into the floor.
        hover = cube + np.array([0.0, 0.0, 0.155])
        # At this site height, the broad collision mesh spans the cube sides;
        # the small fingertip pads remain clear of the tabletop.
        grasp = cube + np.array([0.0, 0.0, 0.045])
        carry = self.goal + np.array([0.0, 0.0, 0.165])
        # Hold the pads above the cube while it settles at the marker.  Driving
        # the pads down to the floor makes opening fingers sweep the cube aside.
        drop = self.goal + np.array([0.0, 0.0, 0.080])
        retreat = self.goal + np.array([0.0, 0.0, 0.18])
        if self._teacher_stage == "approach" and np.linalg.norm(ee - hover) < 0.014:
            self._teacher_stage = "descend"
        elif self._teacher_stage == "descend" and np.linalg.norm(ee - grasp) < 0.010:
            self._teacher_stage = "close"
        elif self._teacher_stage == "close":
            # The gate is intentionally contact-based.  A fully closed empty
            # gripper is not allowed to progress into a fake lift.
            self._teacher_contact_steps = self._teacher_contact_steps + 1 if left and right else 0
            if self._teacher_contact_steps >= 16:
                self._teacher_stage = "lift"
        elif self._teacher_stage == "lift" and cube[2] >= self._LIFT_HEIGHT:
            self._teacher_stage = "transport"
        elif self._teacher_stage == "transport" and np.linalg.norm(ee - carry) < 0.017:
            self._teacher_stage = "lower"
        elif self._teacher_stage == "lower" and np.linalg.norm(ee - drop) < 0.012:
            self._teacher_stage = "open"
        elif self._teacher_stage == "open" and self._gripper_opening() > 0.80:
            self._teacher_stage = "retreat"

        desired, grip = {
            "approach": (hover, 1.0),
            "descend": (grasp, 1.0),
            "close": (grasp, -1.0),
            "lift": (hover, -1.0),
            "transport": (carry, -1.0),
            "lower": (drop, -1.0),
            "open": (drop, 1.0),
            "retreat": (retreat, 1.0),
        }[self._teacher_stage]
        xyz = np.clip((desired - ee) / 0.014, -1.0, 1.0)
        if self._teacher_stage == "descend":
            # A slower descent avoids bouncing the light cube into MuJoCo's
            # floor-contact softness before the lateral squeeze is engaged.
            xyz[2] = max(xyz[2], -0.25)
        return np.concatenate((xyz, [grip])).astype(np.float32)

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
