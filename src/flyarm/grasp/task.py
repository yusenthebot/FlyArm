"""The grasp task's rules: action scaling, observation layout, shaped reward and success.

Everything here is vectorized over a leading environment axis and shared by the batched
environment and the single environment, so the two cannot drift apart in their rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Control ------------------------------------------------------------------------------------
ACTION_DIM = 5  # dx, dy, dz, dyaw, gripper, each in [-1, 1]
SUBSTEPS = 25  # 2-ms MuJoCo steps per action: 20 Hz control, as in pick-and-place
STEP_METERS = 0.014  # Cartesian step of the EE site at full action, as in pick-and-place
YAW_STEP = 0.05  # rad of gripper yaw per step at full action (under the 0.055 joint clip)
YAW_LIMIT = 1.75  # commanded yaw stays within this of the reset yaw: more than pi / 2
JOINT_STEP = 0.055  # IK joint increment clip, as in pick-and-place
POSTURE_GAIN = 0.05  # null-space pull toward the home posture, per step
PAD_BELOW_SITE = 0.0434  # the rubber pad centres sit this far below the EE site
SITE_FLOOR = 0.058  # lowest EE-site height the teacher commands (fingers touch at ~0.054)

# Task ---------------------------------------------------------------------------------------
LIFT_HEIGHT = 0.10  # object origin must rise this far above its resting height
HOLD_STEPS = 20  # consecutive lifted steps with both fingers in contact
DEFAULT_HORIZON = 200
# Object placement, in the world frame: a band in front of the robot that the Panda reaches
# comfortably at table height (the base is at the origin, the home EE at about x = 0.41).
WORKSPACE_LOW = np.array([0.33, -0.18])
WORKSPACE_HIGH = np.array([0.58, 0.18])

# Reward -------------------------------------------------------------------------------------
REACH_WEIGHT, ALIGN_WEIGHT, GRASP_WEIGHT, LIFT_WEIGHT, HOLD_WEIGHT = 0.5, 0.25, 0.5, 1.0, 1.0
STEP_REWARD_MAX = REACH_WEIGHT + ALIGN_WEIGHT + GRASP_WEIGHT + LIFT_WEIGHT + HOLD_WEIGHT
DEFAULT_GAMMA = 0.99
DEFAULT_SUCCESS_BONUS = 400.0
DEFAULT_ACTION_COST = 0.01


def minimum_success_bonus(gamma: float = DEFAULT_GAMMA) -> float:
    """The stalling floor: the discounted value of collecting the per-step maximum forever.

    A terminal bonus at or below it lets a policy that hovers in the best non-terminal state
    (grasped and lifted but never held long enough, or reached and aligned but never lifted)
    earn as much as one that finishes the task, so the bonus must exceed it.
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    return STEP_REWARD_MAX / (1.0 - gamma)


# Observation layout ---------------------------------------------------------------------------
OBS_FIELDS: tuple[tuple[str, int], ...] = (
    ("joint_pos", 7),
    ("joint_vel", 7),
    ("ee_pos", 3),
    ("gripper_yaw", 2),  # sin 2 phi, cos 2 phi of the jaw-closing axis (two-fold symmetric)
    ("gripper_opening", 1),
    ("object_pos", 3),
    ("object_minus_ee", 3),
    ("object_rot6d", 6),  # first two columns of the object rotation matrix
    ("object_axis_yaw", 2),  # sin 2 psi, cos 2 psi of the object's narrow axis
    ("relative_yaw", 2),  # sin 2 (psi - phi), cos 2 (psi - phi): jaws across the narrow side
    ("object_vel", 3),
    ("object_descriptor", 5),  # box long, narrow, height; grasp offset; grasp pad height
    ("grasp_minus_ee", 3),  # geometry-derived grasp target for the EE site, minus the EE
    ("lift_target", 1),  # object height that counts as lifted
    ("lift_error", 1),  # lift target minus object height
    ("contacts", 2),  # left and right finger touching the object
    ("ever_grasped", 1),
    ("lifted", 1),  # object at or above the lift target now
    ("hold_fraction", 1),  # consecutive held-and-lifted steps / HOLD_STEPS
)
PRIVILEGED_FIELDS: tuple[tuple[str, int], ...] = (
    ("object_mass", 1),
    ("object_angvel", 3),
    ("commanded_yaw", 1),
)
OBS_DIM = sum(size for _, size in OBS_FIELDS)
PRIVILEGED_DIM = OBS_DIM + sum(size for _, size in PRIVILEGED_FIELDS)


def field_slices(fields: tuple[tuple[str, int], ...] = OBS_FIELDS) -> dict[str, slice]:
    slices, start = {}, 0
    for name, size in fields:
        slices[name] = slice(start, start + size)
        start += size
    return slices


# Geometry helpers -----------------------------------------------------------------------------
def closing_axis_yaw(site_xmat: np.ndarray) -> np.ndarray:
    """Yaw of the jaw-closing axis (the EE site's y axis) projected on the table, [N]."""
    matrix = site_xmat.reshape(-1, 3, 3)
    return np.arctan2(matrix[:, 1, 1], matrix[:, 0, 1])


def object_axis_yaw(object_xmat: np.ndarray) -> np.ndarray:
    """Yaw of the object's narrow axis (body y) projected on the table, [N]."""
    matrix = object_xmat.reshape(-1, 3, 3)
    return np.arctan2(matrix[:, 1, 1], matrix[:, 0, 1])


def half_turn_wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap to [-pi/2, pi/2): parallel jaws are the same after a half turn."""
    return np.mod(angle + np.pi / 2, np.pi) - np.pi / 2


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """[N, 4] unit quaternions (w, x, y, z) to [N, 3, 3] rotation matrices."""
    w, x, y, z = (quat[:, index] for index in range(4))
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            np.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        axis=1,
    )


def yaw_matrix(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    zero, one = np.zeros_like(yaw), np.ones_like(yaw)
    return np.stack(
        (np.stack((c, -s, zero), -1), np.stack((s, c, zero), -1), np.stack((zero, zero, one), -1)),
        axis=1,
    )


def rotation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` to ``desired``, both [N, 3, 3].

    Equal to MuJoCo's ``mju_subQuat`` rotated into the world frame, which is what the
    pick-and-place IK uses; this is the same quantity without a per-row Python loop.
    """
    error = desired @ current.transpose(0, 2, 1)
    vee = 0.5 * np.stack(
        (
            error[:, 2, 1] - error[:, 1, 2],
            error[:, 0, 2] - error[:, 2, 0],
            error[:, 1, 0] - error[:, 0, 1],
        ),
        -1,
    )
    sine = np.linalg.norm(vee, axis=1)
    cosine = np.clip((np.trace(error, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arctan2(sine, cosine)
    scale = np.where(sine > 1e-12, angle / np.maximum(sine, 1e-12), 1.0)
    return vee * scale[:, None]


def hinge_jacobian(site: np.ndarray, axes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Site Jacobian of an all-hinge chain, [N, 6, 7]: cross(axis, site - anchor), then axis.

    For the Panda arm this equals ``mj_jacSite`` restricted to the arm joints; computing it from
    each hinge's world axis and anchor vectorizes over environments.
    """
    jacp = np.cross(axes, site[:, None, :] - anchors).transpose(0, 2, 1)
    return np.concatenate((jacp, axes.transpose(0, 2, 1)), axis=1)


def ik_step(
    jacobian: np.ndarray,
    joint_pos: np.ndarray,
    limits: np.ndarray,
    displacement: np.ndarray,
    rotation: np.ndarray,
    posture: np.ndarray | None = None,
) -> np.ndarray:
    """Damped least-squares joint targets for a site displacement and rotation, [N, 7].

    The same weighting (0.25 on rotation), damping and per-joint clip as pick-and-place, plus
    an optional null-space pull toward ``posture``: the Panda has one redundant joint, and with
    yaw commands the pick-and-place IK lets the elbow drift through self-motion into joint
    limits near the base. The pull only moves the arm where it leaves the site still.
    ``jacobian`` is [N, 6, 7]: translational rows, then rotational rows.
    """
    weighted = np.concatenate((jacobian[:, :3], 0.25 * jacobian[:, 3:]), axis=1)
    error = np.concatenate((displacement, 0.25 * rotation), axis=1)[..., None]
    gram = weighted @ weighted.transpose(0, 2, 1) + 1e-3 * np.eye(6)
    pseudo_inverse = weighted.transpose(0, 2, 1) @ np.linalg.inv(gram)  # [N, 7, 6]
    delta = (pseudo_inverse @ error)[..., 0]
    if posture is not None:
        null = np.eye(7) - pseudo_inverse @ weighted
        delta += (null @ (POSTURE_GAIN * (posture - joint_pos))[..., None])[..., 0]
    command = joint_pos + np.clip(delta, -JOINT_STEP, JOINT_STEP)
    return np.clip(command, limits[:, 0] + 0.01, limits[:, 1] - 0.01)


# Reward and success --------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardConfig:
    success_bonus: float = DEFAULT_SUCCESS_BONUS
    action_cost: float = DEFAULT_ACTION_COST
    reach_slope: float = 10.0
    gamma: float = DEFAULT_GAMMA

    def __post_init__(self) -> None:
        floor = minimum_success_bonus(self.gamma)
        if self.success_bonus <= floor:
            raise ValueError(
                f"success_bonus {self.success_bonus} must exceed the stalling floor "
                f"{STEP_REWARD_MAX} / (1 - {self.gamma}) = {floor:.1f}"
            )
        if self.action_cost < 0 or self.reach_slope <= 0:
            raise ValueError("action_cost must be non-negative and reach_slope positive")


def shaped_reward(
    ee: np.ndarray,
    grasp_target: np.ndarray,
    relative_yaw: np.ndarray,
    grasped: np.ndarray,
    height_gain: np.ndarray,
    success: np.ndarray,
    action: np.ndarray,
    config: RewardConfig,
) -> np.ndarray:
    """Staged dense reward: reach and align, grasp, lift, hold; a bonus ends the episode.

    Per-step terms lie in [0, STEP_REWARD_MAX] before the action cost, which only subtracts,
    so the terminal bonus above ``minimum_success_bonus`` keeps finishing better than stalling.
    The reward shapes training only; evaluation reports the success rule.
    """
    reach = 1.0 - np.tanh(config.reach_slope * np.linalg.norm(ee - grasp_target, axis=1))
    align = reach * np.cos(relative_yaw) ** 2
    lift = np.clip(height_gain / LIFT_HEIGHT, 0.0, 1.0)
    lifted = height_gain >= LIFT_HEIGHT
    reward = (
        REACH_WEIGHT * reach
        + ALIGN_WEIGHT * align
        + GRASP_WEIGHT * grasped
        + LIFT_WEIGHT * grasped * lift
        + HOLD_WEIGHT * (grasped & lifted)
        - config.action_cost * np.sum(action**2, axis=1)
        + config.success_bonus * success
    )
    return reward.astype(np.float32)


def update_hold(hold: np.ndarray, grasped: np.ndarray, height_gain: np.ndarray) -> np.ndarray:
    """Consecutive steps with both fingers on the object and the object lifted."""
    return np.where(grasped & (height_gain >= LIFT_HEIGHT), hold + 1, 0)


def is_success(hold: np.ndarray) -> np.ndarray:
    return hold >= HOLD_STEPS
