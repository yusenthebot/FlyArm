"""Start detours: train every skill from hand poses other skills leave behind (research log E64).

Every training template opens its drawer first and opens its lid with the same jaw heading, so
skill-level DAgger only ever starts a skill from the poses the teacher's own previous subgoal
leaves. Held-out compositions and held-out furniture start skills elsewhere: the drawer right
after the lid (``retrieve_to_shelf``), the lid's knob a quarter turn from the bar's heading.

A detour moves the hand, before a subgoal-reset episode starts, to a random point of the
workspace and a random jaw heading, keeping the teacher's grip command (an object that is held
stays held). It rises to a transit height before any lateral move, so it does not sweep through
the furniture. Its steps are recorded with the teacher's labels, the teacher re-deriving its
phase from the scene at every detour step, so each detour state is labelled with the way back;
from there the round runs as before (learner, beta mixing, every visited state labelled).
Nothing about the held-out splits is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyarm.grasp import task as arm_task

# Workspace of the detour targets, inside the 5th to 95th percentile of the hand positions the
# teacher and learners visit on the train split (x forward, y left, z up, metres).
TARGET_LOW = np.array([0.25, -0.40, 0.12])
TARGET_HIGH = np.array([0.65, 0.40, 0.40])
TRANSIT_Z = 0.48  # above every furniture top of the train ranges
NEAR_XY = 0.03  # lateral distance under which the hand descends to the target height
YAW_SPAN = 1.2  # rad either side of the reset heading (the yaw limit is 1.75)


@dataclass(frozen=True)
class Detour:
    """Per environment row: detour length in steps (0 for none), target point and heading."""

    steps: np.ndarray
    target: np.ndarray
    yaw: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.steps)
        if self.target.shape != (n, 3) or self.yaw.shape != (n,):
            raise ValueError("a detour needs one target point and one heading per row")
        if (self.steps < 0).any():
            raise ValueError("detour lengths must be nonnegative")


def sample_detours(
    rows: int, share: float, length: tuple[int, int], generator: np.random.Generator
) -> Detour:
    """A detour for each row with probability ``share``, its length uniform in ``length``."""
    if not 0.0 <= share <= 1.0 or not 1 <= length[0] <= length[1]:
        raise ValueError("share must lie in [0, 1] and length be a positive range")
    chosen = generator.random(rows) < share
    steps = np.where(chosen, generator.integers(length[0], length[1] + 1, rows), 0)
    target = generator.uniform(TARGET_LOW, TARGET_HIGH, (rows, 3))
    yaw = generator.uniform(-YAW_SPAN, YAW_SPAN, rows)
    return Detour(steps=steps.astype(np.int64), target=target, yaw=yaw)


def detour_action(
    ee: np.ndarray, yaw_command: np.ndarray, detour: Detour, grip: np.ndarray
) -> np.ndarray:
    """[N, 5] action toward each row's detour target: rise, move across, then descend."""
    lateral = detour.target[:, :2] - ee[:, :2]
    far = np.linalg.norm(lateral, axis=1) > NEAR_XY
    rising = far & (ee[:, 2] < TRANSIT_Z - 0.02)
    goal_z = np.where(far, TRANSIT_Z, detour.target[:, 2])
    move = np.concatenate(
        [np.where(rising[:, None], 0.0, lateral), (goal_z - ee[:, 2])[:, None]], 1
    )
    action = np.zeros((len(ee), 5))
    action[:, :3] = np.clip(move / arm_task.STEP_METERS, -1.0, 1.0)
    action[:, 3] = np.clip((detour.yaw - yaw_command) / arm_task.YAW_STEP, -1.0, 1.0)
    action[:, 4] = grip
    return action
