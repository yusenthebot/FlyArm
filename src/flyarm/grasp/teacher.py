"""Privileged scripted grasp teacher, closed loop over the object's current pose.

Every step it recomputes the grasp target from the object pose it reads now (the grasp point
on the principal axis, the geometry-derived pad height, the narrow axis the jaws must close
across) and outputs the same 5-D action a controller would. It never writes simulator state.
Its only memory is a phase and three counters, and a resync rule re-derives the phase from the
physical state, so it can label states that a learner reached by other paths (DAgger).

Phases: ``approach`` (open, rise if low, move above the grasp point, turn the jaws across the
narrow axis), ``descend`` (open, go down to the grasp height, keep aligned), ``close``
(hold still, close, wait for both fingers to touch for a few steps), ``lift`` (closed, rise
until the object is well past the lift target, then hold there).
"""

from __future__ import annotations

import numpy as np

from flyarm.grasp import task
from flyarm.grasp.sim import GraspSim

APPROACH, DESCEND, CLOSE, LIFT = 0, 1, 2, 3
PHASES = ("approach", "descend", "close", "lift")
HOVER_CLEARANCE = 0.075  # EE site above the object top while moving sideways (pads ~5 cm below)
ALIGN_XY = 0.008  # m, EE over the grasp point before descending
ALIGN_YAW = 0.06  # rad, jaws across the narrow axis before descending
ABORT_XY = 0.02  # the object moved away during the descent: back to approach
ABORT_YAW = 0.15
AT_HEIGHT = 0.008  # m, close when the site is this near its grasp height
SETTLE_STEPS = 6  # both fingers in contact this long before lifting
CLOSE_TIMEOUT = 30  # closing this long without a grasp is a miss: reopen and retry
DROP_STEPS = 4  # lost contact this long while lifting: reopen and retry
LIFT_MARGIN = 0.03  # lift the object this far past the lift target
DESCENT_SPEED = 0.6  # fraction of the full step while descending onto the object


class GraspTeacher:
    """Scripted demonstrator for every environment of a :class:`GraspSim`."""

    def __init__(self, sim: GraspSim) -> None:
        self.sim = sim
        n = sim.num_envs
        self.phase = np.zeros(n, dtype=np.int64)
        self.contact_steps = np.zeros(n, dtype=np.int64)
        self.close_steps = np.zeros(n, dtype=np.int64)
        self.lost_steps = np.zeros(n, dtype=np.int64)
        self.anchor = np.zeros((n, 2))  # EE xy held while closing and lifting

    def reset(self, ids: np.ndarray | None = None) -> None:
        ids = self.sim.rows if ids is None else np.asarray(ids, dtype=np.int64)
        for counter in (self.phase, self.contact_steps, self.close_steps, self.lost_steps):
            counter[ids] = 0

    def phase_names(self) -> list[str]:
        return [PHASES[int(phase)] for phase in self.phase]

    def act(self) -> np.ndarray:
        """One action per environment for the current state; advances the phase memory."""
        sim = self.sim
        ee, target = sim.ee(), sim.grasp_target()
        left, right = sim.contacts()
        grasped = left & right
        opening = sim.gripper_opening()
        descriptor = sim.descriptors[sim.object_index]
        top = sim.object_pos()[:, 2] + descriptor[:, 2] / 2
        hover_z = np.maximum(target[:, 2] + 0.06, top + HOVER_CLEARANCE)
        xy_error = np.linalg.norm(ee[:, :2] - target[:, :2], axis=1)
        yaw_error = task.half_turn_wrap(sim.object_yaw() - sim.closing_yaw())
        # Closed loop on the object's own height: an object that hangs tilted in the jaws sits
        # lower than planned, so keep rising until it is LIFT_MARGIN past the lift target.
        lift_z = ee[:, 2] + task.LIFT_HEIGHT + LIFT_MARGIN - sim.height_gain()

        self._resync(grasped, opening, xy_error, yaw_error, ee, target, hover_z)
        phase = self.phase
        # Transitions -----------------------------------------------------------------------
        ready = (
            (phase == APPROACH)
            & (xy_error < ALIGN_XY)
            & (np.abs(ee[:, 2] - hover_z) < 0.02)
            & (np.abs(yaw_error) < ALIGN_YAW)
            & (opening > 0.9)
        )
        aborted = (phase == DESCEND) & ((xy_error > ABORT_XY) | (np.abs(yaw_error) > ABORT_YAW))
        arrived = (phase == DESCEND) & (np.abs(ee[:, 2] - target[:, 2]) < AT_HEIGHT) & ~aborted
        closing = phase == CLOSE
        self.contact_steps = np.where(closing & grasped, self.contact_steps + 1, 0)
        self.close_steps = np.where(closing, self.close_steps + 1, 0)
        settled = closing & (self.contact_steps >= SETTLE_STEPS)
        missed = closing & ~grasped & (self.close_steps >= CLOSE_TIMEOUT)
        lifting = phase == LIFT
        self.lost_steps = np.where(lifting & ~grasped, self.lost_steps + 1, 0)
        dropped = lifting & (self.lost_steps >= DROP_STEPS)
        phase = np.where(ready, DESCEND, phase)
        phase = np.where(aborted | missed | dropped, APPROACH, phase)
        phase = np.where(arrived, CLOSE, phase)
        phase = np.where(settled, LIFT, phase)
        entering = (phase == CLOSE) & (self.phase != CLOSE)
        self.anchor[entering] = ee[entering, :2]
        self.phase = phase
        self.close_steps = np.where(phase == CLOSE, self.close_steps, 0)

        # Targets ---------------------------------------------------------------------------
        desired = ee.copy()
        grip = np.ones(sim.num_envs)
        approach = phase == APPROACH
        low = approach & (ee[:, 2] < hover_z - 0.02) & (xy_error > 0.02)
        desired[approach] = np.column_stack((target[approach, :2], hover_z[approach]))
        desired[low, :2] = ee[low, :2]  # rise straight up before moving sideways
        descend = phase == DESCEND
        desired[descend] = target[descend]
        wait_open = descend & (opening < 0.85)  # a learner closed early: open before going on
        desired[wait_open, 2] = ee[wait_open, 2]
        close = phase == CLOSE
        desired[close, 2] = target[close, 2]
        grip[close] = -1.0
        lift = phase == LIFT
        # Hold the xy where the jaws closed: re-commanding the current (sagging) joints would
        # let the arm creep sideways under the load.
        desired[close | lift, :2] = self.anchor[close | lift]
        desired[lift, 2] = lift_z[lift]
        grip[lift] = -1.0

        xyz = np.clip((desired - ee) / task.STEP_METERS, -1.0, 1.0)
        xyz[descend, 2] = np.maximum(xyz[descend, 2], -DESCENT_SPEED)
        turn = self._yaw_action(sim.object_yaw())
        turn[close | lift] = 0.0
        return np.column_stack((xyz, turn, grip)).astype(np.float32)

    def _yaw_action(self, object_yaw: np.ndarray) -> np.ndarray:
        """Turn the commanded jaw axis onto the object's narrow axis within the yaw limit."""
        sim = self.sim
        delta = task.half_turn_wrap(object_yaw - sim.commanded_closing_yaw())
        goal = sim.yaw_command + delta
        delta = np.where(goal > task.YAW_LIMIT, delta - np.pi, delta)
        delta = np.where(goal < -task.YAW_LIMIT, delta + np.pi, delta)
        return np.clip(delta / task.YAW_STEP, -1.0, 1.0)

    def _resync(
        self,
        grasped: np.ndarray,
        opening: np.ndarray,
        xy_error: np.ndarray,
        yaw_error: np.ndarray,
        ee: np.ndarray,
        target: np.ndarray,
        hover_z: np.ndarray,
    ) -> None:
        """Re-derive the phase for states the teacher's own trajectory does not visit.

        None of these fire on its own rollouts: it never touches the object with both fingers
        while approaching or descending, and it is never below the hover height and centred
        above the grasp point with open jaws while still approaching.
        """
        phase = self.phase
        squeezing = ((phase == APPROACH) | (phase == DESCEND)) & grasped
        centred = (
            (phase == APPROACH)
            & (xy_error < ALIGN_XY)
            & (np.abs(yaw_error) < ALIGN_YAW)
            & (ee[:, 2] < hover_z - 0.02)
            & (ee[:, 2] > target[:, 2])
            & (opening > 0.9)
        )
        self.anchor[squeezing] = ee[squeezing, :2]
        phase = np.where(squeezing, CLOSE, phase)
        phase = np.where(centred, DESCEND, phase)
        self.phase = phase
