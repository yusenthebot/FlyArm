"""The current motor phase, computed only from observable scene predicates (the phase cue).

The scripted teacher is a phase machine (approach, descend, close, move, lift, carry, lower,
release, retreat, clear), and most of what a smooth controller cannot fit is the switching
between phases: one linear map per teacher phase fits the teacher's commands about three times
better than one map overall (docs/ARCHITECTURE_ANALYSIS.md). The teacher's phase is memory;
this module derives the phase a controller should be in from what the scene shows, with the
teacher's own geometric predicates (alignment at the hover, the closing gates, a pinch, a held
object over its destination), never from the teacher's state. It is part of the memoryless cue
(``phase_cue``), like the current skill, and is zero in the no-phase-cue control.

Per subgoal (the current one, in task order), the phase is decided in this order:

- the previous subgoal's hand is still on what it finished: a handle not yet at the teacher's
  stop: ``move``; closed on it: ``release``; open and still within reach of it, below where
  the teacher's retreat ends: ``retreat``;
- a hand closed on nothing it needs: ``clear``;
- articulations: a pad on the handle within reach, at its height: ``move`` (before the
  previous subgoal's release); at the handle within the closing gate, or closing on it:
  ``close``; below the hover over the handle (or aligned at the hover): ``descend``;
  otherwise ``approach``;
- objects held (both pads on the object): over the destination so that it fits (the teacher's
  gate) and down at its surface: ``release``; over the destination: ``lower``; high enough to
  travel: ``carry``; otherwise ``lift`` (a pick subgoal: ``lift``);
- objects not held: placed and settling where the subgoal wants it: ``retreat``; at the grasp
  pose within the closing gates, or closing on the object: ``close``; below the hover over the
  object: ``descend``; otherwise ``approach``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flyarm.manipulation import furniture as fu
from flyarm.manipulation import tasks as tk

PHASE_NAMES = (
    "approach",
    "descend",
    "close",
    "move",
    "lift",
    "carry",
    "lower",
    "release",
    "retreat",
    "clear",
)
PHASE_DIM = len(PHASE_NAMES)
(APPROACH, DESCEND, CLOSE, MOVE, LIFT, CARRY, LOWER, RELEASE, RETREAT, CLEAR) = range(PHASE_DIM)
ARTICULATIONS = (tk.OPEN_DRAWER, tk.CLOSE_DRAWER, tk.OPEN_DOOR, tk.CLOSE_DOOR)
OBJECTS = (tk.PICK, tk.PLACE, tk.STACK)
OPEN = 0.8  # opening above which the hand counts as open (the teacher's release is done)
NEAR = 0.04  # m (xy) within which the hand is still at the handle the previous subgoal moved
NEAR_PLACED = 0.015  # m (xy): the teacher rises straight up from a placed object
SURFACE = 0.006  # m: a lowered object this close to its target height is released there
PINCH_REACH = 0.03  # m from the hand to the handle within which a closed hand holds it
FULL_GAP = 0.08  # m between the pads fully open
PINCHED = 0.18  # opening above the handle's width at which the pads squeeze it (measured)
PINCH_HEIGHT = 0.015  # m: the hand holding a handle is at its height (a descent is above it)
SETTLING = 0.003  # m an object held but not yet risen is still being closed on


def _helper(sim: Any) -> Any:
    """A teacher instance used only for its memoryless geometry (never its phase or memory)."""
    helper = getattr(sim, "_phase_helper", None)
    if helper is None:
        from flyarm.manipulation.teacher import ManipulationTeacher

        helper = ManipulationTeacher(sim)
        sim._phase_helper = helper
    return helper


def observable_phases(sim: Any, leading: np.ndarray) -> np.ndarray:
    """[N] phase index of every environment's current subgoal, -1 when all are done."""
    from flyarm.manipulation import teacher as mt

    helper = _helper(sim)
    scene = helper._scene()
    out = np.full(sim.num_envs, -1, dtype=np.int64)
    for row in range(sim.num_envs):
        if leading[row] >= sim.sub_count[row]:
            continue
        index = int(sim.current(leading)[row])
        out[row] = _phase(sim, helper, mt, scene, row, index)
    return out


def _previous(sim: Any, helper: Any, mt: Any, scene: dict, row: int, index: int) -> int | None:
    """Move, release or retreat while the hand is still at what the previous subgoal finished.

    The scene counts a drawer open at 75% of its travel and a lid at 1.58 rad, and the teacher
    carries the motion on to its own stop (97%, 1.645 rad) before letting go: until the joint
    is there, a hand on the handle is still moving it.
    """
    if index <= 0 or index <= int(sim.preset[row]) - 1:
        return None
    previous = index - 1
    kind = int(sim.sub_kind[row, previous])
    ee = scene["ee"][row]
    opening = float(scene["opening"][row])
    if kind in ARTICULATIONS:
        lid = kind in (tk.OPEN_DOOR, tk.CLOSE_DOOR)
        articulation = 2 if lid else int(sim.sub_target[row, previous])
        site = scene["handles"][row, articulation]
        touching = bool(
            scene["handle_fingers"][0][row, articulation]
            or scene["handle_fingers"][1][row, articulation]
        )
        xy = float(np.linalg.norm(ee[:2] - site[:2]))
        if kind == tk.OPEN_DOOR:  # slide off the standing lid's bar, then step back: no rise
            reach = mt.LID_STEP_BACK + 0.015
            if xy > reach:
                return None
            if touching and opening < OPEN:
                return MOVE if not _finished(sim, mt, scene, row, kind, articulation) else RELEASE
            return RETREAT if opening >= OPEN - 0.1 else RELEASE
        if xy > NEAR:
            return None
        if touching and not _finished(sim, mt, scene, row, kind, articulation):
            return MOVE
        at_height = abs(float(ee[2] - site[2])) < PINCH_HEIGHT
        if opening < OPEN and (touching or at_height):
            return RELEASE
        return RETREAT if ee[2] < site[2] + mt.HOVER - 0.01 else None
    if kind not in (tk.PLACE, tk.STACK):
        return None
    slot = int(sim.sub_obj[row, previous])
    site = scene["grasp_points"][row, slot]
    if float(np.linalg.norm(ee[:2] - site[:2])) > NEAR_PLACED:
        return None
    touching = bool(scene["fingers"][0][row, slot] or scene["fingers"][1][row, slot])
    if opening < OPEN and touching:
        return RELEASE
    top = max(float(scene["top"][row, slot]) + 0.09, site[2] + 0.08, helper._transit_z(row, scene))
    return RETREAT if ee[2] < top - 0.01 else None


def _finished(sim: Any, mt: Any, scene: dict, row: int, kind: int, articulation: int) -> bool:
    """The joint is where the teacher stops moving it (past where the scene counts it)."""
    joint = float(scene["joints"][row, articulation])
    if kind == tk.OPEN_DOOR:
        return joint >= mt.LID_DONE_OPEN
    if kind == tk.CLOSE_DOOR:
        return joint <= mt.LID_DONE_CLOSED
    if kind == tk.OPEN_DRAWER:
        return joint >= mt.DRAWER_DONE_OPEN * float(sim.travel[row, articulation])
    return joint <= mt.DRAWER_DONE_CLOSED


def _on_handle(sim: Any, scene: dict, row: int, index: int, kind: int) -> bool:
    """A pad on the current subgoal's handle, the hand within reach of it and at its height."""
    articulation = 2 if kind in (tk.OPEN_DOOR, tk.CLOSE_DOOR) else int(sim.sub_target[row, index])
    site = scene["handles"][row, articulation]
    ee = scene["ee"][row]
    touching = bool(
        scene["handle_fingers"][0][row, articulation]
        or scene["handle_fingers"][1][row, articulation]
    )
    if not (
        touching
        and float(np.linalg.norm(ee - site)) < PINCH_REACH
        and abs(float(ee[2] - site[2])) < PINCH_HEIGHT
    ):
        return False
    if articulation == 2 and sim.lid_knob[row] < 0.5:
        return True  # the lid's bar swivels in a part-open hand: touching is holding
    # Still closing onto the handle (the pads touch before they squeeze): not yet moving it.
    knob = (sim.lid_knob[row] if articulation == 2 else sim.drawer_knob[row]) > 0.5
    radius = fu.KNOB_RADIUS if knob else fu.BAR_RADIUS
    return float(scene["opening"][row]) < 2 * radius / FULL_GAP + PINCHED


def _phase(sim: Any, helper: Any, mt: Any, scene: dict, row: int, index: int) -> int:
    kind = int(sim.sub_kind[row, index])
    holding = kind in OBJECTS and bool(scene["grasped"][row, int(sim.sub_obj[row, index])])
    if kind in ARTICULATIONS and _on_handle(sim, scene, row, index, kind):
        return MOVE
    finishing = None if holding else _previous(sim, helper, mt, scene, row, index)
    if finishing is not None:
        return finishing
    if kind in ARTICULATIONS:
        return _articulation(sim, helper, mt, scene, row, index, kind)
    return _object(sim, helper, mt, scene, row, index, kind)


def _articulation(
    sim: Any, helper: Any, mt: Any, scene: dict, row: int, index: int, kind: int
) -> int:
    lid = kind in (tk.OPEN_DOOR, tk.CLOSE_DOOR)
    articulation = 2 if lid else int(sim.sub_target[row, index])
    site = scene["handles"][row, articulation]
    hover = site + np.array([0.0, 0.0, mt.HOVER])
    across = lid and sim.lid_knob[row] < 0.5
    grip = mt.LID_BAR_GRIP if across else 1.0
    yaw = float(scene["articulation_headings"][row, articulation])
    ee = scene["ee"][row]
    opening = float(scene["opening"][row])
    opened = (grip + 1.0) / 2.0
    near = float(np.linalg.norm(ee - site)) < PINCH_REACH
    if opening < opened - 0.1:  # closing on nothing yet, or closed on nothing
        return CLOSE if near else CLEAR
    at_height = abs(ee[2] - site[2]) < mt.AT_HEIGHT
    if at_height and helper._straddles(
        row, scene, site, helper._handle_tolerance(row, articulation, grip)
    ):
        return CLOSE
    xy = float(np.linalg.norm(ee[:2] - site[:2]))
    aligned = (
        xy < mt.ALIGN_XY
        and abs(ee[2] - hover[2]) < 0.02
        and helper._yaw_error(row, yaw) < mt.ALIGN_YAW
    )
    if aligned or (ee[2] < hover[2] - 0.02 and xy < mt.ABORT_XY):
        return DESCEND
    return APPROACH


def _object(sim: Any, helper: Any, mt: Any, scene: dict, row: int, index: int, kind: int) -> int:
    slot = int(sim.sub_obj[row, index])
    ee = scene["ee"][row]
    opening = float(scene["opening"][row])
    grasp = scene["grasp_points"][row, slot]
    bottom = float(scene["bottom"][row, slot])
    top = float(scene["top"][row, slot])
    if bool(scene["grasped"][row, slot]):
        risen = float(scene["object_pos"][row, slot, 2] - sim.start_pos[row, slot, 2])
        if kind == tk.PICK or risen < SETTLING:
            return CLOSE if risen < SETTLING else LIFT
        target = scene["targets"][row, index]
        centre = scene["object_pos"][row, slot]
        offset = target[:2] - centre[:2]
        place_yaw = float(scene["place_headings"][row, index])
        over = (
            np.linalg.norm(offset) < 0.008 and helper._yaw_error(row, place_yaw) < 0.05
        ) or helper._placement_fits(row, index, slot, kind, scene, offset, place_yaw)
        if over and target[2] + SURFACE - bottom > -0.004:
            return RELEASE
        safe = helper._safe_bottom(row, scene, slot)
        if over and bottom < safe - 0.01 + 0.03:
            return LOWER
        if bottom >= safe - 0.01:
            return LOWER if over else CARRY
        return LIFT
    # Not held.
    if kind in (tk.PLACE, tk.STACK):
        settled = bool(
            scene["inside"][row, slot, int(np.clip(sim.sub_target[row, index], 0, None))]
            if kind == tk.PLACE
            else scene["stacked_geometry"][row, slot].any()
        )
        if settled:
            return RELEASE if opening < OPEN else RETREAT
    touching = bool(scene["fingers"][0][row, slot] or scene["fingers"][1][row, slot])
    at = helper._pads_on_object(row, ee, grasp, bottom) and helper._straddles(
        row, scene, grasp, helper._object_tolerance(row, slot)
    )
    if opening < OPEN - 0.1:
        return CLOSE if (at or touching) else CLEAR
    if at:
        return CLOSE
    hover = np.array([grasp[0], grasp[1], max(grasp[2] + 0.06, top + 0.075)])
    yaw = helper._pick_yaw(row, slot, scene)
    xy = float(np.linalg.norm(ee[:2] - grasp[:2]))
    aligned = (
        xy < mt.ALIGN_XY
        and abs(ee[2] - hover[2]) < 0.02
        and helper._yaw_error(row, yaw) < mt.ALIGN_YAW
    )
    if aligned or (ee[2] < hover[2] - 0.02 and xy < 0.02):
        return DESCEND
    return APPROACH


def _in_source(sim: Any, scene: dict, row: int, slot: int) -> bool:
    source = int(sim.source[row, slot])
    return source >= 0 and bool(scene["inside"][row, slot, source])


__all__ = ["PHASE_DIM", "PHASE_NAMES", "observable_phases"]
