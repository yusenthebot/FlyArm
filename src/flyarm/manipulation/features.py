"""Control-scale input features for manipulation controllers (a fixed policy-side expansion).

The scripted teacher is a proportional controller with a gain of one full action per control
step of error (``STEP_METERS``, 14 mm; ``YAW_STEP``, 0.05 rad) and phase gates at millimetre
and hundredth-of-a-radian tolerances (aligned within 8 mm and 0.05 rad before descending,
centred within 2 to 3 mm before closing). The observation carries the offsets it acts on
(hand to handle, object, grasp and place targets; heading errors), but normalized by their
spread over whole episodes (about 0.1 m), where a smooth regressor cannot resolve the few
millimetres that decide between hovering and descending. docs/MANIPULATION_ENV.md ("Skill-level
DAgger failure analysis") shows the learners stalling there.

``control_expansion`` appends tanh(offset / scale) of every such offset at two scales: one
control step (fine) and a few (coarse). It is fixed, not trained, and computed inside the
policy from the observation it already receives, so the environment is unchanged.
"""

from __future__ import annotations

from flyarm.manipulation import sim as ms
from flyarm.whole_brain.policy import Expansion

POSITION_SCALES = (0.01, 0.04)  # m: under one control step, and about three
ANGLE_SCALES = (0.05, 0.2)  # rad: one yaw step, and four
SLOT_OFFSETS = ("minus_ee",)
FURNITURE_OFFSETS = ("drawer_handle_minus_ee", "lid_handle_minus_ee")
CUE_OFFSETS = ("object_minus_ee", "target_minus_ee", "target_minus_object", "handle_minus_ee")
CUE_ANGLES = ("grasp_heading_error", "place_heading_error")


def _layout(fields: tuple[tuple[str, int], ...], start: int) -> tuple[dict[str, range], int]:
    out = {}
    for name, size in fields:
        out[name] = range(start, start + size)
        start += size
    return out, start


def control_indices() -> tuple[list[int], list[int]]:
    """Observation indices of the hand-relative offsets and of the heading errors."""
    _, end = _layout(ms.ROBOT_FIELDS, 0)
    positions: list[int] = []
    for _ in range(ms.S):
        slot, end = _layout(ms.SLOT_FIELDS, end)
        positions += [i for name in SLOT_OFFSETS for i in slot[name]]
    furniture, end = _layout(ms.FURNITURE_FIELDS, end)
    positions += [i for name in FURNITURE_OFFSETS for i in furniture[name]]
    cue, end = _layout(ms.CUE_FIELDS, end)
    if end != ms.OBS_DIM:
        raise ValueError("observation layout changed; update the control features")
    positions += [i for name in CUE_OFFSETS for i in cue[name]]
    angles = [i for name in CUE_ANGLES for i in cue[name]]
    return positions, angles


def control_expansion() -> Expansion:
    positions, angles = control_indices()
    index: list[int] = []
    scale: list[float] = []
    for s in POSITION_SCALES:
        index += positions
        scale += [s] * len(positions)
    for s in ANGLE_SCALES:
        index += angles
        scale += [s] * len(angles)
    return index, scale


__all__ = ["control_expansion", "control_indices"]
