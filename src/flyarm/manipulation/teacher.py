"""Privileged scripted teacher that executes any task by chaining skill controllers.

Every step it reads the scene (what the environment's cue is computed from: the first subgoal
that is not done) and runs the controller of that subgoal's skill, recomputed from the current
state: approach a handle, pinch it, pull or push it along its joint and let go; or pick an
object with the grasp task's rule (jaws across its narrow side, pads at its geometry-derived
height), carry it above everything, turn it to the receptacle's axis, lower it until its bottom
reaches the target surface (a stack's target is the base object's top) and release.

Its memory is the subgoal it is executing, a phase, a counter and a held xy. It finishes an
articulation (release and step back) or a placement (release and rise) before it moves on, even
when the scene already counts the subgoal done mid-motion, and it re-derives its phase from the
scene when the subgoal changes (an object already in the hand is carried on, not re-picked), so
it can label states a learner reached for DAgger. It never writes simulator state.
"""

from __future__ import annotations

import math

import numpy as np

from flyarm.grasp import task as arm_task
from flyarm.manipulation import furniture as fu
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.sim import ManipulationSim, _lid_rotation

(
    APPROACH,
    DESCEND,
    CLOSE,
    MOVE,
    LIFT,
    CARRY,
    LOWER,
    RELEASE,
    RETREAT,
    FINISHED,
    CLEAR,
) = range(11)
PHASES = (
    "approach",
    "descend",
    "close",
    "move",
    "lift",
    "carry",
    "lower",
    "release",
    "retreat",
    "finished",
    "clear",
)
HOVER = 0.07
ALIGN_XY = 0.008
ALIGN_YAW = 0.05
ABORT_XY = 0.025
AT_HEIGHT = 0.007
SETTLE_STEPS = 6
CLOSE_TIMEOUT = 30
LOST_STEPS = 4
SAFE_MARGIN = 0.05  # carried objects clear every furniture top and object by this much
DESCENT = 0.6
DRAWER_STEP = 0.02  # m of drawer travel commanded ahead of the handle per step
# rad of lid angle commanded ahead of the handle per step. Opening lifts the lid's weight, which
# a stiff position servo only pushes through with a lead of about 0.16 rad (the teacher's removed
# integral used to add the rest); closing past vertical with that lead pulls the pinch off the bar.
LID_OPEN_STEP = 0.16
LID_CLOSE_STEP = 0.08
# Aim past the joint's stop, so the servos' steady-state lag does not leave the drawer or the
# lid a few millimetres short; the stop itself ends the motion.
OVERSHOOT = 0.015  # m
OVERSHOOT_ANGLE = 0.1  # rad
LID_OPEN_GOAL = fu.LID_OPEN_LIMIT + OVERSHOOT_ANGLE
DRAWER_DONE_OPEN = 0.97
LID_DONE_OPEN = 1.645  # push it onto its stop (1.66): gravity barely holds it near vertical
DRAWER_DONE_CLOSED = 0.0015
LID_DONE_CLOSED = 0.008
SLIP = 0.035
PARTIAL_OPEN = 0.7
TRANSIT_CLEARANCE = 0.09  # EE site above the tallest furniture top when travelling empty
CENTRING_STEP = 0.002
LID_STEP_BACK = 0.1  # m toward the robot after letting go of an opened lid
LID_SLIDE = 0.08  # m along the lid bar to slide the fingers off it
LID_BAR_GRIP = -0.3  # grip command that opens the pads about 1.4 cm each side of the bar
FINISHED_PATIENCE = 30  # steps a finished motion waits for the scene to count its subgoal
ARRIVING = 0.15  # m from the place target inside which the hand turns before moving on
TURN_BEFORE_ARRIVING = 0.3  # rad of heading error that still counts as turning
TRAVEL_SINK = 0.002  # m per control step
PHASE_PATIENCE = 250  # steps any phase may last before the subgoal starts over
# Memoryless re-entry into the descent: a hand already below its hover point, within this of the
# handle or grasp point in xy, turned within REENTRY_YAW and open, continues down instead of
# being sent back up to the hover to pass the strict alignment gate there. The teacher itself
# always passes that gate on the way, but a learner that descends without the pause otherwise
# gets "rise" labels at states the teacher's own data labels "descend" or "close", and stalls
# there (docs/MANIPULATION_ENV.md, "Skill-level DAgger failure analysis"). For objects, not
# while the object settles where it was placed (_object_down).
REENTRY_XY = 0.012
REENTRY_YAW = 0.15
# Proportional commands with a floor while aligning (APPROACH, DESCEND, CARRY, LOWER): a command
# whose proportional size is under COMMAND_FLOOR is raised to it, unless the error is already inside
# half a floor step (so the hand cannot overshoot by more than that). A learner a few millimetres
# off then gets a label that moves it, never a near-zero hover (a P command at 3 mm is 0.2, and
# regressions smoothed such labels to zero; docs/MANIPULATION_ENV.md).
COMMAND_FLOOR = 0.25
# Closing gates from geometry rather than a fixed 2 to 3 mm: the jaws close once the target lies
# between the pads with MARGIN to spare along the jaw axis (half the opening the grip command
# settles at, minus the handle's radius or the object's half width), the pads overlap it across
# the jaw axis (PAD_HALF, the Panda fingertip pad's half width, for a vertical bar or a knob), and
# the pad height is within AT_HEIGHT (objects: _pads_on_object). Capped (10 mm for handles, 8 mm
# for objects), so the fingers still meet an object together instead of one pushing it far
# before the other arrives; the closing hand keeps servoing to the grasp point and centring on
# the finger contacts, as before.
FULL_GAP = 0.08  # m between the pads with the gripper fully open
PAD_HALF = 0.0085
MARGIN = 0.003
ARTICULATION_CAP = 0.01
OBJECT_CAP = 0.008
PAD_DEPTH = 0.02  # m below the grasp point the pads may still close on an object


def _angle(vector: np.ndarray) -> float:
    return math.atan2(float(vector[1]), float(vector[0]))


class ManipulationTeacher:
    """Scripted demonstrator for every environment of a :class:`ManipulationSim`."""

    def __init__(self, sim: ManipulationSim) -> None:
        self.sim = sim
        n = sim.num_envs
        self.active = np.full(n, -1, dtype=np.int64)
        self.phase = np.full(n, FINISHED, dtype=np.int64)
        self.counter = np.zeros(n, dtype=np.int64)
        self.close_steps = np.zeros(n, dtype=np.int64)
        self.retry = np.zeros(n, dtype=bool)
        self.anchor = np.zeros((n, 2))
        self.previous_z = np.zeros(n)
        self.grip_offset = np.zeros((n, 3))
        self.centring = np.zeros((n, 2))
        self.seen = np.zeros((n, 2), dtype=bool)
        self.phase_steps = np.zeros(n, dtype=np.int64)
        self.release_z = np.zeros(n)
        self.single = np.zeros(n, dtype=np.int64)  # consecutive steps with one pad touching

    def reset(self, ids: np.ndarray | None = None) -> None:
        ids = self.sim.rows if ids is None else np.asarray(ids, dtype=np.int64)
        self.active[ids] = -1
        self.phase[ids] = FINISHED
        self.counter[ids] = 0
        self.close_steps[ids] = 0
        self.retry[ids] = False
        self.phase_steps[ids] = 0

    def phase_names(self) -> list[str]:
        return [PHASES[int(phase)] for phase in self.phase]

    # ------------------------------------------------------------------------------ inputs
    def _scene(self) -> dict[str, np.ndarray]:
        sim = self.sim
        state = sim.scene_state()
        done = sim.subgoal_done(sim.effects(state))
        leading = sim.leading(done)
        rotation = sim.object_rotation()
        return {
            **state,
            "leading": leading,
            "current": sim.current(leading),
            "ee": sim.ee(),
            "opening": sim.gripper_opening(),
            "handles": sim.handle_sites(),
            "grasp_points": sim.grasp_points(),
            "object_yaw": np.arctan2(rotation[..., 1, 1], rotation[..., 0, 1]),
            "pick_headings": (pick := sim.pick_headings(state)),
            "place_headings": sim.place_headings(pick),
            "articulation_headings": sim.articulation_headings(),
            "object_pos": sim.object_pos(),
            "targets": sim.place_targets(),
            "receptacle_rot": sim.receptacle_frames()[1],
            "rotation": rotation,
            "fingers": sim.finger_contacts(),
            "handle_fingers": sim.handle_contacts(),
            "jaw_axis": sim.ee_rotation()[:, :2, 1],  # world xy of the site's y (left finger)
        }

    def _safe_bottom(self, row: int, scene: dict[str, np.ndarray], carried: int) -> float:
        """Lowest height a carried object's bottom may travel at: above everything else."""
        sim = self.sim
        episode = sim.episodes[row]
        assert episode is not None
        config = episode.furniture
        tops = [
            config.drawer_cabinet_height,
            config.cabinet_height,
            fu.BIN_FLOOR + fu.BIN_WALL_HEIGHT,
        ]
        for slot in range(tk.MAX_OBJECTS):
            if sim.present[row, slot] and slot != carried:
                tops.append(float(scene["top"][row, slot]))
        return max(tops) + SAFE_MARGIN

    # ------------------------------------------------------------------------------ act
    def act(self) -> np.ndarray:
        """One action per environment for the current state; advances the teacher's memory."""
        sim = self.sim
        scene = self._scene()
        actions = np.zeros((sim.num_envs, 5), dtype=np.float32)
        for row in range(sim.num_envs):
            before = int(self.phase[row])
            self._choose_subgoal(row, scene)
            actions[row] = self._act_row(row, scene)
            same = int(self.phase[row]) == before
            self.phase_steps[row] = self.phase_steps[row] + 1 if same else 0
        return actions

    def _choose_subgoal(self, row: int, scene: dict[str, np.ndarray]) -> None:
        sim = self.sim
        current = int(scene["current"][row])
        complete = scene["leading"][row] >= sim.sub_count[row]
        active = int(self.active[row])
        if complete:
            if active >= 0 and self.phase[row] not in (FINISHED,):
                return  # finish the motion in hand (release, step back)
            self.active[row] = int(sim.sub_count[row]) - 1
            self.phase[row] = FINISHED
            return
        # Watchdogs: a finished motion whose subgoal the scene still does not count, or any phase
        # that makes no progress for too long, gives up the motion in hand.
        if self.phase_steps[row] > PHASE_PATIENCE and self.phase[row] != FINISHED:
            self.phase[row] = FINISHED if active != current else CLEAR
            self.counter[row] = 0
            self.phase_steps[row] = 0
        if current == active:
            if self.phase[row] == FINISHED and self.phase_steps[row] > FINISHED_PATIENCE:
                self.phase[row] = CLEAR
                self.counter[row] = 0
                self.phase_steps[row] = 0
            return
        phase = int(self.phase[row])
        kind = int(sim.sub_kind[row, active]) if active >= 0 else tk.NONE
        new_kind = int(sim.sub_kind[row, current])
        new_obj = int(sim.sub_obj[row, current])
        busy_articulation = kind in (
            tk.OPEN_DRAWER,
            tk.CLOSE_DRAWER,
            tk.OPEN_DOOR,
            tk.CLOSE_DOOR,
        ) and phase in (CLOSE, MOVE, RELEASE, RETREAT)
        busy_placing = kind in (tk.PLACE, tk.STACK) and phase in (RELEASE, RETREAT)
        if (busy_articulation or busy_placing) and active < current:
            return  # finish letting go before the next subgoal
        self.active[row] = current
        self.counter[row] = 0
        holding = scene["opening"][row] < PARTIAL_OPEN
        carrying = new_kind in (tk.PICK, tk.PLACE, tk.STACK) and bool(
            scene["grasped"][row, new_obj]
        )
        if carrying:
            self.phase[row] = LIFT
            self.anchor[row] = scene["ee"][row, :2]
        elif holding:
            self.phase[row] = CLEAR  # let go of whatever is in the hand, then start
        else:
            self.phase[row] = APPROACH

    def _act_row(self, row: int, scene: dict[str, np.ndarray]) -> np.ndarray:
        sim = self.sim
        index = int(self.active[row])
        ee = scene["ee"][row]
        if index < 0 or self.phase[row] == FINISHED:
            return self._command(row, ee, ee, 1.0, None)
        kind = int(sim.sub_kind[row, index])
        if kind in (tk.PICK, tk.PLACE, tk.STACK) and self.phase[row] in (CLEAR, APPROACH):
            if self._object_down(row, index, scene):
                self.phase[row] = DESCEND  # memoryless: see _object_down
                self.counter[row] = 0
        if self.phase[row] == CLEAR:
            return self._clear(row, scene)
        if kind in (tk.OPEN_DRAWER, tk.CLOSE_DRAWER, tk.OPEN_DOOR, tk.CLOSE_DOOR):
            return self._articulate(row, index, kind, scene)
        return self._pick_place(row, index, kind, scene)

    # ------------------------------------------------------------------------------ helpers
    def _command(
        self,
        row: int,
        ee: np.ndarray,
        desired: np.ndarray,
        grip: float,
        yaw: float | None,
        descent: float = 1.0,
        speed: float = 1.0,
        floor: float = 0.0,
    ) -> np.ndarray:
        """P control toward ``desired``, a function of the scene only (``floor``: COMMAND_FLOOR).

        An integral term on the xy error used to make up for the hand sagging under its own
        weight between steps; with the arm gravity-compensated it is not needed, and it was
        hidden state that no policy imitating the teacher could see (docs/MANIPULATION_ENV.md,
        "Imitation failure analysis").
        """
        xyz = (desired - ee) / arm_task.STEP_METERS
        if floor > 0.0:
            size = float(np.linalg.norm(xyz))
            if floor / 2 < size < floor:
                xyz = xyz * (floor / size)
        xyz = np.clip(xyz, -speed, speed)
        xyz[2] = max(xyz[2], -descent)
        turn = 0.0 if yaw is None else self._turn(row, yaw, floor)
        return np.array([*xyz, turn, grip], dtype=np.float32)

    def _turn(self, row: int, yaw: float, floor: float = 0.0) -> float:
        """Turn the commanded jaw axis onto ``yaw`` (mod pi) within the yaw limit."""
        sim = self.sim
        commanded = float(sim.commanded_closing_yaw()[row])
        delta = float(arm_task.half_turn_wrap(np.array(yaw - commanded)))
        goal = float(sim.yaw_command[row]) + delta
        if goal > arm_task.YAW_LIMIT:
            delta -= math.pi
        elif goal < -arm_task.YAW_LIMIT:
            delta += math.pi
        turn = delta / arm_task.YAW_STEP
        if floor / 2 < abs(turn) < floor:
            turn = math.copysign(floor, turn)
        return float(np.clip(turn, -1.0, 1.0))

    def _yaw_error(self, row: int, yaw: float) -> float:
        closing = float(self.sim.closing_yaw()[row])
        return abs(float(arm_task.half_turn_wrap(np.array(yaw - closing))))

    def _retreat(self, row: int, ee: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Rise to ``up``; done there, or when the arm stops rising (the edge of its reach)."""
        rising = ee[2] - self.previous_z[row] > 0.0005
        self.counter[row] = 0 if rising else self.counter[row] + 1
        self.previous_z[row] = ee[2]
        if ee[2] >= up[2] - 0.01 or self.counter[row] >= 10:
            self.phase[row] = APPROACH if self.retry[row] else FINISHED
            self.retry[row] = False
            self.counter[row] = 0
        return self._command(row, ee, up, 1.0, None)

    def _release(self, row: int, ee: np.ndarray, scene: dict[str, np.ndarray]) -> np.ndarray:
        """Open the hand where it is; the retreat then rises straight up from this xy."""
        if self.counter[row] == 0:
            self.anchor[row] = ee[:2]
            self.release_z[row] = ee[2]
        self.counter[row] += 1
        if scene["opening"][row] > 0.8 or self.counter[row] > 12:
            self.phase[row] = RETREAT
            self.counter[row] = 0
        return self._command(row, ee, np.array([*self.anchor[row], ee[2]]), 1.0, None)

    def _centre(self, row: int, scene: dict[str, np.ndarray], left: bool, right: bool) -> None:
        """While closing, move the hand toward the only finger touching, so both close on it.

        The fingers are coupled: when one pad meets a handle or a heavy object first, the other
        stops short. The arm is stiff, so the hand has to shift itself.
        """
        self.single[row] = self.single[row] + 1 if left != right else 0
        if left != right and self.single[row] >= 3 and not self.seen[row].all():
            direction = scene["jaw_axis"][row] * (1.0 if left else -1.0)
            self.centring[row] = np.clip(
                self.centring[row] + CENTRING_STEP * direction, -0.015, 0.015
            )

    def _transit_z(self, row: int, scene: dict[str, np.ndarray]) -> float:
        """EE height for moving sideways with an empty hand: fingertips above all furniture."""
        return self._safe_bottom(row, scene, -1) - SAFE_MARGIN + TRANSIT_CLEARANCE

    def _approach(
        self,
        row: int,
        scene: dict[str, np.ndarray],
        hover: np.ndarray,
        yaw: float,
        grip: float = 1.0,
    ) -> tuple[np.ndarray, bool]:
        """Command toward ``hover``, travelling above the furniture; True once aligned there."""
        ee = scene["ee"][row]
        xy_error = float(np.linalg.norm(ee[:2] - hover[:2]))
        self.centring[row] = 0.0
        self.seen[row] = False
        if xy_error > 0.03:
            transit = max(hover[2], self._transit_z(row, scene))
            if ee[2] < transit - 0.02:
                rise = np.array([ee[0], ee[1], transit])
                return self._command(row, ee, rise, grip, yaw, floor=COMMAND_FLOOR), False
            # Sink only slowly while still travelling (an opened lid stands up to 0.45 m high):
            # TRAVEL_SINK per step toward the transit height, the rate at which the hand used to
            # sag under its own weight before the arm was gravity-compensated, so the episodes
            # keep the time budget they were measured with.
            height = max(transit, ee[2] - TRAVEL_SINK)
            travel = np.array([hover[0], hover[1], height])
            return self._command(row, ee, travel, grip, yaw, floor=COMMAND_FLOOR), False
        opened = (grip + 1.0) / 2.0  # the opening this grip command settles at
        aligned = (
            xy_error < ALIGN_XY
            and abs(ee[2] - hover[2]) < 0.02
            and self._yaw_error(row, yaw) < ALIGN_YAW
            and abs(scene["opening"][row] - opened) < 0.1
        )
        return self._command(row, ee, hover, grip, yaw, floor=COMMAND_FLOOR), aligned

    def _already_down(
        self,
        row: int,
        scene: dict[str, np.ndarray],
        site: np.ndarray,
        hover: np.ndarray,
        yaw: float,
        grip: float = 1.0,
    ) -> bool:
        """The hand is below the hover, over the site, roughly turned and open (REENTRY_XY)."""
        ee = scene["ee"][row]
        opened = (grip + 1.0) / 2.0
        return (
            ee[2] < hover[2] - 0.02
            and float(np.linalg.norm(ee[:2] - site[:2])) < REENTRY_XY
            and self._yaw_error(row, yaw) < REENTRY_YAW
            and abs(float(scene["opening"][row]) - opened) < 0.15
        )

    def _object_down(self, row: int, index: int, scene: dict[str, np.ndarray]) -> bool:
        """The open hand is already down at the object it has to pick, turned onto it, and the
        object is not settling in a receptacle it was placed in or on another object.

        The teacher's own path reaches such a state only through the hover's alignment gate,
        so from it the teacher's data says "descend" or "close"; a fresh teacher (a learner's
        state, or its own CLEAR after a subgoal) said "rise to the hover", the same state with
        opposite labels by path, and learners stalled there with the hand open at the grasp
        point (docs/MANIPULATION_ENV.md, "Place and stack"). A just-placed object is excluded:
        rising from it while it settles is how the teacher lets the rest test count, and a pick
        subgoal can read undone for a moment while it does.
        """
        sim = self.sim
        slot = int(sim.sub_obj[row, index])
        if bool(scene["grasped"][row, slot]):
            return False
        source = int(sim.source[row, slot])
        inside = scene["inside"][row, slot].copy()
        if source >= 0:
            inside[source] = False
        if inside.any() or bool(scene["stacked_geometry"][row, slot].any()):
            return False
        grasp = scene["grasp_points"][row, slot]
        top = float(scene["top"][row, slot])
        hover = np.array([grasp[0], grasp[1], max(grasp[2] + 0.06, top + 0.075)])
        return self._already_down(row, scene, grasp, hover, self._pick_yaw(row, slot, scene))

    def _straddles(
        self, row: int, scene: dict[str, np.ndarray], target: np.ndarray, tolerance: np.ndarray
    ) -> bool:
        """The xy offset of ``target`` from the hand, split along and across the jaw axis, is
        within ``tolerance`` (along, across)."""
        offset = target[:2] - scene["ee"][row, :2]
        axis = scene["jaw_axis"][row] / max(float(np.linalg.norm(scene["jaw_axis"][row])), 1e-9)
        along = abs(float(offset @ axis))
        across = abs(float(offset[0] * axis[1] - offset[1] * axis[0]))
        return along < tolerance[0] and across < tolerance[1]

    def _handle_tolerance(self, row: int, articulation: int, grip: float) -> np.ndarray:
        """Along the jaw axis: half the gap the grip opens to, minus the handle's radius and the
        margin; across: the pad's half width (a vertical bar or a knob) or the lid bar's half
        length (it runs across the jaws). Floored at the old 2 mm gate, capped."""
        sim = self.sim
        knob = (sim.lid_knob[row] if articulation == 2 else sim.drawer_knob[row]) > 0.5
        radius = fu.KNOB_RADIUS if knob else fu.BAR_RADIUS
        gap = FULL_GAP * (grip + 1.0) / 2.0
        along = gap / 2 - radius - MARGIN
        across_bar = articulation == 2 and not knob
        across = fu.LID_BAR_LENGTH / 2 - MARGIN if across_bar else PAD_HALF - MARGIN
        return np.clip(np.array([along, across]), 0.002, ARTICULATION_CAP)

    def _pads_on_object(self, row: int, ee: np.ndarray, grasp: np.ndarray, bottom: float) -> bool:
        """The pads' height closes on the object: at most 8 mm above the grasp point (as before),
        and below it down to where the pads' lower edge would reach the object's bottom (with
        a millimetre to spare), at most PAD_DEPTH below. The teacher's own descent stops within
        8 mm above; a learner that went a little lower used to be sent back up, labels the
        teacher's data never has (docs/MANIPULATION_ENV.md, "Place and stack")."""
        lowest = bottom + arm_task.PAD_BELOW_SITE + PAD_HALF + 0.001
        low = min(grasp[2] - 0.008, max(grasp[2] - PAD_DEPTH, lowest))
        return bool(low < float(ee[2]) < grasp[2] + 0.008)

    def _object_tolerance(self, row: int, slot: int) -> np.ndarray:
        """Along the jaw axis: half the open gap minus half the object's narrow side and the
        margin; across: the pad's half width. Floored at the old 3 mm gate, capped."""
        size = self.sim.object_size()[row, slot]
        narrow = float(min(size[0], size[1]))
        along = FULL_GAP / 2 - narrow / 2 - MARGIN
        return np.clip(np.array([along, PAD_HALF - MARGIN]), 0.003, OBJECT_CAP)

    def _clear(self, row: int, scene: dict[str, np.ndarray]) -> np.ndarray:
        """Open the hand and rise before starting the next subgoal."""
        ee = scene["ee"][row]
        self.counter[row] += 1
        if (scene["opening"][row] > 0.8 and self.counter[row] > 4) or self.counter[row] > 15:
            safe = self._safe_bottom(row, scene, -1) + 0.05
            if ee[2] >= safe - 0.01 or self.counter[row] > 40:
                self.phase[row] = APPROACH
                self.counter[row] = 0
            return self._command(row, ee, np.array([ee[0], ee[1], safe]), 1.0, None)
        return self._command(row, ee, ee, 1.0, None)

    # ------------------------------------------------------------------------------ articulations
    def _handle_at(self, row: int, articulation: int, value: float) -> np.ndarray:
        """EE site target to hold the handle of ``articulation`` at joint value ``value``."""
        sim = self.sim
        if articulation < 2:
            local = (
                sim.drawer_body[row, articulation]
                + np.array([value, 0.0, 0.0])
                + sim.drawer_handle_local[row]
            )
            world = sim.dc_pos[row] + sim.dc_rot[row] @ local
        else:
            lid = _lid_rotation(np.array([value]))[0]
            world = sim.cab_pos[row] + sim.cab_rot[row] @ (
                sim.lid_hinge[row] + lid @ sim.lid_handle_local[row]
            )
        return world + np.array([0.0, 0.0, arm_task.PAD_BELOW_SITE])

    def _articulate(
        self, row: int, index: int, kind: int, scene: dict[str, np.ndarray]
    ) -> np.ndarray:
        sim = self.sim
        ee = scene["ee"][row]
        lid = kind in (tk.OPEN_DOOR, tk.CLOSE_DOOR)
        articulation = 2 if lid else int(sim.sub_target[row, index])
        opening = kind in (tk.OPEN_DRAWER, tk.OPEN_DOOR)
        site = scene["handles"][row, articulation]
        frame = sim.cab_rot[row] if lid else sim.dc_rot[row]
        # Jaws close along the drawer's front and along the lid's hinge (a knob turns freely
        # between them there), or front to back across the lid's swivelling bar.
        across = lid and sim.lid_knob[row] < 0.5
        yaw = float(scene["articulation_headings"][row, articulation])
        joint = float(scene["joints"][row, articulation])
        phase = int(self.phase[row])
        xy_error = float(np.linalg.norm(ee[:2] - site[:2]))
        hover = site + np.array([0.0, 0.0, HOVER])
        # Across the lid's bar the back finger goes down between the bar and the lid's edge, so
        # the hand opens only part way there.
        grip = LID_BAR_GRIP if across else 1.0
        if phase == APPROACH and self._already_down(row, scene, site, hover, yaw, grip):
            phase = self.phase[row] = DESCEND
        if phase == APPROACH:
            action, aligned = self._approach(row, scene, hover, yaw, grip)
            if aligned:
                self.phase[row] = DESCEND
            return action
        if phase == DESCEND:
            if xy_error > ABORT_XY:
                self.phase[row] = APPROACH
            elif abs(ee[2] - site[2]) < AT_HEIGHT and self._straddles(
                row, scene, site, self._handle_tolerance(row, articulation, grip)
            ):
                self.phase[row] = CLOSE
                self.counter[row] = 0
            return self._command(row, ee, site, grip, yaw, descent=DESCENT, floor=COMMAND_FLOOR)
        if phase == CLOSE:
            self.counter[row] += 1
            left = bool(scene["handle_fingers"][0][row, articulation])
            right = bool(scene["handle_fingers"][1][row, articulation])
            # A pinched rigid handle can flicker between the two pads' contacts from one step
            # to the next; the pinch holds once each pad has touched and one touches now.
            self.seen[row] |= np.array([left, right])
            self._centre(row, scene, left, right)
            pinched = bool(self.seen[row].all()) and (left or right)
            if self.counter[row] >= SETTLE_STEPS and pinched:
                self.phase[row] = MOVE
                self.grip_offset[row] = ee - site
            elif self.counter[row] >= CLOSE_TIMEOUT:
                self.phase[row] = RELEASE
                self.counter[row] = 0
                self.retry[row] = True
            desired = site.copy()
            desired[:2] += self.centring[row]
            return self._command(row, ee, desired, -1.0, yaw)
        if phase == MOVE:
            if np.linalg.norm(ee - site - self.grip_offset[row]) > SLIP:
                self.phase[row] = RELEASE  # lost the handle: let go and try again
                self.counter[row] = 0
                self.retry[row] = True
                return self._command(row, ee, ee, 1.0, None)
            if lid:
                goal = LID_OPEN_GOAL if opening else -OVERSHOOT_ANGLE
                step = LID_OPEN_STEP if opening else LID_CLOSE_STEP
                finished = joint >= LID_DONE_OPEN if opening else joint <= LID_DONE_CLOSED
            else:
                travel = float(sim.travel[row, articulation])
                goal = travel + OVERSHOOT if opening else -OVERSHOOT
                step = DRAWER_STEP
                finished = (
                    joint >= DRAWER_DONE_OPEN * travel if opening else joint <= DRAWER_DONE_CLOSED
                )
            if finished:
                self.phase[row] = RELEASE
                self.counter[row] = 0
                return self._command(row, ee, ee, 1.0, None)
            # Lead the handle along its path by one joint step; the hand keeps the offset it
            # had from the handle when the pinch closed.
            target = joint + float(np.clip(goal - joint, -step, step))
            desired = self._handle_at(row, articulation, target) + self.grip_offset[row]
            return self._command(row, ee, desired, -1.0, yaw)
        if phase == RELEASE:
            return self._release(row, ee, scene)
        if phase == RETREAT:
            if lid and opening and not self.retry[row]:
                # The opened lid's grip stands at the edge of the arm's reach, too high to rise
                # above. Slide off it sideways, along the bar (a knob is held from the sides, so
                # nothing is in the way), then step back toward the robot over the cabinet.
                z = self.release_z[row]
                side = self.anchor[row] + (LID_SLIDE * frame[:2, 1] if across else 0.0)
                back = side + LID_STEP_BACK * frame[:2, 0]
                if self.counter[row] == 0 and np.linalg.norm(ee[:2] - side) > 0.01:
                    return self._command(row, ee, np.array([*side, z]), 1.0, None)
                self.counter[row] = 1
                if np.linalg.norm(ee[:2] - back) < 0.015:
                    self.phase[row] = FINISHED
                    self.counter[row] = 0
                return self._command(row, ee, np.array([*back, z]), 1.0, None)
            up = np.array([self.anchor[row, 0], self.anchor[row, 1], self.release_z[row] + HOVER])
            return self._retreat(row, ee, up)
        return self._command(row, ee, ee, 1.0, None)

    # ------------------------------------------------------------------------------ objects
    def _pick_yaw(self, row: int, slot: int, scene: dict[str, np.ndarray]) -> float:
        """Jaw heading to pick ``slot``: across its narrow side (the grasp task's rule).

        In a drawer the jaws close along the drawer's width instead whenever that is still
        across the object (round objects, or ones turned less than DRAWER_JAW_SLACK), so the
        open fingers stay parallel to the front panel: skewed, a finger lands on the panel and
        pushes the drawer shut.
        """
        return float(scene["pick_headings"][row, slot])  # rule: ManipulationSim.pick_headings

    def _pick_place(
        self, row: int, index: int, kind: int, scene: dict[str, np.ndarray]
    ) -> np.ndarray:
        sim = self.sim
        ee = scene["ee"][row]
        slot = int(sim.sub_obj[row, index])
        grasp = scene["grasp_points"][row, slot]
        object_yaw = self._pick_yaw(row, slot, scene)
        grasped = bool(scene["grasped"][row, slot])
        bottom = float(scene["bottom"][row, slot])
        top = float(scene["top"][row, slot])
        centre = scene["object_pos"][row, slot]
        phase = int(self.phase[row])
        xy_error = float(np.linalg.norm(ee[:2] - grasp[:2]))
        safe = self._safe_bottom(row, scene, slot)
        carry_z = ee[2] + (safe - bottom)
        place_yaw = float(scene["place_headings"][row, index])
        target = scene["targets"][row, index]
        if phase == APPROACH:
            hover = np.array([grasp[0], grasp[1], max(grasp[2] + 0.06, top + 0.075)])
            action, aligned = self._approach(row, scene, hover, object_yaw)
            if aligned:
                self.phase[row] = DESCEND
            return action
        if phase == DESCEND:
            if xy_error > 0.02 or self._yaw_error(row, object_yaw) > 0.15:
                self.phase[row] = APPROACH
            elif self._pads_on_object(row, ee, grasp, bottom) and self._straddles(
                row, scene, grasp, self._object_tolerance(row, slot)
            ):
                # Centred within OBJECT_CAP: the coupled fingers then meet the object together
                # instead of one pushing it over before the other arrives.
                self.phase[row] = CLOSE
                self.counter[row] = 0
                self.close_steps[row] = 0
                self.anchor[row] = grasp[:2]
            return self._command(
                row, ee, grasp, 1.0, object_yaw, descent=DESCENT, floor=COMMAND_FLOOR
            )
        if phase == CLOSE:
            self.close_steps[row] += 1
            self.counter[row] = self.counter[row] + 1 if grasped else 0
            if self.counter[row] >= SETTLE_STEPS:
                self.phase[row] = LIFT
                self.counter[row] = 0
            elif self.close_steps[row] >= CLOSE_TIMEOUT:
                self.phase[row] = CLEAR
                self.counter[row] = 0
            self._centre(
                row,
                scene,
                bool(scene["fingers"][0][row, slot]),
                bool(scene["fingers"][1][row, slot]),
            )
            desired = np.array([self.anchor[row, 0], self.anchor[row, 1], grasp[2]])
            desired[:2] += self.centring[row]
            return self._command(row, ee, desired, -1.0, None)
        if phase in (LIFT, CARRY, LOWER) and not grasped:
            self.counter[row] += 1
            if self.counter[row] >= LOST_STEPS:
                self.phase[row] = CLEAR
                self.counter[row] = 0
        elif phase in (LIFT, CARRY, LOWER):
            self.counter[row] = 0
        if phase == LIFT:
            desired = np.array([self.anchor[row, 0], self.anchor[row, 1], carry_z])
            if bottom >= safe - 0.01:
                self.phase[row] = FINISHED if kind == tk.PICK else CARRY
            return self._command(row, ee, desired, -1.0, None)
        if phase == CARRY:
            offset = target[:2] - centre[:2]
            # Near the target, finish turning before closing in: a hand still turning sweeps
            # its 20 cm width through whatever stands beside the target (an opened lid).
            turning = self._yaw_error(row, place_yaw) > TURN_BEFORE_ARRIVING
            if turning and np.linalg.norm(offset) < ARRIVING:
                offset = np.zeros(2)
            desired = np.array([ee[0] + offset[0], ee[1] + offset[1], carry_z])
            if np.linalg.norm(offset) < 0.008 and self._yaw_error(row, place_yaw) < 0.05:
                self.phase[row] = LOWER
                self.counter[row] = 0
                self.previous_z[row] = ee[2] + 1.0
            return self._command(row, ee, desired, -1.0, place_yaw, floor=COMMAND_FLOOR)
        if phase == LOWER:
            offset = target[:2] - centre[:2]
            drop = target[2] + 0.006 - bottom
            # The fingers can meet the support before the object does (an object that shifted
            # up in the jaws): stop when the hand stops going down, and stop steering sideways.
            stalled = self.previous_z[row] - ee[2] < 0.001
            self.counter[row] = self.counter[row] + 1 if stalled else 0
            self.previous_z[row] = ee[2]
            if drop > -0.004 or self.counter[row] >= 3:
                self.phase[row] = RELEASE
                self.counter[row] = 0
                return self._command(row, ee, ee, -1.0, None)
            desired = np.array([ee[0] + offset[0], ee[1] + offset[1], ee[2] + drop])
            return self._command(
                row, ee, desired, -1.0, place_yaw, descent=0.5, floor=COMMAND_FLOOR
            )
        if phase == RELEASE:
            return self._release(row, ee, scene)
        if phase == RETREAT:
            height = max(top + 0.09, grasp[2] + 0.08, self._transit_z(row, scene))
            up = np.array([self.anchor[row, 0], self.anchor[row, 1], height])
            return self._retreat(row, ee, up)
        return self._command(row, ee, ee, 1.0, None)
