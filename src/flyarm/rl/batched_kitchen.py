"""N copies of the D4RL FrankaKitchen benchmark stepped in parallel with mjbatch.

Everything that defines the task is read out of the benchmark's own environment, which is
constructed once per process and then discarded: the compiled MuJoCo model, the initial
state, the joint position and velocity bounds, the observation-noise amplitudes and the
control timing. Only the bookkeeping is vectorized, so a batched episode is the same episode
gymnasium_robotics.envs.franka_kitchen would run.

Control semantics (FrankaRobot.step): 9 actions clipped to [-1, 1] and scaled by
``act_rng = 2`` into joint velocity commands, clipped to the configured velocity bounds,
integrated for one control step of ``frame skip 40 x 2 ms = 80 ms`` from the *last observed*
joint positions, clipped to the configured position bounds, and written to the 9 position
actuators. The dependence on the last observation rather than on the true state is the
benchmark's own (it breaks the Markov property; upstream flags it and keeps it), so it is
kept here: with a nonzero ``robot_noise_ratio`` the observation noise enters the control law.

Observation: the 30 position features of flyarm.benchmarks.kitchen.POLICY_FEATURES, i.e. the
9 robot joint positions and the 21 object joint positions of the 59-D benchmark observation.
Joint velocities are withheld from controllers because behavior cloning with them copies its
own last action (kitchen.POLICY_FEATURES); the critic's privileged observation has them.

Task completion is the benchmark's: the element goals of
gymnasium_robotics.envs.franka_kitchen.kitchen_env and its 0.3 threshold, for the four tasks
of the "complete" split in their order (microwave, kettle, light switch, slide cabinet). A
task counts for the episode once its element has ever been within the threshold, which is
what KitchenEnv records in ``episode_task_completions``; the dataset's recovered environment
sets ``remove_task_when_completed`` False, so a completed element is re-checked every step
and a task that is undone later still counts, exactly as here.

Reward. The benchmark's own reward is the number of elements at goal in the current step,
which the D4RL score ignores, so training uses the shaped reward below instead; evaluation
always reports the benchmark's completion count. The shaped reward pays, for the first task
of the four that is not yet completed:

* ``APPROACH_WEIGHT`` x how close the gripper is to that element's handle, and
* ``PROGRESS_WEIGHT`` x how far that element has travelled from its initial joint position
  toward its goal (0 at the start of the episode, 1 at the completion threshold),

so at most ``MAX_STEP_REWARD = 1.0`` per step, and ``COMPLETION_BONUS`` once per newly
completed task. Research log E34: at gamma 0.99 a per-step term of r sustained forever is
worth 100 r, so a bonus below ``MAX_STEP_REWARD / (1 - gamma) = 100`` would make hovering at
the completion threshold of one element worth more than completing it and moving on; the
default 200 is twice that floor. ``minimum_completion_bonus`` states the rule for other
discounts, and the constructor rejects a bonus below it.

Reference tracking (optional, ``tracking_weight``, default 0 so every earlier run is
unchanged). In the spirit of DeepMimic and AMP, the reward can carry a term built from a
reference joint trajectory, ``q_ref = reference[min(step, len - 1)]``, taken from one benchmark
demonstration chosen by index. Two forms, selected by ``tracking_form``:

``"potential"`` (the default and the one to use): ``tracking_weight * (phi(s') - phi(s))`` with
``phi(s) = -||q - q_ref||``, that is ``tracking_weight`` times the distance closed this step, in
radians. It pays for *closing* the distance rather than for *being* close, so it is scale-free
and alive at any distance: shaving 0.1 rad pays the same at 5 rad as at 0.5 rad. It telescopes
exactly and without discounting to ``phi(s_T) - phi(s_0)``, so its total over an episode is
bounded by ``tracking_weight * ||q_0 - q_ref,0||`` (about 0.45 rad from a 0.3 rad start),
standing still pays exactly 0, and the E34 floor below is untouched. Strict policy invariance
(Ng, Harada and Russell 1999) holds for the undiscounted objective; at gamma 0.99 the shaped
optimum can differ, but by at most ``tracking_weight * ||q_0 - q_ref,0||`` in return, which is
about 0.2% of a single completion bonus.

``"potential_discounted"`` (kept for the record, do not reach for it): the literal Ng, Harada
and Russell form ``tracking_weight * (gamma * phi(s') - phi(s))``. Its discounted sum telescopes
to ``gamma^T phi(s_T) - phi(s_0)``, but the undiscounted reward the agent collects carries a
``(1 - gamma) * ||q - q_ref||`` residual per step that **pays more for being far from the
reference than for tracking it**: measured at weight 1.0, gamma 0.99 and 0.3 rad starts, the
demonstration tracker earns +0.0058 per step and a random policy +0.0250, the wrong way round.
Standing still far from the reference pays ``(1 - gamma) * d`` forever instead of 0.

Either way, the time index has to count as part of the state for the invariance argument, which
it does: the reference is a function of the step and the critic reads the episode fraction.

``"gaussian"`` (kept for the record, do not reach for it):
``tracking_weight * exp(-||q - q_ref||^2 / tracking_sigma^2)``. **This form produced a negative
result (research log E41, runs/ppo-kitchen-scratch-004).** A learning policy sits about 5.5 rad
from the reference in joint space, where ``exp(-d^2 / 0.6^2)`` is about ``e^-85`` and its
derivative is exactly 0, so the term is numerically dead where it would have to do its work; it
earned 0.017 per step against 0.025 for a random policy. Its only live effect was to raise the
per-step maximum and with it the E34 floor, cutting the completion bonus's headroom from 2.0x
to 1.33x. Widening sigma enough to reach 5.5 rad makes it nearly constant instead. Any sigma
for it must be chosen from the distance the *learning* policy occupies, never from how well it
separates an expert from a random policy.

**Only the reference's states are used, never its actions**, so nothing is cloned and a run
that uses it is still reward-only; ``reference_joints`` reads the demonstration's joint
positions and never touches ``KitchenData.actions``.

Indexing the reference by time is defensible on this benchmark and only on this benchmark:
every kitchen episode starts from the same physical state (research log E29), so step t of any
episode is comparable to step t of any demonstration. It would not be defensible for
pick-and-place, where the cube and the goal move every episode and step t means nothing.

The term is the whole per-step budget's third member, so it raises the per-step maximum to
``MAX_STEP_REWARD + tracking_weight`` and the E34 floor with it: at gamma 0.99 and the default
bonus of 200 the weight must stay below 1.0, and the constructor's check enforces exactly that.

Because a time-indexed reference could in principle be maximised by replaying it open loop,
which E29 shows the clean kitchen rewards, a run that uses this term should train from
perturbed starts and select on perturbed validation episodes (E30: blind replay falls to 16
to 19 from 0.2 to 0.3 rad starts while closed-loop control still scores 100).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import (
    BONUS_THRESH,
    OBS_ELEMENT_GOALS,
    OBS_ELEMENT_INDICES,
)
from mjbatch import Batch

from flyarm.benchmarks import _robotics_compat, kitchen

ACTION_DIM = kitchen.ACTION_DIM  # 9 joint velocity commands
OBS_DIM = kitchen.FEATURE_DIM  # 30 position features
ROBOT_JOINTS = 9
ARM_JOINTS = 7  # the 7 revolute arm joints; 7 and 8 are the gripper fingers
HORIZON = 280  # FrankaKitchen-v1 max_episode_steps
ACT_RANGE = 2.0  # FrankaRobot.act_rng: action 1.0 commands 2 rad/s
COMPLETE_TASKS = ("microwave", "kettle", "light switch", "slide cabinet")
# Handle sites of the kitchen model, one per element, used for the approach term only.
ELEMENT_SITES = {
    "bottom burner": "knob2_site",
    "top burner": "knob4_site",
    "light switch": "light_site",
    "slide cabinet": "slide_site",
    "hinge cabinet": "hinge_site2",
    "microwave": "microhandle_site",
    "kettle": "kettle_site",
}
GRIPPER_SITE = "end_effector"

APPROACH_WEIGHT = 0.3
PROGRESS_WEIGHT = 0.7
MAX_STEP_REWARD = APPROACH_WEIGHT + PROGRESS_WEIGHT  # 1.0
APPROACH_SLOPE = 3.0  # 1 - tanh(slope d); research log E38 on starving a far-away policy
COMPLETION_BONUS = 200.0
STALL_GAMMA = 0.99  # the discount the default bonus is sized against
TRACKING_SIGMA = 0.6  # rad, "gaussian" form only; see the module docstring and research log E41
TRACKING_FORMS = ("potential", "potential_discounted", "gaussian")
# Which uncompleted task the approach and progress terms are paid for.
# "split_order": the first one in the split's order, the rule of every run before research log
#   E43. It is wrong whenever the controller cannot reach that element: measured on
#   runs/ppo-kitchen-scratch-003, the target is the microwave for 280 of 280 steps while the
#   microwave's progress stays at exactly 0 and the kettle is what gets completed.
# "nearest": the one whose handle is nearest the gripper.
# "progress": the one with the largest normalised progress, ties by the split's order. Measured
#   worse than "split_order" for the expert, because normalised progress saturates fastest for
#   the element with the smallest span to its threshold (slide cabinet 0.07 rad against the
#   microwave's 0.45), so the rule follows an artefact of the normalisation (E43).
# "progress_then_nearest": "progress" once any element has moved, "nearest" before that.
# "moved": the one whose element has travelled furthest in absolute terms, ties by the split's
#   order, which removes the span artefact; before anything moves this is the split's order.
# "moved_then_nearest": "moved" once any element has moved, "nearest" before that.
TARGET_RULES = (
    "split_order",
    "nearest",
    "progress",
    "progress_then_nearest",
    "moved",
    "moved_then_nearest",
)
MOVED_EPSILON = 0.02  # rad of element travel that counts as "this element has started moving"
PROGRESS_EPSILON = 0.05  # normalised progress that counts as "this element has started moving"
REFERENCE_SPLIT = "complete"


def minimum_completion_bonus(
    gamma: float = STALL_GAMMA, max_step_reward: float = MAX_STEP_REWARD
) -> float:
    """Value of stalling forever on the per-step maximum; a bonus must exceed it (E34)."""
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    if max_step_reward <= 0:
        raise ValueError("max_step_reward must be positive")
    return max_step_reward / (1.0 - gamma)


@functools.lru_cache(maxsize=4)
def reference_joints(episode: int, split: str = REFERENCE_SPLIT) -> np.ndarray:
    """The 9 robot joint positions of one benchmark demonstration, [steps, 9].

    Only the demonstration's states are read; its actions are never touched, so a run that
    tracks this reference clones nothing and stays reward-only.
    """
    data = kitchen.load(split, download=False)
    if not 0 <= episode < len(data.obs):
        raise ValueError(f"Reference episode {episode} is outside the {len(data.obs)} of {split}")
    steps = int(data.mask[episode].sum())
    return data.obs[episode, :steps, :ROBOT_JOINTS].astype(np.float64)


@dataclass(frozen=True)
class KitchenVariant:
    """Per-episode task conditions; the defaults are the benchmark's own noiseless episode."""

    # Uniform offset of the 7 arm joints at reset, the perturbed-start protocol of E30.
    initial_joint_offset: float = 0.0
    # Log-uniform factor on the kettle's mass and inertia.
    kettle_mass_scale: tuple[float, float] = (1.0, 1.0)
    # The benchmark's observation-noise ratios (0.01 and 0.0005 in the recorded environment);
    # robot noise also enters the control law through the last observed joint positions.
    robot_noise_ratio: float = 0.0
    object_noise_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.initial_joint_offset <= 0.5:
            raise ValueError("initial_joint_offset must be in [0, 0.5] rad")
        low, high = self.kettle_mass_scale
        if not 0 < low <= high:
            raise ValueError("kettle_mass_scale must satisfy 0 < low <= high")
        for ratio in (self.robot_noise_ratio, self.object_noise_ratio):
            if not 0 <= ratio <= 1:
                raise ValueError("noise ratios must be in [0, 1]")

    @property
    def randomizes_physics(self) -> bool:
        return self.kettle_mass_scale != (1.0, 1.0)

    @property
    def adds_noise(self) -> bool:
        return self.robot_noise_ratio > 0 or self.object_noise_ratio > 0


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray  # [N, 30], after automatic resets for finished episodes
    reward: np.ndarray  # [N]
    terminated: np.ndarray  # [N] bool, all four tasks completed
    truncated: np.ndarray  # [N] bool, horizon reached
    completed: np.ndarray  # [N, 4] bool, tasks ever completed this episode (before reset)
    tasks_completed: np.ndarray  # [N] int in 0..4 (before reset)
    newly_completed: np.ndarray  # [N] int completed at this step
    target: np.ndarray  # [N] index of the task being shaped for, 4 when all are done
    goal_distance: np.ndarray  # [N, 4] element distance to its goal
    tracking: np.ndarray  # [N] the reference term actually paid this step, before its weight


@dataclass(frozen=True)
class _Specs:
    """Everything the batched environment copies out of the benchmark's own environment."""

    xml_path: str
    frame_skip: int
    control_dt: float
    init_qpos: np.ndarray
    pos_bound: np.ndarray
    vel_bound: np.ndarray
    pos_noise_amp: np.ndarray
    vel_noise_amp: np.ndarray


@functools.lru_cache(maxsize=1)
def benchmark_specs() -> _Specs:
    """Construct the benchmark's KitchenEnv once and read its model path, state and limits."""
    _robotics_compat.install()
    from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv

    env = KitchenEnv(
        tasks_to_complete=list(COMPLETE_TASKS),
        remove_task_when_completed=False,
        terminate_on_tasks_completed=False,
    )
    robot = env.robot_env
    specs = _Specs(
        xml_path=str(robot.fullpath),
        frame_skip=int(robot.frame_skip),
        control_dt=float(robot.dt),
        init_qpos=np.asarray(robot.init_qpos, dtype=np.float64).copy(),
        pos_bound=np.asarray(robot.robot_pos_bound, dtype=np.float64)[:ROBOT_JOINTS].copy(),
        vel_bound=np.asarray(robot.robot_vel_bound, dtype=np.float64)[:ROBOT_JOINTS].copy(),
        pos_noise_amp=np.asarray(robot.robot_pos_noise_amp, dtype=np.float64).copy(),
        vel_noise_amp=np.asarray(robot.robot_vel_noise_amp, dtype=np.float64).copy(),
    )
    env.close()
    return specs


class BatchedKitchen:
    """``num_envs`` FrankaKitchen episodes stepped together; the API of BatchedPickPlace."""

    def __init__(
        self,
        num_envs: int,
        *,
        horizon: int = HORIZON,
        first_seed: int = 0,
        num_threads: int = 0,
        variant: KitchenVariant | None = None,
        completion_bonus: float = COMPLETION_BONUS,
        approach_slope: float = APPROACH_SLOPE,
        gamma: float = STALL_GAMMA,
        tracking_weight: float = 0.0,
        tracking_form: str = "potential",
        target_rule: str = "split_order",
        tracking_sigma: float = TRACKING_SIGMA,
        reference_episode: int = 0,
        terminate_on_all_tasks: bool = True,
        tasks: tuple[str, ...] = COMPLETE_TASKS,
    ) -> None:
        if num_envs < 1 or horizon < 20:
            raise ValueError("num_envs must be positive and horizon at least 20")
        if tracking_weight < 0 or tracking_sigma <= 0:
            raise ValueError("tracking_weight must be non-negative and tracking_sigma positive")
        if tracking_form not in TRACKING_FORMS:
            raise ValueError(f"tracking_form must be one of {TRACKING_FORMS}")
        if target_rule not in TARGET_RULES:
            raise ValueError(f"target_rule must be one of {TARGET_RULES}")
        self.target_rule = target_rule
        self.tracking_weight = float(tracking_weight)
        self.tracking_form = tracking_form
        self.tracking_sigma = float(tracking_sigma)
        self.gamma = float(gamma)
        # Potential-based shaping telescopes, so it adds nothing to a sustained trajectory and
        # leaves the E34 floor alone; the Gaussian form joins the per-step budget and raises it.
        self.max_step_reward = MAX_STEP_REWARD + (
            self.tracking_weight if tracking_form == "gaussian" else 0.0
        )
        floor = minimum_completion_bonus(gamma, self.max_step_reward)
        if completion_bonus <= floor:
            raise ValueError(
                f"completion_bonus {completion_bonus} must exceed the value of stalling on the "
                f"per-step maximum of {self.max_step_reward} (approach, progress and reference "
                f"tracking), {floor:.1f} at gamma {gamma} (research log E34)"
            )
        if approach_slope <= 0:
            raise ValueError("approach_slope must be positive")
        self.reference = (
            reference_joints(int(reference_episode)) if self.tracking_weight > 0 else None
        )
        unknown = [task for task in tasks if task not in OBS_ELEMENT_GOALS]
        if unknown or not tasks:
            raise ValueError(f"Unknown kitchen tasks {unknown}")
        specs = benchmark_specs()
        self.specs = specs
        self.tasks = tuple(tasks)
        self.completion_bonus = float(completion_bonus)
        self.approach_slope = float(approach_slope)
        self.terminate_on_all_tasks = bool(terminate_on_all_tasks)
        self.num_envs, self.horizon = num_envs, horizon
        self.variant = variant or KitchenVariant()
        self.model = model = mujoco.MjModel.from_xml_path(specs.xml_path)
        if model.nu != ACTION_DIM:
            raise ValueError(f"Kitchen model has {model.nu} actuators, expected {ACTION_DIM}")
        self.batch = Batch(model, num_envs, num_threads)
        self._element_indices = [OBS_ELEMENT_INDICES[task] for task in self.tasks]
        self._element_goals = [OBS_ELEMENT_GOALS[task] for task in self.tasks]
        self._element_sites = np.array([model.site(ELEMENT_SITES[task]).id for task in self.tasks])
        self._gripper_site = model.site(GRIPPER_SITE).id
        self._kettle_body = model.body("kettle").id

        b = self.batch
        self.qpos, self.qvel, self.ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
        self.site_xpos = b.bind("site_xpos")
        if self.variant.randomizes_physics:
            self.body_mass = b.expand("body_mass")
            self.body_inertia = b.expand("body_inertia")
        self.mass_scale = np.ones(num_envs)

        n, t = num_envs, len(self.tasks)
        self.steps = np.zeros(n, dtype=np.int64)
        self.completed = np.zeros((n, t), dtype=bool)
        self.initial_goal_distance = np.ones((n, t))
        self.last_robot_qpos = np.zeros((n, ROBOT_JOINTS))
        self.previous_potential = np.zeros(n)
        self.episode_seed = np.zeros(n, dtype=np.int64)
        self._noise = [np.random.default_rng(0) for _ in range(n)]
        self._next_seed = first_seed

    # State readers ------------------------------------------------------------------------
    def object_qpos(self) -> np.ndarray:
        return self.qpos[:, ROBOT_JOINTS:].copy()

    def goal_distance(self) -> np.ndarray:
        """[N, tasks] distance of each element's joints to its goal; the benchmark's measure."""
        qpos = self.qpos
        return np.stack(
            [
                np.linalg.norm(qpos[:, indices] - goal, axis=1)
                for indices, goal in zip(self._element_indices, self._element_goals, strict=True)
            ],
            axis=1,
        )

    def element_progress(self) -> np.ndarray:
        """[N, tasks] normalised progress: 0 at the episode start, 1 at the completion threshold."""
        span = np.maximum(self.initial_goal_distance - BONUS_THRESH, 1e-6)
        return np.clip((self.initial_goal_distance - self.goal_distance()) / span, 0.0, 1.0)

    def handle_distances(self) -> np.ndarray:
        """[N, tasks] gripper-to-handle distance for every element of the split."""
        gripper = self.site_xpos[:, self._gripper_site][:, None, :]
        return np.linalg.norm(self.site_xpos[:, self._element_sites] - gripper, axis=2)

    def target(self) -> np.ndarray:
        """[N] index of the uncompleted task the shaping is paid for (len when all are done)."""
        remaining = ~self.completed
        if self.target_rule == "split_order":
            index = remaining.argmax(1)
        else:
            index = np.where(remaining, self._target_score(), -np.inf).argmax(1)
        return np.where(remaining.any(1), index, len(self.tasks))

    def element_travel(self) -> np.ndarray:
        """[N, tasks] how far each element has moved toward its goal, in radians, never negative."""
        return np.maximum(self.initial_goal_distance - self.goal_distance(), 0.0)

    def _target_score(self) -> np.ndarray:
        """[N, tasks] higher is a better shaping target under the configured rule."""
        if self.target_rule == "nearest":
            return -self.handle_distances()
        if self.target_rule in ("moved", "moved_then_nearest"):
            travel = self.element_travel()
            if self.target_rule == "moved":
                return travel
            started = travel.max(1, keepdims=True) >= MOVED_EPSILON
            return np.where(started, travel, -self.handle_distances())
        progress = self.element_progress()
        if self.target_rule == "progress":
            return progress
        # "progress_then_nearest": commit to whatever has started moving, otherwise go nearest.
        started = progress.max(1, keepdims=True) >= PROGRESS_EPSILON
        return np.where(started, progress, -self.handle_distances())

    def potential(self) -> np.ndarray:
        """[N] phi(s) = -||q - q_ref|| at the current step index; the shaping potential."""
        if self.reference is None:
            return np.zeros(self.num_envs)
        index = np.minimum(self.steps, len(self.reference) - 1)
        return -np.linalg.norm(self.qpos[:, :ROBOT_JOINTS] - self.reference[index], axis=1)

    def tracking_reward(self) -> np.ndarray:
        """[N] the reference term for the current state.

        The potential forms read ``previous_potential``, which ``step`` advances once per step
        right after paying the term, so after a step this returns 0 rather than what was paid;
        ``StepResult.tracking`` carries the value that was actually paid.
        """
        if self.reference is None:
            return np.zeros(self.num_envs)
        if self.tracking_form == "gaussian":
            index = np.minimum(self.steps, len(self.reference) - 1)
            error = self.qpos[:, :ROBOT_JOINTS] - self.reference[index]
            return np.exp(-(error**2).sum(1) / self.tracking_sigma**2)
        discount = self.gamma if self.tracking_form == "potential_discounted" else 1.0
        return discount * self.potential() - self.previous_potential

    def approach_distance(self, target: np.ndarray) -> np.ndarray:
        """[N] gripper-to-handle distance for each environment's current target element."""
        active = target < len(self.tasks)
        sites = self._element_sites[np.where(active, target, 0)]
        rows = np.arange(self.num_envs)
        handle = self.site_xpos[rows, sites]
        gripper = self.site_xpos[:, self._gripper_site]
        return np.linalg.norm(handle - gripper, axis=1)

    def observation(self, *, privileged: bool = False) -> np.ndarray:
        """The controller's 30 features, or the critic's (plus velocities and task state)."""
        robot_qpos, robot_qvel = self.qpos[:, :ROBOT_JOINTS], self.qvel[:, :ROBOT_JOINTS]
        object_qpos, object_qvel = self.qpos[:, ROBOT_JOINTS:], self.qvel[:, ROBOT_JOINTS:]
        variant = self.variant
        if variant.adds_noise:
            robot_qpos = robot_qpos + self._draw_noise(
                variant.robot_noise_ratio, self.specs.pos_noise_amp[:ROBOT_JOINTS]
            )
            object_qpos = object_qpos + self._draw_noise(
                variant.object_noise_ratio, self.specs.pos_noise_amp[ROBOT_JOINTS - 1 :]
            )
        features = np.concatenate((robot_qpos, object_qpos), axis=1)
        if not privileged:
            return features.astype(np.float32)
        return np.concatenate(
            (
                features,
                robot_qvel,
                object_qvel,
                self.goal_distance(),
                self.completed.astype(np.float64),
            ),
            axis=1,
        ).astype(np.float32)

    @property
    def privileged_dim(self) -> int:
        return OBS_DIM + self.model.nv + 2 * len(self.tasks)

    def _draw_noise(self, ratio: float, amplitude: np.ndarray) -> np.ndarray:
        draws = np.stack([rng.uniform(-1.0, 1.0, len(amplitude)) for rng in self._noise])
        return ratio * amplitude * draws

    # Reset ---------------------------------------------------------------------------------
    def reset(self, ids: np.ndarray | None = None, seeds: np.ndarray | None = None) -> np.ndarray:
        """Reset the given envs (all by default) with explicit or consecutive seeds."""
        ids = np.arange(self.num_envs) if ids is None else np.asarray(ids, dtype=np.int64)
        if seeds is None:
            seeds = np.arange(self._next_seed, self._next_seed + len(ids))
            self._next_seed += len(ids)
        seeds = np.asarray(seeds, dtype=np.int64)
        if len(seeds) != len(ids):
            raise ValueError("One seed per reset environment is required")
        self.batch.reset(ids)
        offset = self.variant.initial_joint_offset
        for row, seed in zip(ids, seeds, strict=True):
            self.qpos[row] = self.specs.init_qpos
            self.qvel[row] = 0.0
            if offset:
                # The perturbed start of flyarm.benchmarks.kitchen._offset_initial_joints.
                draw = np.random.default_rng(int(seed) + 424242).uniform(
                    -offset, offset, ARM_JOINTS
                )
                self.qpos[row, :ARM_JOINTS] = self.specs.init_qpos[:ARM_JOINTS] + draw
            self._noise[row] = np.random.default_rng([int(seed), 2])
        if self.variant.randomizes_physics:
            self._randomize_physics(ids, seeds)
        self.batch.forward(ids)
        self.steps[ids] = 0
        self.completed[ids] = False
        self.episode_seed[ids] = seeds
        distance = self.goal_distance()
        self.initial_goal_distance[ids] = np.maximum(distance[ids], BONUS_THRESH + 1e-6)
        observation = self.observation()
        # As in FrankaRobot.reset_model, the control law's reference is the observed position.
        self.last_robot_qpos[ids] = observation[ids, :ROBOT_JOINTS]
        self.previous_potential[ids] = self.potential()[ids]
        return observation

    def _randomize_physics(self, ids: np.ndarray, seeds: np.ndarray) -> None:
        """Kettle mass (inertia scaled with it), from a stream of its own."""
        low, high = self.variant.kettle_mass_scale
        body = self._kettle_body
        for row, seed in zip(ids, seeds, strict=True):
            generator = np.random.default_rng([int(seed), 1])
            scale = float(np.exp(generator.uniform(np.log(low), np.log(high))))
            self.mass_scale[row] = scale
            self.body_mass[row, body] = self.model.body_mass[body] * scale
            self.body_inertia[row, body] = self.model.body_inertia[body] * scale
        self.batch.set_const(ids)

    # Step ----------------------------------------------------------------------------------
    def _control(self, action: np.ndarray) -> np.ndarray:
        """FrankaRobot.step: velocity command, velocity clip, integrate, position clip."""
        velocity = np.clip(action, -1.0, 1.0) * ACT_RANGE
        velocity = np.clip(velocity, self.specs.vel_bound[:, 0], self.specs.vel_bound[:, 1])
        target = self.last_robot_qpos + velocity * self.specs.control_dt
        return np.clip(target, self.specs.pos_bound[:, 0], self.specs.pos_bound[:, 1])

    def step(self, action: np.ndarray, *, auto_reset: bool = True) -> StepResult:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.num_envs, ACTION_DIM) or not np.all(np.isfinite(action)):
            raise ValueError(f"actions must be finite with shape ({self.num_envs}, {ACTION_DIM})")
        self.ctrl[:] = self._control(action)
        self.batch.step(nstep=self.specs.frame_skip)
        self.steps += 1
        observation = self.observation()
        self.last_robot_qpos = observation[:, :ROBOT_JOINTS].astype(np.float64)

        distance = self.goal_distance()
        target = self.target()
        was_completed = self.completed.copy()
        self.completed |= distance < BONUS_THRESH
        newly = (self.completed & ~was_completed).sum(1)
        rows, shaped = np.arange(self.num_envs), np.minimum(target, len(self.tasks) - 1)
        tracking = self.tracking_reward()
        reward = shaped_reward(
            self.approach_distance(target),
            distance[rows, shaped],
            self.initial_goal_distance[rows, shaped],
            newly,
            target < len(self.tasks),
            self.completion_bonus,
            self.approach_slope,
            tracking,
            self.tracking_weight,
        )
        self.previous_potential = self.potential()
        tasks_completed = self.completed.sum(1)
        terminated = (tasks_completed == len(self.tasks)) & self.terminate_on_all_tasks
        truncated = self.steps >= self.horizon
        result = StepResult(
            obs=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated & ~terminated,
            completed=self.completed.copy(),
            tasks_completed=tasks_completed,
            newly_completed=newly,
            target=target,
            goal_distance=distance,
            tracking=tracking,
        )
        done = terminated | truncated
        if auto_reset and done.any():
            finished = np.flatnonzero(done)
            fresh = observation.copy()
            fresh[finished] = self.reset(finished)[finished]
            return replace(result, obs=fresh)
        return result


def shaped_reward(
    approach_distance: np.ndarray,
    goal_distance: np.ndarray,
    initial_goal_distance: np.ndarray,
    newly_completed: np.ndarray,
    has_target: np.ndarray,
    completion_bonus: float = COMPLETION_BONUS,
    approach_slope: float = APPROACH_SLOPE,
    tracking: np.ndarray | None = None,
    tracking_weight: float = 0.0,
) -> np.ndarray:
    """Approach the current target element, move it toward its goal, complete tasks.

    The approach and progress terms are bounded by ``MAX_STEP_REWARD`` and are paid for the
    first uncompleted task of the split only, so completing a task never costs shaping reward:
    the shaping simply moves on to the next element. The optional reference term is added
    whether or not a target remains, because it is about where the arm is, not about which
    element is next, and in its potential-based form it may be negative when the arm moves away
    from the reference.

    ``completion_bonus`` must exceed ``minimum_completion_bonus(gamma, max_step_reward)``,
    otherwise stalling is worth more than completing a task (research log E34). The
    potential-based reference term does not enter ``max_step_reward``, because its discounted
    sum telescopes: over any trajectory, however long, it contributes at most
    ``tracking_weight * ||q_0 - q_ref,0||`` in total. The Gaussian form does enter it, because
    it can be collected every step forever.
    """
    approach = 1.0 - np.tanh(approach_slope * approach_distance)
    span = np.maximum(initial_goal_distance - BONUS_THRESH, 1e-6)
    progress = np.clip((initial_goal_distance - goal_distance) / span, 0.0, 1.0)
    per_step = has_target * (APPROACH_WEIGHT * approach + PROGRESS_WEIGHT * progress)
    if tracking is not None and tracking_weight:
        per_step = per_step + tracking_weight * tracking
    return (per_step + completion_bonus * newly_completed).astype(np.float32)


def rollout(
    env: BatchedKitchen,
    controller: Callable[[np.ndarray], np.ndarray],
    steps: int,
    *,
    seeds: np.ndarray | None = None,
    auto_reset: bool = False,
) -> dict[str, Any]:
    """Drive ``env`` with ``controller(obs_30) -> [N, 9]`` and report the benchmark score.

    Used for the random-policy and scripted-teacher checks; the score is the benchmark's, the
    number of the split's tasks each episode ever completed.
    """
    obs = env.reset(seeds=seeds)
    total = np.zeros(env.num_envs)
    completed = np.zeros((env.num_envs, len(env.tasks)), dtype=bool)
    taken = 0
    for _ in range(steps):
        result = env.step(np.asarray(controller(obs), dtype=np.float64), auto_reset=auto_reset)
        total += result.reward
        completed |= result.completed
        obs = result.obs
        taken += 1
        if not auto_reset and (result.terminated | result.truncated).all():
            break
    counts = completed.sum(1)
    return {
        "steps": taken,
        "mean_episode_reward": float(total.mean()),
        "mean_step_reward": float(total.mean() / max(taken, 1)),
        "tasks_completed": counts.tolist(),
        "mean_tasks": float(counts.mean()),
        "normalized_score": float(25.0 * counts.mean()),
        "per_task_success": {
            task: float(completed[:, i].mean()) for i, task in enumerate(env.tasks)
        },
    }
