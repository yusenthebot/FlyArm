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


def minimum_completion_bonus(gamma: float = STALL_GAMMA) -> float:
    """Value of stalling forever on the per-step maximum; a bonus must exceed it (E34)."""
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    return MAX_STEP_REWARD / (1.0 - gamma)


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
        terminate_on_all_tasks: bool = True,
        tasks: tuple[str, ...] = COMPLETE_TASKS,
    ) -> None:
        if num_envs < 1 or horizon < 20:
            raise ValueError("num_envs must be positive and horizon at least 20")
        if completion_bonus <= minimum_completion_bonus(gamma):
            raise ValueError(
                f"completion_bonus {completion_bonus} must exceed the value of stalling on the "
                f"per-step maximum, {minimum_completion_bonus(gamma):.1f} at gamma {gamma} "
                "(research log E34)"
            )
        if approach_slope <= 0:
            raise ValueError("approach_slope must be positive")
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

    def target(self) -> np.ndarray:
        """[N] index of the first task of the split that is not yet completed (len when done)."""
        remaining = ~self.completed
        return np.where(remaining.any(1), remaining.argmax(1), len(self.tasks))

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
        reward = shaped_reward(
            self.approach_distance(target),
            distance[rows, shaped],
            self.initial_goal_distance[rows, shaped],
            newly,
            target < len(self.tasks),
            self.completion_bonus,
            self.approach_slope,
        )
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
) -> np.ndarray:
    """Approach the current target element, move it toward its goal, complete tasks.

    The per-step terms are bounded by ``MAX_STEP_REWARD`` and are paid for the first
    uncompleted task of the split only, so completing a task never costs shaping reward: the
    shaping simply moves on to the next element. ``completion_bonus`` must exceed
    ``minimum_completion_bonus(gamma)``, otherwise stalling at the completion threshold of one
    element is worth more than completing it (research log E34).
    """
    approach = 1.0 - np.tanh(approach_slope * approach_distance)
    span = np.maximum(initial_goal_distance - BONUS_THRESH, 1e-6)
    progress = np.clip((initial_goal_distance - goal_distance) / span, 0.0, 1.0)
    per_step = has_target * (APPROACH_WEIGHT * approach + PROGRESS_WEIGHT * progress)
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
