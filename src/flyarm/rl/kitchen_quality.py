"""Motion-quality terms for the batched kitchen: finish each task, touch only the task, move gently.

The benchmark counts a task as done the moment its element comes within 0.3 of its goal, so a
controller that opens the slide cabinet less than halfway, nudges the kettle to the edge of its
goal region, sweeps a burner knob on the way and drives every joint at full speed scores the same
as one that works like the demonstrations (research log E50). These terms price those
differences; every weight defaults to 0, which leaves the reward of every earlier run unchanged.

* depth: potential-based, ``w * (phi(s') - phi(s))`` with ``phi = -sum_k min(d_k, 0.3) / 0.3``
  over the split's elements, that is only the last stretch, from the benchmark's success
  threshold to the exact goal, weighted the same for every task (the kettle's distance includes
  its orientation, so a fraction of its whole way would make its last stretch nearly free). The
  task shaping already pays for the way up to the threshold. Undoing a finished task is charged.
  It telescopes, so over an episode it is bounded by ``w`` times the number of tasks and does
  not enter the E34 stalling floor.
* disturbance: potential-based on every object joint outside the split (burner knobs, burners,
  the hinge cabinet), ``phi = -sum_j |q_j - q_j0|`` in radians, so knocking something pays
  negative once and putting it back pays it back.
* collision: ``-w`` for every control step in which any robot geom touches anything other than
  the movable body of the task being shaped for or of a task already completed (the microwave
  door, the kettle, the light switch, the sliding door), read from MuJoCo contact sensors. A
  completed task's body is allowed because the depth term asks for it to be pushed on to its
  exact goal; charging that contact made the two terms fight (E50: stray contact rose from
  0.15 to 0.21 to 0.32 of steps). The sensors are added to the benchmark's own model and change
  nothing in its physics (checked bit for bit).
* action and smoothness: ``-w * mean(a^2)`` and ``-w * mean((a_t - a_{t-1})^2)`` over the 9
  normalized joint-velocity commands; the demonstrations average ``|a| = 0.23`` with 2.4% of
  commands saturated, the four-task controller of E49 ``|a| = 0.74`` with 47.5% saturated.

The penalties are never positive, so none of them raises the per-step maximum either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

# Success under the strict reading: the element ends the episode within this distance of its
# goal. The benchmark's threshold is 0.3; the demonstrations end at a median of 0.065, 0.087,
# 0.008 and 0.021 for the microwave, kettle, light switch and slide cabinet (E50).
STRICT_THRESHOLD = 0.1
ROBOT_ROOT = "panda0_link0"
# The body a task is done with: contact with it is the task, contact with anything else is not.
TASK_CONTACT_BODIES = {
    "microwave": "microdoorroot",
    "kettle": "kettle",
    "light switch": "lightswitchroot",
    "slide cabinet": "slidelink",
}
_FOUND = 1  # contact sensor data: number of matching contacts
_REDUCE_MAXFORCE, _ONE = 2, 1


@dataclass(frozen=True)
class QualityWeights:
    depth: float = 0.0
    disturbance: float = 0.0
    collision: float = 0.0
    action: float = 0.0
    smoothness: float = 0.0

    def __post_init__(self) -> None:
        for name in ("depth", "disturbance", "collision", "action", "smoothness"):
            if getattr(self, name) < 0:
                raise ValueError(f"quality weight {name} must be non-negative")

    @property
    def active(self) -> bool:
        return any((self.depth, self.disturbance, self.collision, self.action, self.smoothness))


def build_model_with_contact_sensors(xml_path: str, tasks: tuple[str, ...]) -> Any:
    """The benchmark's model plus one contact counter for the robot and one per task body."""
    spec = mujoco.MjSpec.from_file(xml_path)
    names = ["robot_any"] + [f"robot_{task}" for task in tasks if task in TASK_CONTACT_BODIES]
    spec.add_sensor(
        name=names[0],
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,  # the robot's whole subtree
        objname=ROBOT_ROOT,
        intprm=[_FOUND, _REDUCE_MAXFORCE, _ONE],
    )
    for task in tasks:
        if task not in TASK_CONTACT_BODIES:
            continue
        spec.add_sensor(
            name=f"robot_{task}",
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=ROBOT_ROOT,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=TASK_CONTACT_BODIES[task],
            intprm=[_FOUND, _REDUCE_MAXFORCE, _ONE],
        )
    return spec.compile()


def contact_addresses(model: Any, tasks: tuple[str, ...]) -> tuple[int, np.ndarray]:
    """sensordata addresses of the robot's contact count and of each task body's (-1 if none)."""
    anything = int(model.sensor_adr[model.sensor("robot_any").id])
    per_task = np.array(
        [
            int(model.sensor_adr[model.sensor(f"robot_{task}").id])
            if task in TASK_CONTACT_BODIES
            else -1
            for task in tasks
        ]
    )
    return anything, per_task


def stray_contact(
    sensordata: np.ndarray,
    anything: int,
    per_task: np.ndarray,
    target: np.ndarray,
    completed: np.ndarray,
) -> np.ndarray:
    """[N] bool: the robot touches something other than the target's or a finished task's body.

    Each contact involves one robot geom and one other geom, so the per-body counts add up to
    at most the robot's total and the remainder is the stray contacts.
    """
    tasks = len(per_task)
    allowed_task = completed | (target[:, None] == np.arange(tasks)[None, :])
    allowed_task &= (per_task >= 0)[None, :]
    counts = sensordata[:, np.maximum(per_task, 0)]
    allowed = (counts * allowed_task).sum(1)
    return (sensordata[:, anything] - allowed) > 0


def depth_potential(goal_distance: np.ndarray, threshold: float) -> np.ndarray:
    """[N] -sum of each element's remaining share of the last stretch, in [-tasks, 0]."""
    return -(np.minimum(goal_distance, threshold) / threshold).sum(1)


def disturbance_potential(object_qpos: np.ndarray, initial: np.ndarray, mask: np.ndarray):
    """[N] -total displacement of the non-task object joints from where the episode began."""
    return -(np.abs(object_qpos - initial) * mask).sum(1)


def action_costs(action: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N] mean squared command and mean squared change of command."""
    return (action**2).mean(1), ((action - previous) ** 2).mean(1)
