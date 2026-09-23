"""Official-environment evaluation of kitchen checkpoints, with the motion-quality metrics.

Runs every episode to the 280-step horizon (the benchmark's termination on four tasks is turned
off) so that the final state is what the strict score measures, and reports per condition:
the benchmark score (tasks ever completed, threshold 0.3), the strict score (elements within
0.1 of their goal at the end), per-task final distances, and the share of saturated commands.

Usage: uv run python scripts/kitchen_final_eval.py OUTPUT.json LABEL=RUN/policy-XXXX.safetensors ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import (
    OBS_ELEMENT_GOALS,
    OBS_ELEMENT_INDICES,
)

from flyarm.benchmarks import kitchen
from flyarm.cli import DEFAULT_PACK
from flyarm.config import KitchenPPOConfig
from flyarm.rl.kitchen_quality import STRICT_THRESHOLD
from flyarm.rl.ppo_kitchen import load_kitchen_policy
from flyarm.whole_brain.policy import MlxController

TASKS = ("microwave", "kettle", "light switch", "slide cabinet")
CONDITIONS = {"benchmark start": 0.0, "0.2 rad perturbed": 0.2, "0.3 rad perturbed": 0.3}
EPISODES = 50


def evaluate(checkpoint: Path) -> dict[str, dict[str, object]]:
    config = KitchenPPOConfig.model_validate_json((checkpoint.parent / "config.json").read_text())
    policy = load_kitchen_policy(config, Path(DEFAULT_PACK))
    policy.load(checkpoint)
    controller = kitchen.PositionFeatures(MlxController(policy))
    env = kitchen.recover_env("complete")
    env.unwrapped.terminate_on_tasks_completed = False
    results: dict[str, dict[str, object]] = {}
    try:
        for name, offset in CONDITIONS.items():
            counts, strict, saturated, final = [], [], [], []
            for seed in range(EPISODES):
                observation, _ = env.reset(seed=seed)
                if offset:
                    observation = kitchen._offset_initial_joints(env, seed, offset)
                controller.reset()
                completed: list[str] = []
                commands = []
                for _ in range(env.spec.max_episode_steps):
                    action = controller.act(np.asarray(observation["observation"], np.float32))
                    commands.append(np.abs(action) > 0.95)
                    observation, _, _, truncated, info = env.step(action.astype(np.float64))
                    completed = list(info["episode_task_completions"])
                    if truncated:
                        break
                qpos = env.unwrapped.data.qpos
                distance = [
                    float(np.linalg.norm(qpos[OBS_ELEMENT_INDICES[t]] - OBS_ELEMENT_GOALS[t]))
                    for t in TASKS
                ]
                counts.append(len(completed))
                strict.append(sum(d < STRICT_THRESHOLD for d in distance))
                saturated.append(float(np.mean(commands)))
                final.append(distance)
            final_array = np.array(final)
            results[name] = {
                "episodes": EPISODES,
                "benchmark_score": 25.0 * float(np.mean(counts)),
                "all_four": int(sum(c == 4 for c in counts)),
                "strict_score": 25.0 * float(np.mean(strict)),
                "strict_per_task": {
                    t: float(np.mean(final_array[:, i] < STRICT_THRESHOLD))
                    for i, t in enumerate(TASKS)
                },
                "final_distance_median": {
                    t: float(np.median(final_array[:, i])) for i, t in enumerate(TASKS)
                },
                "saturated_fraction": float(np.mean(saturated)),
            }
            print(checkpoint, name, json.dumps(results[name]), flush=True)
    finally:
        env.close()
    return results


def main() -> None:
    output = Path(sys.argv[1])
    report = {}
    for item in sys.argv[2:]:
        label, path = item.split("=", 1)
        report[label] = {"checkpoint": path, "conditions": evaluate(Path(path))}
        output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
