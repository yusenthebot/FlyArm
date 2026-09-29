"""Open-loop baselines for the kitchen benchmark: how much needs no perception at all?

Every baseline ignores the observation:
- constant actions: zero, the mean demonstration action over whole episodes and over the
  first MICROWAVE_STEPS steps, and RANDOM_CONSTANTS uniform draws from [-1, 1]^9;
- replay: the recorded action sequence of a training demonstration, replayed from the
  benchmark's reset, and from resets with the protocol's initial joint offsets.
The benchmark's robot and object noise ratios only perturb observations, so they cannot
affect an open-loop baseline; only the joint offsets perturb the physical start state.
Test seeds are the kitchen protocol v2 evaluation seeds (runs/flyleg-kitchen-complete-chunk-001).

Usage:

    PYTHONPATH=src uv run python scripts/kitchen_open_loop.py docs/results/kitchen-open-loop.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from flyarm.benchmarks import kitchen

RUN = Path("runs/flyleg-kitchen-complete-chunk-001")
MICROWAVE_STEPS, RANDOM_CONSTANTS, RANDOM_SEEDS, REPLAYS = 60, 16, 10, 5
JOINT_OFFSETS = (0.05, 0.1)  # the protocol's OOD initial joint offsets, in rad


class Constant:
    def __init__(self, action: np.ndarray) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.action


def summary(result: dict) -> dict:
    return {
        "normalized_score": result["normalized_score"],
        "per_task_success": result["per_task_success"],
        "episodes": len(result["episodes"]),
    }


def main(output: Path) -> None:
    splits = json.loads((RUN / "splits.json").read_text())
    seeds = [int(s) for s in splits["evaluation_seeds"]]
    data = kitchen.load("complete")
    env = kitchen.recover_env("complete")
    valid = data.mask.astype(bool)
    constants = {
        "zero": np.zeros(kitchen.ACTION_DIM),
        "mean demonstration action": data.actions[valid].mean(0),
        f"mean action of the first {MICROWAVE_STEPS} steps": data.actions[:, :MICROWAVE_STEPS][
            valid[:, :MICROWAVE_STEPS]
        ].mean(0),
    }
    result: dict = {"test_seeds": seeds, "constant": {}, "random_constant": [], "replay": {}}
    for name, action in constants.items():
        result["constant"][name] = summary(kitchen.evaluate(env, Constant(action), seeds))
        print(name, result["constant"][name], flush=True)
    generator = np.random.default_rng(0)
    for _ in range(RANDOM_CONSTANTS):
        action = generator.uniform(-1, 1, kitchen.ACTION_DIM)
        row = summary(kitchen.evaluate(env, Constant(action), seeds[:RANDOM_SEEDS]))
        result["random_constant"].append({"action": action.tolist(), **row})
        print("random", round(row["normalized_score"], 2), flush=True)
    index = {episode: i for i, episode in enumerate(data.episode_ids.tolist())}
    for offset in (0.0, *JOINT_OFFSETS):
        key = "replay" if offset == 0.0 else f"replay_joint_offset_{offset:g}rad"
        result[key] = {}
        for episode in splits["train_episode_ids"][:REPLAYS]:
            row = index[episode]
            actions = data.actions[row][valid[row]]
            replay = kitchen.evaluate(
                env, None, seeds, replay_actions=actions, initial_joint_offset=offset
            )
            result[key][str(episode)] = summary(replay)
            print(key, episode, result[key][str(episode)]["normalized_score"], flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
