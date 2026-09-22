"""Open-loop baselines for B1a pick-and-place: does the task need perception?

The cube and goal are placed anew in every episode, so a policy that ignores its observation
should fail. Baselines on the protocol's 24 test seeds (60000 to 60023):
- zero action;
- replay of the recorded teacher actions of REPLAYS training demonstrations, each replayed on
  every test seed.
The teacher (closed loop, full state) is the reference.

Usage:

    PYTHONPATH=src uv run python scripts/pick_place_open_loop.py \
        docs/results/pick-place-open-loop.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from flyarm.config import WholeBrainConfig
from flyarm.whole_brain.experiment import Task

RUN = Path("runs/whole-brain-pick-place-v2a")
MODEL = Path("assets/menagerie/franka_emika_panda/scene.xml")
REPLAYS = 10


class Replay:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        action = self.actions[min(self.step, len(self.actions) - 1)]
        self.step += 1
        return action


def counts(result: dict) -> dict:
    episodes = result["episodes"]
    return {
        "episodes": len(episodes),
        "placed": int(sum(e["success"] for e in episodes)),
        "lifted": int(sum(e["ever_lifted"] for e in episodes)),
        "grasped": int(sum(e["ever_grasped"] for e in episodes)),
    }


def main(output: Path) -> None:
    config = WholeBrainConfig.model_validate_json((RUN / "config.json").read_text())
    splits = json.loads((RUN / "splits.json").read_text())
    seeds = [int(s) for s in splits["test"]]
    data = np.load(RUN / "train.npz")
    task = Task(config, MODEL)
    try:
        result: dict = {"test_seeds": seeds, "replay": {}}
        result["teacher"] = counts(task.baseline("teacher", seeds))
        result["zero"] = counts(task.baseline("zero", seeds))
        print("teacher", result["teacher"], "zero", result["zero"], flush=True)
        for row in range(REPLAYS):
            steps = int(data["mask"][row].sum())
            replay = Replay(data["actions"][row][:steps])
            result["replay"][str(int(data["seeds"][row]))] = counts(task.evaluate(replay, seeds))
            print(
                "replay",
                int(data["seeds"][row]),
                result["replay"][str(int(data["seeds"][row]))],
                flush=True,
            )
    finally:
        task.close()
    totals = {
        key: sum(r[key] for r in result["replay"].values())
        for key in ("placed", "lifted", "grasped")
    }
    result["replay_totals"] = {"episodes": REPLAYS * len(seeds), **totals}
    print("replay totals", result["replay_totals"], flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
