"""Can a harder kitchen start separate feedback control from trajectory replay?

For initial joint offsets of 0.1 to 0.3 rad (seeded, uniform per arm joint, as in the
protocol's OOD variants), scores on TEST_SEEDS seeds: the closed-loop demonstration tracker
(nearest demonstrated state plus a joint correction, the kitchen teacher) and the open-loop
replay of one training demonstration (E29).
If the tracker keeps its score where the replay collapses, perturbed starts make the kitchen a
closed-loop test, and the tracker can label recovery data for them.

Usage:

    PYTHONPATH=src uv run python scripts/kitchen_perturbation_probe.py \
        docs/results/kitchen-perturbation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flyarm.benchmarks import kitchen
from flyarm.benchmarks.kitchen_expert import DemonstrationTracker

RUN = Path("runs/flyleg-kitchen-complete-chunk-001")
OFFSETS, TEST_SEEDS = (0.0, 0.1, 0.2, 0.3), 20


def main(output: Path) -> None:
    splits = json.loads((RUN / "splits.json").read_text())
    seeds = [int(s) for s in splits["evaluation_seeds"]][:TEST_SEEDS]
    data = kitchen.load("complete")
    tracker = DemonstrationTracker.from_data(data)
    index = {episode: row for row, episode in enumerate(data.episode_ids.tolist())}
    row = index[splits["train_episode_ids"][0]]
    replay_actions = data.actions[row][data.mask[row].astype(bool)]
    env = kitchen.recover_env("complete")
    result: dict = {"test_seeds": seeds, "offsets": {}}
    for offset in OFFSETS:
        tracked = kitchen.evaluate(env, tracker, seeds, initial_joint_offset=offset)
        replayed = kitchen.evaluate(
            env, None, seeds, replay_actions=replay_actions, initial_joint_offset=offset
        )
        result["offsets"][str(offset)] = {
            "tracker": tracked["normalized_score"],
            "tracker_per_task": tracked["per_task_success"],
            "replay": replayed["normalized_score"],
            "replay_per_task": replayed["per_task_success"],
        }
        print(
            offset,
            "tracker",
            tracked["normalized_score"],
            "replay",
            replayed["normalized_score"],
            flush=True,
        )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
