"""Pick among checkpoints on a larger validation set: train-split true starts from the
validation seed block (never used for training or testing).

    PYTHONPATH=src:scripts .venv/bin/python scripts/manipulation_select.py --per-template 16 \\
        --policy r08=file:runs/X:connectome:0:runs/X/connectome-0/round-08/policy.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_evaluation import resolve, wilson

from flyarm.manipulation import rollout
from flyarm.manipulation.imitation import PolicyActor, Workbench


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, help="LABEL=SPEC")
    parser.add_argument("--per-template", type=int, default=16)
    parser.add_argument(
        "--split",
        default="train",
        help="train (validation seeds) or a test split on its in-run evaluation seeds",
    )
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.output.read_text()) if args.output.is_file() else {}
    for item in args.policy:
        label, spec = item.split("=", 1)
        if label in report:
            continue
        config, policy, record = resolve(spec, args.pack)
        bench = Workbench(
            args.model, args.asset_root, config.cue, config.velocities, config.phase_cue
        )
        offset = rollout.VALIDATION_OFFSET if args.split == "train" else 0
        plan = rollout.plan(args.split, args.per_template, offset)
        (log,) = bench.run([plan], lambda n, p=policy: PolicyActor(p, n))
        summary = rollout.summarize(log)
        low, high = wilson(summary["successes"], summary["episodes"])
        report[label] = {
            **record,
            "episodes": summary["episodes"],
            "successes": summary["successes"],
            "success_rate": round(summary["success_rate"], 4),
            "wilson_95": [round(low, 4), round(high, 4)],
            "subgoal_fraction": round(summary["subgoal_fraction"], 4),
        }
        print(
            label,
            report[label]["successes"],
            "/",
            report[label]["episodes"],
            report[label]["wilson_95"],
            "subgoals",
            report[label]["subgoal_fraction"],
            flush=True,
        )
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
