"""Where does a manipulation checkpoint fail? Per split and template: success of the checkpoint
and of the teacher on the same fresh episodes, and the skill at which failed episodes stop.

    FLYARM_SIM_THREADS=8 PYTHONPATH=src .venv/bin/python scripts/manipulation_failure_analysis.py \\
        --policy ppo:runs/ppo-manipulation-skill-dagger-002 --output docs/results/failures.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from final_evaluation import FINAL_OFFSET, SPLITS, resolve

from flyarm.manipulation import rollout
from flyarm.manipulation.imitation import PolicyActor, Workbench
from flyarm.manipulation.tasks import TEMPLATES


def stop_skill(template: str, done: int) -> str:
    steps = TEMPLATES[template].steps
    if done >= len(steps):
        return "done"
    skill, _, target = steps[done]
    return f"{skill}:{target}" if skill in ("place", "stack") and target else skill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes-per-template", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument(
        "--offset",
        type=int,
        default=FINAL_OFFSET,
        help="seed offset per template block (100 is the development block, research log)",
    )
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, policy, record = resolve(args.policy, args.pack)
    bench = Workbench(args.model, args.asset_root, config.cue, config.velocities, config.phase_cue)
    report: dict = {"policy": record, "splits": {}}
    for split in args.splits:
        plan = rollout.plan(split, args.episodes_per_template, args.offset)
        (learner,) = bench.run([plan], lambda n: PolicyActor(policy, n))
        (teacher,) = bench.run([plan], lambda n: None)
        rows: dict = {}
        stops: Counter = Counter()
        for template in dict.fromkeys(plan.templates):
            idx = [i for i, t in enumerate(plan.templates) if t == template]
            failed = [i for i in idx if not learner.success[i]]
            where = Counter(stop_skill(template, int(learner.subgoals[i])) for i in failed)
            stops.update(where)
            rows[template] = {
                "episodes": len(idx),
                "learner": int(learner.success[idx].sum()),
                "teacher": int(teacher.success[idx].sum()),
                "learner_fails_teacher_succeeds": int(sum(1 for i in failed if teacher.success[i])),
                "failed_at": dict(where),
            }
        n = len(plan.seeds)
        report["splits"][split] = {
            "episodes": n,
            "learner": int(learner.success.sum()),
            "teacher": int(teacher.success.sum()),
            "failed_at": dict(stops),
            "per_template": rows,
        }
        print(
            split,
            f"learner {learner.success.sum()}/{n} teacher {teacher.success.sum()}/{n}",
            dict(stops.most_common()),
            flush=True,
        )
        args.output.write_text(json.dumps(report, indent=1))
    total = Counter()
    for split in report["splits"].values():
        total.update(split["failed_at"])
    report["failed_at_all_splits"] = dict(total.most_common())
    args.output.write_text(json.dumps(report, indent=1))
    print("all splits", dict(total.most_common()))


if __name__ == "__main__":
    main()
