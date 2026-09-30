"""Test-time lesions of the reported connectome controller (research log E63).

    PYTHONPATH=src:scripts .venv/bin/python scripts/lesion_analysis.py \\
        --policy ppo:runs/ppo-manipulation-nophase-001@500 \\
        --output docs/results/manipulation-lesions.json

Every condition runs the same trained encoder and decoder on a lesioned copy of the frozen
connectome (flyarm.whole_brain.lesion) over the same test episodes as the intact controller
(true starts from FINAL_OFFSET, so paired with the final evaluation's first episodes). Nothing
is retrained and no checkpoint is selected on these episodes. Per condition: successes, Wilson
interval, subgoal fraction, per-episode outcomes, and an exact McNemar test against intact.
Conditions already in the output are skipped (rerun to continue).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from final_evaluation import FINAL_OFFSET, resolve, wilson

from flyarm.io import save_json
from flyarm.manipulation import rollout
from flyarm.manipulation.imitation import PolicyActor, Workbench
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.lesion import REGIONS, LesionedDynamics, mcnemar, random_mask, region_mask
from flyarm.whole_brain.policy import BrainPolicy

RANDOM_DRAWS = 2


def conditions(pack: ConnectomePack, policy: BrainPolicy) -> dict[str, dict[str, Any]]:
    """Name -> {silenced mask or None, reset flag, description} of every lesion."""
    dynamics = policy.dynamics
    interface = np.concatenate(
        [np.asarray(dynamics.input_indices), np.asarray(dynamics.output_indices)]
    )
    neurons = pack.neurons()
    out: dict[str, dict[str, Any]] = {
        "intact": {"silenced": None, "reset": False, "what": "no lesion"},
        "state_reset": {
            "silenced": None,
            "reset": True,
            "what": "recurrent state cleared before every control step",
        },
    }
    for region in REGIONS:
        mask = region_mask(neurons, region, interface)
        out[f"silence_{region}"] = {
            "silenced": mask,
            "reset": False,
            "what": f"silence {', '.join(REGIONS[region])}",
        }
        for draw in range(RANDOM_DRAWS):
            out[f"random_as_{region}_{draw}"] = {
                "silenced": random_mask(int(mask.sum()), interface, pack.nodes, draw),
                "reset": False,
                "what": f"silence as many random non-interface neurons as {region} (draw {draw})",
            }
    everything = np.ones(pack.nodes, dtype=bool)
    everything[interface] = False
    out["silence_all_but_interface"] = {
        "silenced": everything,
        "reset": False,
        "what": "only the input and output neurons and the edges among them remain",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="a final_evaluation SPEC")
    parser.add_argument("--split", default="iid_test")
    parser.add_argument("--per-template", type=int, default=16)
    parser.add_argument("--only", nargs="*", help="run only these conditions")
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, policy, provenance = resolve(args.policy, args.pack)
    if not isinstance(policy, BrainPolicy):
        raise ValueError("lesions need a connectome controller")
    pack = ConnectomePack.load(args.pack)
    report: dict[str, Any] = (
        json.loads(args.output.read_text())
        if args.output.is_file()
        else {
            "policy": provenance,
            "protocol": {
                "split": args.split,
                "episodes_per_template": args.per_template,
                "seed_offset": FINAL_OFFSET,
                "test": "exact two-sided McNemar against intact on the same episodes",
            },
            "conditions": {},
        }
    )
    if report["protocol"]["episodes_per_template"] != args.per_template:
        raise ValueError("the output was written with another number of episodes per template")
    bench = Workbench(args.model, args.asset_root, config.cue, config.velocities, config.phase_cue)
    plan = rollout.plan(args.split, args.per_template, FINAL_OFFSET)
    table = conditions(pack, policy)
    names = ["intact", *(args.only or [name for name in table if name != "intact"])]
    for name in names:
        if name in report["conditions"]:
            continue
        spec = table[name]
        lesioned = policy.with_dynamics(
            policy.kind,
            LesionedDynamics(policy.dynamics, spec["silenced"], spec["reset"]),  # type: ignore[arg-type]
        )
        started = time.monotonic()
        (log,) = bench.run([plan], lambda n, p=lesioned: PolicyActor(p, n))
        summary = rollout.summarize(log)
        low, high = wilson(summary["successes"], summary["episodes"])
        row: dict[str, Any] = {
            "what": spec["what"],
            "silenced_neurons": 0 if spec["silenced"] is None else int(spec["silenced"].sum()),
            "episodes": summary["episodes"],
            "successes": summary["successes"],
            "success_rate": round(summary["success_rate"], 4),
            "wilson_95": [round(low, 4), round(high, 4)],
            "subgoal_fraction": round(summary["subgoal_fraction"], 4),
            "per_template": summary["per_template"],
            "outcomes": [int(s) for s in log.success],
            "seconds": round(time.monotonic() - started, 1),
        }
        if name != "intact":
            row["mcnemar"] = mcnemar(
                np.array(report["conditions"]["intact"]["outcomes"], bool), log.success
            )
        report["conditions"][name] = row
        save_json(args.output, report)
        print(
            f"{name}: {row['successes']}/{row['episodes']} ({low:.3f} to {high:.3f}), "
            f"subgoals {row['subgoal_fraction']:.3f}"
            + (f", McNemar p {row['mcnemar']['p']:.2g}" if "mcnemar" in row else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
