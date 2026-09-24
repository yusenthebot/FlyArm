"""Short imitation diagnostic: train one controller through the real pipeline, test closed loop.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/manipulation_imitation_probe.py \\
        --policy mlp --train-per-template 8 --epochs 12 --output runs/probe-mlp

Collects teacher demonstrations of the train split, trains ``--policy`` with
flyarm.manipulation.imitation (normalization, calibration, sequence behavior cloning, DAgger
rounds, closed-loop phase selection) and evaluates the selected checkpoint closed loop on
train-split episodes that no stage trains or selects on (offset 150,000), per template: success
and how many episodes completed at least k subgoals.

``--mask-heading-features`` zeroes the three observation features added by the imitation
failure analysis (the yaw command and the two heading errors) in the training data and in
every closed-loop step, which reproduces the earlier observation; it runs behavior cloning only
(the DAgger rollouts would see the features). See docs/MANIPULATION_ENV.md, "Imitation failure
analysis".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from flyarm.config import ManipulationImitationConfig
from flyarm.manipulation import rollout
from flyarm.manipulation.env import DEFAULT_ASSET_ROOT
from flyarm.manipulation.imitation import (
    PolicyActor,
    Workbench,
    _train,
    brain_budget,
    build_policy,
    dynamics_for,
)
from flyarm.manipulation.sim import ROBOT_DIM, cue_slices

PROBE_OFFSET = 150_000  # train-split seeds between validation (100,000) and DAgger (200,000)


def heading_columns() -> list[int]:
    cue = cue_slices()
    return [
        ROBOT_DIM - 1,  # yaw_command, the last robot field
        cue["grasp_heading_error"].start,
        cue["place_heading_error"].start,
    ]


class MaskedActor:
    """Zeroes the given observation columns before the policy sees them."""

    def __init__(self, inner: PolicyActor, columns: list[int]) -> None:
        self.inner, self.columns = inner, columns

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        obs[:, self.columns] = 0.0
        return self.inner.act(obs)


def histogram(log: rollout.EpisodeLog) -> dict[str, Any]:
    names = np.array(log.plan.templates)
    out = {}
    for name in dict.fromkeys(log.plan.templates):
        rows = names == name
        total = int(log.subgoals_total[rows][0])
        out[name] = {
            "success": f"{int(log.success[rows].sum())}/{int(rows.sum())}",
            "at_least_k_subgoals": [
                int((log.subgoals[rows] >= k).sum()) for k in range(1, total + 1)
            ],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["mlp", "gru", "connectome"], default="mlp")
    parser.add_argument("--train-per-template", type=int, default=8)
    parser.add_argument("--val-per-template", type=int, default=2)
    parser.add_argument("--test-per-template", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--dagger-iterations", type=int, default=1)
    parser.add_argument("--dagger-per-template", type=int, default=4)
    parser.add_argument("--dagger-epochs", type=int, default=4)
    parser.add_argument("--select-every", type=int, default=4)
    parser.add_argument("--mask-heading-features", action="store_true")
    parser.add_argument(
        "--bc-only",
        action="store_true",
        help="behavior cloning selected on imitation loss, no DAgger (as the masked run)",
    )
    parser.add_argument("--templates", nargs="+", default=None)
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    masked = args.mask_heading_features
    bc_only = masked or args.bc_only
    config = ManipulationImitationConfig(
        train_episodes_per_template=args.train_per_template,
        val_episodes_per_template=args.val_per_template,
        epochs=args.epochs,
        select_every=args.select_every,
        selection="validation_loss" if bc_only else "closed_loop",
        phase_selection="last" if bc_only else "validation_success",
        dagger_iterations=0 if bc_only else args.dagger_iterations,
        dagger_episodes_per_template=args.dagger_per_template,
        dagger_epochs=args.dagger_epochs,
        policies=[args.policy],
    )
    started = time.monotonic()
    bench = Workbench(args.model, args.asset_root, config.cue, config.velocities)
    train, _ = bench.collect(rollout.plan("train", args.train_per_template, 0, args.templates))
    validation, _ = bench.collect(
        rollout.plan("train", args.val_per_template, rollout.VALIDATION_OFFSET, args.templates)
    )
    columns = heading_columns()
    if masked:
        for data in (train, validation):
            data["obs"][..., columns] = 0.0
    dynamics, budget = None, 412_543
    if args.policy == "connectome":
        from flyarm.whole_brain.compiler import ConnectomePack
        from flyarm.whole_brain.interface import annotation_interface

        pack = ConnectomePack.load(args.pack)
        pack.validate_b1a_provenance()
        interface = annotation_interface(pack)
        dynamics, _ = dynamics_for("connectome", 0, pack, interface)
        budget = brain_budget(pack, interface, config.neural_steps)
    policy = build_policy(args.policy, config, 0, dynamics, budget)
    curves, training, _ = _train(
        policy, config, bench, train, validation, 0, time.monotonic() + 36_000, args.output
    )
    test = rollout.plan("train", args.test_per_template, PROBE_OFFSET, args.templates)

    def actor_for(n: int) -> Any:
        actor = PolicyActor(policy, n)
        return MaskedActor(actor, columns) if masked else actor

    (log,) = bench.run([test], actor_for)
    summary = rollout.summarize(log)
    record = {
        "policy": args.policy,
        "masked_heading_features": masked,
        "bc_only": bc_only,
        "config": config.model_dump(),
        "demonstration_steps": int(train["mask"].sum()),
        "final_train_loss": curves[-1]["train_loss"],
        "final_validation_loss": curves[-1]["validation_loss"],
        "selected_phase": training["selected_phase"],
        "test": {
            "episodes": summary["episodes"],
            "successes": summary["successes"],
            "subgoal_fraction": summary["subgoal_fraction"],
            "per_template": histogram(log),
            "stray_contact_fraction": summary["stray_contact_fraction"],
        },
        "minutes": round((time.monotonic() - started) / 60, 1),
    }
    (args.output / "probe.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["test"], indent=1))
    print(
        f"train L1 {record['final_train_loss']:.4f} val {record['final_validation_loss']:.4f} "
        f"selected {record['selected_phase']} in {record['minutes']} min"
    )


if __name__ == "__main__":
    main()
