"""Is held-out performance lost to out-of-range inputs? Probe observations at test time.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src:scripts .venv/bin/python scripts/ood_clamp_probe.py \\
        --policy ppo:runs/ppo-manipulation-nophase-001@500 \\
        --run runs/skill-dagger-connectome-nophase-001/connectome-0 \\
        --output docs/results/ood-clamp-probe.json

The training range of every observation dimension is its minimum and maximum over the
imitation run's aggregated data (every round's data.npz). The trained
controller then runs unchanged except that its observation is clipped to that range before
normalization: on the furniture block only, or on every dimension. If held-out furniture fails
because the encoder extrapolates poorly beyond the training ranges, clamping should help there
and do nothing on iid episodes. Episodes come from the development seeds (DEV_OFFSET onward in
each template block, research log "Generalization goal"), never the final ones, because this
probe informs design; paired McNemar tests against the unclamped controller.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from final_evaluation import resolve, wilson

from flyarm.io import save_json
from flyarm.manipulation import rollout
from flyarm.manipulation.imitation import PolicyActor, Workbench
from flyarm.manipulation.sim import FURNITURE_DIM, OBS_DIM, ROBOT_DIM, SLOT_DIM, S, cue_slices
from flyarm.whole_brain.lesion import mcnemar

DEV_OFFSET = 100  # per template block: in-run evaluation uses 0 to 7, the final 500 onward
FURNITURE = slice(ROBOT_DIM + S * SLOT_DIM, ROBOT_DIM + S * SLOT_DIM + FURNITURE_DIM)


def clip_to(low: mx.array, high: mx.array) -> Callable[[mx.array], mx.array]:
    return lambda obs: mx.clip(obs, low, high)


def scale_by(mask: mx.array) -> Callable[[mx.array], mx.array]:
    return lambda obs: obs * mask


def training_range(run: Path) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(OBS_DIM, np.inf, dtype=np.float32)
    high = np.full(OBS_DIM, -np.inf, dtype=np.float32)
    for path in sorted(run.glob("round-*/data.npz")):
        data = np.load(path)
        obs = data["obs"]  # episodes stored end to end, no padding
        low, high = np.minimum(low, obs.min(0)), np.maximum(high, obs.max(0))
    if not np.isfinite(low).all():
        raise ValueError(f"no round data under {run}")
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--run", type=Path, required=True, help="imitation run member directory")
    parser.add_argument("--splits", nargs="+", default=["unseen_furniture", "iid_test"])
    parser.add_argument("--per-template", type=int, default=16)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["none", "furniture", "all"],
        help="none, furniture, all (clamps), progress_zero (subgoals-done fraction set to 0)",
    )
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, policy, provenance = resolve(args.policy, args.pack)
    low, high = training_range(args.run)
    transforms: dict[str, Any] = {"none": None}
    for name, dims in (("furniture", FURNITURE), ("all", slice(0, OBS_DIM))):
        lo = np.full(OBS_DIM, -np.inf, dtype=np.float32)
        hi = np.full(OBS_DIM, np.inf, dtype=np.float32)
        lo[dims], hi[dims] = low[dims], high[dims]
        transforms[name] = clip_to(mx.array(lo), mx.array(hi))
    # Every training template opens its drawer as the first subgoal, so the cue's
    # subgoals-done fraction is always 0 there; this shows the controller that value.
    done = np.ones(OBS_DIM, dtype=np.float32)
    done[cue_slices()["progress"].start] = 0.0
    transforms["progress_zero"] = scale_by(mx.array(done))
    report: dict[str, Any] = (
        json.loads(args.output.read_text())
        if args.output.is_file()
        else {"policy": provenance, "run": str(args.run), "splits": {}}
    )
    bench = Workbench(args.model, args.asset_root, config.cue, config.velocities, config.phase_cue)
    normalize = policy.normalize
    for split in args.splits:
        rows = report["splits"].setdefault(split, {})
        plan = rollout.plan(split, args.per_template, DEV_OFFSET)
        for clamp in args.conditions:
            if clamp in rows:
                continue
            transform = transforms[clamp]
            policy.normalize = (  # type: ignore[method-assign]
                normalize if transform is None else (lambda obs, t=transform: normalize(t(obs)))
            )
            (log,) = bench.run([plan], lambda n: PolicyActor(policy, n))
            summary = rollout.summarize(log)
            low_ci, high_ci = wilson(summary["successes"], summary["episodes"])
            row: dict[str, Any] = {
                "successes": summary["successes"],
                "episodes": summary["episodes"],
                "wilson_95": [round(low_ci, 4), round(high_ci, 4)],
                "subgoal_fraction": round(summary["subgoal_fraction"], 4),
                "per_template": summary["per_template"],
                "outcomes": [int(s) for s in log.success],
            }
            if clamp != "none":
                row["mcnemar"] = mcnemar(np.array(rows["none"]["outcomes"], bool), log.success)
            rows[clamp] = row
            save_json(args.output, report)
            print(
                f"{split} clamp={clamp}: {row['successes']}/{row['episodes']}, "
                f"subgoals {row['subgoal_fraction']:.3f}"
                + (f", McNemar p {row['mcnemar']['p']:.2g}" if "mcnemar" in row else ""),
                flush=True,
            )
    policy.normalize = normalize  # type: ignore[method-assign]


if __name__ == "__main__":
    main()
