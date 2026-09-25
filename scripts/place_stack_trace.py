"""Why do learners fail place and stack? Per-episode traces and failure reasons from resets.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/place_stack_trace.py \\
        --run runs/skill-dagger-connectome-002 --kind connectome --round 3

Runs every place and stack start of the run's validation bank as a single-subgoal episode,
driven by the controller (or by the teacher with ``--teacher``), the teacher labelling every
state. Each episode ends in one reason, from what the scene shows over the episode:

- ``success``;
- ``carry``: still holding the object at the end, never within 8 mm (xy) of the target;
- ``turn``: holding, within 8 mm of the target but the heading off by more than 0.05 rad;
- ``lower``: holding, over the target and turned, never released;
- ``dropped``: let go (not grasped) with the object's bottom-centre more than 2 cm (xy) from
  the target;
- ``touching``: released over the target and inside it (a stack: on the base) but still
  touched by the hand at the end, so the rest test never counts;
- ``outside``: released over the target but the object is not inside the receptacle (a stack:
  not on the base's footprint, or not upright);
- ``unsettled``: inside and untouched at the end but not yet counted (still moving or too late).

Plus the teacher's phase shares, and the steps spent in each phase, per skill.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flyarm.manipulation import curriculum as cu
from flyarm.manipulation import rollout
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.imitation import load_manipulation_policy
from flyarm.manipulation.teacher import PHASES, ManipulationTeacher

sys.path.insert(0, str(Path(__file__).parent))

ARRIVED = 0.008
TURNED = 0.05
DROPPED = 0.02


def trace(env: Any, bank: cu.SubgoalBank, picks: np.ndarray, policy: Any) -> dict[str, Any]:
    n = len(picks)
    rows = env.rows
    obs = bank.reset(env, rows, picks, np.ones(n, dtype=np.int64))
    teacher = ManipulationTeacher(env)
    teacher.reset()
    state = None if policy is None else policy.initial_state(n)
    index = env.preset.copy()
    kind = env.sub_kind[rows, index].copy()
    obj = env.sub_obj[rows, index].copy()
    target = env.sub_target[rows, index].copy()
    horizon = int(env.horizons.max())
    active = np.ones(n, bool)
    success = np.zeros(n, bool)
    steps = np.zeros(n, np.int64)
    ever_arrived = np.zeros(n, bool)
    ever_released = np.zeros(n, bool)
    phase_steps: list[Counter] = [Counter() for _ in range(n)]
    last: dict[str, np.ndarray] = {}
    for _ in range(horizon):
        if not active.any():
            break
        scene = teacher._scene()
        label = teacher.act()
        if policy is None:
            action = label.astype(np.float64)
        else:
            output, state = policy.step(mx.array(obs.astype(np.float32)), state)
            mx.eval(output, state)
            action = np.clip(np.asarray(output, dtype=np.float64), -1, 1)
        goal = scene["targets"][rows, index]
        centre = scene["object_pos"][rows, obj]
        xy = np.linalg.norm(goal[:, :2] - centre[:, :2], axis=1)
        yaw = np.array(
            [teacher._yaw_error(r, float(scene["place_headings"][r, index[r]])) for r in rows]
        )
        grasped = scene["grasped"][rows, obj]
        for r in np.flatnonzero(active):
            phase_steps[r][PHASES[int(teacher.phase[r])]] += 1
        ever_arrived |= active & (xy < ARRIVED) & (yaw < TURNED)
        ever_released |= active & ~grasped
        snapshot = {
            "xy": xy,
            "yaw": yaw,
            "grasped": grasped,
            "touched": scene["touched"][rows, obj],
            "inside": scene["inside"][rows, obj, np.clip(target, 0, scene["inside"].shape[2] - 1)],
            "stack_geometry": scene["stacked_geometry"][
                rows, obj, np.clip(target, 0, scene["stacked_geometry"].shape[2] - 1)
            ],
        }
        for key, value in snapshot.items():
            last.setdefault(key, np.zeros(n, value.dtype))
            last[key][active] = value[active]
        result = env.step(action, auto_reset=False)
        steps += active
        success |= active & result.success
        active &= ~(result.success | result.truncated)
        obs = result.obs
    reason = np.empty(n, dtype=object)
    for r in rows:
        stack = kind[r] == tk.STACK
        if success[r]:
            reason[r] = "success"
        elif last["grasped"][r]:
            if last["xy"][r] >= ARRIVED:
                reason[r] = "carry"
            elif last["yaw"][r] >= TURNED:
                reason[r] = "turn"
            else:
                reason[r] = "lower"
        elif last["xy"][r] > DROPPED:
            reason[r] = "dropped"
        elif (
            not (last["stack_geometry"][r] if stack else last["inside"][r])
            and not last["touched"][r]
        ):
            reason[r] = "outside"
        elif last["touched"][r]:
            reason[r] = "touching"
        else:
            reason[r] = "unsettled"
    out: dict[str, Any] = {}
    for skill in (tk.PLACE, tk.STACK):
        mask = kind == skill
        if not mask.any():
            continue
        phases: Counter = Counter()
        for r in np.flatnonzero(mask):
            phases.update(phase_steps[r])
        total = sum(phases.values())
        out[tk.SKILLS[skill]] = {
            "episodes": int(mask.sum()),
            "success_rate": round(float(success[mask].mean()), 3),
            "reasons": dict(Counter(reason[mask]).most_common()),
            "mean_steps": round(float(steps[mask].mean()), 1),
            "mean_steps_to_success": (
                round(float(steps[mask & success].mean()), 1) if (mask & success).any() else None
            ),
            "ever_over_target_and_turned": round(float(ever_arrived[mask].mean()), 3),
            "ever_released": round(float(ever_released[mask].mean()), 3),
            "teacher_phase_share": {k: round(v / total, 3) for k, v in phases.most_common()},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", default="connectome")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--teacher", action="store_true", help="the teacher drives")
    parser.add_argument("--bank", type=Path, default=None)
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    bank = cu.SubgoalBank.load(args.bank or args.run / "validation-bank.npz")
    picks = np.flatnonzero(np.isin(bank.skill, (tk.PLACE, tk.STACK)))
    policy = None
    cue, velocities = True, False
    if not args.teacher:
        checkpoint = None
        if args.round is not None:
            checkpoint = args.run / f"{args.kind}-{args.seed}" / f"round-{args.round:02d}"
            checkpoint = checkpoint / "policy.safetensors"
        config, policy = load_manipulation_policy(
            args.run, args.kind, args.seed, args.pack, checkpoint
        )
        cue, velocities = config.cue, config.velocities
    plan = rollout.EpisodePlan(
        "train",
        tuple(int(s) for s in bank.seeds[picks]),
        tuple(str(t) for t in bank.templates[picks]),
    )
    env = rollout.make_env(
        args.model, plan, asset_root=args.asset_root, cue=cue, velocities=velocities
    )
    row = {
        "run": str(args.run),
        "driver": "teacher" if args.teacher else f"{args.kind} round {args.round}",
        **trace(env, bank, picks, policy),
    }
    print(json.dumps(row, indent=1), flush=True)
    if args.output is not None:
        rows = json.loads(args.output.read_text()) if args.output.is_file() else []
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
