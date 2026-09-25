"""Why do skill-DAgger controllers fail closed loop? One hypothesis per ``--test``.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/skill_dagger_diagnostics.py \\
        --run runs/skill-dagger-connectome-001 --kind connectome --round 1 --test gain \\
        --output docs/results/skill-dagger-diagnostics.json

Every test runs single-subgoal episodes from the run's validation bank (up to
``--per-skill`` per skill), each started by the curriculum's exact restore, with the scripted
teacher labelling every state the controller visits.

- ``gain`` (H-A): the controller's actions multiplied by 1.0, 1.5 and 2.0 (then clipped) at
  evaluation; per-skill success and failure reason; per-dimension magnitude of the controller's
  commands and of the teacher's labels on the same states, and the slope of the controller's
  command on the teacher's label (1 is no shrinkage).
- ``labels`` (H-B): smoothness of the teacher's labels along the controller's trajectories and
  along the teacher's own: the fraction of steps where a dimension's label flips sign (both
  sides above 0.05 in magnitude) or jumps by more than 0.5.
- ``burnin`` (H-C): the teacher drives the first ``--burn-in`` steps while the controller
  watches, then the controller takes over with its state kept, or reset to zero at the handover.
- ``stuck`` (H-E): the share of the controller's steps where the hand moved under 5 mm over
  10 steps, the teacher's phase there, and the labels against the controller's commands.

Failure reasons, from the subgoal's own progress (the shaping potential's term, in [0, 1)):
``lost_grasp`` (an object skill held its object and let go without the subgoal),
``wrong_direction`` (progress ended more than 0.02 below where it started), ``stalled`` (within
0.02), ``timeout`` (progressed but ran out of steps).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flyarm.manipulation import curriculum as cu
from flyarm.manipulation import rollout
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.imitation import load_manipulation_policy
from flyarm.manipulation.teacher import ManipulationTeacher
from flyarm.whole_brain.policy import SequencePolicy

DIMENSIONS = ("x", "y", "z", "yaw", "grip")
OBJECT_SKILLS = (tk.PICK, tk.PLACE, tk.STACK)
BINS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0001)
FLIP_FLOOR = 0.05
JUMP = 0.5


def picks_per_skill(bank: cu.SubgoalBank, per_skill: int) -> np.ndarray:
    return np.concatenate(
        [np.flatnonzero(bank.skill == skill)[:per_skill] for skill in np.unique(bank.skill)]
    )


def run(
    env: Any,
    bank: cu.SubgoalBank,
    picks: np.ndarray,
    policy: SequencePolicy | None,
    *,
    gain: float = 1.0,
    teacher_steps: int = 0,
    reset_at_handover: bool = False,
) -> dict[str, Any]:
    """Single-subgoal episodes from ``picks``; ``policy`` None lets the teacher drive."""
    n = len(picks)
    rows = env.rows
    obs = bank.reset(env, rows, picks, np.ones(n, dtype=np.int64))
    teacher = ManipulationTeacher(env)
    teacher.reset()
    state = None if policy is None else policy.initial_state(n)
    index = env.preset.copy()
    kind = env.sub_kind[rows, index].copy()
    obj = env.sub_obj[rows, index].copy()
    horizon = int(env.horizons.max())
    labels = np.zeros((n, horizon, 5), np.float32)
    commands = np.zeros((n, horizon, 5), np.float32)
    live_mask = np.zeros((n, horizon), bool)
    progress = np.zeros((n, horizon + 1))
    held = np.zeros((n, horizon + 1), bool)
    success = np.zeros(n, bool)
    active = np.ones(n, bool)
    steps = np.zeros(n, np.int64)

    def measure(t: int) -> None:
        scene = env.scene_state()
        leading = env.leading(env.subgoal_done(env.effects(scene)))
        progress[:, t] = np.clip(env.potential(scene, leading) - index, 0.0, 1.0)
        held[:, t] = scene["grasped"][rows, obj]

    measure(0)
    for t in range(horizon):
        if not active.any():
            break
        label = teacher.act().astype(np.float64)
        executed = label
        if policy is not None:
            if reset_at_handover and t == teacher_steps:
                state = policy.initial_state(n)
            output, state = policy.step(mx.array(obs.astype(np.float32)), state)
            mx.eval(output, state)
            command = np.asarray(output, dtype=np.float64)
            commands[active, t] = command[active]
            if t >= teacher_steps:
                executed = np.clip(gain * command, -1.0, 1.0)
        labels[active, t] = label[active]
        live_mask[active, t] = t >= teacher_steps
        result = env.step(executed, auto_reset=False)
        steps += active
        success |= active & result.success
        finished = active & (result.success | result.truncated)
        measure(t + 1)
        progress[~active, t + 1] = progress[~active, t]
        active &= ~finished
        obs = result.obs
    last = np.clip(steps, 0, horizon)
    end = progress[rows, last]
    start = progress[rows, np.minimum(teacher_steps, last)]
    ever_held = np.array([held[r, teacher_steps : last[r] + 1].any() for r in rows])
    reason = np.where(end > start + 0.02, "timeout", "stalled").astype(object)
    reason[end < start - 0.02] = "wrong_direction"
    lost = np.isin(kind, OBJECT_SKILLS) & ever_held & ~held[rows, last]
    reason[lost] = "lost_grasp"
    reason[success] = "success"
    return {
        "kind": kind,
        "success": success,
        "reason": reason,
        "steps": steps,
        "labels": labels,
        "commands": commands,
        "mask": live_mask,
    }


def per_skill(result: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for skill in np.unique(result["kind"]):
        rows = result["kind"] == skill
        reasons, counts = np.unique(result["reason"][rows], return_counts=True)
        out[tk.SKILLS[int(skill)]] = {
            "episodes": int(rows.sum()),
            "success_rate": round(float(result["success"][rows].mean()), 3),
            "reasons": {str(r): int(c) for r, c in zip(reasons, counts, strict=True)},
        }
    out["mean_success"] = round(float(np.mean([v["success_rate"] for v in out.values()])), 3)
    return out


def magnitudes(result: dict[str, Any]) -> dict[str, Any]:
    """Controller's commands versus the teacher's labels on the states the controller visited."""
    mask = result["mask"]
    command = np.clip(result["commands"][mask], -1, 1)
    label = result["labels"][mask]
    out: dict[str, Any] = {"steps": int(mask.sum())}
    for d, name in enumerate(DIMENSIONS):
        c, lab = command[:, d], label[:, d]
        slope = float(np.dot(c, lab) / max(np.dot(lab, lab), 1e-9))
        out[name] = {
            "mean_abs_command": round(float(np.abs(c).mean()), 3),
            "mean_abs_label": round(float(np.abs(lab).mean()), 3),
            "saturated_command": round(float((np.abs(c) > 0.95).mean()), 3),
            "saturated_label": round(float((np.abs(lab) > 0.95).mean()), 3),
            "slope_on_label": round(slope, 3),
            "sign_agreement": round(
                float((np.sign(c) == np.sign(lab))[np.abs(lab) > 0.1].mean()), 3
            ),
            "l1": round(float(np.abs(c - lab).mean()), 3),
            "histogram_command": np.histogram(np.abs(c), BINS)[0].tolist(),
            "histogram_label": np.histogram(np.abs(lab), BINS)[0].tolist(),
        }
    big = np.abs(label[:, :3]).max(1) > 0.95  # the teacher at full speed in some direction
    ratio = np.linalg.norm(command[big, :3], axis=1) / np.linalg.norm(label[big, :3], axis=1)
    cosine = np.sum(command[big, :3] * label[big, :3], 1) / np.maximum(
        np.linalg.norm(command[big, :3], axis=1) * np.linalg.norm(label[big, :3], axis=1), 1e-9
    )
    out["full_speed_label_steps"] = int(big.sum())
    out["full_speed_norm_ratio"] = round(float(np.median(ratio)), 3) if big.any() else None
    out["full_speed_cosine"] = round(float(np.median(cosine)), 3) if big.any() else None
    return out


def phase_table(
    env: Any, bank: cu.SubgoalBank, picks: np.ndarray, policy: SequencePolicy | None
) -> dict[str, Any]:
    """Share of steps the teacher spends in each phase along the controller's trajectories."""
    from flyarm.manipulation.teacher import PHASES

    n = len(picks)
    obs = bank.reset(env, env.rows, picks, np.ones(n, dtype=np.int64))
    teacher = ManipulationTeacher(env)
    teacher.reset()
    state = None if policy is None else policy.initial_state(n)
    counts = np.zeros(len(PHASES))
    z_sign = {phase: [0, 0, 0] for phase in PHASES}
    active = np.ones(n, bool)
    for _ in range(int(env.horizons.max())):
        if not active.any():
            break
        label = teacher.act()
        for row in np.flatnonzero(active):
            phase = PHASES[int(teacher.phase[row])]
            counts[int(teacher.phase[row])] += 1
            z_sign[phase][int(np.sign(label[row, 2])) + 1] += 1
        if policy is None:
            executed = label.astype(np.float64)
        else:
            output, state = policy.step(mx.array(obs.astype(np.float32)), state)
            mx.eval(output, state)
            executed = np.clip(np.asarray(output, dtype=np.float64), -1, 1)
        result = env.step(executed, auto_reset=False)
        active &= ~(result.success | result.truncated)
        obs = result.obs
    share = counts / counts.sum()
    return {
        PHASES[k]: {"share": round(float(share[k]), 3), "z_down_zero_up": z_sign[PHASES[k]]}
        for k in range(len(PHASES))
        if counts[k]
    }


def stuck(
    env: Any, bank: cu.SubgoalBank, picks: np.ndarray, policy: SequencePolicy
) -> dict[str, Any]:
    """Where the controller's hand stands still, what does the teacher ask for, in which phase?"""
    from flyarm.manipulation.teacher import PHASES

    n = len(picks)
    obs = bank.reset(env, env.rows, picks, np.ones(n, dtype=np.int64))
    teacher = ManipulationTeacher(env)
    teacher.reset()
    state = policy.initial_state(n)
    horizon = int(env.horizons.max())
    ee = np.zeros((n, horizon, 3))
    labels = np.zeros((n, horizon, 5))
    commands = np.zeros((n, horizon, 5))
    phases = np.zeros((n, horizon), np.int64)
    alive = np.zeros((n, horizon), bool)
    active = np.ones(n, bool)
    for t in range(horizon):
        if not active.any():
            break
        labels[:, t] = teacher.act()
        output, state = policy.step(mx.array(obs.astype(np.float32)), state)
        mx.eval(output, state)
        commands[:, t] = np.clip(np.asarray(output, dtype=np.float64), -1, 1)
        ee[:, t], phases[:, t], alive[:, t] = env.ee(), teacher.phase, active
        result = env.step(commands[:, t], auto_reset=False)
        active &= ~(result.success | result.truncated)
        obs = result.obs
    moved = np.full((n, horizon), np.inf)
    moved[:, 10:] = np.linalg.norm(ee[:, 10:] - ee[:, :-10], axis=2)
    still = alive & (moved < 0.005)
    out: dict[str, Any] = {
        "alive_steps": int(alive.sum()),
        "stationary_share": round(float(still.sum() / alive.sum()), 3),
        "phases_when_stationary": {},
    }
    for phase in np.unique(phases[still]):
        rows = still & (phases == phase)
        out["phases_when_stationary"][PHASES[int(phase)]] = {
            "share": round(float(rows.sum() / still.sum()), 3),
            "mean_abs_label": np.round(np.abs(labels[rows]).mean(0), 3).tolist(),
            "mean_abs_command": np.round(np.abs(commands[rows]).mean(0), 3).tolist(),
        }
    descend = still & (phases == PHASES.index("descend"))
    if descend.any():  # the xy correction the teacher asks for, in mm (one action = 14 mm)
        xy = np.linalg.norm(labels[descend][:, :2], axis=1) * 14.0
        out["descend_xy_error_mm_quartiles"] = np.round(np.percentile(xy, [25, 50, 75]), 1).tolist()
    return out


def smoothness(labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Sign flips and jumps of consecutive labels inside each episode."""
    pair = mask[:, 1:] & mask[:, :-1]
    a, b = labels[:, :-1][pair], labels[:, 1:][pair]
    flip = (np.sign(a) != np.sign(b)) & (np.abs(a) > FLIP_FLOOR) & (np.abs(b) > FLIP_FLOOR)
    jump = np.abs(b - a) > JUMP
    out: dict[str, Any] = {"pairs": int(pair.sum())}
    for d, name in enumerate(DIMENSIONS):
        out[name] = {
            "flip": round(float(flip[:, d].mean()), 4),
            "jump": round(float(jump[:, d].mean()), 4),
        }
    out["any_flip_or_jump"] = round(float((flip | jump).any(1).mean()), 4)
    out["xyz_flip_or_jump"] = round(float((flip | jump)[:, :3].any(1).mean()), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, default=None, help="round checkpoint (skill DAgger)")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--test", choices=("gain", "labels", "burnin", "stuck"), required=True)
    parser.add_argument("--per-skill", type=int, default=12)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--bank", type=Path, default=None, help="default: the run's bank")
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if args.round is not None:
        checkpoint = args.run / f"{args.kind}-{args.seed}" / f"round-{args.round:02d}"
        checkpoint = checkpoint / "policy.safetensors"
    config, policy = load_manipulation_policy(args.run, args.kind, args.seed, args.pack, checkpoint)
    bank_path = args.bank or args.run / "validation-bank.npz"
    bank = cu.SubgoalBank.load(bank_path)
    picks = picks_per_skill(bank, args.per_skill)
    plan = rollout.EpisodePlan(
        "train",
        tuple(int(s) for s in bank.seeds[picks]),
        tuple(str(t) for t in bank.templates[picks]),
    )
    env = rollout.make_env(
        args.model, plan, asset_root=args.asset_root, cue=config.cue, velocities=config.velocities
    )
    started = time.monotonic()
    row: dict[str, Any] = {
        "run": str(args.run),
        "kind": args.kind,
        "seed": args.seed,
        "checkpoint": str(checkpoint) if checkpoint else "final",
        "test": args.test,
        "episodes": len(picks),
    }
    if args.test == "gain":
        for gain in (1.0, 1.5, 2.0):
            result = run(env, bank, picks, policy, gain=gain)
            row[f"gain_{gain}"] = {"skills": per_skill(result), "magnitudes": magnitudes(result)}
            print(f"gain {gain}: {json.dumps(row[f'gain_{gain}']['skills'])}", flush=True)
    elif args.test == "labels":
        learner = run(env, bank, picks, policy)
        teacher = run(env, bank, picks, None)
        row["learner_trajectories"] = smoothness(learner["labels"], learner["mask"])
        row["teacher_trajectories"] = smoothness(teacher["labels"], teacher["mask"])
        row["teacher_success"] = per_skill(teacher)
        row["teacher_label_phases"] = phase_table(env, bank, picks, None)
        row["learner_success"] = per_skill(learner)
        row["learner_magnitudes"] = magnitudes(learner)
        # The same controller watching the teacher drive (its training distribution): its
        # commands against the labels on the teacher's own states.
        watching = run(env, bank, picks, policy, teacher_steps=10_000)
        watching["mask"] = watching["labels"].any(2)
        row["watching_magnitudes"] = magnitudes(watching)
        row["label_phases"] = phase_table(env, bank, picks, policy)
        print(json.dumps({k: row[k] for k in row if k.endswith("trajectories")}), flush=True)
    elif args.test == "stuck":
        row.update(stuck(env, bank, picks, policy))
        print(json.dumps(row), flush=True)
    else:
        for name, reset in (("state_kept", False), ("state_reset", True)):
            result = run(
                env, bank, picks, policy, teacher_steps=args.burn_in, reset_at_handover=reset
            )
            row[f"teacher_{args.burn_in}_{name}"] = per_skill(result)
            print(f"{name}: {json.dumps(per_skill(result))}", flush=True)
        row["no_burn_in"] = per_skill(run(env, bank, picks, policy))
        print(f"no burn-in: {json.dumps(row['no_burn_in'])}", flush=True)
    row["seconds"] = round(time.monotonic() - started, 1)
    rows = json.loads(args.output.read_text()) if args.output.is_file() else []
    rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
