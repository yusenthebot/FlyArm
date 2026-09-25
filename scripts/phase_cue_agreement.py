"""Does the observable phase (flyarm.manipulation.phases) agree with the teacher's own phase?

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/phase_cue_agreement.py \\
        --per-template 3 --output docs/results/phase-cue-agreement.json

The teacher runs whole templates of the train split from true starts; at every step the phase
the teacher decides from the state (its phase after acting, which it acts in from the next step)
is compared with the observable phase of the same state; the phase it acted in is reported too.
The teacher's FINISHED (a finished motion waiting for the scene) counts as ``retreat``.
Reports the agreement overall, per skill and per teacher phase, and the confusion counts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from flyarm.manipulation import phases as ph
from flyarm.manipulation import rollout
from flyarm.manipulation import teacher as mt


def teacher_as_observable(phase: int) -> int:
    name = mt.PHASES[phase]
    return ph.PHASE_NAMES.index("retreat" if name == "finished" else name)


def measure(model: Path, split: str, per_template: int, offset: int) -> dict:
    plan = rollout.plan(split, per_template, offset)
    env = rollout.make_env(model, plan, velocities=False)
    env.reset(seeds=np.array(plan.seeds), templates=list(plan.templates))
    teacher = mt.ManipulationTeacher(env)
    teacher.reset()
    acting = np.zeros(env.num_envs, dtype=np.int64)
    original = teacher._act_row

    def record(row: int, scene: dict) -> np.ndarray:
        acting[row] = teacher.phase[row]
        return original(row, scene)

    teacher._act_row = record  # type: ignore[method-assign]
    active = np.ones(env.num_envs, bool)
    pairs: list[tuple[int, int, str]] = []
    for _ in range(int(env.horizons.max())):
        if not active.any():
            break
        state = env.scene_state()
        leading = env.leading(env.subgoal_done(env.effects(state)))
        observed = ph.observable_phases(env, leading)
        action = teacher.act().astype(np.float64)
        decided = teacher.phase.copy()  # the phase the teacher judged this state to be in
        for row in np.flatnonzero(active & (observed >= 0)):
            skill = mt.tk.SKILLS[int(env.sub_kind[row, env.current(leading)[row]])]
            pairs.append(
                (
                    teacher_as_observable(int(decided[row])),
                    int(observed[row]),
                    skill,
                    teacher_as_observable(int(acting[row])),
                )
            )
        result = env.step(action, auto_reset=False)
        active &= ~(result.success | result.truncated)
    truth = np.array([p[0] for p in pairs])
    guess = np.array([p[1] for p in pairs])
    per_phase = {}
    for k, name in enumerate(ph.PHASE_NAMES):
        m = truth == k
        if m.any():
            per_phase[name] = {
                "steps": int(m.sum()),
                "agreement": round(float((guess[m] == k).mean()), 3),
                "observed_as": {
                    ph.PHASE_NAMES[g]: int(c) for g, c in Counter(guess[m]).most_common(4)
                },
            }
    skills = np.array([p[2] for p in pairs])
    acted = np.array([p[3] for p in pairs])
    return {
        "split": split,
        "episodes": len(plan),
        "steps": int(len(truth)),
        # The teacher decides a transition from the state at step t and acts in the new phase
        # from t + 1, so the phase it acts in lags the state's by one step at every switch;
        # the phase it decided from the state is the like-for-like comparison.
        "agreement": round(float((truth == guess).mean()), 4),
        "agreement_with_acting_phase": round(float((acted == guess).mean()), 4),
        "per_teacher_phase": per_phase,
        "per_skill": {
            s: round(float((truth[skills == s] == guess[skills == s]).mean()), 3)
            for s in np.unique(skills)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-template", type=int, default=3)
    parser.add_argument("--split", default="train")
    parser.add_argument("--offset", type=int, default=rollout.VALIDATION_OFFSET)
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    row = measure(args.model, args.split, args.per_template, args.offset)
    print(json.dumps(row, indent=1))
    if args.output is not None:
        args.output.write_text(json.dumps(row, indent=1) + "\n")


if __name__ == "__main__":
    main()
