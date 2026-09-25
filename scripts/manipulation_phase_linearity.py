"""How linear is the teacher within a phase? Global vs per-phase vs per-skill linear fits."""

import json
import sys
from pathlib import Path

import numpy as np

from flyarm.manipulation import rollout
from flyarm.manipulation.phases import PHASE_NAMES, observable_phases
from flyarm.manipulation.teacher import PHASES, ManipulationTeacher

MODEL = Path("assets/menagerie/franka_emika_panda/scene.xml")
per_template = int(sys.argv[1]) if len(sys.argv) > 1 else 3
data = {"obs": [], "act": [], "phase": [], "observed": [], "skill": [], "episode": []}
for offset, tag in ((0, "fit"), (500, "held")):
    p = rollout.plan("train", per_template, offset=offset)
    env = rollout.make_env(MODEL, p, velocities=False)
    obs = env.reset(seeds=np.array(p.seeds), templates=list(p.templates))
    teacher = ManipulationTeacher(env)
    teacher.reset()
    active = np.ones(env.num_envs, bool)
    for _ in range(int(env.horizons.max())):
        state = env.scene_state()
        observed = observable_phases(env, env.leading(env.subgoal_done(env.effects(state))))
        a = teacher.act().astype(np.float64)
        ph = teacher.phase.copy()
        sk = env.current_skill()
        for r in np.flatnonzero(active):
            data["obs"].append(obs[r])
            data["act"].append(a[r])
            data["phase"].append(ph[r])
            data["observed"].append(observed[r])
            data["skill"].append(sk[r])
            data["episode"].append((tag, r))
        res = env.step(a, auto_reset=False)
        active &= ~(res.success | res.truncated)
        obs = res.obs
        if not active.any():
            break
X = np.array(data["obs"], np.float64)
Y = np.clip(np.array(data["act"]), -1, 1)
phase = np.array(data["phase"])
observed = np.array(data["observed"])
skill = np.array(data["skill"])
held = np.array([e[0] == "held" for e in data["episode"]])
mu, sd = X[~held].mean(0), X[~held].std(0)
keep = sd > 1e-3
Xn = np.clip((X[:, keep] - mu[keep]) / sd[keep], -5, 5)


def fit_eval(groups):
    pred = np.zeros_like(Y)
    for g in np.unique(groups):
        tr = (~held) & (groups == g)
        te = groups == g
        if tr.sum() < 20:
            pred[te] = Y[tr].mean(0) if tr.any() else 0
            continue
        A = np.c_[Xn[tr], np.ones(tr.sum())]
        W = np.linalg.lstsq(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ Y[tr], rcond=None)[0]
        pred[te] = np.clip(np.c_[Xn[te], np.ones(te.sum())] @ W, -1, 1)
    err = np.abs(pred - Y).mean(1)
    return round(float(err[~held].mean()), 3), round(float(err[held].mean()), 3)


out = {
    "steps": int(len(Y)),
    "held_steps": int(held.sum()),
    "global linear": fit_eval(np.zeros(len(Y), int)),
    "linear per skill": fit_eval(skill),
    "linear per teacher phase": fit_eval(phase),
    "linear per skill x phase": fit_eval(skill * 100 + phase),
    # The observable phase (flyarm.manipulation.phases), what the phase cue gives a controller.
    "linear per observable phase": fit_eval(observed),
    "linear per skill x observable phase": fit_eval(skill * 100 + observed),
    "observable phase agreement": round(
        float(
            np.mean(
                [
                    PHASE_NAMES.index("retreat" if PHASES[p] == "finished" else PHASES[p]) == o
                    for p, o in zip(phase, observed, strict=True)
                ]
            )
        ),
        3,
    ),
    "phase shares": {PHASES[p]: round(float((phase == p).mean()), 3) for p in np.unique(phase)},
}
per_phase = {}
for p in np.unique(phase):
    m = phase == p
    per_phase[PHASES[p]] = {
        "share": round(float(m.mean()), 3),
        "saturated": round(float((np.abs(Y[m]) > 0.95).mean()), 3),
    }
out["per phase"] = per_phase
print(json.dumps(out, indent=1))
Path("docs/results/manipulation-phase-linearity.json").write_text(json.dumps(out, indent=1))
