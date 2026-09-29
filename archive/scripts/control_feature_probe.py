"""Skill-level DAgger with a frame-wise MLP, with and without control-scale features (H-E).

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/control_feature_probe.py \\
        --banks runs/skill-dagger-connectome-001 --variant fine --rounds 6

Isolates the input representation from the sequence trainer: a memoryless MLP (two ReLU layers
of 674, tanh output, about the matched budget) is trained frame by frame (1,024 random steps per
update, L1 with the per-skill weights, about 10 passes over the aggregate a round, Adam 1e-3,
warm-started) inside the skill-DAgger loop of flyarm.manipulation.skill_dagger: round 0 the
teacher, round 1 beta 0.5, then beta 0; single-subgoal episodes from the banks of ``--banks``
(256 a round, 10% true starts). Each round prints the mean single-subgoal success per skill
over the validation bank (12 episodes per skill, 8 for pick). ``--variant fine`` appends
flyarm.manipulation.features.control_expansion() to the normalized observation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from flyarm.config import DaggerStage
from flyarm.manipulation import curriculum as cu
from flyarm.manipulation import rollout
from flyarm.manipulation import skill_dagger as sd
from flyarm.manipulation.features import control_expansion
from flyarm.manipulation.imitation import Workbench

sys.path.insert(0, str(Path(__file__).parent))
import skill_dagger_diagnostics as dg  # noqa: E402

HIDDEN = 674
BATCH = 1024
BETAS = (1.0, 0.5)


class FrameNet:
    def __init__(self, fine: bool) -> None:
        self.index, self.scale = (np.asarray(v) for v in control_expansion())
        self.fine = fine
        self.model: nn.Module | None = None
        self.mean = self.std = np.zeros(0)

    def features(self, obs: np.ndarray) -> np.ndarray:
        if not self.fine:
            return obs
        return np.concatenate([obs, np.tanh(obs[:, self.index] / self.scale)], 1)

    def inputs(self, obs: np.ndarray) -> np.ndarray:
        return ((self.features(obs) - self.mean) / self.std).astype(np.float32)

    def fit_normalization(self, obs: np.ndarray) -> None:
        x = self.features(obs)
        self.mean, self.std = x.mean(0), np.maximum(x.std(0), 0.05)
        self.mean[obs.shape[1] :], self.std[obs.shape[1] :] = 0.0, 1.0  # already at scale
        mx.random.seed(0)
        self.model = nn.Sequential(
            nn.Linear(x.shape[1], HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 5),
            nn.Tanh(),
        )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model(mx.array(self.inputs(obs))), dtype=np.float64)


class Actor:
    def __init__(self, net: FrameNet) -> None:
        self.net = net

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.net(obs)


class Sequence:
    """The SequencePolicy interface the diagnostics' rollout expects."""

    def __init__(self, net: FrameNet) -> None:
        self.net = net

    def initial_state(self, n: int) -> mx.array:
        return mx.zeros((n, 1))

    def step(self, obs: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        return mx.array(self.net(np.asarray(obs)).astype(np.float32)), state


def train(net: FrameNet, data: sd.StepData, updates: int, seed: int) -> float:
    assert net.model is not None
    x = net.inputs(data.obs)

    def loss(model: nn.Module, a: mx.array, b: mx.array, w: mx.array) -> mx.array:
        return mx.sum(mx.abs(model(a) - b).mean(1) * w) / mx.sum(w)

    grad = nn.value_and_grad(net.model, loss)
    optimizer = optim.Adam(learning_rate=1e-3)
    generator = np.random.default_rng(seed)
    losses = []
    for _ in range(updates):
        rows = generator.integers(0, len(x), BATCH)
        value, gradients = grad(
            net.model,
            mx.array(x[rows]),
            mx.array(data.actions[rows]),
            mx.array(data.weights[rows]),
        )
        optimizer.update(net.model, gradients)
        mx.eval(net.model.parameters(), optimizer.state)
        losses.append(float(value))
    return float(np.mean(losses[-300:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks", type=Path, required=True, help="a run with bank.npz files")
    parser.add_argument("--variant", choices=("plain", "fine"), required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    args = parser.parse_args()
    bank = cu.SubgoalBank.load(args.banks / "bank.npz")
    validation = cu.SubgoalBank.load(args.banks / "validation-bank.npz")
    bench = Workbench(args.model, args.asset_root, cue=True, velocities=False)
    picks = dg.picks_per_skill(validation, 12)
    plan = rollout.EpisodePlan(
        "train",
        tuple(int(s) for s in validation.seeds[picks]),
        tuple(str(t) for t in validation.templates[picks]),
    )
    eval_env = rollout.make_env(
        args.model, plan, asset_root=args.asset_root, cue=True, velocities=False
    )
    stage = DaggerStage(name="single_subgoal", rounds=args.rounds, true_start_share=0.1)
    net = FrameNet(args.variant == "fine")
    parts: list[dict[str, np.ndarray]] = []
    for index in range(args.rounds):
        started = time.monotonic()
        beta = BETAS[index] if index < len(BETAS) else 0.0
        generator = np.random.default_rng([0, index, 13])
        plans = sd.round_plans(stage, index, args.episodes, bank, generator)
        actor = None if index == 0 else Actor(net)
        logs = bench.run(
            plans,
            lambda n, actor=actor: actor,
            record=True,
            beta=beta if index else 0.0,
            generator=generator,
        )
        parts.append(sd.ragged(logs))
        data = sd.aggregate(parts, 5.0)
        if index == 0:
            net.fit_normalization(data.obs)
        updates = min(15_000, max(3000, int(10 * len(data.obs) / BATCH)))
        loss = train(net, data, updates, index)
        skills = dg.per_skill(dg.run(eval_env, validation, picks, Sequence(net)))
        print(
            f"{args.variant} round {index} beta {beta}: aggregate {len(data.obs)}, "
            f"updates {updates}, L1 {loss:.3f}, mean skill success {skills['mean_success']} "
            + " ".join(f"{k} {v['success_rate']}" for k, v in skills.items() if k != "mean_success")
            + f" ({time.monotonic() - started:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
