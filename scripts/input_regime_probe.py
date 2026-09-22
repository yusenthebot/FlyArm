"""Do the kitchen fly controller's sensory input neurons operate in their nonlinear range?

For trained kitchen fly checkpoints, runs the demonstration episodes of the training split
through the policy and reports, for the input currents I = encoder(normalized observation)
of the proprioceptive and head-sensory channels: standard deviation, share with |I| > 1
(where tanh departs from linear by more than 20%), and the input neurons' own states.
The untrained policy (same seed) is the reference.

Usage:

    PYTHONPATH=src uv run python scripts/input_regime_probe.py docs/results/input-regime.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from flyarm.benchmarks import kitchen
from flyarm.config import FlyLegConfig
from flyarm.flyleg.experiment import leg_dynamics, make_policy
from flyarm.flyleg.interface import front_leg_interface, leg_channels
from flyarm.flyleg.record import load_flyleg_policy
from flyarm.whole_brain.compiler import ConnectomePack

PACK = Path("data/whole_brain/malecns-v1.0-c3")
ANNOTATIONS = Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
RUNS = (
    Path("runs/flyleg-kitchen-complete-chunk-001"),
    Path("runs/flyleg-kitchen-leg-unitnorm-dev-001"),
)


def currents(policy, obs: np.ndarray, mask: np.ndarray) -> dict:
    split = policy.channels[0][2]
    state = policy.initial_state(obs.shape[0])
    inputs = np.asarray(policy.dynamics.input_indices)
    drive, own = [], []
    for step in range(obs.shape[1]):
        current = policy.encode(policy.normalize(mx.array(obs[:, step])))
        state, _ = policy.dynamics.advance(state, current, policy.neural_steps)
        mx.eval(state, current)
        keep = mask[:, step] > 0
        drive.append(np.asarray(current)[keep])
        own.append(np.asarray(state)[inputs][:, keep].T)
    drive_all, own_all = np.concatenate(drive), np.concatenate(own)

    def stats(block: np.ndarray, states: np.ndarray) -> dict:
        return {
            "current_std": float(block.std()),
            "share_abs_current_above_1": float((np.abs(block) > 1).mean()),
            "state_mean_abs": float(np.abs(states).mean()),
            "share_abs_state_above_0.5": float((np.abs(states) > 0.5).mean()),
        }

    return {
        "proprioceptive": stats(drive_all[:, :split], own_all[:, :split]),
        "head_sensory": stats(drive_all[:, split:], own_all[:, split:]),
    }


def main(output: Path) -> None:
    result = {}
    pack = ConnectomePack.load(PACK)
    pack.validate_b1a_provenance()
    for run in RUNS:
        config = FlyLegConfig.model_validate_json((run / "config.json").read_text())
        splits = json.loads((run / "splits.json").read_text())
        data = kitchen.load(config.split)
        index = {episode: row for row, episode in enumerate(data.episode_ids.tolist())}
        train = data.subset(np.array([index[e] for e in splits["train_episode_ids"]]))
        trained = load_flyleg_policy(run, "flyleg", 0, PACK, ANNOTATIONS)
        leg = front_leg_interface(pack, ANNOTATIONS)
        untrained = make_policy(
            "flyleg",
            config,
            0,
            dynamics=leg_dynamics(config, pack, leg.interface),
            channels=leg_channels(leg),
            fly_budget=0,
        )
        samples = train["obs"][train["mask"].astype(bool)]
        untrained.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
        result[str(run)] = {
            "trained_seed_0": currents(trained, train["obs"], train["mask"]),
            "untrained_seed_0": currents(untrained, train["obs"], train["mask"]),
        }
        print(run, json.dumps(result[str(run)], indent=1), flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
