"""The first decoder warm-up updates of a kitchen fly run, per readout calibration.

Replays the start of flyarm.whole_brain.training.train_sequence_policy exactly (seed-0 policy,
frozen encoder, the run's batch of episodes, 8-step BPTT windows, gradient clip 1.0, Adam at
the run's learning rate) for WINDOWS updates and records the loss and the decoder's
pre-activation spread after each update.
Calibrations: the 68-motor readout standardized (reference, E22) and the 1,382-neuron
readout standardized (E22), divided by sqrt(outputs) ("unit_norm") and PCA-whitened.

Usage:

    PYTHONPATH=src uv run python scripts/readout_warmup_probe.py docs/results/readout-warmup.json
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from flyarm.benchmarks import kitchen
from flyarm.config import FlyLegConfig
from flyarm.flyleg.experiment import leg_dynamics, make_policy
from flyarm.flyleg.interface import front_leg_interface, leg_channels
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.training import _chunk_loss, _set_input_frozen, chunk_targets

PACK = Path("data/whole_brain/malecns-v1.0-c3")
ANNOTATIONS = Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
RUN = Path("runs/flyleg-kitchen-descending-std-dev-001")
WINDOWS, WHITEN_FLOOR = 20, 1e-6


def _whiten(policy, obs: np.ndarray, mask: np.ndarray) -> int:
    """Replace the readout by a frozen PCA whitening of the calibration activity."""
    state = policy.initial_state(obs.shape[0])
    rows = []
    for step in range(obs.shape[1]):
        current = policy.encode(policy.normalize(mx.array(obs[:, step])))
        state, pooled = policy.dynamics.advance(state, current, policy.neural_steps)
        mx.eval(state, pooled)
        rows.append(np.asarray(pooled, dtype=np.float64)[mask[:, step] > 0])
    activity = np.concatenate(rows)
    mean = activity.mean(0)
    values, vectors = np.linalg.eigh(np.cov(activity - mean, rowvar=False))
    keep = values >= WHITEN_FLOOR * values.max()
    projection = mx.array((vectors[:, keep] / np.sqrt(values[keep])).astype(np.float32))
    offset = mx.array(mean.astype(np.float32))
    policy.readout = lambda pooled: (pooled - offset) @ projection
    policy.decoder = nn.Linear(int(keep.sum()), policy.output_dim)
    return int(keep.sum())


def replay(pack, config, train, descending: bool, calibration: str) -> dict:
    leg = front_leg_interface(pack, ANNOTATIONS, include_descending=descending)
    policy = make_policy(
        "flyleg",
        config,
        0,
        dynamics=leg_dynamics(config, pack, leg.interface),
        channels=leg_channels(leg),
        fly_budget=0,
    )
    samples = train["obs"][train["mask"].astype(bool)]
    policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
    policy.calibrate_readout(
        train["obs"],
        train["mask"],
        center=calibration == "standardize",
        unit_norm=calibration == "unit_norm",
    )
    components = _whiten(policy, train["obs"], train["mask"]) if calibration == "whiten" else None
    obs = mx.array(train["obs"])
    targets_np, weights_np = chunk_targets(
        train["actions"], train["mask"], train["mask"].astype(np.float32), policy.chunk
    )
    targets, weights = mx.array(targets_np), mx.array(weights_np)
    _set_input_frozen(policy, True)
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    gradient_fn = nn.value_and_grad(policy, partial(_chunk_loss, loss=config.loss))
    batch = mx.array(np.arange(min(config.batch_size, obs.shape[0])))
    state = policy.initial_state(batch.size)
    losses, spreads = [], []
    for window in range(WINDOWS):
        start, stop = window * config.bptt_steps, (window + 1) * config.bptt_steps
        (loss, state), gradients = gradient_fn(
            policy,
            obs[batch][:, start:stop],
            targets[batch][:, start:stop],
            weights[batch][:, start:stop],
            state,
        )
        gradients, _ = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(policy, gradients)
        state = mx.stop_gradient(state)
        mx.eval(policy.parameters(), optimizer.state, state, loss)
        _, pooled = policy.dynamics.advance(
            state, policy.encode(policy.normalize(obs[batch][:, stop])), policy.neural_steps
        )
        spreads.append(float(np.std(np.asarray(policy.decoder(policy.readout(pooled))))))
        losses.append(float(loss))
    return {"components": components, "loss": losses, "pre_activation_std": spreads}


def main(output: Path) -> None:
    config = FlyLegConfig.model_validate_json((RUN / "config.json").read_text())
    splits = json.loads((RUN / "splits.json").read_text())
    pack = ConnectomePack.load(PACK)
    pack.validate_b1a_provenance()
    data = kitchen.load(config.split)
    index = {episode: row for row, episode in enumerate(data.episode_ids.tolist())}
    train = data.subset(np.array([index[e] for e in splits["train_episode_ids"]]))
    cases = {
        "68 motor, standardize": (False, "standardize"),
        "1,382 neurons, standardize": (True, "standardize"),
        "1,382 neurons, unit_norm": (True, "unit_norm"),
        "1,382 neurons, whiten": (True, "whiten"),
    }
    result = {"run": str(RUN), "windows": WINDOWS, "cases": {}}
    for name, (descending, calibration) in cases.items():
        result["cases"][name] = replay(pack, config, train, descending, calibration)
        row = result["cases"][name]
        print(name, [round(v, 3) for v in row["loss"]], flush=True)
        print("   pre-activation std", [round(v, 2) for v in row["pre_activation_std"]], flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
