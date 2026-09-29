"""Why the 1,382-neuron kitchen readout does not train: decoder-only fits on frozen features.

Reproduces the decoder warm-up of a kitchen fly run (untrained seed-0 encoder, frozen
connectome, tanh(linear) decoder, L1 on 10-step action chunks, Adam at the run's learning
rate) on every demonstration step of the split, for two readouts (68 left front-leg motor
neurons; the same plus 1,314 descending neurons) and two frozen calibrations:

- standardize: per-neuron mean and standard deviation (runs E22);
- whiten: PCA of the same activity, components with at least WHITEN_FLOOR of the top
  variance kept and scaled to unit variance (a fixed linear map before the trainable decoder,
  so the model class is unchanged; only the optimization geometry differs).

Usage:

    PYTHONPATH=src uv run python scripts/readout_conditioning.py \
        docs/results/readout-conditioning.json
"""

from __future__ import annotations

import json
import sys
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
from flyarm.whole_brain.training import chunk_targets

PACK = Path("data/whole_brain/malecns-v1.0-c3")
ANNOTATIONS = Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
CONFIG = Path("configs/flyleg-kitchen-descending-std-dev.json")
STEPS, BATCH, WHITEN_FLOOR, SEED = 3000, 256, 1e-6, 0


def features(policy, obs: np.ndarray) -> np.ndarray:
    """Pooled readout activity [episodes, steps, outputs] of the untrained policy."""
    state = policy.initial_state(obs.shape[0])
    rows = []
    for step in range(obs.shape[1]):
        current = policy.encode(policy.normalize(mx.array(obs[:, step])))
        state, pooled = policy.dynamics.advance(state, current, policy.neural_steps)
        mx.eval(state, pooled)
        rows.append(np.asarray(pooled, dtype=np.float64))
    return np.stack(rows, axis=1)


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)


def whiten(x: np.ndarray) -> tuple[np.ndarray, int]:
    centered = x - x.mean(0)
    values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    keep = values >= WHITEN_FLOOR * values.max()
    projection = vectors[:, keep] / np.sqrt(values[keep])
    return centered @ projection, int(keep.sum())


def fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, learning_rate: float) -> list[float]:
    """Decoder-only Adam on L1; returns the weighted L1 over all samples every 250 steps."""
    mx.random.seed(SEED)
    decoder = nn.Linear(x.shape[1], y.shape[1])
    optimizer = optim.Adam(learning_rate=learning_rate)
    xs, ys, ws = mx.array(x, dtype=mx.float32), mx.array(y), mx.array(w)

    def loss(model, xb, yb, wb):
        return (mx.abs(mx.tanh(model(xb)) - yb) * wb).sum() / mx.maximum(wb.sum(), 1.0)

    step_fn = nn.value_and_grad(decoder, loss)
    generator = np.random.default_rng(SEED)
    curve = []
    for step in range(STEPS + 1):
        if step % 250 == 0:
            curve.append(float(loss(decoder, xs, ys, ws)))
        rows = mx.array(generator.integers(0, x.shape[0], BATCH))
        _, grads = step_fn(decoder, xs[rows], ys[rows], ws[rows])
        optimizer.update(decoder, grads)
        mx.eval(decoder.parameters(), optimizer.state)
    return curve


def main(output: Path) -> None:
    config = FlyLegConfig.model_validate_json(CONFIG.read_text())
    pack = ConnectomePack.load(PACK)
    pack.validate_b1a_provenance()
    data = kitchen.load(config.split)
    train = data.subset(np.arange(data.obs.shape[0]))
    valid = train["mask"].astype(bool)
    targets, weights = chunk_targets(
        train["actions"], train["mask"], train["mask"].astype(np.float32), config.action_chunk
    )
    y = targets[valid].reshape(int(valid.sum()), -1)
    w = np.repeat(weights[valid], kitchen.ACTION_DIM, axis=1)
    result = {"config": str(CONFIG), "samples": int(valid.sum()), "steps": STEPS, "readouts": {}}
    for name, descending in (("68 motor", False), ("68 motor + 1,314 descending", True)):
        leg = front_leg_interface(pack, ANNOTATIONS, include_descending=descending)
        dynamics = leg_dynamics(config, pack, leg.interface)
        policy = make_policy(
            "flyleg", config, SEED, dynamics=dynamics, channels=leg_channels(leg), fly_budget=0
        )
        samples = train["obs"][valid]
        policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
        raw = features(policy, train["obs"])[valid]
        white, kept = whiten(raw)
        row = {
            "outputs": int(raw.shape[1]),
            "whitened_components": kept,
            "standardize_l1": fit(standardize(raw), y, w, config.learning_rate),
            "whiten_l1": fit(white, y, w, config.learning_rate),
        }
        result["readouts"][name] = row
        print(name, json.dumps(row), flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
