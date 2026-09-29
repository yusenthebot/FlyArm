"""Is the kitchen fly controller's fit that of a linear policy? A no-connectome reference.

Fits, on the train/validation split of a kitchen run, the 10-step action-chunk targets from
the 30 normalized position features with (a) a tanh(linear) policy and (b) a two-layer MLP
of width 256, both with L1 and Adam at 1e-3 on shuffled mini-batches, and reports the
validation L1 next to the runs' own best validation L1 of every controller.

Usage:

    PYTHONPATH=src uv run python scripts/linear_policy_probe.py docs/results/linear-policy.json
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
from flyarm.whole_brain.training import chunk_targets

RUN = Path("runs/flyleg-kitchen-complete-chunk-001")
REFERENCES = {
    "68 motor, unit_norm": Path("runs/flyleg-kitchen-leg-unitnorm-dev-001"),
    "1,382 neurons, unit_norm": Path("runs/flyleg-kitchen-descending-unitnorm-dev-001"),
}
STEPS, BATCH, SEED = 20000, 256, 0


def samples(data, episodes: list[int], chunk: int, mean=None, scale=None):
    index = {episode: row for row, episode in enumerate(data.episode_ids.tolist())}
    subset = data.subset(np.array([index[e] for e in episodes]))
    valid = subset["mask"].astype(bool)
    targets, weights = chunk_targets(
        subset["actions"], subset["mask"], subset["mask"].astype(np.float32), chunk
    )
    x = subset["obs"][valid]
    if mean is None:
        mean, scale = x.mean(0), np.maximum(x.std(0), 0.05)
    x = (x - mean) / scale
    y = targets[valid].reshape(int(valid.sum()), -1)
    w = np.repeat(weights[valid], kitchen.ACTION_DIM, axis=1)
    return x.astype(np.float32), y, w, mean, scale


def fit(model: nn.Module, train, validation) -> dict:
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss(net, x, y, w):
        return (mx.abs(mx.tanh(net(x)) - y) * w).sum() / mx.maximum(w.sum(), 1.0)

    step = nn.value_and_grad(model, loss)
    tx, ty, tw = (mx.array(a) for a in train)
    vx, vy, vw = (mx.array(a) for a in validation)
    generator = np.random.default_rng(SEED)
    best = float("inf")
    for index in range(STEPS):
        rows = mx.array(generator.integers(0, tx.shape[0], BATCH))
        _, grads = step(model, tx[rows], ty[rows], tw[rows])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if index % 500 == 499:
            best = min(best, float(loss(model, vx, vy, vw)))
    return {"train_l1": float(loss(model, tx, ty, tw)), "best_validation_l1": best}


def main(output: Path) -> None:
    config = FlyLegConfig.model_validate_json((RUN / "config.json").read_text())
    splits = json.loads((RUN / "splits.json").read_text())
    data = kitchen.load(config.split)
    x, y, w, mean, scale = samples(data, splits["train_episode_ids"], config.action_chunk)
    validation = samples(data, splits["validation_episode_ids"], config.action_chunk, mean, scale)
    mx.random.seed(SEED)
    width = y.shape[1]
    models = {
        "linear (tanh of an affine map of the 30 features)": nn.Linear(x.shape[1], width),
        "MLP 2 x 256": nn.Sequential(
            nn.Linear(x.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, width),
        ),
    }
    result = {"split_from": str(RUN), "steps": STEPS, "fits": {}, "run_references": {}}
    for name, model in models.items():
        result["fits"][name] = fit(model, (x, y, w), validation[:3])
        print(name, result["fits"][name], flush=True)
    runs = json.loads((RUN / "results.json").read_text())["models"]
    for model in runs:
        result["run_references"][f"{model['kind']} seed {model['seed']}"] = model.get(
            "best_validation_loss"
        )
    for name, run in REFERENCES.items():
        model = json.loads((run / "results.json").read_text())["models"][0]
        result["run_references"][name] = model["best_validation_loss"]
    print(json.dumps(result["run_references"], indent=1))
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
