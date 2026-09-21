"""Kitchen protocol sweep for the cheap controllers (MLP, GRU): loss, action chunk, epochs,
checkpoint selection and tracker DART data, each evaluated on the benchmark's clean episodes.

This is how the B2 protocol was chosen before any fly run; the fly controllers are never part
of the sweep. Usage:

    PYTHONPATH=src uv run python scripts/kitchen_protocol_sweep.py GRID.json OUTPUT.json

GRID.json is a list of settings, each with kind (mlp|gru), chunk, loss (mse|l1), epochs, seed
and optionally split, hidden (GRU width, default 160), select_every and select_episodes
(closed-loop selection), dart_episodes and dart_noise. Results are appended to OUTPUT.json,
so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from flyarm.benchmarks import kitchen
from flyarm.benchmarks.kitchen_expert import DemonstrationTracker, noisy_teacher_episodes
from flyarm.flyleg.experiment import DART_SEED, SELECTION_SEED, VALIDATION_SEED, split_episodes
from flyarm.whole_brain.policy import GRUPolicy, MLPPolicy, MlxController, SequencePolicy
from flyarm.whole_brain.training import Budget, train_sequence_policy

EVAL_EPISODES = 40


def _pad(data: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    extra = horizon - data["obs"].shape[1]
    return {
        key: np.pad(data[key], [(0, 0), (0, extra)] + [(0, 0)] * (data[key].ndim - 2))
        for key in ("obs", "actions", "mask")
    }


def run(setting: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    split = setting.get("split", "complete")
    if split not in cache:
        data = kitchen.load(split)
        train_rows, validation_rows = split_episodes(len(data.episode_ids), 0.1, VALIDATION_SEED)
        cache[split] = (data, data.subset(train_rows), data.subset(validation_rows))
    data, train, validation = cache[split]
    env = kitchen.recover_env(split)
    try:
        if setting.get("dart_episodes"):
            tracker = DemonstrationTracker.from_data(data)
            dart, _ = noisy_teacher_episodes(
                tracker, env, setting["dart_episodes"], setting.get("dart_noise", 0.1), DART_SEED
            )
            padded = _pad(train, dart["obs"].shape[1])
            train = {key: np.concatenate((padded[key], dart[key])) for key in padded}
        dims = {"obs_dim": kitchen.FEATURE_DIM, "action_dim": kitchen.ACTION_DIM}
        policy: SequencePolicy
        if setting["kind"] == "mlp":
            policy = MLPPolicy(**dims, seed=setting["seed"], chunk=setting["chunk"])
        else:
            hidden = setting.get("hidden", 160)
            policy = GRUPolicy(**dims, hidden=hidden, seed=setting["seed"], chunk=setting["chunk"])
        episodes = list(range(SELECTION_SEED, SELECTION_SEED + setting.get("select_episodes", 5)))

        def closed_loop(candidate: SequencePolicy) -> float:
            controller = kitchen.PositionFeatures(MlxController(candidate))
            return float(kitchen.evaluate(env, controller, episodes)["mean_tasks"])

        started = time.monotonic()
        _, summary = train_sequence_policy(
            policy,
            train,
            train["mask"],
            validation,
            validation["mask"],
            Budget(
                epochs=setting["epochs"],
                decoder_warmup_epochs=0,
                batch_size=4,
                bptt_steps=8,
                learning_rate=1e-3,
                deadline=started + 36_000,
                loss=setting["loss"],
            ),
            setting["seed"],
            selector=closed_loop if setting.get("select_every") else None,
            select_every=setting.get("select_every", 1),
        )
        result = kitchen.evaluate(
            env, kitchen.PositionFeatures(MlxController(policy)), list(range(EVAL_EPISODES))
        )
    finally:
        env.close()
    return {
        **setting,
        "score": result["normalized_score"],
        "per_task": result["per_task_success"],
        "best_epoch": summary["best_epoch"],
        "selection_score": summary["selection_score"],
        "training_seconds": summary["training_seconds"],
    }


def main(grid_path: Path, output: Path) -> None:
    grid = json.loads(grid_path.read_text())
    results = json.loads(output.read_text()) if output.exists() else []
    cache: dict[str, Any] = {}
    for setting in grid[len(results) :]:
        results.append(run(setting, cache))
        output.write_text(json.dumps(results, indent=1))
        print(json.dumps({k: results[-1][k] for k in ("kind", "chunk", "loss", "seed", "score")}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
