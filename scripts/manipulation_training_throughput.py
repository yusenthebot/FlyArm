"""Measured costs of training the connectome controller on the manipulation benchmark.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/manipulation_training_throughput.py

1. Closed-loop rollouts: the calibrated connectome policy (encoder, frozen MaleCNS, decoder)
   driving N batched environments of the train split, in environment steps per second, next
   to the environment alone under random actions and the connectome alone.
2. Imitation: one behavior-cloning epoch with the encoder training (the expensive phase) on
   teacher demonstrations, at the batch size and BPTT window of
   configs/whole-brain-manipulation.json, in seconds per demonstration step.
3. Evaluation: every split of the imitation config, all in lockstep, in seconds per control
   step of the longest episode (a policy that fails runs every episode to its horizon).

The numbers feed the wall-clock estimates in docs/MANIPULATION_ENV.md and are written to
docs/results/manipulation-training-throughput.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flyarm.config import ManipulationImitationConfig
from flyarm.manipulation import rollout
from flyarm.manipulation.env import DEFAULT_ASSET_ROOT, BatchedManipulation
from flyarm.manipulation.imitation import PolicyActor, Workbench, evaluation_plans
from flyarm.manipulation.sim import OBS_DIM
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.interface import annotation_interface
from flyarm.whole_brain.policy import BrainPolicy
from flyarm.whole_brain.training import Budget, train_sequence_policy

MEASURE_OFFSET = 900_000  # train-split seeds that no run trains, selects or tests on


def rollout_rate(policy: BrainPolicy, model: Path, assets: Path, envs: int, steps: int) -> dict:
    env = BatchedManipulation(
        model, envs, split="train", asset_root=assets, first_seed=MEASURE_OFFSET
    )
    generator = np.random.default_rng(0)
    env.reset()
    started = time.perf_counter()
    for _ in range(steps):
        env.step(generator.uniform(-1, 1, (envs, 5)))
    random_rate = envs * steps / (time.perf_counter() - started)
    actor = PolicyActor(policy, envs)
    obs = env.reset()
    actor.act(obs)
    started = time.perf_counter()
    brain_seconds = 0.0
    for _ in range(steps):
        tick = time.perf_counter()
        action = actor.act(obs)
        brain_seconds += time.perf_counter() - tick
        obs = env.step(np.clip(action, -1, 1)).obs
    total = time.perf_counter() - started
    return {
        "num_envs": envs,
        "steps": steps,
        "environment_random_actions_steps_per_second": round(random_rate, 1),
        "connectome_closed_loop_steps_per_second": round(envs * steps / total, 1),
        "connectome_ms_per_control_step": round(1000 * brain_seconds / steps, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/whole-brain-manipulation.json")
    )
    parser.add_argument("--demonstrations-per-template", type=int, default=3)
    parser.add_argument("--envs", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/manipulation-training-throughput.json")
    )
    arguments = parser.parse_args()
    config = ManipulationImitationConfig.model_validate_json(arguments.config.read_text())

    pack = ConnectomePack.load(arguments.pack)
    pack.validate_b1a_provenance()
    policy = BrainPolicy(
        "connectome",
        RateDynamics(pack, annotation_interface(pack)),
        obs_dim=OBS_DIM,
        action_dim=5,
        neural_steps=config.neural_steps,
    )
    bench = Workbench(arguments.model, arguments.asset_root, config.cue)
    started = time.perf_counter()
    demonstrations = rollout.plan("train", arguments.demonstrations_per_template, MEASURE_OFFSET)
    data, summary = bench.collect(demonstrations)
    collect_seconds = time.perf_counter() - started
    samples = data["obs"][data["mask"] > 0]
    policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
    policy.calibrate_readout(data["obs"], data["mask"], unit_norm=True)
    record: dict[str, Any] = {
        "machine": platform.platform(),
        "device": str(mx.default_device()),
        "threads": int(os.environ.get("FLYARM_SIM_THREADS", "0")),
        "teacher_collection": {
            "episodes": summary["episodes"],
            "steps": int(data["mask"].sum()),
            "seconds": round(collect_seconds, 1),
        },
        "rollouts": [
            rollout_rate(policy, arguments.model, arguments.asset_root, envs, arguments.steps)
            for envs in arguments.envs
        ],
    }
    print(json.dumps(record["rollouts"], indent=1), flush=True)

    weights = rollout.skill_weights(data, config.max_skill_weight)
    budget = Budget(
        epochs=1,
        decoder_warmup_epochs=0,
        batch_size=config.batch_size,
        bptt_steps=config.bptt_steps,
        learning_rate=config.learning_rate,
        deadline=time.monotonic() + 36_000,
        loss=config.loss,
        input_learning_rate=config.learning_rate * config.encoder_learning_rate_scale,
    )
    _, fit = train_sequence_policy(
        policy,
        data,
        weights,
        data,
        weights,
        budget,
        0,
        set_normalization=False,
        length_buckets=config.length_buckets,
    )
    steps = int(data["mask"].sum())
    record["imitation_epoch"] = {
        "episodes": int(data["obs"].shape[0]),
        "demonstration_steps": steps,
        "batch_size": config.batch_size,
        "bptt_steps": config.bptt_steps,
        "seconds_including_validation_loss": round(fit["training_seconds"], 1),
        "ms_per_demonstration_step": round(1000 * fit["training_seconds"] / steps, 3),
    }
    print(json.dumps(record["imitation_epoch"], indent=1), flush=True)

    plans = evaluation_plans(config, 1)
    started = time.perf_counter()
    logs = bench.run(plans, lambda n: PolicyActor(policy, n))
    seconds = time.perf_counter() - started
    longest = max(int(log.steps.max()) for log in logs)
    record["evaluation_one_per_template"] = {
        "episodes": sum(len(p) for p in plans),
        "longest_episode_steps": longest,
        "seconds": round(seconds, 1),
        "ms_per_lockstep_step": round(1000 * seconds / longest, 2),
    }
    print(json.dumps(record["evaluation_one_per_template"], indent=1), flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
