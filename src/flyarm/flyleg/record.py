"""Rebuild saved B2 checkpoints and record labelled FrankaKitchen rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from flyarm.benchmarks import kitchen
from flyarm.config import FlyLegConfig
from flyarm.experiment import save_json
from flyarm.flyleg.interface import front_leg_interface
from flyarm.interfaces import NeuralInterface
from flyarm.video import annotate, tile, write_video
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.policy import (
    BrainPolicy,
    GRUPolicy,
    MLPPolicy,
    MlxController,
    SequencePolicy,
    gru_hidden_for_budget,
)
from flyarm.whole_brain.shuffle import shuffle_pack

TITLES = {
    "flyleg": "Fly CNS (MaleCNS) as the left front leg",
    "flyleg_shuffled": "Degree-matched shuffled CNS, same leg interface",
    "mlp": "MLP behavior cloning (D4RL BC)",
    "gru": "GRU, parameter-matched",
}


def load_flyleg_policy(
    run_root: Path, kind: str, seed: int, pack_root: Path, annotations: Path
) -> SequencePolicy:
    config = FlyLegConfig.model_validate_json((run_root / "config.json").read_text())
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    leg = front_leg_interface(pack, annotations)
    if NeuralInterface.load(run_root / "interface.json") != leg.interface:
        raise ValueError("Saved interface differs from the one regenerated from annotations")
    channels = [
        (kitchen.PROPRIOCEPTION.start, kitchen.PROPRIOCEPTION.stop, len(leg.proprioceptors)),
        (kitchen.EXTEROCEPTION.start, kitchen.EXTEROCEPTION.stop, len(leg.exteroceptors)),
    ]
    dims = {"obs_dim": kitchen.FEATURE_DIM, "action_dim": kitchen.ACTION_DIM}
    policy: SequencePolicy
    if kind == "mlp":
        policy = MLPPolicy(**dims, seed=seed)
    elif kind == "gru":
        budget = BrainPolicy(
            "flyleg", RateDynamics(pack, leg.interface), channels=channels, **dims
        ).trainable_parameter_count()
        hidden = gru_hidden_for_budget(kitchen.FEATURE_DIM, kitchen.ACTION_DIM, budget)
        policy = GRUPolicy(**dims, hidden=hidden, seed=seed)
    else:
        graph_pack, interface = pack, leg.interface
        if kind == "flyleg_shuffled":
            graph_pack = shuffle_pack(pack, seed + 17000)
            recorded = json.loads((run_root / f"shuffled-{seed}.json").read_text())
            if graph_pack.fingerprint() != recorded["fingerprint"]:
                raise ValueError("Regenerated shuffle differs from the one used in training")
            interface = NeuralInterface.bind(
                graph_pack,
                leg.interface.input_body_ids,
                leg.interface.output_body_ids,
                label=leg.interface.label,
            )
        policy = BrainPolicy(
            kind,
            RateDynamics(graph_pack, interface),
            neural_steps=config.neural_steps,
            seed=seed,
            channels=channels,
            **dims,
        )
    policy.load(run_root / f"{kind}-{seed}" / "policy.safetensors")
    return policy


def rollout_frames(
    env: Any, policy: SequencePolicy, episode_seed: int, title: str
) -> tuple[list[np.ndarray], list[str]]:
    controller = kitchen.PositionFeatures(MlxController(policy))
    observation, _ = env.reset(seed=episode_seed)
    controller.reset()
    completed: list[str] = []
    total = len(env.unwrapped.goal)

    def frame(step: int) -> np.ndarray:
        done = ", ".join(completed) if completed else "none yet"
        detail = (
            f"episode {episode_seed} · step {step}/280 · tasks {len(completed)}/{total}: {done}"
        )
        return annotate(env.render(), title, detail, success=len(completed) == total)

    frames = [frame(0)]
    for step in range(env.spec.max_episode_steps):
        action = controller.act(np.asarray(observation["observation"], dtype=np.float32))
        observation, _, terminated, truncated, info = env.step(action.astype(np.float64))
        completed = list(info["episode_task_completions"])
        frames.append(frame(step + 1))
        if terminated or truncated:
            break
    return frames, completed


def record_kitchen(
    run_root: Path,
    seed: int,
    episodes: list[int],
    pack_root: Path,
    annotations: Path,
    output: Path,
) -> dict[str, Any]:
    """Per-controller clips plus a 2x2 comparison of the same episode for every controller."""
    config = FlyLegConfig.model_validate_json((run_root / "config.json").read_text())
    kinds = [kind for kind in config.policies if (run_root / f"{kind}-{seed}").is_dir()]
    env = (
        kitchen._minari()
        .load_dataset(kitchen.DATASETS[config.split])
        .recover_environment(render_mode="rgb_array")
    )
    manifest: dict[str, Any] = {"split": config.split, "seed": seed, "videos": []}
    try:
        clips: dict[str, dict[int, list[np.ndarray]]] = {}
        for kind in kinds:
            policy = load_flyleg_policy(run_root, kind, seed, pack_root, annotations)
            clips[kind] = {}
            for episode in episodes:
                frames, completed = rollout_frames(
                    env, policy, episode, f"{TITLES[kind]} · seed {seed}"
                )
                clips[kind][episode] = frames
                manifest["videos"].append(
                    {"kind": kind, "episode": episode, "completed": completed}
                )
            path = output / f"kitchen-{config.split}-{kind}-seed{seed}.mp4"
            write_video(path, [f for episode in episodes for f in clips[kind][episode]], 12)
        for episode in episodes:
            grid = tile([clips[kind][episode] for kind in kinds])
            path = output / f"kitchen-{config.split}-comparison-seed{seed}-episode{episode}.mp4"
            write_video(path, grid, 12)
    finally:
        env.close()
    save_json(output / f"kitchen-{config.split}-seed{seed}.json", manifest)
    return manifest
