"""Rebuild saved B2 checkpoints and record labelled FrankaKitchen rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from flyarm.benchmarks import kitchen
from flyarm.config import FlyLegConfig
from flyarm.experiment import save_json
from flyarm.flyleg.experiment import make_policy
from flyarm.flyleg.interface import front_leg_interface, leg_channels
from flyarm.interfaces import NeuralInterface
from flyarm.video import annotate, tile, write_video
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.policy import (
    ACTPolicy,
    BrainPolicy,
    GRUPolicy,
    MLPPolicy,
    MlxController,
    SequencePolicy,
)
from flyarm.whole_brain.shuffle import shuffle_pack

TITLES = {
    "flyleg": "Fly CNS (MaleCNS) as the left front leg",
    "flyleg_shuffled": "Degree-matched shuffled CNS, same leg interface",
    "mlp": "MLP behavior cloning (D4RL BC)",
    "gru": "GRU, parameter-matched",
    "act": "ACT reference (transformer + CVAE), not a fly model",
}


def load_flyleg_policy(
    run_root: Path, kind: str, seed: int, pack_root: Path, annotations: Path
) -> SequencePolicy:
    config = FlyLegConfig.model_validate_json((run_root / "config.json").read_text())
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    leg = front_leg_interface(
        pack, annotations, include_head=config.sensory_channels == "proprioception+head"
    )
    if NeuralInterface.load(run_root / "interface.json") != leg.interface:
        raise ValueError("Saved interface differs from the one regenerated from annotations")
    channels = leg_channels(leg)
    measured = RateDynamics(pack, leg.interface)
    dynamics: RateDynamics | None = measured
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
        dynamics = RateDynamics(graph_pack, interface)
    fly_budget = make_policy(
        "flyleg", config, 0, dynamics=measured, channels=channels, fly_budget=0
    ).trainable_parameter_count()
    policy = make_policy(
        kind, config, seed, dynamics=dynamics, channels=channels, fly_budget=fly_budget
    )
    policy.load(run_root / f"{kind}-{seed}" / "policy.safetensors")
    return policy


def _lesions(
    run_root: Path, seed: int, pack_root: Path, annotations: Path
) -> list[tuple[str, str, SequencePolicy]]:
    """Causal variants of the trained measured-CNS policy, most informative first."""
    policy = load_flyleg_policy(run_root, "flyleg", seed, pack_root, annotations)
    if not isinstance(policy, BrainPolicy) or policy.channels is None:
        return []
    pack = ConnectomePack.load(pack_root)
    interface = NeuralInterface.load(run_root / "interface.json")
    variants: list[tuple[str, str, SequencePolicy]] = []
    if len(policy.channels) > 1:
        variants.append(
            (
                "head_sensory_deprived",
                "Lesion: head senses removed",
                policy.silence_channel("head_sensory_deprived", 1),
            )
        )
    variants.append(
        (
            "deafferented_leg",
            "Lesion: leg proprioceptors silenced",
            policy.silence_channel("deafferented_leg", 0),
        )
    )
    variants.append(
        (
            "edges_off",
            "Lesion: every connectome edge removed",
            policy.with_dynamics("edges_off", RateDynamics(pack, interface, edges=False)),
        )
    )
    return variants


def rollout_frames(
    env: Any, policy: SequencePolicy | kitchen.Controller, episode_seed: int, title: str
) -> tuple[list[np.ndarray], list[str]]:
    """Annotated frames of one episode; a plain Controller reads the full observation."""
    controller: kitchen.Controller = (
        kitchen.PositionFeatures(MlxController(policy))
        if isinstance(policy, (BrainPolicy, GRUPolicy, MLPPolicy, ACTPolicy))
        else policy
    )
    observation, _ = env.reset(seed=episode_seed)
    controller.reset()
    completed: list[str] = []
    total = len(env.unwrapped.goal)

    def frame(step: int) -> np.ndarray:
        done = ", ".join(completed) if completed else "none yet"
        detail = (
            f"episode {episode_seed} · step {step}/280 · tasks {len(completed)}/{total}\n{done}"
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
    """Per-controller clips plus a grid comparing every controller on the same episode."""
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
            grid = tile([clips[kind][episode] for kind in kinds], 3 if len(kinds) > 4 else 2)
            path = output / f"kitchen-{config.split}-comparison-seed{seed}-episode{episode}.mp4"
            write_video(path, grid, 12)
        if "flyleg" in kinds:
            lesion_clips = [clips["flyleg"][episodes[0]]]
            for name, title, lesioned in _lesions(run_root, seed, pack_root, annotations):
                frames, completed = rollout_frames(
                    env, lesioned, episodes[0], f"{title} · seed {seed}"
                )
                lesion_clips.append(frames)
                manifest["videos"].append(
                    {"kind": f"flyleg:{name}", "episode": episodes[0], "completed": completed}
                )
            path = output / f"kitchen-{config.split}-lesions-seed{seed}-episode{episodes[0]}.mp4"
            write_video(path, tile(lesion_clips[:4]), 12)
    finally:
        env.close()
    save_json(output / f"kitchen-{config.split}-seed{seed}.json", manifest)
    return manifest


def record_teacher(split: str, episodes: list[int], output: Path) -> dict[str, Any]:
    """Rollouts of the demonstration tracker: what the benchmark's demonstrations support."""
    from flyarm.benchmarks.kitchen_expert import DemonstrationTracker

    tracker = DemonstrationTracker.from_data(kitchen.load(split))
    env = (
        kitchen._minari()
        .load_dataset(kitchen.DATASETS[split])
        .recover_environment(render_mode="rgb_array")
    )
    frames: list[np.ndarray] = []
    completed: dict[int, list[str]] = {}
    try:
        for episode in episodes:
            clip, done = rollout_frames(
                env, tracker, episode, "Demonstration tracker (teacher, not a learned controller)"
            )
            frames += clip
            completed[episode] = done
    finally:
        env.close()
    write_video(output, frames, 12)
    return {"split": split, "episodes": completed, "video": str(output)}
