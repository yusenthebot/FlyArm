"""Render one episode and record the fly connectome's activity on every frame (the site's data).

    FLYARM_SIM_THREADS=4 PYTHONPATH=src:scripts .venv/bin/python scripts/record_brain_activity.py \\
        --policy ppo:runs/ppo-manipulation-nophase-001@500 --split iid_test --template shelve \\
        --seed 3002502 --conditions intact state_reset silence_vnc_interneurons \\
        --output site/data/shelve

For each condition (flyarm.whole_brain.lesion) the episode is rendered without labels into
``OUTPUT/CONDITION.mp4`` and, on the same frames, the rate of a fixed sample of neurons is
stored as uint8 in ``OUTPUT/CONDITION.bin`` ([frames, neurons]) on a signed square-root scale,
u = 127.5 (1 + sign(r) sqrt(|r|)), so the small rates of interior neurons (0.01 to 0.1) keep
enough levels. ``OUTPUT/neurons.json`` names the sample: per group (ascending inputs, central brain,
optic lobes, VNC interneurons, sensory neurons, descending and motor outputs) the neurons'
pack indices and body IDs, drawn once with a fixed seed. ``OUTPUT/episode.json`` keeps, per
condition, the subgoal shown on each frame and the outcome. The activity is the state after
the control step that precedes the frame, so frame and activity are the same moment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from final_evaluation import resolve

from flyarm.manipulation.env import PandaManipulationEnv
from flyarm.video import write_video
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.lesion import REGIONS, LesionedDynamics, region_mask
from flyarm.whole_brain.policy import BrainPolicy, MlxController

SAMPLE = {
    "ascending": 360,
    "central_brain": 700,
    "optic_lobes": 700,
    "vnc_interneurons": 500,
    "sensory": 300,
    "descending": 360,
    "motor": 280,
}


def neuron_sample(pack: ConnectomePack, policy: BrainPolicy) -> dict[str, np.ndarray]:
    dynamics = policy.dynamics
    inputs = np.asarray(dynamics.input_indices)
    outputs = np.asarray(dynamics.output_indices)
    interface = np.concatenate([inputs, outputs])
    neurons = pack.neurons()
    superclass = neurons.superclass.to_numpy()
    pools = {
        "ascending": inputs,
        "descending": outputs[superclass[outputs] == "descending_neuron"],
        "motor": outputs[superclass[outputs] == "vnc_motor"],
    } | {region: np.flatnonzero(region_mask(neurons, region, interface)) for region in REGIONS}
    generator = np.random.default_rng(7)
    return {
        group: np.sort(generator.choice(pools[group], size=count, replace=False))
        for group, count in SAMPLE.items()
    }


def record(
    env: PandaManipulationEnv,
    policy: BrainPolicy,
    seed: int,
    template: str,
    stride: int,
    index: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, list[str], dict[str, Any]]:
    obs, info = env.reset(seed=seed, options={"template": template})
    controller = MlxController(policy)
    frames, activity, labels = [], [], []
    current = np.zeros(len(index), dtype=np.float32)
    for step in range(env.horizon):
        if step % stride == 0:
            frames.append(env.render())
            activity.append(current)
            labels.append(f"{info['subgoals_done']}/{len(info['subgoals'])} {info['current']}")
        obs, _, terminated, truncated, info = env.step(controller.act(obs))
        current = np.asarray(controller.state)[index, 0].astype(np.float32)
        if terminated or truncated:
            break
    frames.append(env.render())
    activity.append(current)
    labels.append(f"{info['subgoals_done']}/{len(info['subgoals'])} done")
    stacked = np.stack(activity)
    scaled = np.sign(stacked) * np.sqrt(np.abs(stacked))
    rates = np.clip(np.round((scaled + 1.0) * 127.5), 0, 255).astype(np.uint8)
    outcome = {
        "success": bool(info["is_success"]),
        "subgoals_done": int(info["subgoals_done"]),
        "subgoals": int(len(info["subgoals"])),
    }
    return frames, rates, labels, outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--split", default="iid_test")
    parser.add_argument("--template", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conditions", nargs="+", default=["intact"])
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, policy, provenance = resolve(args.policy, args.pack)
    if not isinstance(policy, BrainPolicy):
        raise ValueError("activity needs a connectome controller")
    pack = ConnectomePack.load(args.pack)
    sample = neuron_sample(pack, policy)
    index = np.concatenate(list(sample.values()))
    args.output.mkdir(parents=True, exist_ok=True)
    neurons = pack.neurons()
    (args.output / "neurons.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "name": group,
                        "count": int(len(chosen)),
                        "body_ids": [int(b) for b in pack.body_ids[chosen]],
                        "types": [str(t) for t in neurons.type.to_numpy()[chosen]],
                    }
                    for group, chosen in sample.items()
                ],
                "encoding": "uint8 u: s = u / 127.5 - 1, rate = sign(s) s^2",
            }
        )
    )
    interface = np.concatenate(
        [np.asarray(policy.dynamics.input_indices), np.asarray(policy.dynamics.output_indices)]
    )
    episode: dict[str, Any] = {
        **provenance,
        "split": args.split,
        "template": args.template,
        "seed": args.seed,
        "stride": args.stride,
        "fps": args.fps,
        "conditions": {},
    }
    env = PandaManipulationEnv(
        args.model,
        split=args.split,
        asset_root=args.asset_root,
        cue=config.cue,
        velocities=config.velocities,
        phase_cue=config.phase_cue,
        render_size=(args.height, args.width),
    )
    try:
        for condition in args.conditions:
            if condition == "intact":
                lesioned = policy
            elif condition == "state_reset":
                lesioned = policy.with_dynamics(
                    policy.kind, LesionedDynamics(policy.dynamics, reset=True)
                )
            else:
                region = condition.removeprefix("silence_")
                lesioned = policy.with_dynamics(
                    policy.kind,
                    LesionedDynamics(policy.dynamics, region_mask(neurons, region, interface)),
                )
            frames, rates, labels, outcome = record(
                env, lesioned, args.seed, args.template, args.stride, index
            )
            write_video(args.output / f"{condition}.mp4", frames, args.fps)
            (args.output / f"{condition}.bin").write_bytes(rates.tobytes())
            episode["conditions"][condition] = {
                **outcome,
                "frames": len(frames),
                "labels": labels,
            }
            print(condition, outcome, len(frames), "frames", flush=True)
    finally:
        env.close()
    (args.output / "episode.json").write_text(json.dumps(episode))


if __name__ == "__main__":
    main()
