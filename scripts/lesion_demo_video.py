"""Demo: the same episode with the fly connectome intact, its state reset, and a lesion.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src:scripts .venv/bin/python scripts/lesion_demo_video.py \\
        --policy ppo:runs/ppo-manipulation-nophase-001@500 \\
        --output docs/videos/fly-brain-lesions.mp4

Three panels play one episode side by side with the same trained encoder and decoder: the
intact connectome, the connectome with its state cleared before every control step, and the
connectome with the VNC interneurons silenced (flyarm.whole_brain.lesion, research log E63).
Candidates are the iid test episodes of docs/results/manipulation-lesions.json where the intact
controller succeeded and the VNC lesion failed; the first whose single-environment replay
keeps that contrast is shown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from final_evaluation import FINAL_OFFSET, resolve
from manipulation_demo_video import SIZE, render_episode

from flyarm.manipulation import rollout
from flyarm.manipulation.env import PandaManipulationEnv
from flyarm.video import write_video
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.lesion import LesionedDynamics, region_mask

PANELS = (
    ("intact", "fly connectome, intact"),
    ("state_reset", "state cleared every step"),
    ("silence_vnc_interneurons", "VNC interneurons silenced"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--lesions", type=Path, default=Path("docs/results/manipulation-lesions.json")
    )
    parser.add_argument("--templates", nargs="*", default=["tidy", "unpack", "shelve"])
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config, policy, provenance = resolve(args.policy, args.pack)
    record = json.loads(args.lesions.read_text())
    conditions = record["conditions"]
    episodes = rollout.plan(
        record["protocol"]["split"], record["protocol"]["episodes_per_template"], FINAL_OFFSET
    )
    candidates = [
        (template, seed)
        for index, (seed, template) in enumerate(
            zip(episodes.seeds, episodes.templates, strict=True)
        )
        if template in args.templates
        and conditions["intact"]["outcomes"][index]
        and not conditions["silence_vnc_interneurons"]["outcomes"][index]
    ][: args.candidates]
    dynamics = policy.dynamics
    interface = np.concatenate(
        [np.asarray(dynamics.input_indices), np.asarray(dynamics.output_indices)]
    )
    vnc = region_mask(ConnectomePack.load(args.pack).neurons(), "vnc_interneurons", interface)
    policies = {
        "intact": policy,
        "state_reset": policy.with_dynamics(policy.kind, LesionedDynamics(dynamics, reset=True)),
        "silence_vnc_interneurons": policy.with_dynamics(
            policy.kind, LesionedDynamics(dynamics, vnc)
        ),
    }
    size = SIZE
    env = PandaManipulationEnv(
        args.model,
        split=record["protocol"]["split"],
        asset_root=args.asset_root,
        cue=config.cue,
        velocities=config.velocities,
        phase_cue=config.phase_cue,
        render_size=size,
    )
    shown = None
    try:
        for template, seed in candidates:
            clips, outcomes = {}, {}
            for key, label in PANELS:
                clips[key], outcomes[key] = render_episode(
                    env, policies[key], seed, template, label, args.stride, size
                )
            print(template, seed, outcomes, flush=True)
            if outcomes["intact"] and not outcomes["silence_vnc_interneurons"]:
                shown = (template, seed, clips, outcomes)
                break
    finally:
        env.close()
    if shown is None:
        raise RuntimeError("no candidate kept the contrast in the single environment")
    template, seed, clips, outcomes = shown
    length = max(len(clip) for clip in clips.values())
    frames = [
        np.concatenate([clips[key][min(i, len(clips[key]) - 1)] for key, _ in PANELS], axis=1)
        for i in range(length)
    ]
    write_video(args.output, frames, args.fps)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {**provenance, "template": template, "seed": seed, "outcomes": outcomes}, indent=1
        )
        + "\n"
    )
    print(f"wrote {args.output} ({len(frames)} frames): {template} {seed} {outcomes}", flush=True)


if __name__ == "__main__":
    main()
