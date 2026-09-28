"""Labelled demo video of one checkpoint: one successful and one failed episode per template.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/manipulation_demo_video.py \\
        --policy ppo:runs/ppo-manipulation-skill-dagger-002 --split iid_test \\
        --output runs/manipulation/demo-connectome-ppo-iid.mp4

``--policy`` takes the specs of scripts/final_evaluation.py. For every template of the split,
up to ``--candidates`` episodes (seeds from the final-evaluation block, disjoint from
selection) are rendered in a single-episode environment until one success and one failure
are found, each labelled with the checkpoint, the template, the seed, the subgoal count and
the current subgoal. Outcomes are the single-episode environment's own: the batched
environment of the evaluation rebuilds the same episodes, but contact-rich episodes can end
differently between the two simulators, so seeds are not preselected there. A template with
no success (or no failure) among the candidates shows a card saying so. The two episodes of a
template play side by side (success left, failure right), one template after the other,
every ``--stride``-th control step. ``--successes-only`` makes a highlight reel instead: the
first successful episode of every template, full width, templates without one left out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from flyarm.manipulation import rollout
from flyarm.manipulation.env import PandaManipulationEnv
from flyarm.video import annotate, write_video
from flyarm.whole_brain.policy import MlxController

sys.path.insert(0, str(Path(__file__).parent))
from final_evaluation import FINAL_OFFSET, resolve  # noqa: E402

SIZE = (304, 400)


def render_episode(
    env: PandaManipulationEnv,
    policy,
    seed: int,
    template: str,
    title: str,
    stride: int,
    size: tuple[int, int] = SIZE,
) -> tuple[list[np.ndarray], bool]:
    obs, info = env.reset(seed=seed, options={"template": template})
    controller = MlxController(policy)
    frames = []
    for step in range(env.horizon):
        if step % stride == 0:
            status = f"seed {seed} · step {step} · {info['subgoals_done']}/{len(info['subgoals'])}"
            frames.append(
                annotate(
                    env.render(),
                    f"{title} · {template}",
                    f"{status}\nnow: {info['current']}",
                    success=info["is_success"],
                )
            )
        obs, _, terminated, truncated, info = env.step(controller.act(obs))
        if terminated or truncated:
            break
    outcome = "success" if info["is_success"] else "failed"
    final = annotate(
        env.render(),
        f"{title} · {template}",
        f"seed {seed} · {outcome} · {info['subgoals_done']}/{len(info['subgoals'])} subgoals",
        success=info["is_success"],
    )
    return frames + [final] * 10, bool(info["is_success"])


def card(title: str, text: str, size: tuple[int, int] = SIZE) -> list[np.ndarray]:
    return [annotate(np.full((*size, 3), 40, np.uint8), title, text)] * 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="a spec of scripts/final_evaluation.py")
    parser.add_argument("--label", default=None, help="title on every frame (default: the spec)")
    parser.add_argument("--split", default="iid_test")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--successes-only", action="store_true", help="a highlight reel")
    parser.add_argument("--height", type=int, default=SIZE[0])
    parser.add_argument("--width", type=int, default=SIZE[1])
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
    title = args.label or args.policy
    env = PandaManipulationEnv(
        args.model,
        split=args.split,
        asset_root=args.asset_root,
        cue=config.cue,
        velocities=config.velocities,
        phase_cue=config.phase_cue,
        render_size=(args.height, args.width),
    )
    size = (args.height, args.width)
    frames: list[np.ndarray] = []
    shown: dict[str, dict[str, int | None]] = {}
    plan = rollout.plan(args.split, args.candidates, FINAL_OFFSET)
    try:
        for template in env.split.templates:
            seeds = [s for s, t in zip(plan.seeds, plan.templates, strict=True) if t == template]
            found: dict[str, tuple[int, list[np.ndarray]] | None] = {
                "success": None,
                "failure": None,
            }
            for seed in seeds:  # render candidates until one of each outcome is found
                clip, success = render_episode(
                    env, policy, seed, template, title, args.stride, size
                )
                key = "success" if success else "failure"
                if found[key] is None:
                    found[key] = (int(seed), clip)
                if args.successes_only and found["success"] is not None:
                    break
                if all(value is not None for value in found.values()):
                    break
            shown[template] = {k: (None if v is None else v[0]) for k, v in found.items()}
            print(f"{template}: {shown[template]}", flush=True)
            if args.successes_only:
                if found["success"] is not None:
                    frames.extend(found["success"][1])
                continue
            pair = [
                found[key][1]
                if found[key] is not None
                else card(f"{title} · {template}", f"no {key} in {len(seeds)} episodes", size)
                for key in ("success", "failure")
            ]
            length = max(len(clip) for clip in pair)
            for index in range(length):
                frames.append(
                    np.concatenate([clip[min(index, len(clip) - 1)] for clip in pair], axis=1)
                )
    finally:
        env.close()
    write_video(args.output, frames, args.fps)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {**provenance, "split": args.split, "candidates": args.candidates, "episodes": shown},
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {args.output} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
