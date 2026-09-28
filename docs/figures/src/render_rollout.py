"""Render frames of one real rollout of the reported connectome checkpoint for the pipeline figure.

The controller is the frozen MaleCNS connectome with its trained encoder and decoder (the PPO
checkpoint named below); the episode is an iid test episode it solves. Frames use the studio look
and camera of render_tasks.py. Writes assets/rollout-<step>.png for every --every-th step, so the
figure can pick the frame that shows the task in progress. Run from the repository root:

    PYTHONPATH=src:scripts .venv/bin/python docs/figures/src/render_rollout.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from final_evaluation import resolve  # noqa: E402
from PIL import Image  # noqa: E402
from render_tasks import MANIP_LOOKAT, MANIP_VIEW, MODEL, OBJECTS, SIZE, autocrop, camera, studio  # noqa: E402

from flyarm.manipulation.env import PandaManipulationEnv  # noqa: E402
from flyarm.whole_brain.policy import MlxController  # noqa: E402

POLICY = "ppo:runs/ppo-manipulation-encoder-002@150"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="iid_test")
    parser.add_argument("--template", default="retrieve")
    parser.add_argument("--seed", type=int, default=3001500)
    parser.add_argument("--every", type=int, default=60)
    args = parser.parse_args()
    config, policy, _ = resolve(POLICY, ROOT / "data" / "whole_brain" / "malecns-v1.0-c3")
    env = PandaManipulationEnv(
        MODEL,
        split=args.split,
        asset_root=OBJECTS,
        cue=config.cue,
        velocities=config.velocities,
        phase_cue=config.phase_cue,
        render_size=SIZE,
    )
    studio(env.model)
    obs, info = env.reset(seed=args.seed, options={"template": args.template})
    controller = MlxController(policy)
    for step in range(env.horizon):
        if step % args.every == 0:
            frame = autocrop(env.render(camera(MANIP_LOOKAT, *MANIP_VIEW)))
            out = HERE / "assets" / f"rollout-{step:04d}.png"
            Image.fromarray(frame).save(out, optimize=True)
            print(out.name, info["subgoals_done"], info["current"], flush=True)
        obs, _, terminated, truncated, info = env.step(controller.act(obs))
        if terminated or truncated:
            break
    print("success", info["is_success"], "steps", step + 1)


if __name__ == "__main__":
    main()
