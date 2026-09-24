"""Manipulation benchmark smoke run: teacher table per task and split, throughput and a video.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/manipulation_smoke.py

1. Teacher table. For every split and every task template of that split, the scripted teacher
   runs ``--episodes`` episodes in one batched environment per split (each split's own seed
   block), and the success rate, the mean steps to success and, for failures, the subgoal
   skill the episode was stuck at are written to ``docs/results/manipulation-teacher.json``.
2. Throughput. 128 batched environments on the train split, random actions and then the
   teacher in the loop, in environment steps per second.
3. Video. The teacher on every task template (the held-out compositions from their own
   split), plus one episode from the unseen-objects split and one from the unseen-furniture
   split, tiled into ``runs/manipulation/teacher-smoke.mp4`` with every frame labelled.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from flyarm.grasp.task import ACTION_DIM
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.env import DEFAULT_ASSET_ROOT, BatchedManipulation, PandaManipulationEnv
from flyarm.manipulation.splits import SPLITS
from flyarm.manipulation.teacher import ManipulationTeacher
from flyarm.video import annotate, tile, write_video

VIDEO_STRIDE = 3  # render every third control step; the video plays at 20 fps (3x real time)


def teacher_cells(model: Path, split: str, episodes: int, asset_root: Path) -> dict[str, Any]:
    """Run the teacher on ``episodes`` episodes of every template of ``split``."""
    templates = list(SPLITS[split].templates)
    names = [name for name in templates for _ in range(episodes)]
    env = BatchedManipulation(model, len(names), split=split, asset_root=asset_root)
    teacher = ManipulationTeacher(env)
    seeds = np.arange(len(names)) + SPLITS[split].seed_start
    env.reset(seeds=seeds, templates=names)
    teacher.reset()
    active = np.ones(len(names), dtype=bool)
    success = np.zeros(len(names), dtype=bool)
    steps = np.zeros(len(names), dtype=np.int64)
    reached = np.zeros(len(names), dtype=np.int64)
    started = time.perf_counter()
    for step in range(env.horizon):
        result = env.step(teacher.act(), auto_reset=False)
        reached = np.where(active, np.maximum(reached, result.subgoals_done), reached)
        finished = active & result.success
        steps[finished] = step + 1
        success |= finished
        active &= ~(result.success | result.truncated)
        if not active.any():
            break
    elapsed = time.perf_counter() - started
    cells = {}
    for name in templates:
        mine = np.array(names) == name
        stuck = collections.Counter(
            tk.SKILLS[env.episodes[row].subgoals[int(reached[row])].kind]
            for row in np.flatnonzero(mine & ~success)
        )
        cells[name] = {
            "episodes": int(mine.sum()),
            "successes": int(success[mine].sum()),
            "success_rate": float(success[mine].mean()),
            "subgoals": len(tk.TEMPLATES[name].steps),
            "horizon": tk.horizon(len(tk.TEMPLATES[name].steps)),
            "mean_steps_to_success": (
                float(steps[mine & success].mean()) if (mine & success).any() else None
            ),
            "mean_subgoals_done": float(reached[mine].mean()),
            "failures_stuck_at": dict(stuck),
        }
    return {
        "split": split,
        "seeds": [int(seeds[0]), int(seeds[-1])],
        "wall_seconds": round(elapsed, 1),
        "cells": cells,
    }


def throughput(model: Path, num_envs: int, steps: int, asset_root: Path) -> dict[str, Any]:
    env = BatchedManipulation(model, num_envs, split="train", asset_root=asset_root)
    generator = np.random.default_rng(0)
    env.reset()
    env.step(generator.uniform(-1, 1, (num_envs, ACTION_DIM)))
    started = time.perf_counter()
    for _ in range(steps):
        env.step(generator.uniform(-1, 1, (num_envs, ACTION_DIM)))
    random_rate = num_envs * steps / (time.perf_counter() - started)
    teacher = ManipulationTeacher(env)
    env.reset()
    teacher.reset()
    started = time.perf_counter()
    for _ in range(steps):
        result = env.step(teacher.act())
        teacher.reset(np.flatnonzero(result.terminated | result.truncated))
    teacher_rate = num_envs * steps / (time.perf_counter() - started)
    return {
        "num_envs": num_envs,
        "steps": steps,
        "threads": int(os.environ.get("FLYARM_SIM_THREADS", "0")),
        "random_actions_steps_per_second": round(random_rate, 1),
        "teacher_steps_per_second": round(teacher_rate, 1),
    }


def record_video(model: Path, output: Path, asset_root: Path) -> list[dict[str, Any]]:
    """One labelled teacher episode per template, plus unseen objects and unseen furniture."""
    plan = [("train", name) for name in SPLITS["train"].templates]
    plan += [("unseen_composition", name) for name in SPLITS["unseen_composition"].templates]
    plan += [("unseen_objects", "tidy"), ("unseen_furniture", "shelve")]
    clips, outcomes = [], []
    environments: dict[str, PandaManipulationEnv] = {}
    for k, (split, template) in enumerate(plan):
        if split not in environments:
            environments[split] = PandaManipulationEnv(
                model, split=split, asset_root=asset_root, render_size=(300, 400)
            )
        env = environments[split]
        teacher = ManipulationTeacher(env.sim)
        seed = SPLITS[split].seed_start + 700_000 + k
        _, info = env.reset(seed=seed, options={"template": template})
        teacher.reset()
        title = f"{template} ({split.replace('_', ' ')})"
        frames = [annotate(env.render(), title, _status(info))]
        for step in range(env.horizon):
            _, _, terminated, truncated, info = env.step(teacher.act()[0])
            if step % VIDEO_STRIDE == 0 or terminated:
                frames.append(
                    annotate(env.render(), title, _status(info), success=info["is_success"])
                )
            if terminated or truncated:
                break
        clips.append(frames)
        outcomes.append(
            {
                "split": split,
                "template": template,
                "seed": seed,
                "success": info["is_success"],
                "subgoals": info["subgoals"],
                "steps": info["steps"],
            }
        )
    for env in environments.values():
        env.close()
    write_video(output, tile(clips, columns=4), fps=20)
    return outcomes


def _status(info: dict[str, Any]) -> str:
    total = len(info["subgoals"])
    return f"step {info['steps']} · {info['subgoals_done']}/{total} done\nnow: {info['current']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/manipulation-teacher.json")
    )
    parser.add_argument("--video", type=Path, default=Path("runs/manipulation/teacher-smoke.mp4"))
    parser.add_argument("--throughput-envs", type=int, default=128)
    parser.add_argument("--throughput-steps", type=int, default=200)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    arguments = parser.parse_args()
    if not arguments.skip_video and arguments.video.exists():
        # flyarm.video never overwrites a video; fail now rather than after the table.
        parser.error(f"{arguments.video} exists; move it away or pass --video")

    table = {}
    for split in arguments.splits:
        table[split] = teacher_cells(
            arguments.model, split, arguments.episodes, arguments.asset_root
        )
        for name, cell in table[split]["cells"].items():
            print(
                f"{split:20s} {name:22s} {cell['successes']:3d}/{cell['episodes']} "
                f"steps {cell['mean_steps_to_success']} stuck {cell['failures_stuck_at']}"
            )
    record: dict[str, Any] = {
        "benchmark": "articulated multi-step manipulation (flyarm.manipulation)",
        "machine": platform.platform(),
        "episodes_per_cell": arguments.episodes,
        "teacher": table,
    }
    if not arguments.skip_throughput:
        record["throughput"] = throughput(
            arguments.model,
            arguments.throughput_envs,
            arguments.throughput_steps,
            arguments.asset_root,
        )
        print(json.dumps(record["throughput"]))
    _write(arguments.output, record)  # the table survives a failure while recording the video
    if not arguments.skip_video:
        record["video"] = {
            "path": str(arguments.video),
            "stride": VIDEO_STRIDE,
            "episodes": record_video(arguments.model, arguments.video, arguments.asset_root),
        }
        print(f"wrote {arguments.video}")
        _write(arguments.output, record)


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
