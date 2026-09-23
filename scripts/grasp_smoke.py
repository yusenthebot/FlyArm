"""Grasp environment smoke run: per-object teacher success, batched throughput and a video.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/grasp_smoke.py \
        --model assets/menagerie/franka_emika_panda/scene.xml

1. Teacher table. The scripted teacher runs ``--episodes`` episodes on every object of the
   manifest (the candidate pool, not only the kept split) in one batched environment, on the
   same seeds for every object, and the per-object success, grasp and lift rates are written
   to ``docs/results/grasp-teacher.json``. ``scripts/grasp_split.py`` reads that table to drop
   objects the teacher cannot grasp and to split the rest.
2. Throughput. 128 batched environments stepped with random actions (physics and task rules),
   then with the teacher in the loop, in environment steps per second.
3. Video. The teacher on four train and four held-out objects in the single environment,
   every frame labelled with the object, its split, the step and the physical state.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from flyarm.grasp import task
from flyarm.grasp.env import DEFAULT_ASSET_ROOT, BatchedGrasp, PandaGraspEnv
from flyarm.grasp.objects import FAMILIES, GraspObject, load_manifest, load_split
from flyarm.grasp.teacher import GraspTeacher
from flyarm.video import annotate, tile, write_video

TEACHER_SEED = 900_000  # teacher evaluation seeds, disjoint from any training seed block
VIDEO_SEED = 910_000


def teacher_table(
    model: Path, objects: list[GraspObject], episodes: int, asset_root: Path
) -> dict[str, Any]:
    """Run the teacher on every object for ``episodes`` seeds; per-object rates."""
    count = len(objects)
    env = BatchedGrasp(model, count * episodes, objects=objects, asset_root=asset_root)
    teacher = GraspTeacher(env)
    index = np.repeat(np.arange(count), episodes)
    seeds = np.tile(np.arange(TEACHER_SEED, TEACHER_SEED + episodes), count)
    env.reset(seeds=seeds, objects=index)
    teacher.reset()
    active = np.ones(len(index), dtype=bool)
    success = np.zeros(len(index), dtype=bool)
    grasped = np.zeros(len(index), dtype=bool)
    lifted = np.zeros(len(index), dtype=bool)
    steps = np.full(len(index), env.horizon)
    started = time.perf_counter()
    for step in range(env.horizon):
        result = env.step(teacher.act(), auto_reset=False)
        grasped |= active & result.grasped
        lifted |= active & result.lifted
        finished = active & result.success
        steps[finished] = step + 1
        success |= finished
        active &= ~(result.success | result.truncated)
        if not active.any():
            break
    elapsed = time.perf_counter() - started
    rows = []
    for k, item in enumerate(objects):
        mine = index == k
        rows.append(
            {
                "object": item.name,
                "family": item.family,
                "success_rate": float(success[mine].mean()),
                "grasp_rate": float(grasped[mine].mean()),
                "lift_rate": float(lifted[mine].mean()),
                "successes": int(success[mine].sum()),
                "episodes": episodes,
                "mean_steps_to_success": (
                    float(steps[mine & success].mean()) if (mine & success).any() else None
                ),
            }
        )
    return {
        "seeds": [TEACHER_SEED, TEACHER_SEED + episodes - 1],
        "horizon": env.horizon,
        "wall_seconds": round(elapsed, 2),
        "objects": rows,
    }


def throughput(model: Path, num_envs: int, steps: int, asset_root: Path) -> dict[str, float]:
    """Environment steps per second on the train split, random actions and teacher actions."""
    env = BatchedGrasp(model, num_envs, split="train", asset_root=asset_root)
    generator = np.random.default_rng(0)
    env.reset()
    env.step(generator.uniform(-1, 1, (num_envs, task.ACTION_DIM)))  # warm the thread pool
    started = time.perf_counter()
    for _ in range(steps):
        env.step(generator.uniform(-1, 1, (num_envs, task.ACTION_DIM)))
    random_rate = num_envs * steps / (time.perf_counter() - started)
    teacher = GraspTeacher(env)
    env.reset()
    teacher.reset()
    started = time.perf_counter()
    for _ in range(steps):
        result = env.step(teacher.act())
        done = np.flatnonzero(result.terminated | result.truncated)
        teacher.reset(done)
    teacher_rate = num_envs * steps / (time.perf_counter() - started)
    return {
        "num_envs": num_envs,
        "steps": steps,
        "threads": int(os.environ.get("FLYARM_SIM_THREADS", "0")),
        "random_actions_steps_per_second": round(random_rate, 1),
        "teacher_steps_per_second": round(teacher_rate, 1),
    }


def record(
    model: Path, names: list[tuple[str, str]], output: Path, asset_root: Path
) -> list[dict[str, Any]]:
    """Teacher rollouts on the named (object, split) pairs, tiled into one labelled video."""
    objects = [load_manifest()[name] for name, _ in names]
    env = PandaGraspEnv(
        model,
        objects=objects,
        asset_root=asset_root,
        render_mode="rgb_array",
        render_size=(360, 480),
    )
    teacher = GraspTeacher(env.sim)
    clips, outcomes = [], []
    for k, (name, split) in enumerate(names):
        _, info = env.reset(seed=VIDEO_SEED + k, options={"object": name})
        teacher.reset()
        camera = env.side_camera()
        title = f"{name} ({split})"
        frames = [annotate(env.render(camera), title, _status(info, 0, teacher))]
        for step in range(env.horizon):
            _, _, terminated, truncated, info = env.step(teacher.act()[0])
            frames.append(
                annotate(
                    env.render(camera),
                    title,
                    _status(info, step + 1, teacher),
                    success=info["is_success"],
                )
            )
            if terminated or truncated:
                break
        clips.append(frames)
        outcomes.append({"object": name, "split": split, "success": info["is_success"]})
    env.close()
    write_video(output, tile(clips, columns=4), fps=20)
    return outcomes


def _status(info: dict[str, Any], step: int, teacher: GraspTeacher) -> str:
    state = (
        "held"
        if info["is_success"]
        else "lifted"
        if info["height_gain"] >= task.LIFT_HEIGHT
        else "grasped"
        if info["grasped"]
        else "free"
    )
    return (
        f"step {step} · {teacher.phase_names()[0]} · {state}\n"
        f"lift {100 * info['height_gain']:.1f} cm · hold {info['hold_steps']}/{task.HOLD_STEPS}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("docs/results/grasp-teacher.json"))
    parser.add_argument("--video", type=Path, default=Path("runs/grasp/teacher-smoke.mp4"))
    parser.add_argument("--throughput-envs", type=int, default=128)
    parser.add_argument("--throughput-steps", type=int, default=400)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    arguments = parser.parse_args()

    manifest = load_manifest()
    try:
        split = load_split()
    except FileNotFoundError:  # first run: the split is made from this table
        split = {"train": [], "test": [], "dropped": []}
    membership = {name: part for part in ("train", "test") for name in split[part]}
    membership |= {name: "dropped" for name in split["dropped"]}
    table = teacher_table(
        arguments.model, list(manifest.values()), arguments.episodes, arguments.asset_root
    )
    for row in table["objects"]:
        row["split"] = membership.get(row["object"], "unsplit")
        print(
            f"{row['object']:20s} {row['family']:8s} {row['split']:8s} "
            f"success {row['success_rate']:.2f} grasp {row['grasp_rate']:.2f} "
            f"lift {row['lift_rate']:.2f}"
        )
    summary = {}
    for part in ("train", "test", "dropped"):
        rates = [row["success_rate"] for row in table["objects"] if row["split"] == part]
        if rates:
            summary[part] = {"objects": len(rates), "mean_success": round(float(np.mean(rates)), 4)}
    record_: dict[str, Any] = {
        "task": "generalizable grasp: lift 10 cm and hold 20 steps with both fingers",
        "machine": platform.platform(),
        "teacher": table,
        "summary": summary,
    }
    print(json.dumps(summary))
    if not arguments.skip_throughput:
        record_["throughput"] = throughput(
            arguments.model,
            arguments.throughput_envs,
            arguments.throughput_steps,
            arguments.asset_root,
        )
        print(json.dumps(record_["throughput"]))
    if not arguments.skip_video:
        chosen = _video_objects(split)
        record_["video"] = {
            "path": str(arguments.video),
            "seed": VIDEO_SEED,
            "episodes": record(arguments.model, chosen, arguments.video, arguments.asset_root),
        }
        print(f"wrote {arguments.video}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record_, indent=2) + "\n")
    print(f"wrote {arguments.output}")


def _video_objects(split: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Four train objects and four held-out objects, eight different families between them."""
    manifest = load_manifest()
    chosen: list[tuple[str, str]] = []
    for part, families in (("train", FAMILIES[0::2]), ("test", FAMILIES[1::2])):
        for family in families:
            names = [name for name in split[part] if manifest[name].family == family]
            if names:
                chosen.append((names[0], part))
    return chosen


if __name__ == "__main__":
    main()
