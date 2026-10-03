"""Vertical demo video: the robot and the fly connectome's activity on the same frames.

    PYTHONPATH=src .venv/bin/python scripts/brain_demo_video.py --episode site/data/unpack \\
        --output docs/videos/fly-brain-demo-vertical.mp4

Reads an episode recorded by scripts/record_brain_activity.py (CONDITION.mp4, CONDITION.bin,
neurons.json, episode.json) and composes a 1088 x 1440 video: a title band, the robot, the
current subgoal, and the sampled neurons drawn on a schematic of the fly CNS (optic lobes and
central brain, the neck connective with descending and ascending neurons, the ventral nerve
cord with interneurons, motor and sensory neurons), coloured by their rate on each frame. The
layout and colours follow the project site (site/index.html).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flyarm.video import write_video

WIDTH, HEIGHT = 1088, 1440  # multiples of 16 for H.264
TITLE_H, ROBOT_H, FOOT_H = 210, 608, 64
BRAIN_H = HEIGHT - TITLE_H - ROBOT_H - FOOT_H
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
BG = (7, 11, 20)
PANEL = (10, 15, 28)
INK = (232, 237, 245)
MUTED = (147, 161, 187)
WARM = np.array([255, 181, 71])
COOL = np.array([76, 201, 240])
QUIET = np.array([38, 46, 68])
GOOD = (123, 216, 143)
GAIN = {"ascending": 1.0}  # every other group: 10 (rates of interior neurons are about 0.1)
LABELS = {
    "ascending": "上行神经元（输入）",
    "central_brain": "中央脑",
    "optic_lobes": "视叶",
    "vnc_interneurons": "VNC 中间神经元",
    "sensory": "感觉神经元",
    "descending": "下行神经元（输出）",
    "motor": "运动神经元（输出）",
}


SKILLS = {
    "open_drawer": "拉开抽屉",
    "close_drawer": "关上抽屉",
    "open_door": "掀开柜盖",
    "close_door": "合上柜盖",
    "pick": "拿起物体",
    "stack": "叠放",
}
TARGETS = {"region": "绿色区域", "shelf": "架子", "bin": "垃圾桶"}


def describe(label: str) -> str:
    """'2/7 place(coffee_can, region)' -> '第 3 步 / 7：把物体放到绿色区域'."""
    count, _, action = label.partition(" ")
    done, total = (int(x) for x in count.split("/"))
    skill = action.split("(", 1)[0]
    if skill == "done":
        return f"{done}/{total} 步完成"
    if skill == "place":
        target = action.rstrip(")").split(",")[-1].strip()
        text = f"放到{TARGETS.get(target, '抽屉里')}"
    else:
        text = SKILLS.get(skill, skill)
    return f"第 {done + 1}/{total} 步：{text}"


def _ellipse(rng: np.random.Generator, n: int, cx: float, cy: float, rx: float, ry: float):
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.sqrt(rng.uniform(0, 1, n))
    return np.stack([cx + np.cos(angle) * rx * radius, cy + np.sin(angle) * ry * radius], 1)


def place(name: str, count: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-square positions of a group on the CNS schematic (x right, y down)."""
    if name == "optic_lobes":
        left = _ellipse(rng, count, 0.15, 0.24, 0.12, 0.17)
        right = _ellipse(rng, count, 0.85, 0.24, 0.12, 0.17)
        return np.where((np.arange(count) % 2 == 0)[:, None], left, right)
    if name == "central_brain":
        return _ellipse(rng, count, 0.5, 0.22, 0.2, 0.14)
    if name == "descending":
        return np.stack([0.43 + rng.uniform(0, 0.06, count), 0.38 + rng.uniform(0, 0.17, count)], 1)
    if name == "ascending":
        return np.stack([0.51 + rng.uniform(0, 0.06, count), 0.38 + rng.uniform(0, 0.17, count)], 1)
    if name == "vnc_interneurons":
        return _ellipse(rng, count, 0.5, 0.76, 0.13, 0.18)
    if name == "motor":
        angle = rng.uniform(0, 2 * np.pi, count)
        scale = 0.9 + rng.uniform(0, 0.12, count)
        return np.stack(
            [0.5 + np.cos(angle) * 0.15 * scale, 0.76 + np.sin(angle) * 0.205 * scale], 1
        )
    return np.stack([0.25 + rng.uniform(0, 0.07, count), 0.6 + rng.uniform(0, 0.32, count)], 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--condition", default="intact")
    parser.add_argument("--title", default="一只果蝇的完整大脑，在控制机械臂")
    parser.add_argument(
        "--subtitle",
        default="166,700 个神经元一根线都不改 · 训练时没见过的物体 · 7 步任务",
    )
    parser.add_argument("--footer", default="github.com/yusenthebot/FlyArm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode = json.loads((args.episode / "episode.json").read_text())
    info = episode["conditions"][args.condition]
    groups = json.loads((args.episode / "neurons.json").read_text())["groups"]
    total = sum(g["count"] for g in groups)
    raw = np.frombuffer((args.episode / f"{args.condition}.bin").read_bytes(), np.uint8)
    scaled = raw.reshape(-1, total).astype(np.float32) / 127.5 - 1.0
    rates = np.sign(scaled) * scaled**2
    gains = np.concatenate([np.full(g["count"], GAIN.get(g["name"], 10.0)) for g in groups])
    rng = np.random.default_rng(3)
    unit = np.concatenate([place(g["name"], g["count"], rng) for g in groups])
    side = BRAIN_H - 70
    origin = np.array([(WIDTH - side) / 2, TITLE_H + ROBOT_H + 36])
    points = origin + unit * side

    title = ImageFont.truetype(FONT, 54)
    subtitle = ImageFont.truetype(FONT, 28)
    label = ImageFont.truetype(FONT, 22)
    status = ImageFont.truetype(FONT, 26)
    frame_labels = info["labels"]
    reader = imageio.get_reader(args.episode / f"{args.condition}.mp4")
    frames = []
    for index, robot in enumerate(reader):
        if index >= len(rates):
            break
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((48, 46), args.title, font=title, fill=INK)
        draw.text((50, 128), args.subtitle, font=subtitle, fill=MUTED)
        robot_image = Image.fromarray(robot).resize((WIDTH, ROBOT_H), Image.Resampling.LANCZOS)
        canvas.paste(robot_image, (0, TITLE_H))
        text = frame_labels[min(index, len(frame_labels) - 1)]
        done = index >= len(rates) - 2
        badge = "成功：7 步全部完成" if done and info["success"] else describe(text)
        box = draw.textbbox((0, 0), badge, font=status)
        draw.rounded_rectangle(
            (28, TITLE_H + 24, 28 + box[2] + 32, TITLE_H + 24 + box[3] + 22),
            radius=22,
            fill=(7, 11, 20),
            outline=GOOD if done else (70, 82, 110),
        )
        draw.text((44, TITLE_H + 33), badge, font=status, fill=GOOD if done else INK)
        draw.rectangle((0, TITLE_H + ROBOT_H, WIDTH, HEIGHT - FOOT_H), fill=PANEL)
        magnitude = np.clip(np.abs(rates[index]) * gains, 0, 1)[:, None]
        colour = np.where((rates[index] >= 0)[:, None], WARM, COOL)
        mixed = (QUIET + (colour - QUIET) * magnitude).astype(int)
        # Dots are drawn at twice the size and scaled down, so they come out round.
        layer = Image.new("RGB", (WIDTH * 2, BRAIN_H * 2), PANEL)
        dots = ImageDraw.Draw(layer)
        top = TITLE_H + ROBOT_H
        for (x, y), c in zip(points, mixed, strict=True):
            cx, cy = 2 * x, 2 * (y - top)
            dots.ellipse((cx - 5.2, cy - 5.2, cx + 5.2, cy + 5.2), fill=tuple(int(v) for v in c))
        canvas.paste(layer.resize((WIDTH, BRAIN_H), Image.Resampling.LANCZOS), (0, top))
        draw = ImageDraw.Draw(canvas)
        for name, x, y, anchor in (
            ("optic_lobes", 0.15, 0.025, "mm"),
            ("optic_lobes", 0.85, 0.025, "mm"),
            ("central_brain", 0.5, 0.04, "mm"),
            ("descending", 0.41, 0.465, "rm"),
            ("ascending", 0.59, 0.465, "lm"),
            ("sensory", 0.285, 0.565, "mm"),
            ("motor", 0.68, 0.76, "lm"),
            ("vnc_interneurons", 0.5, 0.995, "mm"),
        ):
            draw.text(
                tuple(origin + np.array([x, y]) * side),
                LABELS[name],
                font=label,
                fill=MUTED,
                anchor=anchor,
            )
        draw.text(
            (WIDTH - 36, TITLE_H + ROBOT_H + 30),
            "每个点是一个真实神经元\n橙色兴奋 · 蓝色抑制",
            font=label,
            fill=MUTED,
            anchor="ra",
            align="right",
        )
        draw.text(
            (WIDTH / 2, HEIGHT - FOOT_H / 2), args.footer, font=subtitle, fill=MUTED, anchor="mm"
        )
        frames.append(np.asarray(canvas))
    reader.close()
    write_video(args.output, frames, int(episode["fps"]))
    print(f"wrote {args.output} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
