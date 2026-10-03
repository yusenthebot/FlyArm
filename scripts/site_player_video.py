"""Video in the style of the project site's player: robot, brain map and activity strip together.

    PYTHONPATH=src .venv/bin/python scripts/site_player_video.py --episode site/data/unpack \\
        --layout landscape --output docs/videos/fly-brain-player.mp4

Reads an episode recorded by scripts/record_brain_activity.py and draws every frame the way
site/index.html does: the rendered episode with the current step, the sampled neurons on the
CNS schematic coloured by their rate (orange excited, blue inhibited, grey quiet), and the
strip of each region's activity change over the whole episode (mean |r(t) - r(t-1)|, scaled
to its 98th percentile with the site's floor) with the subgoal marks and a moving playhead.
``--layout vertical`` stacks the robot, the brain map and the strip for phone screens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flyarm.video import write_video

SANS = "/System/Library/Fonts/Hiragino Sans GB.ttc"
SERIF = "/System/Library/Fonts/Supplemental/Songti.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
PAPER = (243, 245, 248)
SURFACE = (255, 255, 255)
INK = (18, 22, 28)
GRAPHITE = (90, 100, 114)
FAINT = (138, 147, 160)
HAIR = (221, 226, 232)
EXCITE = np.array([217, 72, 15])
INHIBIT = np.array([24, 100, 171])
QUIET = np.array([207, 213, 220])
GOOD = (43, 138, 62)
FLOOR = 0.004  # as site/index.html: a silent region stays pale
STRIP_ORDER = (
    "ascending",
    "central_brain",
    "optic_lobes",
    "descending",
    "vnc_interneurons",
    "motor",
    "sensory",
)
LABELS = {
    "ascending": ("上行神经元（输入）", "上行"),
    "central_brain": ("中央脑", "中央脑"),
    "optic_lobes": ("视叶", "视叶"),
    "vnc_interneurons": ("VNC 中间神经元", "VNC"),
    "sensory": ("感觉神经元", "感觉"),
    "descending": ("下行神经元（输出）", "下行"),
    "motor": ("运动神经元（输出）", "运动"),
}
MAP_LABELS = (
    ("optic_lobes", 0.15, 0.03, "mm"),
    ("optic_lobes", 0.85, 0.03, "mm"),
    ("central_brain", 0.5, 0.05, "mm"),
    ("descending", 0.41, 0.465, "rm"),
    ("ascending", 0.59, 0.465, "lm"),
    ("sensory", 0.285, 0.565, "mm"),
    ("vnc_interneurons", 0.5, 0.995, "mm"),
    ("motor", 0.68, 0.76, "lm"),
)
SKILLS = {
    "open_drawer": "开抽屉",
    "close_drawer": "关抽屉",
    "open_door": "开柜盖",
    "close_door": "合柜盖",
    "pick": "拿起来",
    "stack": "叠上去",
}
PLACES = {"region": "放到绿色区域", "shelf": "放上架子", "bin": "扔进垃圾桶"}


def describe(action: str) -> str:
    skill = action.split("(", 1)[0]
    if skill == "place":
        return PLACES.get(action.rstrip(")").split(",")[-1].strip(), "放进抽屉")
    return SKILLS.get(skill, skill)


def _ellipse(rng: np.random.Generator, n: int, cx: float, cy: float, rx: float, ry: float):
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.sqrt(rng.uniform(0, 1, n))
    return np.stack([cx + np.cos(angle) * rx * radius, cy + np.sin(angle) * ry * radius], 1)


def place(name: str, count: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-square positions on the CNS schematic, the site's layout."""
    if name == "optic_lobes":
        left = _ellipse(rng, count, 0.15, 0.24, 0.12, 0.17)
        right = _ellipse(rng, count, 0.85, 0.24, 0.12, 0.17)
        return np.where((np.arange(count) % 2 == 0)[:, None], left, right)
    if name == "central_brain":
        return _ellipse(rng, count, 0.5, 0.22, 0.2, 0.14)
    if name in ("descending", "ascending"):
        x0 = 0.43 if name == "descending" else 0.51
        return np.stack([x0 + rng.uniform(0, 0.06, count), 0.38 + rng.uniform(0, 0.17, count)], 1)
    if name == "vnc_interneurons":
        return _ellipse(rng, count, 0.5, 0.76, 0.13, 0.18)
    if name == "motor":
        angle = rng.uniform(0, 2 * np.pi, count)
        scale = 0.9 + rng.uniform(0, 0.12, count)
        return np.stack(
            [0.5 + np.cos(angle) * 0.15 * scale, 0.76 + np.sin(angle) * 0.205 * scale], 1
        )
    return np.stack([0.25 + rng.uniform(0, 0.07, count), 0.6 + rng.uniform(0, 0.32, count)], 1)


class Layout:
    """Boxes (x, y, w, h) of every panel for one canvas size."""

    def __init__(self, kind: str) -> None:
        if kind == "landscape":
            self.size = (1920, 1088)  # H.264 needs multiples of 16
            self.title = (40, 26)
            card = (32, 104, 1856, 952)
            self.video = (card[0], card[1], 1180, 664)
            self.map = (card[0] + 1180, card[1], card[2] - 1180, 664)
            self.strip = (card[0], card[1] + 664, card[2], card[3] - 664)
        else:
            self.size = (1088, 1440)
            self.title = (32, 24)
            card = (24, 120, 1040, 1266)
            video_h = round(1040 * 9 / 16)
            self.video = (card[0], card[1], 1040, video_h)
            self.map = (card[0], card[1] + video_h, 1040, 440)
            self.strip = (card[0], card[1] + video_h + 440, 1040, card[3] - video_h - 440)
        self.card = card
        # Strip margins (top for the subgoal marks, bottom for the time) and label sizes.
        self.strip_top, self.strip_bottom = (56, 54) if kind == "landscape" else (44, 42)
        self.label_px, self.row_px = (20, 19) if kind == "landscape" else (17, 15)


def change_envelopes(rates: np.ndarray, groups: list[dict]) -> dict[str, np.ndarray]:
    out, start = {}, 0
    for group in groups:
        block = rates[:, start : start + group["count"]]
        change = np.zeros(len(block))
        change[1:] = np.abs(np.diff(block, axis=0)).mean(1)
        out[group["name"]] = change
        start += group["count"]
    return out


def strip_image(layout: Layout, envelopes, marks, frames: int, fonts) -> tuple[Image.Image, tuple]:
    """The static part of the strip and the plot box (x0, y0, x1, y1) in canvas pixels."""
    x, y, w, h = layout.strip
    image = Image.new("RGB", (w, h), SURFACE)
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, w, 0), fill=HAIR, width=2)
    left, top, bottom = 230, layout.strip_top, h - layout.strip_bottom
    right = w - 28
    row_h = (bottom - top) / len(STRIP_ORDER)
    for k, name in enumerate(STRIP_ORDER):
        env = envelopes[name]
        scale = max(float(np.percentile(env, 98)), FLOOR)
        m = np.clip(env / scale, 0, 1) ** 0.7
        y0 = top + k * row_h
        draw.text(
            (left - 12, y0 + row_h / 2),
            LABELS[name][1],
            font=fonts["small"],
            fill=GRAPHITE,
            anchor="rm",
        )
        cols = right - left
        idx = np.minimum(frames - 1, (np.arange(cols) / cols * frames).astype(int))
        colours = (np.array(PAPER) + (EXCITE - np.array(PAPER)) * m[idx, None]).astype(np.uint8)
        band = np.repeat(colours[None], max(1, int(row_h) - 2), axis=0)
        image.paste(Image.fromarray(band), (left, int(y0) + 1))
    last = -1e9
    for frame, action in marks:
        mx = left + frame / frames * (right - left)
        draw.line((mx, top, mx, bottom), fill=(170, 176, 184), width=1)
        if mx - last > 110:
            draw.text(
                (mx + 6, top - 22),
                describe(action),
                font=fonts["small"],
                fill=GRAPHITE,
                anchor="lm",
            )
            last = mx
    cx, cy = 60, (top + bottom) / 2
    draw.ellipse((cx - 34, cy - 34, cx + 34, cy + 34), fill=PAPER, outline=HAIR, width=2)
    draw.rectangle((cx - 11, cy - 13, cx - 4, cy + 13), fill=INK)
    draw.rectangle((cx + 4, cy - 13, cx + 11, cy + 13), fill=INK)
    draw.text((w - 28, h - 26), "活动变化 · 按脑区", font=fonts["small"], fill=FAINT, anchor="rm")
    return image, (x + left, y + top, x + right, y + bottom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--condition", default="intact")
    parser.add_argument("--layout", choices=("landscape", "vertical"), default="landscape")
    parser.add_argument("--title", default="一整颗冻结的果蝇大脑，控制机械臂")
    parser.add_argument("--note", default="没见过的物体 · 7 步任务 · 166,700 个神经元一根线都不改")
    parser.add_argument("--link", default="yusenthebot.github.io/FlyArm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layout = Layout(args.layout)
    episode = json.loads((args.episode / "episode.json").read_text())
    info = episode["conditions"][args.condition]
    groups = json.loads((args.episode / "neurons.json").read_text())["groups"]
    total = sum(g["count"] for g in groups)
    raw = np.frombuffer((args.episode / f"{args.condition}.bin").read_bytes(), np.uint8)
    scaled = raw.reshape(-1, total).astype(np.float32) / 127.5 - 1.0
    rates = np.sign(scaled) * scaled**2
    frames = len(rates)
    gains = np.concatenate(
        [np.full(g["count"], 1.0 if g["name"] == "ascending" else 10.0) for g in groups]
    )
    rng = np.random.default_rng(3)
    unit = np.concatenate([place(g["name"], g["count"], rng) for g in groups])
    labels = info["labels"]
    marks, previous = [], None
    for f, text in enumerate(labels):
        count, _, action = text.partition(" ")
        if count != previous and action and action != "done":
            marks.append((f, action))
        previous = count

    fonts = {
        "title": ImageFont.truetype(SERIF, 46 if args.layout == "landscape" else 44, index=1),
        "note": ImageFont.truetype(SANS, 24),
        "chip": ImageFont.truetype(SANS, 26),
        "label": ImageFont.truetype(SANS, layout.label_px),
        "row": ImageFont.truetype(SANS, layout.row_px),
        "small": ImageFont.truetype(SANS, 19),
        "mono": ImageFont.truetype(MONO, 22),
        "mono_small": ImageFont.truetype(MONO, 18),
    }
    strip, plot = strip_image(layout, change_envelopes(rates, groups), marks, frames, fonts)
    mx, my, mw, mh = layout.map
    side = min(mw - 60, mh - 110)
    origin = np.array([mx + (mw - side) / 2, my + 30 + (mh - 90 - side) / 2])
    points = origin + unit * side

    mask = Image.new("L", layout.size, 0)
    cx, cy, cw, ch = layout.card
    ImageDraw.Draw(mask).rounded_rectangle((cx - 1, cy - 1, cx + cw, cy + ch), radius=22, fill=255)
    card_mask = np.asarray(mask) > 0
    title_box = np.zeros(layout.size[::-1], dtype=bool)
    title_box[: cy - 2] = True
    card_mask |= title_box  # the title and link above the card stay
    footer_box = np.zeros(layout.size[::-1], dtype=bool)
    footer_box[cy + ch + 2 :] = True
    card_mask |= footer_box
    reader = imageio.get_reader(args.episode / f"{args.condition}.mp4")
    out = []
    for index, robot in enumerate(reader):
        if index >= frames:
            break
        canvas = Image.new("RGB", layout.size, PAPER)
        draw = ImageDraw.Draw(canvas)
        tx, ty = layout.title
        draw.text((tx, ty), args.title, font=fonts["title"], fill=INK)
        if args.layout == "landscape":
            draw.text(
                (layout.size[0] - 40, ty + 30),
                args.link,
                font=fonts["mono"],
                fill=GRAPHITE,
                anchor="rm",
            )
        else:
            draw.text((tx + 2, ty + 62), args.note, font=fonts["note"], fill=GRAPHITE)
        cx, cy, cw, ch = layout.card
        draw.rounded_rectangle(
            (cx - 1, cy - 1, cx + cw, cy + ch), radius=22, fill=SURFACE, outline=HAIR, width=2
        )
        vx, vy, vw, vh = layout.video
        canvas.paste(Image.fromarray(robot).resize((vw, vh), Image.Resampling.LANCZOS), (vx, vy))
        text = labels[min(index, len(labels) - 1)]
        count, _, action = text.partition(" ")
        done, total_steps = (int(v) for v in count.split("/"))
        last = index >= frames - 2
        chips = (
            [f"完成 {done}/{total_steps} 步", "成功" if info["success"] else "失败"]
            if last
            else [f"第 {min(done + 1, total_steps)}/{total_steps} 步", describe(action)]
        )
        x = vx + 22
        for chip in chips:
            box = draw.textbbox((0, 0), chip, font=fonts["chip"])
            wbox = box[2] - box[0] + 36
            colour = GOOD if last and info["success"] else INK
            draw.rounded_rectangle(
                (x, vy + 20, x + wbox, vy + 70),
                radius=25,
                fill=(255, 255, 255),
                outline=HAIR,
                width=2,
            )
            draw.text((x + wbox / 2, vy + 45), chip, font=fonts["chip"], fill=colour, anchor="mm")
            x += wbox + 12
        # brain map, drawn at twice the size and scaled down so the dots come out round
        layer = Image.new("RGB", (mw * 2, mh * 2), SURFACE)
        dots = ImageDraw.Draw(layer)
        v = rates[index]
        mag = np.clip(np.abs(v) * gains, 0, 1)[:, None]
        colour = np.where((v >= 0)[:, None], EXCITE, INHIBIT)
        mixed = (QUIET + (colour - QUIET) * mag).astype(int)
        for (px, py), c in zip(points, mixed, strict=True):
            qx, qy = 2 * (px - mx), 2 * (py - my)
            dots.ellipse((qx - 5.4, qy - 5.4, qx + 5.4, qy + 5.4), fill=tuple(int(t) for t in c))
        for cxe, cye, rxe, rye in (
            (0.15, 0.24, 0.13, 0.18),
            (0.85, 0.24, 0.13, 0.18),
            (0.5, 0.22, 0.21, 0.15),
            (0.5, 0.76, 0.16, 0.22),
        ):
            ex, ey = 2 * (origin[0] - mx + cxe * side), 2 * (origin[1] - my + cye * side)
            dots.ellipse(
                (
                    ex - 2 * rxe * side,
                    ey - 2 * rye * side,
                    ex + 2 * rxe * side,
                    ey + 2 * rye * side,
                ),
                outline=HAIR,
                width=3,
            )
        canvas.paste(layer.resize((mw, mh), Image.Resampling.LANCZOS), (mx, my))
        draw = ImageDraw.Draw(canvas)
        if args.layout == "landscape":
            draw.line((mx, my, mx, my + mh), fill=HAIR, width=2)
        else:
            draw.line((mx, my, mx + mw, my), fill=HAIR, width=2)
        for name, lx, ly, anchor in MAP_LABELS:
            draw.text(
                tuple(origin + np.array([lx, ly]) * side),
                LABELS[name][0],
                font=fonts["label"],
                fill=GRAPHITE,
                anchor=anchor,
            )
        ky = my + mh - 34
        draw.text((mx + 24, ky), "抑制", font=fonts["small"], fill=GRAPHITE, anchor="lm")
        ramp = np.linspace(0, 1, 130)
        ramp_cols = np.array(
            [
                INHIBIT + (QUIET - INHIBIT) * (t * 2)
                if t < 0.5
                else QUIET + (EXCITE - QUIET) * ((t - 0.5) * 2)
                for t in ramp
            ]
        ).astype(np.uint8)
        canvas.paste(Image.fromarray(np.repeat(ramp_cols[None], 9, axis=0)), (mx + 70, int(ky) - 4))
        draw.text((mx + 210, ky), "兴奋", font=fonts["small"], fill=GRAPHITE, anchor="lm")
        draw.text(
            (mx + mw - 24, ky), "3,200 个神经元", font=fonts["small"], fill=GRAPHITE, anchor="rm"
        )
        # strip and playhead
        canvas.paste(strip, layout.strip[:2])
        draw = ImageDraw.Draw(canvas)
        x0, y0, x1, y1 = plot
        px = x0 + index / frames * (x1 - x0)
        draw.rectangle((px - 2, y0 - 8, px + 2, y1), fill=INK)
        draw.ellipse((px - 8, y0 - 16, px + 8, y0), fill=INK)
        seconds = index / episode["fps"]
        draw.text(
            (x0, layout.strip[1] + layout.strip[3] - 26),
            f"{seconds:4.1f} / {frames / episode['fps']:.1f} s",
            font=fonts["mono_small"],
            fill=FAINT,
            anchor="lm",
        )
        if args.layout == "vertical":
            draw.text(
                (layout.size[0] / 2, layout.size[1] - 12),
                args.link,
                font=fonts["mono"],
                fill=GRAPHITE,
                anchor="mb",
            )
        # Round the card's corners over everything pasted into it, then redraw its outline.
        frame = np.asarray(canvas).copy()
        frame[~card_mask] = PAPER
        canvas = Image.fromarray(frame)
        ImageDraw.Draw(canvas).rounded_rectangle(
            (cx - 1, cy - 1, cx + cw, cy + ch), radius=22, outline=HAIR, width=2
        )
        out.append(np.asarray(canvas))
    reader.close()
    write_video(args.output, out, int(episode["fps"]))
    print(f"wrote {args.output} ({len(out)} frames, {layout.size[0]}x{layout.size[1]})", flush=True)


if __name__ == "__main__":
    main()
