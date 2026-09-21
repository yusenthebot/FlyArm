"""Labelled rollout frames and side-by-side comparison videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT = ImageFont.load_default(size=17)
_SMALL = ImageFont.load_default(size=14)


def annotate(frame: np.ndarray, title: str, detail: str, *, success: bool = False) -> np.ndarray:
    """Draw a title and a status line on a translucent band at the top of the frame."""
    image = Image.fromarray(frame).convert("RGBA")
    band = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)
    draw.rectangle((0, 0, image.width, 50), fill=(12, 24, 32, 170))
    draw.text((10, 6), title, font=_FONT, fill=(255, 255, 255, 255))
    color = (120, 235, 170, 255) if success else (205, 220, 228, 255)
    draw.text((10, 28), detail, font=_SMALL, fill=color)
    return np.asarray(Image.alpha_composite(image, band).convert("RGB"))


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with cast(Any, imageio.get_writer(path, fps=fps, codec="libx264", quality=7)) as writer:
        for frame in frames:
            writer.append_data(frame)


def tile(clips: list[list[np.ndarray]], columns: int = 2) -> list[np.ndarray]:
    """Grid of equally sized clips; shorter clips hold their last frame."""
    if not clips or any(not clip for clip in clips):
        raise ValueError("Every clip needs at least one frame")
    length = max(len(clip) for clip in clips)
    height, width, _ = clips[0][0].shape
    rows = -(-len(clips) // columns)
    blank = np.full((height, width, 3), 248, dtype=np.uint8)
    grid: list[np.ndarray] = []
    for index in range(length):
        cells = [clip[min(index, len(clip) - 1)] for clip in clips]
        cells += [blank] * (rows * columns - len(cells))
        lines = [
            np.concatenate(cells[r * columns : (r + 1) * columns], axis=1) for r in range(rows)
        ]
        grid.append(np.concatenate(lines, axis=0))
    return grid
