"""Render the two task scenes that have no live-UI screenshot: the generalizable grasp and the
complex manipulation benchmark, each at its reset state from the environment's own camera.

Read-only on the repository; writes assets/render-grasp.png and assets/render-manipulation.png.
Run from the repository root with the project environment:

    PYTHONPATH=src .venv/bin/python docs/figures/src/render_tasks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from flyarm.grasp.env import PandaGraspEnv
from flyarm.manipulation.env import PandaManipulationEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = ROOT / "assets" / "menagerie" / "franka_emika_panda" / "scene.xml"
OBJECTS = ROOT / "assets" / "objects"
SIZE = (720, 960)  # (height, width) of the raw render; the figure shows it at about 150 px
GRASP_VIEW = (1.25, 150.0, -22.0)  # distance, azimuth, elevation; aims between the base and the object
OBJECT_VIEW = (0.42, 150.0, -30.0)
CLOSE_UP = 330  # px of the close-up kept around the object
SHEET_OBJECTS = ("classic_blue_mug", "scirocco_bowl", "coq10_bottle", "sand_cup", "htp_bottle", "turquoise_bowl")
MANIP_LOOKAT, MANIP_VIEW = (0.40, 0.02, 0.22), (1.85, 160.0, -30.0)


def studio(model) -> None:
    """Neutral look that matches the live-UI renders, applied in memory (the XML is not modified):
    the reflective blue checker floor becomes a matte light-gray one with a faint checker, the blue
    skybox a flat light background, and the offscreen buffer holds the render size."""
    model.vis.global_.offheight, model.vis.global_.offwidth = SIZE
    for m in range(model.nmat):
        texture = int(model.mat_texid[m].max())
        if texture < 0 or model.mat_reflectance[m] <= 0:
            continue
        model.mat_reflectance[m] = 0.0
        model.mat_rgba[m] = [1.0, 1.0, 1.0, 1.0]
        a = model.tex_adr[texture]
        n = model.tex_width[texture] * model.tex_height[texture] * model.tex_nchannel[texture]
        pixels = model.tex_data[a : a + n].astype(np.float64)
        span = max(pixels.max() - pixels.min(), 1.0)
        model.tex_data[a : a + n] = (226 + 10 * (pixels - pixels.min()) / span).astype(np.uint8)
    for t in range(model.ntex):
        if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            n = model.tex_width[t] * model.tex_height[t] * model.tex_nchannel[t]
            model.tex_data[model.tex_adr[t] : model.tex_adr[t] + n] = 250
    model.vis.rgba.haze[:] = [0.98, 0.98, 0.98, 1.0]


def camera(lookat, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def autocrop(frame: np.ndarray, aspect: float = 4 / 3, pad: int = 24) -> np.ndarray:
    """Crop to the drawn content (anything darker than the light background, shadows included),
    padded and widened to a fixed aspect so all task thumbnails share one shape."""
    ink = (frame < 226).any(axis=2)  # darker than the faint floor and horizon
    ys, xs = np.nonzero(ink)
    top, bottom = max(ys.min() - pad, 0), min(ys.max() + pad, frame.shape[0])
    left, right = max(xs.min() - pad, 0), min(xs.max() + pad, frame.shape[1])
    h, w = bottom - top, right - left
    if w / h < aspect:  # widen around the centre
        grow = int(h * aspect) - w
        left, right = max(left - grow // 2, 0), min(right + grow - grow // 2, frame.shape[1])
    else:
        grow = int(w / aspect) - h
        top, bottom = max(top - grow // 2, 0), min(bottom + grow - grow // 2, frame.shape[0])
    return frame[top:bottom, left:right]


def save(frame, name: str) -> None:
    out = HERE / "assets" / name
    frame = autocrop(frame)
    Image.fromarray(frame).save(out, optimize=True)
    print(out, frame.shape)


def main() -> None:
    grasp_object = sys.argv[1] if len(sys.argv) > 1 else None
    template = sys.argv[2] if len(sys.argv) > 2 else None

    grasp = PandaGraspEnv(MODEL, split="train", asset_root=OBJECTS, render_mode="rgb_array", render_size=SIZE)
    print("grasp objects:", ", ".join(grasp.sim.object_names[:8]), "...")
    studio(grasp.model)
    grasp.reset(seed=3, options={"object": grasp_object} if grasp_object else None)
    target = np.asarray(grasp.sim.object_pos()[0]) * [0.5, 0.5, 0.0] + [0.0, 0.0, 0.32]
    save(grasp.render(camera(target, *GRASP_VIEW)), "render-grasp.png")

    sheet = []  # close-ups of six scanned objects: the grasp task is about many shapes, not one scene
    grasp._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False  # the arm's shadow is not the object
    for name in SHEET_OBJECTS:
        grasp.reset(seed=3, options={"object": name})
        target = np.asarray(grasp.sim.object_pos()[0]) + [0.0, 0.0, 0.02]
        frame = grasp.render(camera(target, *OBJECT_VIEW))
        cy, cx, half = frame.shape[0] // 2, frame.shape[1] // 2, CLOSE_UP // 2  # the camera aims at the object
        sheet.append(Image.fromarray(frame[cy - half : cy + half, cx - half : cx + half]))
    tile = 240
    board = Image.new("RGB", (3 * tile, 2 * tile), (250, 250, 250))
    for k, image in enumerate(sheet):
        board.paste(image.resize((tile, tile), Image.LANCZOS), ((k % 3) * tile, (k // 3) * tile))
    out = HERE / "assets" / "render-grasp-objects.png"
    board.save(out, optimize=True)
    print(out, board.size)

    manip = PandaManipulationEnv(MODEL, split="train", asset_root=OBJECTS, render_size=SIZE)
    print("templates:", ", ".join(manip.sim.templates))
    studio(manip.model)
    manip.reset(seed=3, options={"template": template} if template else None)
    save(manip.render(camera(MANIP_LOOKAT, *MANIP_VIEW)), "render-manipulation.png")


if __name__ == "__main__":
    main()
