"""Build the raster inputs of fig_control_network.py from the project's own data.

- assets/cns-soma-map.png: every MaleCNS v1.0 neuron with a measured soma location (gray
  density), the declared input neurons of both interfaces (amber) and the declared output
  neurons (teal). Sensory afferents have no soma inside the CNS; as in the live UI
  (flyarm.whole_brain.live.build_whole_brain_payload) they are drawn at the contact-weighted
  centroid of their postsynaptic partners' somata. Same view as the UI: brain above the VNC.
- assets/render-pick-place.png, assets/render-kitchen.png: crops of the live-UI renders in
  docs/images (MuJoCo Menagerie Panda; FrankaKitchen-v1 scene meshes).

Read-only on data/ and docs/images; CPU only (no MLX). Run from the repository root with the project environment:

    PYTHONPATH=src .venv/bin/python docs/figures/src/make_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from flyarm.flyleg.interface import front_leg_interface
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.interface import annotation_interface

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE / "assets"
PACK = ROOT / "data" / "whole_brain" / "malecns-v1.0-c3"
ANNOTATIONS = ROOT / "data" / "raw" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
IMAGES = ROOT / "docs" / "images"

GRAY = (120, 116, 108)
AMBER = (207, 154, 34)  # figkit PAL["amber"].accent: input neurons
TEAL = (59, 142, 165)  # figkit PAL["blue"].accent: output neurons
WIDTH = 900  # output pixels; the figure shows the map at about 280 px, so this is > 3x


def soma_positions(pack: ConnectomePack, afferents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(xyz per neuron, drawn mask); afferents without a soma sit at their partners' centroid."""
    table = feather.read_table(ANNOTATIONS, columns=["bodyId", "somaLocation"]).to_pandas()
    soma = table.set_index("bodyId").somaLocation.reindex(pack.body_ids)
    measured = soma.notna().to_numpy()
    xyz = np.zeros((pack.nodes, 3))
    xyz[measured] = np.stack(soma[measured].to_numpy()).astype(np.float64)
    drawn = measured.copy()
    rows = pack.rows()
    by_source = np.argsort(pack.col_idx, kind="stable")
    source_ptr = np.concatenate(([0], np.cumsum(np.bincount(pack.col_idx, minlength=pack.nodes))))
    contacts = np.asarray(pack.contacts)
    for neuron in afferents[~measured[afferents]]:
        slots = by_source[source_ptr[neuron] : source_ptr[neuron + 1]]
        targets = rows[slots]
        keep = measured[targets]
        if not np.any(keep):
            continue
        weights = contacts[slots][keep].astype(np.float64)
        xyz[neuron] = (xyz[targets[keep]] * weights[:, None]).sum(0) / weights.sum()
        drawn[neuron] = True
    return xyz, drawn


def render_map(pack: ConnectomePack) -> None:
    b1a = annotation_interface(pack)
    leg = front_leg_interface(pack, ANNOTATIONS, include_head=True)
    in_a, out_a = b1a.resolve_indices(pack)
    in_b, out_b = leg.interface.resolve_indices(pack)
    inputs = np.union1d(in_a, in_b)
    outputs = np.union1d(out_a, out_b)
    print(f"interfaces: B1a {len(in_a)} in / {len(out_a)} out; B2 {len(in_b)} in / {len(out_b)} out")

    xyz, drawn = soma_positions(pack, inputs)
    print(f"drawn {int(drawn.sum()):,} of {pack.nodes:,} neurons")
    # UI view: screen x = -x, screen down = +z (brain above the VNC).
    sx, sy = -xyz[:, 0], xyz[:, 2]
    lo_x, hi_x = sx[drawn].min(), sx[drawn].max()
    lo_y, hi_y = sy[drawn].min(), sy[drawn].max()
    pad = 0.02 * (hi_x - lo_x)
    scale = (WIDTH - 1) / (hi_x - lo_x + 2 * pad)
    height = int(np.ceil((hi_y - lo_y + 2 * pad) * scale)) + 1
    px = (sx - lo_x + pad) * scale
    py = (sy - lo_y + pad) * scale

    internal = drawn.copy()
    internal[inputs] = False
    internal[outputs] = False
    counts, _, _ = np.histogram2d(
        py[internal], px[internal], bins=(height, WIDTH), range=((0, height), (0, WIDTH))
    )
    density = gaussian_filter(counts, 0.8)
    # gamma < 1 lifts sparse regions (the VNC) without saturating the dense optic lobes
    alpha = 0.58 * (1.0 - np.exp(-1.2 * density)) ** 0.6
    layer = np.zeros((height, WIDTH, 4), dtype=np.uint8)
    layer[..., :3] = GRAY
    layer[..., 3] = (alpha * 255).astype(np.uint8)
    image = Image.fromarray(layer, mode="RGBA")

    ss = 3  # supersample the highlighted dots
    dots = Image.new("RGBA", (WIDTH * ss, height * ss), (0, 0, 0, 0))
    pen = ImageDraw.Draw(dots)
    for group, color in ((inputs, AMBER), (outputs, TEAL)):
        keep = group[drawn[group]]
        radius = 2.4 * ss
        for x, y in zip(px[keep] * ss, py[keep] * ss, strict=True):
            pen.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 150))
        print(f"highlighted {len(keep):,} of {len(group):,}")
    image.alpha_composite(dots.resize((WIDTH, height), Image.LANCZOS))
    out = ASSETS / "cns-soma-map.png"
    image.save(out, optimize=True)
    print(out, image.size)


def crop(source: str, box: tuple[int, int, int, int], name: str) -> None:
    image = Image.open(IMAGES / source).convert("RGB").crop(box)
    out = ASSETS / name
    image.save(out, optimize=True)
    print(out, image.size)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    render_map(ConnectomePack.load(PACK))
    # Render regions of the live-UI screenshots (1600 x 913), clear of the UI overlays.
    crop("ui-pick-place.png", (652, 190, 1148, 540), "render-pick-place.png")
    crop("ui-kitchen.png", (620, 282, 1146, 632), "render-kitchen.png")


if __name__ == "__main__":
    main()
