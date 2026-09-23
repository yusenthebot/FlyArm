"""Download and measure the scanned household objects of the generalizable-grasp task.

Each object is one mesh (``model.obj``) and one texture (``texture.png``) from
github.com/kevinzakka/mujoco_scanned_objects, fetched at a pinned upstream commit and checked
against the SHA-256 recorded in the committed manifest ``src/flyarm/grasp/catalog/objects.json``.
The texture is also saved at 256 x 256 (``texture_256.png``), which is what the scene renders.

    PYTHONPATH=src .venv/bin/python scripts/fetch_grasp_objects.py            # fetch and verify
    PYTHONPATH=src .venv/bin/python scripts/fetch_grasp_objects.py --measure  # rebuild manifest

``--measure`` fetches every candidate below, measures it with flyarm.grasp.objects.measure_mesh
and rewrites the manifest (objects the gripper cannot take are listed under ``rejected``).
Meshes and textures are CC-BY 4.0 (Google Scanned Objects), the MJCF conversion is MIT; see
docs/GRASP_ENV.md and THIRD_PARTY.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx
from PIL import Image

from flyarm.grasp.objects import (
    LICENSE,
    MESH_FILE,
    TEXTURE_FILE,
    TEXTURE_SMALL,
    UPSTREAM_COMMIT,
    UPSTREAM_REPO,
    GraspObject,
    data_path,
    load_obj_vertices,
    measure_mesh,
    upstream_url,
)

DEFAULT_ROOT = Path("assets/objects")
MAX_BYTES = 16 * 1024 * 1024  # every pinned file is under 2 MB; refuse anything absurd
TEXTURE_SIZE = 256

# (family, short name, upstream directory). The short name is what configs and results use.
CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("mug", "paper_cup", "ACE_Coffee_Mug_Kristen_16_oz_cup"),
    ("mug", "classic_blue_mug", "Cole_Hardware_Mug_Classic_Blue"),
    ("mug", "white_yellow_mug", "Room_Essentials_Mug_White_Yellow"),
    ("mug", "bead_mug", "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White"),
    ("mug", "sand_cup", "Ecoforms_Cup_B4_SAN"),
    ("bowl", "scirocco_bowl", "Cole_Hardware_Bowl_Scirocco_YellowBlue"),
    ("bowl", "glazed_ramekin", "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup"),
    ("bowl", "cereal_bowl", "Threshold_Bead_Cereal_Bowl_White"),
    ("bowl", "turquoise_bowl", "Room_Essentials_Bowl_Turquiose"),
    ("bottle", "coq10_bottle", "CoQ10"),
    ("bottle", "htp_bottle", "5_HTP"),
    ("bottle", "sprinkles_jar", "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold"),
    ("bottle", "tonic_bottle", "Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh"),
    ("bottle", "supplement_tub", "Twinlab_Nitric_Fuel"),
    ("can", "coffee_can", "Don_Franciscos_Gourmet_Coffee_Medium_Decaf_100_Colombian_12_oz_340_g"),
    ("can", "xylitol_tub", "Xyli_Pure_Xylitol"),
    (
        "can",
        "instant_coffee_jar",
        "Nescafe_Tasters_Choice_Instant_Coffee_Decaf_House_Blend_Light_7_oz",
    ),
    (
        "can",
        "cocoa_canister",
        "Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister",
    ),
    ("can", "camera_lens", "Nikon_1_AW1_w11275mm_Lens_Silver"),
    ("box", "crayon_box", "Crayola_Crayons_24_count"),
    ("box", "caplet_box", "Phillips_Caplets_Size_24"),
    ("box", "raisinets_box", "Nestle_Raisinets_Milk_Chocolate_35_oz_992_g"),
    ("box", "lipstick_box", "Perricone_MD_No_Lipstick_Lipstick"),
    ("box", "cereal_box", "Vans_Cereal_Honey_Nut_Crunch_11_oz_box"),
    ("box", "hair_color_box", "Just_For_Men_ShampooIn_Haircolor_Jet_Black_60"),
    ("tool", "hammer", "Cole_Hardware_Hammer_Black"),
    ("tool", "screwdriver", "Craftsman_Grip_Screwdriver_Phillips_Cushion"),
    ("tool", "can_opener", "OXO_Soft_Works_Can_Opener_SnapLock"),
    ("tool", "lime_squeezer", "Focus_8643_Lime_Squeezer_10x35x188_Enamelled_Aluminum_Light"),
    ("tool", "flashlight", "HeavyDuty_Flashlight"),
    ("tool", "tape_roll", "Shurtape_Tape_Purple_CP28"),
    ("tool", "tape_measure", "SNAIL_MEASURING_TAPE"),
    ("tool", "hand_bell", "Cole_Hardware_School_Bell_Solid_Brass_38"),
    ("toy", "android_figure", "Android_Figure_Orange"),
    ("toy", "mario_figure", "Nintendo_Mario_Action_Figure"),
    ("toy", "yoshi_figure", "Nintendo_Yoshi_Action_Figure"),
    ("toy", "triceratops", "Great_Dinos_Triceratops_Toy"),
    ("toy", "school_bus", "SCHOOL_BUS"),
    ("toy", "henry_engine", "Thomas_Friends_Woodan_Railway_Henry"),
    ("toy", "lion_figure", "Schleich_Lion_Action_Figure"),
    ("rounded", "pineapple_maraca", "PINEAPPLE_MARACA_6_PCSSET"),
    ("rounded", "fruit_basket_toy", "Squirt_Strain_Fruit_Basket"),
    ("rounded", "ladybug_bead", "LADYBUG_BEAD"),
    ("rounded", "rubber_chew_toy", "Kong_Puppy_Teething_Rubber_Small_Pink"),
    ("rounded", "whale_whistle", "WHALE_WHISTLE_6PCS_SET"),
    ("rounded", "bird_rattle", "BIRD_RATTLE"),
)


def _digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(client: httpx.Client, url: str, target: Path, sha256: str | None) -> str:
    """Fetch ``url`` to ``target`` unless a copy with the pinned digest is already there."""
    if target.is_file():
        digest = _digest(target)
        if sha256 is None or digest == sha256:
            return digest
        raise ValueError(f"{target} does not match its pinned SHA-256; delete it and retry")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    written = 0
    with client.stream("GET", url) as response, partial.open("wb") as output:
        response.raise_for_status()
        for chunk in response.iter_bytes(2**20):
            written += len(chunk)
            if written > MAX_BYTES:
                partial.unlink(missing_ok=True)
                raise ValueError(f"{url}: download exceeded {MAX_BYTES} bytes")
            output.write(chunk)
    digest = _digest(partial)
    if sha256 is not None and digest != sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{url}: SHA-256 {digest} does not match the pinned {sha256}")
    partial.rename(target)
    return digest


def shrink_texture(source: Path, target: Path) -> None:
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    with Image.open(source) as image:
        image.convert("RGB").resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.Resampling.LANCZOS).save(
            target
        )


def fetch_pinned(root: Path, client: httpx.Client) -> None:
    manifest = json.loads(data_path("objects.json").read_text())
    for record in manifest["objects"]:
        item = GraspObject.from_json(record)
        download(
            client, upstream_url(item.upstream, MESH_FILE), item.mesh_path(root), item.mesh_sha256
        )
        source = root / item.upstream / TEXTURE_FILE
        texture = upstream_url(item.upstream, TEXTURE_FILE)
        download(client, texture, source, item.texture_sha256)
        shrink_texture(source, item.texture_path(root))
        print(f"{item.name:22s} {item.family:7s} {item.mesh_path(root)}")


def measure_all(root: Path, client: httpx.Client, output: Path) -> None:
    pinned: dict[str, dict] = {}
    if output.is_file():
        previous = json.loads(output.read_text())
        pinned = {record["upstream"]: record for record in previous["objects"]}
        pinned |= {record["upstream"]: record for record in previous.get("rejected", [])}
    objects, rejected = [], []
    for family, name, upstream in CANDIDATES:
        known = pinned.get(upstream, {})
        mesh = root / upstream / MESH_FILE
        mesh_sha = download(
            client, upstream_url(upstream, MESH_FILE), mesh, known.get("mesh_sha256")
        )
        source = root / upstream / TEXTURE_FILE
        texture = upstream_url(upstream, TEXTURE_FILE)
        texture_sha = download(client, texture, source, known.get("texture_sha256"))
        shrink_texture(source, root / upstream / TEXTURE_SMALL)
        try:
            geometry = measure_mesh(load_obj_vertices(mesh))
        except ValueError as error:
            rejected.append(
                {
                    "name": name,
                    "upstream": upstream,
                    "family": family,
                    "reason": str(error),
                    "mesh_sha256": mesh_sha,
                    "texture_sha256": texture_sha,
                }
            )
            print(f"{name:22s} rejected: {error}")
            continue
        item = GraspObject(
            name=name,
            upstream=upstream,
            family=family,
            mesh_sha256=mesh_sha,
            texture_sha256=texture_sha,
            **{key: value for key, value in vars(geometry).items() if key != "hull_volume"},
            hull_volume=round(geometry.hull_volume, 9),
        )
        objects.append(item.to_json())
        size = " x ".join(f"{100 * value:.1f}" for value in item.size)
        print(f"{name:22s} {family:7s} scale {item.scale:.2f} size {size} cm mass {item.mass:.3f}")
    record = {
        "source": UPSTREAM_REPO,
        "commit": UPSTREAM_COMMIT,
        "license": LICENSE,
        "units": "metres and kilograms; size is (long, narrow, height) of the oriented box",
        "objects": objects,
        "rejected": rejected,
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {output}: {len(objects)} objects, {len(rejected)} rejected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--measure", action="store_true", help="re-measure and rewrite manifest")
    arguments = parser.parse_args()
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        if arguments.measure:
            measure_all(arguments.root, client, data_path("objects.json"))
        else:
            fetch_pinned(arguments.root, client)


if __name__ == "__main__":
    main()
