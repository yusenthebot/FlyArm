"""Scanned household objects for the generalizable-grasp task: manifest, split and geometry.

Every object is one Google Scanned Objects mesh from the MJCF conversion in
``kevinzakka/mujoco_scanned_objects``, pinned to one upstream commit and one SHA-256 per file.
``scripts/fetch_grasp_objects.py`` downloads the meshes and textures and measures each object
with :func:`measure_mesh`; the result is the committed manifest ``catalog/objects.json``.
``scripts/grasp_split.py`` writes the committed train and held-out split ``catalog/split.json``.

The body frame of every object is fixed by its geometry, not by hand:

- z points up and the scanned resting pose is kept (the scans stand on z = 0);
- x is the long side and y the narrow side of the minimum-area rectangle around the footprint
  of the convex hull, so y is the axis the jaws close across;
- the origin is the centre of that oriented bounding box, so the object rests at
  ``z = size[2] / 2`` with identity orientation.

Collision uses the convex hull of the scan (MuJoCo's rule for mesh geoms), so every
geometric quantity here, including the grasp point, is measured on the hull.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

UPSTREAM_COMMIT = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"
UPSTREAM_REPO = "https://github.com/kevinzakka/mujoco_scanned_objects"
RAW_BASE = f"https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/{UPSTREAM_COMMIT}"
LICENSE = "CC-BY-4.0 (Google Scanned Objects mesh and texture); MIT (MJCF conversion)"
MESH_FILE = "model.obj"
TEXTURE_FILE = "texture.png"
TEXTURE_SMALL = "texture_256.png"  # derived by the fetch script; rendering only
FAMILIES = ("mug", "bowl", "bottle", "can", "box", "tool", "toy", "rounded")
SPLITS = ("train", "test")

# Gripper-derived limits (measured on the FlyArm Panda, see docs/GRASP_ENV.md).
PAD_GAP = 0.068  # distance between the rubber pads with the gripper fully open
MAX_NARROW = PAD_GAP - 0.013  # widest graspable narrow side: 6.5 mm per pad to approach
MAX_LONG = 0.16  # longest side, keeps an object inside the workspace at every yaw
MAX_HEIGHT = 0.10  # the fingers are 5 cm long: taller objects are only graspable by the top
MIN_HEIGHT = 0.02  # the pads cannot centre below PAD_FLOOR without touching the table
PAD_FLOOR = 0.013  # lowest pad-centre height above the table (fingers touch at ~1.1 cm)
COM_MARGIN = 0.003  # close this far above the centre of mass
PAD_OVERLAP = 0.008  # the pads must overlap at least this much of the object vertically
PALM_CLEARANCE = 0.03  # object top may rise this far above the pad centre before the palm
PALM_HALF_DEPTH = 0.035  # half the hand's extent along the object's long axis
DENSITY = 350.0  # kg / m^3 of hull volume: household items are mostly hollow or light
MASS_RANGE = (0.03, 0.30)
GEOMETRY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class GraspObject:
    """One graspable object as it is simulated.

    ``scale`` is the uniform factor applied to the scan, ``yaw`` and ``offset`` place the scaled
    scan in the body frame (``p_body = Rz(yaw) (scale p_scan) + offset``), ``size`` is the full
    oriented bounding box (long, narrow, height). ``grasp_offset`` is the grasp point along the
    long axis (the hull's centre of mass) and ``grasp_height`` the pad-centre height above the
    table at which the jaws close there.
    """

    name: str
    upstream: str
    family: str
    scale: float
    yaw: float
    offset: tuple[float, float, float]
    size: tuple[float, float, float]
    mass: float
    grasp_offset: float
    grasp_height: float
    hull_volume: float
    mesh_sha256: str
    texture_sha256: str

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: unknown family {self.family!r}")
        long, narrow, height = self.size
        if not (0 < narrow <= long + 1e-9 and narrow <= MAX_NARROW + 1e-9):
            raise ValueError(f"{self.name}: narrow side {narrow:.4f} m is not graspable")
        if long > MAX_LONG + 1e-9 or not MIN_HEIGHT - 1e-9 <= height <= MAX_HEIGHT + 1e-9:
            raise ValueError(f"{self.name}: size {self.size} is outside the gripper limits")
        if not MASS_RANGE[0] - 1e-9 <= self.mass <= MASS_RANGE[1] + 1e-9:
            raise ValueError(f"{self.name}: mass {self.mass} kg is outside {MASS_RANGE}")
        if not PAD_FLOOR - 1e-9 <= self.grasp_height <= height:
            raise ValueError(f"{self.name}: grasp height {self.grasp_height} is not on the object")
        if abs(self.grasp_offset) > long / 2:
            raise ValueError(f"{self.name}: grasp offset lies outside the object")

    @property
    def rest_z(self) -> float:
        """Height of the body origin above the table when the object rests upright."""
        return self.size[2] / 2

    @property
    def descriptor(self) -> np.ndarray:
        """The controller's shape descriptor: box (long, narrow, height) and grasp point."""
        return np.array([*self.size, self.grasp_offset, self.grasp_height])

    @property
    def source_url(self) -> str:
        return f"{UPSTREAM_REPO}/tree/{UPSTREAM_COMMIT}/models/{self.upstream}"

    def mesh_path(self, root: Path) -> Path:
        return root / self.upstream / MESH_FILE

    def texture_path(self, root: Path) -> Path:
        return root / self.upstream / TEXTURE_SMALL

    def to_json(self) -> dict[str, Any]:
        record = asdict(self)
        record["source_url"] = self.source_url
        record["license"] = LICENSE
        return record

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> GraspObject:
        fields = {
            key: value for key, value in record.items() if key not in ("source_url", "license")
        }
        fields["offset"] = tuple(fields["offset"])
        fields["size"] = tuple(fields["size"])
        return cls(**fields)


def upstream_url(upstream: str, file: str) -> str:
    """Raw download URL of one file of one upstream object, at the pinned commit."""
    return f"{RAW_BASE}/models/{upstream}/{file}"


# --------------------------------------------------------------------------- manifest and split


def data_path(name: str) -> Path:
    return Path(str(resources.files("flyarm.grasp").joinpath("catalog", name)))


def load_manifest(path: Path | None = None) -> dict[str, GraspObject]:
    """Every measured object, by name, from the committed manifest."""
    source = path or data_path("objects.json")
    records = json.loads(source.read_text())["objects"]
    objects = [GraspObject.from_json(record) for record in records]
    names = [item.name for item in objects]
    if len(set(names)) != len(names):
        raise ValueError(f"{source}: duplicate object names")
    return {item.name: item for item in objects}


def load_split(path: Path | None = None) -> dict[str, list[str]]:
    """The committed split: ``train`` and ``test`` object names plus the ``dropped`` ones."""
    source = path or data_path("split.json")
    record = json.loads(source.read_text())
    split = {key: list(record[key]) for key in (*SPLITS, "dropped")}
    train, test = set(split["train"]), set(split["test"])
    if train & test:
        raise ValueError(f"{source}: objects in both splits: {sorted(train & test)}")
    return split


def split_objects(split: str, manifest: dict[str, GraspObject] | None = None) -> list[GraspObject]:
    """The objects of one split (``train``, ``test`` or ``all`` kept objects), in split order."""
    manifest = manifest or load_manifest()
    names = load_split()
    if split == "all":
        chosen = names["train"] + names["test"]
    elif split in SPLITS:
        chosen = names[split]
    else:
        raise ValueError(f"split must be one of {(*SPLITS, 'all')}, got {split!r}")
    return [manifest[name] for name in chosen]


def stratified_split(
    objects: Sequence[GraspObject], test_fraction: float = 1 / 3, seed: int = 0
) -> dict[str, list[str]]:
    """Split by object, per family, so both splits hold every family.

    Each family of ``n`` objects sends ``max(1, round(n * test_fraction))`` of them to the
    held-out test split (and keeps at least one for training), chosen by a seeded shuffle.
    """
    generator = np.random.default_rng(seed)
    split: dict[str, list[str]] = {"train": [], "test": []}
    for family in FAMILIES:
        names = sorted(item.name for item in objects if item.family == family)
        if not names:
            continue
        if len(names) < 2:
            raise ValueError(f"family {family!r} needs at least two objects to appear in both")
        order = [names[index] for index in generator.permutation(len(names))]
        held_out = min(len(names) - 1, max(1, round(len(names) * test_fraction)))
        split["test"] += sorted(order[:held_out])
        split["train"] += sorted(order[held_out:])
    return split


# --------------------------------------------------------------------------- geometry


def load_obj_vertices(path: Path) -> np.ndarray:
    """The ``v`` records of a Wavefront OBJ file as an (n, 3) array."""
    vertices = [
        [float(value) for value in line.split()[1:4]]
        for line in path.read_text().splitlines()
        if line.startswith("v ")
    ]
    if len(vertices) < 4:
        raise ValueError(f"{path}: fewer than four vertices")
    result = np.asarray(vertices)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{path}: non-finite vertices")
    return result


def min_area_rectangle(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Angle of the long side, centre and (long, narrow) extents of the 2-D bounding rectangle."""
    hull = points[ConvexHull(points).vertices]
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        angle = math.atan2(edge[1], edge[0])
        rotation = _rot2(-angle)
        projected = hull @ rotation.T
        low, high = projected.min(0), projected.max(0)
        area = float(np.prod(high - low))
        if best is None or area < best[0] - 1e-12:
            best = (area, angle, (low + high) / 2, high - low)
    assert best is not None
    _, angle, centre, extents = best
    if extents[1] > extents[0]:  # make the first axis the long one
        angle += math.pi / 2
        centre, extents = np.array([centre[1], -centre[0]]), extents[::-1].copy()
    centre_world = _rot2(angle) @ centre
    # Keep the angle in (-pi/2, pi/2]: the rectangle has a two-fold symmetry.
    angle = math.atan2(math.sin(angle), math.cos(angle))
    if angle <= -math.pi / 2:
        angle += math.pi
    elif angle > math.pi / 2:
        angle -= math.pi
    return angle, centre_world, extents


def _rot2(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class MeshGeometry:
    scale: float
    yaw: float
    offset: tuple[float, float, float]
    size: tuple[float, float, float]
    mass: float
    grasp_offset: float
    grasp_height: float
    hull_volume: float


def measure_mesh(vertices: np.ndarray) -> MeshGeometry:
    """Scale, body frame, mass and grasp point of one scan, from its convex hull alone.

    The scale is the largest (at most 1, rounded down to 0.01) that fits the gripper limits.
    The grasp point is the hull's centre of mass projected on the long axis; the pad height
    there is half the local hull height, raised until the palm clears everything under the hand
    and never below the pad floor. Raises ``ValueError`` for an object the gripper cannot take.
    """
    angle, _, extents = min_area_rectangle(vertices[:, :2])
    height = float(np.ptp(vertices[:, 2]))
    long, narrow = float(extents[0]), float(extents[1])
    limit = min(1.0, MAX_NARROW / narrow, MAX_LONG / long, MAX_HEIGHT / height)
    scale = math.floor(limit * 100 + 1e-9) / 100
    if scale <= 0:
        raise ValueError("object is far too large for the gripper")
    rotation = rotz(-angle)
    body = (scale * vertices) @ rotation.T
    low, high = body.min(0), body.max(0)
    centre = (low + high) / 2
    body = body - centre
    size = tuple(float(value) for value in high - low)
    if size[2] < MIN_HEIGHT:
        raise ValueError(f"only {100 * size[2]:.1f} cm tall once it fits the gripper")
    hull = ConvexHull(body)
    volume, centroid = _hull_volume_centroid(body, hull)
    mass = float(np.clip(DENSITY * volume, *MASS_RANGE))
    half_long = size[0] / 2
    grasp_x = float(np.clip(centroid[0], -half_long + 0.005, half_long - 0.005))
    bottom = -size[2] / 2
    section = _section_points(body, hull, grasp_x)[:, 1:]  # (y, z) outline at the grasp plane
    top_here = float(section[:, 1].max()) - bottom
    window = _window_top(body, hull, grasp_x - PALM_HALF_DEPTH, grasp_x + PALM_HALF_DEPTH) - bottom
    low = max(PAD_FLOOR, window - PALM_CLEARANCE)
    high = top_here - PAD_OVERLAP
    if low > high:
        raise ValueError(
            f"the palm would hit the object before the pads overlap it (lowest pad centre "
            f"{low:.3f} m, local top {top_here:.3f} m)"
        )
    # Close at or above the centre of mass, so the object hangs below the jaw axis (a pendulum)
    # instead of balancing above it; only if the palm band lies wholly below it, close lower.
    above = centroid[2] - bottom + COM_MARGIN
    pad = _widest_height(section, bottom, max(low, above) if above <= high else low, high)
    return MeshGeometry(
        scale=scale,
        yaw=-angle,
        offset=tuple(float(value) for value in -centre),  # type: ignore[arg-type]
        size=size,  # type: ignore[arg-type]
        mass=round(mass, 4),
        grasp_offset=round(grasp_x, 4),
        grasp_height=round(float(pad), 4),
        hull_volume=float(volume),
    )


def body_vertices(vertices: np.ndarray, item: GraspObject | MeshGeometry) -> np.ndarray:
    """Scan vertices mapped into the object's body frame."""
    return (item.scale * vertices) @ rotz(item.yaw).T + np.asarray(item.offset)


def _hull_volume_centroid(points: np.ndarray, hull: ConvexHull) -> tuple[float, np.ndarray]:
    origin = points[hull.vertices].mean(0)
    triangles = points[hull.simplices] - origin
    volumes = np.abs(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    )
    volumes /= 6.0
    centroids = (triangles.sum(1)) / 4.0 + origin
    total = float(volumes.sum())
    return total, (volumes[:, None] * centroids).sum(0) / total


def _hull_edges(points: np.ndarray, hull: ConvexHull) -> tuple[np.ndarray, np.ndarray]:
    edges = {tuple(sorted((a, b))) for tri in hull.simplices for a, b in _pairs(tri)}
    index = np.array(sorted(edges))
    return points[index[:, 0]], points[index[:, 1]]


def _pairs(triangle: Iterable[int]) -> list[tuple[int, int]]:
    a, b, c = (int(value) for value in triangle)
    return [(a, b), (b, c), (a, c)]


def _section_points(points: np.ndarray, hull: ConvexHull, x: float) -> np.ndarray:
    """Points of the hull's cross-section with the plane ``body x = x``."""
    start, end = _hull_edges(points, hull)
    crossing = (start[:, 0] - x) * (end[:, 0] - x) <= 0
    start, end = start[crossing], end[crossing]
    span = end[:, 0] - start[:, 0]
    safe = np.where(np.abs(span) < 1e-12, 1.0, span)
    t = np.where(np.abs(span) < 1e-12, 0.0, (x - start[:, 0]) / safe)
    return start + t[:, None] * (end - start)


def _chord_width(outline: np.ndarray, z: float) -> float:
    """Width in y of a convex (y, z) outline along the horizontal line at height ``z``."""
    ring = outline[ConvexHull(outline).vertices]
    start, end = ring, np.roll(ring, -1, axis=0)
    crossing = (start[:, 1] - z) * (end[:, 1] - z) <= 0
    start, end = start[crossing], end[crossing]
    span = end[:, 1] - start[:, 1]
    t = np.where(np.abs(span) < 1e-12, 0.0, (z - start[:, 1]) / np.where(span == 0, 1.0, span))
    ys = start[:, 0] + t * (end[:, 0] - start[:, 0])
    return float(ys.max() - ys.min()) if len(ys) else 0.0


def _widest_height(outline: np.ndarray, bottom: float, low: float, high: float) -> float:
    """Lowest pad height in [low, high] where the grasp section is within 1 mm of its widest.

    Pads on a wall that narrows upward (a bottle's shoulder or neck) wedge the object down out
    of the grasp; the widest band of the section is a vertical wall or a rim that narrows
    downward, which both hold. Among near-widest heights the lowest is nearest the centre of
    mass, which the caller puts at or below ``low``.
    """
    heights = np.append(np.arange(low, high, 0.002), high)
    widths = np.array([_chord_width(outline, bottom + height) for height in heights])
    return float(heights[np.flatnonzero(widths >= widths.max() - 0.001)[0]])


def _window_top(points: np.ndarray, hull: ConvexHull, low: float, high: float) -> float:
    inside = points[hull.vertices]
    inside = inside[(inside[:, 0] >= low) & (inside[:, 0] <= high)]
    tops = [float(inside[:, 2].max())] if len(inside) else []
    for x in (low, high):
        section = _section_points(points, hull, x)
        if len(section):
            tops.append(float(section[:, 2].max()))
    return max(tops)
