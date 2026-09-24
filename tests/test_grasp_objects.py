from __future__ import annotations

import math

import numpy as np
import pytest

from flyarm.grasp.objects import (
    FAMILIES,
    MAX_NARROW,
    GraspObject,
    body_vertices,
    load_manifest,
    load_split,
    measure_mesh,
    rotz,
    split_objects,
    stratified_split,
)


def _box(size: tuple[float, float, float], yaw: float, shift: tuple[float, float]) -> np.ndarray:
    """The 8 corners of a box resting on z = 0, turned by ``yaw`` and moved by ``shift``."""
    half = np.array(size) / 2
    corners = np.array(
        [
            [sx * half[0], sy * half[1], (sz + 1) * half[2]]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    return corners @ rotz(yaw).T + np.array([*shift, 0.0])


def test_manifest_objects_are_valid_and_licensed() -> None:
    manifest = load_manifest()
    assert 30 <= len(manifest) <= 50
    for name, item in manifest.items():
        assert item.name == name
        record = item.to_json()
        assert "CC-BY-4.0" in record["license"]
        assert record["source_url"].startswith(
            "https://github.com/kevinzakka/mujoco_scanned_objects"
        )
        assert len(item.mesh_sha256) == 64 and len(item.texture_sha256) == 64
        assert GraspObject.from_json(record) == item


def test_split_is_disjoint_complete_and_stratified() -> None:
    manifest = load_manifest()
    split = load_split()
    train, test, dropped = set(split["train"]), set(split["test"]), set(split["dropped"])
    assert not train & test and not (train | test) & dropped
    assert train | test | dropped == set(manifest)
    for family in FAMILIES:
        assert any(manifest[name].family == family for name in train), family
        assert any(manifest[name].family == family for name in test), family
    assert 0.25 <= len(test) / (len(train) + len(test)) <= 0.40
    assert [item.name for item in split_objects("train")] == split["train"]
    assert len(split_objects("all")) == len(train) + len(test)


def test_stratified_split_is_deterministic_and_keeps_families_on_both_sides() -> None:
    objects = list(load_manifest().values())
    first, second = stratified_split(objects, seed=3), stratified_split(objects, seed=3)
    assert first == second
    assert not set(first["train"]) & set(first["test"])
    assert stratified_split(objects, seed=4) != first


@pytest.mark.parametrize("yaw", [0.0, 0.4, -1.2, 2.0])
def test_measure_mesh_recovers_a_turned_box(yaw: float) -> None:
    size = (0.08, 0.04, 0.06)
    geometry = measure_mesh(_box(size, yaw, (0.3, -0.2)))
    assert geometry.scale == 1.0
    assert np.allclose(geometry.size, size, atol=1e-9)
    body = body_vertices(_box(size, yaw, (0.3, -0.2)), geometry)
    assert np.allclose(body.min(0), -np.array(size) / 2, atol=1e-9)
    assert np.allclose(body.max(0), np.array(size) / 2, atol=1e-9)
    assert abs(geometry.grasp_offset) < 1e-6  # a uniform box balances at its centre
    # The pads close at or above the centre of mass, so the box hangs below the jaw axis.
    assert geometry.grasp_height >= size[2] / 2
    assert math.isclose(geometry.mass, 350.0 * np.prod(size), rel_tol=1e-3)


def test_measure_mesh_scales_an_object_too_wide_for_the_gripper() -> None:
    geometry = measure_mesh(_box((0.12, 0.10, 0.06), 0.0, (0.0, 0.0)))
    assert geometry.scale == 0.55
    assert geometry.size[1] <= MAX_NARROW + 1e-12


def test_measure_mesh_rejects_what_the_fingers_cannot_reach() -> None:
    with pytest.raises(ValueError, match="tall"):
        measure_mesh(_box((0.30, 0.10, 0.02), 0.0, (0.0, 0.0)))  # a plate
