from __future__ import annotations

import numpy as np
import pytest

from flyarm.manipulation.detour import (
    TRANSIT_Z,
    Detour,
    detour_action,
    sample_detours,
)


def _detour(target: list[float], yaw: float = 0.0) -> Detour:
    return Detour(steps=np.array([30]), target=np.array([target]), yaw=np.array([yaw]))


def test_a_low_hand_far_from_the_target_rises_before_moving_across() -> None:
    action = detour_action(
        np.array([[0.3, 0.0, 0.2]]), np.zeros(1), _detour([0.6, 0.3, 0.2]), np.ones(1)
    )
    assert action[0, 0] == 0.0 and action[0, 1] == 0.0 and action[0, 2] == 1.0


def test_at_transit_height_it_moves_across_then_descends_near_the_target() -> None:
    across = detour_action(
        np.array([[0.3, 0.0, TRANSIT_Z]]), np.zeros(1), _detour([0.6, 0.3, 0.2]), np.ones(1)
    )
    assert across[0, 0] == 1.0 and across[0, 1] == 1.0 and abs(across[0, 2]) < 1e-9
    near = detour_action(
        np.array([[0.6, 0.29, TRANSIT_Z]]), np.zeros(1), _detour([0.6, 0.3, 0.2]), np.ones(1)
    )
    assert near[0, 2] == -1.0


def test_the_heading_turns_toward_the_target_and_the_grip_is_kept() -> None:
    action = detour_action(
        np.array([[0.6, 0.3, 0.2]]),
        np.array([0.5]),
        _detour([0.6, 0.3, 0.2], yaw=0.49),
        np.array([-1.0]),
    )
    np.testing.assert_allclose(action[0, :3], 0.0, atol=1e-9)
    assert action[0, 3] == pytest.approx(-0.2) and action[0, 4] == -1.0


def test_sampling_respects_the_share_and_the_length_range() -> None:
    none = sample_detours(100, 0.0, (20, 60), np.random.default_rng(0))
    assert not none.steps.any()
    every = sample_detours(1000, 1.0, (20, 60), np.random.default_rng(0))
    assert every.steps.min() >= 20 and every.steps.max() <= 60
    half = sample_detours(4000, 0.5, (20, 60), np.random.default_rng(1))
    assert 0.45 < (half.steps > 0).mean() < 0.55
    with pytest.raises(ValueError):
        sample_detours(10, 1.5, (20, 60), np.random.default_rng(0))
