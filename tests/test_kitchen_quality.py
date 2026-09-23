from __future__ import annotations

import numpy as np
import pytest

from flyarm.rl.kitchen_quality import (
    QualityWeights,
    action_costs,
    depth_potential,
    disturbance_potential,
    stray_contact,
)


def test_only_contact_with_the_target_body_is_allowed() -> None:
    # sensordata columns: [any, microwave, kettle]; row 0 touches only the kettle, its target;
    # row 1 touches the kettle while the microwave is the target; row 2 touches nothing;
    # row 3 has finished every task, so any contact is stray.
    data = np.array([[2.0, 0.0, 2.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    per_task = np.array([1, 2])
    target = np.array([1, 0, 0, 2])
    assert stray_contact(data, 0, per_task, target).tolist() == [False, True, False, True]


def test_depth_pays_only_the_last_stretch_and_equally_per_task() -> None:
    far = depth_potential(np.array([[2.0, 0.5]]), 0.3)
    at_threshold = depth_potential(np.array([[0.3, 0.3]]), 0.3)
    done = depth_potential(np.array([[0.0, 0.0]]), 0.3)
    assert far.tolist() == at_threshold.tolist() == [-2.0]
    assert done.tolist() == [0.0]
    halfway = depth_potential(np.array([[0.15, 0.3]]), 0.3)
    assert halfway.tolist() == pytest.approx([-1.5])


def test_disturbance_counts_only_joints_outside_the_split() -> None:
    initial = np.zeros((1, 3))
    moved = np.array([[0.4, -0.2, 5.0]])
    mask = np.array([1.0, 1.0, 0.0])  # the third joint belongs to a task
    assert disturbance_potential(moved, initial, mask).tolist() == pytest.approx([-0.6])


def test_action_costs_and_weights() -> None:
    magnitude, change = action_costs(np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]]))
    assert magnitude.tolist() == [0.5] and change.tolist() == [0.5]
    assert not QualityWeights().active
    assert QualityWeights(collision=1.0).active
    with pytest.raises(ValueError, match="collision"):
        QualityWeights(collision=-1.0)
