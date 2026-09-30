from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from graph_fixtures import make_interface as interface_for
from graph_fixtures import make_random_graph as random_graph

from flyarm.whole_brain.compiler import ConnectomePack

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal device unavailable (e.g. CI VM)", allow_module_level=True)

from flyarm.whole_brain.backend_mlx import RateDynamics  # noqa: E402
from flyarm.whole_brain.lesion import LesionedDynamics, mcnemar, random_mask  # noqa: E402
from flyarm.whole_brain.policy import BrainPolicy  # noqa: E402


def _dynamics() -> RateDynamics:
    pack = ConnectomePack.from_graph(random_graph(n=40, edges=300))
    return RateDynamics(pack, interface_for(pack))


def _interface(dynamics: RateDynamics) -> np.ndarray:
    return np.concatenate([np.asarray(dynamics.input_indices), np.asarray(dynamics.output_indices)])


def _drive(dynamics: RateDynamics, batch: int = 3) -> Any:
    rng = np.random.default_rng(1)
    return mx.array(rng.normal(size=(batch, dynamics.input_count)).astype(np.float32))


def test_an_empty_lesion_is_the_intact_graph() -> None:
    dynamics = _dynamics()
    state, current = dynamics.zeros(3), _drive(dynamics)
    for _ in range(4):
        expected, pooled = dynamics.advance(state, current, 3)
        got, got_pooled = LesionedDynamics(dynamics, np.zeros(40, bool)).advance(state, current, 3)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-6)
        np.testing.assert_allclose(np.asarray(got_pooled), np.asarray(pooled), atol=1e-6)
        state = expected


def test_silenced_neurons_stay_at_zero_and_change_the_outputs() -> None:
    dynamics = _dynamics()
    mask = random_mask(20, _interface(dynamics), 40, seed=0)
    lesioned = LesionedDynamics(dynamics, mask)
    state, current = dynamics.zeros(3), _drive(dynamics)
    intact_state = state
    for _ in range(5):
        state, pooled = lesioned.advance(state, current, 3)
        intact_state, intact_pooled = dynamics.advance(intact_state, current, 3)
    assert np.abs(np.asarray(state)[mask]).max() == 0.0
    assert not np.allclose(np.asarray(pooled), np.asarray(intact_pooled))


def test_interface_neurons_cannot_be_silenced() -> None:
    dynamics = _dynamics()
    mask = np.zeros(40, bool)
    mask[int(np.asarray(dynamics.output_indices)[0])] = True
    with pytest.raises(ValueError, match="interface"):
        LesionedDynamics(dynamics, mask)


def test_state_reset_forgets_the_previous_control_step() -> None:
    dynamics = _dynamics()
    lesioned = LesionedDynamics(dynamics, reset=True)
    current = _drive(dynamics)
    warm, _ = dynamics.advance(dynamics.zeros(3), current, 3)
    _, from_warm = lesioned.advance(warm, current, 3)
    _, from_zero = dynamics.advance(dynamics.zeros(3), current, 3)
    np.testing.assert_allclose(np.asarray(from_warm), np.asarray(from_zero), atol=1e-6)


def test_a_trained_policy_runs_on_the_lesioned_graph() -> None:
    dynamics = _dynamics()
    policy = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2)
    lesioned = policy.with_dynamics("connectome", LesionedDynamics(dynamics, reset=True))  # type: ignore[arg-type]
    obs = mx.zeros((3, 4))
    action, _ = lesioned.step(obs, lesioned.initial_state(3))
    assert action.shape == (3, 2)


def test_mcnemar_counts_discordant_pairs() -> None:
    intact = np.array([1, 1, 1, 1, 0, 0], bool)
    lesioned = np.array([0, 0, 0, 1, 0, 1], bool)
    result = mcnemar(intact, lesioned)
    assert (result["lost"], result["gained"]) == (3, 1)
    assert result["p"] == pytest.approx(0.625)
