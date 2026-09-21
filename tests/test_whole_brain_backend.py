from __future__ import annotations

import numpy as np
import pytest
import torch
from graph_fixtures import make_interface as interface_for
from graph_fixtures import make_random_graph as random_graph

from flyarm.pick_place_models import PickPlacePolicy
from flyarm.whole_brain.compiler import ConnectomePack

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal device unavailable (e.g. CI VM)", allow_module_level=True)

from flyarm.whole_brain.backend_mlx import (  # noqa: E402
    FrozenCSR,
    MlxRateBackend,
    RateDynamics,
)


def dense(pack: ConnectomePack) -> np.ndarray:
    matrix = np.zeros((pack.nodes, pack.nodes))
    matrix[pack.rows(), pack.col_idx] = pack.normalized_weights()
    return matrix


def test_kernel_and_adjoint_match_dense_reference() -> None:
    pack = ConnectomePack.from_graph(random_graph())
    matrix = FrozenCSR(pack.row_ptr, pack.col_idx, pack.normalized_weights())
    state = np.random.default_rng(1).standard_normal((pack.nodes, 5)).astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(matrix.apply(mx.array(state))), dense(pack) @ state, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(matrix.transpose_apply(mx.array(state))), dense(pack).T @ state, atol=1e-5
    )


def test_gradient_through_dynamics_matches_finite_differences() -> None:
    pack = ConnectomePack.from_graph(random_graph())
    dynamics = RateDynamics(pack, interface_for(pack))
    rng = np.random.default_rng(2)
    current = rng.standard_normal((2, dynamics.input_count)).astype(np.float32)
    readout = mx.array(rng.standard_normal((2, dynamics.output_count)).astype(np.float32))

    def loss(values):
        state = dynamics.zeros(2)
        for _ in range(4):
            state, pooled = dynamics.advance(state, values, 3)
        return (pooled * readout).sum()

    gradient = np.asarray(mx.grad(loss)(mx.array(current)))
    epsilon = 1e-3
    for index in [(0, 0), (1, 3), (0, 5)]:
        up, down = current.copy(), current.copy()
        up[index] += epsilon
        down[index] -= epsilon
        numeric = (float(loss(mx.array(up))) - float(loss(mx.array(down)))) / (2 * epsilon)
        assert gradient[index] == pytest.approx(numeric, rel=2e-2, abs=1e-4)
    assert np.any(np.abs(gradient) > 1e-6)


def test_rate_dynamics_reproduce_the_torch_subgraph_policy() -> None:
    graph = random_graph()
    torch_interface = interface_for(graph)
    policy = PickPlacePolicy(
        "restricted_connectome", graph, torch_interface, seed=0, obs_dim=7, internal_steps=3
    )
    pack = ConnectomePack.from_graph(graph)
    dynamics = RateDynamics(pack, interface_for(pack))
    obs = torch.randn(3, 7)
    state_t = policy.initial_state(3)
    state_m = dynamics.zeros(3)
    for _ in range(5):
        with torch.no_grad():
            _, state_t = policy(obs, state_t)
            current = policy.encoder((obs - policy.obs_mean) / policy.obs_scale).numpy()
        state_m, _ = dynamics.advance(state_m, mx.array(current), 3)
    np.testing.assert_allclose(np.asarray(state_m).T, state_t.numpy(), atol=2e-6)


def test_backend_is_deterministic_and_edges_off_has_no_io_path() -> None:
    pack = ConnectomePack.from_graph(random_graph())
    interface = interface_for(pack)
    current = np.random.default_rng(3).standard_normal((1, 6)).astype(np.float32)
    runs = []
    for _ in range(2):
        backend = MlxRateBackend(RateDynamics(pack, interface))
        backend.reset(1)
        for _ in range(6):
            output = backend.step(current, 3)
        runs.append(np.asarray(output.activity))
    assert np.array_equal(runs[0], runs[1])
    assert np.any(np.abs(runs[0]) > 0)

    silent = MlxRateBackend(RateDynamics(pack, interface, edges=False))
    silent.reset(1)
    for _ in range(6):
        output = silent.step(current * 100, 3)
    assert np.array_equal(np.asarray(output.activity), np.zeros((1, 5), dtype=np.float32))
    assert np.any(silent.read_nodes(interface.input_body_ids) != 0)


def test_step_rejects_currents_for_undeclared_neurons() -> None:
    pack = ConnectomePack.from_graph(random_graph())
    backend = MlxRateBackend(RateDynamics(pack, interface_for(pack)))
    backend.reset(2)
    with pytest.raises(ValueError, match="Input current"):
        backend.step(np.zeros((2, pack.nodes), dtype=np.float32), 3)
