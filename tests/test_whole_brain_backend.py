from __future__ import annotations

import numpy as np
import pytest
import torch
from graph_fixtures import make_interface as interface_for
from graph_fixtures import make_random_graph as random_graph

from flyarm.graph import Graph
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


def torch_rate_steps(
    graph: Graph, inputs: np.ndarray, state: torch.Tensor, current: torch.Tensor, steps: int
) -> torch.Tensor:
    """Independent sparse torch reference: the original 256-neuron subgraph update."""
    nodes = len(graph.ids)
    adjacency = torch.sparse_coo_tensor(
        torch.as_tensor(np.stack((graph.post, graph.pre)), dtype=torch.int64),
        torch.from_numpy(graph.normalized_weights()),
        (nodes, nodes),
        check_invariants=True,
    ).coalesce()
    drive = state.new_zeros(state.shape)
    drive[:, torch.from_numpy(inputs)] = current
    for _ in range(steps):
        recurrent = torch.sparse.mm(adjacency, state.T).T
        state = 0.5 * state + 0.5 * torch.tanh(drive + 0.8 * recurrent)
    return state


def test_rate_dynamics_reproduce_the_torch_subgraph_policy() -> None:
    graph = random_graph()
    inputs, _ = interface_for(graph).resolve_indices(graph)
    pack = ConnectomePack.from_graph(graph)
    dynamics = RateDynamics(pack, interface_for(pack))
    current = torch.randn(3, len(inputs), generator=torch.Generator().manual_seed(0))
    state_t = torch.zeros(3, len(graph.ids))
    state_m = dynamics.zeros(3)
    for _ in range(5):
        state_t = torch_rate_steps(graph, inputs, state_t, current, 3)
        state_m, _ = dynamics.advance(state_m, mx.array(current.numpy()), 3)
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


def test_pooled_readout_is_the_window_mean_and_matches_last_state_for_one_step() -> None:
    pack = ConnectomePack.from_graph(random_graph())
    interface = interface_for(pack)
    dynamics = RateDynamics(pack, interface)
    outputs = interface.resolve_indices(pack)[1]
    current = mx.array(np.random.default_rng(4).standard_normal((2, 6)).astype(np.float32))
    state = dynamics.zeros(2)
    window = []
    for _ in range(3):
        state, _ = dynamics.advance(state, current, 1)
        window.append(np.asarray(state)[outputs].T)
    _, pooled = dynamics.advance(dynamics.zeros(2), current, 3)
    np.testing.assert_allclose(np.asarray(pooled), np.mean(window, axis=0), atol=1e-6)
    last_state, single = dynamics.advance(dynamics.zeros(2), current, 1)
    np.testing.assert_array_equal(np.asarray(single), np.asarray(last_state)[outputs].T)


def test_float16_weights_stay_close_to_float32() -> None:
    pack = ConnectomePack.from_graph(random_graph())
    state = np.random.default_rng(5).standard_normal((pack.nodes, 3)).astype(np.float32)
    exact = FrozenCSR(pack.row_ptr, pack.col_idx, pack.normalized_weights())
    half = FrozenCSR(pack.row_ptr, pack.col_idx, pack.normalized_weights(), "float16")
    np.testing.assert_allclose(
        np.asarray(half.apply(mx.array(state))), np.asarray(exact.apply(mx.array(state))), atol=5e-3
    )
    with pytest.raises(ValueError, match="weight_dtype"):
        FrozenCSR(pack.row_ptr, pack.col_idx, pack.normalized_weights(), "int8")
