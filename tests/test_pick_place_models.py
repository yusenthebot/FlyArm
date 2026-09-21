from __future__ import annotations

import numpy as np
import pytest
import torch

from flyarm.graph import Graph, shuffle_graph
from flyarm.interfaces import NeuralInterface
from flyarm.pick_place_models import RestrictedController, RestrictedPolicy


@pytest.fixture
def graph() -> Graph:
    result = Graph(
        ids=np.array([101, 102, 103, 104, 105, 106], dtype=np.int64),
        pre=np.array([0, 1, 2, 1, 3], dtype=np.int64),
        post=np.array([1, 2, 3, 4, 5], dtype=np.int64),
        contacts=np.ones(5, dtype=np.float32),
        signs=np.ones(6, dtype=np.int8),
        metadata={},
    )
    result.validate()
    return result


@pytest.fixture
def interface(graph: Graph) -> NeuralInterface:
    return NeuralInterface.bind(
        graph, np.array([101], dtype=np.int64), np.array([104], dtype=np.int64)
    )


def test_restricted_wiring_has_trainable_endpoints_but_frozen_graph(
    graph: Graph, interface: NeuralInterface
) -> None:
    policy = RestrictedPolicy("restricted_connectome", graph, interface, seed=3, internal_steps=3)
    assert policy.encoder.out_features == 1
    assert policy.readout.in_features == 1
    before = policy.adjacency.values().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    action, _ = policy(torch.randn(2, 30), policy.initial_state(2))
    action.square().sum().backward()
    optimizer.step()
    assert policy.encoder.weight.grad is not None
    assert policy.readout.weight.grad is not None
    assert "adjacency" not in dict(policy.named_parameters())
    assert not policy.adjacency.requires_grad
    assert torch.equal(policy.adjacency.values(), before)


def test_restricted_model_has_no_observation_bypass(
    graph: Graph, interface: NeuralInterface
) -> None:
    disconnected = RestrictedPolicy("restricted_disconnected", graph, interface, seed=4)
    direct_interface = NeuralInterface.bind(
        graph, np.array([101], dtype=np.int64), np.array([102], dtype=np.int64)
    )
    connected = RestrictedPolicy(
        "restricted_connectome", graph, direct_interface, seed=4, internal_steps=3
    )
    first, _ = connected(torch.zeros(1, 30), connected.initial_state(1))
    second, _ = connected(torch.ones(1, 30), connected.initial_state(1))
    assert not torch.allclose(first, second)
    disconnected_first, _ = disconnected(torch.zeros(1, 30), disconnected.initial_state(1))
    disconnected_second, _ = disconnected(torch.ones(1, 30), disconnected.initial_state(1))
    assert torch.allclose(disconnected_first, disconnected_second)


def test_controller_reset_discards_recurrent_history(
    graph: Graph, interface: NeuralInterface
) -> None:
    controller = RestrictedController(
        RestrictedPolicy("restricted_connectome", graph, interface, seed=8, internal_steps=3)
    )
    observation = np.arange(30, dtype=np.float32)
    first = controller.act(observation)
    controller.act(observation)
    controller.reset()
    assert np.allclose(controller.act(observation), first)


def test_shuffled_interface_requires_a_graph_bound_to_that_control(graph: Graph) -> None:
    shuffled = shuffle_graph(graph, seed=6, swaps_per_edge=1)
    interface = NeuralInterface.bind(
        shuffled, np.array([101], dtype=np.int64), np.array([104], dtype=np.int64)
    )
    policy = RestrictedPolicy("restricted_shuffled", shuffled, interface)
    assert policy.input_indices.tolist() == [0]
    assert policy.output_indices.tolist() == [3]


def test_mlp_matches_restricted_parameter_count_and_gru_is_nearby() -> None:
    # The preregistered 23-input/43-output interface has an 889-parameter budget.
    ids = np.arange(1000, 1066, dtype=np.int64)
    canonical_sized_graph = Graph(
        ids=ids,
        pre=np.arange(66, dtype=np.int64),
        post=np.roll(np.arange(66, dtype=np.int64), -1),
        contacts=np.ones(66, dtype=np.float32),
        signs=np.ones(66, dtype=np.int8),
        metadata={},
    )
    canonical_sized_graph.validate()
    canonical_sized_interface = NeuralInterface.bind(canonical_sized_graph, ids[:23], ids[23:])
    restricted = RestrictedPolicy(
        "restricted_connectome", canonical_sized_graph, canonical_sized_interface
    )
    mlp = RestrictedPolicy("mlp", canonical_sized_graph, canonical_sized_interface)
    gru = RestrictedPolicy("gru", canonical_sized_graph, canonical_sized_interface)
    assert restricted.trainable_parameters() == mlp.trainable_parameters()
    assert (
        abs(gru.trainable_parameters() - restricted.trainable_parameters())
        / restricted.trainable_parameters()
        < 0.05
    )
