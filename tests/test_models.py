from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from flyarm.config import ExperimentConfig
from flyarm.graph import Graph
from flyarm.models import Controller, Policy


@pytest.fixture
def graph() -> Graph:
    result = Graph(
        ids=np.array([1, 2, 3, 4], dtype=np.int64),
        pre=np.array([0, 1, 2, 3], dtype=np.int64),
        post=np.array([1, 2, 3, 0], dtype=np.int64),
        contacts=np.array([1, 2, 3, 4], dtype=np.float32),
        signs=np.array([1, -1, 1, 0], dtype=np.int8),
        metadata={},
    )
    result.validate()
    return result


def test_encoder_and_readout_have_gradients_but_adjacency_is_frozen(
    graph: Graph,
) -> None:
    policy = Policy("connectome", graph, seed=5)
    action, _ = policy(torch.randn(3, 20), policy.initial_state(3))
    action.square().sum().backward()
    assert policy.encoder.weight.grad is not None
    assert policy.readout.weight.grad is not None
    assert not policy.adjacency.requires_grad
    assert "adjacency" not in dict(policy.named_parameters())


def test_connectome_and_mlp_have_exact_trainable_parameter_match(graph: Graph) -> None:
    connectome = Policy("connectome", graph, seed=2)
    mlp = Policy("mlp", graph, seed=2)
    assert connectome.state_size == mlp.state_size == len(graph.ids)
    assert connectome.trainable_parameters() == mlp.trainable_parameters()
    assert sum(p.numel() for p in connectome.parameters()) == sum(
        p.numel() for p in mlp.parameters()
    )
    assert torch.equal(connectome.encoder.weight, mlp.encoder.weight)
    assert torch.equal(connectome.readout.weight, mlp.readout.weight)


def test_controller_reset_discards_episode_state(graph: Graph) -> None:
    controller = Controller(Policy("connectome", graph, seed=3))
    observation = np.arange(20, dtype=np.float32)
    first = controller.act(observation)
    assert not torch.equal(controller.state, torch.zeros_like(controller.state))
    controller.reset()
    assert torch.equal(controller.state, torch.zeros_like(controller.state))
    assert np.allclose(controller.act(observation), first)


@pytest.mark.parametrize(
    "kwargs",
    [{"seeds": [1, 1]}, {"policies": ["mlp", "mlp"]}, {"policies": ["invalid"]}],
)
def test_config_rejects_duplicate_or_unknown_experiment_entries(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(**kwargs)
