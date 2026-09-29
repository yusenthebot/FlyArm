from __future__ import annotations

import numpy as np
import pytest

from flyarm.graph import Graph
from flyarm.interfaces import NeuralInterface


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


def test_interface_resolves_ordered_disjoint_ids_and_round_trips(graph: Graph, tmp_path) -> None:
    interface = NeuralInterface.bind(
        graph, np.array([102, 101], dtype=np.int64), np.array([106, 104], dtype=np.int64)
    )
    inputs, outputs = interface.resolve_indices(graph)
    assert np.array_equal(inputs, [1, 0])
    assert np.array_equal(outputs, [5, 3])
    path = tmp_path / "interface.json"
    interface.save(path)
    assert NeuralInterface.load(path) == interface


def test_interface_rejects_graph_fingerprint_mismatch(graph: Graph) -> None:
    interface = NeuralInterface.bind(
        graph, np.array([101], dtype=np.int64), np.array([106], dtype=np.int64)
    )
    changed = Graph(
        ids=graph.ids,
        pre=graph.pre,
        post=np.array([1, 2, 4, 3, 5], dtype=np.int64),
        contacts=graph.contacts,
        signs=graph.signs,
        metadata={},
    )
    changed.validate()
    with pytest.raises(ValueError, match="different graph"):
        interface.resolve_indices(changed)


def test_interface_rejects_overlap(graph: Graph) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        NeuralInterface.bind(
            graph, np.array([101], dtype=np.int64), np.array([101], dtype=np.int64)
        )
