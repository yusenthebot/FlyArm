from __future__ import annotations

import numpy as np
import pytest
import torch

from flyarm.graph import Graph, shuffle_graph
from flyarm.models import Policy


def graph() -> Graph:
    """A sparse, swappable directed graph with nonuniform source strength."""
    n = 12
    pre = np.repeat(np.arange(n, dtype=np.int64), 3)
    post = np.array(
        [(source + offset) % n for source in range(n) for offset in (1, 3, 5)],
        dtype=np.int64,
    )
    result = Graph(
        ids=np.arange(100, 100 + n, dtype=np.int64),
        pre=pre,
        post=post,
        contacts=np.arange(1, len(pre) + 1, dtype=np.float32),
        signs=np.array([1, -1, 0] * 4, dtype=np.int8),
        metadata={"fixture": True},
    )
    result.validate()
    return result


def test_graph_int64_round_trip(tmp_path) -> None:
    original = graph()
    path = tmp_path / "graph.npz"
    original.save(path)
    restored = Graph.load(path)
    assert restored.ids.dtype == np.int64
    assert restored.pre.dtype == np.int64
    assert restored.post.dtype == np.int64
    assert np.array_equal(restored.ids, original.ids)
    assert np.array_equal(restored.pre, original.pre)
    assert np.array_equal(restored.post, original.post)
    assert np.array_equal(restored.contacts, original.contacts)
    assert np.array_equal(restored.signs, original.signs)
    assert restored.metadata == original.metadata


def test_duplicate_directed_edge_is_rejected() -> None:
    duplicate = Graph(
        ids=np.array([1, 2, 3], dtype=np.int64),
        pre=np.array([0, 0], dtype=np.int64),
        post=np.array([1, 1], dtype=np.int64),
        contacts=np.array([1.0, 2.0], dtype=np.float32),
        signs=np.array([1, -1, 0], dtype=np.int8),
        metadata={},
    )
    with pytest.raises(ValueError, match="duplicate"):
        duplicate.validate()


def test_adjacency_uses_post_by_pre_direction_for_sparse_mm() -> None:
    directed = Graph(
        ids=np.array([10, 11, 12], dtype=np.int64),
        pre=np.array([0, 1], dtype=np.int64),
        post=np.array([1, 2], dtype=np.int64),
        contacts=np.array([1.0, 1.0], dtype=np.float32),
        signs=np.array([1, 1, 1], dtype=np.int8),
        metadata={},
    )
    policy = Policy("connectome", directed)
    state = torch.tensor([[2.0, 3.0, 5.0]])
    recurrent = torch.sparse.mm(policy.adjacency, state.T).T
    assert torch.equal(recurrent, torch.tensor([[0.0, 2.0, 3.0]]))


def test_shuffle_preserves_directed_degrees_source_strength_and_seed() -> None:
    original = graph()
    first = shuffle_graph(original, seed=23, swaps_per_edge=2)
    second = shuffle_graph(original, seed=23, swaps_per_edge=2)
    assert np.array_equal(first.pre, second.pre)
    assert np.array_equal(first.post, second.post)
    assert np.array_equal(
        np.bincount(first.pre, minlength=len(first.ids)),
        np.bincount(original.pre, minlength=len(original.ids)),
    )
    assert np.array_equal(
        np.bincount(first.post, minlength=len(first.ids)),
        np.bincount(original.post, minlength=len(original.ids)),
    )
    assert np.array_equal(
        np.bincount(first.pre, weights=first.contacts, minlength=len(first.ids)),
        np.bincount(original.pre, weights=original.contacts, minlength=len(original.ids)),
    )
