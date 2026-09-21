"""Shared synthetic connectomes for the whole-brain tests."""

from __future__ import annotations

import numpy as np

from flyarm.graph import Graph
from flyarm.interfaces import BindableGraph, NeuralInterface


def make_random_graph(n: int = 40, edges: int = 200, seed: int = 0) -> Graph:
    """Sparse signed directed graph with Body IDs 1000..1000+n and integer contacts."""
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < edges:
        a, b = (int(x) for x in rng.integers(n, size=2))
        if a != b:
            pairs.add((a, b))
    pre, post = (np.array(values, dtype=np.int64) for values in zip(*sorted(pairs), strict=True))
    graph = Graph(
        ids=np.arange(1000, 1000 + n, dtype=np.int64),
        pre=pre,
        post=post,
        contacts=rng.integers(3, 30, size=edges).astype(np.float32),
        signs=rng.choice([-1.0, 0.0, 1.0], size=n).astype(np.float32),
        metadata={},
    )
    graph.validate()
    return graph


def make_interface(graph: BindableGraph, n: int = 40) -> NeuralInterface:
    """First six neurons are inputs, last five are outputs."""
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    return NeuralInterface.bind(graph, ids[:6], ids[-5:])
