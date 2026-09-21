"""Annotation-defined whole-CNS interface: observations enter ascending neurons only and
actions are read from descending and VNC motor neurons only.

As in the subgraph protocol these roles are an engineering proxy for feedback and
command channels, not a claim that fly neurons natively encode a robot arm.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.compiler import ConnectomePack

INPUT_SUPERCLASSES = ("ascending_neuron",)
OUTPUT_SUPERCLASSES = ("descending_neuron", "vnc_motor")
LABEL = "whole-CNS ascending_neuron inputs -> descending_neuron + vnc_motor outputs (proxy)"


def annotation_interface(pack: ConnectomePack) -> NeuralInterface:
    """Select every neuron of the declared superclasses, ordered by Body ID."""
    neurons = pack.neurons()
    inputs = neurons.bodyId[neurons.superclass.isin(INPUT_SUPERCLASSES)].to_numpy(np.int64)
    outputs = neurons.bodyId[neurons.superclass.isin(OUTPUT_SUPERCLASSES)].to_numpy(np.int64)
    return NeuralInterface.bind(pack, inputs, outputs, label=LABEL)


def _ranges(ptr: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Concatenated CSR slot indices of ``ptr[node]:ptr[node + 1]`` for every node."""
    starts, lengths = ptr[nodes], ptr[nodes + 1] - ptr[nodes]
    offsets = np.repeat(starts - np.cumsum(lengths) + lengths, lengths)
    return offsets + np.arange(lengths.sum())


def hop_distances(ptr: np.ndarray, neighbors: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Multi-source breadth-first hop count over a CSR adjacency; -1 means unreachable."""
    distance = np.full(len(ptr) - 1, -1, dtype=np.int64)
    distance[sources] = 0
    frontier, level = np.unique(sources), 0
    while frontier.size:
        level += 1
        reached = neighbors[_ranges(ptr, frontier)]
        frontier = np.unique(reached[distance[reached] < 0])
        distance[frontier] = level
    return distance


def _histogram(distances: np.ndarray) -> dict[str, int]:
    reachable = distances[distances >= 0]
    values, counts = np.unique(reachable, return_counts=True)
    histogram = {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}
    histogram["unreachable"] = int(np.sum(distances < 0))
    return histogram


def interface_report(pack: ConnectomePack, interface: NeuralInterface) -> dict[str, Any]:
    """Content-addressed interface record plus directed topology between its two sets."""
    inputs, outputs = interface.resolve_indices(pack)
    if np.intersect1d(inputs, outputs).size:
        raise ValueError("Input and output neurons must be disjoint")
    rows = pack.rows()
    by_source = np.argsort(pack.col_idx, kind="stable")
    source_ptr = np.concatenate(([0], np.cumsum(np.bincount(pack.col_idx, minlength=pack.nodes))))
    downstream = hop_distances(source_ptr, rows[by_source], inputs)
    upstream = hop_distances(np.asarray(pack.row_ptr), np.asarray(pack.col_idx), outputs)
    output_hops, input_hops = downstream[outputs], upstream[inputs]
    reachable = output_hops[output_hops >= 0]
    in_degree, out_degree = pack.degrees()
    is_input = np.zeros(pack.nodes, dtype=bool)
    is_input[inputs] = True
    is_output = np.zeros(pack.nodes, dtype=bool)
    is_output[outputs] = True
    neurons = pack.neurons()

    def describe(index: np.ndarray) -> dict[str, Any]:
        rows_ = neurons.iloc[index]
        return {
            "count": len(index),
            "superclass_counts": rows_.superclass.value_counts().to_dict(),
            "distinct_types": int(rows_.type.nunique()),
            "top_types": rows_.type.value_counts().head(12).to_dict(),
            "zero_sign": int(np.sum(pack.signs[index] == 0)),
        }

    return {
        **interface.to_dict(),
        "pack_fingerprint": pack.fingerprint(),
        "input_rule": {"superclass": list(INPUT_SUPERCLASSES)},
        "output_rule": {"superclass": list(OUTPUT_SUPERCLASSES)},
        "inputs": describe(inputs),
        "outputs": describe(outputs),
        "input_types": neurons.type.iloc[inputs].fillna("").tolist(),
        "output_types": neurons.type.iloc[outputs].fillna("").tolist(),
        "output_superclasses": neurons.superclass.iloc[outputs].tolist(),
        "direct_input_to_output_edges": int(np.sum(is_input[pack.col_idx] & is_output[rows])),
        "outputs_reachable_from_inputs": len(reachable),
        "output_reachability": len(reachable) / len(outputs),
        "output_hops_from_inputs": _histogram(output_hops),
        "output_hops_mean": float(reachable.mean()) if len(reachable) else None,
        "inputs_reaching_outputs": int(np.sum(input_hops >= 0)),
        "input_hops_to_outputs": _histogram(input_hops),
        "whole_cns_hops_from_inputs": _histogram(downstream),
        "input_out_degree_median": float(np.median(out_degree[inputs])),
        "output_in_degree_median": float(np.median(in_degree[outputs])),
        "bypass": "none: encoder writes input rows, decoder reads output rows, sets disjoint",
    }
