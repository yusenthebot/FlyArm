"""How far the controller's outputs are from its inputs in the frozen connectome (E63).

    PYTHONPATH=src .venv/bin/python scripts/connectome_paths.py \\
        --run runs/skill-dagger-connectome-nophase-001 --output docs/results/connectome-paths.json

Hop distance, along edges that carry weight (a presynaptic neuron with an unknown transmitter
sign has zero weight in the rate model), from the nearest input neuron to every output neuron,
in the intact graph and with each lesion group of flyarm.whole_brain.lesion removed. The rate
model runs three neural steps per control step, so an output more than three hops from every
input first responds to a new observation one or more control steps later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from flyarm.interfaces import NeuralInterface
from flyarm.io import save_json
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.lesion import REGIONS, region_mask

MAX_HOPS = 12


def hops_from(adjacency: csr_matrix, sources: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """Multi-source BFS hop count from ``sources`` (source -> target edges), -1 unreachable."""
    n = adjacency.shape[0]
    keep = ~removed
    # A super-source (index n) with an edge to every input gives multi-source distances.
    pruned = adjacency.multiply(keep[:, None]).multiply(keep[None, :]).tocsr()
    extra = csr_matrix(
        (np.ones(len(sources)), (np.full(len(sources), n), sources)), shape=(n + 1, n + 1)
    )
    padded = csr_matrix(
        (pruned.data, pruned.indices, np.append(pruned.indptr, pruned.nnz)), shape=(n + 1, n + 1)
    )
    graph = (padded + extra).tocsr()
    order, predecessors = breadth_first_order(graph, n, directed=True, return_predecessors=True)
    depth = np.full(n + 1, -1, dtype=np.int64)
    depth[n] = 0
    for node in order[1:]:
        depth[node] = depth[predecessors[node]] + 1
    return depth[:n] - (depth[:n] > 0)  # an input is one hop from the super-source


def distribution(hops: np.ndarray) -> dict[str, Any]:
    reached = hops[hops >= 0]
    return {
        "neurons": int(len(hops)),
        "unreachable": int((hops < 0).sum()),
        "median": float(np.median(reached)) if reached.size else None,
        "within": {str(k): int(((hops >= 0) & (hops <= k)).sum()) for k in (1, 2, 3, 6, 9)},
        "histogram": {str(k): int((hops == k).sum()) for k in range(MAX_HOPS + 1)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="imitation run with interface.json")
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack = ConnectomePack.load(args.pack)
    interface = NeuralInterface.load(args.run / "interface.json")
    inputs, outputs = interface.resolve_indices(pack)
    weights = pack.normalized_weights()
    carrying = weights != 0
    # Pack rows are targets and columns sources; BFS walks source -> target.
    adjacency = csr_matrix(
        (np.ones(int(carrying.sum())), (pack.col_idx[carrying], pack.rows()[carrying])),
        shape=(pack.nodes, pack.nodes),
    )
    neurons = pack.neurons()
    kinds = neurons.superclass.to_numpy()[outputs]
    interface_ids = np.concatenate([inputs, outputs])
    groups = {"intact": np.zeros(pack.nodes, dtype=bool)} | {
        region: region_mask(neurons, region, interface_ids) for region in REGIONS
    }
    everything = np.ones(pack.nodes, dtype=bool)
    everything[interface_ids] = False
    groups["all_but_interface"] = everything
    report: dict[str, Any] = {
        "edges_with_weight": int(carrying.sum()),
        "edges": int(pack.edges),
        "inputs": int(len(inputs)),
        "outputs": {kind: int((kinds == kind).sum()) for kind in np.unique(kinds)},
        "neural_steps_per_control_step": 3,
        "removed": {},
    }
    # Share of each output neuron's incoming absolute weight by presynaptic origin.
    origin = np.full(pack.nodes, "interior", dtype=object)
    origin[inputs], origin[outputs] = "input", "output"
    targets = pack.rows()
    report["incoming_weight_share"] = {}
    for kind in np.unique(kinds):
        chosen = np.zeros(pack.nodes, dtype=bool)
        chosen[outputs[kinds == kind]] = True
        edge = chosen[targets]
        magnitude = np.abs(weights[edge])
        sources = origin[pack.col_idx[edge]]
        total = magnitude.sum()
        report["incoming_weight_share"][kind] = {
            name: round(float(magnitude[sources == name].sum() / total), 4)
            for name in ("input", "output", "interior")
        }
    print("incoming weight share", report["incoming_weight_share"], flush=True)
    for name, removed in groups.items():
        hops = hops_from(adjacency, inputs, removed)[outputs]
        report["removed"][name] = {
            "removed_neurons": int(removed.sum()),
            **{kind: distribution(hops[kinds == kind]) for kind in np.unique(kinds)},
        }
        row = report["removed"][name]
        print(
            name,
            {
                kind: (row[kind]["median"], row[kind]["within"]["3"], row[kind]["unreachable"])
                for kind in np.unique(kinds)
            },
            flush=True,
        )
    save_json(args.output, report)


if __name__ == "__main__":
    main()
