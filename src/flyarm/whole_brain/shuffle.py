"""Degree-preserving null connectome at full scale.

The 256-node control uses sequential double-edge swaps; at 10.5M edges that loop is too
slow in Python, so this uses the equivalent vectorized configuration-model construction:
permute every edge's target, then repair self loops and duplicate pairs by swapping targets
with random valid edges until none remain. Exactly as in ``graph.shuffle_graph``, contact
counts stay with their source slot, so every neuron keeps its identity, in-degree,
out-degree, outgoing contact multiset and transmitter sign, and the ascending/descending/
motor interface sets are unchanged. Incoming weighted strength, locality and motifs are
not preserved; weights are re-normalized by the same rule as the measured graph.
"""

from __future__ import annotations

import numpy as np

from flyarm.whole_brain.compiler import ConnectomePack


def _conflicts(pre: np.ndarray, post: np.ndarray, nodes: int) -> np.ndarray:
    """Self loops, and every duplicate (source, target) pair after its first occurrence."""
    keys = pre * nodes + post
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    bad = pre == post
    bad[order[1:][ordered[1:] == ordered[:-1]]] = True
    return bad


def shuffle_pack(pack: ConnectomePack, seed: int, max_rounds: int = 200) -> ConnectomePack:
    pack.validate()
    rng = np.random.default_rng(seed)
    nodes = pack.nodes
    pre = np.asarray(pack.col_idx, dtype=np.int64)
    post = pack.rows().astype(np.int64)[rng.permutation(pack.edges)]
    rounds = 0
    while True:
        bad = _conflicts(pre, post, nodes)
        broken = np.flatnonzero(bad)
        if broken.size == 0:
            break
        rounds += 1
        if rounds > max_rounds:
            raise ValueError(f"{broken.size} conflicting edges remain after {max_rounds} rounds")
        partners = rng.choice(np.flatnonzero(~bad), size=broken.size, replace=False)
        post[broken], post[partners] = post[partners], post[broken].copy()

    order = np.lexsort((pre, post))
    original = pack.rows().astype(np.int64) * nodes + np.asarray(pack.col_idx)
    shuffled_keys = post * nodes + pre
    overlap = float(np.isin(shuffled_keys, original, assume_unique=True).mean())
    row_ptr = np.concatenate(([0], np.cumsum(np.bincount(post, minlength=nodes))))
    result = ConnectomePack(
        body_ids=np.asarray(pack.body_ids),
        row_ptr=row_ptr.astype(np.int64),
        col_idx=pre[order].astype(np.int32),
        contacts=np.asarray(pack.contacts)[order],
        signs=np.asarray(pack.signs),
        manifest={
            **{key: value for key, value in pack.manifest.items() if key != "fingerprint"},
            "control": "configuration_model_target_permutation_with_conflict_repair",
            "source_fingerprint": pack.fingerprint(),
            "shuffle_seed": seed,
            "repair_rounds": rounds,
            "edge_overlap_fraction": overlap,
            "preserves": [
                "node_identity",
                "in_degree",
                "out_degree",
                "outgoing_contact_multiset",
                "source_sign",
                "interface_sets",
            ],
            "does_not_preserve": ["incoming_weighted_strength", "spatial_locality", "motifs"],
        },
    )
    in_before, out_before = pack.degrees()
    in_after, out_after = result.degrees()
    if not (np.array_equal(in_before, in_after) and np.array_equal(out_before, out_after)):
        raise AssertionError("Shuffle changed a node degree")
    result.validate()
    return result
