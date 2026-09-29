"""Measured directed subgraphs and degree-preserving null controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECIPE_VERSION = "descending-contact-v1"


@dataclass(frozen=True)
class Graph:
    ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    contacts: np.ndarray
    signs: np.ndarray
    metadata: dict

    def validate(self) -> None:
        n = len(self.ids)
        if n < 2 or len(np.unique(self.ids)) != n or self.ids.dtype != np.int64:
            raise ValueError("Node IDs must be unique int64, with at least two nodes")
        if self.pre.dtype.kind not in "iu" or self.post.dtype.kind not in "iu":
            raise ValueError("Edge endpoints must be integer indices")
        if len(self.pre) == 0 or not (len(self.pre) == len(self.post) == len(self.contacts)):
            raise ValueError("Edge arrays must have equal, nonzero lengths")
        if min(self.pre.min(), self.post.min()) < 0 or max(self.pre.max(), self.post.max()) >= n:
            raise ValueError("Edge endpoint outside graph")
        if np.any(self.pre == self.post) or len(np.unique(self.pre * n + self.post)) != len(
            self.pre
        ):
            raise ValueError("Graph must have neither self loops nor duplicate directed edges")
        if not np.all(np.isfinite(self.contacts)) or np.any(self.contacts <= 0):
            raise ValueError("Contact counts must be finite and positive")
        if self.signs.shape != (n,) or not np.all(np.isin(self.signs, [-1, 0, 1])):
            raise ValueError("One neurotransmitter sign per node is required")

    def normalized_weights(self) -> np.ndarray:
        # Incoming absolute strength <= 1; unknown/modulatory transmitters contribute zero.
        signed = self.contacts * self.signs[self.pre]
        denom = np.bincount(self.post, weights=np.abs(signed), minlength=len(self.ids))
        return (signed / np.maximum(denom[self.post], 1)).astype(np.float32)

    def fingerprint(self) -> str:
        """Hash ordered numeric graph contents, independent of NPZ compression metadata."""
        digest = hashlib.sha256()
        for key, dtype in [
            ("ids", "<i8"),
            ("pre", "<i8"),
            ("post", "<i8"),
            ("contacts", "<f4"),
            ("signs", "<f4"),
        ]:
            digest.update(key.encode())
            digest.update(getattr(self, key).astype(dtype).tobytes())
        return digest.hexdigest()

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(
            path,
            ids=self.ids,
            pre=self.pre,
            post=self.post,
            contacts=self.contacts,
            signs=self.signs,
            metadata=np.array(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path: Path) -> Graph:
        with np.load(path, allow_pickle=False) as pack:
            graph = cls(
                ids=pack["ids"].copy(),
                pre=pack["pre"].copy(),
                post=pack["post"].copy(),
                contacts=pack["contacts"].copy(),
                signs=pack["signs"].copy(),
                metadata=json.loads(str(pack["metadata"])),
            )
        graph.validate()
        return graph


def shuffle_graph(graph: Graph, seed: int, swaps_per_edge: int = 10) -> Graph:
    """Directed double-edge swaps preserving each node's in/out degree exactly.

    Weights stay with source-edge slots: source sign and outgoing strength are preserved,
    but incoming weighted strength is not. Re-normalization is identical for both graphs.
    """
    graph.validate()
    rng = np.random.default_rng(seed)
    pre, post = graph.pre.copy(), graph.post.copy()
    pairs = set(zip(pre.tolist(), post.tolist(), strict=True))
    wanted = swaps_per_edge * len(pre)
    accepted = 0
    attempts = 0
    while accepted < wanted and attempts < wanted * 30:
        attempts += 1
        i, j = rng.integers(len(pre), size=2)
        a, b, c, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
        if a == c or b == d or a == d or c == b or (a, d) in pairs or (c, b) in pairs:
            continue
        pairs.remove((a, b))
        pairs.remove((c, d))
        pairs.update([(a, d), (c, b)])
        post[i], post[j] = d, b
        accepted += 1
    if accepted < wanted:
        raise ValueError(f"Only {accepted}/{wanted} valid swaps; graph too constrained")
    overlap = len(pairs.intersection(zip(graph.pre.tolist(), graph.post.tolist(), strict=True)))
    result = Graph(
        graph.ids.copy(),
        pre,
        post,
        graph.contacts.copy(),
        graph.signs.copy(),
        {
            **graph.metadata,
            "control": "directed_degree_preserving_swaps",
            "shuffle_seed": seed,
            "accepted_swaps": accepted,
            "edge_overlap_fraction": overlap / len(pre),
            "preserves": [
                "node_identity",
                "in_degree",
                "out_degree",
                "outgoing_strength",
                "source_sign",
                "contact_count_multiset",
            ],
            "does_not_preserve": ["incoming_weighted_strength", "spatial_locality", "motifs"],
        },
    )
    result.validate()
    return result
