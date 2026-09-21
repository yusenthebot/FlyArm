"""Measured directed subgraphs and degree-preserving null controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from flyarm.assets import SOURCE_SHA256, SOURCES, verify_raw_sources

RECIPE_VERSION = "descending-contact-v1"
CANONICAL_256 = "7a5018c5481d14307e1efec305f4f448648545e5feda7caf9297dab518f4da5d"


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

    def validate_mvp_provenance(self) -> None:
        """Require the independently pinned graph, not merely self-declared metadata."""
        self.validate()
        if (
            self.metadata.get("schema_version") != 1
            or self.metadata.get("recipe_version") != RECIPE_VERSION
        ):
            raise ValueError("Unsupported graph schema/recipe; regenerate with flyarm prepare")
        if self.metadata.get("dataset") != "MaleCNS v1.0":
            raise ValueError("MVP requires MaleCNS v1.0")
        files = self.metadata.get("sources", {}).get("files", {})
        if any(files.get(key, {}).get("sha256") != value for key, value in SOURCE_SHA256.items()):
            raise ValueError("Graph source hashes do not match pinned MaleCNS exports")
        if len(self.ids) != 256 or self.fingerprint() != CANONICAL_256:
            raise ValueError(
                "MVP run requires the pinned 256-node graph; new graphs need a new protocol"
            )

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


def prepare_graph(raw: Path, output: Path, max_nodes: int = 256) -> Graph:
    """Grow an anatomically seeded subgraph without using any task outcomes."""
    if not 32 <= max_nodes <= 4096:
        raise ValueError("MVP subgraph limit must be between 32 and 4096 nodes")
    manifest = verify_raw_sources(raw)
    neurons = feather.read_table(raw / SOURCES["annotations"][0]).to_pandas()
    transmitters = feather.read_table(raw / SOURCES["neurotransmitters"][0]).to_pandas()
    edges = feather.read_table(raw / SOURCES["weights"][0]).to_pandas()
    required = {"body_pre", "body_post", "weight"}
    if not required.issubset(edges.columns):
        raise ValueError(f"Unexpected weight columns: {list(edges.columns)}")
    if "body" not in transmitters or "consensus_nt" not in transmitters:
        raise ValueError(f"Unexpected transmitter columns: {list(transmitters.columns)}")
    neurons = neurons[neurons.superclass.notna() & (neurons.superclass != "glia")]
    ids = neurons.bodyId.to_numpy(dtype=np.int64)
    edges = edges[edges.body_pre.isin(ids) & edges.body_post.isin(ids)]
    edges = edges[(edges.body_pre != edges.body_post) & (edges.weight >= 3)]
    edges = edges.groupby(["body_pre", "body_post"], as_index=False, sort=True).weight.sum()
    seeds = neurons[neurons.type.isin(["DNa02", "DNg13", "DNge104", "DNp01"])].bodyId
    if seeds.empty:
        raise ValueError("None of the preregistered descending cell types were found")
    selected = set(int(x) for x in seeds)
    while len(selected) < max_nodes:
        incoming = edges[edges.body_post.isin(selected) & ~edges.body_pre.isin(selected)]
        outgoing = edges[edges.body_pre.isin(selected) & ~edges.body_post.isin(selected)]
        candidates = (
            pd.concat(
                [
                    incoming.rename(columns={"body_pre": "candidate"})[["candidate", "weight"]],
                    outgoing.rename(columns={"body_post": "candidate"})[["candidate", "weight"]],
                ]
            )
            .groupby("candidate", as_index=False)
            .weight.sum()
        )
        candidates = candidates.sort_values(["weight", "candidate"], ascending=[False, True])
        if candidates.empty:
            raise ValueError("Seed-connected component too small for requested graph")
        # Breadth-wise growth, deterministic score ties resolved by original integer ID.
        selected.update(candidates.candidate.iloc[: max_nodes - len(selected)].astype(int))
    kept_ids = np.array(sorted(selected), dtype=np.int64)
    kept = edges[edges.body_pre.isin(selected) & edges.body_post.isin(selected)]
    pre = np.searchsorted(kept_ids, kept.body_pre.to_numpy(dtype=np.int64))
    post = np.searchsorted(kept_ids, kept.body_post.to_numpy(dtype=np.int64))
    nt = transmitters.set_index("body").consensus_nt.reindex(kept_ids)
    signs = (
        nt.map({"acetylcholine": 1, "gaba": -1, "glutamate": -1, "GABA": -1})
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    meta = {
        "schema_version": 1,
        "recipe_version": RECIPE_VERSION,
        "dataset": "MaleCNS v1.0",
        "scope": "measured_subgraph_not_whole_brain",
        "selection": "descending seeds, strongest adjacent contact sums, ID tie break",
        "seed_types": ["DNa02", "DNg13", "DNge104", "DNp01"],
        "min_contacts": 3,
        "self_loops": "excluded",
        "nodes": len(kept_ids),
        "edges": len(pre),
        "retained_annotated_nodes": len(ids),
        "eligible_edges": len(edges),
        "node_fraction": len(kept_ids) / len(ids),
        "edge_fraction": len(pre) / len(edges),
        "zero_sign_nodes": int(np.sum(signs == 0)),
        "nt_counts": nt.fillna("unknown").value_counts().to_dict(),
        "sign_assumption": "ACh +1; GABA/glutamate -1; other/unknown 0; no receptor modeling",
        "dynamics": "abstract leaky tanh state; not physiological spikes",
        "sources": manifest,
    }
    graph = Graph(kept_ids, pre, post, kept.weight.to_numpy(dtype=np.float32), signs, meta)
    graph.save(output)
    return graph
