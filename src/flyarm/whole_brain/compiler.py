"""Compile the complete annotated MaleCNS v1.0 connectome into a memory-mapped CSR pack.

Rows are postsynaptic targets and columns are presynaptic sources, so one recurrent update
reads ``drive[i] = sum_j W[i, j] h[j]`` straight off row ``i``. The recipe deliberately
matches the 256-node MVP graph (annotated non-glia neurons, at least three contacts, no
self loops, ACh +1 / GABA and glutamate -1 / other 0, incoming-strength normalization) so
that B1a changes the graph scale and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from flyarm.assets import SOURCE_SHA256, SOURCES, verify_raw_sources
from flyarm.graph import Graph

PACK_SCHEMA = 1
RECIPE_VERSION = "whole-malecns-contact-v1"
DATASET = "MaleCNS v1.0"
DEFAULT_MIN_CONTACTS = 3
SIGN_MAP = {"acetylcholine": 1, "gaba": -1, "GABA": -1, "glutamate": -1}
NEURON_COLUMNS = ("bodyId", "type", "superclass", "class", "subclass", "instance", "status")

# Measured once from the pinned exports with the default recipe; a pack that differs is a
# different experiment and needs its own protocol.
CANONICAL_NODES = 166_700
CANONICAL_EDGES = 10_520_377
CANONICAL_FINGERPRINT = "7aa88acd850bd908317d205edfaf53bbb65e17438d98112cca8722a521c172e6"

_ARRAYS: dict[str, str] = {
    "body_ids": "<i8",
    "row_ptr": "<i8",
    "col_idx": "<i4",
    "contacts": "<i4",
    "signs": "i1",
}


def _content_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, dtype in _ARRAYS.items():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(arrays[name], dtype=dtype).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class ConnectomePack:
    """Read-only CSR connectome; target-major rows, source-index columns."""

    body_ids: np.ndarray
    row_ptr: np.ndarray
    col_idx: np.ndarray
    contacts: np.ndarray
    signs: np.ndarray
    manifest: dict[str, Any]
    root: Path | None = None

    @property
    def ids(self) -> np.ndarray:
        """Body IDs in index order (the attribute name NeuralInterface binds against)."""
        return self.body_ids

    @property
    def nodes(self) -> int:
        return len(self.body_ids)

    @property
    def edges(self) -> int:
        return len(self.col_idx)

    def rows(self) -> np.ndarray:
        """Target index of every stored edge, expanded from ``row_ptr``."""
        return np.repeat(np.arange(self.nodes, dtype=np.int32), np.diff(self.row_ptr))

    def validate(self) -> None:
        """Structural checks; arrays are read-only, so one pass per instance suffices."""
        if not self._structurally_valid:  # the property raises on failure
            raise ValueError("Invalid connectome pack")

    @cached_property
    def _structurally_valid(self) -> bool:
        n, m = self.nodes, self.edges
        for name, dtype in _ARRAYS.items():
            array = getattr(self, name)
            if array.ndim != 1 or array.dtype != np.dtype(dtype):
                raise ValueError(f"{name} must be one-dimensional {dtype}")
        if n < 2 or np.any(np.diff(self.body_ids) <= 0):
            raise ValueError("body_ids must be strictly increasing with at least two nodes")
        if len(self.row_ptr) != n + 1 or self.row_ptr[0] != 0 or self.row_ptr[-1] != m:
            raise ValueError("row_ptr must have nodes + 1 entries spanning every edge")
        if np.any(np.diff(self.row_ptr) < 0):
            raise ValueError("row_ptr must be nondecreasing")
        if m == 0 or len(self.contacts) != m:
            raise ValueError("Edge arrays must have equal, nonzero lengths")
        if self.col_idx.min() < 0 or self.col_idx.max() >= n:
            raise ValueError("Edge source outside graph")
        rows = self.rows()
        if np.any(rows == self.col_idx):
            raise ValueError("Pack must not contain self loops")
        same_row = rows[1:] == rows[:-1]
        if np.any(np.diff(self.col_idx)[same_row] <= 0):
            raise ValueError("Columns must be strictly increasing within each row (no duplicates)")
        if np.any(self.contacts <= 0):
            raise ValueError("Contact counts must be positive")
        if len(self.signs) != n or not np.all(np.isin(self.signs, (-1, 0, 1))):
            raise ValueError("One neurotransmitter sign in {-1, 0, 1} per node is required")
        return True

    @cached_property
    def _fingerprint(self) -> str:
        return _content_fingerprint({name: getattr(self, name) for name in _ARRAYS})

    def fingerprint(self) -> str:
        """SHA-256 of the ordered numeric content, independent of file layout."""
        return self._fingerprint

    def normalized_weights(self) -> np.ndarray:
        """Same rule as Graph.normalized_weights: incoming absolute strength <= 1 per target."""
        signed = self.contacts.astype(np.float64) * self.signs[self.col_idx]
        rows = self.rows()
        denom = np.bincount(rows, weights=np.abs(signed), minlength=self.nodes)
        return (signed / np.maximum(denom[rows], 1.0)).astype(np.float32)

    def degrees(self) -> tuple[np.ndarray, np.ndarray]:
        """(in_degree, out_degree) per node index."""
        return np.diff(self.row_ptr), np.bincount(self.col_idx, minlength=self.nodes)

    def neurons(self) -> pd.DataFrame:
        """Annotation rows in pack index order."""
        if self.root is None:
            raise ValueError("In-memory packs carry no annotation table")
        table = feather.read_table(self.root / "neurons.feather").to_pandas()
        if not np.array_equal(table.bodyId.to_numpy(np.int64), self.body_ids):
            raise ValueError("neurons.feather is not aligned with body_ids")
        return table

    def validate_b1a_provenance(self) -> None:
        """Require the independently pinned full pack, not merely self-declared metadata."""
        self.validate()
        meta = self.manifest
        if meta.get("schema_version") != PACK_SCHEMA or meta.get("recipe") != RECIPE_VERSION:
            raise ValueError("Unsupported pack schema/recipe; recompile with flyarm whole-brain")
        if meta.get("dataset") != DATASET or meta.get("min_contacts") != DEFAULT_MIN_CONTACTS:
            raise ValueError("B1a requires the default MaleCNS v1.0 recipe")
        files = meta.get("sources", {}).get("files", {})
        if any(files.get(key, {}).get("sha256") != value for key, value in SOURCE_SHA256.items()):
            raise ValueError("Pack source hashes do not match pinned MaleCNS exports")
        if (self.nodes, self.edges) != (CANONICAL_NODES, CANONICAL_EDGES):
            raise ValueError(f"B1a requires {CANONICAL_NODES} nodes / {CANONICAL_EDGES} edges")
        if self.fingerprint() != CANONICAL_FINGERPRINT:
            raise ValueError("Pack content differs from the pinned B1a connectome")

    def save(self, root: Path, neurons: pd.DataFrame | None = None) -> None:
        """Write atomically: a partial directory is renamed only after every file exists."""
        self.validate()
        if root.exists():
            raise FileExistsError(root)
        partial = root.with_name(root.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        for name, dtype in _ARRAYS.items():
            np.save(partial / f"{name}.npy", np.ascontiguousarray(getattr(self, name), dtype))
        manifest = {**self.manifest, "fingerprint": self.fingerprint()}
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if neurons is not None:
            if not np.array_equal(neurons.bodyId.to_numpy(np.int64), self.body_ids):
                raise ValueError("Annotation rows must align with body_ids")
            table = pa.Table.from_pandas(neurons, preserve_index=False)
            feather.write_feather(table, partial / "neurons.feather")
        partial.rename(root)

    @classmethod
    def load(cls, root: Path) -> ConnectomePack:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_version") != PACK_SCHEMA:
            raise ValueError("Unsupported connectome pack schema")
        arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in _ARRAYS}
        pack = cls(**arrays, manifest=manifest, root=root)
        pack.validate()
        if manifest.get("fingerprint") != pack.fingerprint():
            raise ValueError("Pack content fingerprint does not match its manifest")
        return pack

    @classmethod
    def from_graph(cls, graph: Graph) -> ConnectomePack:
        """In-memory pack for a measured subgraph (the 256-node SubgraphBackend baseline)."""
        graph.validate()
        order = np.argsort(graph.ids, kind="stable")
        if not np.array_equal(order, np.arange(len(order))):
            raise ValueError("Graph IDs must already be sorted")
        if not np.all(graph.contacts == np.round(graph.contacts)):
            raise ValueError("Contact counts must be integers")
        edge_order = np.lexsort((graph.pre, graph.post))
        counts = np.bincount(graph.post, minlength=len(graph.ids))
        row_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        pack = cls(
            body_ids=graph.ids.astype(np.int64),
            row_ptr=row_ptr,
            col_idx=graph.pre[edge_order].astype(np.int32),
            contacts=graph.contacts[edge_order].astype(np.int32),
            signs=graph.signs.astype(np.int8),
            manifest={
                "schema_version": PACK_SCHEMA,
                "recipe": "from_graph",
                "graph_fingerprint": graph.fingerprint(),
                "dataset": graph.metadata.get("dataset"),
            },
        )
        pack.validate()
        return pack


def _eligible_neurons(raw: Path) -> tuple[pd.DataFrame, np.ndarray]:
    table = feather.read_table(
        raw / SOURCES["annotations"][0], columns=list(NEURON_COLUMNS)
    ).to_pandas()
    table = table[table.superclass.notna() & (table.superclass != "glia")]
    table = table.sort_values("bodyId", kind="stable").reset_index(drop=True)
    ids = table.bodyId.to_numpy(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Annotation table contains duplicate body IDs")
    return table, ids


def _index_of(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of each value in sorted ``ids``, or -1 when absent."""
    position = np.searchsorted(ids, values)
    clipped = np.minimum(position, len(ids) - 1)
    return np.where(ids[clipped] == values, clipped, -1)


def compile_connectome(
    raw: Path, output: Path, *, min_contacts: int = DEFAULT_MIN_CONTACTS
) -> ConnectomePack:
    """Build the full annotated connectome from hash-verified official exports."""
    if not 1 <= min_contacts <= 100:
        raise ValueError("min_contacts must be between 1 and 100")
    if output.exists():
        raise FileExistsError(output)
    manifest = verify_raw_sources(raw)
    neurons, ids = _eligible_neurons(raw)
    weights = feather.read_table(raw / SOURCES["weights"][0], memory_map=True)
    if not {"body_pre", "body_post", "weight"}.issubset(weights.column_names):
        raise ValueError(f"Unexpected weight columns: {weights.column_names}")
    pre = _index_of(ids, weights.column("body_pre").to_numpy())
    post = _index_of(ids, weights.column("body_post").to_numpy())
    contacts = weights.column("weight").to_numpy()
    total_rows, total_synapses = len(contacts), int(contacts.sum())
    between = (pre >= 0) & (post >= 0) & (pre != post)
    annotated_synapses = int(contacts[between].sum())
    keep = between & (contacts >= min_contacts)
    pre, post, contacts = pre[keep], post[keep], contacts[keep]
    del weights, between, keep

    # Aggregate any duplicated (post, pre) rows exactly as the subgraph recipe does.
    keys = post.astype(np.int64) * len(ids) + pre
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    summed = np.bincount(inverse, weights=contacts).astype(np.int64)
    if summed.max() > np.iinfo(np.int32).max:
        raise ValueError("Aggregated contact count overflows int32")
    rows, cols = np.divmod(unique_keys, len(ids))
    row_ptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=len(ids)))))

    transmitters = feather.read_table(
        raw / SOURCES["neurotransmitters"][0], columns=["body", "consensus_nt"]
    ).to_pandas()
    nt = transmitters.set_index("body").consensus_nt.reindex(ids)
    signs = nt.map(SIGN_MAP).fillna(0).to_numpy(np.int8)
    neurons = neurons.assign(consensus_nt=nt.fillna("unknown").to_numpy())

    pack = ConnectomePack(
        body_ids=ids,
        row_ptr=row_ptr.astype(np.int64),
        col_idx=cols.astype(np.int32),
        contacts=summed.astype(np.int32),
        signs=signs,
        manifest={
            "schema_version": PACK_SCHEMA,
            "recipe": RECIPE_VERSION,
            "dataset": DATASET,
            "scope": "complete_annotated_connectome_not_physiology",
            "node_rule": "annotated superclass, excluding glia",
            "edge_rule": f"summed contacts >= {min_contacts} between eligible neurons",
            "min_contacts": min_contacts,
            "self_loops": "excluded",
            "layout": "CSR rows=postsynaptic target, columns=presynaptic source",
            "nodes": len(ids),
            "edges": len(cols),
            "synapses_kept": int(summed.sum()),
            "synapses_between_eligible_neurons": annotated_synapses,
            "export_rows": total_rows,
            "export_synapses": total_synapses,
            "zero_sign_nodes": int(np.sum(signs == 0)),
            "superclass_counts": neurons.superclass.value_counts().to_dict(),
            "nt_counts": nt.fillna("unknown").value_counts().to_dict(),
            "sign_assumption": "ACh +1; GABA/glutamate -1; other/unknown 0; no receptors",
            "dynamics": "abstract leaky tanh rate state; not physiological spikes",
            "sources": manifest,
        },
    )
    pack.save(output, neurons)
    return ConnectomePack.load(output)
