"""Restricted engineering interfaces between a measured graph and a robot task.

The ascending/descending labels refer to the data annotations only.  They are
explicitly an engineering proxy for feedback and commands, not a claim that a
fly's native neurons implement robot anatomy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class BindableGraph(Protocol):
    """Any measured connectome with ordered Body IDs and a content fingerprint."""

    @property
    def ids(self) -> np.ndarray: ...

    def validate(self) -> None: ...

    def fingerprint(self) -> str: ...


def _interface_fingerprint(inputs: np.ndarray, outputs: np.ndarray, graph_fingerprint: str) -> str:
    digest = hashlib.sha256()
    digest.update(inputs.astype("<i8", copy=False).tobytes())
    digest.update(outputs.astype("<i8", copy=False).tobytes())
    digest.update(graph_fingerprint.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class NeuralInterface:
    """Ordered, graph-bound I/O body IDs for a constrained experiment."""

    input_body_ids: np.ndarray
    output_body_ids: np.ndarray
    graph_fingerprint: str
    label: str = "ascending-feedback/descending-command engineering proxy"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NeuralInterface):
            return NotImplemented
        return (
            self.graph_fingerprint == other.graph_fingerprint
            and self.label == other.label
            and np.array_equal(self.input_body_ids, other.input_body_ids)
            and np.array_equal(self.output_body_ids, other.output_body_ids)
        )

    def validate(self) -> None:
        for name, values in (("input", self.input_body_ids), ("output", self.output_body_ids)):
            if values.ndim != 1 or values.dtype != np.int64 or len(values) == 0:
                raise ValueError(f"{name} body IDs must be a nonempty one-dimensional int64 array")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"{name} body IDs must be unique")
        if np.intersect1d(self.input_body_ids, self.output_body_ids).size:
            raise ValueError("Input and output body IDs must be disjoint")
        if len(self.graph_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.graph_fingerprint
        ):
            raise ValueError("graph_fingerprint must be a lowercase SHA-256 digest")

    @property
    def fingerprint(self) -> str:
        self.validate()
        return _interface_fingerprint(
            self.input_body_ids, self.output_body_ids, self.graph_fingerprint
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable, content-addressed interface record."""
        self.validate()
        return {
            "schema_version": 1,
            "input_body_ids": self.input_body_ids.tolist(),
            "output_body_ids": self.output_body_ids.tolist(),
            "graph_fingerprint": self.graph_fingerprint,
            "label": self.label,
            "interface_fingerprint": self.fingerprint,
        }

    @classmethod
    def bind(
        cls,
        graph: BindableGraph,
        input_body_ids: np.ndarray,
        output_body_ids: np.ndarray,
        *,
        label: str = "ascending-feedback/descending-command engineering proxy",
    ) -> NeuralInterface:
        interface = cls(
            np.asarray(input_body_ids, dtype=np.int64).copy(),
            np.asarray(output_body_ids, dtype=np.int64).copy(),
            graph.fingerprint(),
            label,
        )
        interface.resolve_indices(graph)
        return interface

    def resolve_indices(self, graph: BindableGraph) -> tuple[np.ndarray, np.ndarray]:
        """Resolve body IDs without changing their declared order."""
        self.validate()
        graph.validate()
        if self.graph_fingerprint != graph.fingerprint():
            raise ValueError("Neural interface is bound to a different graph fingerprint")
        id_to_index = {int(body_id): index for index, body_id in enumerate(graph.ids)}
        missing = [
            int(body_id)
            for body_id in np.concatenate((self.input_body_ids, self.output_body_ids))
            if int(body_id) not in id_to_index
        ]
        if missing:
            raise ValueError(f"Neural interface IDs absent from graph: {missing[:5]}")
        inputs = np.array(
            [id_to_index[int(body_id)] for body_id in self.input_body_ids], dtype=np.int64
        )
        outputs = np.array(
            [id_to_index[int(body_id)] for body_id in self.output_body_ids], dtype=np.int64
        )
        return inputs, outputs

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> NeuralInterface:
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported neural interface schema")
        interface = cls(
            np.asarray(payload["input_body_ids"], dtype=np.int64),
            np.asarray(payload["output_body_ids"], dtype=np.int64),
            payload["graph_fingerprint"],
            payload.get("label", "ascending-feedback/descending-command engineering proxy"),
        )
        interface.validate()
        if payload.get("interface_fingerprint") != interface.fingerprint:
            raise ValueError("Neural interface content fingerprint mismatch")
        return interface
