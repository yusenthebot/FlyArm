"""Restricted engineering interfaces between a measured graph and a robot task.

The ascending/descending labels refer to the data annotations only.  They are
explicitly an engineering proxy for feedback and commands, not a claim that a
fly's native neurons implement robot anatomy.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flyarm.graph import Graph

ASCENDING_FEEDBACK_BODY_IDS = np.array(
    [
        10180,
        10671,
        10689,
        10797,
        13137,
        13538,
        13656,
        14498,
        14899,
        15884,
        16510,
        16897,
        17713,
        18050,
        22499,
        25185,
        27797,
        56630,
        58725,
        60372,
        66518,
        519660,
        522822,
    ],
    dtype=np.int64,
)

DESCENDING_COMMAND_BODY_IDS = np.array(
    [
        10001,
        10010,
        10091,
        10106,
        10141,
        10192,
        10223,
        10283,
        10360,
        10417,
        10580,
        10732,
        10967,
        10971,
        10975,
        11074,
        11133,
        11158,
        11233,
        11424,
        11466,
        11610,
        11625,
        11687,
        12175,
        12218,
        12223,
        12628,
        12781,
        33335,
        230783,
        512006,
        513052,
        515029,
        519228,
        519624,
        519896,
        521190,
        521197,
        521377,
        523769,
        524412,
        556329,
    ],
    dtype=np.int64,
)


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
        graph: Graph,
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

    @classmethod
    def canonical(cls, graph: Graph) -> NeuralInterface:
        """Bind the preregistered MaleCNS proxy IDs to one exact graph."""
        return cls.bind(
            graph,
            ASCENDING_FEEDBACK_BODY_IDS,
            DESCENDING_COMMAND_BODY_IDS,
        )

    def resolve_indices(self, graph: Graph) -> tuple[np.ndarray, np.ndarray]:
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

    def path_statistics(self, graph: Graph) -> dict[str, object]:
        """Return JSON-safe directed I/O topology statistics."""
        return directed_io_stats(graph, self).to_dict()


@dataclass(frozen=True)
class DirectedIOStats:
    """Topology-only reachability summary for a declared interface."""

    input_count: int
    output_count: int
    reachable_output_count: int
    unreachable_output_body_ids: tuple[int, ...]
    min_hops: int | None
    mean_hops: float | None
    max_hops: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "reachable_output_count": self.reachable_output_count,
            "unreachable_output_body_ids": list(self.unreachable_output_body_ids),
            "min_hops": self.min_hops,
            "mean_hops": self.mean_hops,
            "max_hops": self.max_hops,
        }


def directed_io_stats(graph: Graph, interface: NeuralInterface) -> DirectedIOStats:
    """Report paths from any declared feedback input to every command output."""
    input_indices, output_indices = interface.resolve_indices(graph)
    neighbors: list[list[int]] = [[] for _ in graph.ids]
    for source, target in zip(graph.pre, graph.post, strict=True):
        neighbors[int(source)].append(int(target))
    distances = np.full(len(graph.ids), -1, dtype=np.int64)
    queue: deque[int] = deque()
    for index in input_indices:
        distances[index] = 0
        queue.append(int(index))
    while queue:
        source = queue.popleft()
        for target in neighbors[source]:
            if distances[target] == -1:
                distances[target] = distances[source] + 1
                queue.append(target)
    output_distances = distances[output_indices]
    reachable = output_distances[output_distances >= 0]
    missing = tuple(
        int(body_id)
        for body_id, distance in zip(interface.output_body_ids, output_distances, strict=True)
        if distance < 0
    )
    return DirectedIOStats(
        input_count=len(input_indices),
        output_count=len(output_indices),
        reachable_output_count=len(reachable),
        unreachable_output_body_ids=missing,
        min_hops=int(reachable.min()) if len(reachable) else None,
        mean_hops=float(reachable.mean()) if len(reachable) else None,
        max_hops=int(reachable.max()) if len(reachable) else None,
    )


def male_cns_proxy_interface(graph: Graph) -> NeuralInterface:
    """Bind the preregistered MaleCNS proxy IDs to this exact graph."""
    return NeuralInterface.canonical(graph)


def canonical_interface(graph: Graph) -> NeuralInterface:
    """Public factory for the preregistered constrained MaleCNS interface."""
    return NeuralInterface.canonical(graph)


default_interface = canonical_interface
