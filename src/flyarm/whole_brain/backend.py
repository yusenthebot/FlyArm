"""Backend contract shared by the full-connectome, subgraph and future spiking backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BrainOutput:
    """Activity of the declared output neurons only, pooled over one control period.

    ``activity`` is a backend-native array shaped ``[batch, output_count]``; the full
    neural state never leaves the backend through this record.
    """

    activity: Any
    neural_steps: int


class BrainBackend(Protocol):
    """Stateful frozen-connectome simulator driven only through declared input neurons."""

    input_count: int
    output_count: int

    def reset(self, batch_size: int) -> None: ...

    def step(self, input_current: Any, neural_steps: int) -> BrainOutput: ...

    def read_nodes(self, body_ids: Any) -> Any: ...
