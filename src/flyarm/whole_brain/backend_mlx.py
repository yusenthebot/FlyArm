"""MLX/Metal rate backend: a frozen signed CSR recurrence with a hand-written adjoint.

One control period applies ``neural_steps`` updates of

    h <- (1 - alpha) h + alpha tanh(I + g W h)

where ``I`` is nonzero only on declared input neurons and ``W`` is the normalized, signed,
frozen connectome. Constants default to the 256-node MVP (alpha 0.5, g 0.8). The one
deliberate difference from that policy is the readout: B1a decodes the output neurons'
activity averaged over the ``neural_steps`` updates of a control period (a pooled time
window), whereas the subgraph policy read only the last update. With ``neural_steps=1``
the two coincide. Because every row of ``W`` has absolute sum at most one, the update is a
contraction for ``g < 1`` and cannot blow up numerically.

The whole state stays in unified memory. ``W h`` is a Metal kernel over target-major CSR
(one thread per target neuron and batch column, fixed summation order, so results are
bitwise deterministic); its vector-Jacobian product is the same kernel on the transposed
CSR, so training never forms gradients for the 10.5M frozen edges.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
import numpy as np

from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.backend import BrainOutput
from flyarm.whole_brain.compiler import ConnectomePack

WEIGHT_DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}

_SPMM_SOURCE = """
    uint gid = thread_position_in_grid.x;
    uint row = gid / BATCH;
    if (row >= ROWS) {
        return;
    }
    uint column = gid - row * BATCH;
    int start = ptr[row];
    int stop = ptr[row + 1];
    float total = 0.0f;
    for (int edge = start; edge < stop; ++edge) {
        total += float(values[edge]) * state[indices[edge] * BATCH + column];
    }
    out[gid] = total;
"""
_kernel: Any = None


def _spmm_kernel() -> Any:
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="flyarm_csr_spmm",
            input_names=["ptr", "indices", "values", "state"],
            output_names=["out"],
            source=_SPMM_SOURCE,
        )
    return _kernel


class FrozenCSR:
    """A frozen square matrix and its transpose, resident on the GPU."""

    def __init__(
        self,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        weights: np.ndarray,
        weight_dtype: str = "float32",
    ) -> None:
        if weight_dtype not in WEIGHT_DTYPES:
            raise ValueError(f"weight_dtype must be one of {sorted(WEIGHT_DTYPES)}")
        rows = len(row_ptr) - 1
        if len(col_idx) >= np.iinfo(np.int32).max or rows >= np.iinfo(np.int32).max:
            raise ValueError("CSR kernel uses int32 offsets")
        if len(weights) != len(col_idx) or not np.all(np.isfinite(weights)):
            raise ValueError("One finite weight per stored edge is required")
        self.rows = rows
        dtype = WEIGHT_DTYPES[weight_dtype]
        targets = np.repeat(np.arange(rows, dtype=np.int32), np.diff(row_ptr))
        transpose = np.lexsort((targets, col_idx))
        counts = np.bincount(col_idx, minlength=rows)
        self._forward = (
            mx.array(row_ptr.astype(np.int32)),
            mx.array(np.asarray(col_idx, dtype=np.int32)),
            mx.array(np.asarray(weights, dtype=np.float32)).astype(dtype),
        )
        self._adjoint = (
            mx.array(np.concatenate(([0], np.cumsum(counts))).astype(np.int32)),
            mx.array(targets[transpose]),
            mx.array(np.asarray(weights, dtype=np.float32)[transpose]).astype(dtype),
        )
        # Materialize the constants once; lazy arrays are bound to the creating thread's stream.
        mx.eval(*self._forward, *self._adjoint)
        self.apply: Callable[[mx.array], mx.array] = self._differentiable()

    def _run(self, matrix: tuple[mx.array, mx.array, mx.array], state: mx.array) -> mx.array:
        if state.ndim != 2 or state.shape[0] != self.rows or state.dtype != mx.float32:
            raise ValueError("CSR state must be float32 shaped [neurons, batch]")
        batch = state.shape[1]
        return _spmm_kernel()(
            inputs=[*matrix, state],
            template=[("BATCH", batch), ("ROWS", self.rows)],
            grid=(self.rows * batch, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(self.rows, batch)],
            output_dtypes=[mx.float32],
        )[0]

    def _differentiable(self) -> Callable[[mx.array], mx.array]:
        @mx.custom_function
        def apply(state: mx.array) -> mx.array:
            return self._run(self._forward, state)

        @apply.vjp
        def apply_vjp(primals: Any, cotangent: Any, output: Any) -> tuple[mx.array]:
            gradient = cotangent[0] if isinstance(cotangent, (list, tuple)) else cotangent
            return (self._run(self._adjoint, gradient),)

        return cast(Callable[[mx.array], mx.array], apply)

    def transpose_apply(self, state: mx.array) -> mx.array:
        """``W^T state``; exposed so tests can check the adjoint independently."""
        return self._run(self._adjoint, state)


class RateDynamics:
    """Pure, differentiable neural update bound to one pack and one declared interface."""

    def __init__(
        self,
        pack: ConnectomePack,
        interface: NeuralInterface,
        *,
        edges: bool = True,
        alpha: float = 0.5,
        recurrent_gain: float = 0.8,
        weight_dtype: str = "float32",
        weights: np.ndarray | None = None,
    ) -> None:
        if not 0 < alpha <= 1 or not 0 <= recurrent_gain < 1:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= recurrent_gain < 1")
        inputs, outputs = interface.resolve_indices(pack)
        self.body_ids = np.asarray(pack.body_ids)
        self.neurons = pack.nodes
        self.alpha, self.recurrent_gain = alpha, recurrent_gain
        self.edges = edges
        self.input_count, self.output_count = len(inputs), len(outputs)
        self.input_indices = mx.array(inputs.astype(np.int32))
        self.output_indices = mx.array(outputs.astype(np.int32))
        self.interface_fingerprint = interface.fingerprint
        self.pack_fingerprint = pack.fingerprint()
        self.matrix = (
            FrozenCSR(
                np.asarray(pack.row_ptr),
                np.asarray(pack.col_idx),
                pack.normalized_weights() if weights is None else weights,
                weight_dtype,
            )
            if edges
            else None
        )

    def zeros(self, batch: int) -> mx.array:
        if batch < 1:
            raise ValueError("batch must be positive")
        return mx.zeros((self.neurons, batch), dtype=mx.float32)

    def advance(
        self, state: mx.array, current: mx.array, neural_steps: int
    ) -> tuple[mx.array, mx.array]:
        """Return (new state [neurons, batch], mean output activity [batch, outputs])."""
        if neural_steps < 1:
            raise ValueError("neural_steps must be positive")
        batch = state.shape[1]
        if current.shape != (batch, self.input_count):
            raise ValueError(f"Input current must be shaped [{batch}, {self.input_count}]")
        drive = mx.zeros_like(state).at[self.input_indices].add(current.T)
        pooled = mx.zeros((self.output_count, batch), dtype=mx.float32)
        for _ in range(neural_steps):
            total = drive
            if self.matrix is not None:
                total = total + self.recurrent_gain * self.matrix.apply(state)
            state = (1 - self.alpha) * state + self.alpha * mx.tanh(total)
            pooled = pooled + mx.take(state, self.output_indices, axis=0)
        return state, (pooled / neural_steps).T


class MlxRateBackend:
    """Stateful BrainBackend over RateDynamics; state persists until reset()."""

    def __init__(self, dynamics: RateDynamics) -> None:
        self.dynamics = dynamics
        self.input_count = dynamics.input_count
        self.output_count = dynamics.output_count
        self.state = dynamics.zeros(1)

    def reset(self, batch_size: int) -> None:
        self.state = self.dynamics.zeros(batch_size)

    def step(self, input_current: Any, neural_steps: int) -> BrainOutput:
        current = mx.array(input_current, dtype=mx.float32)
        self.state, pooled = self.dynamics.advance(self.state, current, neural_steps)
        # Materialize each control period so lazy graphs never span an episode.
        mx.eval(self.state, pooled)
        return BrainOutput(pooled, neural_steps)

    def read_nodes(self, body_ids: Any) -> np.ndarray:
        """Host copy of selected neurons' state, shaped [batch, len(body_ids)]."""
        wanted = np.asarray(body_ids, dtype=np.int64)
        index = np.searchsorted(self.dynamics.body_ids, wanted)
        clipped = np.minimum(index, len(self.dynamics.body_ids) - 1)
        if np.any(self.dynamics.body_ids[clipped] != wanted):
            raise ValueError("Requested body IDs are not in this connectome")
        values = mx.take(self.state, mx.array(clipped.astype(np.int32)), axis=0)
        return np.asarray(values).T
