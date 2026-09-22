"""Go/no-go measurements for the full connectome before any robot training.

Every number here comes from actually running the backend on the pinned pack: load and
provenance, bitwise determinism, the no-bypass property, numerical stability, memory,
control-rate latency, training-step cost and whether input activity reaches the outputs.
"""

from __future__ import annotations

import os
import platform
import resource
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.backend_mlx import MlxRateBackend, RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.interface import annotation_interface, interface_report

CONTROL_HZ = 20.0


def direct_only_weights(
    pack: ConnectomePack, interface: NeuralInterface, power: float = 1.0
) -> np.ndarray:
    """Normalized weights with every edge removed except input-to-output synapses."""
    inputs, outputs = interface.resolve_indices(pack)
    is_input = np.zeros(pack.nodes, dtype=bool)
    is_input[inputs] = True
    is_output = np.zeros(pack.nodes, dtype=bool)
    is_output[outputs] = True
    keep = is_input[pack.col_idx] & is_output[pack.rows()]
    return np.where(keep, pack.normalized_weights(power), 0.0).astype(np.float32)


def _currents(count: int, steps: int, batch: int, seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (scale * rng.standard_normal((steps, batch, count))).astype(np.float32)


def _run(backend: MlxRateBackend, currents: np.ndarray, neural_steps: int) -> np.ndarray:
    backend.reset(currents.shape[1])
    return np.stack(
        [np.asarray(backend.step(current, neural_steps).activity) for current in currents]
    )


def propagation(
    dynamics: RateDynamics, neural_steps: int, control_steps: int, seed: int
) -> dict[str, Any]:
    """Constant unit-variance input currents; output activity per control step."""
    current = _currents(dynamics.input_count, 1, 1, seed, 1.0)[0]
    backend = MlxRateBackend(dynamics)
    activity = _run(backend, np.repeat(current[None], control_steps, axis=0), neural_steps)[:, 0]
    rms = np.sqrt(np.mean(activity**2, axis=1))
    return {
        "output_rms_by_control_step": [float(value) for value in rms],
        "active_fraction_by_control_step": [float(np.mean(np.abs(row) > 1e-4)) for row in activity],
        "final_output_activity": activity[-1],
    }


def sensitivity(dynamics: RateDynamics, neural_steps: int, control_steps: int, seed: int) -> float:
    """Norm of d(random projection of pooled outputs)/d(input currents) over an episode."""
    rng = np.random.default_rng(seed)
    currents = mx.array(_currents(dynamics.input_count, control_steps, 1, seed, 1.0))
    projection = mx.array(rng.standard_normal((1, dynamics.output_count)).astype(np.float32))

    def readout(values: mx.array) -> mx.array:
        state = dynamics.zeros(1)
        total = mx.array(0.0)
        for step in range(control_steps):
            state, pooled = dynamics.advance(state, values[step], neural_steps)
            total = total + (pooled * projection).sum()
        return total

    gradient = mx.grad(readout)(currents)
    return float(mx.sqrt((gradient**2).sum()))


def training_step_seconds(
    dynamics: RateDynamics, batch: int, bptt: int, neural_steps: int, repeats: int = 3
) -> float:
    """Wall time of one truncated-BPTT chunk (forward and backward) at training batch size."""
    currents = mx.array(_currents(dynamics.input_count, bptt, batch, 7, 1.0))

    def loss(values: mx.array) -> mx.array:
        state = dynamics.zeros(batch)
        total = mx.array(0.0)
        for step in range(bptt):
            state, pooled = dynamics.advance(state, values[step], neural_steps)
            total = total + (pooled**2).mean()
        return total

    gradient_fn = mx.value_and_grad(loss)
    mx.eval(*gradient_fn(currents))
    started = time.perf_counter()
    for _ in range(repeats):
        mx.eval(*gradient_fn(currents))
    return (time.perf_counter() - started) / repeats


def go_no_go(pack_root: Path, *, neural_steps: int = 3, seed: int = 0) -> dict[str, Any]:
    started = time.perf_counter()
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    load_seconds = time.perf_counter() - started
    interface = annotation_interface(pack)
    report = interface_report(pack, interface)
    full = RateDynamics(pack, interface)

    # Determinism: identical inputs through two independently built backends.
    currents = _currents(full.input_count, 20, 1, seed, 1.0)
    first = _run(MlxRateBackend(full), currents, neural_steps)
    second = _run(MlxRateBackend(RateDynamics(pack, interface)), currents, neural_steps)
    deterministic = bool(np.array_equal(first, second))

    # No bypass: with every edge removed, strong inputs must leave outputs exactly zero.
    silent = _run(MlxRateBackend(RateDynamics(pack, interface, edges=False)), currents * 10, 3)
    no_bypass = (
        bool(np.all(silent == 0.0))
        and not np.intersect1d(interface.input_body_ids, interface.output_body_ids).size
    )

    # Stability: a long episode with large inputs stays finite and inside tanh bounds.
    backend = MlxRateBackend(full)
    backend.reset(1)
    strong = _currents(full.input_count, 400, 1, seed + 1, 3.0)
    timings = []
    for current in strong:
        tick = time.perf_counter()
        output = backend.step(current, neural_steps)
        np.asarray(output.activity)  # the only host transfer per control step
        timings.append(time.perf_counter() - tick)
    state = np.asarray(backend.state)
    stable = bool(np.all(np.isfinite(state)) and np.abs(state).max() <= 1.0)
    median = float(np.median(timings))

    reach = propagation(full, neural_steps, 20, seed)
    direct = propagation(
        RateDynamics(pack, interface, weights=direct_only_weights(pack, interface)),
        neural_steps,
        20,
        seed,
    )
    beyond_direct = reach.pop("final_output_activity") - direct.pop("final_output_activity")
    full_sensitivity = sensitivity(full, neural_steps, 8, seed)
    train_seconds = training_step_seconds(full, batch=8, bptt=8, neural_steps=neural_steps)
    peak_gpu = float(mx.get_peak_memory()) / 2**30
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes

    checks = {
        "pack_loaded_with_pinned_provenance": True,
        "single_step_deterministic": deterministic,
        "no_input_output_bypass": no_bypass,
        "numerically_stable_400_steps": stable,
        "memory_under_8_gib": peak_rss < 8.0,
        "control_step_meets_20_hz": median < 1.0 / CONTROL_HZ,
        "input_reaches_outputs": reach["active_fraction_by_control_step"][-1] > 0.5
        and full_sensitivity > 0,
    }
    return {
        "go": all(checks.values()),
        "checks": checks,
        "pack": {
            "root": str(pack_root),
            "fingerprint": pack.fingerprint(),
            "nodes": pack.nodes,
            "edges": pack.edges,
            "synapses_kept": pack.manifest["synapses_kept"],
            "load_seconds": load_seconds,
        },
        "interface": {
            key: report[key]
            for key in (
                "interface_fingerprint",
                "inputs",
                "outputs",
                "direct_input_to_output_edges",
                "output_reachability",
                "output_hops_from_inputs",
                "input_hops_to_outputs",
                "whole_cns_hops_from_inputs",
            )
        },
        "dynamics": {
            "update": "h <- 0.5 h + 0.5 tanh(I + 0.8 W h); W row-normalized, frozen",
            "neural_steps_per_control_step": neural_steps,
            "weight_dtype": "float32",
            "max_abs_state_after_400_steps": float(np.abs(state).max()),
        },
        "performance": {
            "control_step_ms_median": median * 1000,
            "control_step_ms_p95": float(np.quantile(timings, 0.95)) * 1000,
            "max_control_hz": 1.0 / median,
            "bptt_chunk_seconds_batch8_8steps": train_seconds,
            "peak_gpu_gib": peak_gpu,
            "peak_rss_gib": peak_rss,
        },
        "propagation": {
            "input": "constant N(0,1) current on every input neuron, 20 control steps",
            "full": reach,
            "direct_input_to_output_edges_only": direct,
            "final_rms_beyond_direct_edges": float(np.sqrt(np.mean(beyond_direct**2))),
            "episode_sensitivity_norm": full_sensitivity,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "device": str(mx.default_device()),
            "cpu_count": os.cpu_count(),
        },
    }
