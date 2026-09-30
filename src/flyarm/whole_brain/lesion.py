"""Test-time lesions of a trained controller's frozen connectome (research log E63).

A lesion changes only the frozen dynamics a trained BrainPolicy runs on; encoder, decoder and
interface stay as trained (BrainPolicy.with_dynamics). Two kinds:

- silencing: a set of neurons is clamped to zero after every neural step, so nothing passes
  through them; interface neurons (inputs and outputs) are never silenced;
- state reset: the recurrent state is cleared before every control step, so the controller
  keeps its within-step dynamics (neural_steps updates) but no memory across control steps.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pandas as pd

from flyarm.whole_brain.backend_mlx import RateDynamics

# Superclass groups of the MaleCNS annotation (neurons.feather) that a lesion silences together.
REGIONS: dict[str, tuple[str, ...]] = {
    "central_brain": ("cb_intrinsic",),
    "optic_lobes": ("ol_intrinsic", "visual_projection", "visual_centrifugal"),
    "vnc_interneurons": ("vnc_intrinsic",),
    "sensory": ("vnc_sensory", "ol_sensory", "cb_sensory", "sensory_ascending"),
}


class LesionedDynamics:
    """RateDynamics with silenced neurons and/or a state reset before every control step."""

    def __init__(
        self, inner: RateDynamics, silenced: np.ndarray | None = None, reset: bool = False
    ) -> None:
        if silenced is not None:
            silenced = np.asarray(silenced, dtype=bool)
            if silenced.shape != (inner.neurons,):
                raise ValueError(f"silenced must be a mask over {inner.neurons} neurons")
            interface = np.concatenate(
                [np.asarray(inner.input_indices), np.asarray(inner.output_indices)]
            )
            if silenced[interface].any():
                raise ValueError("a lesion may not silence interface neurons")
        self.inner = inner
        self.reset = reset
        self.silenced_count = 0 if silenced is None else int(silenced.sum())
        self.keep = (
            None
            if silenced is None or not silenced.any()
            else mx.array((~silenced).astype(np.float32))[:, None]
        )

    def __getattr__(self, name: str) -> Any:
        # Everything but advance (sizes, indices, fingerprints, zeros) is the intact graph's.
        return getattr(self.inner, name)

    def advance(
        self, state: mx.array, current: mx.array, neural_steps: int
    ) -> tuple[mx.array, mx.array]:
        if self.reset:
            state = mx.zeros_like(state)
        if self.keep is None:
            return self.inner.advance(state, current, neural_steps)
        # One neural step at a time with the same drive equals the intact loop; interface
        # neurons are never silenced, so each step's pooled outputs are already the masked ones.
        pooled = None
        for _ in range(neural_steps):
            state, step = self.inner.advance(state, current, 1)
            state = state * self.keep
            pooled = step if pooled is None else pooled + step
        assert pooled is not None  # advance rejects neural_steps < 1
        return state, pooled / neural_steps


def region_mask(neurons: pd.DataFrame, region: str, interface: np.ndarray) -> np.ndarray:
    """Neurons of one REGIONS group, excluding the interface."""
    mask = neurons.superclass.isin(REGIONS[region]).to_numpy().copy()
    mask[interface] = False
    return mask


def random_mask(count: int, interface: np.ndarray, neurons: int, seed: int) -> np.ndarray:
    """``count`` neurons drawn uniformly from the non-interface neurons."""
    eligible = np.setdiff1d(np.arange(neurons), interface)
    if count > len(eligible):
        raise ValueError("more neurons to silence than there are outside the interface")
    mask = np.zeros(neurons, dtype=bool)
    mask[np.random.default_rng([seed, 23]).choice(eligible, size=count, replace=False)] = True
    return mask


def mcnemar(intact: np.ndarray, lesioned: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired episode outcomes (same seeds, both controllers)."""
    from scipy.stats import binomtest

    intact, lesioned = np.asarray(intact, bool), np.asarray(lesioned, bool)
    lost = int((intact & ~lesioned).sum())
    gained = int((~intact & lesioned).sum())
    discordant = lost + gained
    p = 1.0 if discordant == 0 else float(binomtest(lost, discordant, 0.5).pvalue)
    return {"lost": lost, "gained": gained, "p": p}
