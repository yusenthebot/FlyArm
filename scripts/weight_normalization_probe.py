"""Operating regime of the rate model under two synaptic weight normalizations.

The default ("l1", ConnectomePack.normalized_weights) divides each neuron's signed input
weights by their absolute sum, so a neuron with k similar inputs weights each by about 1/k
and independent input fluctuations shrink by about 1/sqrt(k) per synapse: deep networks of
such neurons stay quiet and close to linear (E25).
The L2 alternative divides by the Euclidean norm instead, which preserves the variance of
independent inputs, the usual normalization of random recurrent network theory; L-p norms with
1 < p < 2 lie in between (p = 1 reproduces the default for every neuron with at least one
contact, since contacts are at least 3).

For each normalization and recurrent gain, with the kitchen front-leg interface:
- activity: mean |h| over all neurons and the share above 0.1 after CONTROL_STEPS control steps
  of random +-0.5 drive on every input neuron;
- persistence: max |h| 1, 5 and 20 control steps after the drive is removed;
- linear response of the 68 motor neurons (participation ratio, singular values for 90%,
  head-sensory energy share) and the error of the linear prediction at the training drive.

Usage (an optional interface.json replaces the kitchen interface, e.g. the pick-and-place one):

    PYTHONPATH=src uv run python scripts/weight_normalization_probe.py \
        docs/results/weight-normalization.json [runs/RUN/interface.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from flyarm.flyleg.interface import front_leg_interface
from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack

PACK = Path("data/whole_brain/malecns-v1.0-c3")
ANNOTATIONS = Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
EPSILON, DRIVE, BATCH, CONTROL_STEPS, NEURAL_STEPS = 1e-2, 0.5, 512, 10, 3
SETTINGS = (
    (1.0, 0.8),
    (2.0, 0.5),
    (2.0, 0.8),
    (2.0, 0.95),
    (2.0, 0.2),
    (2.0, 0.3),
    (1.25, 0.8),
    (1.5, 0.5),
    (1.5, 0.8),
)


def lp_weights(pack: ConnectomePack, power: float) -> np.ndarray:
    """Signed contacts divided by each target's L-p norm of incoming weights."""
    signed = pack.contacts.astype(np.float64) * pack.signs[pack.col_idx]
    rows = pack.rows()
    total = np.bincount(rows, weights=np.abs(signed) ** power, minlength=pack.nodes)
    norm = total ** (1.0 / power)
    return (signed / np.maximum(norm[rows], 1e-12)).astype(np.float32)


def run(dynamics: RateDynamics, current: np.ndarray, steps: int, state=None):
    state = dynamics.zeros(current.shape[0]) if state is None else state
    drive = mx.array(current)
    for _ in range(steps):
        state, pooled = dynamics.advance(state, drive, NEURAL_STEPS)
    mx.eval(state, pooled)
    out = np.asarray(pooled)
    return state, out if out.shape[0] == current.shape[0] else out.T


def response(dynamics: RateDynamics, inputs: int) -> np.ndarray:
    columns = []
    for start in range(0, inputs, BATCH):
        stop = min(start + BATCH, inputs)
        current = np.zeros((stop - start, inputs), dtype=np.float32)
        current[np.arange(stop - start), np.arange(start, stop)] = EPSILON
        columns.append(run(dynamics, current, CONTROL_STEPS)[1] / EPSILON)
    return np.concatenate(columns, axis=0).T


def spectrum(matrix: np.ndarray, split: int | None) -> dict:
    values = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    energy = np.cumsum(values**2) / np.sum(values**2)
    row = {
        "participation_ratio": float(np.sum(values**2) ** 2 / np.sum(values**4)),
        "rank_90": int(np.searchsorted(energy, 0.90) + 1),
        "rank_99": int(np.searchsorted(energy, 0.99) + 1),
        "frobenius": float(np.linalg.norm(matrix)),
    }
    if split is not None:
        head, proprio = np.sum(matrix[:, split:] ** 2), np.sum(matrix[:, :split] ** 2)
        row["head_energy_share"] = float(head / (head + proprio))
    return row


def main(output: Path, interface_path: Path | None = None) -> None:
    pack = ConnectomePack.load(PACK)
    pack.validate_b1a_provenance()
    if interface_path is None:
        leg = front_leg_interface(pack, ANNOTATIONS)
        interface, split = leg.interface, len(leg.proprioceptors)
        described = "kitchen front leg: 23 proprioceptors + 4,868 head sensory in, 68 motor"
    else:
        interface, split = NeuralInterface.load(interface_path), None
        described = str(interface_path)
    inputs = len(interface.input_body_ids)
    weights = {power: lp_weights(pack, power) for power in sorted({p for p, _ in SETTINGS})}
    generator = np.random.default_rng(0)
    drive = generator.choice([-DRIVE, DRIVE], size=(8, inputs)).astype(np.float32)
    silent = np.zeros_like(drive)
    rows = {}
    for power, gain in SETTINGS:
        name = f"L{power:g} gain {gain}"
        dynamics = RateDynamics(pack, interface, recurrent_gain=gain, weights=weights[power])
        state, driven = run(dynamics, drive, CONTROL_STEPS)
        activity = np.abs(np.asarray(state))
        persistence = {}
        for step in range(1, 21):
            state, _ = run(dynamics, silent, 1, state)
            if step in (1, 5, 20):
                persistence[str(step)] = float(np.abs(np.asarray(state)).max())
        lingering = np.abs(np.asarray(state))
        persistence["mean_abs_at_20"] = float(lingering.mean())
        persistence["neurons_above_0.1_at_20"] = int((lingering > 0.1).any(axis=1).sum())
        matrix = response(dynamics, inputs)
        predicted = drive @ matrix.T
        rows[name] = {
            "mean_abs_activity": float(activity.mean()),
            "share_above_0.1": float((activity > 0.1).mean()),
            "motor_rms_under_drive": float(np.sqrt(np.mean(driven**2))),
            "max_abs_after_drive_removed": persistence,
            "motor_linear_response": spectrum(matrix, split),
            "linear_prediction_relative_error": float(
                np.linalg.norm(driven - predicted) / np.linalg.norm(driven)
            ),
        }
        print(name, json.dumps(rows[name]), flush=True)
    result = {
        "pack_fingerprint": pack.fingerprint(),
        "protocol": {
            "drive": f"random +-{DRIVE} on every input neuron, batch 8",
            "control_steps": CONTROL_STEPS,
            "neural_steps": NEURAL_STEPS,
            "epsilon": EPSILON,
            "interface": described,
        },
        "settings": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
