"""Linear response of the kitchen fly interface: how many independent motor signals can it carry?

For small inputs the rate model is close to linear, so after a few control steps of constant
drive the readout is R = M u, with M of shape [outputs, inputs] fixed by the connectome.
This script measures M column by column (one-hot drive of size EPSILON, steady state after
CONTROL_STEPS control steps of NEURAL_STEPS updates), then reports its singular spectrum:
participation ratio, the number of singular values needed for 90% and 99% of the energy, and
the condition number, separately for the proprioceptive and head-sensory input blocks.
It also checks linearity at the drive used in training (random +-0.5 on every input neuron).

Networks: measured connectome with the 68 left front-leg motor neurons as readout, the same
with 1,314 descending neurons added, and the kitchen seed-0 degree-preserving shuffle.

Usage: PYTHONPATH=src uv run python scripts/linear_response.py docs/results/linear-response.json
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
from flyarm.whole_brain.shuffle import shuffle_pack

PACK = Path("data/whole_brain/malecns-v1.0-c3")
ANNOTATIONS = Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
EPSILON, DRIVE, BATCH, CONTROL_STEPS, NEURAL_STEPS = 1e-2, 0.5, 512, 10, 3
KITCHEN_SHUFFLE_SEED = 17000


def steady_output(dynamics: RateDynamics, current: np.ndarray) -> np.ndarray:
    """Readout [batch, outputs] after CONTROL_STEPS control steps of constant drive."""
    state = dynamics.zeros(current.shape[0])
    drive = mx.array(current)
    for _ in range(CONTROL_STEPS):
        state, pooled = dynamics.advance(state, drive, NEURAL_STEPS)
    mx.eval(pooled)
    out = np.asarray(pooled)
    return out if out.shape[0] == current.shape[0] else out.T


def response_matrix(dynamics: RateDynamics, inputs: int) -> np.ndarray:
    columns = []
    for start in range(0, inputs, BATCH):
        stop = min(start + BATCH, inputs)
        current = np.zeros((stop - start, inputs), dtype=np.float32)
        current[np.arange(stop - start), np.arange(start, stop)] = EPSILON
        columns.append(steady_output(dynamics, current) / EPSILON)
        print(f"  inputs {stop}/{inputs}", flush=True)
    return np.concatenate(columns, axis=0).T


def spectrum(matrix: np.ndarray) -> dict:
    values = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    energy = np.cumsum(values**2) / np.sum(values**2)
    return {
        "shape": list(matrix.shape),
        "frobenius": float(np.linalg.norm(matrix)),
        "participation_ratio": float(np.sum(values**2) ** 2 / np.sum(values**4)),
        "rank_90": int(np.searchsorted(energy, 0.90) + 1),
        "rank_99": int(np.searchsorted(energy, 0.99) + 1),
        "condition_number": float(values[0] / values[-1]) if values[-1] > 0 else None,
        "top_singular_values": [float(v) for v in values[:10]],
    }


def linearity(dynamics: RateDynamics, matrix: np.ndarray, inputs: int) -> dict:
    generator = np.random.default_rng(0)
    current = generator.choice([-DRIVE, DRIVE], size=(8, inputs)).astype(np.float32)
    actual = steady_output(dynamics, current)
    predicted = current @ matrix.T
    return {
        "drive": f"random +-{DRIVE} on every input neuron",
        "relative_error_of_linear_prediction": float(
            np.linalg.norm(actual - predicted) / np.linalg.norm(actual)
        ),
    }


def analyse(name: str, pack: ConnectomePack, interface: NeuralInterface, split: int) -> dict:
    print(name, flush=True)
    dynamics = RateDynamics(pack, interface)
    inputs = len(interface.input_body_ids)
    matrix = response_matrix(dynamics, inputs)
    row = {
        "outputs": int(matrix.shape[0]),
        "all_inputs": spectrum(matrix),
        "proprioceptive_inputs": spectrum(matrix[:, :split]),
        "head_sensory_inputs": spectrum(matrix[:, split:]),
        "linearity": linearity(dynamics, matrix, inputs),
    }
    print(json.dumps({k: v for k, v in row.items() if k != "outputs"}, indent=1), flush=True)
    return row


def main(output: Path) -> None:
    pack = ConnectomePack.load(PACK)
    pack.validate_b1a_provenance()
    leg = front_leg_interface(pack, ANNOTATIONS)
    wide = front_leg_interface(pack, ANNOTATIONS, include_descending=True)
    split = len(leg.proprioceptors)
    if list(leg.interface.input_body_ids[:split]) != list(leg.proprioceptors):
        raise ValueError("expected proprioceptors first in the interface input order")
    shuffled = shuffle_pack(pack, KITCHEN_SHUFFLE_SEED)
    rebound = NeuralInterface.bind(
        shuffled, leg.interface.input_body_ids, leg.interface.output_body_ids, label="shuffled"
    )
    result = {
        "pack_fingerprint": pack.fingerprint(),
        "protocol": {
            "epsilon": EPSILON,
            "control_steps": CONTROL_STEPS,
            "neural_steps": NEURAL_STEPS,
            "recurrent_gain": 0.8,
            "kitchen_shuffle_seed": KITCHEN_SHUFFLE_SEED,
        },
        "measured_motor_readout": analyse("measured, 68 motor", pack, leg.interface, split),
        "measured_motor_plus_descending": analyse("measured, wide", pack, wide.interface, split),
        "shuffled_motor_readout": analyse("shuffled, 68 motor", shuffled, rebound, split),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
