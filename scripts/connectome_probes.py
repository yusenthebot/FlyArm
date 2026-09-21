"""Probes of the whole-connectome rate model, independent of any trained controller.

1. Pathway gain: drive one input population with random +-0.5 currents for 10 control steps
   and report the RMS activity of the 68 left front-leg motor neurons.
2. Recurrent-gain sweep: head-sensory and proprioceptive pathway gain and the residual state
   1, 5, 10 and 20 control steps after all inputs are removed, for several recurrent gains.

Usage: PYTHONPATH=src uv run python scripts/connectome_probes.py docs/results/connectome-probes.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow.feather as feather

from flyarm.flyleg.interface import front_leg_interface
from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack

PACK = Path("data/whole_brain/malecns-v1.0-c3")
ANNOTATIONS = Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
BATCH, DRIVE, STEPS, NEURAL_STEPS = 8, 0.5, 10, 3


def _run(dynamics: RateDynamics, current: np.ndarray, steps: int, state: mx.array | None = None):
    state = dynamics.zeros(current.shape[0]) if state is None else state
    pooled = None
    for _ in range(steps):
        state, pooled = dynamics.advance(state, mx.array(current), NEURAL_STEPS)
    mx.eval(state, pooled)
    return state, np.asarray(pooled)


def pathway_gains(pack: ConnectomePack, motor: np.ndarray, populations: dict[str, np.ndarray]):
    generator = np.random.default_rng(0)
    rows = {}
    for name, inputs in populations.items():
        interface = NeuralInterface.bind(pack, inputs, motor, label=name)
        dynamics = RateDynamics(pack, interface)
        drive = generator.choice([-DRIVE, DRIVE], size=(BATCH, len(inputs))).astype(np.float32)
        _, out = _run(dynamics, drive, STEPS)
        rows[name] = {
            "inputs": int(len(inputs)),
            "motor_rms": float(np.sqrt((out**2).mean())),
            "motor_max": float(np.abs(out).max()),
        }
        print(name, rows[name], flush=True)
    return rows


def gain_sweep(pack: ConnectomePack, leg, gains: list[float]):
    generator = np.random.default_rng(0)
    n_prop, n_head = len(leg.proprioceptors), len(leg.exteroceptors)
    prop = generator.choice([-DRIVE, DRIVE], size=(4, n_prop)).astype(np.float32)
    head = generator.choice([-DRIVE, DRIVE], size=(4, n_head)).astype(np.float32)
    silent_p, silent_h = np.zeros_like(prop), np.zeros_like(head)
    rows = {}
    for gain in gains:
        dynamics = RateDynamics(pack, leg.interface, recurrent_gain=gain)
        _, head_out = _run(dynamics, np.concatenate([silent_p, head], 1), STEPS)
        _, prop_out = _run(dynamics, np.concatenate([prop, silent_h], 1), STEPS)
        state, _ = _run(dynamics, np.concatenate([prop, head], 1), STEPS)
        residual = {}
        for step in range(1, 21):
            state, _ = _run(dynamics, np.concatenate([silent_p, silent_h], 1), 1, state)
            if step in (1, 5, 10, 20):
                residual[str(step)] = float(np.abs(np.asarray(state)).max())
        rows[str(gain)] = {
            "head_to_motor_rms": float(np.sqrt((head_out**2).mean())),
            "proprio_to_motor_rms": float(np.sqrt((prop_out**2).mean())),
            "residual_max_state_after_steps": residual,
        }
        print(gain, rows[str(gain)], flush=True)
    return rows


def main(output: Path) -> None:
    pack = ConnectomePack.load(PACK)
    pack.validate_b1a_provenance()
    leg = front_leg_interface(pack, ANNOTATIONS)
    table = feather.read_table(ANNOTATIONS, columns=["bodyId", "superclass", "type"])
    annotations = table.to_pandas().set_index("bodyId").reindex(pack.body_ids)
    ids, kind = pack.body_ids, annotations.superclass
    types = annotations["type"].fillna("").astype(str)
    lobula = kind.eq("visual_projection") & (
        types.str.startswith("LC") | types.str.startswith("LPLC")
    )
    populations = {
        "left front-leg proprioceptors": leg.proprioceptors,
        "head sensory (cb_sensory)": ids[kind.eq("cb_sensory").to_numpy()],
        "visual projection (all)": ids[kind.eq("visual_projection").to_numpy()],
        "visual projection LC/LPLC": ids[lobula.to_numpy()],
        "ascending neurons": ids[kind.eq("ascending_neuron").to_numpy()],
        "descending neurons (reference)": ids[kind.eq("descending_neuron").to_numpy()],
    }
    result = {
        "pack_fingerprint": pack.fingerprint(),
        "protocol": {
            "drive": f"random +-{DRIVE} per input neuron",
            "control_steps": STEPS,
            "neural_steps_per_control_step": NEURAL_STEPS,
            "readout": "68 left front-leg motor neurons, mean over neural steps",
        },
        "pathway_gain": pathway_gains(pack, leg.motor_neurons, populations),
        "recurrent_gain_sweep": gain_sweep(pack, leg, [0.8, 0.9, 0.95, 0.99, 0.9999]),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
