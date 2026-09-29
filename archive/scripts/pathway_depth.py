"""How far, and how fast, sensory input reaches the left front-leg motor neurons.

1. Structural depth: multi-source breadth-first search over signed edges (edges whose source
   has transmitter sign 0 carry no signal and are skipped) from each input population; for
   each hop count, the share of the 68 motor neurons first reached at that hop.
2. Neural steps per control step: RMS motor activity per control step when one population is
   driven with random +-0.5 currents, for several numbers of neural updates per control step
   (the kitchen default is 3, while one kitchen control step is 80 ms of simulated time).

Usage: PYTHONPATH=src uv run python scripts/pathway_depth.py docs/results/pathway-depth.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy import sparse

from flyarm.flyleg.interface import front_leg_interface
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack

PACK = Path("data/whole_brain/malecns-v1.0-c3")
ANNOTATIONS = Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
BATCH, DRIVE, CONTROL_STEPS, MAX_HOPS = 8, 0.5, 10, 12
STEP_COUNTS = (3, 6, 12, 24)


def hop_profile(pack: ConnectomePack, sources: np.ndarray, targets: np.ndarray) -> dict:
    """Share of targets first reached at each hop from any source, over signed edges."""
    signed = pack.signs[pack.col_idx] != 0
    adjacency = sparse.csr_matrix(
        (signed.astype(np.float32), pack.col_idx, pack.row_ptr), shape=(pack.nodes, pack.nodes)
    )
    index = {body: i for i, body in enumerate(pack.body_ids.tolist())}
    reached = np.zeros(pack.nodes, dtype=bool)
    reached[[index[b] for b in sources.tolist()]] = True
    frontier = reached.copy()
    target_index = np.array([index[b] for b in targets.tolist()])
    first_hop = np.full(len(targets), -1)
    for hop in range(1, MAX_HOPS + 1):
        frontier = (adjacency @ frontier.astype(np.float32) > 0) & ~reached
        reached |= frontier
        newly = (first_hop < 0) & frontier[target_index]
        first_hop[newly] = hop
        if not frontier.any():
            break
    hops = {str(h): int(np.sum(first_hop == h)) for h in range(1, MAX_HOPS + 1)}
    return {
        "targets": len(targets),
        "first_reached_at_hop": {h: n for h, n in hops.items() if n},
        "unreached": int(np.sum(first_hop < 0)),
        "median_hop": float(np.median(first_hop[first_hop > 0])) if (first_hop > 0).any() else None,
    }


def motor_series(dynamics: RateDynamics, current: np.ndarray, neural_steps: int) -> list[float]:
    state = dynamics.zeros(current.shape[0])
    series = []
    for _ in range(CONTROL_STEPS):
        state, pooled = dynamics.advance(state, mx.array(current), neural_steps)
        mx.eval(state, pooled)
        series.append(float(np.sqrt(np.mean(np.asarray(pooled) ** 2))))
    return series


def main(output: Path) -> None:
    pack = ConnectomePack.load(PACK)
    pack.validate_b1a_provenance()
    leg = front_leg_interface(pack, ANNOTATIONS)
    populations = {
        "left front-leg proprioceptors": leg.proprioceptors,
        "head sensory (cb_sensory)": leg.exteroceptors,
    }
    depth = {name: hop_profile(pack, ids, leg.motor_neurons) for name, ids in populations.items()}
    for name, row in depth.items():
        print(name, row, flush=True)
    generator = np.random.default_rng(0)
    n_prop, n_head = len(leg.proprioceptors), len(leg.exteroceptors)
    prop = generator.choice([-DRIVE, DRIVE], size=(BATCH, n_prop)).astype(np.float32)
    head = generator.choice([-DRIVE, DRIVE], size=(BATCH, n_head)).astype(np.float32)
    dynamics = RateDynamics(pack, leg.interface)
    steps = {}
    for count in STEP_COUNTS:
        steps[str(count)] = {
            "head_to_motor_rms_per_control_step": motor_series(
                dynamics, np.concatenate([np.zeros_like(prop), head], 1), count
            ),
            "proprio_to_motor_rms_per_control_step": motor_series(
                dynamics, np.concatenate([prop, np.zeros_like(head)], 1), count
            ),
        }
        row = steps[str(count)]
        print(
            count,
            "head",
            [f"{v:.2e}" for v in row["head_to_motor_rms_per_control_step"][:4]],
            "proprio",
            [f"{v:.2e}" for v in row["proprio_to_motor_rms_per_control_step"][:4]],
            flush=True,
        )
    result = {
        "pack_fingerprint": pack.fingerprint(),
        "protocol": {
            "drive": f"random +-{DRIVE} per input neuron, batch {BATCH}",
            "control_steps": CONTROL_STEPS,
            "recurrent_gain": 0.8,
            "readout": "68 left front-leg motor neurons, mean over a control step's neural steps",
            "signed_edges_only": True,
        },
        "structural_depth": depth,
        "neural_steps_per_control_step": steps,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
