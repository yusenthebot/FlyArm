"""Which fly interface can carry the kitchen policy? Decoder-only fits on frozen features.

For each interface, the untrained seed-0 encoder drives the frozen connectome over every
demonstration step of the kitchen-complete split; a tanh(linear) decoder on the standardized
readout is fitted to the 10-step action chunks with L1 and Adam (shuffled mini-batches, as in
scripts/readout_conditioning.py, where the 68-motor readout reaches about 0.106).
Interfaces:
- leg: 23 left front-leg proprioceptors + 4,868 head sensory neurons in, 68 left front-leg
  motor neurons out (kitchen default; two encoders);
- whole body: the B1a pick-and-place interface, 1,846 ascending neurons in, 1,314 descending
  and 708 VNC motor neurons out (one encoder over all 30 features);
- leg senses, whole-body readout: the leg's inputs and the B1a outputs.

Usage:

    PYTHONPATH=src uv run python scripts/kitchen_interface_probe.py \
        docs/results/kitchen-interface.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from flyarm.benchmarks import kitchen
from flyarm.config import FlyLegConfig
from flyarm.flyleg.experiment import leg_dynamics
from flyarm.flyleg.interface import front_leg_interface, leg_channels
from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.policy import BrainPolicy
from flyarm.whole_brain.training import chunk_targets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readout_conditioning import features, fit, standardize  # noqa: E402

PACK = Path("data/whole_brain/malecns-v1.0-c3")
ANNOTATIONS = Path("data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather")
CONFIG = Path("configs/flyleg-kitchen-complete-chunk.json")
WHOLE_BODY = Path("runs/whole-brain-pick-place-v2a/interface.json")


def main(output: Path) -> None:
    config = FlyLegConfig.model_validate_json(CONFIG.read_text())
    pack = ConnectomePack.load(PACK)
    pack.validate_b1a_provenance()
    data = kitchen.load(config.split)
    train = data.subset(np.arange(data.obs.shape[0]))
    valid = train["mask"].astype(bool)
    targets, weights = chunk_targets(
        train["actions"], train["mask"], train["mask"].astype(np.float32), config.action_chunk
    )
    y = targets[valid].reshape(int(valid.sum()), -1)
    w = np.repeat(weights[valid], kitchen.ACTION_DIM, axis=1)
    samples = train["obs"][valid]
    leg = front_leg_interface(pack, ANNOTATIONS)
    body = NeuralInterface.load(WHOLE_BODY)
    leg_to_body = NeuralInterface.bind(
        pack, leg.interface.input_body_ids, body.output_body_ids, label="leg senses, body out"
    )
    cases = {
        "leg (kitchen default)": (leg.interface, leg_channels(leg)),
        "whole body (B1a interface)": (body, None),
        "leg senses, whole-body readout": (leg_to_body, leg_channels(leg)),
    }
    result = {"samples": int(valid.sum()), "cases": {}}
    for name, (interface, channels) in cases.items():
        dynamics = (
            leg_dynamics(config, pack, interface)
            if channels is not None
            else RateDynamics(pack, interface, recurrent_gain=config.recurrent_gain)
        )
        policy = BrainPolicy(
            "flyleg",
            dynamics,
            obs_dim=kitchen.FEATURE_DIM,
            action_dim=kitchen.ACTION_DIM,
            neural_steps=config.neural_steps,
            seed=0,
            channels=channels,
            chunk=config.action_chunk,
        )
        policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
        raw = features(policy, train["obs"])[valid]
        curve = fit(standardize(raw), y, w, config.learning_rate)
        result["cases"][name] = {
            "inputs": len(interface.input_body_ids),
            "outputs": len(interface.output_body_ids),
            "l1": curve,
        }
        print(
            name,
            len(interface.input_body_ids),
            len(interface.output_body_ids),
            [round(v, 4) for v in curve[-3:]],
            flush=True,
        )
        mx.clear_cache()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
