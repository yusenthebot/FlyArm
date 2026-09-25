"""Can a trainable nonlinear sensory encoder make the frozen connectome a useful controller?

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/sensory_encoder_probe.py \\
        --demonstrations demos.npz --variant connectome_mlp \\
        --output docs/results/sensory-encoder-probe.json

The demonstrations and held-out split of docs/results/connectome-capacity.json (one episode in
seven held out). Every variant trains with the imitation pipeline's own trainer (window
sampling with burn-in, BPTT through the frozen graph, L1, every demonstrated step weighted
equally as in the frame-wise table), for the same number of epochs; the kept weights are the
epoch with the lowest training loss, so the held-out episodes never select anything.

Variants: an MLP encoder (``--hidden``, ``--activation``) into the measured connectome, a
degree-preserving shuffle of it, or its direct input-to-output synapses only, each with the
linear readout; the linear encoder into the measured connectome; and an MLP policy or a GRU with
the MLP-encoder connectome policy's trainable-parameter count. Each run appends its row to
``--output``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flyarm.config import ManipulationImitationConfig
from flyarm.interfaces import NeuralInterface
from flyarm.manipulation.imitation import build_policy, load_data
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.diagnostics import direct_only_weights
from flyarm.whole_brain.interface import annotation_interface
from flyarm.whole_brain.policy import BrainPolicy
from flyarm.whole_brain.shuffle import shuffle_pack
from flyarm.whole_brain.training import Budget, sequence_loss, train_sequence_policy

VARIANTS = (
    "connectome_mlp",
    "shuffled_mlp",
    "direct_mlp",
    "connectome_linear",
    "matched_mlp",
    "matched_gru",
)
SHUFFLE_SEED = 17000


def split(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    held = np.zeros(data["mask"].shape[0], dtype=bool)
    held[::7] = True
    return (
        {key: value[~held] for key, value in data.items()},
        {key: value[held] for key, value in data.items()},
    )


def current_rms(policy: BrainPolicy, obs: np.ndarray) -> float:
    current = policy.encode(policy.normalize(mx.array(obs)))
    return float(mx.sqrt(mx.mean(current**2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--activation", choices=["tanh", "gelu"], default="tanh")
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--sampling", choices=["windows", "episodes"], default="windows")
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_data(args.demonstrations)
    data = {key: data[key] for key in ("obs", "actions", "mask")}
    train, held = split(data)
    config = ManipulationImitationConfig(
        encoder="linear" if args.variant == "connectome_linear" else "mlp",
        encoder_hidden=args.hidden,
        encoder_activation=args.activation,
        mlp_control="matched",
        encoder_learning_rate_scale=args.encoder_lr_scale,
    )
    pack = ConnectomePack.load(args.pack)
    pack.validate_b1a_provenance()
    interface = annotation_interface(pack)
    budget = build_policy(
        "connectome", config, 0, RateDynamics(pack, interface), 0
    ).trainable_parameter_count()
    kind = {"matched_mlp": "mlp", "matched_gru": "gru"}.get(args.variant, "connectome")
    dynamics = None
    if args.variant in ("connectome_mlp", "connectome_linear"):
        dynamics = RateDynamics(pack, interface)
    elif args.variant == "direct_mlp":
        dynamics = RateDynamics(pack, interface, weights=direct_only_weights(pack, interface))
    elif args.variant == "shuffled_mlp":
        shuffled = shuffle_pack(pack, SHUFFLE_SEED)
        rebound = NeuralInterface.bind(
            shuffled, interface.input_body_ids, interface.output_body_ids, label=interface.label
        )
        dynamics = RateDynamics(shuffled, rebound)
    policy = build_policy(kind, config, 0, dynamics, budget)
    brain = isinstance(policy, BrainPolicy)
    started = time.monotonic()
    extra: dict[str, Any] = {}
    if brain:
        samples = train["obs"][train["mask"] > 0]
        policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
        extra["encoder_scale"] = policy.scale_encoder(samples)
        policy.calibrate_readout(train["obs"], train["mask"], unit_norm=True)
        extra["current_rms_start"] = current_rms(policy, samples[::20])
    # The training set is its own "validation": the kept epoch has the lowest training loss.
    curves, _ = train_sequence_policy(
        policy,
        train,
        train["mask"],
        train,
        train["mask"],
        Budget(
            epochs=args.epochs,
            decoder_warmup_epochs=2 if brain else 0,
            batch_size=16,
            bptt_steps=16,
            learning_rate=1e-3,
            deadline=time.monotonic() + 36_000,
            loss="l1",
            input_learning_rate=1e-3 * args.encoder_lr_scale if brain else None,
            window_batch=32 if args.sampling == "windows" else None,
            burn_in=args.burn_in,
        ),
        0,
        set_normalization=not brain,
    )
    if brain:
        extra["current_rms_end"] = current_rms(policy, train["obs"][train["mask"] > 0][::20])
    row = {
        "variant": args.variant,
        "encoder": config.encoder,
        "encoder_hidden": args.hidden if config.encoder == "mlp" else None,
        "activation": args.activation if config.encoder == "mlp" else None,
        "encoder_lr_scale": args.encoder_lr_scale if brain else None,
        "trainable_parameters": policy.trainable_parameter_count(),
        "parameter_budget": budget,
        "epochs": args.epochs,
        "sampling": args.sampling,
        "burn_in": args.burn_in if args.sampling == "windows" else None,
        "train_l1": sequence_loss(policy, train, train["mask"], loss="l1"),
        "held_out_l1": sequence_loss(policy, held, held["mask"], loss="l1"),
        "training_curve": [round(c["train_loss"], 4) for c in curves],
        "minutes": round((time.monotonic() - started) / 60, 1),
        **extra,
    }
    record = json.loads(args.output.read_text()) if args.output.exists() else {"rows": []}
    record["rows"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "training_curve"}), flush=True)


if __name__ == "__main__":
    main()
