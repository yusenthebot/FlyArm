"""Final evaluation: re-score checkpoints on every test split with many fresh episodes.

    FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/final_evaluation.py \\
        --policy connectome-dagger=runs/skill-dagger-connectome-002 \\
        --policy connectome-ppo=ppo:runs/ppo-manipulation-skill-dagger-002 \\
        --policy gru-dagger=runs/skill-dagger-gru-002:gru \\
        --episodes-per-template 32 --output docs/results/manipulation-final.json

Each ``--policy`` is ``LABEL=SPEC``:

- ``RUN[:KIND[:SEED]]``: an imitation or skill-DAgger run directory, its selected checkpoint
  ``KIND-SEED/policy.safetensors`` (kind defaults to the run's only policy, seed to 0);
- ``ppo:RUN``: a manipulation PPO run, the checkpoint selected on validation
  (``results.json`` best iteration) over its base run's interface and policy kind;
- ``ppo:RUN@ITERATION``: that PPO run's checkpoint of a given iteration;
- ``file:RUN:KIND:SEED:PATH``: any weights file over an imitation run's interface.

Every checkpoint runs ``--episodes-per-template`` episodes of every template of each split
from true starts, from seeds ``FINAL_OFFSET`` onward in each template's block: selection during
training (imitation, skill DAgger, PPO) uses the train split and the first 8 seeds of each
held-out template's block, so these episodes are disjoint from everything any checkpoint was
selected on. The policy runs deterministically with the observation settings (cue, phase cue,
velocities) of its own run. Per split: successes, the success rate with a Wilson 95% interval,
subgoal fraction and the motion-quality metrics of flyarm.manipulation.rollout.summarize. The
output JSON keeps one row per label; labels already there are skipped (rerun to continue).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from flyarm.config import ManipulationImitationConfig, ManipulationPPOConfig
from flyarm.manipulation import rollout
from flyarm.manipulation.imitation import PolicyActor, Workbench, load_manipulation_policy
from flyarm.whole_brain.policy import SequencePolicy

FINAL_OFFSET = 500  # per template block: selection uses seeds 0 to 7, the final 500 onward
SPLITS = ("iid_test", "unseen_objects", "unseen_furniture", "unseen_composition")
QUALITY = (
    "subgoal_fraction",
    "mean_subgoals",
    "mean_steps_to_success",
    "stray_contact_fraction",
    "mean_abs_action",
    "saturated_fraction",
    "mean_abs_action_change",
    "disturbance",
)


def wilson(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion (95% by default)."""
    if total <= 0:
        raise ValueError("an interval needs at least one episode")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _only_policy(run: Path) -> str:
    kinds = {path.name.rsplit("-", 1)[0] for path in run.iterdir() if path.is_dir()}
    kinds &= {"connectome", "shuffled", "mlp", "gru"}
    if len(kinds) != 1:
        raise ValueError(f"{run} holds {sorted(kinds)}; name the kind (RUN:KIND)")
    return kinds.pop()


def resolve(
    spec: str, pack: Path
) -> tuple[ManipulationImitationConfig, SequencePolicy, dict[str, Any]]:
    """The policy a spec names, the imitation config of its interface, and a provenance record."""
    if spec.startswith("ppo:"):
        body = spec[4:]
        run_text, _, iteration_text = body.partition("@")
        run = Path(run_text)
        ppo = ManipulationPPOConfig.model_validate_json((run / "config.json").read_text())
        if iteration_text:
            iteration = int(iteration_text)
        else:
            best = json.loads((run / "results.json").read_text()).get("best") or {}
            if best.get("iteration") is None:
                raise ValueError(f"{run} has no selected checkpoint yet")
            iteration = int(best["iteration"])
        checkpoint = run / f"policy-{iteration:04d}.safetensors"
        config, policy = load_manipulation_policy(
            Path(ppo.base_run), ppo.base_kind, ppo.base_seed, pack, checkpoint
        )
        record = {
            "source": "ppo",
            "run": str(run),
            "iteration": iteration,
            "base_run": ppo.base_run,
            "kind": ppo.base_kind,
            "seed": ppo.base_seed,
            "checkpoint": str(checkpoint),
        }
        return config, policy, record
    if spec.startswith("file:"):
        run_text, kind, seed_text, path_text = spec[5:].split(":", 3)
        config, policy = load_manipulation_policy(
            Path(run_text), kind, int(seed_text), pack, Path(path_text)
        )
        record = {"source": "file", "run": run_text, "kind": kind, "seed": int(seed_text)}
        return config, policy, record | {"checkpoint": path_text}
    parts = spec.split(":")
    run = Path(parts[0])
    kind = parts[1] if len(parts) > 1 else _only_policy(run)
    seed = int(parts[2]) if len(parts) > 2 else 0
    config, policy = load_manipulation_policy(run, kind, seed, pack)
    checkpoint = run / f"{kind}-{seed}" / "policy.safetensors"
    record = {"source": "imitation", "run": str(run), "kind": kind, "seed": seed}
    return config, policy, record | {"checkpoint": str(checkpoint)}


def evaluate(
    config: ManipulationImitationConfig,
    policy: SequencePolicy,
    model: Path,
    asset_root: Path,
    per_template: int,
    splits: list[str],
) -> dict[str, Any]:
    bench = Workbench(model, asset_root, config.cue, config.velocities, config.phase_cue)
    out: dict[str, Any] = {}
    for split in splits:
        started = time.monotonic()
        episodes = rollout.plan(split, per_template, FINAL_OFFSET)
        (log,) = bench.run([episodes], lambda n: PolicyActor(policy, n))
        summary = rollout.summarize(log)
        low, high = wilson(summary["successes"], summary["episodes"])
        out[split] = {
            "episodes": summary["episodes"],
            "successes": summary["successes"],
            "success_rate": round(summary["success_rate"], 4),
            "wilson_95": [round(low, 4), round(high, 4)],
            **{key: summary[key] for key in QUALITY},
            "per_template": summary["per_template"],
            "seconds": round(time.monotonic() - started, 1),
        }
        print(
            f"  {split}: {summary['successes']}/{summary['episodes']} "
            f"({summary['success_rate']:.3f}, 95% {low:.3f} to {high:.3f}), "
            f"subgoals {summary['subgoal_fraction']:.3f}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, help="LABEL=SPEC")
    parser.add_argument("--episodes-per-template", type=int, default=32)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--pack", type=Path, default=Path("data/whole_brain/malecns-v1.0-c3"))
    parser.add_argument(
        "--model", type=Path, default=Path("assets/menagerie/franka_emika_panda/scene.xml")
    )
    parser.add_argument("--asset-root", type=Path, default=Path("assets/objects"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.episodes_per_template <= rollout.TEMPLATE_STRIDE - FINAL_OFFSET:
        raise ValueError("episodes per template must fit in the final seed block")
    record: dict[str, Any] = (
        json.loads(args.output.read_text())
        if args.output.is_file()
        else {
            "protocol": {
                "episodes_per_template": args.episodes_per_template,
                "seed_offset": FINAL_OFFSET,
                "splits": args.splits,
                "interval": "Wilson score, 95%",
                "note": "seeds disjoint from every selection episode; deterministic policies",
            },
            "policies": {},
        }
    )
    if record["protocol"]["episodes_per_template"] != args.episodes_per_template:
        raise ValueError("the output was written with another number of episodes per template")
    for item in args.policy:
        label, _, spec = item.partition("=")
        if not spec:
            raise ValueError(f"--policy needs LABEL=SPEC, got {item!r}")
        if label in record["policies"]:
            print(f"{label}: already scored, skipped", flush=True)
            continue
        print(f"{label} ({spec})", flush=True)
        config, policy, provenance = resolve(spec, args.pack)
        splits = evaluate(
            config, policy, args.model, args.asset_root, args.episodes_per_template, args.splits
        )
        record["policies"][label] = {
            **provenance,
            "trainable_parameters": policy.trainable_parameter_count(),
            "observation": {
                "cue": config.cue,
                "phase_cue": config.phase_cue,
                "velocities": config.velocities,
            },
            "splits": splits,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
