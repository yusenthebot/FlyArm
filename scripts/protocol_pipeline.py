"""Run the 2026-09-30 report protocol (research log, "Report rigor plan") for one seed.

    PYTHONPATH=src:scripts .venv/bin/python scripts/protocol_pipeline.py --kind connectome --seed 1

Steps, each skipped when its output exists, so a stopped pipeline is rerun to continue:
1. skill-level DAgger with configs/skill-dagger-{kind}-nophase-seed{seed}.json (resumed);
2. imitation selection: rounds 7, 8 and 9 on the 112-episode validation set, most successes,
   ties to the later round (docs/results/protocol/TAG-selection.json);
3. PPO with configs/ppo-manipulation-nophase.json, only base_run, base_kind, base_seed, seed and
   init_checkpoint changed (written to configs/ppo-{kind}-nophase-seed{seed}.json);
4. PPO selection: iterations 150, 300 and 500, same rule;
5. final evaluation of the selected imitation and PPO checkpoints, 32 episodes per template on
   every test split (docs/results/protocol/TAG-final.json).
One file per seed and kind, so several pipelines can run at once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

ROUNDS = (7, 8, 9)
ITERATIONS = (150, 300, 500)
RESULTS = Path("docs/results/protocol")


def run(args: list[str], log: Path) -> None:
    env = {**os.environ, "PYTHONPATH": "src:scripts", "FLYARM_SIM_THREADS": "4"}
    print("$", " ".join(args), flush=True)
    with log.open("a") as out:
        subprocess.run(args, check=True, stdout=out, stderr=subprocess.STDOUT, env=env)  # nosec B603


def select(labels: dict[str, str], output: Path, log: Path) -> str:
    """Score every LABEL=SPEC on the validation set; the label with most successes, ties late."""
    policies = [arg for label, spec in labels.items() for arg in ("--policy", f"{label}={spec}")]
    run(
        [sys.executable, "scripts/manipulation_select.py", "--per-template", "16"]
        + ["--output", str(output), *policies],
        log,
    )
    scores = json.loads(output.read_text())
    best = max(labels, key=lambda label: (scores[label]["successes"], list(labels).index(label)))
    print("selected", best, {label: scores[label]["successes"] for label in labels}, flush=True)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("connectome", "shuffled"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    kind, seed = args.kind, args.seed
    tag = f"{kind}-nophase-seed{seed}"
    log = Path(f"runs/protocol-{tag}.log")
    imitation = Path(f"runs/skill-dagger-{tag}")
    member = imitation / f"{kind}-{seed}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    selection = RESULTS / f"{tag}-selection.json"
    # A skill-DAgger run started by hand for this seed: wait for it rather than resume twice.
    running = ["pgrep", "-f", "--", f"--output {imitation}$"]
    while subprocess.run(running, capture_output=True).returncode == 0:  # nosec B603
        time.sleep(300)
    if not (member / "policy.safetensors").is_file():
        command = ["flyarm", "manipulation", "skill-dagger"]
        command += ["--config", f"configs/skill-dagger-{tag}.json", "--output", str(imitation)]
        run(
            [str(Path(sys.executable).parent / command[0]), *command[1:]]
            + (["--resume"] if imitation.exists() else []),
            log,
        )

    weights = f"{member}/round-{{:02d}}/policy.safetensors"
    rounds = {
        f"{tag}-r{r:02d}": f"file:{imitation}:{kind}:{seed}:{weights.format(r)}" for r in ROUNDS
    }
    start = select(rounds, selection, log)
    round_index = int(start.rsplit("-r", 1)[1])

    ppo = Path(f"runs/ppo-{tag}")
    ppo_config = Path(f"configs/ppo-{tag}.json")
    config = json.loads(Path("configs/ppo-manipulation-nophase.json").read_text())
    config |= {
        "base_run": str(imitation),
        "base_kind": kind,
        "base_seed": seed,
        "seed": seed,
        "init_checkpoint": f"{member}/round-{round_index:02d}/policy.safetensors",
    }
    ppo_config.write_text(json.dumps(config, indent=2) + "\n")
    last = ppo / f"policy-{ITERATIONS[-1]:04d}.safetensors"
    if not last.is_file():
        if ppo.exists():
            raise RuntimeError(f"{ppo} exists without {last.name}; PPO does not resume, move it")
        command = ["rl", "manipulation", "--config", str(ppo_config), "--output", str(ppo)]
        run([str(Path(sys.executable).parent / "flyarm"), *command], log)

    iterations = {f"{tag}-ppo{i}": f"ppo:{ppo}@{i}" for i in ITERATIONS}
    chosen = select(iterations, selection, log)
    iteration = int(chosen.rsplit("ppo", 1)[1])

    run(
        [sys.executable, "scripts/final_evaluation.py", "--episodes-per-template", "32"]
        + ["--output", str(RESULTS / f"{tag}-final.json")]
        + ["--policy", f"{tag}-imitation={rounds[start]}"]
        + ["--policy", f"{tag}-ppo=ppo:{ppo}@{iteration}"],
        log,
    )
    print("done", tag, flush=True)


if __name__ == "__main__":
    main()
