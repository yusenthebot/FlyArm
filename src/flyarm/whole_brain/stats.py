"""Topology statistics for B1a pick-and-place across seeds and shuffle replicates.

The replication unit is the training seed: each seed trains one measured-connectome policy
and one or more degree-preserving shuffles on the same data and test episodes. Two tests:

- seed level: exact sign-flip permutation test on per-seed differences (measured minus the
  mean over that seed's shuffles), one-sided for measured > shuffled;
- episode level: exact binomial test on discordant (measured, shuffle) pairs of the same
  test episode, pooled over seeds and replicates; it treats episodes as independent, so it
  is reported as secondary.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

OUTCOMES = {"grasp": "ever_grasped", "lift": "ever_lifted", "place": "success"}


def _outcomes(model: dict[str, Any], field: str) -> np.ndarray:
    return np.array([bool(episode[field]) for episode in model["clean"]["episodes"]])


def _binomial_tail(wins: int, trials: int) -> float:
    """P(X >= wins) for X ~ Binomial(trials, 1/2)."""
    return sum(math.comb(trials, k) for k in range(wins, trials + 1)) / 2**trials


def sign_flip_p(differences: list[float]) -> float:
    """One-sided exact permutation p for mean(differences) > 0 under sign symmetry."""
    observed = float(np.mean(differences))
    flips = list(itertools.product((1.0, -1.0), repeat=len(differences)))
    extreme = sum(np.mean(np.multiply(signs, differences)) >= observed - 1e-12 for signs in flips)
    return extreme / len(flips)


def topology_statistics(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-seed counts and both tests for every outcome, from pooled run results."""
    seeds = sorted({m["seed"] for m in models if m["kind"] == "connectome"})
    by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        measured = [m for m in models if m["kind"] == "connectome" and m["seed"] == seed]
        shuffles = [m for m in models if m["kind"] == "shuffled" and m["seed"] == seed]
        grus = [m for m in models if m["kind"] == "gru" and m["seed"] == seed]
        if len(measured) != 1 or not shuffles:
            continue
        by_seed[seed] = {"connectome": measured[0], "shuffled": shuffles, "gru": grus}
    report: dict[str, Any] = {
        "seeds": list(by_seed),
        "shuffles_per_seed": {seed: len(v["shuffled"]) for seed, v in by_seed.items()},
        "outcomes": {},
    }
    for name, field in OUTCOMES.items():
        per_seed = []
        wins = losses = 0
        for seed, group in by_seed.items():
            measured = _outcomes(group["connectome"], field)
            shuffled = [_outcomes(model, field) for model in group["shuffled"]]
            for replicate in shuffled:
                if len(replicate) != len(measured):
                    raise ValueError("Measured and shuffled policies saw different test sets")
                wins += int(np.sum(measured & ~replicate))
                losses += int(np.sum(~measured & replicate))
            per_seed.append(
                {
                    "seed": seed,
                    "episodes": len(measured),
                    "connectome": int(measured.sum()),
                    "shuffled": [int(replicate.sum()) for replicate in shuffled],
                    "gru": [int(_outcomes(model, field).sum()) for model in group["gru"]],
                    "difference": float(measured.mean() - np.mean([r.mean() for r in shuffled])),
                }
            )
        differences = [row["difference"] for row in per_seed]
        report["outcomes"][name] = {
            "per_seed": per_seed,
            "connectome_rate": float(np.mean([r["connectome"] / r["episodes"] for r in per_seed]))
            if per_seed
            else None,
            "shuffled_rate": float(
                np.mean([np.mean(r["shuffled"]) / r["episodes"] for r in per_seed])
            )
            if per_seed
            else None,
            "seeds_measured_better": int(sum(d > 0 for d in differences)),
            "seeds_measured_worse": int(sum(d < 0 for d in differences)),
            "seed_level_p": sign_flip_p(differences) if differences else None,
            "episode_pairs": {"measured_only": wins, "shuffled_only": losses},
            "episode_level_p": _binomial_tail(wins, wins + losses) if wins + losses else None,
        }
    return report
