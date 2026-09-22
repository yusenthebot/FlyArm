"""Connectome-versus-shuffle statistics pooled over whole-brain runs.

Reads the models of every given run directory, runs flyarm.whole_brain.stats.topology_statistics
(seed-level exact sign-flip test and episode-level exact binomial test for grasp, lift and place)
and writes the report as JSON. Usage:

    PYTHONPATH=src uv run python scripts/topology_report.py OUTPUT.json RUN [RUN ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from flyarm.whole_brain.stats import topology_statistics


def _models(run: Path) -> list[dict[str, Any]]:
    results = json.loads((run / "results.json").read_text())
    if results.get("status") != "complete":
        raise SystemExit(f"{run} is not complete (status {results.get('status')!r})")
    return list(results["models"])


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value.item() if hasattr(value, "item") else value


def main(output: Path, runs: list[Path]) -> None:
    models = [model for run in runs for model in _models(run)]
    report = {"runs": [str(run) for run in runs], **topology_statistics(models)}
    output.write_text(json.dumps(_plain(report), indent=2) + "\n")
    for name, outcome in report["outcomes"].items():
        print(
            f"{name}: connectome {outcome['connectome_rate']:.3f} shuffled "
            f"{outcome['shuffled_rate']:.3f}; seeds better {outcome['seeds_measured_better']} "
            f"worse {outcome['seeds_measured_worse']} (p {float(outcome['seed_level_p']):.3g}); "
            f"episodes {outcome['episode_pairs']} (p {outcome['episode_level_p']:.3g})"
        )
    print(output)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), [Path(arg) for arg in sys.argv[2:]])
