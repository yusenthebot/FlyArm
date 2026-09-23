"""Drop the objects the scripted teacher cannot grasp, then split the rest by object.

    PYTHONPATH=src .venv/bin/python scripts/grasp_split.py

Reads the per-object teacher table written by ``scripts/grasp_smoke.py``
(``docs/results/grasp-teacher.json``), drops every object whose teacher success is below
``--min-success`` (0.8), and splits the kept objects per family into train (about 2/3) and a
held-out test split (about 1/3) with a seeded shuffle, so both splits contain every family.
Writes the committed split ``src/flyarm/grasp/catalog/split.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyarm.grasp.objects import data_path, load_manifest, stratified_split

DEFAULT_TABLE = Path("docs/results/grasp-teacher.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--min-success", type=float, default=0.8)
    parser.add_argument("--test-fraction", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    manifest = load_manifest()
    rows = {
        row["object"]: row for row in json.loads(arguments.table.read_text())["teacher"]["objects"]
    }
    missing = sorted(set(manifest) - set(rows))
    if missing:
        raise SystemExit(f"no teacher result for {missing}; re-run scripts/grasp_smoke.py")
    dropped = {
        name: rows[name]["success_rate"]
        for name in manifest
        if rows[name]["success_rate"] < arguments.min_success
    }
    kept = [item for name, item in manifest.items() if name not in dropped]
    split = stratified_split(kept, arguments.test_fraction, arguments.seed)
    record = {
        "rule": (
            f"objects with scripted-teacher success below {arguments.min_success} over "
            f"{rows[next(iter(rows))]['episodes']} episodes are dropped; the rest are split per "
            f"family with test fraction {arguments.test_fraction:.3f} and seed {arguments.seed}"
        ),
        "train": split["train"],
        "test": split["test"],
        "dropped": sorted(dropped),
        "dropped_teacher_success": {name: dropped[name] for name in sorted(dropped)},
    }
    target = data_path("split.json")
    target.write_text(json.dumps(record, indent=2) + "\n")
    print(f"train {len(split['train'])}, test {len(split['test'])}, dropped {sorted(dropped)}")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
