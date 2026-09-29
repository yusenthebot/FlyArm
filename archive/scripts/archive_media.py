"""Move superseded rollout media out of the dashboard into runs/_archive (reversible).

Every moved file keeps its path under runs/_archive, and runs/_archive/MANIFEST.json lists
each move with its reason. `--restore` moves everything back. Usage:

    uv run python scripts/archive_media.py            # archive
    uv run python scripts/archive_media.py --restore  # undo
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
ARCHIVE = RUNS / "_archive"
MANIFEST = ARCHIVE / "MANIFEST.json"
SIDECARS = (".mp4", ".json", "-outputs.npz")

KEEP_PROPRIO = {"kitchen-complete-lesions-seed2-episode0", "kitchen-complete-flyleg-seed2"}


def planned() -> list[tuple[Path, str]]:
    moves: list[tuple[Path, str]] = []
    videos = RUNS / "overnight" / "videos"
    for path in sorted(videos.glob("kitchen-*")):
        moves.append((path, "first kitchen protocol: every controller near 0, superseded by v2"))
    for path in sorted((videos / "proprio").glob("*")):
        if path.stem.removesuffix("-outputs") not in KEEP_PROPRIO:
            moves.append((path, "proprioception-only variant: only the seed-2 skill is kept"))
    for path in sorted((RUNS / "overnight" / "site").rglob("*.mp4")):
        moves.append((path, "copy of runs/overnight/videos made for the published page"))
    for run in (
        "pick-place-001",
        "pick-place-dagger-001",
        "pick-place-observable-001",
        "pick-place-smoke-001",
        "reach-001",
        "reach-002",
        "reach-003",
        "smoke-001",
    ):
        for path in sorted((RUNS / run).rglob("*.mp4")):
            moves.append((path, "256-node subgraph prototype, not part of the paper"))
    for path in sorted((RUNS / "whole-brain-pick-place-001" / "connectome-0").glob("rollout*")):
        moves.append((path, "duplicate of overnight/videos/pick-place-connectome-seed0"))
    return [(path, reason) for path, reason in moves if path.is_file()]


def archive() -> None:
    record = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else []
    for path, reason in planned():
        target = ARCHIVE / path.relative_to(RUNS)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), target)
        record.append({"from": str(path.relative_to(RUNS)), "reason": reason})
    ARCHIVE.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(record, indent=1) + "\n")
    print(f"archived {len(record)} files; manifest {MANIFEST}")


def restore() -> None:
    record = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else []
    for entry in record:
        source = ARCHIVE / entry["from"]
        if source.is_file():
            target = RUNS / entry["from"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), target)
    MANIFEST.unlink(missing_ok=True)
    print(f"restored {len(record)} files")


if __name__ == "__main__":
    restore() if "--restore" in sys.argv else archive()
