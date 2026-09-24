"""Preregistered generalization splits of the manipulation benchmark.

Five splits share the task, reward and success rules and differ in what they draw from:

- ``train``: training objects, training furniture ranges, training task templates;
- ``iid_test``: the same distributions as ``train`` on a disjoint seed block;
- ``unseen_objects``: the grasp task's held-out objects, otherwise as ``train``;
- ``unseen_furniture``: every furniture factor from its held-out range (sizes, handle height,
  placement, facing, and the knob handle on the lid), otherwise as ``train``;
- ``unseen_composition``: task templates whose skill orderings never occur in training.

``catalog/manipulation_splits.json`` is the committed record of these definitions (including the
furniture ranges and every template's steps); ``tests/test_manipulation_tasks.py`` checks that
the code still matches it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from flyarm.grasp.objects import data_path
from flyarm.manipulation import furniture as fu
from flyarm.manipulation import tasks as tk


@dataclass(frozen=True)
class Split:
    name: str
    objects: str  # grasp object split: "train" or "test"
    furniture: str  # "train" or "held_out"
    templates: tuple[str, ...]
    seed_start: int

    @property
    def ranges(self) -> fu.FurnitureRanges:
        return fu.TRAIN_RANGES if self.furniture == "train" else fu.HELD_OUT_RANGES


SPLITS: dict[str, Split] = {
    split.name: split
    for split in (
        Split("train", "train", "train", tk.TRAIN_TEMPLATES, 0),
        Split("iid_test", "train", "train", tk.TRAIN_TEMPLATES, 3_000_000),
        Split("unseen_objects", "test", "train", tk.TRAIN_TEMPLATES, 4_000_000),
        Split("unseen_furniture", "train", "held_out", tk.TRAIN_TEMPLATES, 5_000_000),
        Split("unseen_composition", "train", "train", tk.HELD_OUT_TEMPLATES, 6_000_000),
    )
}


def record() -> dict:
    """The preregistration record: splits, furniture ranges and template steps."""
    return {
        "splits": {name: asdict(split) for name, split in SPLITS.items()},
        "furniture_ranges": {
            "train": asdict(fu.TRAIN_RANGES),
            "held_out": asdict(fu.HELD_OUT_RANGES),
        },
        "templates": {
            name: {
                "steps": [list(step) for step in template.steps],
                "starts_in_drawer": list(template.in_drawer),
                "signature": list(template.signature()),
            }
            for name, template in tk.TEMPLATES.items()
        },
    }


def record_path() -> Path:
    return data_path("manipulation_splits.json")


def write_record() -> Path:
    path = record_path()
    path.write_text(json.dumps(record(), indent=2) + "\n")
    return path
