from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from graph_fixtures import make_random_graph

from flyarm.flyleg.interface import front_leg_interface, front_leg_report
from flyarm.whole_brain.compiler import ConnectomePack

# (superclass, class, subclass, entryNerve, side): the selection rule must pick exactly
# left ProLN proprioceptors, every head sensory neuron and left front-leg motor neurons.
ROWS = [
    ("vnc_sensory", "mechanosensory_proprioceptive", "chordotonal organ", "ProLN", "L"),  # in
    ("vnc_sensory", "mechanosensory_proprioceptive", "hair plate", "ProLN", "L"),  # in
    ("vnc_sensory", "mechanosensory_proprioceptive", "chordotonal organ", "ProLN", "R"),
    ("vnc_sensory", "mechanosensory_proprioceptive", "chordotonal organ", "MesoLN", "L"),
    ("vnc_sensory", "mechanosensory_tactile", "leg bristle", "ProLN", "L"),
    ("cb_sensory", "olfactory", None, None, "L"),  # exteroceptor
    ("cb_sensory", "mechanosensory", "auditory", None, "R"),  # exteroceptor
    ("vnc_motor", None, "fl", None, "L"),  # motor
    ("vnc_motor", None, "fl", None, "R"),
    ("vnc_motor", None, "hl", None, "L"),
]


def write_annotations(path: Path, ids: np.ndarray) -> None:
    rows = ROWS + [("cb_intrinsic", None, None, None, "L")] * (len(ids) - len(ROWS))
    frame = pd.DataFrame(
        {
            "bodyId": ids,
            "superclass": [row[0] for row in rows],
            "class": [row[1] for row in rows],
            "subclass": [row[2] for row in rows],
            "type": [f"t{index}" for index in range(len(ids))],
            "entryNerve": [row[3] for row in rows],
            "exitNerve": [None] * len(ids),
            "somaSide": [None if row[0].endswith("sensory") else row[4] for row in rows],
            "rootSide": [row[4] for row in rows],
        }
    )
    feather.write_feather(pa.Table.from_pandas(frame), path)


def test_front_leg_rule_selects_left_proprioceptors_head_sensors_and_leg_motor(
    tmp_path: Path,
) -> None:
    graph = make_random_graph(n=40, edges=300)
    pack = ConnectomePack.from_graph(graph)
    annotations = tmp_path / "annotations.feather"
    write_annotations(annotations, pack.body_ids)
    leg = front_leg_interface(pack, annotations)
    ids = pack.body_ids
    assert leg.proprioceptors.tolist() == ids[[0, 1]].tolist()
    assert leg.exteroceptors.tolist() == ids[[5, 6]].tolist()
    assert leg.motor_neurons.tolist() == ids[[7]].tolist()
    assert leg.channel_sizes == (2, 2)
    assert leg.interface.input_body_ids.tolist() == ids[[0, 1, 5, 6]].tolist()
    report = front_leg_report(pack, leg, annotations)
    assert report["proprioceptors"]["count"] == 2
    assert report["motor_neurons"]["count"] == 1
