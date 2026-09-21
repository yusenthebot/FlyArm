"""The fly's left front leg, from MaleCNS annotations: its proprioceptors, its motor neurons,
and the head sensory neurons that carry exteroception.

Every selection is a rule over official annotation columns (superclass, subclass, entry
nerve, side), so the interface is reproducible and auditable; nothing is hand-picked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.interface import _histogram, hop_distances

SIDE = "L"
PROPRIOCEPTOR_SUBCLASSES = ("chordotonal organ", "hair plate", "campaniform sensilla")
FRONT_LEG_NERVE = "ProLN"
LABEL = "left front leg: ProLN proprioceptors + head sensory in, fl motor neurons out"
PROPRIOCEPTIVE_LABEL = "left front leg: ProLN proprioceptors in, fl motor neurons out"
COLUMNS = ["bodyId", "superclass", "class", "subclass", "type", "entryNerve", "exitNerve"]


@dataclass(frozen=True)
class FrontLegInterface:
    """Ordered channels bound to one pack: inputs = proprioceptors then head sensory."""

    interface: NeuralInterface
    proprioceptors: np.ndarray
    exteroceptors: np.ndarray
    motor_neurons: np.ndarray

    @property
    def channel_sizes(self) -> tuple[int, int]:
        return len(self.proprioceptors), len(self.exteroceptors)


def _annotations(pack: ConnectomePack, path: Path) -> pd.DataFrame:
    table = feather.read_table(path, columns=[*COLUMNS, "somaSide", "rootSide"]).to_pandas()
    table = table.set_index("bodyId").reindex(pack.body_ids)
    if table.superclass.isna().any():
        raise ValueError("Annotation table does not cover every pack neuron")
    return table.assign(side=table.somaSide.fillna(table.rootSide))


def front_leg_interface(
    pack: ConnectomePack, annotations_path: Path, *, include_head: bool = True
) -> FrontLegInterface:
    table = _annotations(pack, annotations_path)
    ids = pack.body_ids
    left = table.side == SIDE
    proprio = (
        (table.superclass == "vnc_sensory")
        & table.subclass.isin(PROPRIOCEPTOR_SUBCLASSES)
        & (table.entryNerve == FRONT_LEG_NERVE)
        & left
    )
    head = table.superclass == "cb_sensory"
    motor = (table.superclass == "vnc_motor") & (table.subclass == "fl") & left
    proprioceptors = ids[proprio.to_numpy()]
    exteroceptors = ids[head.to_numpy()] if include_head else np.array([], dtype=np.int64)
    motor_neurons = ids[motor.to_numpy()]
    interface = NeuralInterface.bind(
        pack,
        np.concatenate((proprioceptors, exteroceptors)),
        motor_neurons,
        label=LABEL if include_head else PROPRIOCEPTIVE_LABEL,
    )
    return FrontLegInterface(interface, proprioceptors, exteroceptors, motor_neurons)


def leg_channels(leg: FrontLegInterface) -> list[tuple[int, int, int]]:
    """Encoder channels over the position features: joints -> proprioceptors, scene -> head."""
    from flyarm.benchmarks import kitchen

    channels = [
        (kitchen.PROPRIOCEPTION.start, kitchen.PROPRIOCEPTION.stop, len(leg.proprioceptors))
    ]
    if len(leg.exteroceptors):
        channels.append(
            (kitchen.EXTEROCEPTION.start, kitchen.EXTEROCEPTION.stop, len(leg.exteroceptors))
        )
    return channels


def front_leg_report(
    pack: ConnectomePack, leg: FrontLegInterface, annotations_path: Path
) -> dict[str, Any]:
    """Channel composition and directed reachability of the motor neurons per channel."""
    table = _annotations(pack, annotations_path)
    index = {int(body): position for position, body in enumerate(pack.body_ids)}
    rows = pack.rows()
    by_source = np.argsort(pack.col_idx, kind="stable")
    source_ptr = np.concatenate(([0], np.cumsum(np.bincount(pack.col_idx, minlength=pack.nodes))))
    motor = np.array([index[int(body)] for body in leg.motor_neurons])

    def channel(body_ids: np.ndarray) -> dict[str, Any]:
        if len(body_ids) == 0:
            return {"count": 0}
        positions = np.array([index[int(body)] for body in body_ids])
        hops = hop_distances(source_ptr, rows[by_source], positions)[motor]
        members = table.iloc[positions]
        return {
            "count": len(positions),
            "class_counts": members["class"].fillna("unknown").value_counts().to_dict(),
            "subclass_counts": members.subclass.fillna("unknown").value_counts().head(12).to_dict(),
            "motor_neuron_hops": _histogram(hops),
        }

    motor_rows = table.iloc[motor]
    return {
        **leg.interface.to_dict(),
        "pack_fingerprint": pack.fingerprint(),
        "rules": {
            "proprioceptors": {
                "superclass": "vnc_sensory",
                "subclass": list(PROPRIOCEPTOR_SUBCLASSES),
                "entryNerve": FRONT_LEG_NERVE,
                "side": SIDE,
            },
            "exteroceptors": {"superclass": "cb_sensory"},
            "motor_neurons": {"superclass": "vnc_motor", "subclass": "fl", "side": SIDE},
        },
        "proprioceptors": channel(leg.proprioceptors),
        "exteroceptors": channel(leg.exteroceptors),
        "motor_neurons": {
            "count": len(motor),
            "types": motor_rows.type.fillna("untyped").value_counts().to_dict(),
            "exit_nerves": motor_rows.exitNerve.fillna("unknown").value_counts().to_dict(),
        },
        "vision": "not used: photoreceptors are histaminergic (sign 0 in this recipe)",
    }
