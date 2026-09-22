from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flyarm.assets import SOURCES
from flyarm.graph import Graph
from flyarm.whole_brain import compiler
from flyarm.whole_brain.compiler import ConnectomePack


def small_graph() -> Graph:
    n = 12
    pre = np.repeat(np.arange(n, dtype=np.int64), 3)
    post = np.array(
        [(source + offset) % n for source in range(n) for offset in (1, 3, 5)], dtype=np.int64
    )
    graph = Graph(
        ids=np.arange(100, 100 + n, dtype=np.int64),
        pre=pre,
        post=post,
        contacts=np.arange(1, len(pre) + 1, dtype=np.float32),
        signs=np.array([1, -1, 0] * 4, dtype=np.float32),
        metadata={"dataset": "fixture"},
    )
    graph.validate()
    return graph


def test_from_graph_matches_graph_normalization_and_round_trips(tmp_path: Path) -> None:
    graph = small_graph()
    pack = ConnectomePack.from_graph(graph)
    assert (pack.nodes, pack.edges) == (12, 36)
    dense_pack = np.zeros((12, 12))
    dense_pack[pack.rows(), pack.col_idx] = pack.normalized_weights()
    dense_graph = np.zeros((12, 12))
    dense_graph[graph.post, graph.pre] = graph.normalized_weights()
    np.testing.assert_allclose(dense_pack, dense_graph, rtol=1e-6)
    pack.save(tmp_path / "pack")
    restored = ConnectomePack.load(tmp_path / "pack")
    assert restored.fingerprint() == pack.fingerprint()
    assert np.array_equal(restored.col_idx, pack.col_idx)
    with pytest.raises(FileExistsError):
        pack.save(tmp_path / "pack")


def test_weight_norm_power_interpolates_between_l1_and_l2_rows() -> None:
    pack = ConnectomePack.from_graph(small_graph())
    rows = pack.rows()
    assert np.array_equal(pack.normalized_weights(1.0), pack.normalized_weights())
    for power in (1.0, 1.5, 2.0):
        weights = pack.normalized_weights(power).astype(np.float64)
        norms = np.bincount(rows, weights=np.abs(weights) ** power, minlength=pack.nodes)
        signed_rows = np.unique(rows[weights != 0])
        np.testing.assert_allclose(norms[signed_rows], 1.0, rtol=1e-5)
    # Larger powers shrink each input less.
    assert np.all(
        np.abs(pack.normalized_weights(2.0)) >= np.abs(pack.normalized_weights(1.5)) - 1e-7
    )
    with pytest.raises(ValueError, match="power"):
        pack.normalized_weights(0.5)


def test_load_rejects_tampered_content(tmp_path: Path) -> None:
    pack = ConnectomePack.from_graph(small_graph())
    pack.save(tmp_path / "pack")
    contacts = np.load(tmp_path / "pack" / "contacts.npy")
    contacts[0] += 1
    np.save(tmp_path / "pack" / "contacts.npy", contacts)
    with pytest.raises(ValueError, match="fingerprint"):
        ConnectomePack.load(tmp_path / "pack")


@pytest.mark.parametrize(
    ("col_idx", "message"),
    [
        (np.array([0, 2], dtype=np.int32), "self loops"),
        (np.array([2, 2], dtype=np.int32), "strictly increasing"),
    ],
)
def test_validate_rejects_self_loops_and_duplicates(col_idx: np.ndarray, message: str) -> None:
    pack = ConnectomePack(
        body_ids=np.array([1, 2, 3], dtype=np.int64),
        row_ptr=np.array([0, 2, 2, 2], dtype=np.int64),
        col_idx=col_idx,
        contacts=np.ones(2, dtype=np.int32),
        signs=np.ones(3, dtype=np.int8),
        manifest={},
    )
    with pytest.raises(ValueError, match=message):
        pack.validate()


def test_b1a_provenance_rejects_self_declared_synthetic_pack() -> None:
    pack = ConnectomePack.from_graph(small_graph())
    with pytest.raises(ValueError, match="schema/recipe"):
        pack.validate_b1a_provenance()


def write_raw(raw: Path) -> None:
    raw.mkdir()
    annotations = pd.DataFrame(
        {
            "bodyId": np.array([40, 10, 30, 20, 50], dtype=np.int64),
            "type": ["t40", "t10", "t30", "t20", "glia"],
            "superclass": ["descending_neuron", "ascending_neuron", None, "cb_intrinsic", "glia"],
            "class": [None] * 5,
            "subclass": [None] * 5,
            "instance": [None] * 5,
            "status": ["Traced"] * 5,
        }
    )
    edges = pd.DataFrame(
        {
            # 10->20 (3), 20->40 (5), 10->40 (2, below threshold), 30->40 (unannotated),
            # 20->20 (self loop), 40->50 (glia).
            "body_pre": np.array([10, 20, 10, 30, 20, 40], dtype=np.int64),
            "body_post": np.array([20, 40, 40, 40, 20, 50], dtype=np.int64),
            "weight": np.array([3, 5, 2, 9, 7, 4], dtype=np.int64),
        }
    )
    transmitters = pd.DataFrame(
        {"body": [10, 20, 40], "consensus_nt": ["acetylcholine", "gaba", "dopamine"]}
    )
    for key, frame in (
        ("annotations", annotations),
        ("weights", edges),
        ("neurotransmitters", transmitters),
    ):
        feather.write_feather(pa.Table.from_pandas(frame), raw / SOURCES[key][0])


def test_compile_applies_the_subgraph_recipe(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    write_raw(raw)
    manifest = {"dataset": "fixture", "files": {}}
    monkeypatch.setattr(compiler, "verify_raw_sources", lambda _: manifest)
    pack = compiler.compile_connectome(raw, tmp_path / "pack")
    assert pack.body_ids.tolist() == [10, 20, 40]
    # Row = target: 20 <- 10 (3 contacts), 40 <- 20 (5 contacts).
    assert pack.row_ptr.tolist() == [0, 0, 1, 2]
    assert pack.col_idx.tolist() == [0, 1]
    assert pack.contacts.tolist() == [3, 5]
    assert pack.signs.tolist() == [1, -1, 0]
    assert pack.manifest["synapses_between_eligible_neurons"] == 10
    assert pack.manifest["synapses_kept"] == 8
    neurons = pack.neurons()
    assert neurons.superclass.tolist() == ["ascending_neuron", "cb_intrinsic", "descending_neuron"]
    assert neurons.consensus_nt.tolist() == ["acetylcholine", "gaba", "dopamine"]
    saved = json.loads((tmp_path / "pack" / "manifest.json").read_text())
    assert saved["fingerprint"] == pack.fingerprint()
    assert not (tmp_path / "pack.partial").exists()
