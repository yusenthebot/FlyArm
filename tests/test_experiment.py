from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flyarm import experiment
from flyarm.assets import SOURCE_SHA256
from flyarm.config import ExperimentConfig
from flyarm.graph import RECIPE_VERSION, Graph


def test_self_declared_provenance_does_not_admit_synthetic_graph() -> None:
    graph = Graph(
        ids=np.arange(256, dtype=np.int64),
        pre=np.array([0]),
        post=np.array([1]),
        contacts=np.array([3.0], dtype=np.float32),
        signs=np.ones(256),
        metadata={
            "schema_version": 1,
            "recipe_version": RECIPE_VERSION,
            "dataset": "MaleCNS v1.0",
            "sources": {"files": {key: {"sha256": value} for key, value in SOURCE_SHA256.items()}},
        },
    )
    with pytest.raises(ValueError, match="pinned 256-node graph"):
        graph.validate_mvp_provenance()
    original = graph.fingerprint()
    graph.contacts[0] += 1
    assert graph.fingerprint() != original


def test_failed_run_is_marked_and_cannot_be_overwritten(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"

    def fail(_graph, _model, destination, _config):
        destination.mkdir()
        experiment.save_json(destination / "results.json", {"status": "running", "models": []})
        raise TimeoutError("test budget")

    monkeypatch.setattr(experiment, "_run_experiment", fail)
    with pytest.raises(TimeoutError):
        experiment.run_experiment(Path("unused"), Path("unused"), output, ExperimentConfig())
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "failed"
    assert result["error_type"] == "TimeoutError"
    with pytest.raises(FileExistsError):
        experiment.run_experiment(Path("unused"), Path("unused"), output, ExperimentConfig())
    assert json.loads((output / "results.json").read_text()) == result
