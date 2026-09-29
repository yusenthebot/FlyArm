from __future__ import annotations

import json
from pathlib import Path

import pytest

from flyarm import experiment
from flyarm.config import ExperimentConfig


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
