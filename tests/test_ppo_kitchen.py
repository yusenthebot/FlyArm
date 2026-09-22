from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from flyarm.config import KitchenPPOConfig
from flyarm.rl.ppo_kitchen import kitchen_base_config, load_kitchen_policy

WHOLE_BODY: dict[str, Any] = {
    "split": "complete",
    "interface": "whole_body",
    "readout_calibration": "unit_norm",
    "action_chunk": 1,
    "seeds": [0],
    "policies": ["flyleg"],
}


def _run(tmp_path: Path, name: str, **overrides: Any) -> Path:
    run_root = tmp_path / name
    run_root.mkdir()
    (run_root / "config.json").write_text(json.dumps({**WHOLE_BODY, **overrides}))
    return run_root


def test_a_whole_body_chunk_one_run_is_accepted(tmp_path: Path) -> None:
    config = kitchen_base_config(_run(tmp_path, "good"))
    assert config.interface == "whole_body" and config.action_chunk == 1


def test_the_front_leg_interface_chunked_actions_and_other_splits_are_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="whole-body"):
        kitchen_base_config(_run(tmp_path, "leg", interface="front_leg"))
    with pytest.raises(ValueError, match="action_chunk 1"):
        kitchen_base_config(_run(tmp_path, "chunked", action_chunk=10))
    with pytest.raises(ValueError, match="'complete'"):
        kitchen_base_config(_run(tmp_path, "partial", split="partial"))
    with pytest.raises(FileNotFoundError, match="no config.json"):
        kitchen_base_config(tmp_path / "missing")


def test_a_missing_checkpoint_names_the_file_it_looked_for(tmp_path: Path) -> None:
    run_root = _run(tmp_path, "empty")
    config = KitchenPPOConfig(base_run=str(run_root), base_kind="flyleg", base_seed=0)
    with pytest.raises(FileNotFoundError, match="flyleg-0/policy.safetensors"):
        load_kitchen_policy(config, tmp_path / "pack")


def test_from_scratch_needs_no_imitation_checkpoint() -> None:
    """The reward-only path reads only the base run's interface, so it skips these checks."""
    config = KitchenPPOConfig(from_scratch=True)
    assert config.from_scratch and config.base_run.endswith("push2-s3")
