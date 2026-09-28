from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import pytest
from graph_fixtures import make_random_graph

from flyarm.config import WholeBrainConfig
from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.compiler import ConnectomePack

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal device unavailable (e.g. CI VM)", allow_module_level=True)

from flyarm.whole_brain.backend_mlx import RateDynamics  # noqa: E402
from flyarm.whole_brain.experiment import Task, record_rollouts  # noqa: E402
from flyarm.whole_brain.policy import BrainPolicy  # noqa: E402

MODEL = os.environ.get("FLYARM_MODEL")
pytestmark = [
    pytest.mark.skipif(
        not MODEL or not Path(MODEL).is_file(), reason="set FLYARM_MODEL to Panda scene.xml"
    ),
    pytest.mark.skipif(
        os.environ.get("CI") == "true" and platform.system() == "Darwin",
        reason="GitHub macOS runners do not expose a CGL pixel format",
    ),
]


def test_recorded_rollout_has_real_frames_and_output_activity(tmp_path: Path) -> None:
    graph = make_random_graph(n=40, edges=300)
    pack = ConnectomePack.from_graph(graph)
    ids = graph.ids
    interface = NeuralInterface.bind(pack, ids[:6], ids[-5:])
    task = Task(WholeBrainConfig(task="reach", horizon=20), Path(str(MODEL)))
    policy = BrainPolicy(
        "connectome", RateDynamics(pack, interface), obs_dim=20, action_dim=3, seed=0
    )
    try:
        record_rollouts(task, policy, [30000], tmp_path / "rollout.mp4")
    finally:
        task.close()
    frames = imageio.mimread(tmp_path / "rollout.mp4", memtest=False)
    trace = json.loads((tmp_path / "rollout.json").read_text())["trajectory"]
    outputs = np.load(tmp_path / "rollout-outputs.npz")["outputs"]
    assert len(frames) == len(trace) + 1 == 21
    assert int(np.asarray(frames[-1]).max()) > int(np.asarray(frames[-1]).min())
    assert outputs.shape == (20, 5)
    assert all(len(row["action"]) == 3 and "distance" in row for row in trace)
    with pytest.raises(FileExistsError):
        record_rollouts(task, policy, [30000], tmp_path / "rollout.mp4")


def test_phase_selector_saves_every_phase_and_restores_the_best(tmp_path: Path) -> None:
    from mlx.utils import tree_flatten

    from flyarm.whole_brain.experiment import _PhaseSelector

    graph = make_random_graph(n=40, edges=300)
    pack = ConnectomePack.from_graph(graph)
    ids = graph.ids
    interface = NeuralInterface.bind(pack, ids[:6], ids[-4:])
    config = WholeBrainConfig(
        task="pick-place", horizon=200, val_episodes=4, phase_selection="validation_success"
    )
    task = Task(config, Path(str(MODEL)))
    policy = BrainPolicy("connectome", RateDynamics(pack, interface), obs_dim=37, action_dim=4)
    leaves = cast(list[tuple[str, Any]], tree_flatten(policy.parameters()))
    first = {key: np.asarray(value) for key, value in leaves}
    try:
        selector = _PhaseSelector(policy, task, config, tmp_path)
        phases = [{"phase": "behavior_cloning"}]
        selector.score(phases[0])
        policy.decoder.weight = policy.decoder.weight + 1.0
        phases.append({"phase": "dagger_1"})
        selector.score(phases[1])
        selector.restore(phases)
    finally:
        task.close()
    assert (tmp_path / "policy-behavior_cloning.safetensors").is_file()
    assert (tmp_path / "policy-dagger_1.safetensors").is_file()
    assert all("validation_success_rate" in phase for phase in phases)
    # An untrained controller places nothing, so the tie keeps the earlier phase.
    assert [phase["selected"] for phase in phases] == [True, False]
    restored = dict(tree_flatten(policy.parameters()))
    assert np.array_equal(np.asarray(restored["decoder.weight"]), first["decoder.weight"])
