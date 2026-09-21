from __future__ import annotations

import json
import os
import platform
from pathlib import Path

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
