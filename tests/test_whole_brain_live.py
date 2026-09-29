from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from graph_fixtures import make_interface, make_random_graph

from flyarm.whole_brain.compiler import ConnectomePack

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal device unavailable (e.g. CI VM)", allow_module_level=True)

from flyarm.live_common import decode_activity  # noqa: E402
from flyarm.pick_place_env import PandaPickPlaceEnv  # noqa: E402
from flyarm.whole_brain.backend_mlx import RateDynamics  # noqa: E402
from flyarm.whole_brain.live import WholeBrainRuntime  # noqa: E402
from flyarm.whole_brain.policy import BrainPolicy, MlxController  # noqa: E402

MODEL = os.environ.get("FLYARM_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file(), reason="set FLYARM_MODEL to Panda scene.xml"
)


def test_worker_thread_owns_mlx_and_publishes_full_state() -> None:
    pack = ConnectomePack.from_graph(make_random_graph())
    interface = make_interface(pack)
    policy = BrainPolicy("connectome", RateDynamics(pack, interface), obs_dim=37, action_dim=4)
    mx.eval(policy.parameters())
    controllers = {
        "connectome": MlxController(policy),
        "edges_off": MlxController(
            policy.with_dynamics("edges_off", RateDynamics(pack, interface, edges=False))
        ),
    }
    runtime = WholeBrainRuntime(
        PandaPickPlaceEnv(Path(str(MODEL)), horizon=400), controllers, {}, seed=60000
    )
    runtime.start()
    try:
        for _ in range(3):
            runtime.step_once()  # executed on the worker thread, which owns an MLX stream
        snapshot = runtime.snapshot()
        assert snapshot["step"] == 3 and snapshot["hidden_count"] == pack.nodes
        assert np.any(decode_activity(snapshot["hidden_q"]) != 0)
        with pytest.raises(ValueError, match="not available"):
            runtime.set_mode("shuffled")
        runtime.set_mode("edges_off")
        assert runtime.snapshot()["mode"] == "edges_off" and runtime.snapshot()["step"] == 0
    finally:
        runtime.close()
