from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from flyarm.benchmarks import _robotics_compat, kitchen

MODEL = """
<mujoco><worldbody>
  <body><joint name="hinge" type="hinge"/><geom size=".1"/></body>
  <body pos="1 0 0"><joint name="slide" type="slide"/><geom size=".1"/></body>
  <body pos="2 0 0"><joint name="ball" type="ball"/><geom size=".1"/></body>
  <body pos="3 0 0"><freejoint name="free"/><geom size=".1"/></body>
</worldbody></mujoco>
"""


def test_compat_accessors_follow_joint_types() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    data = mujoco.MjData(model)
    data.qpos[:] = np.arange(model.nq)
    data.qvel[:] = np.arange(model.nv) * 10
    assert _robotics_compat.get_joint_qpos(model, data, "hinge").tolist() == [0.0]
    assert _robotics_compat.get_joint_qpos(model, data, "slide").tolist() == [1.0]
    assert len(_robotics_compat.get_joint_qpos(model, data, "ball")) == 4
    assert len(_robotics_compat.get_joint_qpos(model, data, "free")) == 7
    assert len(_robotics_compat.get_joint_qvel(model, data, "free")) == 6
    _robotics_compat.set_joint_qpos(model, data, "slide", 5.0)
    assert data.qpos[1] == 5.0
    with pytest.raises(ValueError, match="incorrect shape"):
        _robotics_compat.set_joint_qvel(model, data, "free", [1.0, 2.0])
    with pytest.raises(ValueError, match="not part"):
        _robotics_compat.get_joint_qpos(model, data, "missing")


def test_position_features_drop_every_velocity() -> None:
    observation = np.arange(59, dtype=np.float32)
    seen: list[np.ndarray] = []

    class Echo:
        def reset(self) -> None:
            pass

        def act(self, features: np.ndarray) -> np.ndarray:
            seen.append(features)
            return np.zeros(9, dtype=np.float32)

    kitchen.PositionFeatures(Echo()).act(observation)
    assert seen[0].tolist() == [*range(0, 9), *range(18, 39)]
    assert kitchen.FEATURE_DIM == 30


def _dataset_available() -> bool:
    return (Path.home() / ".minari" / "datasets" / "D4RL" / "kitchen" / "complete-v2").is_dir()


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_complete_split_replays_to_a_full_score() -> None:
    data = kitchen.load("complete", download=False)
    assert data.obs.shape == (19, 236, 59) and data.provenance["transitions"] == 4209
    env = kitchen.recover_env("complete")
    try:
        first = data.actions[0][data.mask[0].astype(bool)]
        replay = kitchen.evaluate(env, None, [0], replay_actions=first)
        zero = kitchen.evaluate(env, None, [0])
    finally:
        env.close()
    assert replay["normalized_score"] == 100.0
    assert zero["normalized_score"] == 0.0
    assert data.subset(np.array([0]))["obs"].shape == (1, 236, kitchen.FEATURE_DIM)
