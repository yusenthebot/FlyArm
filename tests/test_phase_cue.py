from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flyarm.manipulation import phases as ph
from flyarm.manipulation import rollout
from flyarm.manipulation import sim as ms

MODEL = os.environ.get("FLYARM_MODEL")
OBJECTS = Path(os.environ.get("FLYARM_OBJECTS", "assets/objects"))
needs_env = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file() or not OBJECTS.is_dir(),
    reason="set FLYARM_MODEL to the Panda scene.xml and fetch the grasp objects",
)


def _model_path() -> Path:
    if MODEL is None:
        pytest.skip("set FLYARM_MODEL to the Panda scene.xml")
    return Path(MODEL)


def phase_slice() -> slice:
    names = [name for name, _ in ms.CUE_FIELDS]
    start = ms.CUE_START + sum(size for _, size in ms.CUE_FIELDS[: names.index("motor_phase")])
    return slice(start, start + ph.PHASE_DIM)


def test_the_cue_has_one_slot_per_observable_phase_before_the_progress() -> None:
    assert dict(ms.CUE_FIELDS)["motor_phase"] == ph.PHASE_DIM == len(ph.PHASE_NAMES)
    assert ms.CUE_FIELDS[-1][0] == "progress"  # the cue's zeroing keeps the last two fields


@needs_env
def test_the_phase_cue_is_one_hot_when_on_and_zero_in_the_control() -> None:
    from flyarm.manipulation.teacher import ManipulationTeacher

    episodes = rollout.plan("train", 1, rollout.VALIDATION_OFFSET, templates=["put_away"])
    seen = {}
    for on in (True, False):
        env = rollout.make_env(
            _model_path(), episodes, asset_root=OBJECTS, velocities=False, phase_cue=on
        )
        obs = env.reset(seeds=np.array(episodes.seeds), templates=list(episodes.templates))
        teacher = ManipulationTeacher(env)
        teacher.reset()
        rows = []
        for _ in range(250):
            rows.append(obs[0].copy())
            obs = env.step(teacher.act().astype(np.float64), auto_reset=False).obs
        seen[on] = np.array(rows)
    cue = seen[True][:, phase_slice()]
    assert np.all(cue.sum(1) == 1.0) and set(np.unique(cue)) == {0.0, 1.0}
    assert len(np.unique(cue.argmax(1))) >= 4  # approach, descend, close, move at least
    assert np.all(seen[False][:, phase_slice()] == 0.0)
    rest = np.ones(ms.OBS_DIM, bool)
    rest[phase_slice()] = False
    np.testing.assert_array_equal(seen[True][:, rest], seen[False][:, rest])  # nothing else


@needs_env
def test_the_observable_phase_follows_the_teacher_on_its_own_path() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from phase_cue_agreement import measure

    row = measure(_model_path(), "train", 1, rollout.VALIDATION_OFFSET)
    assert row["agreement"] > 0.93
