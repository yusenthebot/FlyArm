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


def test_tracker_returns_the_demo_action_on_the_demo_and_corrects_joint_error() -> None:
    from flyarm.benchmarks.kitchen_expert import (
        ACTION_TO_VELOCITY,
        CONTROL_DT,
        DemonstrationTracker,
    )

    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, (2, 5, 59)).astype(np.float32)
    actions = rng.uniform(-0.2, 0.2, (2, 5, 9)).astype(np.float32)
    mask = np.ones((2, 5), np.float32)
    mask[1, 3:] = 0
    tracker = DemonstrationTracker(obs, actions, mask, gain=0.5)
    assert np.allclose(tracker.label(obs[1, 2]), actions[1, 2])
    shifted = obs[0, 1].copy()
    shifted[0] += 0.01  # a small joint error keeps the same nearest demonstration state
    correction = 0.5 * -0.01 / (ACTION_TO_VELOCITY * CONTROL_DT)
    assert tracker.label(shifted)[0] == pytest.approx(actions[0, 1, 0] + correction, abs=1e-6)
    # Padding past the end of an episode is never a neighbour.
    assert not np.allclose(tracker.label(obs[1, 4]), actions[1, 4])
    with pytest.raises(ValueError, match="gain"):
        DemonstrationTracker(obs, actions, mask, gain=1.5)


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_tracker_solves_the_benchmark_from_perturbed_starts() -> None:
    from flyarm.benchmarks.kitchen_expert import DemonstrationTracker

    tracker = DemonstrationTracker.from_data(kitchen.load("complete", download=False))
    env = kitchen.recover_env("complete")
    try:
        result = kitchen.evaluate(env, tracker, [0, 1], initial_joint_offset=0.1)
    finally:
        env.close()
    assert result["normalized_score"] == 100.0


def test_dagger_is_only_configured_where_the_tracker_is_validated() -> None:
    from pydantic import ValidationError

    from flyarm.config import FlyLegConfig

    assert FlyLegConfig(split="complete", dagger_iterations=2).dagger_iterations == 2
    with pytest.raises(ValidationError, match="tracker"):
        FlyLegConfig(split="partial", dagger_iterations=1)
    with pytest.raises(ValidationError, match="disjoint"):
        FlyLegConfig(split="complete", dagger_iterations=1, seeds=[3, 12])
    with pytest.raises(ValidationError, match="init_from"):
        FlyLegConfig(split="complete", init_from="runs/earlier")


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_dagger_rollouts_follow_the_learner_and_carry_teacher_labels() -> None:
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("MLX Metal device unavailable")
    from flyarm.benchmarks.kitchen_expert import DemonstrationTracker
    from flyarm.flyleg.experiment import _dagger_rollouts
    from flyarm.whole_brain.policy import MLPPolicy

    data = kitchen.load("complete", download=False)
    tracker = DemonstrationTracker.from_data(data)
    policy = MLPPolicy(
        obs_dim=kitchen.FEATURE_DIM, action_dim=kitchen.ACTION_DIM, hidden=8, chunk=3
    )
    env = kitchen.recover_env("complete")
    try:
        rollouts, stats = _dagger_rollouts(policy, tracker, env, 1, seed=0, iteration=0)
    finally:
        env.close()
    steps = int(rollouts["mask"][0].sum())
    assert rollouts["obs"].shape == (1, 280, kitchen.FEATURE_DIM) and steps == 280
    assert stats["labelled_states"] == steps and stats["rollout_mean_tasks"] == 0.0
    # Labels are the teacher's actions at the learner's own states, not the learner's actions.
    first = np.zeros(59, dtype=np.float32)
    first[kitchen.POLICY_FEATURES] = rollouts["obs"][0, 0]
    assert np.allclose(rollouts["actions"][0, 0], tracker.label(first), atol=1e-6)


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_dart_episodes_label_noisy_teacher_states_with_clean_actions() -> None:
    from flyarm.benchmarks.kitchen_expert import DemonstrationTracker, noisy_teacher_episodes

    tracker = DemonstrationTracker.from_data(kitchen.load("complete", download=False))
    env = kitchen.recover_env("complete")
    try:
        episodes, stats = noisy_teacher_episodes(tracker, env, 2, 0.1, 300_000)
    finally:
        env.close()
    steps = episodes["mask"].sum(1)
    assert stats["labelled_states"] == int(steps.sum()) and stats["episodes"] == 2
    full = np.zeros(59, dtype=np.float32)
    full[kitchen.POLICY_FEATURES] = episodes["obs"][1, 5]
    assert np.allclose(episodes["actions"][1, 5], tracker.label(full), atol=1e-6)
    # Noise moves the arm off the demonstrations, so the two noisy episodes differ.
    assert not np.allclose(episodes["obs"][0, 50], episodes["obs"][1, 50])


@pytest.mark.skipif(not _dataset_available(), reason="Minari kitchen-complete-v2 not downloaded")
def test_beta_one_rollouts_follow_the_teacher() -> None:
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("MLX Metal device unavailable")
    from flyarm.benchmarks.kitchen_expert import DemonstrationTracker
    from flyarm.flyleg.experiment import _dagger_rollouts
    from flyarm.whole_brain.policy import MLPPolicy

    tracker = DemonstrationTracker.from_data(kitchen.load("complete", download=False))
    policy = MLPPolicy(obs_dim=kitchen.FEATURE_DIM, action_dim=kitchen.ACTION_DIM, hidden=8)
    env = kitchen.recover_env("complete")
    try:
        _, stats = _dagger_rollouts(policy, tracker, env, 1, seed=0, iteration=0, beta=1.0)
    finally:
        env.close()
    # Driven entirely by the teacher, the untrained learner's rollout solves the kitchen.
    assert stats["teacher_step_fraction"] == 1.0 and stats["rollout_mean_tasks"] == 4.0
