from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from flyarm.rl.ppo import (  # noqa: E402
    GRURollout,
    MLPRollout,
    MotorHead,
    motor_decoder,
    rollout_for,
)
from flyarm.whole_brain.policy import GRUPolicy, MLPPolicy  # noqa: E402


def _obs(steps: int, n: int, dim: int) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(steps, n, dim)).astype(np.float32)


@pytest.mark.parametrize("kind", ["mlp", "gru"])
def test_the_controls_head_reproduces_the_policy_and_tunes_only_its_last_layer(kind) -> None:
    obs_dim, action_dim, hidden, seed = 7, 3, 16, 1
    policy = (
        MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, seed=seed)
        if kind == "mlp"
        else GRUPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, seed=seed)
    )
    rollout = rollout_for(policy, 4)
    assert isinstance(rollout, MLPRollout if kind == "mlp" else GRURollout)
    head = MotorHead(motor_decoder(policy), -1.0, 3)
    state = policy.initial_state(4)
    for obs in _obs(5, 4, 7):  # the GRU's state is carried, as in the policy's own step
        expected, state = policy.step(mx.array(obs), state)
        got = head.mean(rollout.features(obs))
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-5)
    assert rollout.feature_dim == 16
    # The head's decoder is the policy's own last layer: tuning it changes the saved policy.
    decoder = motor_decoder(policy)
    decoder.weight = decoder.weight + 1.0
    probe = mx.array(_obs(1, 4, 7)[0])
    changed, _ = policy.step(probe, policy.initial_state(4))
    assert np.allclose(
        np.asarray(head.mean(rollout_for(policy, 4).features(np.asarray(probe)))),
        np.asarray(changed),
        atol=1e-5,
    )


def test_the_gru_rollout_zeroes_the_state_of_finished_episodes_only() -> None:
    policy = GRUPolicy(obs_dim=7, action_dim=3, hidden=8, seed=0)
    rollout = GRURollout(policy, 3)
    rollout.features(_obs(1, 3, 7)[0])
    before = np.asarray(rollout.state).copy()
    rollout.reset(np.array([True, False, True]))
    after = np.asarray(rollout.state)
    assert np.all(after[[0, 2]] == 0) and np.array_equal(after[1], before[1])


def test_the_controls_ppo_config_refuses_an_encoder_rate() -> None:
    from pydantic import ValidationError

    from flyarm.config import ManipulationPPOConfig

    assert ManipulationPPOConfig(base_kind="gru").base_kind == "gru"
    with pytest.raises(ValidationError, match="last linear layer"):
        ManipulationPPOConfig(base_kind="mlp", bc_weight=0.0, encoder_lr=1e-4)
