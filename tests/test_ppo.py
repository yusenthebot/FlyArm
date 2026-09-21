from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from graph_fixtures import make_interface as interface_for
from graph_fixtures import make_random_graph as random_graph

from flyarm.whole_brain.compiler import ConnectomePack

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal device unavailable (e.g. CI VM)", allow_module_level=True)

from flyarm.rl.ppo import RunningNorm, gae, gaussian_log_prob  # noqa: E402

MODEL = os.environ.get("FLYARM_MODEL")


def test_gae_matches_a_hand_computation_and_stops_at_episode_ends() -> None:
    rewards = np.array([[1.0], [0.0], [2.0]], np.float32)
    values = np.array([[0.5], [0.2], [0.1]], np.float32)
    dones = np.array([[0.0], [1.0], [0.0]], np.float32)
    advantages, returns = gae(rewards, values, dones, np.array([0.4], np.float32), 0.9, 0.5)
    last = 2.0 + 0.9 * 0.4 - 0.1
    middle = 0.0 - 0.2  # the episode ends after step 1: no bootstrap, no carry
    first = (1.0 + 0.9 * 0.2 - 0.5) + 0.9 * 0.5 * middle
    assert np.allclose(advantages[:, 0], [first, middle, last], atol=1e-6)
    assert np.allclose(returns, advantages + values)


def test_gaussian_log_prob_matches_the_closed_form() -> None:
    actions = mx.array([[0.1, -0.2]])
    mean = mx.array([[0.0, 0.0]])
    log_std = mx.array([np.log(0.5), np.log(2.0)])
    expected = sum(
        -0.5 * (a / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
        for a, s in ((0.1, 0.5), (-0.2, 2.0))
    )
    assert float(gaussian_log_prob(actions, mean, log_std)[0]) == pytest.approx(expected, rel=1e-5)


def test_running_norm_tracks_scale_with_a_floor() -> None:
    norm = RunningNorm()
    with pytest.raises(RuntimeError):
        norm(np.zeros((1, 2)))
    norm.update(np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert np.allclose(norm.mean, [2.0, 5.0]) and np.allclose(norm.scale, [1.0, 1e-3])


@pytest.mark.skipif(not MODEL or not Path(MODEL).is_file(), reason="set FLYARM_MODEL")
def test_ppo_changes_only_the_motor_decoder(tmp_path) -> None:
    from flyarm.config import PPOConfig
    from flyarm.rl.batched_pick_place import OBS_DIM
    from flyarm.rl.ppo import train_ppo
    from flyarm.whole_brain.backend_mlx import RateDynamics
    from flyarm.whole_brain.policy import BrainPolicy

    pack = ConnectomePack.from_graph(random_graph(n=40, edges=300))
    policy = BrainPolicy(
        "connectome", RateDynamics(pack, interface_for(pack)), obs_dim=OBS_DIM, action_dim=4, seed=2
    )
    encoder = np.asarray(policy.encoder.weight).copy()
    decoder = np.asarray(policy.decoder.weight).copy()
    config = PPOConfig(
        num_envs=4,
        rollout_steps=8,
        iterations=2,
        critic_warmup=1,
        minibatch=32,
        eval_every=2,
        eval_episodes=2,
        horizon=20,
    )
    run = train_ppo(policy, Path(MODEL), tmp_path / "ppo", config, [60000, 60001], [50000])
    assert len(run["curves"]) == 2 and run["curves"][1]["policy_trained"]
    # The kept checkpoint is chosen on validation seeds; its reported score is on test seeds.
    assert run["best"]["selected_on"] == "validation seeds"
    assert run["best"]["seeds"] == [60000, 60001] and "selection" in run["evaluations"][-1]
    assert np.array_equal(np.asarray(policy.encoder.weight), encoder)
    assert not np.array_equal(np.asarray(policy.decoder.weight), decoder)
    assert policy.dynamics.matrix is not None  # the connectome itself is untouched and frozen
    assert (tmp_path / "ppo" / "policy-0002.safetensors").is_file()
