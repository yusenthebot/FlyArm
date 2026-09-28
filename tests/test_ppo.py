from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

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
    assert norm.mean is not None and norm.scale is not None
    assert np.allclose(norm.mean, [2.0, 5.0]) and np.allclose(norm.scale, [1.0, 1e-3])


@pytest.mark.skipif(not MODEL or not Path(MODEL).is_file(), reason="set FLYARM_MODEL")
def test_ppo_changes_only_the_motor_decoder(tmp_path) -> None:
    from flyarm.config import PPOConfig
    from flyarm.rl.batched_pick_place import OBS_DIM
    from flyarm.rl.ppo import PickPlaceTask, train_ppo
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
    assert MODEL is not None
    task = PickPlaceTask(Path(MODEL), config)
    run = train_ppo(policy, task, tmp_path / "ppo", config, [60000, 60001], [50000])
    assert len(run["curves"]) == 2 and run["curves"][1]["policy_trained"]
    # The kept checkpoint is chosen on validation seeds; its reported score is on test seeds.
    assert run["best"]["selected_on"] == "validation seeds"
    assert run["best"]["seeds"] == [60000, 60001] and "selection" in run["evaluations"][-1]
    assert np.array_equal(np.asarray(policy.encoder.weight), encoder)
    assert not np.array_equal(np.asarray(policy.decoder.weight), decoder)
    assert policy.dynamics.matrix is not None  # the connectome itself is untouched and frozen
    assert (tmp_path / "ppo" / "policy-0002.safetensors").is_file()


def _scratch_pair(obs_dim: int):
    import mlx.optimizers as optim

    from flyarm.rl.ppo import MotorHead
    from flyarm.whole_brain.backend_mlx import RateDynamics
    from flyarm.whole_brain.policy import BrainPolicy

    pack = ConnectomePack.from_graph(random_graph(n=48, edges=400))
    policy = BrainPolicy(
        "connectome",
        RateDynamics(pack, interface_for(pack)),
        obs_dim=obs_dim,
        action_dim=4,
        seed=0,
    )
    return policy, MotorHead(policy.decoder, -1.0), optim.Adam(learning_rate=1e-2)


def test_encoder_pass_trains_the_encoder_from_advantages_only() -> None:
    from flyarm.rl.ppo import OBS_DIM, encoder_pass

    steps, envs = 4, 3
    generator = np.random.default_rng(0)
    observations = generator.standard_normal((steps, envs, OBS_DIM)).astype(np.float32)
    actions = generator.uniform(-1, 1, (steps, envs, 4)).astype(np.float32)
    done = np.zeros((steps, envs), np.float32)

    policy, head, optimizer = _scratch_pair(OBS_DIM)
    before = np.asarray(policy.encoder.weight).copy()
    loss = encoder_pass(
        policy,
        head,
        optimizer,
        observations,
        actions,
        generator.standard_normal((steps, envs)).astype(np.float32),
        done,
        policy.initial_state(envs),
        0.5,
    )
    assert np.isfinite(loss)
    assert not np.allclose(np.asarray(policy.encoder.weight), before)

    # With zero advantages every gradient is zero, so the encoder must not move.
    policy, head, optimizer = _scratch_pair(OBS_DIM)
    unchanged = np.asarray(policy.encoder.weight).copy()
    encoder_pass(
        policy,
        head,
        optimizer,
        observations,
        actions,
        np.zeros((steps, envs), np.float32),
        done,
        policy.initial_state(envs),
        0.5,
    )
    assert np.array_equal(np.asarray(policy.encoder.weight), unchanged)


def test_advantage_clip_bounds_the_tail_and_is_off_by_default() -> None:
    from flyarm.rl.ppo import normalized_advantages

    # One rare terminal bonus among many small per-step rewards: a 3-sigma-plus outlier.
    advantages = np.array([0.0] * 15 + [100.0], dtype=np.float64)
    unclipped = normalized_advantages(advantages)
    assert unclipped.max() > 3.0 and abs(unclipped.mean()) < 1e-6
    assert np.array_equal(normalized_advantages(advantages, 0.0), unclipped)

    clipped = normalized_advantages(advantages, 2.0)
    assert clipped.max() == pytest.approx(2.0) and clipped.min() >= -2.0
    # Clipping only touches the tail; every other sample keeps its standardized value.
    assert np.allclose(clipped[:15], unclipped[:15])
    with pytest.raises(ValueError, match="advantage_clip"):
        normalized_advantages(advantages, -1.0)


def test_mlp_control_reads_the_normalized_observation_and_checkpoints(tmp_path) -> None:
    from flyarm.config import PPOConfig
    from flyarm.rl.ppo import MotorHead, rollout_for
    from flyarm.whole_brain.policy import DirectPolicy

    policy = DirectPolicy(obs_dim=5, action_dim=3, hidden=16)
    policy.set_normalization(np.full(5, 1.0), np.full(5, 2.0))
    rollout = rollout_for(policy, 4)
    obs = np.arange(20, dtype=np.float64).reshape(4, 5)
    features = np.asarray(rollout.features(obs))
    assert rollout.feature_dim == 5
    assert np.allclose(features, (obs - 1.0) / 2.0)
    head = MotorHead(policy.decoder, -0.7, 3)
    mean = np.asarray(head.mean(mx.array(features)))
    action, _ = policy.step(mx.array(obs, dtype=mx.float32), policy.initial_state(4))
    assert np.allclose(mean, np.asarray(action), atol=1e-6)
    assert np.abs(mean).max() < 0.1  # the small last layer starts near the zero action
    policy.save(tmp_path / "mlp.safetensors")
    restored = DirectPolicy(obs_dim=5, action_dim=3, hidden=16, seed=7)
    restored.load(tmp_path / "mlp.safetensors")
    state = restored.initial_state(4)
    assert np.allclose(np.asarray(restored.step(mx.array(obs, dtype=mx.float32), state)[0]), mean)
    with pytest.raises(ValueError, match="mlp"):
        PPOConfig(controller="mlp")
    with pytest.raises(ValueError, match="mlp"):
        PPOConfig(controller="mlp", from_scratch=True, encoder_lr=1e-4)
    assert PPOConfig(controller="mlp", from_scratch=True).controller == "mlp"


def test_encoder_pass_trains_a_nonlinear_encoder_too() -> None:
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from flyarm.rl.ppo import OBS_DIM, MotorHead, encoder_pass
    from flyarm.whole_brain.backend_mlx import RateDynamics
    from flyarm.whole_brain.policy import BrainPolicy

    steps, envs = 4, 3
    generator = np.random.default_rng(1)
    pack = ConnectomePack.from_graph(random_graph(n=48, edges=400))
    policy = BrainPolicy(
        "connectome",
        RateDynamics(pack, interface_for(pack)),
        obs_dim=OBS_DIM,
        action_dim=4,
        seed=0,
        encoder="mlp",
        encoder_hidden=(16, 16),
    )
    head, optimizer = MotorHead(policy.decoder, -1.0), optim.Adam(learning_rate=1e-2)
    leaves = cast(list[tuple[str, Any]], tree_flatten(policy.encoder.parameters()))
    before = {k: np.asarray(v).copy() for k, v in leaves}
    loss = encoder_pass(
        policy,
        head,
        optimizer,
        generator.standard_normal((steps, envs, OBS_DIM)).astype(np.float32),
        generator.uniform(-1, 1, (steps, envs, 4)).astype(np.float32),
        generator.standard_normal((steps, envs)).astype(np.float32),
        np.zeros((steps, envs), np.float32),
        policy.initial_state(envs),
        0.5,
    )
    after = dict(tree_flatten(policy.encoder.parameters()))
    assert np.isfinite(loss)
    assert all(not np.allclose(np.asarray(after[k]), v) for k, v in before.items())


def test_encoder_pass_fits_demonstrations_through_one_frozen_step() -> None:
    """The encoder's DAPG term alone (zero advantages) lowers the demonstration error."""
    import mlx.optimizers as optim

    from flyarm.rl.ppo import BrainRollout, DemonstrationSet, MotorHead, encoder_pass
    from flyarm.whole_brain.backend_mlx import RateDynamics
    from flyarm.whole_brain.policy import BrainPolicy

    pack = ConnectomePack.from_graph(random_graph(n=40, edges=300))
    dynamics = RateDynamics(pack, interface_for(pack))
    policy = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2, seed=1, encoder="mlp")
    rng = np.random.default_rng(0)
    # Unit-norm readout, as in the manipulation runs, so the outputs' small changes reach the
    # decoder at unit scale.
    calibration = rng.standard_normal((8, 12, 4)).astype(np.float32)
    policy.calibrate_readout(calibration, np.ones((8, 12), np.float32), unit_norm=True)
    head = MotorHead(policy.decoder, -1.0, 2)
    obs = rng.standard_normal((16, 4)).astype(np.float32)
    rollout = BrainRollout(policy, 16)
    rollout.features(rng.standard_normal((16, 4)).astype(np.float32))  # a non-zero state
    demo = DemonstrationSet(
        features=mx.zeros((1, dynamics.output_count)),
        actions=mx.zeros((1, 2)),
        obs=mx.array(obs),
        states=rollout.state,
        state_actions=mx.zeros((16, 2)),
    )
    # DemonstrationSet's obs/states/state_actions are optional in general, but this set builds
    # them all, so they stay plain arrays from here on.
    demo_obs, demo_states = demo.obs, demo.states
    assert demo_obs is not None and demo_states is not None
    # A reachable target: what another encoder makes the same decoder do from the same states.
    teacher = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2, seed=7, encoder="mlp")
    teacher.decoder = policy.decoder
    teacher.readout_offset, teacher.readout_scale = policy.readout_offset, policy.readout_scale
    current = teacher.encode(teacher.normalize(demo_obs))
    _, pooled = dynamics.advance(demo_states, current, teacher.neural_steps)
    state_actions = head.mean(teacher.motor_features(pooled, demo_obs))
    demo.state_actions = state_actions

    def error() -> float:
        current = policy.encode(policy.normalize(demo_obs))
        _, pooled = dynamics.advance(demo_states, current, policy.neural_steps)
        mean = head.mean(policy.motor_features(pooled, demo_obs))
        return float(((mean - state_actions) ** 2).sum(-1).mean())

    before = error()
    optimizer = optim.Adam(learning_rate=1e-4)
    steps = 40
    encoder_pass(
        policy,
        head,
        optimizer,
        rng.standard_normal((steps, 16, 4)).astype(np.float32),
        np.zeros((steps, 16, 2), np.float32),
        np.zeros((steps, 16), np.float32),
        np.zeros((steps, 16), np.float32),
        policy.initial_state(16),
        1.0,
        demo,
        1.0,
        16,
        rng,
    )
    assert error() < 0.5 * before
    assert pack.fingerprint() == dynamics.pack_fingerprint  # the graph is not a parameter


def test_clipped_encoder_pass_stays_near_the_rollout_policy() -> None:
    """With the clipped surrogate, many updates on one rollout stop moving the action
    log-probability soon after the clip range; the plain advantage-weighted update keeps going."""
    import mlx.optimizers as optim

    from flyarm.rl.ppo import MotorHead, encoder_pass
    from flyarm.whole_brain.backend_mlx import RateDynamics
    from flyarm.whole_brain.policy import BrainPolicy

    pack = ConnectomePack.from_graph(random_graph(n=40, edges=300))
    dynamics = RateDynamics(pack, interface_for(pack))
    rng = np.random.default_rng(0)
    calibration = rng.standard_normal((8, 12, 4)).astype(np.float32)
    obs = rng.standard_normal((1, 16, 4)).astype(np.float32)
    actions = np.full((1, 16, 2), 0.5, np.float32)
    advantages = np.ones((1, 16), np.float32)

    def shift(clip: float) -> float:
        policy = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2, seed=1, encoder="mlp")
        policy.calibrate_readout(calibration, np.ones((8, 12), np.float32), unit_norm=True)
        head = MotorHead(policy.decoder, -1.0, 2)

        def log_prob() -> np.ndarray:
            current = policy.encode(policy.normalize(mx.array(obs[0])))
            _, pooled = dynamics.advance(policy.initial_state(16), current, policy.neural_steps)
            mean = head.mean(policy.motor_features(pooled, mx.array(obs[0])))
            return np.asarray(gaussian_log_prob(mx.array(actions[0]), mean, head.log_std))

        before = log_prob()
        optimizer = optim.SGD(learning_rate=0.05)
        for _ in range(60):
            encoder_pass(
                policy,
                head,
                optimizer,
                obs,
                actions,
                advantages,
                np.zeros((1, 16), np.float32),
                policy.initial_state(16),
                1.0,
                old_log_prob=before[None],
                clip=clip,
            )
        return float(np.mean(log_prob() - before))

    # Plain gradient steps, so that only the objective differs (Adam's momentum keeps moving
    # for a while after the clipped gradient vanishes); the mean shift exceeds log(1.2) because
    # samples still inside the range keep moving the shared encoder.
    clipped, plain = shift(0.2), shift(0.0)
    assert clipped < 0.5 and plain > 2 * clipped
