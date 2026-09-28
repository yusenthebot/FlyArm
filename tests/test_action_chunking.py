from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from graph_fixtures import make_interface as interface_for
from graph_fixtures import make_random_graph as random_graph

from flyarm.whole_brain.compiler import ConnectomePack

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal device unavailable (e.g. CI VM)", allow_module_level=True)

from flyarm.whole_brain.backend_mlx import RateDynamics  # noqa: E402
from flyarm.whole_brain.experiment import ResetEveryStep  # noqa: E402
from flyarm.whole_brain.policy import (  # noqa: E402
    ACTPolicy,
    BrainPolicy,
    GRUPolicy,
    MLPPolicy,
    MlxController,
    SequencePolicy,
    gru_hidden_for_budget,
)
from flyarm.whole_brain.training import (  # noqa: E402
    ACTBudget,
    Budget,
    chunk_targets,
    train_act_policy,
    train_sequence_policy,
)


def test_chunk_targets_shift_actions_and_mask_steps_past_the_episode() -> None:
    actions = np.arange(2 * 5 * 1, dtype=np.float32).reshape(2, 5, 1)
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=np.float32)
    targets, weights = chunk_targets(actions, mask, mask, 3)
    assert targets.shape == (2, 5, 3, 1) and weights.shape == (2, 5, 3)
    assert targets[0, 1, :, 0].tolist() == [1, 2, 3]
    assert weights[0, 3].tolist() == [1, 1, 0]  # step 5 does not exist
    assert weights[1, 1].tolist() == [1, 1, 0]  # episode 2 ends after step 2
    assert weights[1, 3].tolist() == [0, 0, 0]  # padding is never a target step
    single, single_weights = chunk_targets(actions, mask, mask, 1)
    assert np.array_equal(single[..., 0, :], actions)
    assert np.array_equal(single_weights[..., 0], mask)


class _Scripted:
    """Stand-in policy whose chunk at call n is n + row index, for exact ensemble checks."""

    chunk, action_dim = 3, 1

    def __init__(self) -> None:
        self.calls = 0

    def initial_state(self, batch: int) -> Any:
        return mx.zeros((batch, 1))

    def step(self, obs: Any, state: Any) -> tuple[Any, Any]:
        self.calls += 1
        plan = mx.array([[10.0 * self.calls + row for row in range(self.chunk)]])
        return plan, state + 1


def test_temporal_ensemble_averages_every_prediction_for_the_current_step() -> None:
    controller = MlxController(_Scripted(), ensemble_decay=0.0)  # type: ignore[arg-type]
    obs = np.zeros(1, dtype=np.float32)
    assert controller.act(obs)[0] == pytest.approx(10.0)
    # Step 2: row 1 of the first chunk (11) and row 0 of the second (20).
    assert controller.act(obs)[0] == pytest.approx((11 + 20) / 2)
    assert controller.act(obs)[0] == pytest.approx((12 + 21 + 30) / 3)
    assert controller.act(obs)[0] == pytest.approx((22 + 31 + 40) / 3)  # first chunk expired
    decayed = MlxController(_Scripted(), ensemble_decay=1.0)  # type: ignore[arg-type]
    decayed.act(obs)
    weights = np.exp(-np.arange(2.0))
    assert decayed.act(obs)[0] == pytest.approx(float(weights @ [11, 20] / weights.sum()))


def test_state_reset_lesion_keeps_the_output_ensemble() -> None:
    controller = MlxController(_Scripted(), ensemble_decay=0.0)  # type: ignore[arg-type]
    lesioned = ResetEveryStep(controller)
    obs = np.zeros(1, dtype=np.float32)
    lesioned.act(obs)
    assert lesioned.act(obs)[0] == pytest.approx((11 + 20) / 2)
    assert float(np.asarray(controller.state)[0, 0]) == 1.0  # recurrent state was cleared
    controller.reset()
    assert controller.plan is None


def test_every_policy_emits_a_flattened_chunk() -> None:
    pack = ConnectomePack.from_graph(random_graph(n=40, edges=300))
    dynamics = RateDynamics(pack, interface_for(pack))
    brain = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2, chunk=5)
    assert brain.decoder.weight.shape == (10, dynamics.output_count)
    assert brain.with_dynamics("edges_off", dynamics).chunk == 5
    policies: list[SequencePolicy] = [
        brain,
        GRUPolicy(obs_dim=4, action_dim=2, hidden=8, chunk=5),
        MLPPolicy(obs_dim=4, action_dim=2, hidden=8, chunk=5),
        ACTPolicy(obs_dim=4, action_dim=2, chunk=5, token_slices=[(0, 2), (2, 4)], dims=16),
    ]
    obs = mx.zeros((3, 4))
    for policy in policies:
        output, _ = policy.step(obs, policy.initial_state(3))
        assert output.shape == (3, 10)
        controller = MlxController(policy)
        assert controller.act(np.zeros(4)).shape == (2,)
        assert controller.plan is not None and controller.plan.shape == (5, 2)


def test_gru_budget_counts_the_chunked_readout() -> None:
    hidden = gru_hidden_for_budget(30, 90, 108_000)
    gru = GRUPolicy(obs_dim=30, action_dim=9, hidden=hidden, chunk=10)
    assert abs(gru.trainable_parameter_count() - 108_000) / 108_000 < 0.01


def _smooth_demos(episodes: int = 6, steps: int = 24) -> dict[str, np.ndarray]:
    phase = np.linspace(0, 2 * np.pi, steps, dtype=np.float32)
    offsets = np.random.default_rng(0).uniform(0, np.pi, episodes).astype(np.float32)
    signal = np.sin(phase[None] + offsets[:, None])
    obs = np.stack((signal, np.cos(phase[None] + offsets[:, None])), -1).astype(np.float32)
    actions = (0.8 * np.stack((signal, -signal), -1)).astype(np.float32)
    return {"obs": obs, "actions": actions, "mask": np.ones((episodes, steps), np.float32)}


@pytest.mark.parametrize("loss", ["mse", "l1"])
def test_chunked_sequence_training_reduces_the_chunk_loss(loss: str) -> None:
    data = _smooth_demos()
    policy = MLPPolicy(obs_dim=2, action_dim=2, hidden=32, chunk=4, seed=1)
    budget = Budget(
        epochs=30,
        decoder_warmup_epochs=0,
        batch_size=3,
        bptt_steps=6,
        learning_rate=0.01,
        deadline=time.monotonic() + 120,
        loss=loss,  # type: ignore[arg-type]
    )
    mask = data["mask"]
    curves, summary = train_sequence_policy(policy, data, mask, data, mask, budget, 0)
    assert summary["loss"] == loss
    assert summary["best_validation_loss"] < 0.5 * curves[0]["validation_loss"]


def test_act_training_reduces_validation_l1_and_round_trips(tmp_path) -> None:
    data = _smooth_demos()
    policy = ACTPolicy(obs_dim=2, action_dim=2, chunk=4, token_slices=[(0, 1), (1, 2)], dims=32)
    budget = ACTBudget(
        steps=300,
        batch_size=32,
        learning_rate=1e-3,
        kl_weight=10.0,
        eval_every=100,
        deadline=time.monotonic() + 300,
    )
    curves, summary = train_act_policy(policy, data, data, budget, seed=0)
    assert summary["best_validation_loss"] < 0.5 * curves[0]["validation_loss"]
    assert not policy.training
    obs = mx.array(data["obs"][:, 3])
    first, _ = policy.step(obs, policy.initial_state(len(obs)))
    again, _ = policy.step(obs, policy.initial_state(len(obs)))
    assert np.array_equal(np.asarray(first), np.asarray(again))  # z = 0 and no dropout
    policy.save(tmp_path / "act.safetensors")
    restored = ACTPolicy(obs_dim=2, action_dim=2, chunk=4, token_slices=[(0, 1), (1, 2)], dims=32)
    restored.load(tmp_path / "act.safetensors")
    output, _ = restored.step(obs, restored.initial_state(len(obs)))
    assert np.allclose(np.asarray(output), np.asarray(first), atol=1e-6)


def test_closed_loop_selector_keeps_the_best_scored_checkpoint() -> None:
    data = _smooth_demos()
    policy = MLPPolicy(obs_dim=2, action_dim=2, hidden=16, chunk=2, seed=3)
    snapshots: list[np.ndarray] = []
    scores = [0.0, 3.0, 1.0, 3.0, 2.0, 0.5]  # epochs 2 and 4 tie; validation loss decides

    def selector(candidate: SequencePolicy) -> float:
        assert isinstance(candidate, MLPPolicy)  # this test only ever hands it the MLP above
        snapshots.append(np.asarray(candidate.layers[0].weight))
        return scores[len(snapshots) - 1]

    budget = Budget(
        epochs=6,
        decoder_warmup_epochs=0,
        batch_size=3,
        bptt_steps=6,
        learning_rate=0.01,
        deadline=time.monotonic() + 120,
    )
    mask = data["mask"]
    curves, summary = train_sequence_policy(
        policy, data, mask, data, mask, budget, 0, selector=selector, select_every=1
    )
    tied = [1, 3]
    winner = min(tied, key=lambda index: curves[index]["validation_loss"])
    assert summary["selection"] == "closed_loop_validation" and summary["selection_score"] == 3.0
    assert summary["best_epoch"] == winner + 1
    assert np.array_equal(np.asarray(policy.layers[0].weight), snapshots[winner])
    sparse: list[float] = []

    def sparse_selector(_: SequencePolicy) -> float:
        sparse.append(1.0)
        return 1.0

    train_sequence_policy(
        MLPPolicy(obs_dim=2, action_dim=2, hidden=16, chunk=2),
        data,
        mask,
        data,
        mask,
        budget,
        0,
        selector=sparse_selector,
        select_every=4,
    )
    assert len(sparse) == 2  # epoch 4 and the final epoch
