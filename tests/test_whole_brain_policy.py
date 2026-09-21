from __future__ import annotations

import time

import numpy as np
import pytest
from graph_fixtures import make_interface as interface_for
from graph_fixtures import make_random_graph as random_graph

from flyarm.interfaces import NeuralInterface
from flyarm.whole_brain.compiler import ConnectomePack

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal device unavailable (e.g. CI VM)", allow_module_level=True)

from flyarm.whole_brain.backend_mlx import RateDynamics  # noqa: E402
from flyarm.whole_brain.experiment import evidence  # noqa: E402
from flyarm.whole_brain.policy import (  # noqa: E402
    BrainPolicy,
    GRUPolicy,
    MlxController,
    gru_hidden_for_budget,
)
from flyarm.whole_brain.shuffle import shuffle_pack  # noqa: E402
from flyarm.whole_brain.training import Budget, train_sequence_policy  # noqa: E402


@pytest.fixture
def pack() -> ConnectomePack:
    return ConnectomePack.from_graph(random_graph(n=40, edges=300))


def test_shuffle_preserves_degrees_signs_and_contacts_but_moves_edges(pack) -> None:
    shuffled = shuffle_pack(pack, seed=5)
    assert np.array_equal(shuffled.degrees()[0], pack.degrees()[0])
    assert np.array_equal(shuffled.degrees()[1], pack.degrees()[1])
    assert np.array_equal(shuffled.signs, pack.signs)
    for source in range(pack.nodes):
        before = np.sort(pack.contacts[pack.col_idx == source])
        after = np.sort(shuffled.contacts[shuffled.col_idx == source])
        assert np.array_equal(before, after)
    assert shuffled.fingerprint() != pack.fingerprint()
    assert shuffled.manifest["edge_overlap_fraction"] < 0.5
    assert shuffle_pack(pack, seed=5).fingerprint() == shuffled.fingerprint()


def test_only_adapters_are_trainable_and_training_reduces_loss(pack) -> None:
    dynamics = RateDynamics(pack, interface_for(pack))
    policy = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2, seed=1)
    expected = (4 + 1) * dynamics.input_count + (dynamics.output_count + 1) * 2
    assert policy.trainable_parameter_count() == expected
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((6, 12, 4)).astype(np.float32)
    # A target that needs the observation, so the adapters must use the graph path.
    actions = np.tanh(obs[..., :2] * 0.8).astype(np.float32)
    data = {"obs": obs, "actions": actions, "mask": np.ones((6, 12), dtype=np.float32)}
    weights = data["mask"]
    before = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2, seed=1)
    curves, summary = train_sequence_policy(
        policy,
        data,
        weights,
        data,
        weights,
        Budget(
            epochs=12,
            decoder_warmup_epochs=2,
            batch_size=3,
            bptt_steps=4,
            learning_rate=0.02,
            deadline=time.monotonic() + 120,
        ),
        seed=0,
    )
    assert curves[0]["decoder_only"] and not curves[-1]["decoder_only"]
    assert summary["best_validation_loss"] < curves[0]["validation_loss"]
    assert not np.array_equal(np.asarray(policy.encoder.weight), np.asarray(before.encoder.weight))
    assert dynamics.matrix is not None


def test_edges_off_clone_keeps_adapters_and_gives_constant_action(pack) -> None:
    interface = interface_for(pack)
    policy = BrainPolicy("connectome", RateDynamics(pack, interface), obs_dim=4, action_dim=2)
    clone = policy.with_dynamics("edges_off", RateDynamics(pack, interface, edges=False))
    assert np.array_equal(np.asarray(clone.decoder.weight), np.asarray(policy.decoder.weight))
    controller = MlxController(clone)
    actions = [controller.act(np.random.default_rng(i).standard_normal(4)) for i in range(5)]
    assert all(np.array_equal(actions[0], action) for action in actions)
    other = ConnectomePack.from_graph(random_graph(n=40, edges=300, seed=9))
    with pytest.raises(ValueError, match="different graph"):
        policy.with_dynamics("x", RateDynamics(other, interface))


def test_gru_budget_matches_brain_adapters(pack) -> None:
    hidden = gru_hidden_for_budget(37, 4, 78_240)
    gru = GRUPolicy(obs_dim=37, action_dim=4, hidden=hidden)
    assert abs(gru.trainable_parameter_count() - 78_240) / 78_240 < 0.01


def test_evidence_rules_do_not_promote_weak_results() -> None:
    def model(kind: str, clean: float, **ablations: float) -> dict:
        item = {"kind": kind, "seed": 0, "clean": {"success_rate": clean}}
        item.update({key: {"success_rate": value} for key, value in ablations.items()})
        return item

    weak = evidence([model("connectome", 0.5, edges_off=0.0, direct_only=0.0)])
    assert weak["readings"] == ["graph-mediated control not established"]
    strong = evidence(
        [
            model("connectome", 0.9, edges_off=0.0, direct_only=0.9),
            model("shuffled", 0.85),
            model("gru", 1.0),
        ]
    )
    assert "graph-mediated control supported (MaleCNS > 70%, edges-off < 10%)" in strong["readings"]
    assert "direct ascending->output synapses alone reproduce the policy" in strong["readings"]
    assert "measured and shuffled graphs equivalent: graph is a usable medium" in strong["readings"]
    assert all("topology advantage" not in reading for reading in strong["readings"])


def test_interface_binding_is_graph_specific(pack) -> None:
    interface = interface_for(pack)
    shuffled = shuffle_pack(pack, seed=1)
    with pytest.raises(ValueError, match="different graph"):
        interface.resolve_indices(shuffled)
    rebound = NeuralInterface.bind(shuffled, interface.input_body_ids, interface.output_body_ids)
    assert np.array_equal(rebound.resolve_indices(shuffled)[0], interface.resolve_indices(pack)[0])


def test_direct_only_weights_keep_exactly_the_input_to_output_edges(pack) -> None:
    from flyarm.whole_brain.diagnostics import direct_only_weights

    interface = interface_for(pack)
    inputs, outputs = interface.resolve_indices(pack)
    kept = direct_only_weights(pack, interface) != 0
    rows = pack.rows()
    expected = np.isin(pack.col_idx, inputs) & np.isin(rows, outputs)
    expected &= pack.normalized_weights() != 0
    assert np.array_equal(kept, expected)


def test_checkpoint_round_trip_restores_adapters_and_normalization(pack, tmp_path) -> None:
    dynamics = RateDynamics(pack, interface_for(pack))
    policy = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2, seed=3)
    policy.set_normalization(np.arange(4, dtype=np.float32), np.full(4, 2.0, dtype=np.float32))
    policy.save(tmp_path / "policy.safetensors")
    restored = BrainPolicy("connectome", dynamics, obs_dim=4, action_dim=2, seed=99)
    restored.load(tmp_path / "policy.safetensors")
    obs = np.random.default_rng(1).standard_normal(4)
    assert np.array_equal(MlxController(policy).act(obs), MlxController(restored).act(obs))
    assert restored.trainable_parameter_count() == policy.trainable_parameter_count()
    with pytest.raises(FileExistsError):
        policy.save(tmp_path / "policy.safetensors")


def test_ablation_clone_reuses_the_policy_seed(pack) -> None:
    interface = interface_for(pack)
    policy = BrainPolicy(
        "connectome", RateDynamics(pack, interface), obs_dim=4, action_dim=2, seed=123
    )
    policy.with_dynamics("edges_off", RateDynamics(pack, interface, edges=False))
    after_clone = np.asarray(mx.random.uniform(shape=(3,)))
    BrainPolicy("connectome", RateDynamics(pack, interface), obs_dim=4, action_dim=2, seed=123)
    after_fresh = np.asarray(mx.random.uniform(shape=(3,)))
    # The clone's hidden re-initialization is tied to the run seed, never a constant.
    assert np.array_equal(after_clone, after_fresh)


def test_config_rejects_dagger_for_reach() -> None:
    from pydantic import ValidationError

    from flyarm.config import WholeBrainConfig

    with pytest.raises(ValidationError, match="DAgger"):
        WholeBrainConfig(task="reach", dagger_iterations=1)
    with pytest.raises(ValidationError, match="warmup"):
        WholeBrainConfig(task="reach", epochs=2, decoder_warmup_epochs=2)


def test_shuffle_replicates_keep_the_original_shuffle_and_names() -> None:
    from pydantic import ValidationError

    from flyarm.config import WholeBrainConfig
    from flyarm.whole_brain.experiment import run_name, shuffle_seed

    assert shuffle_seed(2, 0) == 2 + 17000  # the shuffle of every run before replicates
    assert len({shuffle_seed(seed, r) for seed in range(6) for r in range(3)}) == 18
    assert run_name("shuffled", 3) == "shuffled-3"
    assert run_name("shuffled", 3, 1) == "shuffled-3-r1"
    with pytest.raises(ValidationError, match="unique"):
        WholeBrainConfig(task="pick-place", horizon=400, shuffle_replicates=[1, 1])
    with pytest.raises(ValidationError, match="seeds < 1000"):
        WholeBrainConfig(task="pick-place", horizon=400, seeds=[1000], shuffle_replicates=[0, 1])


def test_channel_encoders_write_only_their_own_input_block(pack) -> None:
    ids = pack.body_ids
    interface = NeuralInterface.bind(pack, ids[:6], ids[-5:])
    dynamics = RateDynamics(pack, interface)
    policy = BrainPolicy(
        "flyleg", dynamics, obs_dim=5, action_dim=2, channels=[(0, 2, 2), (2, 5, 4)]
    )
    obs = mx.array(np.array([[0.0, 0.0, 1.0, -2.0, 3.0]], dtype=np.float32))
    policy.encoders[0].bias = mx.zeros_like(policy.encoders[0].bias)
    current = np.asarray(policy.encode(obs))
    assert current.shape == (1, 6) and np.all(current[0, :2] == 0)
    deprived = policy.silence_channel("head_deprived", 1)
    assert np.all(np.asarray(deprived.encode(obs))[0, 2:] == 0)
    assert np.array_equal(np.asarray(deprived.decoder.weight), np.asarray(policy.decoder.weight))
    with pytest.raises(ValueError, match="cover every"):
        BrainPolicy("x", dynamics, obs_dim=5, action_dim=2, channels=[(0, 5, 3)])
