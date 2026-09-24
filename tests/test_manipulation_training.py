from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from pydantic import ValidationError

from flyarm.config import (
    MANIPULATION_MAX_LEVEL_REWARD,
    ManipulationImitationConfig,
    ManipulationPPOConfig,
)
from flyarm.manipulation import rollout
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.sim import MAX_LEVEL_REWARD, RewardConfig
from flyarm.manipulation.splits import SPLITS
from flyarm.whole_brain.policy import MLPPolicy
from flyarm.whole_brain.training import Budget, _batches, _optimizer, train_sequence_policy

MODEL = os.environ.get("FLYARM_MODEL")
OBJECTS = Path(os.environ.get("FLYARM_OBJECTS", "assets/objects"))
needs_env = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file() or not OBJECTS.is_dir(),
    reason="set FLYARM_MODEL to the Panda scene.xml and fetch the grasp objects",
)
CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# ----------------------------------------------------------------------------------- configs
def test_the_shipped_configs_validate_and_train_the_measured_connectome_only() -> None:
    imitation = ManipulationImitationConfig.model_validate_json(
        (CONFIGS / "whole-brain-manipulation.json").read_text()
    )
    assert imitation.policies == ["connectome"] and imitation.seeds == [0]
    assert imitation.readout_calibration == "unit_norm" and imitation.action_chunk == 1
    ppo = ManipulationPPOConfig.model_validate_json((CONFIGS / "ppo-manipulation.json").read_text())
    assert ppo.base_kind == "connectome" and ppo.gamma == 0.995 and ppo.bc_weight > 0
    assert ppo.best_on == "validation"


def test_imitation_config_refuses_what_breaks_the_wide_readout_or_the_budget() -> None:
    with pytest.raises(ValidationError, match="unit_norm"):
        ManipulationImitationConfig(readout_calibration="standardize")
    with pytest.raises(ValidationError, match="joint epoch"):
        ManipulationImitationConfig(epochs=2, decoder_warmup_epochs=2)
    with pytest.raises(ValidationError, match="unique"):
        ManipulationImitationConfig(policies=["connectome", "connectome"])
    with pytest.raises(ValidationError):
        ManipulationImitationConfig(action_chunk=2)  # PPO drives one action per step
    controls = ManipulationImitationConfig(policies=["connectome", "shuffled", "gru", "mlp"])
    assert controls.policies == ["connectome", "shuffled", "gru", "mlp"]


def test_ppo_config_recomputes_the_stalling_floor_for_its_gamma() -> None:
    assert MANIPULATION_MAX_LEVEL_REWARD == MAX_LEVEL_REWARD  # the literal tracks the env
    config = ManipulationPPOConfig()
    assert config.gamma == 0.995
    reward = RewardConfig(gamma=config.gamma, subgoal_bonus=config.subgoal_bonus)
    assert reward.subgoal_bonus > 0.0  # the floor is 0 at any gamma: every level term costs
    with pytest.raises(ValidationError, match="shaping"):
        ManipulationPPOConfig(subgoal_bonus=5.0, shaping=10.0)
    with pytest.raises(ValidationError, match="frozen encoder"):
        ManipulationPPOConfig(encoder_lr=1e-4, bc_weight=1.0)
    with pytest.raises(ValidationError):
        ManipulationPPOConfig(seed=10)  # PPO seeds would reach the held-out seed blocks
    with pytest.raises(ValidationError):
        ManipulationPPOConfig(gamma=1.0)


def test_ppo_training_seeds_stay_below_every_held_out_block() -> None:
    from flyarm.rl.ppo import TRAIN_SEED

    held_out = min(split.seed_start for name, split in SPLITS.items() if name != "train")
    last = TRAIN_SEED + 100_000 * 9 + 99_999  # seed 9 and 100,000 episodes
    assert last < held_out


# ----------------------------------------------------------------------------------- seeds
def test_seed_blocks_are_disjoint_and_stable_when_episodes_are_added() -> None:
    blocks = [
        rollout.plan("train", 50, rollout.DEMONSTRATION_OFFSET),
        rollout.plan("train", 50, rollout.VALIDATION_OFFSET),
        *[
            rollout.plan("train", 50, rollout.DAGGER_OFFSET + rollout.DAGGER_ROUND_STRIDE * r)
            for r in range(10)
        ],
    ]
    seeds = [seed for block in blocks for seed in block.seeds]
    assert len(seeds) == len(set(seeds))
    small, large = rollout.plan("iid_test", 2), rollout.plan("iid_test", 5)
    assert set(small.seeds) <= set(large.seeds)  # more episodes never change existing ones
    assert min(large.seeds) >= SPLITS["iid_test"].seed_start
    composition = rollout.plan("unseen_composition", 3)
    assert set(composition.templates) == set(tk.HELD_OUT_TEMPLATES)
    with pytest.raises(ValueError, match="not in split"):
        rollout.plan("train", 1, templates=["full_cleanup"])


# ----------------------------------------------------------------------------------- weights
def test_skill_weights_give_every_skill_the_same_total_and_average_one() -> None:
    mask = np.zeros((2, 10), np.float32)
    mask[0, :10], mask[1, :4] = 1, 1
    skill = np.zeros((2, 10), np.int64)
    skill[0, :8], skill[0, 8:], skill[1, :4] = tk.PLACE, tk.OPEN_DRAWER, tk.CLOSE_DRAWER
    weights = rollout.skill_weights({"mask": mask, "skill": skill}, max_weight=100.0)
    assert weights[mask > 0].mean() == pytest.approx(1.0)
    totals = [weights[(skill == k) & (mask > 0)].sum() for k in (tk.PLACE, tk.OPEN_DRAWER)]
    assert totals[0] == pytest.approx(totals[1])
    assert np.all(weights[mask == 0] == 0)
    capped = rollout.skill_weights({"mask": mask, "skill": skill}, max_weight=1.5)
    assert capped.max() / capped[mask > 0].min() <= 1.5 / (14 / (3 * 8)) + 1e-6


def test_concatenate_pads_shorter_sets_with_masked_steps() -> None:
    def part(episodes: int, steps: int) -> dict[str, np.ndarray]:
        return {
            "obs": np.ones((episodes, steps, 217), np.float32),
            "actions": np.ones((episodes, steps, 5), np.float32),
            "mask": np.ones((episodes, steps), np.float32),
            "skill": np.ones((episodes, steps), np.int64),
        }

    joined = rollout.concatenate([part(2, 3), part(1, 5)])
    assert joined["obs"].shape == (3, 5, 217) and joined["mask"].sum() == 2 * 3 + 5
    assert np.all(joined["actions"][:2, 3:] == 0)


def test_selection_score_is_success_first_then_subgoals() -> None:
    episodes = 14
    one_more_success = rollout.selection_score(1 / episodes, 0.0, episodes)
    every_subgoal = rollout.selection_score(0.0, 1.0, episodes)
    assert one_more_success > every_subgoal > rollout.selection_score(0.0, 0.5, episodes)


# ----------------------------------------------------------------------------------- trainer
def test_the_encoder_learning_rate_is_separate_from_the_decoder_s() -> None:
    policy = MLPPolicy(obs_dim=6, action_dim=2, hidden=8, seed=0)
    budget = Budget(1, 0, 4, 4, 1e-2, float("inf"), input_learning_rate=1e-4)
    optimizer = _optimizer(policy, budget)
    before = {name: np.array(value) for name, value in _flat(policy.parameters()).items()}
    gradients = {"layers": [{"weight": mx.ones(layer.weight.shape)} for layer in policy.layers]}
    for layer, grad in zip(policy.layers, gradients["layers"], strict=True):
        grad["bias"] = mx.ones(layer.bias.shape)
    optimizer.update(policy, gradients)
    mx.eval(policy.parameters())
    after = _flat(policy.parameters())
    step = {name: float(np.abs(np.array(after[name]) - before[name]).max()) for name in before}
    # Adam's first step is the same multiple of the learning rate for every weight.
    assert step["layers.0.weight"] / step["layers.1.weight"] == pytest.approx(1e-2, rel=1e-3)
    assert step["layers.2.weight"] == pytest.approx(step["layers.1.weight"], rel=1e-6)


def test_a_separate_encoder_rate_trains_through_the_decoder_only_warmup() -> None:
    generator = np.random.default_rng(2)
    data = {
        "obs": generator.normal(size=(4, 9, 6)).astype(np.float32),
        "actions": np.tanh(generator.normal(size=(4, 9, 2))).astype(np.float32),
        "mask": np.ones((4, 9), np.float32),
    }
    policy = MLPPolicy(obs_dim=6, action_dim=2, hidden=8, seed=0)
    first = np.array(policy.layers[0].weight)
    budget = Budget(3, 1, 2, 4, 1e-2, float("inf"), input_learning_rate=1e-3)
    curves, _ = train_sequence_policy(policy, data, data["mask"], data, data["mask"], budget, 0)
    assert [row["decoder_only"] for row in curves] == [True, False, False]
    assert not np.allclose(np.array(policy.layers[0].weight), first)  # trained after warmup


def _flat(tree: dict) -> dict[str, mx.array]:
    from mlx.utils import tree_flatten

    return dict(tree_flatten(tree))


def test_length_buckets_cover_every_episode_once_with_similar_lengths_together() -> None:
    lengths = np.array([100, 900, 120, 880, 110, 910, 130, 890])
    generator = np.random.default_rng(0)
    batches = _batches(generator.permutation(8), lengths, 4, generator, True)
    assert sorted(np.concatenate(batches).tolist()) == list(range(8))
    spans = sorted(int(np.ptp(lengths[batch])) for batch in batches)
    assert spans[-1] <= 30  # the short and the long episodes are never mixed


def test_training_on_host_side_batches_matches_a_single_batch_run() -> None:
    """Cutting batches at their longest episode leaves the gradient unchanged."""
    generator = np.random.default_rng(1)
    obs = generator.normal(size=(3, 12, 6)).astype(np.float32)
    actions = np.tanh(generator.normal(size=(3, 12, 2))).astype(np.float32)
    mask = np.zeros((3, 12), np.float32)
    mask[0, :12], mask[1, :5], mask[2, :7] = 1, 1, 1
    data = {"obs": obs, "actions": actions, "mask": mask}
    results = []
    for padding in (0, 20):
        padded = {
            key: np.pad(value, [(0, 0), (0, padding)] + [(0, 0)] * (value.ndim - 2))
            for key, value in data.items()
        }
        policy = MLPPolicy(obs_dim=6, action_dim=2, hidden=8, seed=3)
        budget = Budget(2, 0, 3, 4, 1e-2, float("inf"))
        train_sequence_policy(policy, padded, padded["mask"], padded, padded["mask"], budget, 0)
        results.append(np.array(policy.layers[2].weight))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-6, atol=1e-7)


# ----------------------------------------------------------------------------------- env
@needs_env
def test_smoothness_charges_the_change_of_command_and_defaults_to_nothing() -> None:
    from flyarm.manipulation.env import BatchedManipulation

    first = np.array([[0.4, 0.0, 0.0, 0.0, 1.0]])
    second = np.array([[-0.4, 0.2, 0.0, 0.0, 1.0]])

    def rewards(smoothness: float) -> np.ndarray:
        env = BatchedManipulation(
            Path(MODEL), 1, asset_root=OBJECTS, reward=RewardConfig(smoothness=smoothness)
        )
        env.reset(seeds=np.array([5]), templates=["put_away"])
        return np.array([env.step(first).reward[0], env.step(second).reward[0]])

    plain, smooth = rewards(0.0), rewards(2.0)
    # The first step's change is from the zero command every reset leaves behind.
    change = np.array([np.mean(np.square(first)), np.mean(np.square(second - first))])
    np.testing.assert_allclose(plain - smooth, 2.0 * change, atol=1e-5)


@needs_env
def test_teacher_episodes_are_recorded_with_labels_masks_and_skills() -> None:
    from flyarm.manipulation.env import BatchedManipulation

    episodes = rollout.plan("train", 1, rollout.VALIDATION_OFFSET, templates=["put_away"])
    env = BatchedManipulation(Path(MODEL), len(episodes), asset_root=OBJECTS)
    (log,) = rollout.run_episodes([env], [episodes], None, record=True)
    data = log.data()
    assert log.success.all() and log.subgoals[0] == 3
    assert data["mask"].sum() == log.steps.sum() == data["mask"].shape[1]
    assert data["skill"][0, 0] == tk.OPEN_DRAWER and tk.CLOSE_DRAWER in data["skill"][0]
    summary = rollout.summarize(log)
    assert summary["teacher_step_fraction"] == 1.0 and summary["success_rate"] == 1.0

    class Still:
        def act(self, obs: np.ndarray) -> np.ndarray:
            return np.zeros((len(obs), 5))

    (still,) = rollout.run_episodes([env], [episodes], Still(), record=True)
    assert not still.success.any() and still.steps[0] == env.horizons[0]
    assert rollout.summarize(still)["teacher_step_fraction"] == 0.0
    assert np.abs(still.data()["actions"]).sum() > 0  # the teacher still labels every state
    (mixed,) = rollout.run_episodes(
        [env], [episodes], Still(), beta=1.0, generator=np.random.default_rng(0)
    )
    assert mixed.success.all()  # beta 1 executes the teacher's action at every step


@needs_env
def test_the_ppo_adapter_scores_splits_and_validation_and_reads_episode_horizons() -> None:
    from flyarm.rl.ppo import _horizons
    from flyarm.rl.ppo_manipulation import ManipulationTask

    config = ManipulationPPOConfig(eval_splits=["unseen_composition"], smoothness_weight=0.5)
    task = ManipulationTask(config, Path(MODEL), OBJECTS)
    names = [episodes.split for episodes in task.plans(2)]
    assert names == ["unseen_composition", "train"]
    env = task.make_env(2, 1_000_000)
    env.reset()
    assert env.reward_config.smoothness == 0.5 and env.reward_config.gamma == 0.995
    assert env.observation(privileged=True).shape == (2, task.privileged_dim)
    np.testing.assert_array_equal(_horizons(env, config), env.horizons)
    result = SimpleNamespace(high_water=np.array([2, 5]))
    assert task.extra(result, np.array([True, True])) == 7
    assert task.batch_peak(result, np.array([True, False])) == 2.0
    assert task.batch_peak(result, np.array([False, False])) == 0.0


@needs_env
def test_an_action_array_passed_to_step_is_not_changed_by_a_later_reset() -> None:
    from flyarm.manipulation.env import BatchedManipulation

    env = BatchedManipulation(Path(MODEL), 1, asset_root=OBJECTS)
    env.reset(seeds=np.array([1]))
    action = np.array([[0.1, 0.2, 0.3, 0.0, 1.0]])
    env.step(action)
    env.reset(seeds=np.array([2]))
    assert action[0, 0] == 0.1
    assert env.current_skill().shape == (1,)


def test_config_record_is_json_serialisable() -> None:
    json.dumps(ManipulationPPOConfig().model_dump())
    json.dumps(ManipulationImitationConfig().model_dump())
