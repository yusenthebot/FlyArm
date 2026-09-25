from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from flyarm.config import DaggerStage, ManipulationPPOConfig, SkillDaggerConfig
from flyarm.manipulation import rollout
from flyarm.manipulation import skill_dagger as sd
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.splits import SPLITS
from flyarm.whole_brain.training import StepData, window_indices

MODEL = os.environ.get("FLYARM_MODEL")
OBJECTS = Path(os.environ.get("FLYARM_OBJECTS", "assets/objects"))
needs_env = pytest.mark.skipif(
    not MODEL or not Path(MODEL).is_file() or not OBJECTS.is_dir(),
    reason="set FLYARM_MODEL to the Panda scene.xml and fetch the grasp objects",
)
CONFIGS = Path(__file__).resolve().parent.parent / "configs"
# The controls keep encoder "mlp": their matched budget is the MLP-encoder connectome's.
SHIPPED = {
    "skill-dagger-connectome.json": ("connectome", "mlp"),
    "skill-dagger-shuffled.json": ("shuffled", "mlp"),
    "skill-dagger-mlp.json": ("mlp", "mlp"),
    "skill-dagger-gru.json": ("gru", "mlp"),
}


def test_stages_switch_by_round_and_beta_decays_to_zero() -> None:
    config = SkillDaggerConfig()
    assert config.rounds == 10
    stages = [sd.stage_for_round(config, r)[1].name for r in range(config.rounds)]
    assert stages == ["single_subgoal"] * 4 + ["two_or_three"] * 3 + ["full_templates"] * 3
    with pytest.raises(ValueError, match="past the last stage"):
        sd.stage_for_round(config, config.rounds)
    betas = [sd.beta_for_round(config, r) for r in range(5)]
    assert betas == [1.0, 0.5, 0.25, 0.0, 0.0]  # round 0 is the teacher's


def test_the_config_refuses_what_the_loop_cannot_run() -> None:
    with pytest.raises(ValidationError, match="betas"):
        SkillDaggerConfig(betas=[1.5])
    with pytest.raises(ValidationError, match="warmup"):
        SkillDaggerConfig(first_round_updates=100, warmup_updates=100)
    with pytest.raises(ValidationError, match="40 rounds"):
        SkillDaggerConfig(stages=[DaggerStage(name="x", rounds=40, true_start_share=1.0)] * 2)
    with pytest.raises(ValidationError):
        DaggerStage(name="x", rounds=1, true_start_share=0.0)  # true starts in every stage


def test_the_shipped_configs_validate() -> None:
    for name, (kind, encoder) in SHIPPED.items():
        config = SkillDaggerConfig.model_validate_json((CONFIGS / name).read_text())
        assert config.model.policies == [kind] and config.model.encoder == encoder
        assert config.rounds == 10 and config.model.sampling == "windows"
    ppo = ManipulationPPOConfig.model_validate_json(
        (CONFIGS / "ppo-manipulation-skill-dagger.json").read_text()
    )
    assert ppo.base_run is not None and "skill-dagger" in str(ppo.base_run)
    assert ppo.curriculum and ppo.curriculum[0].max_subgoals > 1  # starts at stage 2


def _steps(lengths: list[int], skills: list[int]) -> dict[str, np.ndarray]:
    total = sum(lengths)
    return {
        "obs": np.arange(total, dtype=np.float32)[:, None].repeat(3, 1),
        "actions": np.arange(total, dtype=np.float32)[:, None].repeat(5, 1),
        "skill": np.repeat(skills, lengths).astype(np.int64),
        "lengths": np.array(lengths, dtype=np.int64),
    }


def test_aggregation_keeps_every_round_s_episodes_end_to_end_and_balances_skills() -> None:
    first = _steps([3, 5], [1, 2])
    second = _steps([2, 6, 4], [1, 3, 3])
    data = sd.aggregate([first, second], max_weight=100.0)
    assert np.array_equal(data.lengths, [3, 5, 2, 6, 4])
    assert np.array_equal(data.starts, [0, 3, 8, 10, 16])
    assert len(data.obs) == 20 and np.array_equal(data.obs[8:10, 0], [0, 1])  # round 2 kept
    skill = np.concatenate([first["skill"], second["skill"]])
    totals = [data.weights[skill == k].sum() for k in (1, 2, 3)]
    assert np.allclose(totals, totals[0]) and np.isclose(data.weights.mean(), 1.0)
    capped = sd.aggregate([first, second], max_weight=1.2)
    spread = data.weights.max() / data.weights.min()
    assert capped.weights.max() / capped.weights.min() < spread and capped.weights.min() > 0
    with pytest.raises(ValueError, match="end to end"):
        StepData(data.obs, data.actions, data.weights, data.starts[::-1].copy(), data.lengths)


def test_windows_never_cross_into_another_episode() -> None:
    data = sd.aggregate([_steps([3, 5, 2], [1, 1, 1])], max_weight=10.0)
    picked = np.arange(10)
    flat, inside = window_indices(data, picked, burn_in=4, width=3)
    episode = np.searchsorted(data.starts, flat, side="right") - 1
    own = np.searchsorted(data.starts, picked, side="right") - 1
    assert np.all(episode == own[:, None])  # every index, padding included, is the episode's
    position = picked - data.starts[own]
    expected = (np.arange(-4, 3)[None] + position[:, None] >= 0) & (
        np.arange(-4, 3)[None] + position[:, None] < data.lengths[own][:, None]
    )
    assert np.array_equal(inside, expected)
    assert np.array_equal(flat[inside], (picked[:, None] + np.arange(-4, 3)[None])[inside])


def test_padding_round_trips_the_ragged_steps() -> None:
    part = _steps([3, 5, 2], [1, 2, 3])
    out = sd.padded(part, 10)
    assert out["obs"].shape == (3, 5, 3) and out["mask"].sum() == 10
    assert np.array_equal(out["obs"][out["mask"] > 0], part["obs"])
    assert np.array_equal(out["skill"][out["mask"] > 0], part["skill"])


@pytest.fixture(scope="module")
def bank():
    if not MODEL:
        pytest.skip("needs FLYARM_MODEL")
    from flyarm.manipulation import curriculum as cu

    episodes = cu.bank_plan(1, cu.BANK_OFFSET)
    env = rollout.make_env(Path(MODEL), episodes, asset_root=OBJECTS, velocities=False)
    return cu.record_bank(env, episodes)


@needs_env
def test_round_plans_mix_true_starts_with_skill_balanced_subgoal_starts(bank) -> None:
    stage = DaggerStage(
        name="two_or_three", rounds=1, min_subgoals=2, max_subgoals=3, true_start_share=0.1
    )
    plans = sd.round_plans(stage, 3, 700, bank, np.random.default_rng(0))
    true, resets = plans
    assert (
        len(true) == 70
        and not true.starts
        and set(true.templates) == set(SPLITS["train"].templates)
    )
    assert min(true.seeds) >= sd.DAGGER_OFFSET + 3 * sd.ROUND_STRIDE
    assert max(true.seeds) < sd.DAGGER_OFFSET + 4 * sd.ROUND_STRIDE and len(set(true.seeds)) == 70
    assert len(resets) == 630 and set(resets.budgets) == {2, 3}
    counts = np.bincount(bank.skill[list(resets.starts)], minlength=len(tk.SKILLS))
    present = counts[np.unique(bank.skill)]
    assert present.min() > 0.6 * present.max()  # skills balanced, not bank-frequency weighted
    tiny = sd.round_plans(stage, 0, 4, bank, np.random.default_rng(0))
    assert len(tiny[0]) == 1  # always at least one true start


class Still:
    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros((len(obs), 5))


@needs_env
def test_labels_are_the_teacher_s_on_the_learner_s_states(bank) -> None:
    from flyarm.manipulation.imitation import Workbench
    from flyarm.manipulation.teacher import ManipulationTeacher

    picks = np.flatnonzero(bank.skill == tk.OPEN_DRAWER)[:2]
    episodes = rollout.EpisodePlan(
        "train",
        tuple(int(s) for s in bank.seeds[picks]),
        tuple(str(t) for t in bank.templates[picks]),
        starts=tuple(int(p) for p in picks),
        budget=1,
        bank=bank,
    )
    bench = Workbench(Path(MODEL), OBJECTS, cue=True, velocities=False)
    (log,) = bench.run([episodes], lambda n: Still(), record=True)
    data = log.data()
    assert rollout.summarize(log)["teacher_step_fraction"] == 0.0  # the learner acted
    # Replay the learner's (zero) actions with a teacher that only watches: same states, same
    # labels, and the labels are not what was executed.
    env = rollout.make_env(Path(MODEL), episodes, asset_root=OBJECTS, velocities=False)
    obs = bank.reset(env, env.rows, np.array(episodes.starts), np.ones(len(episodes), int))
    teacher = ManipulationTeacher(env)
    teacher.reset()
    for t in range(200):
        np.testing.assert_array_equal(data["obs"][:, t], obs.astype(np.float32))
        np.testing.assert_allclose(data["actions"][:, t], teacher.act(), atol=1e-6)
        obs = env.step(np.zeros((len(episodes), 5)), auto_reset=False).obs
    assert np.abs(data["actions"][:, :200]).sum() > 1.0


@needs_env
def test_two_rounds_resume_after_a_crash_and_aggregate(bank, tmp_path) -> None:
    from flyarm.config import ManipulationImitationConfig
    from flyarm.manipulation.imitation import Workbench
    from flyarm.manipulation.sim import OBS_DIM
    from flyarm.whole_brain.policy import MLPPolicy

    config = SkillDaggerConfig(
        model=ManipulationImitationConfig(policies=["mlp"], seeds=[0], window_batch=8, burn_in=4),
        stages=[
            DaggerStage(name="single_subgoal", rounds=1, true_start_share=0.1),
            DaggerStage(
                name="two_or_three", rounds=1, min_subgoals=2, max_subgoals=3, true_start_share=0.1
            ),
        ],
        teacher_episodes=6,
        episodes_per_round=6,
        betas=[0.5],
        first_round_updates=20,
        updates_per_round=10,
        warmup_updates=0,
        skill_eval_episodes=1,
        val_episodes_per_template=1,
        eval_episodes_per_template=1,
        eval_splits=["iid_test"],
    )
    bench = Workbench(Path(MODEL), OBJECTS, config.model.cue, config.model.velocities)
    run = tmp_path / "mlp-0"

    def policy() -> MLPPolicy:
        return MLPPolicy(obs_dim=OBS_DIM, action_dim=5, hidden=32, seed=0)

    first = policy()
    stopped = sd.train_rounds(first, run, config, bench, bank, bank, 0, 1e12, stop_after_round=0)
    assert stopped is None and not (run / "round-01").exists()
    saved = (run / "round-00" / "data.npz").read_bytes()
    teacher = dict(np.load(run / "round-00" / "data.npz"))
    assert (tmp_path / "train.npz").is_file()  # the demonstrations curriculum PPO's DAPG reads
    history = sd.train_rounds(policy(), run, config, bench, bank, bank, 0, 1e12)  # resumed
    assert history is not None and [m["round"] for m in history] == [0, 1]
    assert (run / "round-00" / "data.npz").read_bytes() == saved  # not collected again
    assert [m["stage"] for m in history] == ["single_subgoal", "two_or_three"]
    assert [m["beta"] for m in history] == [1.0, 0.5]
    assert history[0]["rollouts"]["dagger-true-starts"]["teacher_step_fraction"] == 1.0
    learner = history[1]["rollouts"]
    fraction = np.average(
        [r["teacher_step_fraction"] for r in learner.values()],
        weights=[r["episodes"] for r in learner.values()],
    )
    assert 0.0 < fraction < 1.0  # beta 0.5 mixes the learner's and the teacher's actions
    assert history[1]["aggregate_steps"] == int(teacher["lengths"].sum()) + int(
        history[1]["labelled_steps"]
    )
    assert history[1]["aggregate_episodes"] == 12
    assert set(history[1]["skills"]) == set(bank.skill_counts())
    assert np.isfinite(history[1]["train_loss_last"]) and history[1]["updates"] == 10
    item = sd.finish(policy(), run, "mlp", 0, config, bench, bank, history)
    assert item["selected_round"] in (0, 1) and "iid_test" in item["evaluation"]
    assert (
        json.loads((run / "evaluation.json").read_text())["selected_round"]
        == (item["selected_round"])
    )
    again = sd.train_rounds(policy(), run, config, bench, bank, bank, 0, 1e12)
    assert again is not None and [m["round"] for m in again] == [0, 1]  # nothing to redo
