"""PPO against the batched manipulation benchmark, reusing the task-agnostic trainer.

ManipulationTask is the TaskAdapter of flyarm.rl.ppo.train_ppo for
flyarm.manipulation.env.BatchedManipulation, so the loop, the critic, the clipped updates, the
DAPG term and the checkpointing are exactly those that solved the kitchen (research log E46 to
E53). What is specific here:

- the reward is the environment's own RewardConfig: the subgoal bonus paid once and in task
  order, potential-based shaping toward the current subgoal, and the motion-quality penalties;
- episodes have their own horizons (200 + 300 per subgoal), and the critic's time feature is
  the step over that episode's horizon;
- the critic reads the 233-feature privileged observation;
- evaluation runs every split of ``eval_splits`` on test episodes and the train split's
  validation episodes, all in lockstep; the checkpoint is selected on validation only, by
  success rate with the mean fraction of subgoals done breaking ties;
- the DAPG demonstrations are the imitation run's own teacher demonstrations (train.npz),
  replayed through the encoder and connectome from a zero state, as in imitation: once when
  the encoder is frozen, and every ``bc_refresh_every`` iterations when PPO trains it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np

from flyarm.config import ManipulationImitationConfig, ManipulationPPOConfig
from flyarm.manipulation import curriculum as cu
from flyarm.manipulation import rollout
from flyarm.manipulation.env import DEFAULT_ASSET_ROOT, BatchedManipulation
from flyarm.manipulation.imitation import Workbench, load_data, load_manipulation_policy
from flyarm.manipulation.sim import OBS_DIM, PRIVILEGED_DIM, RewardConfig
from flyarm.rl.ppo import (
    BrainRollout,
    Controller,
    DemonstrationSet,
    MotorHead,
    motor_decoder,
    rollout_for,
    train_ppo,
    trainable_count,
)
from flyarm.whole_brain.policy import BrainPolicy, GRUPolicy, MLPPolicy

ACTION_DIM = rollout.ACTION_DIM
# What PPO trains for each kind: one linear map into the action's tanh, everything before it
# frozen, so the controls get the stage the connectome gets with the comparable trainable part
# (docs/MANIPULATION_ENV.md, "PPO for the controls").
CLAIMS = {
    "connectome": "PPO tunes the imitation checkpoint's motor decoder; encoder and connectome "
    "frozen",
    "shuffled": "PPO tunes the imitation checkpoint's motor decoder; encoder and shuffled graph "
    "frozen",
    "mlp": "PPO tunes the matched MLP's last linear layer; its hidden layers frozen",
    "gru": "PPO tunes the matched GRU's linear readout; its recurrent cell frozen",
}
VALIDATION = "validation"  # the score entry the checkpoint is selected on


def reward_config(settings: ManipulationPPOConfig) -> RewardConfig:
    return RewardConfig(
        subgoal_bonus=settings.subgoal_bonus,
        shaping=settings.shaping,
        stray_contact=settings.stray_contact_weight,
        disturbance=settings.disturbance_weight,
        action_cost=settings.action_weight,
        smoothness=settings.smoothness_weight,
        gamma=settings.gamma,
    )


class HeadActor:
    """Deterministic mean actions of the PPO head over the policy's frozen part."""

    def __init__(self, policy: Controller, head: MotorHead, num_envs: int) -> None:
        self.brain = rollout_for(policy, num_envs)
        self.head = head

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.head.mean(self.brain.features(obs)), dtype=np.float64)


class ManipulationTask:
    """The manipulation adapter for flyarm.rl.ppo.train_ppo.

    ``score(policy, head, seeds)`` reads ``len(seeds)`` as the number of test episodes per
    template of every evaluated split (seed layout in flyarm.manipulation.rollout); the
    validation entry always has ``val_episodes_per_template`` per template.
    """

    obs_dim = OBS_DIM
    privileged_dim = PRIVILEGED_DIM
    action_dim = ACTION_DIM
    extra_key = "subgoals_per_episode"
    extra_label = "subgoals"
    batch_peak_key: str | None = "max_subgoals_in_one_episode"
    selection_key = "selection_score"
    validation_variant = VALIDATION

    def __init__(
        self,
        settings: ManipulationPPOConfig,
        model_path: Path,
        asset_root: Path = DEFAULT_ASSET_ROOT,
        *,
        cue: bool = True,
        velocities: bool = True,
        phase_cue: bool = False,
        bank: cu.SubgoalBank | None = None,
        validation_bank: cu.SubgoalBank | None = None,
    ) -> None:
        self.settings = settings
        self.model_path, self.asset_root, self.cue = Path(model_path), Path(asset_root), cue
        self.velocities = velocities
        self.phase_cue = phase_cue
        self.bench = Workbench(model_path, asset_root, cue, velocities, phase_cue)
        if settings.curriculum and bank is None:
            raise ValueError("a curriculum needs a subgoal bank")
        self.bank, self.validation_bank = bank, validation_bank
        self.stages = [
            cu.Stage(
                stage.name,
                stage.iterations,
                stage.min_subgoals,
                stage.max_subgoals,
                stage.true_start_share,
            )
            for stage in settings.curriculum
        ]
        self.env: BatchedManipulation | None = None

    def make_env(self, num_envs: int, first_seed: int) -> BatchedManipulation:
        options: dict[str, Any] = {
            "split": "train",
            "asset_root": self.asset_root,
            "first_seed": first_seed,
            "reward": reward_config(self.settings),
            "cue": self.cue,
            "velocities": self.velocities,
            "phase_cue": self.phase_cue,
        }
        if self.stages:
            assert self.bank is not None
            self.env = cu.CurriculumManipulation(
                self.model_path,
                num_envs,
                bank=self.bank,
                stages=self.stages,
                curriculum_seed=self.settings.seed,
                group_weights_by_name=self.settings.reset_group_weights,
                **options,
            )
        else:
            self.env = BatchedManipulation(self.model_path, num_envs, **options)
        return self.env

    def begin_iteration(self, iteration: int) -> dict[str, Any]:
        """Switch the curriculum stage by iteration budget; the stage joins the curve row."""
        if not self.stages or not isinstance(self.env, cu.CurriculumManipulation):
            return {}
        bounds = np.cumsum([stage.iterations for stage in self.stages])
        index = int(np.searchsorted(bounds, iteration, side="right"))
        index = min(index, len(self.stages) - 1)
        if index != self.env.stage_index:
            self.env.set_stage(index)
            print(f"curriculum stage {self.stages[index].name} from iteration {iteration + 1}")
        return {
            "stage": self.stages[index].name,
            "subgoal_starts": self.env.subgoal_starts,
            "true_starts": self.env.true_starts,
        }

    def plans(self, per_template: int) -> list[rollout.EpisodePlan]:
        tests = [
            rollout.plan(split, per_template, rollout.TEST_OFFSET)
            for split in self.settings.eval_splits
        ]
        validation = rollout.plan(
            "train", self.settings.val_episodes_per_template, rollout.VALIDATION_OFFSET
        )
        skills = (
            cu.skill_plans(self.validation_bank, self.settings.skill_eval_episodes)
            if self.validation_bank is not None
            else []
        )
        return [*tests, validation, *skills]

    def score(
        self, policy: Controller, head: MotorHead, seeds: list[int]
    ) -> dict[str, dict[str, Any]]:
        """Per split from true starts (the headline), validation, and per skill from resets."""
        logs = self.bench.run(self.plans(len(seeds)), lambda n: HeadActor(policy, head, n))
        scored = {}
        for log in logs:
            name = log.plan.label or (VALIDATION if log.plan.split == "train" else log.plan.split)
            scored[name] = rollout.summarize(log)
        return scored

    def extra(self, result: Any, done: np.ndarray) -> int:
        """Subgoals the finished episodes completed themselves (preset ones excluded)."""
        return int((result.high_water - result.preset)[done].sum())

    def batch_peak(self, result: Any, done: np.ndarray) -> float:
        """Most subgoals any single finished episode completed in order itself."""
        earned = result.high_water - result.preset
        return float(earned[done].max()) if done.any() else 0.0

    def describe(self, name: str, scored: dict[str, Any], episodes: int) -> str:
        return (
            f"{name} success {scored['successes']}/{scored['episodes']} "
            f"subgoals {scored['subgoal_fraction']:.2f} "
            f"stray {scored['stray_contact_fraction']:.2f} "
            f"|a| {scored['mean_abs_action']:.2f} sat {scored['saturated_fraction']:.2f}"
        )


def base_config(run_root: Path) -> ManipulationImitationConfig:
    """The imitation run's config, checked for what a PPO warm start depends on."""
    path = run_root / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run_root} is not a manipulation imitation run: no config.json")
    config = ManipulationImitationConfig.model_validate_json(path.read_text())
    if config.action_chunk != 1:
        raise ValueError(f"{run_root} emits action chunks; the PPO head drives one action a step")
    return config


def demonstration_features(
    policy: Controller,
    base_run: Path,
    max_steps: int,
    seed: int,
    batch: int = 32,
    encoder_samples: int = 0,
) -> DemonstrationSet:
    """Connectome features and teacher actions of the imitation run's demonstrations.

    At most ``max_steps`` valid steps are kept, drawn without replacement with the run's
    skill-balanced weights so every skill stays represented; each episode is replayed through
    the encoder and connectome from a zero state, as in imitation, and only the drawn steps'
    features are kept. With ``encoder_samples`` (a trained encoder), that many of the drawn
    steps also keep their observation and the connectome state before them, and the set can
    ``refresh`` itself: the same steps replayed through the encoder as it is then.
    """
    config = base_config(base_run)
    data = load_data(base_run / "train.npz")
    weights = rollout.skill_weights(data, config.max_skill_weight)
    valid = np.argwhere(data["mask"] > 0)
    count = min(max_steps, len(valid))
    probabilities = weights[valid[:, 0], valid[:, 1]].astype(np.float64)
    generator = np.random.default_rng([seed, 31])
    picked = generator.choice(
        len(valid), size=count, replace=False, p=probabilities / probabilities.sum()
    )
    keep = np.zeros(data["mask"].shape, dtype=bool)
    keep[valid[picked, 0], valid[picked, 1]] = True
    stored = np.zeros(data["mask"].shape, dtype=bool)
    if encoder_samples:
        subset = picked[generator.choice(count, size=min(encoder_samples, count), replace=False)]
        stored[valid[subset, 0], valid[subset, 1]] = True
    lengths = (data["mask"] > 0).sum(1)

    def replay() -> DemonstrationSet:
        features, actions, obs, states, state_actions = [], [], [], [], []
        for start in range(0, len(data["obs"]), batch):
            rows = slice(start, start + batch)
            horizon = int(lengths[rows].max())
            brain = rollout_for(policy, len(data["obs"][rows]))
            for t in range(horizon):
                chosen, kept = keep[rows, t], stored[rows, t]
                if kept.any():
                    before = cast(BrainRollout, brain).state
                    # Evaluated at once: a lazy slice would hold the whole batch's state of that
                    # step (166,700 x batch) until the end, about 85 MB per stored step.
                    columns = before[:, mx.array(np.flatnonzero(kept))]
                    mx.eval(columns)
                    states.append(columns)
                    obs.append(data["obs"][rows, t][kept])
                    state_actions.append(data["actions"][rows, t][kept])
                step = np.asarray(brain.features(data["obs"][rows, t]))
                if chosen.any():
                    features.append(step[chosen])
                    actions.append(data["actions"][rows, t][chosen])
        result = DemonstrationSet(
            mx.array(np.concatenate(features)), mx.array(np.concatenate(actions))
        )
        if states:
            result.states = mx.concatenate(states, axis=1)
            result.obs = mx.array(np.concatenate(obs))
            result.state_actions = mx.array(np.concatenate(state_actions))
            result.refresh = replay
            mx.eval(result.states)
        mx.clear_cache()  # the replay's per-step states are free now; return them to the system
        return result

    return replay()


def record_banks(
    config: ManipulationPPOConfig,
    model_path: Path,
    asset_root: Path,
    imitation: ManipulationImitationConfig,
    output: Path,
) -> tuple[cu.SubgoalBank, cu.SubgoalBank]:
    """Teacher subgoal-start states for training and for the per-skill evaluation.

    Recorded on the train split in two disjoint seed blocks, with the imitation run's
    observation settings, and saved in the run directory.
    """
    banks = []
    for name, per_template, offset in (
        ("bank", config.bank_episodes_per_template, cu.BANK_OFFSET),
        (
            "validation-bank",
            config.validation_bank_episodes_per_template,
            cu.VALIDATION_BANK_OFFSET,
        ),
    ):
        episodes = cu.bank_plan(per_template, offset)
        env = rollout.make_env(
            model_path,
            episodes,
            asset_root=asset_root,
            cue=imitation.cue,
            velocities=imitation.velocities,
            phase_cue=imitation.phase_cue,
        )
        bank = cu.record_bank(env, episodes)
        bank.save(output / f"{name}.npz")
        print(f"{name}: {len(bank)} subgoal starts {bank.skill_counts()}", flush=True)
        banks.append(bank)
    return banks[0], banks[1]


def run_manipulation_ppo(
    config: ManipulationPPOConfig,
    pack_root: Path,
    model_path: Path,
    output: Path,
    asset_root: Path = DEFAULT_ASSET_ROOT,
) -> dict[str, Any]:
    """Load the imitation checkpoint, score it, train it with PPO and save everything."""
    from flyarm.io import save_json

    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose a new output: {output}")
    base_run = Path(config.base_run)
    imitation = base_config(base_run)
    _, loaded = load_manipulation_policy(base_run, config.base_kind, config.base_seed, pack_root)
    if not isinstance(loaded, BrainPolicy | MLPPolicy | GRUPolicy):
        raise ValueError(f"no PPO fine-tuning is defined for {type(loaded).__name__}")
    policy: Controller = loaded
    output.mkdir(parents=True)
    save_json(output / "config.json", config.model_dump())
    if config.init_checkpoint is not None:
        checkpoint = Path(config.init_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {checkpoint}")
        policy.load(checkpoint)
        print(f"continuing from {checkpoint}", flush=True)
    bank = validation_bank = None
    if config.curriculum:
        bank, validation_bank = record_banks(config, model_path, asset_root, imitation, output)
    task = ManipulationTask(
        config,
        model_path,
        asset_root,
        cue=imitation.cue,
        velocities=imitation.velocities,
        phase_cue=imitation.phase_cue,
        bank=bank,
        validation_bank=validation_bank,
    )
    head = MotorHead(motor_decoder(policy), config.log_std, ACTION_DIM)
    eval_seeds = list(range(config.eval_episodes_per_template))
    before = task.score(policy, head, eval_seeds)
    print(
        "base checkpoint: "
        + "; ".join(task.describe(name, r, r["episodes"]) for name, r in before.items()),
        flush=True,
    )
    results: dict[str, Any] = {
        "status": "running",
        "benchmark": "articulated multi-step manipulation (flyarm.manipulation)",
        "base": before,
        "trainable_parameters": trainable_count(head),
        "claim": CLAIMS[config.base_kind]
        if config.encoder_lr == 0
        else "PPO tunes the imitation checkpoint's decoder and encoder; connectome frozen",
        "base_run": config.base_run,
        "base_kind": config.base_kind,
        "init_checkpoint": config.init_checkpoint,
        "reward": reward_config(config).__dict__,
        "selection": "validation episodes of the train split, success then subgoal fraction",
        "curriculum": [stage.model_dump() for stage in config.curriculum],
        "banks": {
            "train": None if bank is None else bank.skill_counts(),
            "validation": None if validation_bank is None else validation_bank.skill_counts(),
        },
    }
    save_json(output / "results.json", results)
    demonstrations = None
    if config.bc_weight > 0:
        demonstrations = demonstration_features(
            policy,
            base_run,
            config.bc_max_steps,
            config.seed,
            batch=128 if config.encoder_lr > 0 else 32,  # replayed again at every refresh
            encoder_samples=config.bc_encoder_samples if config.encoder_lr > 0 else 0,
        )
        steps = int(demonstrations.features.shape[0])
        stored = 0 if demonstrations.states is None else int(demonstrations.states.shape[1])
        results["demonstration_steps"] = steps
        results["demonstration_states"] = stored
        print(f"DAPG term on {steps} demonstration steps ({stored} with states)", flush=True)
    try:
        run = train_ppo(policy, task, output, config, eval_seeds, None, demonstrations)
    except (Exception, KeyboardInterrupt) as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "results.json", results)
        raise
    best = run["best"]
    chosen = next(
        (entry for entry in run["evaluations"] if entry["iteration"] == best.get("iteration")),
        None,
    )
    results.update(
        status="complete",
        best={
            "iteration": best.get("iteration"),
            "validation": best,
            "test": None if chosen is None else chosen["variants"],
        },
        final=run["evaluations"][-1],
    )
    save_json(output / "results.json", results)
    return results


__all__ = [
    "HeadActor",
    "ManipulationTask",
    "base_config",
    "demonstration_features",
    "record_banks",
    "reward_config",
    "run_manipulation_ppo",
]
