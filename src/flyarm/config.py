"""Validated reproducible experiment budgets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PolicyKind = Literal["connectome", "shuffled", "mlp", "gru"]
PickPlacePolicyKind = Literal["restricted_connectome", "restricted_shuffled", "mlp", "gru"]


def default_policies() -> list[PolicyKind]:
    return ["connectome", "shuffled", "mlp", "gru"]


def default_pick_place_policies() -> list[PickPlacePolicyKind]:
    return ["restricted_connectome", "restricted_shuffled", "mlp", "gru"]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_episodes: int = Field(default=96, ge=8, le=1024)
    val_episodes: int = Field(default=16, ge=4, le=256)
    test_episodes: int = Field(default=24, ge=4, le=256)
    horizon: int = Field(default=100, ge=20, le=300)
    epochs: int = Field(default=35, ge=1, le=300)
    batch_size: int = Field(default=16, ge=1, le=128)
    bptt_steps: int = Field(default=20, ge=1, le=100)
    learning_rate: float = Field(default=0.002, gt=0, le=0.05)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=10)
    policies: list[PolicyKind] = Field(default_factory=default_policies, min_length=1)
    max_seconds: int = Field(default=1800, ge=30, le=86400)
    threads: int = Field(default=2, ge=1, le=8)

    @field_validator("seeds", "policies")
    @classmethod
    def unique_entries(cls, values: list) -> list:
        if len(set(values)) != len(values):
            raise ValueError("entries must be unique")
        if any(isinstance(value, int) and not 0 <= value < 2**32 - 10000 for value in values):
            raise ValueError("seeds must be nonnegative and below 2**32 - 10000")
        return values


class PickPlaceConfig(BaseModel):
    """Bounded behavior-cloning and causal-evaluation budget for pick-and-place."""

    model_config = ConfigDict(extra="forbid")
    train_episodes: int = Field(default=96, ge=16, le=512)
    val_episodes: int = Field(default=16, ge=4, le=128)
    test_episodes: int = Field(default=24, ge=4, le=128)
    horizon: int = Field(default=400, ge=200, le=600)
    epochs: int = Field(default=50, ge=1, le=300)
    batch_size: int = Field(default=8, ge=1, le=64)
    bptt_steps: int = Field(default=40, ge=1, le=200)
    learning_rate: float = Field(default=0.002, gt=0, le=0.05)
    internal_steps: int = Field(default=3, ge=1, le=8)
    dagger_iterations: int = Field(default=0, ge=0, le=5)
    dagger_episodes: int = Field(default=16, ge=4, le=128)
    dagger_epochs: int = Field(default=15, ge=1, le=100)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=10)
    policies: list[PickPlacePolicyKind] = Field(
        default_factory=default_pick_place_policies, min_length=1
    )
    max_seconds: int = Field(default=3600, ge=60, le=86400)
    threads: int = Field(default=2, ge=1, le=8)

    @field_validator("seeds", "policies")
    @classmethod
    def unique_entries(cls, values: list) -> list:
        if len(set(values)) != len(values):
            raise ValueError("entries must be unique")
        if any(isinstance(value, int) and not 0 <= value < 2**32 - 40000 for value in values):
            raise ValueError("seeds must be nonnegative and below 2**32 - 40000")
        return values


WholeBrainPolicyKind = Literal["connectome", "shuffled", "gru"]


def default_whole_brain_policies() -> list[WholeBrainPolicyKind]:
    return ["connectome", "shuffled", "gru"]


class WholeBrainConfig(BaseModel):
    """B1a budget: frozen full MaleCNS, trainable encoder/decoder, same tasks and teachers."""

    model_config = ConfigDict(extra="forbid")
    task: Literal["reach", "pick-place"]
    # "resync" (protocol v2) lets the pick-place teacher re-derive its stage in learner-reached
    # states; "stateful" is the first protocol, whose DAgger labels could say "open the
    # gripper" to a learner holding a lifted cube (research log E23).
    teacher: Literal["stateful", "resync"] = "stateful"
    train_episodes: int = Field(default=96, ge=8, le=512)
    val_episodes: int = Field(default=16, ge=4, le=128)
    test_episodes: int = Field(default=24, ge=4, le=128)
    horizon: int = Field(default=100, ge=20, le=600)
    epochs: int = Field(default=20, ge=1, le=300)
    decoder_warmup_epochs: int = Field(default=2, ge=0, le=50)
    batch_size: int = Field(default=8, ge=1, le=64)
    bptt_steps: int = Field(default=8, ge=1, le=100)
    learning_rate: float = Field(default=0.002, gt=0, le=0.05)
    neural_steps: int = Field(default=3, ge=1, le=8)
    dagger_iterations: int = Field(default=0, ge=0, le=5)
    dagger_episodes: int = Field(default=16, ge=4, le=128)
    dagger_epochs: int = Field(default=10, ge=1, le=100)
    noise_std_m: float = Field(default=0.01, ge=0, le=0.05)
    # Which training phase the evaluated checkpoint comes from: the last one (every run before
    # research log E33) or the one with the best closed-loop success on the validation seeds.
    phase_selection: Literal["last", "validation_success"] = "last"
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=10)
    policies: list[WholeBrainPolicyKind] = Field(
        default_factory=default_whole_brain_policies, min_length=1
    )
    # Independent degree-preserving shuffles trained per seed; replicate r uses shuffle seed
    # seed + 17000 + 1000 r, so replicate 0 is the shuffle of every earlier run.
    shuffle_replicates: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=5)
    max_seconds: int = Field(default=7200, ge=60, le=172800)

    @field_validator("seeds", "policies", "shuffle_replicates")
    @classmethod
    def unique_entries(cls, values: list) -> list:
        if len(set(values)) != len(values):
            raise ValueError("entries must be unique")
        if any(isinstance(value, int) and not 0 <= value < 2**32 - 40000 for value in values):
            raise ValueError("seeds must be nonnegative and below 2**32 - 40000")
        return values

    @model_validator(mode="after")
    def warmup_leaves_joint_epochs(self) -> WholeBrainConfig:
        if self.shuffle_replicates != [0] and (
            max(self.seeds) >= 1000 or max(self.shuffle_replicates) >= 10
        ):
            raise ValueError("shuffle replicates need seeds < 1000 and replicates < 10")
        if self.decoder_warmup_epochs >= self.epochs:
            raise ValueError("decoder_warmup_epochs must leave at least one joint epoch")
        if self.task == "pick-place" and self.horizon < 200:
            raise ValueError("pick-place needs a horizon of at least 200 control steps")
        if self.task == "reach" and self.dagger_iterations:
            raise ValueError("DAgger is defined for the pick-place teacher only")
        return self


FlyLegPolicyKind = Literal["flyleg", "flyleg_shuffled", "mlp", "gru", "act"]


def default_flyleg_policies() -> list[FlyLegPolicyKind]:
    return ["flyleg", "flyleg_shuffled", "mlp", "gru"]


class FlyLegConfig(BaseModel):
    """B2 budget: front-leg MaleCNS vs controls on one D4RL FrankaKitchen split."""

    model_config = ConfigDict(extra="forbid")
    split: Literal["complete", "partial", "mixed"]
    # "front_leg": the arm as the fly's left front leg (the options below apply);
    # "whole_body": the B1a pick-and-place interface, every feature into the 1,846 ascending
    # neurons and the 1,314 descending plus 708 VNC motor neurons read out (research log E32).
    interface: Literal["front_leg", "whole_body"] = "front_leg"
    # "proprioception" wires only the front-leg proprioceptors; scene state is then unused.
    sensory_channels: Literal["proprioception+head", "proprioception"] = "proprioception+head"
    # What the decoder reads: the 68 left front-leg motor neurons, or those plus the brain's
    # 1,314 descending command neurons.
    readout: Literal["leg_motor", "leg_motor+descending"] = "leg_motor"
    # Frozen output normalization before the decoder: divide by RMS ("scale"), subtract the
    # mean activity first and divide by the standard deviation ("standardize"), or
    # standardize and divide by sqrt(outputs) so that a wide readout does not saturate the
    # decoder under Adam ("unit_norm").
    readout_calibration: Literal["scale", "standardize", "unit_norm"] = "scale"
    validation_fraction: float = Field(default=0.1, gt=0, le=0.3)
    epochs: int = Field(default=60, ge=1, le=1000)
    decoder_warmup_epochs: int = Field(default=2, ge=0, le=50)
    batch_size: int = Field(default=8, ge=1, le=128)
    bptt_steps: int = Field(default=8, ge=1, le=100)
    learning_rate: float = Field(default=0.001, gt=0, le=0.05)
    neural_steps: int = Field(default=3, ge=1, le=8)
    # Rate-model regime (weights unchanged): h <- 0.5 h + 0.5 tanh(I + g W h). Near 1 the
    # connectome keeps information for seconds instead of about 0.2 s.
    recurrent_gain: float = Field(default=0.8, ge=0, lt=1)
    # Synaptic weights: signed contacts over the L-p norm of each neuron's inputs. 1 is the
    # default (inputs sum to at most 1: quiet, near-linear); up to 2 (variance preserving).
    weight_norm_power: float = Field(default=1.0, ge=1.0, le=2.0)
    # ACT-style output: every controller predicts the next action_chunk actions at each step,
    # executed through a fixed temporal ensemble. 1 is ordinary single-step control.
    action_chunk: int = Field(default=1, ge=1, le=50)
    loss: Literal["mse", "l1"] = "mse"
    # "act" is the ACT reference (transformer + CVAE); it is not a fly model.
    act_steps: int = Field(default=20000, ge=100, le=500000)
    act_batch_size: int = Field(default=64, ge=8, le=1024)
    act_learning_rate: float = Field(default=1e-4, gt=0, le=0.01)
    act_kl_weight: float = Field(default=10.0, ge=0, le=100)
    # Checkpoint selection: "closed_loop" keeps the weights with the most tasks completed on
    # selection_episodes held-out validation episodes (seeds disjoint from evaluation),
    # scored every select_every epochs; "validation_loss" keeps the lowest imitation loss.
    selection: Literal["validation_loss", "closed_loop"] = "validation_loss"
    selection_episodes: int = Field(default=5, ge=1, le=50)
    select_every: int = Field(default=5, ge=1, le=100)
    # DART: extra training episodes in which the demonstration tracker acts with Gaussian
    # action noise and every visited state is labelled with its clean action.
    dart_episodes: int = Field(default=0, ge=0, le=2000)
    dart_noise: float = Field(default=0.1, gt=0, le=1)
    # DART episodes may also start from a perturbed arm: each of the 7 arm joints offset
    # uniformly in [-dart_start_offset, dart_start_offset] rad (seeded), so the data covers the
    # recoveries that a perturbed-start evaluation needs (research log E29, E30).
    dart_start_offset: float = Field(default=0.0, ge=0, le=0.5)
    # Closed-loop checkpoint selection from starts perturbed the same way (0: clean starts).
    selection_joint_offset: float = Field(default=0.0, ge=0, le=0.5)
    # Interactive imitation: after behavior cloning, each iteration rolls out the learner from
    # clean starts (seeds disjoint from evaluation), labels the visited states with the
    # demonstration tracker and retrains on demonstrations plus every labelled rollout.
    dagger_iterations: int = Field(default=0, ge=0, le=10)
    dagger_episodes: int = Field(default=20, ge=1, le=200)
    dagger_epochs: int = Field(default=30, ge=1, le=500)
    # Mixed rollouts: in iteration i the teacher's action is executed with probability
    # dagger_beta * dagger_beta_decay**i (per step), the learner's otherwise; every visited
    # state is labelled by the teacher either way. 0 is pure learner rollouts.
    dagger_beta: float = Field(default=0.0, ge=0, le=1)
    # DAgger rollouts from arm starts perturbed like the joint-offset evaluation (0: clean).
    dagger_start_offset: float = Field(default=0.0, ge=0, le=0.5)
    # Start every controller from {init_from}/{kind}-{seed}/policy.safetensors of an earlier
    # run with the same interface and action chunk, and skip behavior cloning.
    init_from: str | None = None
    dagger_beta_decay: float = Field(default=0.5, ge=0, le=1)
    act_dagger_steps: int = Field(default=5000, ge=100, le=200000)
    tracker_gain: float = Field(default=0.5, ge=0, le=1)
    eval_episodes: int = Field(default=50, ge=2, le=500)
    ood_joint_offsets: list[float] = Field(default_factory=lambda: [0.05, 0.1], max_length=6)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=10)
    policies: list[FlyLegPolicyKind] = Field(default_factory=default_flyleg_policies, min_length=1)
    max_seconds: int = Field(default=43200, ge=60, le=259200)

    @field_validator("seeds", "policies")
    @classmethod
    def unique_entries(cls, values: list) -> list:
        if len(set(values)) != len(values):
            raise ValueError("entries must be unique")
        return values

    @model_validator(mode="after")
    def warmup_leaves_joint_epochs(self) -> FlyLegConfig:
        if self.decoder_warmup_epochs >= self.epochs:
            raise ValueError("decoder_warmup_epochs must leave at least one joint epoch")
        if any(not 0 < offset <= 0.5 for offset in self.ood_joint_offsets):
            raise ValueError("OOD joint offsets must be in (0, 0.5] rad")
        if (self.dagger_iterations or self.dart_episodes) and self.split != "complete":
            raise ValueError("The demonstration tracker is validated as a teacher on complete only")
        # DAgger rollouts use env seeds 200000 + 10000 seed + ...; keep them below DART's.
        if self.init_from is not None and not self.dagger_iterations:
            raise ValueError("init_from skips behavior cloning, so it needs DAgger iterations")
        if self.dagger_iterations and any(not 0 <= seed < 10 for seed in self.seeds):
            raise ValueError("DAgger needs training seeds in [0, 10) to keep env seeds disjoint")
        if self.interface == "whole_body" and (
            self.sensory_channels != "proprioception+head" or self.readout != "leg_motor"
        ):
            raise ValueError("sensory_channels and readout describe the front-leg interface only")
        return self


class TaskVariantConfig(BaseModel):
    """Harder pick-and-place conditions (see flyarm.rl.batched_pick_place.TaskVariant)."""

    model_config = ConfigDict(extra="forbid")
    mass_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)
    goal_reach: float = Field(default=0.11, ge=0.05, le=0.3)
    goal_visible_steps: int | None = Field(default=None, ge=1)


def default_eval_variants() -> dict[str, TaskVariantConfig]:
    return {"nominal": TaskVariantConfig()}


class PPOConfig(BaseModel):
    """PPO fine-tuning of the motor decoder of a trained B1a pick-and-place checkpoint."""

    model_config = ConfigDict(extra="forbid")
    base_run: str = "runs/whole-brain-pick-place-001"
    base_kind: Literal["connectome", "shuffled", "gru"] = "connectome"
    base_seed: int = Field(default=0, ge=0, le=999)
    num_envs: int = Field(default=128, ge=1, le=4096)
    rollout_steps: int = Field(default=64, ge=8, le=1024)
    iterations: int = Field(default=600, ge=1, le=100_000)
    epochs: int = Field(default=4, ge=1, le=50)
    minibatch: int = Field(default=2048, ge=32, le=1_000_000)
    gamma: float = Field(default=0.99, gt=0, le=1)
    lam: float = Field(default=0.95, ge=0, le=1)
    clip: float = Field(default=0.2, gt=0, le=1)
    decoder_lr: float = Field(default=3e-4, gt=0, le=0.1)
    critic_lr: float = Field(default=1e-3, gt=0, le=0.1)
    value_coef: float = Field(default=0.5, ge=0)
    entropy_coef: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    log_std: float = Field(default=-1.2, ge=-5, le=1)
    critic_warmup: int = Field(default=5, ge=0)
    eval_every: int = Field(default=20, ge=1)
    eval_episodes: int = Field(default=24, ge=1, le=256)
    horizon: int = Field(default=400, ge=20, le=2000)
    seed: int = Field(default=0, ge=0, le=999)
    train_variant: TaskVariantConfig = Field(default_factory=TaskVariantConfig)
    eval_variants: dict[str, TaskVariantConfig] = Field(default_factory=default_eval_variants)
    best_on: str = "nominal"

    @model_validator(mode="after")
    def best_variant_is_evaluated(self) -> PPOConfig:
        if self.best_on not in self.eval_variants:
            raise ValueError("best_on must name one of eval_variants")
        return self
