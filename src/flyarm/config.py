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
    # Learning rate for the encoder (the linear map into the ascending neurons); 0 keeps it
    # frozen, as in every run before research log E38. Above 0, each PPO iteration also
    # replays its rollout to train the encoder from reward with one-step truncated
    # gradients through the connectome, which costs about three extra forward passes.
    encoder_lr: float = Field(default=0.0, ge=0, le=0.1)
    critic_lr: float = Field(default=1e-3, gt=0, le=0.1)
    value_coef: float = Field(default=0.5, ge=0)
    entropy_coef: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Clip standardized advantages to this many standard deviations before the PPO epochs.
    # A rare terminal bonus that dwarfs the per-step terms leaves a few samples tens of sigma
    # out, and the first update after critic warmup then lands the policy on a dead fixed
    # point (research log E39); 0, the default, leaves them unclipped as in every earlier run.
    advantage_clip: float = Field(default=0.0, ge=0, le=100)
    log_std: float = Field(default=-1.2, ge=-5, le=1)
    critic_warmup: int = Field(default=5, ge=0)
    # Terminal reward for a stable placement. Holding the cube earns up to 4 per step, worth
    # 4 / (1 - gamma) = 400 at gamma 0.99, so a bonus below that teaches PPO not to release
    # (research log E34); 50 is the value of every run before E34.
    success_bonus: float = Field(default=50.0, ge=0, le=100_000)
    # Decay of the reach term with distance to the cube; the default 10 leaves almost no
    # signal beyond 30 cm, which matters only when training starts from a random policy.
    reach_slope: float = Field(default=10.0, gt=0, le=100)
    # Reward only: ignore the base run's trained weights and start from a random encoder and
    # decoder, with both frozen normalizations measured from random-action rollouts, so no
    # demonstration touches the controller; the base run still supplies the interface.
    from_scratch: bool = False
    # Which network PPO trains from scratch: the connectome policy, or the deep-RL control, a
    # tanh MLP from the observation to the action (whole_brain.policy.DirectPolicy) trained by
    # the same PPO on the same reward. "mlp" requires from_scratch and a frozen encoder.
    controller: Literal["connectome", "mlp"] = "connectome"
    scratch_envs: int = Field(default=32, ge=1, le=512)
    scratch_steps: int = Field(default=200, ge=20, le=2000)
    neural_steps: int = Field(default=3, ge=1, le=8)
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
        if self.controller == "mlp" and (not self.from_scratch or self.encoder_lr > 0):
            raise ValueError("controller 'mlp' trains from scratch and has no encoder to train")
        return self


class KitchenVariantConfig(BaseModel):
    """Harder kitchen conditions (see flyarm.rl.batched_kitchen.KitchenVariant)."""

    model_config = ConfigDict(extra="forbid")
    initial_joint_offset: float = Field(default=0.0, ge=0, le=0.5)
    kettle_mass_scale: tuple[float, float] = (1.0, 1.0)
    robot_noise_ratio: float = Field(default=0.0, ge=0, le=1)
    object_noise_ratio: float = Field(default=0.0, ge=0, le=1)
    # Prefix curriculum: pre-complete up to this many of the split's first tasks, drawn per
    # environment, with this share of environments always starting at the true beginning.
    # Evaluation variants must leave it at 0, which keeps the reported score the benchmark's.
    curriculum_prefix: int = Field(default=0, ge=0, le=3)
    curriculum_true_start_share: float = Field(default=0.25, ge=0, le=1)
    # Share of episodes that start from a random moment of a random demonstration (states only,
    # research log E53); 0 disables it and evaluation never uses it.
    demo_reset_fraction: float = Field(default=0.0, ge=0, lt=1)


def default_kitchen_eval_variants() -> dict[str, KitchenVariantConfig]:
    return {"nominal": KitchenVariantConfig()}


class KitchenPPOConfig(BaseModel):
    """PPO on the batched FrankaKitchen benchmark (flyarm.rl.batched_kitchen).

    Mirrors PPOConfig field for field; only the reward constants and the checkpoint source
    differ. ``base_run`` always supplies the frozen interface (interface.json); unless
    ``from_scratch``, it also supplies the trained checkpoint that PPO starts from.
    """

    model_config = ConfigDict(extra="forbid")
    base_run: str = "runs/whole-brain-pick-place-push2-s3"
    # Only brain policies can be warm-started: the trainer runs the encoder, the frozen
    # connectome and the readout, which the MLP and GRU controls do not have.
    base_kind: Literal["flyleg", "flyleg_shuffled"] = "flyleg"
    base_seed: int = Field(default=0, ge=0, le=999)
    # Neuron annotations; the whole-body interface is built from the pack alone, so this is
    # read only when a base run declares the front-leg interface.
    annotations: str = "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    # Reward only: ignore any trained weights and start from a random encoder and decoder,
    # with both frozen normalizations measured from random-action rollouts, so that no
    # demonstration touches the controller (research log E36).
    from_scratch: bool = False
    # Which network PPO trains from scratch: the connectome policy, or the deep-RL control, a
    # tanh MLP from the observation to the action (whole_brain.policy.DirectPolicy) trained by
    # the same PPO on the same reward. "mlp" requires from_scratch and a frozen encoder.
    controller: Literal["connectome", "mlp"] = "connectome"
    scratch_envs: int = Field(default=32, ge=1, le=512)
    scratch_steps: int = Field(default=200, ge=20, le=2000)
    # Frozen output normalization measured on those rollouts; "unit_norm" keeps a wide readout
    # from saturating the decoder under Adam (research log E32).
    readout_calibration: Literal["scale", "standardize", "unit_norm"] = "scale"
    num_envs: int = Field(default=128, ge=1, le=4096)
    rollout_steps: int = Field(default=64, ge=8, le=1024)
    iterations: int = Field(default=2000, ge=1, le=100_000)
    epochs: int = Field(default=4, ge=1, le=50)
    minibatch: int = Field(default=2048, ge=32, le=1_000_000)
    gamma: float = Field(default=0.99, gt=0, le=1)
    lam: float = Field(default=0.95, ge=0, le=1)
    clip: float = Field(default=0.2, gt=0, le=1)
    decoder_lr: float = Field(default=3e-4, gt=0, le=0.1)
    # 0 keeps the encoder frozen; above 0 the encoder is trained from reward too, with the
    # one-step truncated gradients through the connectome of research log E38. The kitchen
    # needs it, because the scene reaches the readout only through the encoder.
    encoder_lr: float = Field(default=0.0, ge=0, le=0.1)
    critic_lr: float = Field(default=1e-3, gt=0, le=0.1)
    value_coef: float = Field(default=0.5, ge=0)
    entropy_coef: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Clip standardized advantages to this many standard deviations before the PPO epochs.
    # A rare terminal bonus that dwarfs the per-step terms leaves a few samples tens of sigma
    # out, and the first update after critic warmup then lands the policy on a dead fixed
    # point (research log E39); 0, the default, leaves them unclipped as in every earlier run.
    advantage_clip: float = Field(default=0.0, ge=0, le=100)
    log_std: float = Field(default=-0.7, ge=-5, le=1)
    critic_warmup: int = Field(default=5, ge=0)
    # Reward per newly completed task. The shaped per-step terms are bounded by 1.0, so at
    # gamma 0.99 stalling forever is worth 100; a bonus at or below that teaches the policy to
    # hover at an element instead of completing it (research log E34).
    completion_bonus: float = Field(default=200.0, gt=0, le=100_000)
    # Decay of the approach term with the gripper-to-handle distance.
    approach_slope: float = Field(default=3.0, gt=0, le=100)
    # Reference term built from the joint positions of one benchmark demonstration, indexed by
    # step (in the spirit of DeepMimic and AMP). Only the demonstration's states are used, never
    # its actions, so a run with this on is still reward-only and clones nothing. 0, the default,
    # leaves every earlier run unchanged.
    tracking_weight: float = Field(default=0.0, ge=0, le=10)
    # "potential": tracking_weight * (phi(s') - phi(s)) with phi = -||q - q_ref||, that is the
    # distance closed this step. It telescopes exactly, standing still pays 0, and its total over
    # an episode is bounded by tracking_weight * the starting distance, so it leaves the E34 floor
    # alone.
    # "potential_discounted": the literal gamma * phi(s') - phi(s). Kept for the record only:
    # research log E41 measured its (1 - gamma) * distance residual paying a random policy more
    # (+0.0250 per step) than the demonstration tracker (+0.0058), the wrong way round.
    # "gaussian": tracking_weight * exp(-||q - q_ref||^2 / tracking_sigma^2). Kept for the
    # record only: research log E41 measured it numerically dead at the 5.5 rad the learning
    # policy occupies, and it raises the E34 floor for nothing in return.
    tracking_form: Literal["potential", "potential_discounted", "gaussian"] = "potential"
    tracking_sigma: float = Field(default=0.6, gt=0, le=10)
    reference_episode: int = Field(default=0, ge=0, le=1000)
    # Which uncompleted task the approach and progress terms are paid for. "split_order" is the
    # rule of every run before research log E43 and stays the default. The alternatives were all
    # measured worse there: "nearest" is the only one that does not lower the expert's shaping,
    # and it collapses the expert-to-random ratio from 2.5x to 1.2x because taking a maximum over
    # four elements inflates the term for any policy.
    target_rule: Literal[
        "split_order", "nearest", "progress", "progress_then_nearest", "moved", "moved_then_nearest"
    ] = "split_order"
    # "target" pays the approach and progress terms for one element, "sum" averages them over
    # every uncompleted element. Measured worse on both counts in research log E44 (the expert's
    # shaping falls from 0.2636 to 0.1823 and the expert-to-random ratio from 2.5x to 1.6x), so
    # "target" stays the default.
    shaping_scope: Literal["target", "sum"] = "target"
    # Scale on the approach and progress terms; 0 leaves the completion bonus alone.
    task_shaping_weight: float = Field(default=1.0, ge=0, le=100)
    # "level" pays the current approach and progress levels, the form of every run before
    # research log E45. "potential" pays their change within a step, w * (phi(s') - phi(s)),
    # rebased without payment when a completion moves the target: idleness then pays exactly 0,
    # it telescopes, and it does not enter the per-step maximum or the E34 floor.
    task_shaping_form: Literal["level", "potential"] = "level"
    # Which completions pay the bonus. "any", the default and the rule of every run before
    # research log E47, pays each newly completed task; "split_order" pays a task only once every
    # earlier task of the split is done, the order of the demonstrations. The reported score
    # always counts any order, as the benchmark does.
    completion_order: Literal["any", "split_order"] = "any"
    # Motion quality (flyarm.rl.kitchen_quality, research log E50), all 0 by default: depth pays
    # for taking each element all the way to its goal (potential form), disturbance charges for
    # moving object joints outside the split (potential form), collision charges each step the
    # robot touches anything but the target task's body, action and smoothness charge the mean
    # squared command and the mean squared change of command.
    depth_weight: float = Field(default=0.0, ge=0, le=100)
    disturbance_weight: float = Field(default=0.0, ge=0, le=100)
    collision_weight: float = Field(default=0.0, ge=0, le=100)
    action_weight: float = Field(default=0.0, ge=0, le=100)
    smoothness_weight: float = Field(default=0.0, ge=0, le=100)
    # End an episode the moment all four tasks cross the benchmark's threshold (every run before
    # E50), or run it to the horizon so that every element has to reach and stay at its goal,
    # which is what the strict score measures.
    terminate_on_all_tasks: bool = True
    # Share of every completion bonus paid only when the element comes within the strict
    # threshold (0.1) of its goal instead of the benchmark's 0.3 (research log E51); 0 pays all
    # of it at 0.3, as every earlier run did.
    strict_bonus_fraction: float = Field(default=0.0, ge=0, lt=1)
    # The distance at which the reward counts a task as done (bonus, ordered prefix, next
    # shaping target); the benchmark's 0.3 is the default and still decides the reported score
    # (research log E52).
    completion_threshold: float = Field(default=0.3, gt=0, le=0.3)
    # Paid once at an episode's end per element within the strict 0.1 of its goal that was not
    # there at the start (research log E54); 0 disables it.
    final_strict_bonus: float = Field(default=0.0, ge=0, le=10_000)
    # Demonstration-augmented PPO (DAPG, Rajeswaran et al. 2018): add bc_weight * bc_decay^k x
    # the squared error between the policy mean and the demonstrated action to the PPO loss at
    # iteration k, on the base run's own training demonstrations. It keeps a warm-started policy
    # from trading a demonstrated skill for a rewarded one (research log E46). 0, the default,
    # leaves every earlier run unchanged; above 0 it needs a warm start and a frozen encoder,
    # because the demonstrations' connectome features are computed once, before training.
    bc_weight: float = Field(default=0.0, ge=0, le=1000)
    # Continue from a PPO checkpoint (run/policy-XXXX.safetensors) instead of the base run's
    # imitation checkpoint; the base run still supplies the interface and the demonstrations.
    # The critic and optimizer state start fresh, so keep critic_warmup above 0.
    init_checkpoint: str | None = None
    bc_decay: float = Field(default=1.0, gt=0, le=1)
    bc_minibatch: int = Field(default=1024, ge=32, le=100_000)
    neural_steps: int = Field(default=3, ge=1, le=8)
    eval_every: int = Field(default=50, ge=1)
    eval_episodes: int = Field(default=20, ge=1, le=256)
    horizon: int = Field(default=280, ge=20, le=2000)
    seed: int = Field(default=0, ge=0, le=999)
    train_variant: KitchenVariantConfig = Field(default_factory=KitchenVariantConfig)
    eval_variants: dict[str, KitchenVariantConfig] = Field(
        default_factory=default_kitchen_eval_variants
    )
    best_on: str = "nominal"

    @model_validator(mode="after")
    def bonus_outweighs_stalling(self) -> KitchenPPOConfig:
        if self.best_on not in self.eval_variants:
            raise ValueError("best_on must name one of self.eval_variants")
        if self.bc_weight > 0 and (self.from_scratch or self.encoder_lr > 0):
            raise ValueError("bc_weight needs a warm start and a frozen encoder (encoder_lr 0)")
        if self.controller == "mlp" and (not self.from_scratch or self.encoder_lr > 0):
            raise ValueError("controller 'mlp' trains from scratch and has no encoder to train")
        # flyarm.rl.batched_kitchen.MAX_STEP_REWARD, plus the Gaussian reference term when it
        # is in use; the potential-based form telescopes and adds nothing to a sustained
        # trajectory. Kept as a literal so that validating a config never imports MuJoCo.
        gaussian = self.tracking_weight if self.tracking_form == "gaussian" else 0.0
        max_step_reward = 1.0 + gaussian
        if self.gamma < 1 and self.completion_bonus <= max_step_reward / (1 - self.gamma):
            raise ValueError(
                "completion_bonus must exceed the value of stalling on the per-step maximum of "
                f"{max_step_reward}, "
                f"{max_step_reward / (1 - self.gamma):.1f} at gamma {self.gamma} (log E34)"
            )
        return self


ManipulationPolicyKind = Literal["connectome", "shuffled", "gru", "mlp"]
ManipulationSplitName = Literal[
    "iid_test", "unseen_objects", "unseen_furniture", "unseen_composition"
]


def default_manipulation_eval_splits() -> list[ManipulationSplitName]:
    return ["iid_test", "unseen_objects", "unseen_furniture", "unseen_composition"]


class ManipulationImitationConfig(BaseModel):
    """Imitation (behavior cloning then DAgger) on the articulated manipulation benchmark.

    The controller is the B1a whole-body interface: the 220-feature observation into the 1,846
    ascending neurons, the 1,314 descending and 708 VNC motor neurons read out through a frozen
    unit-norm calibration, 5 actions out, one action per step. Demonstrations, DAgger labels,
    phase selection and evaluation all run in the batched environment
    (flyarm.manipulation.env.BatchedManipulation) with the scripted ManipulationTeacher.
    Counts are per task template, so every template is equally represented.
    """

    model_config = ConfigDict(extra="forbid")
    train_episodes_per_template: int = Field(default=24, ge=1, le=999)
    val_episodes_per_template: int = Field(default=2, ge=1, le=999)
    eval_episodes_per_template: int = Field(default=8, ge=1, le=999)
    eval_splits: list[ManipulationSplitName] = Field(
        default_factory=default_manipulation_eval_splits, min_length=1
    )
    # Teacher baseline on the same evaluation episodes (the reference every number is read
    # against); it costs about as much as one controller evaluation.
    evaluate_teacher: bool = True
    epochs: int = Field(default=20, ge=1, le=300)
    decoder_warmup_epochs: int = Field(default=2, ge=0, le=50)
    batch_size: int = Field(default=16, ge=1, le=128)
    bptt_steps: int = Field(default=16, ge=1, le=100)
    learning_rate: float = Field(default=0.001, gt=0, le=0.05)
    # The encoder's learning rate over learning_rate. For the linear encoder, 1 drives the
    # ascending neurons into tanh saturation within one epoch (input current RMS 0.48 to 2.2)
    # and 0.1 keeps it near 0.5 and fits better; the MLP encoder fits best at 1 and uses that
    # saturation (docs/MANIPULATION_ENV.md, "Training" and "Sensory encoder").
    encoder_learning_rate_scale: float = Field(default=0.1, gt=0, le=1)
    loss: Literal["mse", "l1"] = "l1"
    neural_steps: int = Field(default=3, ge=1, le=8)
    # The sensory periphery in front of the frozen connectome: "linear" (every earlier run) or
    # "mlp", hidden layers then a linear map to the 1,846 ascending currents, scaled at the start
    # so the currents begin at the linear encoder's RMS. The readout stays linear either way.
    encoder: Literal["linear", "mlp"] = "linear"
    encoder_hidden: list[int] = Field(default_factory=lambda: [256, 256], min_length=1)
    encoder_activation: Literal["tanh", "gelu"] = "tanh"
    # The "mlp" control: the D4RL BC architecture (two 256-unit layers), or two layers sized to
    # the connectome policy's trainable parameters ("matched").
    mlp_control: Literal["d4rl", "matched"] = "d4rl"
    # Control-scale input features (flyarm.manipulation.features): tanh of every hand-relative
    # offset at 1 and 4 cm and of the heading errors at 0.05 and 0.2 rad, appended to the
    # normalized observation inside the policy. Without them learners stall a few millimetres
    # short of the teacher's gates (docs/MANIPULATION_ENV.md, "Skill-level DAgger failure
    # analysis"). Every controller of a run gets them; the matched budgets count them.
    control_features: bool = False
    # Frozen readout normalization. At 2,022 outputs anything but "unit_norm" saturates the
    # decoder on its first Adam updates (research log E26), so the other values are refused.
    readout_calibration: Literal["scale", "standardize", "unit_norm"] = "unit_norm"
    # PPO drives one action per step, so the imitation checkpoint it warm-starts from must too.
    action_chunk: int = Field(default=1, ge=1, le=1)
    # Per-step loss weights: "skill_balanced" gives every skill of the current subgoal the same
    # total weight (clipped at max_skill_weight times the mean), "uniform" weights every step 1.
    sample_weights: Literal["skill_balanced", "uniform"] = "skill_balanced"
    max_skill_weight: float = Field(default=5.0, ge=1.0, le=100.0)
    # Batches of similar-length episodes, so a batch does not run far past most of its episodes
    # (episode sweep only).
    length_buckets: bool = True
    # "windows": every update fits window_batch windows of bptt_steps drawn uniformly from all
    # demonstrated steps, each entered after burn_in steps run without gradient from the zero
    # state; "episodes": the sweep of every earlier run, batches of batch_size whole episodes
    # walked window by window, whose correlated consecutive updates left every controller
    # underfit here (docs/MANIPULATION_ENV.md, "Imitation failure analysis").
    sampling: Literal["windows", "episodes"] = "windows"
    window_batch: int = Field(default=32, ge=1, le=1024)
    burn_in: int = Field(default=16, ge=0, le=200)
    # Closed-loop scoring on the train split's validation seeds: "closed_loop" also scores every
    # select_every behavior-cloning epochs and keeps the best epoch; phase_selection keeps the
    # best of behavior cloning and each DAgger round (research log E33).
    selection: Literal["validation_loss", "closed_loop"] = "closed_loop"
    select_every: int = Field(default=5, ge=1, le=100)
    phase_selection: Literal["last", "validation_success"] = "validation_success"
    dagger_iterations: int = Field(default=2, ge=0, le=10)
    dagger_episodes_per_template: int = Field(default=8, ge=1, le=999)
    dagger_epochs: int = Field(default=6, ge=1, le=100)
    # In round i the teacher's action is executed with probability beta * decay**i per step;
    # every visited state is labelled by the teacher either way.
    dagger_beta: float = Field(default=0.5, ge=0, le=1)
    dagger_beta_decay: float = Field(default=0.5, ge=0, le=1)
    # The memoryless sub-task cue in the observation; False is the no-cue control.
    cue: bool = True
    # The observable motor phase in the cue (flyarm.manipulation.phases); False zeroes it,
    # the no-phase-cue control (docs/MANIPULATION_ENV.md, "Phase cue").
    phase_cue: bool = False
    # Joint and object velocities in the policy's observation. Off by default: a controller
    # cloned from states with velocities learns to keep doing what they say, and a well-fit MLP
    # never left the start pose (the copycat problem, docs/MANIPULATION_ENV.md, "Imitation
    # failure analysis"); the kitchen protocol excludes them for the same reason.
    velocities: bool = False
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=10)
    policies: list[ManipulationPolicyKind] = Field(
        default_factory=lambda: ["connectome"], min_length=1
    )
    max_seconds: int = Field(default=43200, ge=60, le=259200)

    @field_validator("seeds", "policies", "eval_splits")
    @classmethod
    def unique_entries(cls, values: list) -> list:
        if len(set(values)) != len(values):
            raise ValueError("entries must be unique")
        if any(isinstance(value, int) and not 0 <= value < 1000 for value in values):
            raise ValueError("seeds must be in [0, 1000)")
        return values

    @model_validator(mode="after")
    def consistent_budget(self) -> ManipulationImitationConfig:
        if self.decoder_warmup_epochs >= self.epochs:
            raise ValueError("decoder_warmup_epochs must leave at least one joint epoch")
        if self.readout_calibration != "unit_norm":
            raise ValueError(
                "the whole-body readout has 2,022 outputs; only 'unit_norm' keeps the decoder "
                "out of tanh saturation under Adam (research log E26)"
            )
        return self


# flyarm.manipulation.sim.MAX_LEVEL_REWARD, kept as a literal so that validating a config never
# imports MuJoCo; tests/test_manipulation_training.py checks that the two agree.
MANIPULATION_MAX_LEVEL_REWARD = 0.0


class CurriculumStage(BaseModel):
    """One stage of the skill curriculum (flyarm.manipulation.curriculum).

    Each training episode starts from a true start with probability ``true_start_share``;
    otherwise from a teacher state at the start of a random subgoal (skills balanced) and ends
    after between ``min_subgoals`` and ``max_subgoals`` further subgoals, or the template's end.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    iterations: int = Field(ge=1, le=100_000)
    min_subgoals: int = Field(default=1, ge=1, le=8)
    max_subgoals: int = Field(default=1, ge=1, le=8)
    # Every stage keeps some true starts, so the full task is never absent from training.
    true_start_share: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def ordered_range(self) -> CurriculumStage:
        if self.max_subgoals < self.min_subgoals:
            raise ValueError("max_subgoals must be at least min_subgoals")
        return self


class ManipulationPPOConfig(BaseModel):
    """PPO on the batched manipulation benchmark from an imitation checkpoint.

    The trainer is flyarm.rl.ppo.train_ppo with flyarm.rl.ppo_manipulation.ManipulationTask.
    The reward is the environment's own (flyarm.manipulation.sim.RewardConfig): a bonus per
    subgoal paid once and in task order, potential-based shaping toward the current subgoal,
    and the motion-quality penalties (stray contact, disturbance, action, smoothness).
    """

    model_config = ConfigDict(extra="forbid")
    base_run: str = "runs/whole-brain-manipulation-001"
    base_kind: Literal["connectome", "shuffled"] = "connectome"
    base_seed: int = Field(default=0, ge=0, le=999)
    num_envs: int = Field(default=128, ge=1, le=4096)
    rollout_steps: int = Field(default=64, ge=8, le=1024)
    iterations: int = Field(default=1000, ge=1, le=100_000)
    epochs: int = Field(default=4, ge=1, le=50)
    minibatch: int = Field(default=2048, ge=32, le=1_000_000)
    # Episodes last 1,100 to 2,600 steps and subgoals are about 300 steps apart; at 0.99 the
    # next subgoal's bonus is discounted to 0.05, at 0.995 to 0.22.
    gamma: float = Field(default=0.995, gt=0, lt=1)
    lam: float = Field(default=0.95, ge=0, le=1)
    clip: float = Field(default=0.2, gt=0, le=1)
    decoder_lr: float = Field(default=3e-4, gt=0, le=0.1)
    # 0 keeps the encoder frozen, which the DAPG term needs (its features are computed once).
    encoder_lr: float = Field(default=0.0, ge=0, le=0.1)
    critic_lr: float = Field(default=1e-3, gt=0, le=0.1)
    value_coef: float = Field(default=0.5, ge=0)
    entropy_coef: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Research log E39: a rare large bonus leaves a few advantages tens of sigma out.
    advantage_clip: float = Field(default=10.0, ge=0, le=100)
    log_std: float = Field(default=-1.2, ge=-5, le=1)
    # Fresh critic on a warm-started actor: fit it before the policy moves (E39, E46).
    critic_warmup: int = Field(default=50, ge=0)
    # Reward (RewardConfig): subgoal_bonus per subgoal in task order, shaping on the potential,
    # and the motion-quality penalties.
    subgoal_bonus: float = Field(default=50.0, gt=0, le=100_000)
    shaping: float = Field(default=10.0, ge=0, le=1000)
    stray_contact_weight: float = Field(default=0.05, ge=0, le=100)
    disturbance_weight: float = Field(default=5.0, ge=0, le=1000)
    action_weight: float = Field(default=0.01, ge=0, le=100)
    smoothness_weight: float = Field(default=0.0, ge=0, le=100)
    # DAPG (Rajeswaran et al. 2018, research log E46): bc_weight * bc_decay^k times the squared
    # error to the imitation run's own teacher demonstrations, replayed once through the frozen
    # encoder and connectome. bc_max_steps caps how many demonstration steps are kept (drawn
    # with the skill-balanced weights, so every skill stays represented).
    bc_weight: float = Field(default=1.0, ge=0, le=1000)
    bc_decay: float = Field(default=1.0, gt=0, le=1)
    bc_minibatch: int = Field(default=1024, ge=32, le=100_000)
    bc_max_steps: int = Field(default=60_000, ge=1000, le=2_000_000)
    # Continue from a PPO checkpoint instead of the imitation checkpoint; base_run still supplies
    # the interface and the demonstrations.
    init_checkpoint: str | None = None
    # Evaluation: every eval_every iterations, eval_episodes_per_template test episodes of every
    # split in eval_splits (reported) and val_episodes_per_template episodes of the train split's
    # validation seeds (the checkpoint is selected on these only).
    eval_every: int = Field(default=50, ge=1)
    eval_episodes_per_template: int = Field(default=4, ge=1, le=999)
    val_episodes_per_template: int = Field(default=2, ge=1, le=999)
    eval_splits: list[ManipulationSplitName] = Field(
        default_factory=default_manipulation_eval_splits, min_length=1
    )
    # PPO environment seeds are 1,000,000 + 100,000 seed onward; seeds below 10 keep them under
    # the first held-out block (iid_test starts at 3,000,000).
    seed: int = Field(default=0, ge=0, le=9)
    # Skill curriculum with subgoal resets (flyarm.manipulation.curriculum); empty trains on
    # true starts only. The stages' iterations must add up to ``iterations``.
    curriculum: list[CurriculumStage] = Field(default_factory=list)
    # Teacher episodes per template whose subgoal-start states the curriculum resets to, and
    # the separate validation bank of the per-skill evaluation (single-subgoal episodes).
    bank_episodes_per_template: int = Field(default=20, ge=1, le=999)
    validation_bank_episodes_per_template: int = Field(default=4, ge=1, le=999)
    skill_eval_episodes: int = Field(default=4, ge=1, le=100)

    @property
    def best_on(self) -> str:
        """The score entry train_ppo selects checkpoints on: validation, never a test split."""
        return "validation"

    @field_validator("eval_splits")
    @classmethod
    def unique_splits(cls, values: list) -> list:
        if len(set(values)) != len(values):
            raise ValueError("entries must be unique")
        return values

    @model_validator(mode="after")
    def bonus_outweighs_stalling(self) -> ManipulationPPOConfig:
        if (
            self.curriculum
            and sum(stage.iterations for stage in self.curriculum) != self.iterations
        ):
            raise ValueError("the curriculum stages' iterations must add up to iterations")
        if self.bc_weight > 0 and self.encoder_lr > 0:
            raise ValueError("bc_weight needs a frozen encoder (encoder_lr 0)")
        # flyarm.manipulation.sim.MAX_LEVEL_REWARD: every level term is a penalty, so stalling
        # is worth at most 0 / (1 - gamma) = 0 at any gamma. Recomputed here so that validating a
        # config never imports MuJoCo; tests check the literal against the environment's.
        floor = MANIPULATION_MAX_LEVEL_REWARD / (1.0 - self.gamma)
        if self.subgoal_bonus <= floor:
            raise ValueError(
                f"subgoal_bonus must exceed the stalling floor {floor:.1f} at gamma {self.gamma}"
            )
        if self.subgoal_bonus <= self.shaping:
            raise ValueError("subgoal_bonus must exceed shaping, the most shaping pays a subgoal")
        return self


class DaggerStage(BaseModel):
    """Rounds of skill-level DAgger with one episode mix (flyarm.manipulation.skill_dagger).

    A share ``true_start_share`` of each round's episodes starts from a true start and runs the
    whole template; the rest start from recorded teacher states at the start of a subgoal
    (skills balanced) and run between ``min_subgoals`` and ``max_subgoals`` subgoals.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    rounds: int = Field(ge=1, le=40)
    min_subgoals: int = Field(default=1, ge=1, le=8)
    max_subgoals: int = Field(default=1, ge=1, le=8)
    true_start_share: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def ordered_range(self) -> DaggerStage:
        if self.max_subgoals < self.min_subgoals:
            raise ValueError("max_subgoals must be at least min_subgoals")
        return self


def default_dagger_stages() -> list[DaggerStage]:
    return [
        DaggerStage(name="single_subgoal", rounds=4, true_start_share=0.1),
        DaggerStage(
            name="two_or_three", rounds=3, min_subgoals=2, max_subgoals=3, true_start_share=0.15
        ),
        DaggerStage(
            name="full_templates", rounds=3, min_subgoals=8, max_subgoals=8, true_start_share=1.0
        ),
    ]


class SkillDaggerConfig(BaseModel):
    """Skill-level DAgger at scale on the manipulation benchmark.

    Round 0 records teacher episodes; every later round rolls out the current learner (the
    teacher's action executed instead with probability ``betas[r - 1]``, 0 once the list ends),
    labels every visited state with the teacher, adds them to the aggregate and trains the
    warm-started controller for a fixed number of updates. Rounds belong to ``stages`` in order
    (round 0 to the first stage). ``model`` is the controller and its trainer (encoder, learning
    rates, windows, burn-in, policies, seeds); its episode counts and DAgger fields are unused.
    """

    model_config = ConfigDict(extra="forbid")
    model: ManipulationImitationConfig = Field(default_factory=ManipulationImitationConfig)
    stages: list[DaggerStage] = Field(default_factory=default_dagger_stages, min_length=1)
    teacher_episodes: int = Field(default=256, ge=1, le=4096)
    episodes_per_round: int = Field(default=256, ge=1, le=4096)
    betas: list[float] = Field(default_factory=lambda: [0.5, 0.25])
    first_round_updates: int = Field(default=3000, ge=1, le=1_000_000)
    updates_per_round: int = Field(default=1500, ge=1, le=1_000_000)
    # Passes over the whole aggregate each round: the round trains for at least this many
    # epochs' worth of windows (``window_batch`` x ``bptt_steps`` labelled steps per update), and
    # never fewer updates than the fixed counts above. 0 keeps the fixed counts, which spread
    # thinner over a growing aggregate (0.4 epochs a round by 2 M steps; docs/MANIPULATION_ENV.md).
    epochs_per_round: float = Field(default=0.0, ge=0.0, le=1000.0)
    max_updates_per_round: int = Field(default=100_000, ge=1, le=10_000_000)
    # Decoder-only updates at the start of round 0 for brain policies (the readout first).
    warmup_updates: int = Field(default=300, ge=0, le=100_000)
    bank_episodes_per_template: int = Field(default=20, ge=1, le=999)
    validation_bank_episodes_per_template: int = Field(default=4, ge=1, le=999)
    skill_eval_episodes: int = Field(default=4, ge=1, le=100)
    val_episodes_per_template: int = Field(default=2, ge=1, le=999)
    eval_episodes_per_template: int = Field(default=8, ge=1, le=999)
    eval_splits: list[ManipulationSplitName] = Field(
        default_factory=default_manipulation_eval_splits, min_length=1
    )
    max_seconds: int = Field(default=86400, ge=60, le=604800)

    @property
    def rounds(self) -> int:
        return sum(stage.rounds for stage in self.stages)

    @model_validator(mode="after")
    def consistent(self) -> SkillDaggerConfig:
        if any(not 0 <= beta <= 1 for beta in self.betas):
            raise ValueError("betas must be in [0, 1]")
        if self.rounds > 40:
            raise ValueError("at most 40 rounds (DAgger seeds end below the PPO block)")
        if self.warmup_updates >= self.first_round_updates:
            raise ValueError("warmup_updates must leave joint updates in round 0")
        if self.model.sampling != "windows":
            raise ValueError("skill DAgger trains by window sampling")
        return self
