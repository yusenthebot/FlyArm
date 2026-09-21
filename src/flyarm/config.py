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
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=10)
    policies: list[WholeBrainPolicyKind] = Field(
        default_factory=default_whole_brain_policies, min_length=1
    )
    max_seconds: int = Field(default=7200, ge=60, le=172800)

    @field_validator("seeds", "policies")
    @classmethod
    def unique_entries(cls, values: list) -> list:
        if len(set(values)) != len(values):
            raise ValueError("entries must be unique")
        if any(isinstance(value, int) and not 0 <= value < 2**32 - 40000 for value in values):
            raise ValueError("seeds must be nonnegative and below 2**32 - 40000")
        return values

    @model_validator(mode="after")
    def warmup_leaves_joint_epochs(self) -> WholeBrainConfig:
        if self.decoder_warmup_epochs >= self.epochs:
            raise ValueError("decoder_warmup_epochs must leave at least one joint epoch")
        if self.task == "pick-place" and self.horizon < 200:
            raise ValueError("pick-place needs a horizon of at least 200 control steps")
        if self.task == "reach" and self.dagger_iterations:
            raise ValueError("DAgger is defined for the pick-place teacher only")
        return self
