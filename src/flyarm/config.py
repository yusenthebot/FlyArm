"""Validated reproducible experiment budgets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PolicyKind = Literal["connectome", "shuffled", "mlp", "gru"]


def default_policies() -> list[PolicyKind]:
    return ["connectome", "shuffled", "mlp", "gru"]


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
