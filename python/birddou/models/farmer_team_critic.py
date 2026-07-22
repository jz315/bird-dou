"""Training-only full-state team critic for sequential farmer cooperation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from birddou.features.ragged import FEATURE_SCHEMA_VERSION, RaggedBatch
from birddou.models.privileged_teacher import (
    PrivilegedTeacher,
    PrivilegedTeacherConfig,
    PrivilegedTeacherOutput,
    load_privileged_teacher_config,
)
from birddou.models.rank_mixer import RmsNorm

FARMER_TEAM_CRITIC_SCHEMA_VERSION = 1
FARMER_TEAM_CRITIC_ARCHITECTURE = "bird_dou_farmer_team_critic_v1"


@dataclass(frozen=True, slots=True)
class FarmerTeamCriticConfig:
    """Versioned centralized-critic dimensions around the privileged trunk."""

    schema_version: int
    architecture: str
    feature_schema_version: int
    teacher: PrivilegedTeacherConfig
    hidden_multiplier: int
    dropout: float

    def __post_init__(self) -> None:
        if self.schema_version != FARMER_TEAM_CRITIC_SCHEMA_VERSION:
            raise ValueError("unsupported Farmer Team Critic schema")
        if self.architecture != FARMER_TEAM_CRITIC_ARCHITECTURE:
            raise ValueError("unsupported Farmer Team Critic architecture")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("Farmer Team Critic feature schema mismatch")
        if self.teacher.feature_schema_version != self.feature_schema_version:
            raise ValueError("Farmer Team Critic and Teacher feature schemas differ")
        if self.hidden_multiplier <= 0:
            raise ValueError("Farmer Team Critic hidden multiplier must be positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("Farmer Team Critic dropout must be in [0, 1)")

    def fingerprint(self) -> str:
        """Return a stable architecture/configuration identity."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class FarmerTeamCriticOutput:
    """Team Q for every current farmer action plus privileged diagnostics."""

    team_q: Tensor
    state_seat: Tensor
    action_seat: Tensor
    privileged: PrivilegedTeacherOutput


def load_farmer_team_critic_config(path: Path) -> FarmerTeamCriticConfig:
    """Load a Critic config and its referenced privileged Teacher config."""
    resolved = path.resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    root = _mapping(raw, "Farmer Team Critic config")
    head = _mapping(root.get("head"), "Farmer Team Critic head")
    teacher_path = Path(_string(root, "teacher_config"))
    if not teacher_path.is_absolute():
        teacher_path = (resolved.parents[2] / teacher_path).resolve()
    return FarmerTeamCriticConfig(
        schema_version=_integer(root, "schema_version"),
        architecture=_string(root, "architecture"),
        feature_schema_version=_integer(root, "feature_schema_version"),
        teacher=load_privileged_teacher_config(teacher_path),
        hidden_multiplier=_integer(head, "hidden_multiplier"),
        dropout=_number(head, "dropout"),
    )


class FarmerTeamCritic(nn.Module):
    """Estimate farmer-team Q from exact hands; never exported with the Actor."""

    def __init__(self, config: FarmerTeamCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.privileged = PrivilegedTeacher(config.teacher)
        width = config.teacher.base.d_model
        hidden = width * config.hidden_multiplier
        self.seat_embedding = nn.Embedding(3, width)
        self.team_head = nn.Sequential(
            RmsNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: RaggedBatch, true_assignment_a: Tensor) -> FarmerTeamCriticOutput:
        """Score every legal action for farmer-only full-state training rows."""
        privileged = self.privileged(batch, true_assignment_a, oracle_dropout=0.0)
        state_seat = privileged.policy.seat
        if torch.any((state_seat != 1) & (state_seat != 2)):
            raise ValueError("Farmer Team Critic accepts only farmer decision states")
        action_seat = state_seat[batch.action_state_index]
        action = privileged.policy.actions.action + self.seat_embedding(action_seat)
        team_q = self.team_head(action).squeeze(-1)
        if team_q.shape != (batch.action_count,) or not torch.isfinite(team_q).all():
            raise RuntimeError("Farmer Team Critic produced invalid action values")
        return FarmerTeamCriticOutput(team_q, state_seat, action_seat, privileged)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Farmer Team Critic config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Farmer Team Critic config {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Farmer Team Critic config {key} must be finite")
    return numeric


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Farmer Team Critic config {key} must be a non-empty string")
    return value


__all__ = (
    "FARMER_TEAM_CRITIC_ARCHITECTURE",
    "FARMER_TEAM_CRITIC_SCHEMA_VERSION",
    "FarmerTeamCritic",
    "FarmerTeamCriticConfig",
    "FarmerTeamCriticOutput",
    "load_farmer_team_critic_config",
)
