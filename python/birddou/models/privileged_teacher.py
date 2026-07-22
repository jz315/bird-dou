"""Training-only full-state Teacher and privileged critic for IS-KD."""

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

from birddou.belief.cardinality_crf import validate_assignment
from birddou.features.ragged import FEATURE_SCHEMA_VERSION, RaggedBatch
from birddou.models.action_encoder import SwiGluMlp
from birddou.models.belief_bird_dou import belief_constraints_from_batch
from birddou.models.bird_dou import BirdDouConfig, BirdDouModel, BirdDouOutput, load_bird_dou_config
from birddou.models.rank_mixer import RmsNorm

PRIVILEGED_TEACHER_ARCHITECTURE = "bird_dou_privileged_teacher_v1"
PRIVILEGED_TEACHER_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PrivilegedTeacherConfig:
    """Full-hand interaction dimensions and Oracle Dropout curriculum setting."""

    schema_version: int
    architecture: str
    feature_schema_version: int
    base: BirdDouConfig
    transformer_layers: int
    attention_heads: int
    feedforward_multiplier: int
    count_embedding_dim: int
    dropout: float
    oracle_dropout: float

    def __post_init__(self) -> None:
        if self.schema_version != PRIVILEGED_TEACHER_SCHEMA_VERSION:
            raise ValueError("unsupported privileged Teacher schema")
        if self.architecture != PRIVILEGED_TEACHER_ARCHITECTURE:
            raise ValueError("unsupported privileged Teacher architecture")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("privileged Teacher feature schema mismatch")
        if self.base.feature_schema_version != self.feature_schema_version:
            raise ValueError("Teacher and base feature schemas differ")
        if (
            min(
                self.transformer_layers,
                self.attention_heads,
                self.feedforward_multiplier,
                self.count_embedding_dim,
            )
            <= 0
        ):
            raise ValueError("privileged Teacher dimensions must be positive")
        if self.base.d_model % self.attention_heads != 0:
            raise ValueError("Teacher d_model must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.oracle_dropout <= 1.0:
            raise ValueError("Teacher dropout settings are out of range")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PrivilegedTeacherOutput:
    """Full-state action heads plus the applied hidden-rank keep mask."""

    policy: BirdDouOutput
    full_state: Tensor
    hand_tokens: Tensor
    oracle_keep_mask: Tensor


def load_privileged_teacher_config(path: Path) -> PrivilegedTeacherConfig:
    """Load a Teacher config and its explicitly referenced public base model."""
    resolved = path.resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    root = _mapping(raw, "Teacher config")
    teacher = _mapping(root.get("teacher"), "teacher")
    base_path = Path(_string(root, "base_model_config"))
    if not base_path.is_absolute():
        base_path = (resolved.parents[2] / base_path).resolve()
    return PrivilegedTeacherConfig(
        schema_version=_integer(root, "schema_version"),
        architecture=_string(root, "architecture"),
        feature_schema_version=_integer(root, "feature_schema_version"),
        base=load_bird_dou_config(base_path),
        transformer_layers=_integer(teacher, "transformer_layers"),
        attention_heads=_integer(teacher, "attention_heads"),
        feedforward_multiplier=_integer(teacher, "feedforward_multiplier"),
        count_embedding_dim=_integer(teacher, "count_embedding_dim"),
        dropout=_number(teacher, "dropout"),
        oracle_dropout=_number(teacher, "oracle_dropout"),
    )


class FullHandInteractionEncoder(nn.Module):
    """Encode three exact remaining hands and mix their 45 rank tokens globally."""

    def __init__(self, config: PrivilegedTeacherConfig) -> None:
        super().__init__()
        width = config.base.d_model
        self.width = width
        self.rank_embedding = nn.Embedding(15, config.count_embedding_dim)
        self.player_embedding = nn.Embedding(3, config.count_embedding_dim)
        self.count_embedding = nn.Embedding(5, config.count_embedding_dim)
        self.mask_embedding = nn.Parameter(torch.zeros(config.count_embedding_dim))
        self.input_projection = nn.Linear(config.count_embedding_dim * 3, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.attention_heads,
            dim_feedforward=width * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=RmsNorm(width),
            enable_nested_tensor=False,
        )

    def forward(self, hands: Tensor, oracle_keep_mask: Tensor) -> Tensor:
        batch_size = hands.shape[0]
        if hands.dtype != torch.int64 or hands.shape != (batch_size, 3, 15):
            raise ValueError("privileged hands must be int64 [B, 3, 15]")
        if oracle_keep_mask.dtype != torch.bool or oracle_keep_mask.shape != hands.shape:
            raise ValueError("Oracle keep mask must be bool [B, 3, 15]")
        if torch.any((hands < 0) | (hands > 4)):
            raise ValueError("privileged hand counts must be in 0..4")
        ranks = torch.arange(15, dtype=torch.int64, device=hands.device)
        players = torch.arange(3, dtype=torch.int64, device=hands.device)
        counts = self.count_embedding(hands)
        hidden_keep = oracle_keep_mask.unsqueeze(-1)
        counts = torch.where(hidden_keep, counts, self.mask_embedding)
        embedded = torch.cat(
            (
                counts,
                self.rank_embedding(ranks)[None, None].expand(batch_size, 3, -1, -1),
                self.player_embedding(players)[None, :, None].expand(batch_size, -1, 15, -1),
            ),
            dim=-1,
        )
        tokens = self.input_projection(embedded).reshape(batch_size, 45, self.width)
        return cast(Tensor, self.interaction(tokens)).reshape(batch_size, 3, 15, self.width)


class PrivilegedTeacher(nn.Module):
    """Full-state Teacher; never used by the Student inference checkpoint."""

    def __init__(self, config: PrivilegedTeacherConfig) -> None:
        super().__init__()
        self.config = config
        self.base = BirdDouModel(config.base)
        self.full_hands = FullHandInteractionEncoder(config)
        width = config.base.d_model
        self.full_state_fusion = SwiGluMlp(width * 4, width * 2, width, config.dropout)

    def forward(
        self,
        batch: RaggedBatch,
        true_assignment_a: Tensor,
        *,
        oracle_dropout: float | None = None,
        generator: torch.Generator | None = None,
    ) -> PrivilegedTeacherOutput:
        public = self.base.encode_public_state(batch)
        unknown, capacity_a, _ = belief_constraints_from_batch(batch)
        validate_assignment(true_assignment_a, unknown, capacity_a)
        own = batch.rank_categorical[..., 1]
        assignment_b = unknown - true_assignment_a
        hands = torch.stack((own, true_assignment_a, assignment_b), dim=1)
        probability = self.config.oracle_dropout if oracle_dropout is None else oracle_dropout
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("oracle_dropout must be finite and in 0..1")
        keep = torch.ones_like(hands, dtype=torch.bool)
        if probability > 0.0:
            random = torch.rand(
                (batch.batch_size, 2, 15),
                dtype=torch.float32,
                device=hands.device,
                generator=generator,
            )
            keep[:, 1:] = random >= probability
        hand_tokens = self.full_hands(hands, keep)
        player_states = hand_tokens.mean(dim=2)
        full_state = self.full_state_fusion(
            torch.cat(
                (
                    public.pre_belief_state,
                    player_states[:, 0],
                    player_states[:, 1],
                    player_states[:, 2],
                ),
                dim=-1,
            )
        )
        policy = self.base.forward_from_state(batch, public, full_state)
        return PrivilegedTeacherOutput(policy, full_state, hand_tokens, keep)


class PrivilegedCritic(nn.Module):
    """Centralized training critic exposing Teacher MC-Q for every legal action."""

    def __init__(self, teacher: PrivilegedTeacher) -> None:
        super().__init__()
        self.teacher = teacher

    def forward(self, batch: RaggedBatch, true_assignment_a: Tensor) -> Tensor:
        return cast(Tensor, self.teacher(batch, true_assignment_a).policy.mc_q)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Teacher config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Teacher config {key} must be a finite number")
    return float(value)


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Teacher config {key} must be a non-empty string")
    return value


__all__ = (
    "PRIVILEGED_TEACHER_ARCHITECTURE",
    "PRIVILEGED_TEACHER_SCHEMA_VERSION",
    "FullHandInteractionEncoder",
    "PrivilegedCritic",
    "PrivilegedTeacher",
    "PrivilegedTeacherConfig",
    "PrivilegedTeacherOutput",
    "load_privileged_teacher_config",
)
