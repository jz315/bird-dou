"""Full-history GRU/causal-Transformer encoder with seat-aware gating."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from birddou.features.ragged import HISTORY_META_COLUMNS, SCALAR_COLUMNS
from birddou.models.rank_mixer import RmsNorm

HISTORY_ENCODER_SCHEMA_VERSION = 1
HISTORY_RANK_COUNT = 15


@dataclass(frozen=True, slots=True)
class HistoryEncoderConfig:
    """Versioned dimensions and independent history-branch switches."""

    schema_version: int = HISTORY_ENCODER_SCHEMA_VERSION
    d_model: int = 256
    max_length: int = 96
    count_embedding_dim: int = 8
    categorical_embedding_dim: int = 8
    gru_layers: int = 2
    transformer_layers: int = 3
    attention_heads: int = 8
    feedforward_multiplier: int = 4
    dropout: float = 0.0
    gru_enabled: bool = True
    transformer_enabled: bool = True
    role_gate_enabled: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.d_model,
            self.max_length,
            self.count_embedding_dim,
            self.categorical_embedding_dim,
            self.feedforward_multiplier,
        )
        if self.schema_version != HISTORY_ENCODER_SCHEMA_VERSION:
            raise ValueError("unsupported history encoder schema")
        if any(value <= 0 for value in positive):
            raise ValueError("history encoder dimensions must be positive")
        if not self.gru_enabled and not self.transformer_enabled:
            raise ValueError("at least one history branch must be enabled")
        if self.gru_enabled and self.gru_layers <= 0:
            raise ValueError("enabled GRU requires at least one layer")
        if self.transformer_enabled:
            if self.transformer_layers <= 0 or self.attention_heads <= 0:
                raise ValueError("enabled Transformer requires positive layers and heads")
            if self.d_model % self.attention_heads != 0:
                raise ValueError("d_model must be divisible by history attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("history dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class HistoryEncoding:
    """Fused state plus branch and gate diagnostics for ablation audits."""

    fused: Tensor
    gru: Tensor
    transformer: Tensor
    gate: Tensor
    scalar: Tensor


def load_history_encoder_config(path: Path) -> HistoryEncoderConfig:
    """Load the history section from the versioned model configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "model config")
    history = _mapping(root.get("history"), "history")
    return HistoryEncoderConfig(
        schema_version=_integer(root, "schema_version"),
        d_model=_integer(root, "d_model"),
        max_length=_integer(history, "max_length"),
        count_embedding_dim=_integer(history, "count_embedding_dim"),
        categorical_embedding_dim=_integer(history, "categorical_embedding_dim"),
        gru_layers=_integer(history, "gru_layers"),
        transformer_layers=_integer(history, "transformer_layers"),
        attention_heads=_integer(history, "attention_heads"),
        feedforward_multiplier=_integer(history, "feedforward_multiplier"),
        dropout=_number(history, "dropout"),
        gru_enabled=_boolean(history, "gru_enabled"),
        transformer_enabled=_boolean(history, "transformer_enabled"),
        role_gate_enabled=_boolean(history, "role_gate_enabled"),
    )


class HistoryEventEncoder(nn.Module):
    """Embed rank counts, categorical event tags, numeric metadata, and position."""

    _CATEGORICAL_CARDINALITIES = (4, 3, 2, 2, 2, 2, 16, 17, 3)
    _CATEGORICAL_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 9)
    _NUMERIC_INDICES = (8, 10, 11, 12, 13, 14)
    _NUMERIC_SCALES = (12.0, 20.0, 20.0, 15.0, 32.0, 96.0)

    def __init__(self, config: HistoryEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.count_embedding = nn.Embedding(5, config.count_embedding_dim)
        self.rank_embedding = nn.Embedding(HISTORY_RANK_COUNT, config.count_embedding_dim)
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.categorical_embedding_dim)
            for cardinality in self._CATEGORICAL_CARDINALITIES
        )
        rank_width = HISTORY_RANK_COUNT * config.count_embedding_dim
        categorical_width = len(self._CATEGORICAL_INDICES) * config.categorical_embedding_dim
        input_width = rank_width + categorical_width + len(self._NUMERIC_INDICES)
        self.projection = nn.Linear(input_width, config.d_model)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.norm = RmsNorm(config.d_model)

    def forward(self, rank_counts: Tensor, metadata: Tensor, mask: Tensor) -> Tensor:
        _validate_history_inputs(rank_counts, metadata, mask, self.config.max_length)
        if torch.any((rank_counts < 0) | (rank_counts > 4)):
            raise ValueError("history rank counts must be in 0..4")
        categorical_pieces: list[Tensor] = []
        for index, (column, embedding, cardinality) in enumerate(
            zip(
                self._CATEGORICAL_INDICES,
                self.categorical_embeddings,
                self._CATEGORICAL_CARDINALITIES,
                strict=True,
            )
        ):
            values = metadata[..., column]
            if torch.any((values < 0) | (values >= cardinality)):
                label = HISTORY_META_COLUMNS[column]
                raise ValueError(f"history {label} values exceed embedding range at field {index}")
            categorical_pieces.append(cast(Tensor, embedding(values)))
        rank_ids = torch.arange(
            HISTORY_RANK_COUNT,
            dtype=torch.int64,
            device=rank_counts.device,
        )
        rank_values = self.count_embedding(rank_counts) + self.rank_embedding(rank_ids)[None, None]
        rank_values = rank_values.flatten(start_dim=2)
        numeric = metadata[..., self._NUMERIC_INDICES].to(torch.float32)
        scales = torch.tensor(
            self._NUMERIC_SCALES,
            dtype=torch.float32,
            device=metadata.device,
        )
        numeric = numeric / scales
        embedded = torch.cat((rank_values, *categorical_pieces, numeric), dim=-1)
        projected = cast(Tensor, self.projection(embedded))
        positions = torch.arange(metadata.shape[1], device=metadata.device)
        projected = projected + self.position_embedding(positions)[None]
        projected = cast(Tensor, self.norm(projected))
        return projected * mask.unsqueeze(-1)


class HistoryGruBranch(nn.Module):
    """Multi-layer recurrent summary selected at the final valid event."""

    def __init__(self, config: HistoryEncoderConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            config.d_model,
            config.d_model,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )

    def forward(self, events: Tensor, mask: Tensor) -> Tensor:
        sequence, _ = self.gru(events)
        return _last_valid(sequence, mask)


class CausalHistoryTransformer(nn.Module):
    """Pre-normalized causal Transformer with explicit all-padding handling."""

    def __init__(self, config: HistoryEncoderConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.d_model * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=RmsNorm(config.d_model),
            enable_nested_tensor=False,
        )

    def forward_sequence(self, events: Tensor, mask: Tensor) -> Tensor:
        """Return every causally encoded row for cache and causality tests."""
        effective_mask = mask.clone()
        empty = ~effective_mask.any(dim=1)
        effective_mask[empty, 0] = True
        safe_events = events.clone()
        safe_events[empty, 0] = 0.0
        length = events.shape[1]
        causal = torch.ones((length, length), dtype=torch.bool, device=events.device).triu(1)
        output = self.encoder(
            safe_events,
            mask=causal,
            src_key_padding_mask=~effective_mask,
        )
        return cast(Tensor, output) * effective_mask.unsqueeze(-1)

    def forward(self, events: Tensor, mask: Tensor) -> Tensor:
        return _last_valid(self.forward_sequence(events, mask), mask)


class ScalarEncoder(nn.Module):
    """Normalize the frozen E016 scalar vector into the shared model width."""

    def __init__(self, config: HistoryEncoderConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(SCALAR_COLUMNS), config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
            RmsNorm(config.d_model),
        )

    def forward(self, scalars: Tensor) -> Tensor:
        if scalars.ndim != 2 or scalars.shape[1] != len(SCALAR_COLUMNS):
            raise ValueError("scalar feature shape mismatch")
        if scalars.dtype != torch.float32 or not torch.isfinite(scalars).all():
            raise ValueError("scalars must be finite float32")
        scale = torch.tensor(
            [2, 1, 3, 20, 20, 20, 2, 15, 15, 96, 96, 256, 3, 20, 65535],
            dtype=scalars.dtype,
            device=scalars.device,
        )
        return cast(Tensor, self.network(scalars / scale))


class RoleHistoryGate(nn.Module):
    """Learn a feature-wise GRU/attention mixture with three seat-role biases."""

    def __init__(self, config: HistoryEncoderConfig) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.seat_bias = nn.Embedding(3, config.d_model)

    def forward(
        self,
        gru: Tensor,
        transformer: Tensor,
        scalar: Tensor,
        seat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if seat.dtype != torch.int64 or seat.ndim != 1 or seat.shape[0] != gru.shape[0]:
            raise ValueError("seat must be int64 [B]")
        if torch.any((seat < 0) | (seat > 2)):
            raise ValueError("seat values must be in 0..2")
        logits = cast(Tensor, self.gate(torch.cat((gru, transformer, scalar), dim=-1)))
        gate = torch.sigmoid(logits + self.seat_bias(seat))
        return gate * gru + (1.0 - gate) * transformer, gate


class RoleGatedHistoryEncoder(nn.Module):
    """Complete E018 encoder with independently switchable dual branches and gate."""

    def __init__(self, config: HistoryEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.event_encoder = HistoryEventEncoder(config)
        self.scalar_encoder = ScalarEncoder(config)
        self.gru = HistoryGruBranch(config) if config.gru_enabled else None
        self.transformer = CausalHistoryTransformer(config) if config.transformer_enabled else None
        self.role_gate = RoleHistoryGate(config) if config.role_gate_enabled else None

    def forward(
        self,
        rank_counts: Tensor,
        metadata: Tensor,
        mask: Tensor,
        scalars: Tensor,
        seat: Tensor,
    ) -> HistoryEncoding:
        events = self.event_encoder(rank_counts, metadata, mask)
        scalar = self.scalar_encoder(scalars)
        zeros = torch.zeros_like(scalar)
        gru = zeros if self.gru is None else self.gru(events, mask)
        transformer = zeros if self.transformer is None else self.transformer(events, mask)
        if self.gru is None:
            fused = transformer
            gate = torch.zeros_like(transformer)
        elif self.transformer is None:
            fused = gru
            gate = torch.ones_like(gru)
        elif self.role_gate is None:
            gate = torch.full_like(gru, 0.5)
            fused = 0.5 * (gru + transformer)
        else:
            fused, gate = self.role_gate(gru, transformer, scalar, seat)
        return HistoryEncoding(fused, gru, transformer, gate, scalar)


def _last_valid(sequence: Tensor, mask: Tensor) -> Tensor:
    lengths = mask.sum(dim=1)
    indices = torch.clamp(lengths - 1, min=0)
    batch = torch.arange(sequence.shape[0], device=sequence.device)
    selected = sequence[batch, indices]
    return selected * (lengths > 0).unsqueeze(-1)


def _validate_history_inputs(
    rank_counts: Tensor,
    metadata: Tensor,
    mask: Tensor,
    max_length: int,
) -> None:
    if rank_counts.dtype != torch.int64 or metadata.dtype != torch.int64:
        raise ValueError("history counts and metadata must use int64")
    if mask.dtype != torch.bool:
        raise ValueError("history mask must use bool")
    if rank_counts.ndim != 3 or rank_counts.shape[2] != HISTORY_RANK_COUNT:
        raise ValueError("history rank-count shape mismatch")
    batch_size, length, _ = rank_counts.shape
    if not 0 < length <= max_length:
        raise ValueError("history length exceeds configured maximum")
    if metadata.shape != (batch_size, length, len(HISTORY_META_COLUMNS)):
        raise ValueError("history metadata shape mismatch")
    if mask.shape != (batch_size, length):
        raise ValueError("history mask shape mismatch")
    valid_after_padding = mask.to(torch.int8).diff(dim=1) > 0
    if torch.any(valid_after_padding):
        raise ValueError("history mask must contain a valid prefix and padded suffix")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"model config {key} must be a finite number")
    return float(value)


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"model config {key} must be a boolean")
    return value


__all__ = (
    "HISTORY_ENCODER_SCHEMA_VERSION",
    "CausalHistoryTransformer",
    "HistoryEncoderConfig",
    "HistoryEncoding",
    "HistoryEventEncoder",
    "HistoryGruBranch",
    "RoleGatedHistoryEncoder",
    "RoleHistoryGate",
    "ScalarEncoder",
    "load_history_encoder_config",
)
