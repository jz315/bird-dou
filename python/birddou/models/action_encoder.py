"""Ragged legal-action encoder with rank cross-attention and set context."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from birddou.features.ragged import ACTION_META_COLUMNS, RaggedBatch
from birddou.models.rank_mixer import RankConvBlock, RankMixerConfig, RmsNorm
from birddou.models.segment_ops import segment_max, segment_mean, segment_state_index

ACTION_ENCODER_SCHEMA_VERSION = 1
ACTION_RANK_COUNT = 15


@dataclass(frozen=True, slots=True)
class ActionEncoderConfig:
    """Versioned dimensions and independent E019 ablation switches."""

    schema_version: int = ACTION_ENCODER_SCHEMA_VERSION
    d_model: int = 256
    rank_blocks: int = 2
    attention_heads: int = 8
    count_embedding_dim: int = 16
    meta_embedding_dim: int = 16
    swiglu_multiplier: int = 2
    dropout: float = 0.0
    decomposition_count_cap: int = 255
    post_hand_enabled: bool = True
    cross_attention_enabled: bool = True
    set_context_enabled: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_ENCODER_SCHEMA_VERSION:
            raise ValueError("unsupported action encoder schema")
        if (
            min(
                self.d_model,
                self.attention_heads,
                self.count_embedding_dim,
                self.meta_embedding_dim,
                self.swiglu_multiplier,
                self.decomposition_count_cap,
            )
            <= 0
        ):
            raise ValueError("action encoder dimensions and cap must be positive")
        if self.rank_blocks < 0:
            raise ValueError("action rank_blocks must be non-negative")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("d_model must be divisible by action attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("action dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ActionEncoding:
    """Final and intermediate per-action states for diagnostics and heads."""

    action: Tensor
    base_action: Tensor
    query: Tensor
    rank_context: Tensor
    attention_weights: Tensor
    set_mean: Tensor
    set_max: Tensor


def load_action_encoder_config(path: Path) -> ActionEncoderConfig:
    """Load the action section from the JSON-subset YAML model configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "model config")
    action = _mapping(root.get("action"), "action")
    return ActionEncoderConfig(
        schema_version=_integer(root, "schema_version"),
        d_model=_integer(root, "d_model"),
        rank_blocks=_integer(action, "blocks"),
        attention_heads=_integer(action, "attention_heads"),
        count_embedding_dim=_integer(action, "count_embedding_dim"),
        meta_embedding_dim=_integer(action, "meta_embedding_dim"),
        swiglu_multiplier=_integer(action, "swiglu_multiplier"),
        dropout=_number(action, "dropout"),
        decomposition_count_cap=_integer(action, "decomposition_count_cap"),
        post_hand_enabled=_boolean(action, "post_hand_enabled"),
        cross_attention_enabled=_boolean(action, "cross_attention_enabled"),
        set_context_enabled=_boolean(action, "set_context_enabled"),
    )


class SwiGluMlp(nn.Module):
    """Two-projection SwiGLU used for query and fusion layers."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gate_value = nn.Linear(input_width, hidden_width * 2)
        self.output = nn.Linear(hidden_width, output_width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, value = self.gate_value(inputs).chunk(2, dim=-1)
        result = self.output(torch.nn.functional.silu(gate) * value)
        return cast(Tensor, self.dropout(result))


class ActionRankCountEncoder(nn.Module):
    """Encode one 15-rank count vector with shared ordered local mixing."""

    def __init__(self, config: ActionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.rank_embedding = nn.Embedding(ACTION_RANK_COUNT, config.count_embedding_dim)
        self.count_embedding = nn.Embedding(5, config.count_embedding_dim)
        self.input_projection = nn.Linear(config.count_embedding_dim * 2, config.d_model)
        mixer_config = RankMixerConfig(
            d_model=config.d_model,
            blocks=max(1, config.rank_blocks),
            attention_every=1,
            attention_heads=config.attention_heads,
            rank_embedding_dim=1,
            count_embedding_dim=1,
            flag_embedding_dim=1,
            swiglu_multiplier=config.swiglu_multiplier,
            dropout=config.dropout,
            drop_path=0.0,
            convolution_enabled=True,
            attention_enabled=False,
            numeric_enabled=False,
        )
        self.blocks = nn.ModuleList(
            RankConvBlock(mixer_config, 0.0) for _ in range(config.rank_blocks)
        )
        self.pool = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            RmsNorm(config.d_model),
        )

    def forward(self, counts: Tensor) -> Tensor:
        if counts.dtype != torch.int64 or counts.ndim != 2 or counts.shape[1] != 15:
            raise ValueError("action rank counts must be int64 [M, 15]")
        if torch.any((counts < 0) | (counts > 4)):
            raise ValueError("action rank counts must be in 0..4")
        ranks = torch.arange(ACTION_RANK_COUNT, dtype=torch.int64, device=counts.device)
        tokens = torch.cat(
            (
                self.rank_embedding(ranks)[None].expand(counts.shape[0], -1, -1),
                self.count_embedding(counts),
            ),
            dim=-1,
        )
        tokens = cast(Tensor, self.input_projection(tokens))
        for block in self.blocks:
            tokens = cast(Tensor, block(tokens))
        pooled = torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)
        return cast(Tensor, self.pool(pooled))


class ActionMetaEncoder(nn.Module):
    """Embed all fourteen frozen E016 action metadata columns."""

    def __init__(self, config: ActionEncoderConfig) -> None:
        super().__init__()
        self.cardinalities = (
            15,
            16,
            13,
            3,
            21,
            2,
            2,
            2,
            2,
            2,
            16,
            16,
            22,
            config.decomposition_count_cap + 2,
        )
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.meta_embedding_dim)
            for cardinality in self.cardinalities
        )
        self.projection = nn.Sequential(
            nn.Linear(len(ACTION_META_COLUMNS) * config.meta_embedding_dim, config.d_model),
            nn.SiLU(),
            RmsNorm(config.d_model),
        )

    def forward(self, metadata: Tensor) -> Tensor:
        if metadata.dtype != torch.int64 or metadata.ndim != 2:
            raise ValueError("action metadata must use int64 [M, Ca]")
        if metadata.shape[1] != len(ACTION_META_COLUMNS):
            raise ValueError("action metadata width differs from schema")
        pieces: list[Tensor] = []
        for column, (embedding, cardinality) in enumerate(
            zip(self.embeddings, self.cardinalities, strict=True)
        ):
            values = metadata[:, column]
            if torch.any((values < 0) | (values >= cardinality)):
                raise ValueError(
                    f"action {ACTION_META_COLUMNS[column]} values exceed embedding range"
                )
            pieces.append(cast(Tensor, embedding(values)))
        return cast(Tensor, self.projection(torch.cat(pieces, dim=-1)))


class ActionRankCrossAttention(nn.Module):
    """One query per action attending only its state's 15 rank tokens."""

    def __init__(self, config: ActionEncoderConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.heads = config.attention_heads
        self.head_width = config.d_model // config.attention_heads
        self.scale = self.head_width**-0.5
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: Tensor,
        rank_tokens: Tensor,
        action_state_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if query.ndim != 2 or query.shape[1] != self.d_model:
            raise ValueError("action attention query must be [M, d_model]")
        if rank_tokens.ndim != 3 or rank_tokens.shape[1:] != (ACTION_RANK_COUNT, self.d_model):
            raise ValueError("action attention rank tokens must be [B, 15, d_model]")
        if action_state_index.dtype != torch.int64 or action_state_index.shape != (query.shape[0],):
            raise ValueError("action attention state index must be int64 [M]")
        if query.device != rank_tokens.device or query.device != action_state_index.device:
            raise ValueError("action attention inputs must share one device")
        if torch.any((action_state_index < 0) | (action_state_index >= rank_tokens.shape[0])):
            raise ValueError("action attention state index is out of range")
        action_count, width = query.shape
        selected = rank_tokens[action_state_index]
        query_heads = self.query(query).reshape(action_count, self.heads, self.head_width)
        key_heads = (
            self.key(selected)
            .reshape(action_count, ACTION_RANK_COUNT, self.heads, self.head_width)
            .transpose(1, 2)
        )
        value_heads = (
            self.value(selected)
            .reshape(action_count, ACTION_RANK_COUNT, self.heads, self.head_width)
            .transpose(1, 2)
        )
        scores = torch.einsum("mhd,mhrd->mhr", query_heads, key_heads) * self.scale
        weights = torch.softmax(scores, dim=-1)
        context_weights = self.dropout(weights)
        context = torch.einsum("mhr,mhrd->mhd", context_weights, value_heads).reshape(
            action_count, width
        )
        return cast(Tensor, self.output(context)), weights


class RaggedActionEncoder(nn.Module):
    """Full action-query, rank-context, and legal-set-context encoder."""

    def __init__(self, config: ActionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.rank_counts = ActionRankCountEncoder(config)
        self.metadata = ActionMetaEncoder(config)
        hidden = config.d_model * config.swiglu_multiplier
        self.query = SwiGluMlp(config.d_model * 4, hidden, config.d_model, config.dropout)
        self.cross_attention = (
            ActionRankCrossAttention(config) if config.cross_attention_enabled else None
        )
        self.base_fusion = SwiGluMlp(
            config.d_model * 2,
            hidden,
            config.d_model,
            config.dropout,
        )
        self.set_fusion = SwiGluMlp(
            config.d_model * 3,
            hidden,
            config.d_model,
            config.dropout,
        )

    def forward(
        self,
        batch: RaggedBatch,
        state: Tensor,
        rank_tokens: Tensor,
    ) -> ActionEncoding:
        _validate_action_context(batch, state, rank_tokens, self.config.d_model)
        expected_state_index = segment_state_index(batch.action_offsets)
        if not torch.equal(expected_state_index, batch.action_state_index):
            raise ValueError("action_state_index differs from offsets")
        action_rank = self.rank_counts(batch.action_rank_counts)
        if self.config.post_hand_enabled:
            post_hand = self.rank_counts(batch.post_hand_counts)
        else:
            post_hand = torch.zeros_like(action_rank)
        metadata = self.metadata(batch.action_meta)
        state_for_action = state[batch.action_state_index]
        query = self.query(torch.cat((action_rank, post_hand, metadata, state_for_action), dim=-1))
        if self.cross_attention is None:
            context = torch.zeros_like(query)
            weights = torch.zeros(
                (batch.action_count, self.config.attention_heads, ACTION_RANK_COUNT),
                dtype=query.dtype,
                device=query.device,
            )
        else:
            context, weights = self.cross_attention(
                query,
                rank_tokens,
                batch.action_state_index,
            )
        base = self.base_fusion(torch.cat((query, context), dim=-1))
        mean = segment_mean(base, batch.action_offsets)
        maximum = segment_max(base, batch.action_offsets)
        if self.config.set_context_enabled:
            action = self.set_fusion(
                torch.cat(
                    (
                        base,
                        mean[batch.action_state_index],
                        maximum[batch.action_state_index],
                    ),
                    dim=-1,
                )
            )
        else:
            action = base
        return ActionEncoding(action, base, query, context, weights, mean, maximum)


def _validate_action_context(
    batch: RaggedBatch,
    state: Tensor,
    rank_tokens: Tensor,
    d_model: int,
) -> None:
    if state.shape != (batch.batch_size, d_model):
        raise ValueError("state shape must be [B, d_model]")
    if rank_tokens.shape != (batch.batch_size, ACTION_RANK_COUNT, d_model):
        raise ValueError("rank token shape must be [B, 15, d_model]")
    if state.dtype != torch.float32 or rank_tokens.dtype != torch.float32:
        raise ValueError("state and rank tokens must use float32")
    if not torch.isfinite(state).all() or not torch.isfinite(rank_tokens).all():
        raise ValueError("action context contains NaN or infinity")
    devices = {
        state.device,
        rank_tokens.device,
        batch.action_rank_counts.device,
        batch.post_hand_counts.device,
        batch.action_meta.device,
        batch.action_state_index.device,
        batch.action_offsets.device,
    }
    if len(devices) != 1:
        raise ValueError("action batch and context must share one device")


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
    "ACTION_ENCODER_SCHEMA_VERSION",
    "ActionEncoderConfig",
    "ActionEncoding",
    "ActionMetaEncoder",
    "ActionRankCountEncoder",
    "ActionRankCrossAttention",
    "RaggedActionEncoder",
    "SwiGluMlp",
    "load_action_encoder_config",
)
