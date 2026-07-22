"""Ordered-rank token encoder and convolution/attention Rank Mixer."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from birddou.features.ragged import RANK_CATEGORICAL_COLUMNS, RANK_NUMERIC_COLUMNS

RANK_MIXER_SCHEMA_VERSION = 1
RANK_COUNT = 15


@dataclass(frozen=True, slots=True)
class RankMixerConfig:
    """Versioned shape and ablation switches for E017."""

    schema_version: int = RANK_MIXER_SCHEMA_VERSION
    d_model: int = 256
    blocks: int = 4
    attention_every: int = 2
    attention_heads: int = 8
    rank_embedding_dim: int = 32
    count_embedding_dim: int = 16
    flag_embedding_dim: int = 8
    swiglu_multiplier: int = 2
    dropout: float = 0.0
    drop_path: float = 0.0
    rms_norm_epsilon: float = 1e-6
    convolution_enabled: bool = True
    attention_enabled: bool = True
    numeric_enabled: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != RANK_MIXER_SCHEMA_VERSION:
            raise ValueError("unsupported Rank Mixer schema")
        positive = (
            self.d_model,
            self.blocks,
            self.rank_embedding_dim,
            self.count_embedding_dim,
            self.flag_embedding_dim,
            self.swiglu_multiplier,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Rank Mixer dimensions and block count must be positive")
        if self.attention_enabled:
            if self.attention_every <= 0 or self.attention_heads <= 0:
                raise ValueError("enabled rank attention needs positive cadence and heads")
            if self.d_model % self.attention_heads != 0:
                raise ValueError("d_model must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.drop_path < 1.0:
            raise ValueError("dropout and drop_path must be in [0, 1)")
        if self.rms_norm_epsilon <= 0.0:
            raise ValueError("rms_norm_epsilon must be positive")


def load_rank_mixer_config(path: Path) -> RankMixerConfig:
    """Load the rank section from the JSON-subset YAML model configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "model config")
    rank = _mapping(root.get("rank_mixer"), "rank_mixer")
    return RankMixerConfig(
        schema_version=_integer(root, "schema_version"),
        d_model=_integer(root, "d_model"),
        blocks=_integer(rank, "blocks"),
        attention_every=_integer(rank, "attention_every"),
        attention_heads=_integer(rank, "attention_heads"),
        rank_embedding_dim=_integer(rank, "rank_embedding_dim"),
        count_embedding_dim=_integer(rank, "count_embedding_dim"),
        flag_embedding_dim=_integer(rank, "flag_embedding_dim"),
        swiglu_multiplier=_integer(rank, "swiglu_multiplier"),
        dropout=_number(rank, "dropout"),
        drop_path=_number(rank, "drop_path"),
        rms_norm_epsilon=_number(rank, "rms_norm_epsilon"),
        convolution_enabled=_boolean(rank, "convolution_enabled"),
        attention_enabled=_boolean(rank, "attention_enabled"),
        numeric_enabled=_boolean(rank, "numeric_enabled"),
    )


class RmsNorm(nn.Module):
    """Parameter-efficient root-mean-square normalization."""

    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        if width <= 0 or epsilon <= 0.0:
            raise ValueError("RmsNorm width and epsilon must be positive")
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, inputs: Tensor) -> Tensor:
        variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance.to(inputs.dtype) + self.epsilon)
        return normalized * self.weight


class DropPath(nn.Module):
    """Per-state stochastic-depth residual drop with deterministic evaluation."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("DropPath probability must be in [0, 1)")
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = torch.empty(shape, dtype=inputs.dtype, device=inputs.device).bernoulli_(keep)
        return inputs * mask / keep


class RankTokenEncoder(nn.Module):
    """Embed the frozen nine categorical and three numeric per-rank fields."""

    def __init__(self, config: RankMixerConfig) -> None:
        super().__init__()
        self.config = config
        self.rank_embedding = nn.Embedding(RANK_COUNT, config.rank_embedding_dim)
        self.count_embeddings = nn.ModuleList(
            nn.Embedding(5, config.count_embedding_dim) for _ in range(7)
        )
        self.straight_embedding = nn.Embedding(2, config.flag_embedding_dim)
        categorical_width = (
            config.rank_embedding_dim + 7 * config.count_embedding_dim + config.flag_embedding_dim
        )
        numeric_width = len(RANK_NUMERIC_COLUMNS) if config.numeric_enabled else 0
        self.projection = nn.Linear(categorical_width + numeric_width, config.d_model)

    def forward(self, categorical: Tensor, numeric: Tensor) -> Tensor:
        _validate_rank_inputs(categorical, numeric)
        if torch.any((categorical[..., 0] < 0) | (categorical[..., 0] >= RANK_COUNT)):
            raise ValueError("rank_id values must be in 0..14")
        count_values = categorical[..., 1:8]
        if torch.any((count_values < 0) | (count_values > 4)):
            raise ValueError("rank count categories must be in 0..4")
        if torch.any((categorical[..., 8] < 0) | (categorical[..., 8] > 1)):
            raise ValueError("straight eligibility must be binary")
        pieces = [self.rank_embedding(categorical[..., 0])]
        pieces.extend(
            embedding(count_values[..., index])
            for index, embedding in enumerate(self.count_embeddings)
        )
        pieces.append(self.straight_embedding(categorical[..., 8]))
        if self.config.numeric_enabled:
            if not torch.isfinite(numeric).all():
                raise ValueError("rank numeric features contain NaN or infinity")
            pieces.append(numeric)
        return cast(Tensor, self.projection(torch.cat(pieces, dim=-1)))


class RankConvBlock(nn.Module):
    """RMSNorm → parallel depthwise 3/5-conv → pointwise SwiGLU residual."""

    def __init__(self, config: RankMixerConfig, drop_path: float) -> None:
        super().__init__()
        self.norm = RmsNorm(config.d_model, config.rms_norm_epsilon)
        self.depthwise3 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            padding=1,
            groups=config.d_model,
        )
        self.depthwise5 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=5,
            padding=2,
            groups=config.d_model,
        )
        hidden = config.d_model * config.swiglu_multiplier
        self.gate_value = nn.Linear(config.d_model, hidden * 2)
        self.output = nn.Linear(hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.norm(inputs).transpose(1, 2)
        mixed = (self.depthwise3(normalized) + self.depthwise5(normalized)).transpose(1, 2)
        gate, value = self.gate_value(mixed).chunk(2, dim=-1)
        update = self.output(torch.nn.functional.silu(gate) * value)
        return inputs + cast(Tensor, self.drop_path(self.dropout(update)))


class RelativeRankAttention(nn.Module):
    """Multi-head rank attention with learned query-key distance bias."""

    def __init__(self, config: RankMixerConfig, drop_path: float) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_width = config.d_model // config.attention_heads
        self.scale = self.head_width**-0.5
        self.norm = RmsNorm(config.d_model, config.rms_norm_epsilon)
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.relative_bias = nn.Parameter(torch.zeros(config.attention_heads, 2 * RANK_COUNT - 1))
        self.output = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path)
        positions = torch.arange(RANK_COUNT)
        relative_index = positions[:, None] - positions[None, :] + RANK_COUNT - 1
        self.relative_index: Tensor
        self.register_buffer("relative_index", relative_index, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, rank_count, width = inputs.shape
        normalized = self.norm(inputs)
        qkv = self.qkv(normalized).reshape(
            batch_size,
            rank_count,
            3,
            self.heads,
            self.head_width,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        bias = self.relative_bias[:, self.relative_index]
        probabilities = self.attention_dropout(torch.softmax(scores + bias[None, ...], dim=-1))
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).reshape(batch_size, rank_count, width)
        update = self.output_dropout(self.output(attended))
        return inputs + cast(Tensor, self.drop_path(update))


class RankMixer(nn.Module):
    """Ordered 15-rank mixer with independently switchable local/global paths."""

    def __init__(self, config: RankMixerConfig) -> None:
        super().__init__()
        self.config = config
        probabilities = torch.linspace(0.0, config.drop_path, config.blocks).tolist()
        self.conv_blocks = nn.ModuleList(
            RankConvBlock(config, probabilities[index])
            if config.convolution_enabled
            else nn.Identity()
            for index in range(config.blocks)
        )
        attentions: list[nn.Module] = []
        for index in range(config.blocks):
            enabled = config.attention_enabled and (index + 1) % config.attention_every == 0
            attentions.append(
                RelativeRankAttention(config, probabilities[index]) if enabled else nn.Identity()
            )
        self.attention_blocks = nn.ModuleList(attentions)

    @property
    def attention_block_count(self) -> int:
        return sum(isinstance(module, RelativeRankAttention) for module in self.attention_blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (RANK_COUNT, self.config.d_model):
            raise ValueError(
                f"RankMixer expects [B, 15, {self.config.d_model}], got {tuple(inputs.shape)}"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("RankMixer inputs contain NaN or infinity")
        output = inputs
        for convolution, attention in zip(
            self.conv_blocks,
            self.attention_blocks,
            strict=True,
        ):
            output = attention(convolution(output))
        return output


def _validate_rank_inputs(categorical: Tensor, numeric: Tensor) -> None:
    if categorical.dtype != torch.int64:
        raise ValueError("rank categorical features must use int64")
    if numeric.dtype != torch.float32:
        raise ValueError("rank numeric features must use float32")
    batch_size = categorical.shape[0]
    if categorical.shape != (batch_size, RANK_COUNT, len(RANK_CATEGORICAL_COLUMNS)):
        raise ValueError("rank categorical feature shape mismatch")
    if numeric.shape != (batch_size, RANK_COUNT, len(RANK_NUMERIC_COLUMNS)):
        raise ValueError("rank numeric feature shape mismatch")


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
    "RANK_COUNT",
    "RANK_MIXER_SCHEMA_VERSION",
    "DropPath",
    "RankConvBlock",
    "RankMixer",
    "RankMixerConfig",
    "RankTokenEncoder",
    "RelativeRankAttention",
    "RmsNorm",
    "load_rank_mixer_config",
)
