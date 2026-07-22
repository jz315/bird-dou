"""Standalone public-history tensor view backed by the canonical Ragged encoder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from birddou.env_types import Observation, RuleConfig
from birddou.features.ragged import (
    FEATURE_SCHEMA_VERSION,
    HISTORY_META_COLUMNS,
    FeatureConfig,
    _encode_history,
)


@dataclass(frozen=True, slots=True)
class PublicHistoryFeatures:
    """Padded public event counts, metadata, and validity mask."""

    schema_version: int
    rank_counts: Tensor
    metadata: Tensor
    mask: Tensor
    truncated_events: Tensor

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported public-history schema")
        if self.rank_counts.dtype != torch.int64 or self.rank_counts.ndim != 3:
            raise ValueError("history rank_counts must be int64 [B, H, 15]")
        batch_size, history_length, rank_width = self.rank_counts.shape
        if rank_width != 15:
            raise ValueError("history rank_counts final dimension must be 15")
        if self.metadata.dtype != torch.int64 or self.metadata.shape != (
            batch_size,
            history_length,
            len(HISTORY_META_COLUMNS),
        ):
            raise ValueError("history metadata shape/dtype mismatch")
        if self.mask.dtype != torch.bool or self.mask.shape != (batch_size, history_length):
            raise ValueError("history mask must be bool [B, H]")
        if self.truncated_events.dtype != torch.int64 or self.truncated_events.shape != (
            batch_size,
        ):
            raise ValueError("history truncated_events must be int64 [B]")
        if any(
            tensor.device != self.rank_counts.device
            for tensor in (self.metadata, self.mask, self.truncated_events)
        ):
            raise ValueError("history tensors must share one device")
        if torch.any(self.truncated_events < 0):
            raise ValueError("history truncated count cannot be negative")

    def to(self, device: str | torch.device) -> PublicHistoryFeatures:
        """Move the complete history view to one device."""
        return PublicHistoryFeatures(
            self.schema_version,
            self.rank_counts.to(device),
            self.metadata.to(device),
            self.mask.to(device),
            self.truncated_events.to(device),
        )


def encode_public_history(
    observations: Sequence[Observation],
    rules: RuleConfig,
    *,
    config: FeatureConfig | None = None,
) -> PublicHistoryFeatures:
    """Encode history independently of candidate-action enumeration."""
    if not observations:
        raise ValueError("history encoding requires at least one observation")
    settings = config if config is not None else FeatureConfig()
    encoded = [_encode_history(observation, settings, rules) for observation in observations]
    return PublicHistoryFeatures(
        schema_version=FEATURE_SCHEMA_VERSION,
        rank_counts=torch.tensor([item[0] for item in encoded], dtype=torch.int64),
        metadata=torch.tensor([item[1] for item in encoded], dtype=torch.int64),
        mask=torch.tensor([item[2] for item in encoded], dtype=torch.bool),
        truncated_events=torch.tensor([item[3] for item in encoded], dtype=torch.int64),
    )


__all__ = ("PublicHistoryFeatures", "encode_public_history")
