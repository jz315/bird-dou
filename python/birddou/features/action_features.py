"""Standalone canonical candidate-action feature view for Proposal and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from birddou.env_types import Action, Observation, RuleConfig
from birddou.features.ragged import (
    ACTION_META_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    encode_ragged_batch,
)


@dataclass(frozen=True, slots=True)
class CandidateActionFeatures:
    """Flat action/post-hand structure and exact per-state segmentation."""

    schema_version: int
    rank_counts: Tensor
    post_hand_counts: Tensor
    metadata: Tensor
    state_index: Tensor
    offsets: Tensor

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported candidate-action schema")
        action_count = self.rank_counts.shape[0]
        if self.rank_counts.dtype != torch.int64 or self.rank_counts.shape != (action_count, 15):
            raise ValueError("action rank_counts must be int64 [M, 15]")
        if self.post_hand_counts.dtype != torch.int64 or self.post_hand_counts.shape != (
            action_count,
            15,
        ):
            raise ValueError("action post_hand_counts must be int64 [M, 15]")
        if self.metadata.dtype != torch.int64 or self.metadata.shape != (
            action_count,
            len(ACTION_META_COLUMNS),
        ):
            raise ValueError("action metadata shape/dtype mismatch")
        if self.offsets.dtype != torch.int64 or self.offsets.ndim != 1:
            raise ValueError("action offsets must be int64 [B+1]")
        if self.state_index.dtype != torch.int64 or self.state_index.shape != (action_count,):
            raise ValueError("action state_index must be int64 [M]")
        counts = torch.diff(self.offsets)
        expected_state_index = torch.repeat_interleave(
            torch.arange(counts.numel(), dtype=torch.int64, device=self.offsets.device),
            counts,
        )
        if not torch.equal(self.state_index, expected_state_index):
            raise ValueError("action state_index differs from offsets")
        if any(
            tensor.device != self.rank_counts.device
            for tensor in (self.post_hand_counts, self.metadata, self.state_index, self.offsets)
        ):
            raise ValueError("candidate-action tensors must share one device")

    @property
    def batch_size(self) -> int:
        return self.offsets.numel() - 1

    @property
    def action_count(self) -> int:
        return self.rank_counts.shape[0]

    def to(self, device: str | torch.device) -> CandidateActionFeatures:
        """Move the complete action view to one device."""
        return CandidateActionFeatures(
            self.schema_version,
            self.rank_counts.to(device),
            self.post_hand_counts.to(device),
            self.metadata.to(device),
            self.state_index.to(device),
            self.offsets.to(device),
        )


def encode_candidate_actions(
    observations: Sequence[Observation],
    legal_actions: Sequence[Sequence[Action]],
    rules: RuleConfig,
    *,
    config: FeatureConfig | None = None,
) -> CandidateActionFeatures:
    """Encode candidates with exactly the same implementation used by RaggedBatch."""
    batch = encode_ragged_batch(observations, legal_actions, rules, config=config)
    return CandidateActionFeatures(
        schema_version=batch.schema_version,
        rank_counts=batch.action_rank_counts,
        post_hand_counts=batch.post_hand_counts,
        metadata=batch.action_meta,
        state_index=batch.action_state_index,
        offsets=batch.action_offsets,
    )


__all__ = ("CandidateActionFeatures", "encode_candidate_actions")
