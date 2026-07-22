"""Public orchestration entry point for versioned information-set feature encoding."""

from __future__ import annotations

from collections.abc import Sequence

from birddou.env_types import Action, Observation, RuleConfig
from birddou.features.ragged import FeatureConfig, RaggedBatch, encode_ragged_batch


def encode_observations(
    observations: Sequence[Observation],
    legal_actions: Sequence[Sequence[Action]],
    rules: RuleConfig,
    *,
    config: FeatureConfig | None = None,
) -> RaggedBatch:
    """Encode public observations and their complete native legal-action segments."""
    return encode_ragged_batch(observations, legal_actions, rules, config=config)


__all__ = ("encode_observations",)
