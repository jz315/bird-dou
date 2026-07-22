"""Offline search-policy and compact-model distillation objectives."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor

from birddou.features.ragged import RaggedBatch
from birddou.models.deployment import CompactPolicyOutput
from birddou.models.segment_ops import segment_softmax, segment_sum

SEARCH_DISTILLATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SearchDistillationConfig:
    """Policy/value weights and temperature for both distillation stages."""

    schema_version: int = SEARCH_DISTILLATION_SCHEMA_VERSION
    policy_weight: float = 1.0
    value_weight: float = 1.0
    temperature: float = 1.0
    minimum_retained_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_DISTILLATION_SCHEMA_VERSION:
            raise ValueError("unsupported search distillation schema")
        if self.policy_weight < 0.0 or self.value_weight < 0.0:
            raise ValueError("search distillation weights must be non-negative")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("search distillation temperature must be finite and positive")
        if not 0.0 <= self.minimum_retained_fraction <= 1.0:
            raise ValueError("minimum retained search gain must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SearchDistillationBatch:
    """Public information set plus search visits/value and belief sample summary."""

    public_batch: RaggedBatch
    visit_probability: Tensor
    search_value: Tensor
    belief_sample_summary: Tensor

    def __post_init__(self) -> None:
        batch_size = self.public_batch.batch_size
        if not self.visit_probability.is_floating_point() or self.visit_probability.shape != (
            self.public_batch.action_count,
        ):
            raise ValueError("search visit probability must be floating [M]")
        if not self.search_value.is_floating_point() or self.search_value.shape != (batch_size,):
            raise ValueError("search value must be floating [B]")
        if (
            not self.belief_sample_summary.is_floating_point()
            or self.belief_sample_summary.ndim != 2
            or self.belief_sample_summary.shape[0] != batch_size
        ):
            raise ValueError("belief sample summary must be floating [B, S]")
        tensors = (self.visit_probability, self.search_value, self.belief_sample_summary)
        if any(value.device != self.public_batch.action_offsets.device for value in tensors):
            raise ValueError("search distillation tensors must share the public batch device")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("search distillation batch contains NaN or infinity")
        sums = segment_sum(self.visit_probability, self.public_batch.action_offsets)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1.0e-5, rtol=1.0e-5):
            raise ValueError("search visits must normalize inside every legal-action segment")
        if torch.any(self.visit_probability < 0.0):
            raise ValueError("search visit probabilities cannot be negative")


@dataclass(frozen=True, slots=True)
class SearchDistillationLoss:
    """Total and individual policy/value losses."""

    total: Tensor
    policy: Tensor
    value: Tensor


@dataclass(frozen=True, slots=True)
class DistillationRetentionReport:
    """Whether the no-search compact model retains the predeclared search gain."""

    accepted: bool
    search_gain: float
    compact_gain: float
    retained_fraction: float
    paired_delta_ci_lower: float
    reasons: tuple[str, ...]


def load_search_distillation_config(path: Path) -> SearchDistillationConfig:
    """Load versioned search/compact distillation controls from JSON-subset YAML."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ValueError("search distillation config must be a string-keyed mapping")
    values = cast(Mapping[str, object], raw)
    schema_version = _integer(values, "schema_version")
    return SearchDistillationConfig(
        schema_version=schema_version,
        policy_weight=_number(values, "policy_weight"),
        value_weight=_number(values, "value_weight"),
        temperature=_number(values, "temperature"),
        minimum_retained_fraction=_number(values, "minimum_retained_fraction"),
    )


def search_distillation_loss(
    policy_logits: Tensor,
    state_value: Tensor,
    batch: SearchDistillationBatch,
    config: SearchDistillationConfig | None = None,
) -> SearchDistillationLoss:
    """Distill root visit distributions and aggregated search values into a network."""
    settings = config if config is not None else SearchDistillationConfig()
    if (
        policy_logits.shape != batch.visit_probability.shape
        or not policy_logits.is_floating_point()
    ):
        raise ValueError("distilled policy logits must be floating [M]")
    if state_value.shape != batch.search_value.shape or not state_value.is_floating_point():
        raise ValueError("distilled state value must be floating [B]")
    probability = segment_softmax(
        policy_logits / settings.temperature,
        batch.public_batch.action_offsets,
    )
    log_probability = torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny))
    policy = -segment_sum(
        batch.visit_probability * log_probability,
        batch.public_batch.action_offsets,
    ).mean()
    value = functional.smooth_l1_loss(state_value, batch.search_value)
    total = settings.policy_weight * policy + settings.value_weight * value
    return SearchDistillationLoss(total, policy, value)


def compact_policy_distillation_loss(
    student: CompactPolicyOutput,
    teacher_policy_logits: Tensor,
    teacher_state_value: Tensor,
    batch: RaggedBatch,
    config: SearchDistillationConfig | None = None,
) -> SearchDistillationLoss:
    """Distill a large no-search/search-trained model into the compact deployable actor."""
    settings = config if config is not None else SearchDistillationConfig()
    if teacher_policy_logits.shape != student.policy_logits.shape:
        raise ValueError("compact and Teacher policy logits differ in action count")
    if teacher_state_value.shape != student.state_value.shape:
        raise ValueError("compact and Teacher state values differ in batch size")
    teacher_probability = segment_softmax(
        teacher_policy_logits.detach() / settings.temperature,
        batch.action_offsets,
    )
    targets = SearchDistillationBatch(
        public_batch=batch,
        visit_probability=teacher_probability,
        search_value=teacher_state_value.detach(),
        belief_sample_summary=torch.zeros(
            (batch.batch_size, 1),
            dtype=teacher_state_value.dtype,
            device=teacher_state_value.device,
        ),
    )
    return search_distillation_loss(
        student.policy_logits,
        student.state_value,
        targets,
        settings,
    )


def evaluate_distillation_retention(
    search_gain: float,
    compact_gain: float,
    paired_delta_ci_lower: float,
    *,
    minimum_retained_fraction: float = 0.8,
) -> DistillationRetentionReport:
    """Require a positive search gain and predeclared compact retention fraction."""
    values = (search_gain, compact_gain, paired_delta_ci_lower, minimum_retained_fraction)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("distillation retention inputs must be finite")
    if not 0.0 <= minimum_retained_fraction <= 1.0:
        raise ValueError("minimum retained fraction must be in [0, 1]")
    retained = 0.0 if search_gain <= 0.0 else compact_gain / search_gain
    reasons: list[str] = []
    if search_gain <= 0.0:
        reasons.append("search did not establish a positive paired gain")
    if retained < minimum_retained_fraction:
        reasons.append("compact model retained too little of the search gain")
    if paired_delta_ci_lower <= 0.0:
        reasons.append("compact paired lower confidence bound is not positive")
    return DistillationRetentionReport(
        not reasons,
        search_gain,
        compact_gain,
        retained,
        paired_delta_ci_lower,
        tuple(reasons),
    )


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"search distillation config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"search distillation config {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"search distillation config {key} must be finite")
    return number


__all__ = (
    "SEARCH_DISTILLATION_SCHEMA_VERSION",
    "DistillationRetentionReport",
    "SearchDistillationBatch",
    "SearchDistillationConfig",
    "SearchDistillationLoss",
    "compact_policy_distillation_loss",
    "evaluate_distillation_retention",
    "load_search_distillation_config",
    "search_distillation_loss",
)
