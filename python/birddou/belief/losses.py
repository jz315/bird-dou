"""Supervised losses and key-card calibration for constrained belief."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from birddou.belief.cardinality_crf import (
    log_partition,
    true_assignment_score,
    validate_assignment,
)

Reduction = Literal["none", "mean", "sum"]


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    empirical_frequency: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    brier_score: float
    expected_calibration_error: float
    bins: tuple[CalibrationBin, ...]


def belief_nll(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    true_assignment_a: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """Exact negative log-likelihood of a capacity-valid hidden hand."""
    validate_assignment(true_assignment_a, unknown_counts, capacity_a)
    losses = log_partition(scores, unknown_counts, capacity_a) - true_assignment_score(
        scores, true_assignment_a
    )
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError(f"unknown belief NLL reduction: {reduction}")


def uniform_belief_nll(
    unknown_counts: Tensor,
    capacity_a: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """NLL baseline that is uniform over every capacity-valid allocation."""
    scores = torch.zeros(
        (*unknown_counts.shape, 5),
        dtype=torch.float32,
        device=unknown_counts.device,
    )
    values = log_partition(scores, unknown_counts, capacity_a)
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"unknown belief NLL reduction: {reduction}")


def calibration_report(
    probability: Tensor,
    target: Tensor,
    bin_count: int = 10,
) -> CalibrationReport:
    """Return Brier score and equal-width reliability bins for binary key cards."""
    if probability.shape != target.shape or probability.ndim != 1:
        raise ValueError("calibration probability and target must be matching vectors")
    if not probability.is_floating_point() or not target.is_floating_point():
        raise ValueError("calibration inputs must use floating dtypes")
    if not torch.isfinite(probability).all() or not torch.isfinite(target).all():
        raise ValueError("calibration inputs contain NaN or infinity")
    if torch.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("calibration probabilities must be in 0..1")
    if torch.any((target < 0.0) | (target > 1.0)):
        raise ValueError("calibration targets must be in 0..1")
    if bin_count <= 0:
        raise ValueError("calibration bin_count must be positive")
    bins: list[CalibrationBin] = []
    total = max(1, probability.numel())
    error = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if index + 1 == bin_count:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        count = int(mask.sum().item())
        mean_probability = float(probability[mask].mean().item()) if count else 0.0
        frequency = float(target[mask].mean().item()) if count else 0.0
        error += count / total * abs(mean_probability - frequency)
        bins.append(CalibrationBin(lower, upper, count, mean_probability, frequency))
    brier = float((probability - target).square().mean().item())
    return CalibrationReport(brier, error, tuple(bins))


__all__ = (
    "CalibrationBin",
    "CalibrationReport",
    "belief_nll",
    "calibration_report",
    "uniform_belief_nll",
)
