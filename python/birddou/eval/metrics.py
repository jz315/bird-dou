"""Deal-clustered, role-separated Arena metrics and confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np

from birddou.eval.bootstrap import (
    BootstrapCI,
    BootstrapConfig,
    bootstrap_mean_ci,
    bootstrap_paired_difference_ci,
)
from birddou.eval.paired_deals import SEAT_ROLES, PairedDealSet, SeatRole

if TYPE_CHECKING:
    from birddou.eval.arena import MatchResult, PairedMatchResult

METRICS_SCHEMA_VERSION = 2
RoleMetricName: TypeAlias = str


@dataclass(frozen=True, slots=True)
class PairedEstimate:
    """Candidate/baseline means and CI for their within-deal difference."""

    sample_count: int
    candidate_mean: float
    baseline_mean: float
    mean_delta: float
    delta_ci: BootstrapCI

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable estimate."""
        return {
            "sample_count": self.sample_count,
            "candidate_mean": self.candidate_mean,
            "baseline_mean": self.baseline_mean,
            "mean_delta": self.mean_delta,
            "delta_ci": self.delta_ci.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RoleReport:
    """Paired estimates for one role or a within-deal role aggregate."""

    role: str
    deal_count: int
    win_rate: PairedEstimate
    raw_payoff: PairedEstimate
    objective_payoff: PairedEstimate

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable role report."""
        return {
            "role": self.role,
            "deal_count": self.deal_count,
            "win_rate": self.win_rate.to_dict(),
            "raw_payoff": self.raw_payoff.to_dict(),
            "objective_payoff": self.objective_payoff.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ArenaReport:
    """Formal paired comparison with every uncertainty unit kept at deal level."""

    schema_version: int
    rules_hash: str
    candidate_policy_id: str
    baseline_policy_id: str
    deal_set: PairedDealSet
    pair_count: int
    match_count: int
    landlord: RoleReport
    landlord_down: RoleReport
    landlord_up: RoleReport
    farmer_team: RoleReport
    overall: RoleReport

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable evaluation artifact."""
        return {
            "schema_version": self.schema_version,
            "rules_hash": self.rules_hash,
            "candidate_policy_id": self.candidate_policy_id,
            "baseline_policy_id": self.baseline_policy_id,
            "deal_set": self.deal_set.to_dict(),
            "pair_count": self.pair_count,
            "match_count": self.match_count,
            "roles": {
                "landlord": self.landlord.to_dict(),
                "landlord_down": self.landlord_down.to_dict(),
                "landlord_up": self.landlord_up.to_dict(),
                "farmer_team": self.farmer_team.to_dict(),
                "overall": self.overall.to_dict(),
            },
        }

    def meets_precision(self, maximum_half_width: float, metric: str = "win_rate") -> bool:
        """Check the overall paired CI against a predeclared stopping precision."""
        if maximum_half_width <= 0.0:
            raise ValueError("maximum_half_width must be positive")
        if metric not in {"win_rate", "raw_payoff", "objective_payoff"}:
            raise ValueError(f"unknown precision metric: {metric}")
        estimate = cast(PairedEstimate, getattr(self.overall, metric))
        return estimate.delta_ci.half_width <= maximum_half_width


@dataclass(frozen=True, slots=True)
class MeanEstimate:
    """One unpaired cross-play mean and bootstrap CI."""

    mean: float
    standard_deviation: float
    ci: BootstrapCI

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mean estimate."""
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "ci": self.ci.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CrossPlayCell:
    """One landlord-policy versus farmer-team-policy matrix cell."""

    landlord_policy_id: str
    farmer_policy_id: str
    deal_count: int
    landlord_win_rate: MeanEstimate
    landlord_raw_payoff: MeanEstimate
    landlord_objective_payoff: MeanEstimate
    mean_bomb_count: MeanEstimate

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable matrix cell."""
        result = cast(dict[str, object], asdict(self))
        result["landlord_win_rate"] = self.landlord_win_rate.to_dict()
        result["landlord_raw_payoff"] = self.landlord_raw_payoff.to_dict()
        result["landlord_objective_payoff"] = self.landlord_objective_payoff.to_dict()
        result["mean_bomb_count"] = self.mean_bomb_count.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class CrossPlayReport:
    """Complete ordered cross-play matrix over one fixed deal set."""

    schema_version: int
    rules_hash: str
    deal_set: PairedDealSet
    cells: tuple[CrossPlayCell, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable matrix artifact."""
        return {
            "schema_version": self.schema_version,
            "rules_hash": self.rules_hash,
            "deal_set": self.deal_set.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Finite-sample distribution and ordered cumulative downside risk."""

    sample_count: int
    mean: float
    standard_deviation: float
    p10: float
    p50: float
    p90: float
    maximum_drawdown: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable distribution summary."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class RoleWinRate:
    """Win rate for one exact controlled game seat role."""

    sample_count: int
    win_rate: float

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError("role sample_count must be non-negative")
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError("role win_rate must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable role result."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class GamePerformanceReport:
    """Per-policy game quality, score tails, bomb use, and spring outcomes."""

    schema_version: int
    policy_id: str
    match_count: int
    controlled_seat_count: int
    role_win_rates: Mapping[RoleMetricName, RoleWinRate]
    raw_score: DistributionSummary
    training_score: DistributionSummary
    bomb_plays: DistributionSummary
    landlord_spring_opportunities: int
    landlord_spring_rate: float
    anti_spring_opportunities: int
    anti_spring_rate: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable game report."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "match_count": self.match_count,
            "controlled_seat_count": self.controlled_seat_count,
            "role_win_rates": {
                role: estimate.to_dict() for role, estimate in self.role_win_rates.items()
            },
            "raw_score": self.raw_score.to_dict(),
            "training_score": self.training_score.to_dict(),
            "bomb_plays": self.bomb_plays.to_dict(),
            "landlord_spring_opportunities": self.landlord_spring_opportunities,
            "landlord_spring_rate": self.landlord_spring_rate,
            "anti_spring_opportunities": self.anti_spring_opportunities,
            "anti_spring_rate": self.anti_spring_rate,
        }


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Probability calibration measured by Brier score and equal-width ECE."""

    sample_count: int
    brier_score: float
    expected_calibration_error: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable calibration result."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class QuantileCoverage:
    """Empirical coverage of declared score quantiles."""

    quantile_levels: tuple[float, ...]
    empirical_coverage: tuple[float, ...]
    mean_absolute_calibration_error: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable coverage result."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class ModelQualityReport:
    """Policy, outcome, score-distribution, and hidden-card quality metrics."""

    schema_version: int
    sample_count: int
    mean_policy_entropy: float
    win: CalibrationSummary
    score_mae: float
    score_quantile_coverage: QuantileCoverage
    belief_nll: float
    belief_count_mae: float
    key_card: CalibrationSummary

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable model report."""
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "mean_policy_entropy": self.mean_policy_entropy,
            "win": self.win.to_dict(),
            "score_mae": self.score_mae,
            "score_quantile_coverage": self.score_quantile_coverage.to_dict(),
            "belief_nll": self.belief_nll,
            "belief_count_mae": self.belief_count_mae,
            "key_card": self.key_card.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Latency distribution in milliseconds."""

    sample_count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable latency result."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class EngineeringPerformanceReport:
    """Environment, inference, queue, learner, lag, and memory telemetry."""

    schema_version: int
    elapsed_seconds: float
    environment_steps_per_second: float
    legal_actions_per_second: float
    gpu_actions_scored_per_second: float
    inference_latency: LatencySummary
    actor_queue_wait: LatencySummary
    learner_gpu_utilization_mean: float
    learner_gpu_utilization_p95: float
    policy_version_lag: DistributionSummary
    actor_peak_memory_mb: float
    learner_peak_memory_mb: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable engineering report."""
        return {
            "schema_version": self.schema_version,
            "elapsed_seconds": self.elapsed_seconds,
            "environment_steps_per_second": self.environment_steps_per_second,
            "legal_actions_per_second": self.legal_actions_per_second,
            "gpu_actions_scored_per_second": self.gpu_actions_scored_per_second,
            "inference_latency": self.inference_latency.to_dict(),
            "actor_queue_wait": self.actor_queue_wait.to_dict(),
            "learner_gpu_utilization_mean": self.learner_gpu_utilization_mean,
            "learner_gpu_utilization_p95": self.learner_gpu_utilization_p95,
            "policy_version_lag": self.policy_version_lag.to_dict(),
            "actor_peak_memory_mb": self.actor_peak_memory_mb,
            "learner_peak_memory_mb": self.learner_peak_memory_mb,
        }


def summarize_game_performance(
    results: Sequence[MatchResult],
    policy_id: str,
) -> GamePerformanceReport:
    """Summarize one policy over every seat it controlled in chronological order."""
    if not policy_id:
        raise ValueError("policy_id must be non-empty")
    if not results:
        raise ValueError("game performance requires at least one match")
    role_scores: dict[str, list[float]] = {
        "landlord": [],
        "landlord_down": [],
        "landlord_up": [],
        "farmer_team": [],
        "overall": [],
    }
    raw_scores: list[float] = []
    training_scores: list[float] = []
    bomb_plays: list[float] = []
    landlord_spring_events = 0
    landlord_spring_opportunities = 0
    anti_spring_events = 0
    anti_spring_opportunities = 0

    for result in results:
        controlled = tuple(
            seat for seat in range(3) if result.assignment.policy_for_seat(seat) == policy_id
        )
        if not controlled:
            continue
        landlord = result.landlord_seat
        down = (landlord + 1) % 3
        up = (landlord + 2) % 3
        controlled_landlord = landlord in controlled
        controlled_farmer = down in controlled or up in controlled
        if controlled_landlord:
            landlord_spring_opportunities += 1
            landlord_spring_events += int(result.spring_outcome == "landlord_spring")
        if controlled_farmer:
            anti_spring_opportunities += 1
            anti_spring_events += int(result.spring_outcome == "anti_spring")
        for seat in controlled:
            raw = float(result.raw_payoff[seat])
            won = float(raw > 0.0)
            role = (
                "landlord"
                if seat == landlord
                else "landlord_down"
                if seat == down
                else "landlord_up"
            )
            role_scores[role].append(won)
            if role != "landlord":
                role_scores["farmer_team"].append(won)
            role_scores["overall"].append(won)
            raw_scores.append(raw)
            training_scores.append(math.copysign(math.log2(1.0 + abs(raw)), raw))
            bomb_plays.append(float(result.bomb_count_by_seat[seat]))

    if not raw_scores:
        raise ValueError(f"policy {policy_id!r} does not control a seat in the supplied matches")
    return GamePerformanceReport(
        schema_version=METRICS_SCHEMA_VERSION,
        policy_id=policy_id,
        match_count=len(results),
        controlled_seat_count=len(raw_scores),
        role_win_rates={
            role: RoleWinRate(
                sample_count=len(values),
                win_rate=float(np.mean(values)) if values else 0.0,
            )
            for role, values in role_scores.items()
        },
        raw_score=_distribution_summary(raw_scores),
        training_score=_distribution_summary(training_scores),
        bomb_plays=_distribution_summary(bomb_plays),
        landlord_spring_opportunities=landlord_spring_opportunities,
        landlord_spring_rate=_safe_rate(
            landlord_spring_events,
            landlord_spring_opportunities,
        ),
        anti_spring_opportunities=anti_spring_opportunities,
        anti_spring_rate=_safe_rate(anti_spring_events, anti_spring_opportunities),
    )


def summarize_model_quality(
    *,
    policy_probabilities: Sequence[float],
    action_offsets: Sequence[int],
    win_probabilities: Sequence[float],
    win_targets: Sequence[float],
    score_predictions: Sequence[float],
    score_targets: Sequence[float],
    score_quantile_predictions: Sequence[Sequence[float]],
    score_quantile_levels: Sequence[float],
    belief_probabilities: Sequence[Sequence[float]],
    belief_targets: Sequence[Sequence[float]],
    belief_count_predictions: Sequence[Sequence[float]],
    belief_count_targets: Sequence[Sequence[float]],
    key_card_probabilities: Sequence[float],
    key_card_targets: Sequence[float],
    calibration_bins: int = 10,
) -> ModelQualityReport:
    """Compute the complete model-quality protocol from held-out predictions."""
    probabilities = _finite_vector(policy_probabilities, "policy_probabilities")
    offsets = np.asarray(action_offsets, dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
        raise ValueError("action_offsets must be a one-dimensional prefix sum starting at zero")
    if offsets[-1] != len(probabilities) or np.any(np.diff(offsets) <= 0):
        raise ValueError("action_offsets must be strictly increasing and end at action count")
    entropies: list[float] = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        segment = probabilities[int(start) : int(stop)]
        if np.any(segment < 0.0) or not np.isclose(segment.sum(), 1.0, atol=1e-6):
            raise ValueError("each ragged policy segment must be a probability distribution")
        positive = segment[segment > 0.0]
        entropies.append(float(-(positive * np.log(positive)).sum()))

    win = _calibration_summary(win_probabilities, win_targets, calibration_bins, "win")
    scores = _finite_vector(score_predictions, "score_predictions")
    score_truth = _finite_vector(score_targets, "score_targets")
    if scores.shape != score_truth.shape or len(scores) != len(entropies):
        raise ValueError("score and ragged policy sample counts must match")
    quantile_levels = _probability_vector(score_quantile_levels, "score_quantile_levels")
    if np.any(np.diff(quantile_levels) <= 0.0):
        raise ValueError("score_quantile_levels must be strictly increasing")
    quantiles = _finite_matrix(score_quantile_predictions, "score_quantile_predictions")
    if quantiles.shape != (len(scores), len(quantile_levels)):
        raise ValueError("score quantile prediction shape must be [sample, quantile]")
    if np.any(np.diff(quantiles, axis=1) < 0.0):
        raise ValueError("score quantile predictions must be non-decreasing per sample")
    empirical = np.mean(score_truth[:, None] <= quantiles, axis=0)

    belief = _probability_matrix(belief_probabilities, "belief_probabilities")
    belief_truth = _binary_matrix(belief_targets, "belief_targets")
    if belief.shape != belief_truth.shape or belief.shape[0] != len(scores):
        raise ValueError("belief probability and target shapes/sample counts must match")
    clipped = np.clip(belief, 1e-12, 1.0 - 1e-12)
    belief_nll = float(
        np.mean(-(belief_truth * np.log(clipped) + (1.0 - belief_truth) * np.log1p(-clipped)))
    )
    belief_counts = _finite_matrix(belief_count_predictions, "belief_count_predictions")
    belief_count_truth = _finite_matrix(belief_count_targets, "belief_count_targets")
    if belief_counts.shape != belief_count_truth.shape or belief_counts.shape[0] != len(scores):
        raise ValueError("belief count prediction and target shapes/sample counts must match")
    if np.any(belief_counts < 0.0) or np.any(belief_count_truth < 0.0):
        raise ValueError("belief counts must be non-negative")
    key_card = _calibration_summary(
        key_card_probabilities,
        key_card_targets,
        calibration_bins,
        "key_card",
    )
    if win.sample_count != len(scores) or key_card.sample_count != len(scores):
        raise ValueError("all model-quality inputs must share the policy sample count")

    return ModelQualityReport(
        schema_version=METRICS_SCHEMA_VERSION,
        sample_count=len(scores),
        mean_policy_entropy=float(np.mean(entropies)),
        win=win,
        score_mae=float(np.mean(np.abs(scores - score_truth))),
        score_quantile_coverage=QuantileCoverage(
            quantile_levels=tuple(float(item) for item in quantile_levels),
            empirical_coverage=tuple(float(item) for item in empirical),
            mean_absolute_calibration_error=float(np.mean(np.abs(empirical - quantile_levels))),
        ),
        belief_nll=belief_nll,
        belief_count_mae=float(np.mean(np.abs(belief_counts - belief_count_truth))),
        key_card=key_card,
    )


def summarize_engineering_performance(
    *,
    elapsed_seconds: float,
    environment_steps: int,
    legal_actions_generated: int,
    gpu_actions_scored: int,
    inference_latency_ms: Sequence[float],
    actor_queue_wait_ms: Sequence[float],
    learner_gpu_utilization: Sequence[float],
    policy_version_lag: Sequence[float],
    actor_peak_memory_mb: float,
    learner_peak_memory_mb: float,
) -> EngineeringPerformanceReport:
    """Summarize bounded operational counters without retaining request histories."""
    elapsed = _finite_nonnegative(elapsed_seconds, "elapsed_seconds")
    if elapsed <= 0.0:
        raise ValueError("elapsed_seconds must be positive")
    for name, count in (
        ("environment_steps", environment_steps),
        ("legal_actions_generated", legal_actions_generated),
        ("gpu_actions_scored", gpu_actions_scored),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    gpu = _finite_vector(learner_gpu_utilization, "learner_gpu_utilization")
    if np.any((gpu < 0.0) | (gpu > 100.0)):
        raise ValueError("learner_gpu_utilization must be in [0, 100]")
    actor_memory = _finite_nonnegative(actor_peak_memory_mb, "actor_peak_memory_mb")
    learner_memory = _finite_nonnegative(learner_peak_memory_mb, "learner_peak_memory_mb")
    lag = _finite_vector(policy_version_lag, "policy_version_lag")
    if np.any(lag < 0.0):
        raise ValueError("policy_version_lag must be non-negative")
    return EngineeringPerformanceReport(
        schema_version=METRICS_SCHEMA_VERSION,
        elapsed_seconds=elapsed,
        environment_steps_per_second=environment_steps / elapsed,
        legal_actions_per_second=legal_actions_generated / elapsed,
        gpu_actions_scored_per_second=gpu_actions_scored / elapsed,
        inference_latency=_latency_summary(inference_latency_ms, "inference_latency_ms"),
        actor_queue_wait=_latency_summary(actor_queue_wait_ms, "actor_queue_wait_ms"),
        learner_gpu_utilization_mean=float(np.mean(gpu)),
        learner_gpu_utilization_p95=float(np.quantile(gpu, 0.95)),
        policy_version_lag=_distribution_summary(lag),
        actor_peak_memory_mb=actor_memory,
        learner_peak_memory_mb=learner_memory,
    )


def summarize_paired(
    results: Sequence[PairedMatchResult],
    deal_set: PairedDealSet,
    rules_hash: str,
    candidate_policy_id: str,
    baseline_policy_id: str,
    bootstrap_config: BootstrapConfig | None = None,
) -> ArenaReport:
    """Aggregate roles after validating a complete, strictly paired deal grid."""
    settings = bootstrap_config if bootstrap_config is not None else BootstrapConfig()
    indexed: dict[tuple[int, SeatRole], PairedMatchResult] = {}
    expected_deals = {deal.deal_index: deal for deal in deal_set.deals}
    for result in results:
        comparison = result.comparison
        key = (comparison.deal.deal_index, comparison.focal_role)
        if key in indexed:
            raise ValueError(f"duplicate paired result for deal/role {key}")
        expected = expected_deals.get(comparison.deal.deal_index)
        if expected != comparison.deal:
            raise ValueError(f"result references a deal outside the fixed set: {comparison.deal}")
        if comparison.candidate_policy_id != candidate_policy_id:
            raise ValueError("candidate policy ID differs from report request")
        if comparison.baseline_policy_id != baseline_policy_id:
            raise ValueError("baseline policy ID differs from report request")
        if result.candidate_match.rules_hash != rules_hash:
            raise ValueError("candidate match rule hash differs from report request")
        if result.baseline_match.rules_hash != rules_hash:
            raise ValueError("baseline match rule hash differs from report request")
        indexed[key] = result

    expected_count = deal_set.count * len(SEAT_ROLES)
    if len(indexed) != expected_count:
        raise ValueError(f"incomplete paired grid: expected {expected_count}, got {len(indexed)}")
    ordered_by_role = {
        role: tuple(indexed[(deal.deal_index, role)] for deal in deal_set.deals)
        for role in SEAT_ROLES
    }
    role_reports = {
        role: _role_report(role.value, ordered_by_role[role], settings) for role in SEAT_ROLES
    }
    farmer_clusters = tuple(
        (
            indexed[(deal.deal_index, SeatRole.LANDLORD_DOWN)],
            indexed[(deal.deal_index, SeatRole.LANDLORD_UP)],
        )
        for deal in deal_set.deals
    )
    all_clusters = tuple(
        tuple(indexed[(deal.deal_index, role)] for role in SEAT_ROLES) for deal in deal_set.deals
    )
    return ArenaReport(
        schema_version=METRICS_SCHEMA_VERSION,
        rules_hash=rules_hash,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        deal_set=deal_set,
        pair_count=expected_count,
        match_count=expected_count * 2,
        landlord=role_reports[SeatRole.LANDLORD],
        landlord_down=role_reports[SeatRole.LANDLORD_DOWN],
        landlord_up=role_reports[SeatRole.LANDLORD_UP],
        farmer_team=_cluster_report("farmer_team", farmer_clusters, settings),
        overall=_cluster_report("overall", all_clusters, settings),
    )


def summarize_cross_play(
    results: Sequence[MatchResult],
    deal_set: PairedDealSet,
    rules_hash: str,
    landlord_policy_ids: tuple[str, ...],
    farmer_policy_ids: tuple[str, ...],
    bootstrap_config: BootstrapConfig | None = None,
) -> CrossPlayReport:
    """Aggregate a complete fixed-deal landlord-versus-farmer matrix."""
    settings = bootstrap_config if bootstrap_config is not None else BootstrapConfig()
    indexed: dict[tuple[str, str, int], MatchResult] = {}
    expected_deals = {deal.deal_index: deal for deal in deal_set.deals}
    expected_cells = set(
        (landlord, farmer) for landlord in landlord_policy_ids for farmer in farmer_policy_ids
    )
    for result in results:
        landlord, farmer_down, farmer_up = result.assignment.policy_ids
        if farmer_down != farmer_up:
            raise ValueError("cross-play requires one shared farmer-team policy")
        cell = (landlord, farmer_down)
        if cell not in expected_cells:
            raise ValueError(f"unexpected cross-play cell: {cell}")
        expected = expected_deals.get(result.deal_index)
        if (
            expected is None
            or expected.seed != result.deal_seed
            or expected.deal_id != result.deal_id
        ):
            raise ValueError("cross-play result references a deal outside the fixed set")
        if result.rules_hash != rules_hash:
            raise ValueError("cross-play match rule hash differs from report request")
        key = (landlord, farmer_down, result.deal_index)
        if key in indexed:
            raise ValueError(f"duplicate cross-play result: {key}")
        indexed[key] = result

    expected_count = len(expected_cells) * deal_set.count
    if len(indexed) != expected_count:
        raise ValueError(
            f"incomplete cross-play grid: expected {expected_count}, got {len(indexed)}"
        )
    cells: list[CrossPlayCell] = []
    for landlord in landlord_policy_ids:
        for farmer in farmer_policy_ids:
            cell_results = tuple(
                indexed[(landlord, farmer, deal.deal_index)] for deal in deal_set.deals
            )
            cells.append(
                CrossPlayCell(
                    landlord_policy_id=landlord,
                    farmer_policy_id=farmer,
                    deal_count=deal_set.count,
                    landlord_win_rate=_mean_estimate(
                        [float(result.raw_payoff[0] > 0) for result in cell_results], settings
                    ),
                    landlord_raw_payoff=_mean_estimate(
                        [float(result.raw_payoff[0]) for result in cell_results], settings
                    ),
                    landlord_objective_payoff=_mean_estimate(
                        [float(result.objective_payoff[0]) for result in cell_results], settings
                    ),
                    mean_bomb_count=_mean_estimate(
                        [float(result.bomb_count) for result in cell_results], settings
                    ),
                )
            )
    return CrossPlayReport(
        schema_version=METRICS_SCHEMA_VERSION,
        rules_hash=rules_hash,
        deal_set=deal_set,
        cells=tuple(cells),
    )


def _role_report(
    role: str,
    results: Sequence[PairedMatchResult],
    config: BootstrapConfig,
) -> RoleReport:
    clusters = tuple((result,) for result in results)
    return _cluster_report(role, clusters, config)


def _cluster_report(
    role: str,
    clusters: Sequence[Sequence[PairedMatchResult]],
    config: BootstrapConfig,
) -> RoleReport:
    return RoleReport(
        role=role,
        deal_count=len(clusters),
        win_rate=_paired_estimate(clusters, _candidate_win, _baseline_win, config),
        raw_payoff=_paired_estimate(
            clusters,
            _candidate_raw,
            _baseline_raw,
            config,
        ),
        objective_payoff=_paired_estimate(
            clusters,
            _candidate_objective,
            _baseline_objective,
            config,
        ),
    )


def _paired_estimate(
    clusters: Sequence[Sequence[PairedMatchResult]],
    candidate_getter: Callable[[PairedMatchResult], float],
    baseline_getter: Callable[[PairedMatchResult], float],
    config: BootstrapConfig,
) -> PairedEstimate:
    candidate = np.asarray(
        [sum(candidate_getter(item) for item in cluster) / len(cluster) for cluster in clusters],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [sum(baseline_getter(item) for item in cluster) / len(cluster) for cluster in clusters],
        dtype=np.float64,
    )
    ci = bootstrap_paired_difference_ci(candidate, baseline, config)
    return PairedEstimate(
        sample_count=len(clusters),
        candidate_mean=float(candidate.mean()),
        baseline_mean=float(baseline.mean()),
        mean_delta=float((candidate - baseline).mean()),
        delta_ci=ci,
    )


def _mean_estimate(values: Sequence[float], config: BootstrapConfig) -> MeanEstimate:
    array = np.asarray(values, dtype=np.float64)
    return MeanEstimate(
        mean=float(array.mean()),
        standard_deviation=float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        ci=bootstrap_mean_ci(array, config),
    )


def _distribution_summary(values: Sequence[float] | np.ndarray) -> DistributionSummary:
    array = _finite_vector(values, "distribution values")
    cumulative = np.cumsum(array, dtype=np.float64)
    cumulative_with_origin = np.concatenate((np.zeros(1, dtype=np.float64), cumulative))
    running_peak = np.maximum.accumulate(cumulative_with_origin)
    maximum_drawdown = float(np.max(running_peak - cumulative_with_origin))
    return DistributionSummary(
        sample_count=len(array),
        mean=float(array.mean()),
        standard_deviation=float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        p10=float(np.quantile(array, 0.10)),
        p50=float(np.quantile(array, 0.50)),
        p90=float(np.quantile(array, 0.90)),
        maximum_drawdown=maximum_drawdown,
    )


def _calibration_summary(
    probabilities: Sequence[float],
    targets: Sequence[float],
    bins: int,
    label: str,
) -> CalibrationSummary:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("calibration_bins must be a positive integer")
    predicted = _probability_vector(probabilities, f"{label}_probabilities")
    actual = _binary_vector(targets, f"{label}_targets")
    if predicted.shape != actual.shape:
        raise ValueError(f"{label} probability and target shapes must match")
    bin_index = np.minimum((predicted * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = bin_index == index
        count = int(mask.sum())
        if count:
            ece += count / len(predicted) * abs(float(predicted[mask].mean() - actual[mask].mean()))
    return CalibrationSummary(
        sample_count=len(predicted),
        brier_score=float(np.mean(np.square(predicted - actual))),
        expected_calibration_error=float(ece),
    )


def _latency_summary(values: Sequence[float], label: str) -> LatencySummary:
    array = _finite_vector(values, label)
    if np.any(array < 0.0):
        raise ValueError(f"{label} must be non-negative")
    return LatencySummary(
        sample_count=len(array),
        mean_ms=float(array.mean()),
        p50_ms=float(np.quantile(array, 0.50)),
        p95_ms=float(np.quantile(array, 0.95)),
        p99_ms=float(np.quantile(array, 0.99)),
        maximum_ms=float(array.max()),
    )


def _finite_vector(values: Sequence[float] | np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _finite_matrix(values: Sequence[Sequence[float]], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{label} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _probability_vector(values: Sequence[float], label: str) -> np.ndarray:
    array = _finite_vector(values, label)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{label} must be in [0, 1]")
    return array


def _probability_matrix(values: Sequence[Sequence[float]], label: str) -> np.ndarray:
    array = _finite_matrix(values, label)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{label} must be in [0, 1]")
    return array


def _binary_vector(values: Sequence[float], label: str) -> np.ndarray:
    array = _finite_vector(values, label)
    if np.any((array != 0.0) & (array != 1.0)):
        raise ValueError(f"{label} must contain only zero or one")
    return array


def _binary_matrix(values: Sequence[Sequence[float]], label: str) -> np.ndarray:
    array = _finite_matrix(values, label)
    if np.any((array != 0.0) & (array != 1.0)):
        raise ValueError(f"{label} must contain only zero or one")
    return array


def _finite_nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _candidate_win(result: PairedMatchResult) -> float:
    return float(result.candidate_win)


def _baseline_win(result: PairedMatchResult) -> float:
    return float(result.baseline_win)


def _candidate_raw(result: PairedMatchResult) -> float:
    return float(result.candidate_raw_payoff)


def _baseline_raw(result: PairedMatchResult) -> float:
    return float(result.baseline_raw_payoff)


def _candidate_objective(result: PairedMatchResult) -> float:
    return float(result.candidate_objective_payoff)


def _baseline_objective(result: PairedMatchResult) -> float:
    return float(result.baseline_objective_payoff)


__all__ = (
    "METRICS_SCHEMA_VERSION",
    "ArenaReport",
    "CalibrationSummary",
    "CrossPlayCell",
    "CrossPlayReport",
    "DistributionSummary",
    "EngineeringPerformanceReport",
    "GamePerformanceReport",
    "LatencySummary",
    "MeanEstimate",
    "ModelQualityReport",
    "PairedEstimate",
    "QuantileCoverage",
    "RoleReport",
    "RoleWinRate",
    "summarize_cross_play",
    "summarize_engineering_performance",
    "summarize_game_performance",
    "summarize_model_quality",
    "summarize_paired",
)
