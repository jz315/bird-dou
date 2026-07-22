"""Explicit empirical acceptance gates for M3-M6 research artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import cast

MILESTONE_GATE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MilestoneGateReport:
    """Auditable pass/fail result that never invents missing experimental evidence."""

    schema_version: int
    milestone: str
    accepted: bool
    metrics: Mapping[str, float | bool]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def evaluate_m3_gate(
    *,
    differential_mismatches: int,
    random_paired_ci_lower: float,
    rlcard_paired_ci_lower: float,
    checkpoint_resume_exact: bool,
) -> MilestoneGateReport:
    """Require exact environment compatibility and stable gains over both weak baselines."""
    if differential_mismatches < 0:
        raise ValueError("differential mismatch count cannot be negative")
    _finite(random_paired_ci_lower, rlcard_paired_ci_lower)
    reasons: list[str] = []
    if differential_mismatches != 0:
        reasons.append("official differential contains mismatches")
    if random_paired_ci_lower <= 0.0:
        reasons.append("paired lower bound against random is not positive")
    if rlcard_paired_ci_lower <= 0.0:
        reasons.append("paired lower bound against RLCard is not positive")
    if not checkpoint_resume_exact:
        reasons.append("checkpoint next-update resume is not exact")
    return _report(
        "M3",
        reasons,
        {
            "differential_mismatches": float(differential_mismatches),
            "random_paired_ci_lower": random_paired_ci_lower,
            "rlcard_paired_ci_lower": rlcard_paired_ci_lower,
            "checkpoint_resume_exact": checkpoint_resume_exact,
        },
    )


def evaluate_m4_gate(
    *,
    paired_delta_point: float,
    paired_delta_ci_lower: float,
    noninferiority_margin: float,
    nonfinite_update_count: int,
    claims_stronger: bool,
    ablations_complete: bool,
) -> MilestoneGateReport:
    """Accept non-inferiority, and require ablations before a stronger claim."""
    _finite(paired_delta_point, paired_delta_ci_lower, noninferiority_margin)
    if noninferiority_margin < 0.0 or nonfinite_update_count < 0:
        raise ValueError("M4 margin/count is invalid")
    reasons: list[str] = []
    if nonfinite_update_count:
        reasons.append("training produced non-finite updates")
    if paired_delta_ci_lower < -noninferiority_margin:
        reasons.append("structured model failed the declared non-inferiority margin")
    if claims_stronger and paired_delta_point <= 0.0:
        reasons.append("stronger claim has no positive point estimate")
    if claims_stronger and not ablations_complete:
        reasons.append("stronger claim is missing required module ablations")
    return _report(
        "M4",
        reasons,
        {
            "paired_delta_point": paired_delta_point,
            "paired_delta_ci_lower": paired_delta_ci_lower,
            "noninferiority_margin": noninferiority_margin,
            "nonfinite_update_count": float(nonfinite_update_count),
            "claims_stronger": claims_stronger,
            "ablations_complete": ablations_complete,
        },
    )


def evaluate_m5_gate(
    *,
    conservation_violation_rate: float,
    trained_nll: float,
    uniform_nll: float,
    joint_policy_ci_lower: float,
    noninferiority_margin: float,
    shuffled_belief_performance_drop: float,
) -> MilestoneGateReport:
    """Require exact conservation, useful likelihood, non-regression, and Belief use."""
    _finite(
        conservation_violation_rate,
        trained_nll,
        uniform_nll,
        joint_policy_ci_lower,
        noninferiority_margin,
        shuffled_belief_performance_drop,
    )
    if conservation_violation_rate < 0.0 or noninferiority_margin < 0.0:
        raise ValueError("M5 rates/margin cannot be negative")
    reasons: list[str] = []
    if conservation_violation_rate != 0.0:
        reasons.append("Belief conservation violation rate is not zero")
    if trained_nll >= uniform_nll:
        reasons.append("trained Belief NLL does not beat the uniform constrained baseline")
    if joint_policy_ci_lower < -noninferiority_margin:
        reasons.append("joint Belief policy failed the non-inferiority margin")
    if shuffled_belief_performance_drop <= 0.0:
        reasons.append("Belief shuffle did not reduce held-out policy performance")
    return _report(
        "M5",
        reasons,
        {
            "conservation_violation_rate": conservation_violation_rate,
            "trained_nll": trained_nll,
            "uniform_nll": uniform_nll,
            "joint_policy_ci_lower": joint_policy_ci_lower,
            "noninferiority_margin": noninferiority_margin,
            "shuffled_belief_performance_drop": shuffled_belief_performance_drop,
        },
    )


def evaluate_m6_gate(
    *,
    teacher_vs_student_ci_lower: float,
    distilled_vs_undistilled_ci_lower: float,
    student_information_set_max_abs_difference: float,
    leakage_tolerance: float,
    direct_kd_gain: float,
    information_set_kd_gain: float,
    direct_vs_is_ablation_complete: bool,
) -> MilestoneGateReport:
    """Require useful privilege/distillation, strict execution safety, and honest ablation."""
    _finite(
        teacher_vs_student_ci_lower,
        distilled_vs_undistilled_ci_lower,
        student_information_set_max_abs_difference,
        leakage_tolerance,
        direct_kd_gain,
        information_set_kd_gain,
    )
    if student_information_set_max_abs_difference < 0.0 or leakage_tolerance < 0.0:
        raise ValueError("M6 leakage values cannot be negative")
    reasons: list[str] = []
    if teacher_vs_student_ci_lower <= 0.0:
        reasons.append("Teacher has no positive paired lower bound over Student")
    if distilled_vs_undistilled_ci_lower <= 0.0:
        reasons.append("distilled Student has no positive paired lower bound")
    if student_information_set_max_abs_difference > leakage_tolerance:
        reasons.append("Student output changes inside one information set")
    if not direct_vs_is_ablation_complete:
        reasons.append("direct-KD versus IS-KD ablation is incomplete")
    return _report(
        "M6",
        reasons,
        {
            "teacher_vs_student_ci_lower": teacher_vs_student_ci_lower,
            "distilled_vs_undistilled_ci_lower": distilled_vs_undistilled_ci_lower,
            "student_information_set_max_abs_difference": (
                student_information_set_max_abs_difference
            ),
            "leakage_tolerance": leakage_tolerance,
            "direct_kd_gain": direct_kd_gain,
            "information_set_kd_gain": information_set_kd_gain,
            "direct_vs_is_ablation_complete": direct_vs_is_ablation_complete,
        },
    )


def _report(
    milestone: str,
    reasons: list[str],
    metrics: Mapping[str, float | bool],
) -> MilestoneGateReport:
    return MilestoneGateReport(
        MILESTONE_GATE_SCHEMA_VERSION,
        milestone,
        not reasons,
        dict(metrics),
        tuple(reasons),
    )


def _finite(*values: float) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError("milestone gate inputs must be finite")


__all__ = (
    "MILESTONE_GATE_SCHEMA_VERSION",
    "MilestoneGateReport",
    "evaluate_m3_gate",
    "evaluate_m4_gate",
    "evaluate_m5_gate",
    "evaluate_m6_gate",
)
