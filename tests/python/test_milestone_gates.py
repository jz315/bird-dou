"""Research gates distinguish completed evidence from merely executable code."""

from birddou.eval import (
    evaluate_m3_gate,
    evaluate_m4_gate,
    evaluate_m5_gate,
    evaluate_m6_gate,
)


def test_m3_and_m4_gates_require_differential_baselines_and_honest_ablations() -> None:
    m3 = evaluate_m3_gate(
        differential_mismatches=0,
        random_paired_ci_lower=0.2,
        rlcard_paired_ci_lower=0.1,
        checkpoint_resume_exact=True,
    )
    rejected_m3 = evaluate_m3_gate(
        differential_mismatches=1,
        random_paired_ci_lower=0.2,
        rlcard_paired_ci_lower=0.0,
        checkpoint_resume_exact=False,
    )
    m4 = evaluate_m4_gate(
        paired_delta_point=0.02,
        paired_delta_ci_lower=-0.01,
        noninferiority_margin=0.02,
        nonfinite_update_count=0,
        claims_stronger=False,
        ablations_complete=False,
    )
    rejected_m4 = evaluate_m4_gate(
        paired_delta_point=0.02,
        paired_delta_ci_lower=0.01,
        noninferiority_margin=0.02,
        nonfinite_update_count=0,
        claims_stronger=True,
        ablations_complete=False,
    )

    assert m3.accepted and m4.accepted
    assert not rejected_m3.accepted and len(rejected_m3.reasons) == 3
    assert not rejected_m4.accepted and "ablations" in rejected_m4.reasons[0]


def test_m5_and_m6_gates_require_policy_evidence_and_information_set_safety() -> None:
    m5 = evaluate_m5_gate(
        conservation_violation_rate=0.0,
        trained_nll=4.6,
        uniform_nll=5.1,
        joint_policy_ci_lower=-0.01,
        noninferiority_margin=0.02,
        shuffled_belief_performance_drop=0.03,
    )
    rejected_m5 = evaluate_m5_gate(
        conservation_violation_rate=0.001,
        trained_nll=5.2,
        uniform_nll=5.1,
        joint_policy_ci_lower=-0.1,
        noninferiority_margin=0.02,
        shuffled_belief_performance_drop=0.0,
    )
    m6 = evaluate_m6_gate(
        teacher_vs_student_ci_lower=0.1,
        distilled_vs_undistilled_ci_lower=0.02,
        student_information_set_max_abs_difference=0.0,
        leakage_tolerance=0.0,
        direct_kd_gain=0.01,
        information_set_kd_gain=0.02,
        direct_vs_is_ablation_complete=True,
    )
    rejected_m6 = evaluate_m6_gate(
        teacher_vs_student_ci_lower=0.0,
        distilled_vs_undistilled_ci_lower=-0.01,
        student_information_set_max_abs_difference=1e-3,
        leakage_tolerance=0.0,
        direct_kd_gain=0.0,
        information_set_kd_gain=0.0,
        direct_vs_is_ablation_complete=False,
    )

    assert m5.accepted and m6.accepted
    assert not rejected_m5.accepted and len(rejected_m5.reasons) == 4
    assert not rejected_m6.accepted and len(rejected_m6.reasons) == 4
