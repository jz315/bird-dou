"""Complete game, model-quality, and engineering metric protocol tests."""

from pathlib import Path

import pytest

from birddou import RuleConfig, load_rule_config
from birddou.eval import (
    Arena,
    FirstLegalPolicy,
    ScheduledMatch,
    SeatAssignment,
    generate_paired_deals,
    summarize_engineering_performance,
    summarize_game_performance,
    summarize_model_quality,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _rules() -> RuleConfig:
    return load_rule_config(REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml")


def test_match_and_game_report_capture_public_bombs_spring_and_tails() -> None:
    arena = Arena(_rules(), (FirstLegalPolicy("landlord"), FirstLegalPolicy("farmers")))
    deal = generate_paired_deals(831, 1).deals[0]
    result = arena.play_match(
        ScheduledMatch(
            "metric-match",
            deal,
            SeatAssignment(("landlord", "farmers", "farmers")),
        )
    )

    assert sum(result.bomb_count_by_seat) == result.bomb_count
    assert result.spring_outcome in {"none", "landlord_spring", "anti_spring"}
    report = summarize_game_performance((result,), "farmers")
    assert report.match_count == 1
    assert report.controlled_seat_count == 2
    assert report.role_win_rates["landlord_down"].sample_count == 1
    assert report.role_win_rates["landlord_up"].sample_count == 1
    assert report.raw_score.sample_count == 2
    assert report.training_score.maximum_drawdown >= 0.0
    assert report.anti_spring_opportunities == 1
    assert report.to_dict()["schema_version"] == 2


def test_model_quality_metrics_are_hand_checkable_and_serializable() -> None:
    report = summarize_model_quality(
        policy_probabilities=(0.5, 0.5, 1.0),
        action_offsets=(0, 2, 3),
        win_probabilities=(0.0, 1.0),
        win_targets=(0.0, 1.0),
        score_predictions=(1.0, -1.0),
        score_targets=(2.0, -2.0),
        score_quantile_predictions=((1.0, 2.0, 3.0), (-3.0, -2.0, -1.0)),
        score_quantile_levels=(0.1, 0.5, 0.9),
        belief_probabilities=((0.25, 0.75), (0.8, 0.2)),
        belief_targets=((0.0, 1.0), (1.0, 0.0)),
        belief_count_predictions=((1.0, 2.0), (2.0, 1.0)),
        belief_count_targets=((1.0, 3.0), (1.0, 1.0)),
        key_card_probabilities=(0.0, 1.0),
        key_card_targets=(0.0, 1.0),
        calibration_bins=2,
    )

    assert report.sample_count == 2
    assert report.mean_policy_entropy == pytest.approx(0.5 * 0.6931471805599453)
    assert report.win.brier_score == 0.0
    assert report.win.expected_calibration_error == 0.0
    assert report.score_mae == 1.0
    assert report.score_quantile_coverage.empirical_coverage == (0.0, 1.0, 1.0)
    assert report.belief_count_mae == 0.5
    assert report.key_card.brier_score == 0.0
    coverage = report.to_dict()["score_quantile_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["quantile_levels"] == (
        0.1,
        0.5,
        0.9,
    )


def test_model_quality_rejects_invalid_ragged_probability_segments() -> None:
    with pytest.raises(ValueError, match="probability distribution"):
        summarize_model_quality(
            policy_probabilities=(0.2, 0.2),
            action_offsets=(0, 2),
            win_probabilities=(0.5,),
            win_targets=(1.0,),
            score_predictions=(0.0,),
            score_targets=(0.0,),
            score_quantile_predictions=((0.0,),),
            score_quantile_levels=(0.5,),
            belief_probabilities=((0.5,),),
            belief_targets=((1.0,),),
            belief_count_predictions=((1.0,),),
            belief_count_targets=((1.0,),),
            key_card_probabilities=(0.5,),
            key_card_targets=(1.0,),
        )


def test_engineering_report_covers_throughput_latency_lag_gpu_and_memory() -> None:
    report = summarize_engineering_performance(
        elapsed_seconds=2.0,
        environment_steps=200,
        legal_actions_generated=1_000,
        gpu_actions_scored=800,
        inference_latency_ms=(1.0, 2.0, 3.0, 4.0),
        actor_queue_wait_ms=(0.0, 1.0, 2.0),
        learner_gpu_utilization=(25.0, 75.0),
        policy_version_lag=(0.0, 1.0, 2.0),
        actor_peak_memory_mb=128.0,
        learner_peak_memory_mb=512.0,
    )

    assert report.environment_steps_per_second == 100.0
    assert report.legal_actions_per_second == 500.0
    assert report.gpu_actions_scored_per_second == 400.0
    assert report.inference_latency.p50_ms == 2.5
    assert report.inference_latency.p99_ms > report.inference_latency.p95_ms
    assert report.learner_gpu_utilization_mean == 50.0
    assert report.policy_version_lag.maximum_drawdown == 0.0
    assert report.to_dict()["learner_peak_memory_mb"] == 512.0
