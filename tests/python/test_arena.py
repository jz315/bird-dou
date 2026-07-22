"""Unified native-Arena acceptance tests for E012."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from birddou import RuleConfig, load_rule_config
from birddou.cli.evaluate import main as evaluate_main
from birddou.env_types import Action, Observation
from birddou.eval.arena import Arena, PolicyDecisionError
from birddou.eval.baselines import (
    FirstLegalPolicy,
    LongestMovePolicy,
    PolicyDecisionContext,
    SeededRandomPolicy,
)
from birddou.eval.bootstrap import BootstrapConfig
from birddou.eval.paired_deals import ScheduledMatch, SeatAssignment, generate_paired_deals

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
FAST_BOOTSTRAP = BootstrapConfig(resamples=200, seed=20260722)


def rules() -> RuleConfig:
    """Load the authoritative post-bid rule profile."""
    return load_rule_config(RULE_CONFIG_PATH)


def test_identical_behaviors_have_exact_paired_zero_and_repeat() -> None:
    """Same deals, rotation, policies, and bootstrap seed reproduce bit for bit."""
    arena = Arena(rules(), (FirstLegalPolicy("A"), FirstLegalPolicy("B")))
    deal_set = generate_paired_deals(321, 2)

    first = arena.evaluate_paired(deal_set, "A", "B", FAST_BOOTSTRAP)
    second = arena.evaluate_paired(deal_set, "A", "B", FAST_BOOTSTRAP)

    assert first == second
    assert len(first.results) == 6
    assert first.report.pair_count == 6
    assert first.report.match_count == 12
    for pair in first.results:
        assert pair.candidate_match.deal_seed == pair.baseline_match.deal_seed
        assert pair.candidate_match.landlord_seat == 0
        assert pair.candidate_match.bidding_record_json == "[]"
        assert (
            pair.candidate_match.terminal_state_sha256 == pair.baseline_match.terminal_state_sha256
        )
        assert pair.candidate_match.raw_payoff == pair.baseline_match.raw_payoff
    for report in (
        first.report.landlord,
        first.report.landlord_down,
        first.report.landlord_up,
        first.report.farmer_team,
        first.report.overall,
    ):
        assert report.raw_payoff.mean_delta == 0.0
        assert report.raw_payoff.delta_ci.lower == 0.0
        assert report.raw_payoff.delta_ci.upper == 0.0
        assert report.win_rate.mean_delta == 0.0
    assert first.report.meets_precision(0.001)


def test_distinct_policies_produce_complete_role_reports() -> None:
    """A nontrivial comparison retains all three roles and deal clusters."""
    arena = Arena(
        rules(),
        (LongestMovePolicy("long"), SeededRandomPolicy("random", seed=17)),
    )
    deal_set = generate_paired_deals(99, 2)
    run = arena.evaluate_paired(deal_set, "long", "random", FAST_BOOTSTRAP)

    assert len(run.results) == deal_set.count * 3
    assert run.report.landlord.role == "landlord"
    assert run.report.landlord_down.role == "landlord_down"
    assert run.report.landlord_up.role == "landlord_up"
    assert run.report.farmer_team.role == "farmer_team"
    assert run.report.overall.raw_payoff.sample_count == deal_set.count
    assert run.report.deal_set.master_seed == 99


@dataclass
class RecordingPolicy:
    """Test policy that records only the observations Arena makes visible."""

    policy_id: str
    observations: list[Observation] = field(default_factory=list)

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        assert observation["observer"] == context.seat
        assert observation["current_player"] == context.seat
        assert "hands" not in observation
        self.observations.append(observation)
        return 0


def test_arena_policy_boundary_is_information_set_safe() -> None:
    """A policy receives its own observation and legal actions, never full state."""
    recorder = RecordingPolicy("record")
    arena = Arena(rules(), (recorder, FirstLegalPolicy("other")))
    deal = generate_paired_deals(44, 1).deals[0]
    result = arena.play_match(
        ScheduledMatch("safe-boundary", deal, SeatAssignment(("record", "other", "other")))
    )

    assert result.action_count > 0
    assert recorder.observations


@dataclass(frozen=True)
class InvalidPolicy:
    """Test policy returning a forbidden action index."""

    policy_id: str = "invalid"

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        del observation, legal_actions, context
        return -1


def test_invalid_policy_decision_fails_without_fallback() -> None:
    """Arena reports policy contract violations instead of silently substituting."""
    arena = Arena(rules(), (InvalidPolicy(), FirstLegalPolicy("other")))
    deal = generate_paired_deals(45, 1).deals[0]
    scheduled = ScheduledMatch(
        "invalid-decision",
        deal,
        SeatAssignment(("invalid", "other", "other")),
    )

    with pytest.raises(PolicyDecisionError, match="outside"):
        arena.play_match(scheduled)


def test_cross_play_builds_every_matrix_cell() -> None:
    """Cross-play reports one deterministic cell per ordered policy pairing."""
    policies = (
        FirstLegalPolicy("first"),
        LongestMovePolicy("long"),
        SeededRandomPolicy("random", 3),
    )
    arena = Arena(rules(), policies)
    deal_set = generate_paired_deals(1234, 1)
    run = arena.evaluate_cross_play(
        deal_set,
        ("first", "long"),
        ("random",),
        FAST_BOOTSTRAP,
    )

    assert len(run.results) == 2
    assert [(cell.landlord_policy_id, cell.farmer_policy_id) for cell in run.report.cells] == [
        ("first", "random"),
        ("long", "random"),
    ]


def test_evaluate_cli_writes_auditable_json(tmp_path: Path) -> None:
    """The installed command boundary emits the versioned formal report."""
    output = tmp_path / "report.json"
    exit_code = evaluate_main(
        (
            "--rules",
            str(RULE_CONFIG_PATH),
            "--candidate",
            "first_legal",
            "--baseline",
            "first_legal",
            "--deals",
            "1",
            "--bootstrap-resamples",
            "100",
            "--output",
            str(output),
        )
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["schema_version"] == 2
    assert len(payload["deal_set"]["deals"]) == 1
    assert payload["pair_count"] == 3
    assert payload["match_count"] == 6
