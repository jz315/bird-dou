"""Deterministic, information-set-safe evaluation over the native environment."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeAlias

from birddou.env import PyDdzEnv
from birddou.env_types import RuleConfig, StepResult
from birddou.eval.baselines import Policy, PolicyDecisionContext
from birddou.eval.bootstrap import BootstrapConfig
from birddou.eval.metrics import (
    ArenaReport,
    CrossPlayReport,
    summarize_cross_play,
    summarize_paired,
)
from birddou.eval.paired_deals import (
    PairedComparison,
    PairedDealSet,
    ScheduledMatch,
    SeatAssignment,
    SeatRole,
    generate_cross_play_schedule,
    generate_paired_comparisons,
    role_for_game_seat,
    role_for_seat,
    splitmix64,
)

ARENA_SCHEMA_VERSION = 2
SpringOutcome: TypeAlias = Literal["none", "landlord_spring", "anti_spring"]


class ArenaError(RuntimeError):
    """Base class for an invalid or failed evaluation run."""


class PolicyDecisionError(ArenaError):
    """A policy failed or returned an invalid local action index."""


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    """Versioned safety controls for deterministic match execution."""

    schema_version: int = ARENA_SCHEMA_VERSION
    max_actions: int = 1_000
    max_redeals: int = 32

    def __post_init__(self) -> None:
        if self.schema_version != ARENA_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Arena schema {self.schema_version}; expected {ARENA_SCHEMA_VERSION}"
            )
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if self.max_redeals < 0:
            raise ValueError("max_redeals must be non-negative")


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Auditable terminal result for one fixed deal and seat assignment."""

    schema_version: int
    match_id: str
    deal_index: int
    deal_seed: int
    final_deal_seed: int
    deal_id: str
    assignment: SeatAssignment
    rule_config_id: int
    rules_hash: str
    landlord_seat: int
    bidding_record_json: str
    redeal_count: int
    action_count: int
    winner_seat: int
    winner_role: SeatRole
    winner_policy_id: str
    raw_payoff: tuple[int, int, int]
    objective_payoff: tuple[int, int, int]
    bomb_count: int
    bomb_count_by_seat: tuple[int, int, int]
    spring_outcome: SpringOutcome
    terminal_state_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != ARENA_SCHEMA_VERSION:
            raise ValueError("unsupported match-result schema")
        role_for_seat(self.landlord_seat)
        if role_for_game_seat(self.winner_seat, self.landlord_seat) is not self.winner_role:
            raise ValueError("winner role does not match winner seat")
        if self.assignment.policy_for_seat(self.winner_seat) != self.winner_policy_id:
            raise ValueError("winner policy does not match the seat assignment")
        if self.action_count <= 0:
            raise ValueError("terminal match must contain at least one action")
        if self.bomb_count < 0:
            raise ValueError("bomb_count must be non-negative")
        if any(count < 0 for count in self.bomb_count_by_seat):
            raise ValueError("bomb_count_by_seat must be non-negative")
        if sum(self.bomb_count_by_seat) != self.bomb_count:
            raise ValueError("per-seat bomb counts must sum to bomb_count")
        if self.spring_outcome not in {"none", "landlord_spring", "anti_spring"}:
            raise ValueError("invalid spring outcome")
        if self.spring_outcome == "landlord_spring" and self.winner_role is not SeatRole.LANDLORD:
            raise ValueError("landlord spring requires a landlord win")
        if self.spring_outcome == "anti_spring" and self.winner_role is SeatRole.LANDLORD:
            raise ValueError("anti-spring requires a farmer win")
        if self.redeal_count < 0:
            raise ValueError("redeal_count must be non-negative")


@dataclass(frozen=True, slots=True)
class PairedMatchResult:
    """Symmetric candidate/baseline results for one deal and focal role."""

    comparison: PairedComparison
    candidate_match: MatchResult
    baseline_match: MatchResult

    def __post_init__(self) -> None:
        expected_candidate_id = f"{self.comparison.pair_id}-candidate"
        expected_baseline_id = f"{self.comparison.pair_id}-baseline"
        if self.candidate_match.match_id != expected_candidate_id:
            raise ValueError("candidate match ID does not match comparison")
        if self.baseline_match.match_id != expected_baseline_id:
            raise ValueError("baseline match ID does not match comparison")
        if self.candidate_match.deal_id != self.baseline_match.deal_id:
            raise ValueError("paired matches must use the same fixed deal")

    @property
    def candidate_raw_payoff(self) -> int:
        """Candidate raw payoff at the focal role."""
        return self.candidate_match.raw_payoff[self.comparison.focal_seat]

    @property
    def baseline_raw_payoff(self) -> int:
        """Baseline raw payoff at the focal role."""
        return self.baseline_match.raw_payoff[self.comparison.focal_seat]

    @property
    def candidate_objective_payoff(self) -> int:
        """Candidate transformed payoff at the focal role."""
        return self.candidate_match.objective_payoff[self.comparison.focal_seat]

    @property
    def baseline_objective_payoff(self) -> int:
        """Baseline transformed payoff at the focal role."""
        return self.baseline_match.objective_payoff[self.comparison.focal_seat]

    @property
    def candidate_win(self) -> int:
        """Whether the candidate's focal side won."""
        return int(self.candidate_raw_payoff > 0)

    @property
    def baseline_win(self) -> int:
        """Whether the baseline's focal side won."""
        return int(self.baseline_raw_payoff > 0)


@dataclass(frozen=True, slots=True)
class PairedArenaRun:
    """Complete paired trajectories and their deal-clustered report."""

    results: tuple[PairedMatchResult, ...]
    report: ArenaReport


@dataclass(frozen=True, slots=True)
class CrossPlayArenaRun:
    """Complete cross-play trajectories and their matrix report."""

    results: tuple[MatchResult, ...]
    report: CrossPlayReport


class Arena:
    """Execute every policy through the same authoritative Rust rules engine."""

    def __init__(
        self,
        rule_config: RuleConfig,
        policies: Sequence[Policy],
        config: ArenaConfig | None = None,
    ) -> None:
        self._config = config if config is not None else ArenaConfig()
        self._rule_config = deepcopy(rule_config)
        canonical_rules = json.dumps(
            self._rule_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._rules_hash = hashlib.sha256(canonical_rules).hexdigest()
        self._policies: dict[str, Policy] = {}
        for policy in policies:
            if not policy.policy_id:
                raise ValueError("policy IDs must be non-empty")
            if policy.policy_id in self._policies:
                raise ValueError(f"duplicate policy ID: {policy.policy_id}")
            self._policies[policy.policy_id] = policy
        if not self._policies:
            raise ValueError("Arena requires at least one policy")

    @property
    def rules_hash(self) -> str:
        """SHA-256 of the canonical rule configuration."""
        return self._rules_hash

    def play_match(self, scheduled: ScheduledMatch) -> MatchResult:
        """Play one fixed deal without exposing hidden hands to a policy."""
        missing = set(scheduled.assignment.policy_ids) - self._policies.keys()
        if missing:
            raise ArenaError(
                f"match {scheduled.match_id} references unknown policies: {sorted(missing)}"
            )

        environment = PyDdzEnv()
        final_deal_seed = scheduled.deal.seed
        environment.reset(final_deal_seed, self._rule_config)
        bidding_attempts: list[list[object]] = []
        redeal_count = 0
        decision_counts = [0, 0, 0]
        action_count = 0
        terminal_result: StepResult | None = None

        while True:
            while not environment.terminal:
                if action_count >= self._config.max_actions:
                    raise ArenaError(
                        f"match {scheduled.match_id} exceeded {self._config.max_actions} actions"
                    )
                seat = environment.current_player
                policy_id = scheduled.assignment.policy_for_seat(seat)
                policy = self._policies[policy_id]
                observation = environment.observe(seat)
                landlord = observation["landlord"]
                role = None if landlord is None else role_for_game_seat(seat, landlord)
                legal_actions = tuple(environment.legal_actions())
                if not legal_actions:
                    raise ArenaError(
                        f"match {scheduled.match_id} has no action at non-terminal seat {seat}"
                    )
                context = PolicyDecisionContext(
                    deal_index=scheduled.deal.deal_index,
                    deal_seed=final_deal_seed,
                    match_id=scheduled.match_id,
                    seat=seat,
                    role=role,
                    decision_index=decision_counts[seat],
                )
                try:
                    selected = policy.select_action(observation, legal_actions, context)
                except Exception as error:
                    raise PolicyDecisionError(
                        f"policy {policy_id} failed in match {scheduled.match_id}, "
                        f"seat {seat}, decision {decision_counts[seat]}: {error}"
                    ) from error
                if isinstance(selected, bool):
                    raise PolicyDecisionError(
                        f"policy {policy_id} returned bool instead of an action index "
                        f"in match {scheduled.match_id}"
                    )
                try:
                    action_index = operator.index(selected)
                except TypeError as error:
                    raise PolicyDecisionError(
                        f"policy {policy_id} returned non-integer action index {selected!r} "
                        f"in match {scheduled.match_id}"
                    ) from error
                if not 0 <= action_index < len(legal_actions):
                    raise PolicyDecisionError(
                        f"policy {policy_id} returned action index {action_index} outside "
                        f"0..{len(legal_actions) - 1} in match {scheduled.match_id}"
                    )
                terminal_result = environment.step(legal_actions[action_index])
                decision_counts[seat] += 1
                action_count += 1

            terminal_observation = environment.observe(environment.current_player)
            if terminal_observation["landlord"] is not None:
                break
            if not self._rule_config["bidding"]["redeal_on_all_pass"]:
                raise ArenaError(f"match {scheduled.match_id} ended all-pass with redeal disabled")
            bidding_attempts.append(list(terminal_observation["bid_history"]))
            if redeal_count >= self._config.max_redeals:
                raise ArenaError(
                    f"match {scheduled.match_id} exceeded {self._config.max_redeals} redeals"
                )
            redeal_count += 1
            final_deal_seed = splitmix64((scheduled.deal.seed + redeal_count) & ((1 << 64) - 1))
            environment.reset(final_deal_seed, self._rule_config)

        if terminal_result is None or not terminal_result["terminal"]:
            raise ArenaError(f"match {scheduled.match_id} ended without a terminal result")
        terminal_state = environment.serialize()
        winner_seat = terminal_result["event"]["actor"]
        winner_observation = environment.observe(winner_seat)
        landlord_seat = winner_observation["landlord"]
        if landlord_seat is None:
            raise ArenaError(f"match {scheduled.match_id} has no landlord after card play")
        final_bids = list(winner_observation["bid_history"])
        bidding_payload: object = (
            final_bids if not bidding_attempts else [*bidding_attempts, final_bids]
        )
        bidding_record_json = json.dumps(bidding_payload, sort_keys=True, separators=(",", ":"))
        bomb_count_by_seat, spring_outcome = _public_terminal_facts(
            winner_observation["history"],
            landlord_seat,
            winner_seat,
            self._rule_config,
        )
        return MatchResult(
            schema_version=ARENA_SCHEMA_VERSION,
            match_id=scheduled.match_id,
            deal_index=scheduled.deal.deal_index,
            deal_seed=scheduled.deal.seed,
            final_deal_seed=final_deal_seed,
            deal_id=scheduled.deal.deal_id,
            assignment=scheduled.assignment,
            rule_config_id=self._rule_config["rule_config_id"],
            rules_hash=self._rules_hash,
            landlord_seat=landlord_seat,
            bidding_record_json=bidding_record_json,
            redeal_count=redeal_count,
            action_count=action_count,
            winner_seat=winner_seat,
            winner_role=role_for_game_seat(winner_seat, landlord_seat),
            winner_policy_id=scheduled.assignment.policy_for_seat(winner_seat),
            raw_payoff=_payoff_tuple(terminal_result["raw_payoff"]),
            objective_payoff=_payoff_tuple(terminal_result["objective_payoff"]),
            bomb_count=winner_observation["bomb_count"],
            bomb_count_by_seat=bomb_count_by_seat,
            spring_outcome=spring_outcome,
            terminal_state_sha256=hashlib.sha256(terminal_state).hexdigest(),
        )

    def run_schedule(self, schedule: Sequence[ScheduledMatch]) -> tuple[MatchResult, ...]:
        """Run a stable arbitrary match schedule in order."""
        if not schedule:
            raise ValueError("schedule must contain at least one match")
        return tuple(self.play_match(match) for match in schedule)

    def run_paired(
        self,
        deal_set: PairedDealSet,
        candidate_policy_id: str,
        baseline_policy_id: str,
    ) -> tuple[PairedMatchResult, ...]:
        """Run six matches per deal: one symmetric pair for every role."""
        results: list[PairedMatchResult] = []
        for comparison in generate_paired_comparisons(
            deal_set,
            candidate_policy_id,
            baseline_policy_id,
        ):
            candidate = self.play_match(_scheduled_from_comparison(comparison, candidate=True))
            baseline = self.play_match(_scheduled_from_comparison(comparison, candidate=False))
            results.append(PairedMatchResult(comparison, candidate, baseline))
        return tuple(results)

    def evaluate_paired(
        self,
        deal_set: PairedDealSet,
        candidate_policy_id: str,
        baseline_policy_id: str,
        bootstrap_config: BootstrapConfig | None = None,
    ) -> PairedArenaRun:
        """Run and summarize a role-balanced paired comparison."""
        results = self.run_paired(deal_set, candidate_policy_id, baseline_policy_id)
        report = summarize_paired(
            results,
            deal_set,
            self._rules_hash,
            candidate_policy_id,
            baseline_policy_id,
            bootstrap_config,
        )
        return PairedArenaRun(results, report)

    def evaluate_cross_play(
        self,
        deal_set: PairedDealSet,
        landlord_policy_ids: tuple[str, ...],
        farmer_policy_ids: tuple[str, ...],
        bootstrap_config: BootstrapConfig | None = None,
    ) -> CrossPlayArenaRun:
        """Run and summarize every ordered landlord-versus-farmer cell."""
        schedule = generate_cross_play_schedule(
            deal_set,
            landlord_policy_ids,
            farmer_policy_ids,
        )
        results = self.run_schedule(schedule)
        report = summarize_cross_play(
            results,
            deal_set,
            self._rules_hash,
            landlord_policy_ids,
            farmer_policy_ids,
            bootstrap_config,
        )
        return CrossPlayArenaRun(results, report)


def _scheduled_from_comparison(
    comparison: PairedComparison,
    *,
    candidate: bool,
) -> ScheduledMatch:
    suffix = "candidate" if candidate else "baseline"
    assignment = comparison.candidate_assignment if candidate else comparison.baseline_assignment
    return ScheduledMatch(
        match_id=f"{comparison.pair_id}-{suffix}",
        deal=comparison.deal,
        assignment=assignment,
    )


def _payoff_tuple(values: list[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ArenaError(f"native payoff must have three seats, got {len(values)}")
    return values[0], values[1], values[2]


def _public_terminal_facts(
    history: Sequence[object],
    landlord_seat: int,
    winner_seat: int,
    rules: RuleConfig,
) -> tuple[tuple[int, int, int], SpringOutcome]:
    """Derive bombs and spring outcome strictly from the public action trace."""
    bombs = [0, 0, 0]
    landlord_non_pass = 0
    farmer_non_pass = 0
    for raw_event in history:
        if not isinstance(raw_event, dict):
            raise ArenaError("public history contains a non-object event")
        actor = raw_event.get("actor")
        action = raw_event.get("action")
        if not isinstance(actor, int) or isinstance(actor, bool) or not 0 <= actor < 3:
            raise ArenaError("public history contains an invalid actor")
        if not isinstance(action, dict) or "play" not in action:
            continue
        move = action["play"]
        if not isinstance(move, dict):
            raise ArenaError("public history contains an invalid play action")
        kind = move.get("kind")
        if not isinstance(kind, str):
            raise ArenaError("public history contains a play without a kind")
        if kind in {"bomb", "rocket"}:
            bombs[actor] += 1
        if kind != "pass":
            if actor == landlord_seat:
                landlord_non_pass += 1
            else:
                farmer_non_pass += 1

    spring: SpringOutcome = "none"
    spring_rules = rules["spring"]
    if (
        winner_seat == landlord_seat
        and spring_rules["landlord_spring_enabled"]
        and farmer_non_pass == 0
    ):
        spring = "landlord_spring"
    elif (
        winner_seat != landlord_seat
        and spring_rules["anti_spring_enabled"]
        and landlord_non_pass == 1
    ):
        spring = "anti_spring"
    return (bombs[0], bombs[1], bombs[2]), spring


__all__ = (
    "ARENA_SCHEMA_VERSION",
    "Arena",
    "ArenaConfig",
    "ArenaError",
    "CrossPlayArenaRun",
    "MatchResult",
    "PairedArenaRun",
    "PairedMatchResult",
    "PolicyDecisionError",
    "SpringOutcome",
)
