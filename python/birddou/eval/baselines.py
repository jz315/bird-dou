"""Deterministic policy protocol and dependency-free rule baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from birddou.env_types import Action, Observation, PlayGameAction
from birddou.eval.paired_deals import SeatRole, splitmix64


@dataclass(frozen=True, slots=True)
class PolicyDecisionContext:
    """Reproducibility metadata supplied to every policy decision."""

    deal_index: int
    deal_seed: int
    match_id: str
    seat: int
    role: SeatRole | None
    decision_index: int


@runtime_checkable
class Policy(Protocol):
    """Minimal information-set-safe policy consumed by the unified Arena."""

    @property
    def policy_id(self) -> str:
        """Stable model or baseline identifier."""
        ...

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        """Return one local index into the supplied stable legal action list."""
        ...


@dataclass(frozen=True, slots=True)
class FirstLegalPolicy:
    """Always select the first canonical legal action."""

    policy_id: str = "first_legal"

    def __post_init__(self) -> None:
        _validate_policy_id(self.policy_id)

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        del observation, context
        if not legal_actions:
            raise ValueError("FirstLegalPolicy received no legal actions")
        return 0


@dataclass(frozen=True, slots=True)
class LongestMovePolicy:
    """Select the first action using the largest number of cards."""

    policy_id: str = "longest_move"

    def __post_init__(self) -> None:
        _validate_policy_id(self.policy_id)

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        del context
        if not legal_actions:
            raise ValueError("LongestMovePolicy received no legal actions")
        if observation["phase"] != "card_play":
            return len(legal_actions) - 1
        return max(
            range(len(legal_actions)),
            key=lambda index: cast(PlayGameAction, legal_actions[index])["play"]["total_cards"],
        )


@dataclass(frozen=True, slots=True)
class FixedBidPolicy:
    """Compose a fixed, auditable bidder/doubler with any card-play policy."""

    policy_id: str
    cardplay: Policy
    score_bid: int = 1
    call: bool = True
    rob: bool = False
    double: bool = False

    def __post_init__(self) -> None:
        _validate_policy_id(self.policy_id)
        if not 1 <= self.score_bid <= 3:
            raise ValueError("score_bid must be in 1..=3")

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        if not legal_actions:
            raise ValueError("FixedBidPolicy received no legal actions")
        if observation["phase"] == "card_play":
            return self.cardplay.select_action(observation, legal_actions, context)
        if observation["phase"] == "doubling":
            desired: object = "double" if self.double else "decline"
            found = _find_phase_action(legal_actions, "double", desired)
            if found is None:
                raise RuntimeError("required doubling action disappeared")
            return found
        if observation["phase"] != "bidding":
            raise ValueError(f"FixedBidPolicy cannot act in phase {observation['phase']}")
        for desired in (
            {"score": self.score_bid},
            "rob" if self.rob else None,
            "call" if self.call else None,
            "pass",
        ):
            if desired is None:
                continue
            found = _find_phase_action(legal_actions, "bid", desired, required=False)
            if found is not None:
                return found
        raise ValueError("fixed bidder found no compatible legal bid")


@dataclass(frozen=True, slots=True)
class StagedPolicy:
    """Compose separate learned bidding and card-play policies for complete games."""

    policy_id: str
    bidding: Policy
    cardplay: Policy
    double: bool = False

    def __post_init__(self) -> None:
        _validate_policy_id(self.policy_id)

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        if not legal_actions:
            raise ValueError("StagedPolicy received no legal actions")
        if observation["phase"] == "bidding":
            return self.bidding.select_action(observation, legal_actions, context)
        if observation["phase"] == "card_play":
            return self.cardplay.select_action(observation, legal_actions, context)
        if observation["phase"] == "doubling":
            desired: object = "double" if self.double else "decline"
            selected = _find_phase_action(legal_actions, "double", desired)
            if selected is None:
                raise RuntimeError("required doubling action disappeared")
            return selected
        raise ValueError(f"StagedPolicy cannot act in phase {observation['phase']}")


@dataclass(frozen=True, slots=True)
class SeededRandomPolicy:
    """Stateless random baseline keyed by deal, seat, turn, and policy seed."""

    policy_id: str = "seeded_random"
    seed: int = 0

    def __post_init__(self) -> None:
        _validate_policy_id(self.policy_id)
        if not 0 <= self.seed < 1 << 64:
            raise ValueError("policy seed must fit uint64")

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        del observation
        if not legal_actions:
            raise ValueError("SeededRandomPolicy received no legal actions")
        key = self.seed ^ context.deal_seed
        key ^= context.seat << 8
        key ^= context.decision_index << 16
        key ^= context.deal_index << 48
        mixed = splitmix64(key & ((1 << 64) - 1))
        return mixed % len(legal_actions)


def make_builtin_policy(name: str, policy_id: str, seed: int = 0) -> Policy:
    """Construct one named built-in policy for CLI and smoke evaluation."""
    if name == "first_legal":
        return FirstLegalPolicy(policy_id)
    if name == "longest_move":
        return LongestMovePolicy(policy_id)
    if name == "seeded_random":
        return SeededRandomPolicy(policy_id, seed)
    raise ValueError(f"unknown built-in policy: {name}")


def _validate_policy_id(policy_id: str) -> None:
    if not policy_id or policy_id.strip() != policy_id:
        raise ValueError("policy_id must be non-empty without surrounding whitespace")


def _find_phase_action(
    legal_actions: Sequence[Action],
    phase_key: str,
    desired: object,
    *,
    required: bool = True,
) -> int | None:
    for index, action in enumerate(legal_actions):
        if action.get(phase_key) == desired:
            return index
    if required:
        raise ValueError(f"required {phase_key} action {desired!r} is not legal")
    return None


__all__ = (
    "FirstLegalPolicy",
    "FixedBidPolicy",
    "LongestMovePolicy",
    "Policy",
    "PolicyDecisionContext",
    "SeededRandomPolicy",
    "StagedPolicy",
    "make_builtin_policy",
)
