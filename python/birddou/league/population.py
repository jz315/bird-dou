"""Validated mutable population for self-play, history, exploiters, and baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

LEAGUE_POPULATION_SCHEMA_VERSION = 1


class LeagueMemberKind(StrEnum):
    """Lifecycle or specialization of one immutable policy artifact."""

    CURRENT_MAIN = "current_main"
    HISTORICAL_MAIN = "historical_main"
    LANDLORD_EXPLOITER = "landlord_exploiter"
    FARMER_EXPLOITER = "farmer_exploiter"
    FIXED_BASELINE = "fixed_baseline"


class LeagueRole(StrEnum):
    """Sides on which a member is eligible to play."""

    BOTH = "both"
    LANDLORD = "landlord"
    FARMER = "farmer"


@dataclass(frozen=True, slots=True)
class LeagueMember:
    """One policy identity and its reproducible artifact/monitoring metadata."""

    policy_id: str
    kind: LeagueMemberKind
    role: LeagueRole
    checkpoint: str | None
    policy_version: int
    created_step: int
    active: bool = True
    games_played: int = 0
    rating: float = 0.0

    def __post_init__(self) -> None:
        if not self.policy_id or self.policy_id.strip() != self.policy_id:
            raise ValueError("league policy_id must be non-empty and trimmed")
        if self.checkpoint is not None and not self.checkpoint:
            raise ValueError("league checkpoint must be null or non-empty")
        if self.kind is not LeagueMemberKind.FIXED_BASELINE and self.checkpoint is None:
            raise ValueError("trainable league members require an immutable checkpoint")
        if self.policy_version < 0 or self.created_step < 0 or self.games_played < 0:
            raise ValueError("league version, step, and game count must be non-negative")
        if not math.isfinite(self.rating):
            raise ValueError("league rating must be finite")
        if (
            self.kind is LeagueMemberKind.LANDLORD_EXPLOITER
            and self.role is not LeagueRole.LANDLORD
        ):
            raise ValueError("landlord exploiter must be landlord-only")
        if self.kind is LeagueMemberKind.FARMER_EXPLOITER and self.role is not LeagueRole.FARMER:
            raise ValueError("farmer exploiter must be farmer-only")
        if self.kind in (LeagueMemberKind.CURRENT_MAIN, LeagueMemberKind.HISTORICAL_MAIN):
            if self.role is not LeagueRole.BOTH:
                raise ValueError("main policies must support both sides")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible record."""
        return {
            "policy_id": self.policy_id,
            "kind": self.kind.value,
            "role": self.role.value,
            "checkpoint": self.checkpoint,
            "policy_version": self.policy_version,
            "created_step": self.created_step,
            "active": self.active,
            "games_played": self.games_played,
            "rating": self.rating,
        }

    @classmethod
    def from_dict(cls, value: object) -> LeagueMember:
        """Strictly decode a serialized member record."""
        if not isinstance(value, dict) or set(value) != {
            "policy_id",
            "kind",
            "role",
            "checkpoint",
            "policy_version",
            "created_step",
            "active",
            "games_played",
            "rating",
        }:
            raise ValueError("league member fields are incomplete or unknown")
        try:
            return cls(
                policy_id=_string(value, "policy_id"),
                kind=LeagueMemberKind(_string(value, "kind")),
                role=LeagueRole(_string(value, "role")),
                checkpoint=_optional_string(value, "checkpoint"),
                policy_version=_integer(value, "policy_version"),
                created_step=_integer(value, "created_step"),
                active=_boolean(value, "active"),
                games_played=_integer(value, "games_played"),
                rating=_number(value, "rating"),
            )
        except ValueError as error:
            raise ValueError(f"invalid league member: {error}") from error


class LeaguePopulation:
    """Own the unique champion and every immutable opponent snapshot."""

    def __init__(self, members: tuple[LeagueMember, ...], champion_id: str) -> None:
        if not members:
            raise ValueError("league population requires at least one member")
        self._members: dict[str, LeagueMember] = {}
        for member in members:
            if member.policy_id in self._members:
                raise ValueError(f"duplicate league member: {member.policy_id}")
            self._members[member.policy_id] = member
        self._champion_id = champion_id
        self._validate_champion()

    @classmethod
    def create(cls, champion: LeagueMember) -> LeaguePopulation:
        """Create a population whose first member is the current main policy."""
        if champion.kind is not LeagueMemberKind.CURRENT_MAIN or not champion.active:
            raise ValueError("initial champion must be an active current_main")
        return cls((champion,), champion.policy_id)

    @property
    def champion_id(self) -> str:
        return self._champion_id

    @property
    def champion(self) -> LeagueMember:
        return self._members[self._champion_id]

    @property
    def members(self) -> tuple[LeagueMember, ...]:
        """Return members in stable policy-ID order."""
        return tuple(self._members[key] for key in sorted(self._members))

    def add(self, member: LeagueMember) -> None:
        """Add an immutable opponent; current-main changes require ``promote``."""
        if member.policy_id in self._members:
            raise ValueError(f"duplicate league member: {member.policy_id}")
        if member.kind is LeagueMemberKind.CURRENT_MAIN:
            raise ValueError("use promote to replace the current main")
        self._members[member.policy_id] = member

    def get(self, policy_id: str) -> LeagueMember:
        """Resolve one member or fail instead of silently substituting a policy."""
        try:
            return self._members[policy_id]
        except KeyError as error:
            raise KeyError(f"unknown league policy: {policy_id}") from error

    def eligible(
        self,
        role: LeagueRole,
        *,
        kinds: tuple[LeagueMemberKind, ...] | None = None,
        include_champion: bool = True,
    ) -> tuple[LeagueMember, ...]:
        """Return active members compatible with one game side."""
        if role is LeagueRole.BOTH:
            raise ValueError("match eligibility must request landlord or farmer")
        result = tuple(
            member
            for member in self.members
            if member.active
            and (include_champion or member.policy_id != self._champion_id)
            and member.role in (LeagueRole.BOTH, role)
            and (kinds is None or member.kind in kinds)
        )
        return result

    def set_active(self, policy_id: str, active: bool) -> None:
        """Activate/deactivate an opponent while keeping the champion active."""
        if policy_id == self._champion_id and not active:
            raise ValueError("league champion cannot be deactivated")
        self._members[policy_id] = replace(self.get(policy_id), active=active)

    def record_games(self, policy_ids: tuple[str, ...], count: int = 1) -> None:
        """Increment bounded scalar usage statistics for scheduled policies."""
        if count <= 0:
            raise ValueError("league game increment must be positive")
        for policy_id in set(policy_ids):
            member = self.get(policy_id)
            self._members[policy_id] = replace(member, games_played=member.games_played + count)

    def set_rating(self, policy_id: str, rating: float) -> None:
        """Replace a finite externally computed rating."""
        if not math.isfinite(rating):
            raise ValueError("league rating must be finite")
        self._members[policy_id] = replace(self.get(policy_id), rating=rating)

    def update_champion_runtime(
        self,
        *,
        checkpoint: str,
        policy_version: int,
    ) -> None:
        """Checkpoint the live champion version without changing its policy identity."""
        if not checkpoint or policy_version < self.champion.policy_version:
            raise ValueError("champion checkpoint must be non-empty and version monotone")
        self._members[self._champion_id] = replace(
            self.champion,
            checkpoint=checkpoint,
            policy_version=policy_version,
        )

    def promote(self, candidate: LeagueMember) -> None:
        """Archive the old champion and atomically install a new current main."""
        if candidate.kind is not LeagueMemberKind.CURRENT_MAIN or not candidate.active:
            raise ValueError("promoted candidate must be an active current_main")
        if candidate.policy_id in self._members:
            raise ValueError("promoted candidate policy_id must be new")
        previous = self.champion
        self._members[previous.policy_id] = replace(
            previous,
            kind=LeagueMemberKind.HISTORICAL_MAIN,
            active=True,
        )
        self._members[candidate.policy_id] = candidate
        self._champion_id = candidate.policy_id
        self._validate_champion()

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete restorable population state."""
        return {
            "schema_version": LEAGUE_POPULATION_SCHEMA_VERSION,
            "champion_id": self._champion_id,
            "members": [member.to_dict() for member in self.members],
        }

    @classmethod
    def from_dict(cls, value: object) -> LeaguePopulation:
        """Restore a strict population snapshot."""
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "champion_id",
            "members",
        }:
            raise ValueError("league population fields are incomplete or unknown")
        if _integer(value, "schema_version") != LEAGUE_POPULATION_SCHEMA_VERSION:
            raise ValueError("unsupported league population schema")
        raw_members = value.get("members")
        if not isinstance(raw_members, list):
            raise ValueError("league population members must be a list")
        return cls(
            tuple(LeagueMember.from_dict(member) for member in raw_members),
            _string(value, "champion_id"),
        )

    def _validate_champion(self) -> None:
        if self._champion_id not in self._members:
            raise ValueError("league champion is absent from population")
        current = [
            member
            for member in self._members.values()
            if member.kind is LeagueMemberKind.CURRENT_MAIN
        ]
        if len(current) != 1 or current[0].policy_id != self._champion_id or not current[0].active:
            raise ValueError("league must have exactly one active current_main champion")


def _string(values: dict[object, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(values: dict[object, object], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _integer(values: dict[object, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(values: dict[object, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _boolean(values: dict[object, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


__all__ = (
    "LEAGUE_POPULATION_SCHEMA_VERSION",
    "LeagueMember",
    "LeagueMemberKind",
    "LeaguePopulation",
    "LeagueRole",
)
