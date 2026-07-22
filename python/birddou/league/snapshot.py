"""Hash-stable league snapshots and confidence-aware champion promotion."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from birddou.league.matchmaking import LeagueMatchmakingConfig
from birddou.league.population import (
    LeagueMember,
    LeagueMemberKind,
    LeaguePopulation,
    LeagueRole,
)

LEAGUE_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LeagueSnapshot:
    """Complete league state required for an exact training resume."""

    schema_version: int
    population: LeaguePopulation
    matchmaking: LeagueMatchmakingConfig
    schedule_cursor: int
    last_promotion_step: int

    def __post_init__(self) -> None:
        if self.schema_version != LEAGUE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported league snapshot schema")
        if self.schedule_cursor < 0 or self.last_promotion_step < 0:
            raise ValueError("league snapshot cursor/step must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible snapshot payload."""
        return {
            "schema_version": self.schema_version,
            "population": self.population.to_dict(),
            "matchmaking": self.matchmaking.to_dict(),
            "schedule_cursor": self.schedule_cursor,
            "last_promotion_step": self.last_promotion_step,
        }

    def canonical_bytes(self) -> bytes:
        """Encode without whitespace or key-order ambiguity."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def fingerprint(self) -> str:
        """SHA-256 identity suitable for checkpoint manifests."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: Path) -> str:
        """Atomically save a human-readable snapshot and return its SHA-256."""
        destination = path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return hashlib.sha256(destination.read_bytes()).hexdigest()

    def with_runtime_progress(
        self,
        *,
        checkpoint: str,
        policy_version: int,
        schedule_cursor: int,
    ) -> LeagueSnapshot:
        """Return an updated, independently owned snapshot for a training checkpoint."""
        population = LeaguePopulation.from_dict(self.population.to_dict())
        population.update_champion_runtime(
            checkpoint=checkpoint,
            policy_version=policy_version,
        )
        return LeagueSnapshot(
            schema_version=self.schema_version,
            population=population,
            matchmaking=self.matchmaking,
            schedule_cursor=schedule_cursor,
            last_promotion_step=self.last_promotion_step,
        )

    @classmethod
    def from_dict(cls, value: object) -> LeagueSnapshot:
        """Strictly restore every population and matchmaking field."""
        root = _mapping(value, "league snapshot")
        if set(root) != {
            "schema_version",
            "population",
            "matchmaking",
            "schedule_cursor",
            "last_promotion_step",
        }:
            raise ValueError("league snapshot fields are incomplete or unknown")
        matchmaking = _mapping(root.get("matchmaking"), "league matchmaking")
        return cls(
            schema_version=_integer(root, "schema_version"),
            population=LeaguePopulation.from_dict(root.get("population")),
            matchmaking=LeagueMatchmakingConfig(
                schema_version=_integer(matchmaking, "schema_version"),
                self_play_weight=_number(matchmaking, "self_play_weight"),
                historical_weight=_number(matchmaking, "historical_weight"),
                exploiter_weight=_number(matchmaking, "exploiter_weight"),
                fixed_baseline_weight=_number(matchmaking, "fixed_baseline_weight"),
                seed=_integer(matchmaking, "seed"),
            ),
            schedule_cursor=_integer(root, "schedule_cursor"),
            last_promotion_step=_integer(root, "last_promotion_step"),
        )

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str | None = None) -> LeagueSnapshot:
        """Verify an optional file checksum and restore a snapshot."""
        source = path.resolve()
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("league snapshot checksum mismatch")
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Predeclared strength, role-safety, and calibration requirements."""

    min_overall_paired_ci_lower: float = 0.0
    min_role_paired_ci_lower: float = -0.02
    max_belief_ece_increase: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.min_overall_paired_ci_lower,
            self.min_role_paired_ci_lower,
            self.max_belief_ece_increase,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("league promotion thresholds must be finite")
        if self.max_belief_ece_increase < 0.0:
            raise ValueError("maximum Belief ECE increase must be non-negative")


@dataclass(frozen=True, slots=True)
class PromotionMetrics:
    """Independent paired evaluation and stability evidence for one candidate."""

    overall_paired_ci_lower: float
    landlord_paired_ci_lower: float
    landlord_down_paired_ci_lower: float
    landlord_up_paired_ci_lower: float
    belief_ece_before: float
    belief_ece_after: float
    numerical_stable: bool
    completed_deals: int

    def __post_init__(self) -> None:
        values = (
            self.overall_paired_ci_lower,
            self.landlord_paired_ci_lower,
            self.landlord_down_paired_ci_lower,
            self.landlord_up_paired_ci_lower,
            self.belief_ece_before,
            self.belief_ece_after,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("league promotion metrics must be finite")
        if self.belief_ece_before < 0.0 or self.belief_ece_after < 0.0:
            raise ValueError("Belief ECE cannot be negative")
        if self.completed_deals <= 0:
            raise ValueError("league promotion requires completed paired deals")


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """Auditable decision; population mutation occurs only after acceptance."""

    accepted: bool
    belief_ece_increase: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def evaluate_promotion(
    metrics: PromotionMetrics,
    thresholds: PromotionThresholds | None = None,
) -> PromotionReport:
    """Require champion improvement without role collapse or numerical/calibration drift."""
    settings = thresholds if thresholds is not None else PromotionThresholds()
    reasons: list[str] = []
    if metrics.overall_paired_ci_lower <= settings.min_overall_paired_ci_lower:
        reasons.append("overall paired lower confidence bound did not clear promotion")
    role_bounds = (
        metrics.landlord_paired_ci_lower,
        metrics.landlord_down_paired_ci_lower,
        metrics.landlord_up_paired_ci_lower,
    )
    if any(value < settings.min_role_paired_ci_lower for value in role_bounds):
        reasons.append("at least one role regressed beyond its safety floor")
    ece_increase = metrics.belief_ece_after - metrics.belief_ece_before
    if ece_increase > settings.max_belief_ece_increase:
        reasons.append("Belief calibration regressed beyond its safety threshold")
    if not metrics.numerical_stable:
        reasons.append("candidate reported numerical instability")
    return PromotionReport(not reasons, ece_increase, tuple(reasons))


def create_self_play_snapshot(
    policy_id: str,
    checkpoint: str,
    *,
    seed: int,
) -> LeagueSnapshot:
    """Create the exact self-play-only league used by pre-League smoke trainers."""
    population = LeaguePopulation.create(
        LeagueMember(
            policy_id=policy_id,
            kind=LeagueMemberKind.CURRENT_MAIN,
            role=LeagueRole.BOTH,
            checkpoint=checkpoint,
            policy_version=0,
            created_step=0,
        )
    )
    return LeagueSnapshot(
        schema_version=LEAGUE_SNAPSHOT_SCHEMA_VERSION,
        population=population,
        matchmaking=LeagueMatchmakingConfig(
            self_play_weight=1.0,
            historical_weight=0.0,
            exploiter_weight=0.0,
            fixed_baseline_weight=0.0,
            seed=seed,
        ),
        schedule_cursor=0,
        last_promotion_step=0,
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"league snapshot {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"league snapshot {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"league snapshot {key} must be finite")
    return number


__all__ = (
    "LEAGUE_SNAPSHOT_SCHEMA_VERSION",
    "LeagueSnapshot",
    "PromotionMetrics",
    "PromotionReport",
    "PromotionThresholds",
    "create_self_play_snapshot",
    "evaluate_promotion",
)
