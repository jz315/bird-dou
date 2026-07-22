"""Seeded league matchmaking with explicit self/history/exploiter/baseline mixtures."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from birddou.eval.paired_deals import splitmix64
from birddou.league.population import (
    LeagueMember,
    LeagueMemberKind,
    LeaguePopulation,
    LeagueRole,
)

LEAGUE_MATCHMAKING_SCHEMA_VERSION = 1


class MatchCategory(StrEnum):
    """Predeclared source of the opponent in a league match."""

    SELF_PLAY = "self_play"
    HISTORICAL = "historical"
    EXPLOITER = "exploiter"
    FIXED_BASELINE = "fixed_baseline"


@dataclass(frozen=True, slots=True)
class LeagueMatchmakingConfig:
    """Versioned category mixture and deterministic deal seed."""

    schema_version: int = LEAGUE_MATCHMAKING_SCHEMA_VERSION
    self_play_weight: float = 0.65
    historical_weight: float = 0.20
    exploiter_weight: float = 0.10
    fixed_baseline_weight: float = 0.05
    seed: int = 20260722

    def __post_init__(self) -> None:
        if self.schema_version != LEAGUE_MATCHMAKING_SCHEMA_VERSION:
            raise ValueError("unsupported league matchmaking schema")
        weights = self.weights
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
            raise ValueError("league matchmaking weights must be finite and non-negative")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1.0e-9):
            raise ValueError("league matchmaking weights must sum to one")
        if self.self_play_weight <= 0.0:
            raise ValueError("league must retain positive current-model self-play")
        if not 0 <= self.seed < 1 << 64:
            raise ValueError("league matchmaking seed must fit uint64")

    @property
    def weights(self) -> dict[MatchCategory, float]:
        return {
            MatchCategory.SELF_PLAY: self.self_play_weight,
            MatchCategory.HISTORICAL: self.historical_weight,
            MatchCategory.EXPLOITER: self.exploiter_weight,
            MatchCategory.FIXED_BASELINE: self.fixed_baseline_weight,
        }

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class LeagueMatch:
    """One landlord policy, shared Farmer policy, deal seed, and audit category."""

    match_id: str
    category: MatchCategory
    landlord_policy_id: str
    farmer_policy_id: str
    deal_seed: int
    training_step: int
    draw_index: int

    def __post_init__(self) -> None:
        if not self.match_id or not self.landlord_policy_id or not self.farmer_policy_id:
            raise ValueError("league match IDs must be non-empty")
        if not 0 <= self.deal_seed < 1 << 64:
            raise ValueError("league match deal seed must fit uint64")
        if self.training_step < 0 or self.draw_index < 0:
            raise ValueError("league match step/index must be non-negative")


def load_league_matchmaking_config(path: Path) -> LeagueMatchmakingConfig:
    """Load a strict JSON-subset YAML league mixture."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = _mapping(raw, "league matchmaking config")
    return LeagueMatchmakingConfig(
        schema_version=_integer(values, "schema_version"),
        self_play_weight=_number(values, "self_play_weight"),
        historical_weight=_number(values, "historical_weight"),
        exploiter_weight=_number(values, "exploiter_weight"),
        fixed_baseline_weight=_number(values, "fixed_baseline_weight"),
        seed=_integer(values, "seed"),
    )


class LeagueMatchmaker:
    """Generate reproducible matches without silently substituting missing pools."""

    def __init__(self, config: LeagueMatchmakingConfig | None = None) -> None:
        self.config = config if config is not None else LeagueMatchmakingConfig()

    def schedule(
        self,
        population: LeaguePopulation,
        count: int,
        *,
        training_step: int,
    ) -> tuple[LeagueMatch, ...]:
        """Draw a deterministic schedule from every configured non-empty pool."""
        if count <= 0 or training_step < 0:
            raise ValueError("league match count must be positive and step non-negative")
        pools = _category_pools(population)
        for category, weight in self.config.weights.items():
            if weight > 0.0 and category is not MatchCategory.SELF_PLAY and not pools[category]:
                raise ValueError(f"configured league pool {category.value} is empty")
        rng = random.Random(splitmix64(self.config.seed ^ training_step))
        categories = tuple(self.config.weights)
        weights = tuple(self.config.weights[category] for category in categories)
        scheduled: list[LeagueMatch] = []
        for draw_index in range(count):
            category = rng.choices(categories, weights=weights, k=1)[0]
            landlord, farmer = self._assign(population, pools, category, rng)
            _validate_side(landlord, LeagueRole.LANDLORD)
            _validate_side(farmer, LeagueRole.FARMER)
            deal_seed = splitmix64(self.config.seed + training_step + draw_index)
            scheduled.append(
                LeagueMatch(
                    match_id=f"league-{training_step}-{draw_index}-{category.value}",
                    category=category,
                    landlord_policy_id=landlord.policy_id,
                    farmer_policy_id=farmer.policy_id,
                    deal_seed=deal_seed,
                    training_step=training_step,
                    draw_index=draw_index,
                )
            )
        return tuple(scheduled)

    @staticmethod
    def _assign(
        population: LeaguePopulation,
        pools: dict[MatchCategory, tuple[LeagueMember, ...]],
        category: MatchCategory,
        rng: random.Random,
    ) -> tuple[LeagueMember, LeagueMember]:
        champion = population.champion
        if category is MatchCategory.SELF_PLAY:
            return champion, champion
        opponent = rng.choice(pools[category])
        if opponent.role is LeagueRole.LANDLORD:
            return opponent, champion
        if opponent.role is LeagueRole.FARMER:
            return champion, opponent
        return (opponent, champion) if rng.randrange(2) == 0 else (champion, opponent)


def _category_pools(
    population: LeaguePopulation,
) -> dict[MatchCategory, tuple[LeagueMember, ...]]:
    historical = tuple(
        member
        for member in population.members
        if member.active and member.kind is LeagueMemberKind.HISTORICAL_MAIN
    )
    exploiters = tuple(
        member
        for member in population.members
        if member.active
        and member.kind in (LeagueMemberKind.LANDLORD_EXPLOITER, LeagueMemberKind.FARMER_EXPLOITER)
    )
    baselines = tuple(
        member
        for member in population.members
        if member.active and member.kind is LeagueMemberKind.FIXED_BASELINE
    )
    return {
        MatchCategory.SELF_PLAY: (population.champion,),
        MatchCategory.HISTORICAL: historical,
        MatchCategory.EXPLOITER: exploiters,
        MatchCategory.FIXED_BASELINE: baselines,
    }


def _validate_side(member: LeagueMember, role: LeagueRole) -> None:
    if not member.active or member.role not in (LeagueRole.BOTH, role):
        raise ValueError(f"league member {member.policy_id} cannot play {role.value}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"league config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"league config {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"league config {key} must be finite")
    return number


__all__ = (
    "LEAGUE_MATCHMAKING_SCHEMA_VERSION",
    "LeagueMatch",
    "LeagueMatchmaker",
    "LeagueMatchmakingConfig",
    "MatchCategory",
    "load_league_matchmaking_config",
)
