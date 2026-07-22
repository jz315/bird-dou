"""League lifecycle, matchmaking, snapshot/resume, and promotion-gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from birddou.league import (
    LeagueMatchmaker,
    LeagueMember,
    LeagueMemberKind,
    LeaguePopulation,
    LeagueRole,
    LeagueSnapshot,
    MatchCategory,
    PromotionMetrics,
    evaluate_promotion,
    load_league_matchmaking_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "train" / "league.yaml"


def _member(
    policy_id: str,
    kind: LeagueMemberKind,
    role: LeagueRole = LeagueRole.BOTH,
    *,
    version: int = 0,
) -> LeagueMember:
    return LeagueMember(
        policy_id=policy_id,
        kind=kind,
        role=role,
        checkpoint=None if kind is LeagueMemberKind.FIXED_BASELINE else f"{policy_id}.pt",
        policy_version=version,
        created_step=version,
    )


def _population() -> LeaguePopulation:
    population = LeaguePopulation.create(_member("main-v2", LeagueMemberKind.CURRENT_MAIN))
    population.add(_member("main-v1", LeagueMemberKind.HISTORICAL_MAIN))
    population.add(
        _member(
            "landlord-exploiter",
            LeagueMemberKind.LANDLORD_EXPLOITER,
            LeagueRole.LANDLORD,
        )
    )
    population.add(
        _member(
            "farmer-exploiter",
            LeagueMemberKind.FARMER_EXPLOITER,
            LeagueRole.FARMER,
        )
    )
    population.add(_member("douzero-adp", LeagueMemberKind.FIXED_BASELINE))
    return population


def test_population_promotion_archives_champion_without_overwriting_artifacts() -> None:
    population = _population()
    population.set_rating("main-v2", 1.25)
    population.record_games(("main-v2", "douzero-adp", "main-v2"), 3)
    assert population.champion.games_played == 3

    population.promote(_member("main-v3", LeagueMemberKind.CURRENT_MAIN, version=3))
    assert population.champion_id == "main-v3"
    archived = population.get("main-v2")
    assert archived.kind is LeagueMemberKind.HISTORICAL_MAIN
    assert archived.rating == pytest.approx(1.25)
    assert archived.checkpoint == "main-v2.pt"
    with pytest.raises(ValueError, match="deactivated"):
        population.set_active("main-v3", False)


def test_matchmaking_is_seeded_covers_all_pools_and_respects_role_specialists() -> None:
    population = _population()
    config = load_league_matchmaking_config(CONFIG_PATH)
    matchmaker = LeagueMatchmaker(config)
    first = matchmaker.schedule(population, 400, training_step=71)
    second = matchmaker.schedule(population, 400, training_step=71)

    assert first == second
    assert {match.category for match in first} == set(MatchCategory)
    assert len({match.deal_seed for match in first}) == len(first)
    for match in first:
        if match.landlord_policy_id == "landlord-exploiter":
            assert match.farmer_policy_id == population.champion_id
        if match.farmer_policy_id == "farmer-exploiter":
            assert match.landlord_policy_id == population.champion_id

    missing = LeaguePopulation.create(_member("only-main", LeagueMemberKind.CURRENT_MAIN))
    with pytest.raises(ValueError, match="historical"):
        matchmaker.schedule(missing, 1, training_step=0)


def test_snapshot_round_trip_is_hash_stable_and_checksum_protected(tmp_path: Path) -> None:
    snapshot = LeagueSnapshot(
        schema_version=1,
        population=_population(),
        matchmaking=load_league_matchmaking_config(CONFIG_PATH),
        schedule_cursor=123,
        last_promotion_step=100,
    )
    destination = tmp_path / "league.json"
    file_digest = snapshot.save(destination)
    restored = LeagueSnapshot.load(destination, expected_sha256=file_digest)

    assert restored.to_dict() == snapshot.to_dict()
    assert restored.fingerprint() == snapshot.fingerprint()
    destination.write_bytes(destination.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        LeagueSnapshot.load(destination, expected_sha256=file_digest)


def test_promotion_gate_requires_positive_overall_role_safety_and_calibration() -> None:
    accepted = evaluate_promotion(
        PromotionMetrics(0.01, 0.0, -0.01, 0.0, 0.05, 0.055, True, 10_000)
    )
    rejected = evaluate_promotion(PromotionMetrics(0.0, -0.03, 0.0, 0.0, 0.05, 0.08, False, 10_000))
    assert accepted.accepted
    assert not rejected.accepted
    assert len(rejected.reasons) == 4
