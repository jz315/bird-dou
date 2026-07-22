"""Unit coverage for the source-pinned differential harness."""

from pathlib import Path

import pytest

from birddou.eval.douzero_differential import (
    DifferentialError,
    deal_from_seed,
    douzero_cards_to_rank_counts,
    load_manifest,
    rank_counts_to_douzero_cards,
    validate_baseline_source,
)


def test_rank_conversion_round_trips_the_complete_douzero_deck() -> None:
    deck = [rank for rank in range(3, 15) for _ in range(4)]
    deck.extend([17] * 4)
    deck.extend([20, 30])

    counts = douzero_cards_to_rank_counts(deck)
    assert rank_counts_to_douzero_cards(counts) == sorted(deck)
    assert sum(counts) == 54


def test_seeded_deals_are_reproducible_and_partition_the_deck() -> None:
    first = deal_from_seed(17)
    second = deal_from_seed(17)

    assert first == second
    assert tuple(sum(hand) for hand in first.hands) == (20, 17, 17)
    assert sum(first.bottom_cards) == 3
    assert all(
        sum(hand[rank] for hand in first.hands) == (4 if rank <= 12 else 1) for rank in range(15)
    )


def test_manifest_is_pinned_and_missing_source_is_an_explicit_error(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(repository_root / "artifacts/baselines/douzero/manifest.toml")

    assert manifest.commit == "718a5c920bf3361e34178a38f3b80458e176b351"
    assert manifest.license == "Apache-2.0"
    assert manifest.weights_required
    assert "douzero/dmc/models.py" in manifest.required_files
    with pytest.raises(DifferentialError, match="source is absent"):
        validate_baseline_source(tmp_path / "missing", manifest)
