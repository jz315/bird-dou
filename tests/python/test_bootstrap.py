"""Deterministic paired-bootstrap acceptance tests for E012."""

import math

import pytest

from birddou.eval.bootstrap import (
    BootstrapConfig,
    bootstrap_mean_ci,
    bootstrap_paired_difference_ci,
)


def test_bootstrap_is_seeded_and_chunk_size_invariant() -> None:
    """Memory chunking cannot change a formal seeded result."""
    values = [1.0, -2.0, 3.0, 4.0, 0.5]
    small = bootstrap_mean_ci(
        values,
        BootstrapConfig(resamples=1_000, seed=71, max_chunk_elements=5),
    )
    large = bootstrap_mean_ci(
        values,
        BootstrapConfig(resamples=1_000, seed=71, max_chunk_elements=100_000),
    )

    assert small == large
    assert small.point_estimate == pytest.approx(1.3)
    assert small.lower <= small.point_estimate <= small.upper
    assert small.width == small.upper - small.lower


def test_paired_difference_resamples_within_deal_deltas() -> None:
    """Identical within-deal shifts have an exact zero-width paired CI."""
    result = bootstrap_paired_difference_ci(
        [3.0, 5.0, -1.0],
        [1.0, 3.0, -3.0],
        BootstrapConfig(resamples=500, seed=8),
    )

    assert result.point_estimate == 2.0
    assert result.lower == 2.0
    assert result.upper == 2.0
    assert result.standard_error == 0.0


def test_bootstrap_rejects_invalid_samples_and_controls() -> None:
    """Malformed uncertainty inputs fail instead of producing misleading output."""
    with pytest.raises(ValueError, match="same number"):
        bootstrap_paired_difference_ci([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        bootstrap_mean_ci([1.0, math.nan])
    with pytest.raises(ValueError, match="confidence_level"):
        BootstrapConfig(confidence_level=1.0)
    with pytest.raises(ValueError, match="at least 100"):
        BootstrapConfig(resamples=99)
