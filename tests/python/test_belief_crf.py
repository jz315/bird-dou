"""Exactness, conservation, sampling, learning, and calibration tests for M5."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from birddou.belief import (
    belief_nll,
    calibration_report,
    cardinality_marginals,
    log_partition,
    sample_hidden_allocations,
    uniform_belief_nll,
    validate_assignment,
)


def small_constraints() -> tuple[torch.Tensor, torch.Tensor]:
    unknown = torch.zeros(1, 15, dtype=torch.int64)
    unknown[0, :3] = torch.tensor([2, 1, 2])
    return unknown, torch.tensor([2], dtype=torch.int64)


def brute_allocations(unknown: torch.Tensor, capacity: int) -> list[tuple[int, ...]]:
    ranges = [range(int(count) + 1) for count in unknown.tolist()]
    return [values for values in itertools.product(*ranges) if sum(values) == capacity]


def test_dynamic_program_and_marginals_match_brute_force() -> None:
    """Small-card exact enumeration agrees with both DP directions."""
    torch.manual_seed(5005)
    unknown, capacity = small_constraints()
    scores = torch.randn(1, 15, 5, dtype=torch.float64, requires_grad=True)
    allocations = brute_allocations(unknown[0], int(capacity[0]))
    ranks = torch.arange(15)
    brute_scores = torch.stack(
        [scores[0, ranks, torch.tensor(values)].sum() for values in allocations]
    )
    expected_log_z = torch.logsumexp(brute_scores, dim=0)
    actual_log_z = log_partition(scores, unknown, capacity)[0]
    torch.testing.assert_close(actual_log_z, expected_log_z, rtol=1e-12, atol=1e-12)

    weights = torch.softmax(brute_scores, dim=0)
    expected_probability = torch.zeros(15, 5, dtype=torch.float64)
    for weight, values in zip(weights, allocations, strict=True):
        for rank, count in enumerate(values):
            expected_probability[rank, count] += weight
    marginals = cardinality_marginals(scores, unknown, capacity)
    torch.testing.assert_close(
        marginals.probability_a[0], expected_probability, rtol=1e-11, atol=1e-12
    )
    torch.testing.assert_close(
        marginals.expected_a.sum(), capacity.to(torch.float64).sum(), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        marginals.expected_a + marginals.expected_b,
        unknown.to(torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.autograd.backward((actual_log_z,))
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_sampler_has_zero_violations_and_matches_exact_marginals() -> None:
    """Backward sampling preserves every card and converges to the DP distribution."""
    unknown, capacity = small_constraints()
    scores = torch.randn(1, 15, 5)
    exact = cardinality_marginals(scores, unknown, capacity)
    generator = torch.Generator().manual_seed(5006)
    samples = sample_hidden_allocations(scores, unknown, capacity, 20_000, generator=generator)

    assert samples.shape == (1, 20_000, 15)
    assert torch.equal(samples.sum(dim=-1), capacity[:, None].expand(1, 20_000))
    assert torch.all(samples <= unknown[:, None])
    for rank in range(3):
        empirical = torch.stack(
            [(samples[0, :, rank] == count).float().mean() for count in range(5)]
        )
        torch.testing.assert_close(
            empirical,
            exact.probability_a[0, rank],
            rtol=0.0,
            atol=0.015,
        )


def test_bomb_and_key_card_probabilities_respect_forced_allocations() -> None:
    """Exact key summaries identify forced bombs, twos, and jokers for either side."""
    unknown = torch.zeros(2, 15, dtype=torch.int64)
    unknown[:, 0] = 4
    unknown[:, 12:] = 1
    capacity = torch.tensor([7, 0], dtype=torch.int64)
    scores = torch.zeros(2, 15, 5)
    marginals = cardinality_marginals(scores, unknown, capacity)

    torch.testing.assert_close(marginals.key_probability_a[0], torch.ones(4))
    torch.testing.assert_close(marginals.key_probability_b[0], torch.zeros(4))
    torch.testing.assert_close(marginals.key_probability_a[1], torch.zeros(4))
    torch.testing.assert_close(marginals.key_probability_b[1], torch.ones(4))


def test_supervised_nll_learns_better_than_uniform_and_stays_finite() -> None:
    """A labeled allocation is learnable under the hard capacity constraint."""
    unknown, capacity = small_constraints()
    label = torch.zeros(1, 15, dtype=torch.int64)
    label[0, :3] = torch.tensor([2, 0, 0])
    scores = torch.nn.Parameter(torch.zeros(1, 15, 5))
    optimizer = torch.optim.Adam((scores,), lr=0.15)
    uniform = float(uniform_belief_nll(unknown, capacity).item())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = belief_nll(scores, unknown, capacity, label)
        torch.autograd.backward((loss,))
        optimizer.step()
    learned = float(belief_nll(scores, unknown, capacity, label).item())

    assert math.isfinite(learned)
    assert learned < uniform - 1.0
    validate_assignment(label, unknown, capacity)


def test_calibration_and_invalid_labels_have_explicit_boundaries() -> None:
    """Reliability metrics are auditable and conservation errors never clip silently."""
    probability = torch.tensor([0.0, 0.25, 0.75, 1.0])
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    report = calibration_report(probability, target, bin_count=4)
    assert report.brier_score == pytest.approx(0.03125)
    assert report.expected_calibration_error == pytest.approx(0.125)
    assert sum(item.count for item in report.bins) == 4

    unknown, capacity = small_constraints()
    invalid = unknown.clone()
    with pytest.raises(ValueError, match="capacity"):
        validate_assignment(invalid, unknown, capacity)
    with pytest.raises(ValueError, match="joker"):
        bad_unknown = unknown.clone()
        bad_unknown[:, 14] = 2
        log_partition(torch.zeros(1, 15, 5), bad_unknown, capacity)
