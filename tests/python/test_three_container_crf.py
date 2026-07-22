"""Exactness and conservation tests for bidding-time three-container belief."""

from itertools import product

import pytest
import torch

from birddou.belief.three_container_crf import (
    sample_three_container_allocations,
    three_container_log_partition,
    three_container_marginals,
    three_container_nll,
    validate_three_container_assignment,
)


def _small_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.zeros(1, 15, 5, 5, dtype=torch.float64)
    scores[0, 0, :3, :3] = torch.tensor(
        [[0.0, -0.2, 0.4], [0.1, 0.3, -0.5], [-0.1, 0.2, 0.7]],
        dtype=torch.float64,
    )
    scores[0, 1, :2, :2] = torch.tensor([[0.2, -0.4], [0.6, 0.1]], dtype=torch.float64)
    unknown = torch.zeros(1, 15, dtype=torch.int64)
    unknown[0, :2] = torch.tensor([2, 1])
    return scores, unknown, torch.tensor([1]), torch.tensor([1])


def test_partition_and_allocation_marginals_match_brute_force() -> None:
    scores, unknown, capacity_a, capacity_b = _small_problem()
    allocations: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for first, second in product(
        ((a, b) for a in range(3) for b in range(3 - a)),
        ((a, b) for a in range(2) for b in range(2 - a)),
    ):
        if first[0] + second[0] == 1 and first[1] + second[1] == 1:
            value = float(scores[0, 0, first[0], first[1]] + scores[0, 1, second[0], second[1]])
            allocations.append((first, second, value))
    brute_scores = torch.tensor([item[2] for item in allocations], dtype=torch.float64)
    brute_probability = torch.softmax(brute_scores, dim=0)
    marginals = three_container_marginals(scores, unknown, capacity_a, capacity_b)

    assert torch.allclose(marginals.log_partition, torch.logsumexp(brute_scores, dim=0)[None])
    for rank in range(2):
        for count_a in range(3):
            for count_b in range(3):
                expected = sum(
                    float(probability)
                    for allocation, probability in zip(allocations, brute_probability, strict=True)
                    if allocation[rank] == (count_a, count_b)
                )
                actual = marginals.allocation_probability[0, rank, count_a, count_b].item()
                assert actual == pytest.approx(expected)
    assert torch.allclose(marginals.expected.sum(dim=1), marginals.capacities.to(torch.float64))


def test_nll_is_differentiable_and_validates_all_three_containers() -> None:
    scores, unknown, capacity_a, capacity_b = _small_problem()
    scores.requires_grad_(True)
    assignment_a = torch.zeros(1, 15, dtype=torch.int64)
    assignment_b = torch.zeros(1, 15, dtype=torch.int64)
    assignment_a[0, 0] = 1
    assignment_b[0, 1] = 1
    loss = three_container_nll(scores, unknown, capacity_a, capacity_b, assignment_a, assignment_b)
    (gradient,) = torch.autograd.grad(loss, scores)
    assert loss.item() > 0.0
    assert torch.isfinite(gradient).all()

    assignment_b[0, 0] = 2
    with pytest.raises(ValueError, match="exceeds"):
        validate_three_container_assignment(
            assignment_a, assignment_b, unknown, capacity_a, capacity_b
        )


def test_exact_sampler_preserves_rank_and_capacity_at_extremes() -> None:
    scores = torch.zeros(2, 15, 5, 5)
    unknown = torch.zeros(2, 15, dtype=torch.int64)
    unknown[:, :5] = torch.tensor([4, 4, 4, 4, 4])
    capacity_a = torch.tensor([0, 17])
    capacity_b = torch.tensor([17, 0])
    generator = torch.Generator().manual_seed(20260722)
    allocation_a, allocation_b, bottom = sample_three_container_allocations(
        scores, unknown, capacity_a, capacity_b, generator=generator
    )

    assert torch.equal(allocation_a + allocation_b + bottom, unknown)
    assert torch.equal(allocation_a.sum(dim=1), capacity_a)
    assert torch.equal(allocation_b.sum(dim=1), capacity_b)
    assert torch.equal(bottom.sum(dim=1), torch.tensor([3, 3]))
    assert torch.isfinite(
        three_container_log_partition(scores, unknown, capacity_a, capacity_b)
    ).all()
