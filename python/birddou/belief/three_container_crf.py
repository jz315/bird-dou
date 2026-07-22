"""Exact three-container cardinality CRF used before bottom-card assignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor

from .cardinality_crf import BELIEF_RANK_COUNT, MAX_RANK_COPIES

THREE_CONTAINER_SCHEMA_VERSION = 1
_NEGATIVE_INFINITY = -1.0e30


@dataclass(frozen=True, slots=True)
class ThreeContainerMarginals:
    """Exact allocation and per-container rank-count marginals."""

    log_partition: Tensor
    allocation_probability: Tensor
    probability: Tensor
    expected: Tensor
    variance: Tensor
    entropy: Tensor
    capacities: Tensor


@dataclass(frozen=True, slots=True)
class _DynamicProgram:
    log_partition: Tensor
    forward: tuple[Tensor, ...]
    backward: tuple[Tensor, ...]


def three_container_log_partition(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    capacity_b: Tensor,
) -> Tensor:
    """Compute exact log-normalizers for A/B capacities; bottom capacity is inferred."""
    _validate_inputs(scores, unknown_counts, capacity_a, capacity_b)
    return _dynamic_program(scores, unknown_counts, capacity_a, capacity_b).log_partition


def three_container_marginals(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    capacity_b: Tensor,
) -> ThreeContainerMarginals:
    """Return exact rank-count marginals for hidden players A/B and the bottom."""
    _validate_inputs(scores, unknown_counts, capacity_a, capacity_b)
    program = _dynamic_program(scores, unknown_counts, capacity_a, capacity_b)
    batch_size = scores.shape[0]
    rank_probabilities: list[Tensor] = []
    for rank in range(BELIEF_RANK_COUNT):
        allocation_values: list[Tensor] = []
        for count_a in range(MAX_RANK_COPIES + 1):
            for count_b in range(MAX_RANK_COPIES + 1):
                suffix = _shift_backward(program.backward[rank + 1], count_a, count_b)
                paths = program.forward[rank] + suffix
                log_value = (
                    torch.logsumexp(paths.flatten(start_dim=1), dim=1)
                    + scores[:, rank, count_a, count_b]
                    - program.log_partition
                )
                valid = count_a + count_b <= unknown_counts[:, rank]
                allocation_values.append(
                    torch.where(
                        valid,
                        log_value,
                        scores.new_full((batch_size,), _NEGATIVE_INFINITY),
                    )
                )
        rank_log_probability = torch.stack(allocation_values, dim=-1).view(
            batch_size, MAX_RANK_COPIES + 1, MAX_RANK_COPIES + 1
        )
        rank_probabilities.append(torch.exp(rank_log_probability))
    allocation_probability = torch.stack(rank_probabilities, dim=1)
    allocation_probability = allocation_probability / allocation_probability.sum(
        dim=(-2, -1), keepdim=True
    )

    probability_a = allocation_probability.sum(dim=-1)
    probability_b = allocation_probability.sum(dim=-2)
    probability_c_values: list[Tensor] = []
    a_values = torch.arange(MAX_RANK_COPIES + 1, device=scores.device)
    b_values = torch.arange(MAX_RANK_COPIES + 1, device=scores.device)
    allocated = a_values[:, None] + b_values[None, :]
    for count_c in range(MAX_RANK_COPIES + 1):
        mask = unknown_counts[:, :, None, None] - allocated[None, None] == count_c
        probability_c_values.append(
            torch.where(mask, allocation_probability, torch.zeros_like(allocation_probability)).sum(
                dim=(-2, -1)
            )
        )
    probability_c = torch.stack(probability_c_values, dim=-1)
    probability = torch.stack((probability_a, probability_b, probability_c), dim=2)

    counts = torch.arange(MAX_RANK_COPIES + 1, dtype=scores.dtype, device=scores.device)
    expected = (probability * counts).sum(dim=-1)
    variance = (probability * (counts - expected.unsqueeze(-1)).square()).sum(dim=-1)
    epsilon = torch.finfo(probability.dtype).tiny
    entropy = -(probability * torch.log(probability.clamp_min(epsilon))).sum(dim=-1)
    capacity_c = unknown_counts.sum(dim=1) - capacity_a - capacity_b
    capacities = torch.stack((capacity_a, capacity_b, capacity_c), dim=-1)
    return ThreeContainerMarginals(
        log_partition=program.log_partition,
        allocation_probability=allocation_probability,
        probability=probability,
        expected=expected,
        variance=variance,
        entropy=entropy,
        capacities=capacities,
    )


def three_container_true_score(
    scores: Tensor,
    assignment_a: Tensor,
    assignment_b: Tensor,
) -> Tensor:
    """Gather an unnormalized score for one complete labeled allocation."""
    if assignment_a.dtype != torch.int64 or assignment_a.shape != scores.shape[:2]:
        raise ValueError("container-A labels must be int64 [B, 15]")
    if assignment_b.dtype != torch.int64 or assignment_b.shape != scores.shape[:2]:
        raise ValueError("container-B labels must be int64 [B, 15]")
    if assignment_a.device != scores.device or assignment_b.device != scores.device:
        raise ValueError("three-container scores and labels must share one device")
    batch = torch.arange(scores.shape[0], device=scores.device)[:, None]
    ranks = torch.arange(BELIEF_RANK_COUNT, device=scores.device)[None, :]
    return scores[batch, ranks, assignment_a, assignment_b].sum(dim=1)


def three_container_nll(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    capacity_b: Tensor,
    assignment_a: Tensor,
    assignment_b: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Exact supervised negative log-likelihood for labeled A/B/bottom allocations."""
    validate_three_container_assignment(
        assignment_a, assignment_b, unknown_counts, capacity_a, capacity_b
    )
    losses = three_container_log_partition(
        scores, unknown_counts, capacity_a, capacity_b
    ) - three_container_true_score(scores, assignment_a, assignment_b)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def validate_three_container_assignment(
    assignment_a: Tensor,
    assignment_b: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    capacity_b: Tensor,
) -> None:
    """Reject labels that violate rank conservation or any container capacity."""
    if assignment_a.dtype != torch.int64 or assignment_a.shape != unknown_counts.shape:
        raise ValueError("container-A allocation must be int64 [B, 15]")
    if assignment_b.dtype != torch.int64 or assignment_b.shape != unknown_counts.shape:
        raise ValueError("container-B allocation must be int64 [B, 15]")
    if assignment_a.device != unknown_counts.device or assignment_b.device != unknown_counts.device:
        raise ValueError("three-container allocations and constraints must share one device")
    if torch.any((assignment_a < 0) | (assignment_b < 0)):
        raise ValueError("three-container allocations cannot be negative")
    if torch.any(assignment_a + assignment_b > unknown_counts):
        raise ValueError("three-container allocation exceeds an unknown rank count")
    if not torch.equal(assignment_a.sum(dim=1), capacity_a):
        raise ValueError("container-A allocation violates capacity")
    if not torch.equal(assignment_b.sum(dim=1), capacity_b):
        raise ValueError("container-B allocation violates capacity")


def sample_three_container_allocations(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    capacity_b: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw exact sequential samples from the constrained CRF."""
    _validate_inputs(scores, unknown_counts, capacity_a, capacity_b)
    program = _dynamic_program(scores, unknown_counts, capacity_a, capacity_b)
    batch_size = scores.shape[0]
    used_a = torch.zeros(batch_size, dtype=torch.int64, device=scores.device)
    used_b = torch.zeros(batch_size, dtype=torch.int64, device=scores.device)
    sampled_a: list[Tensor] = []
    sampled_b: list[Tensor] = []
    batch = torch.arange(batch_size, device=scores.device)
    for rank in range(BELIEF_RANK_COUNT):
        logits: list[Tensor] = []
        choices: list[tuple[int, int]] = []
        for count_a in range(MAX_RANK_COPIES + 1):
            for count_b in range(MAX_RANK_COPIES + 1):
                next_a = used_a + count_a
                next_b = used_b + count_b
                valid = (
                    (count_a + count_b <= unknown_counts[:, rank])
                    & (next_a <= capacity_a)
                    & (next_b <= capacity_b)
                )
                safe_a = next_a.clamp_max(program.backward[rank + 1].shape[1] - 1)
                safe_b = next_b.clamp_max(program.backward[rank + 1].shape[2] - 1)
                value = (
                    scores[:, rank, count_a, count_b]
                    + program.backward[rank + 1][batch, safe_a, safe_b]
                )
                logits.append(
                    torch.where(valid, value, scores.new_full((batch_size,), _NEGATIVE_INFINITY))
                )
                choices.append((count_a, count_b))
        probabilities = torch.softmax(torch.stack(logits, dim=-1), dim=-1)
        selected = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        rank_a = torch.tensor(
            [choices[index][0] for index in selected.tolist()],
            dtype=torch.int64,
            device=scores.device,
        )
        rank_b = torch.tensor(
            [choices[index][1] for index in selected.tolist()],
            dtype=torch.int64,
            device=scores.device,
        )
        sampled_a.append(rank_a)
        sampled_b.append(rank_b)
        used_a = used_a + rank_a
        used_b = used_b + rank_b
    allocation_a = torch.stack(sampled_a, dim=-1)
    allocation_b = torch.stack(sampled_b, dim=-1)
    allocation_c = unknown_counts - allocation_a - allocation_b
    validate_three_container_assignment(
        allocation_a, allocation_b, unknown_counts, capacity_a, capacity_b
    )
    return allocation_a, allocation_b, allocation_c


def _dynamic_program(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    capacity_b: Tensor,
) -> _DynamicProgram:
    batch_size = scores.shape[0]
    maximum_a = int(capacity_a.max().item())
    maximum_b = int(capacity_b.max().item())
    shape = (batch_size, maximum_a + 1, maximum_b + 1)
    negative = scores.new_full(shape, _NEGATIVE_INFINITY)
    initial = negative.clone()
    initial[:, 0, 0] = 0.0
    forward = [initial]
    for rank in range(BELIEF_RANK_COUNT):
        candidates: list[Tensor] = []
        for count_a in range(MAX_RANK_COPIES + 1):
            for count_b in range(MAX_RANK_COPIES + 1):
                shifted = _shift_forward(forward[-1], count_a, count_b)
                candidate = shifted + scores[:, rank, count_a, count_b, None, None]
                valid = count_a + count_b <= unknown_counts[:, rank]
                candidates.append(torch.where(valid[:, None, None], candidate, negative))
        forward.append(torch.logsumexp(torch.stack(candidates, dim=0), dim=0))

    used_a = torch.arange(maximum_a + 1, device=scores.device)
    used_b = torch.arange(maximum_b + 1, device=scores.device)
    terminal_mask = (used_a[None, :, None] == capacity_a[:, None, None]) & (
        used_b[None, None, :] == capacity_b[:, None, None]
    )
    terminal = torch.where(terminal_mask, torch.zeros_like(negative), negative)
    backward: list[Tensor] = [negative] * (BELIEF_RANK_COUNT + 1)
    backward[BELIEF_RANK_COUNT] = terminal
    for rank in range(BELIEF_RANK_COUNT - 1, -1, -1):
        candidates = []
        for count_a in range(MAX_RANK_COPIES + 1):
            for count_b in range(MAX_RANK_COPIES + 1):
                shifted = _shift_backward(backward[rank + 1], count_a, count_b)
                candidate = shifted + scores[:, rank, count_a, count_b, None, None]
                valid = count_a + count_b <= unknown_counts[:, rank]
                candidates.append(torch.where(valid[:, None, None], candidate, negative))
        backward[rank] = torch.logsumexp(torch.stack(candidates, dim=0), dim=0)
    batch = torch.arange(batch_size, device=scores.device)
    partition = forward[-1][batch, capacity_a, capacity_b]
    return _DynamicProgram(partition, tuple(forward), tuple(backward))


def _shift_forward(values: Tensor, count_a: int, count_b: int) -> Tensor:
    if count_a >= values.shape[1] or count_b >= values.shape[2]:
        return torch.full_like(values, _NEGATIVE_INFINITY)
    source = values[:, : values.shape[1] - count_a, : values.shape[2] - count_b]
    return functional.pad(source, (count_b, 0, count_a, 0), value=_NEGATIVE_INFINITY)


def _shift_backward(values: Tensor, count_a: int, count_b: int) -> Tensor:
    if count_a >= values.shape[1] or count_b >= values.shape[2]:
        return torch.full_like(values, _NEGATIVE_INFINITY)
    source = values[:, count_a:, count_b:]
    return functional.pad(source, (0, count_b, 0, count_a), value=_NEGATIVE_INFINITY)


def _validate_inputs(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    capacity_b: Tensor,
) -> None:
    expected_tail = (
        BELIEF_RANK_COUNT,
        MAX_RANK_COPIES + 1,
        MAX_RANK_COPIES + 1,
    )
    if not scores.is_floating_point() or scores.shape[-3:] != expected_tail:
        raise ValueError("three-container scores must be floating [B, 15, 5, 5]")
    batch_size = scores.shape[0]
    if unknown_counts.dtype != torch.int64 or unknown_counts.shape != (
        batch_size,
        BELIEF_RANK_COUNT,
    ):
        raise ValueError("unknown rank counts must be int64 [B, 15]")
    for capacity, label in ((capacity_a, "A"), (capacity_b, "B")):
        if capacity.dtype != torch.int64 or capacity.shape != (batch_size,):
            raise ValueError(f"container-{label} capacity must be int64 [B]")
    if not (scores.device == unknown_counts.device == capacity_a.device == capacity_b.device):
        raise ValueError("three-container scores and constraints must share one device")
    if not torch.isfinite(scores).all():
        raise ValueError("three-container scores contain NaN or infinity")
    if torch.any((unknown_counts < 0) | (unknown_counts > MAX_RANK_COPIES)):
        raise ValueError("unknown rank counts must be in 0..4")
    if torch.any(unknown_counts[:, 13:] > 1):
        raise ValueError("joker unknown counts cannot exceed one")
    total = unknown_counts.sum(dim=1)
    if torch.any((capacity_a < 0) | (capacity_b < 0) | (capacity_a + capacity_b > total)):
        raise ValueError("A/B capacities must fit the unknown pool")


__all__ = (
    "THREE_CONTAINER_SCHEMA_VERSION",
    "ThreeContainerMarginals",
    "sample_three_container_allocations",
    "three_container_log_partition",
    "three_container_marginals",
    "three_container_nll",
    "three_container_true_score",
    "validate_three_container_assignment",
)
