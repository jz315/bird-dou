"""Differentiable two-container cardinality CRF for hidden DouDizhu hands."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

BELIEF_SCHEMA_VERSION = 1
BELIEF_RANK_COUNT = 15
MAX_RANK_COPIES = 4
_NEGATIVE_INFINITY = -1.0e30


@dataclass(frozen=True, slots=True)
class BeliefMarginals:
    """Exact constrained marginals and policy-fusion summaries."""

    log_partition: Tensor
    probability_a: Tensor
    expected_a: Tensor
    expected_b: Tensor
    variance_a: Tensor
    variance_b: Tensor
    entropy_a: Tensor
    entropy_b: Tensor
    key_probability_a: Tensor
    key_probability_b: Tensor


@dataclass(frozen=True, slots=True)
class _DynamicProgram:
    log_partition: Tensor
    forward: tuple[Tensor, ...]
    backward: tuple[Tensor, ...]


def log_partition(scores: Tensor, unknown_counts: Tensor, capacity_a: Tensor) -> Tensor:
    """Return exact log-normalizers under rank and total-card constraints."""
    _validate_inputs(scores, unknown_counts, capacity_a)
    return _dynamic_program(scores, unknown_counts, capacity_a).log_partition


def cardinality_marginals(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
) -> BeliefMarginals:
    """Compute every rank-count marginal, moment, entropy, and key-card probability."""
    _validate_inputs(scores, unknown_counts, capacity_a)
    program = _dynamic_program(scores, unknown_counts, capacity_a)
    batch_size = scores.shape[0]
    max_capacity = program.forward[0].shape[1] - 1
    negative = scores.new_full((batch_size, max_capacity + 1), _NEGATIVE_INFINITY)
    log_probabilities: list[Tensor] = []
    for rank in range(BELIEF_RANK_COUNT):
        rank_values: list[Tensor] = []
        for count in range(MAX_RANK_COPIES + 1):
            if count == 0:
                suffix = program.backward[rank + 1]
            else:
                suffix = torch.cat(
                    (program.backward[rank + 1][:, count:], negative[:, :count]),
                    dim=1,
                )
            paths = program.forward[rank] + scores[:, rank, count, None] + suffix
            log_value = torch.logsumexp(paths, dim=1) - program.log_partition
            valid = count <= unknown_counts[:, rank]
            rank_values.append(
                torch.where(
                    valid,
                    log_value,
                    scores.new_full((batch_size,), _NEGATIVE_INFINITY),
                )
            )
        log_probabilities.append(torch.stack(rank_values, dim=-1))
    raw_probability = torch.exp(torch.stack(log_probabilities, dim=1))
    valid_counts = _valid_count_mask(unknown_counts)
    raw_probability = torch.where(valid_counts, raw_probability, torch.zeros_like(raw_probability))
    probability = raw_probability / raw_probability.sum(dim=-1, keepdim=True)

    count_values = torch.arange(
        MAX_RANK_COPIES + 1,
        dtype=scores.dtype,
        device=scores.device,
    )
    expected_a = (probability * count_values).sum(dim=-1)
    expected_b = unknown_counts.to(scores.dtype) - expected_a
    variance = (probability * (count_values - expected_a.unsqueeze(-1)).square()).sum(dim=-1)
    epsilon = torch.finfo(probability.dtype).tiny
    entropy = -(probability * torch.log(probability.clamp_min(epsilon))).sum(dim=-1)

    probability_a_empty = probability[..., 0]
    probability_b_empty = probability.gather(-1, unknown_counts.unsqueeze(-1)).squeeze(-1)
    key_a = [1.0 - probability_a_empty[:, rank] for rank in (12, 13, 14)]
    key_b = [1.0 - probability_b_empty[:, rank] for rank in (12, 13, 14)]
    key_a.append(_any_bomb_probability(scores, unknown_counts, capacity_a, for_a=True))
    key_b.append(_any_bomb_probability(scores, unknown_counts, capacity_a, for_a=False))
    return BeliefMarginals(
        log_partition=program.log_partition,
        probability_a=probability,
        expected_a=expected_a,
        expected_b=expected_b,
        variance_a=variance,
        variance_b=variance,
        entropy_a=entropy,
        entropy_b=entropy,
        key_probability_a=torch.stack(key_a, dim=-1),
        key_probability_b=torch.stack(key_b, dim=-1),
    )


def true_assignment_score(scores: Tensor, assignment_a: Tensor) -> Tensor:
    """Gather the unnormalized score of one complete labeled allocation."""
    if assignment_a.dtype != torch.int64 or assignment_a.shape != scores.shape[:2]:
        raise ValueError("true hidden allocation must be int64 [B, 15]")
    if assignment_a.device != scores.device:
        raise ValueError("belief scores and labels must share one device")
    return scores.gather(-1, assignment_a.unsqueeze(-1)).squeeze(-1).sum(dim=-1)


def validate_assignment(
    assignment_a: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
) -> None:
    """Reject any hidden label that violates per-rank or total-card conservation."""
    if assignment_a.dtype != torch.int64 or assignment_a.shape != unknown_counts.shape:
        raise ValueError("hidden allocation must be int64 [B, 15]")
    if assignment_a.device != unknown_counts.device:
        raise ValueError("hidden allocation and constraints must share one device")
    if torch.any((assignment_a < 0) | (assignment_a > unknown_counts)):
        raise ValueError("hidden allocation exceeds the unknown rank pool")
    if not torch.equal(assignment_a.sum(dim=1), capacity_a):
        raise ValueError("hidden allocation violates container-A capacity")


def _dynamic_program(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
) -> _DynamicProgram:
    batch_size = scores.shape[0]
    max_capacity = int(capacity_a.max().item())
    used = torch.arange(max_capacity + 1, device=scores.device)
    negative = scores.new_full((batch_size, max_capacity + 1), _NEGATIVE_INFINITY)
    initial = negative.clone()
    initial[:, 0] = 0.0
    forward = [initial]
    cumulative = torch.zeros(batch_size, dtype=torch.int64, device=scores.device)
    for rank in range(BELIEF_RANK_COUNT):
        previous = forward[-1]
        candidates: list[Tensor] = []
        for count in range(MAX_RANK_COPIES + 1):
            if count == 0:
                shifted = previous
            else:
                shifted = torch.cat((negative[:, :count], previous[:, :-count]), dim=1)
            candidate = shifted + scores[:, rank, count, None]
            candidates.append(
                torch.where(
                    (count <= unknown_counts[:, rank]).unsqueeze(-1),
                    candidate,
                    negative,
                )
            )
        current = torch.logsumexp(torch.stack(candidates, dim=0), dim=0)
        cumulative = cumulative + unknown_counts[:, rank]
        reachable = (used[None] <= cumulative[:, None]) & (used[None] <= capacity_a[:, None])
        forward.append(torch.where(reachable, current, negative))

    terminal = torch.where(
        used[None] == capacity_a[:, None],
        torch.zeros_like(negative),
        negative,
    )
    backward: list[Tensor] = [negative] * (BELIEF_RANK_COUNT + 1)
    backward[BELIEF_RANK_COUNT] = terminal
    suffix_total = torch.zeros(batch_size, dtype=torch.int64, device=scores.device)
    for rank in range(BELIEF_RANK_COUNT - 1, -1, -1):
        following = backward[rank + 1]
        candidates = []
        for count in range(MAX_RANK_COPIES + 1):
            if count == 0:
                shifted = following
            else:
                shifted = torch.cat((following[:, count:], negative[:, :count]), dim=1)
            candidate = shifted + scores[:, rank, count, None]
            candidates.append(
                torch.where(
                    (count <= unknown_counts[:, rank]).unsqueeze(-1),
                    candidate,
                    negative,
                )
            )
        current = torch.logsumexp(torch.stack(candidates, dim=0), dim=0)
        suffix_total = suffix_total + unknown_counts[:, rank]
        needed = capacity_a[:, None] - used[None]
        reachable = (needed >= 0) & (needed <= suffix_total[:, None])
        backward[rank] = torch.where(reachable, current, negative)

    batch = torch.arange(batch_size, device=scores.device)
    partition = forward[-1][batch, capacity_a]
    return _DynamicProgram(partition, tuple(forward), tuple(backward))


def _any_bomb_probability(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    *,
    for_a: bool,
) -> Tensor:
    count_values = torch.arange(
        MAX_RANK_COPIES + 1,
        dtype=torch.int64,
        device=scores.device,
    )
    if for_a:
        forbidden = count_values[None, None, :] == 4
    else:
        forbidden = unknown_counts.unsqueeze(-1) - count_values[None, None, :] == 4
    restricted = scores.masked_fill(forbidden, _NEGATIVE_INFINITY)
    unrestricted_z = _dynamic_program(scores, unknown_counts, capacity_a).log_partition
    no_bomb_z = _dynamic_program(restricted, unknown_counts, capacity_a).log_partition
    no_bomb_probability = torch.exp(no_bomb_z - unrestricted_z).clamp(0.0, 1.0)
    return 1.0 - no_bomb_probability


def _valid_count_mask(unknown_counts: Tensor) -> Tensor:
    values = torch.arange(
        MAX_RANK_COPIES + 1,
        dtype=torch.int64,
        device=unknown_counts.device,
    )
    return values[None, None, :] <= unknown_counts.unsqueeze(-1)


def _validate_inputs(scores: Tensor, unknown_counts: Tensor, capacity_a: Tensor) -> None:
    if not scores.is_floating_point() or scores.shape[-2:] != (
        BELIEF_RANK_COUNT,
        MAX_RANK_COPIES + 1,
    ):
        raise ValueError("belief scores must be floating [B, 15, 5]")
    batch_size = scores.shape[0]
    if unknown_counts.dtype != torch.int64 or unknown_counts.shape != (
        batch_size,
        BELIEF_RANK_COUNT,
    ):
        raise ValueError("unknown rank counts must be int64 [B, 15]")
    if capacity_a.dtype != torch.int64 or capacity_a.shape != (batch_size,):
        raise ValueError("container-A capacity must be int64 [B]")
    if scores.device != unknown_counts.device or scores.device != capacity_a.device:
        raise ValueError("belief scores and constraints must share one device")
    if not torch.isfinite(scores).all():
        raise ValueError("belief scores contain NaN or infinity")
    if torch.any((unknown_counts < 0) | (unknown_counts > MAX_RANK_COPIES)):
        raise ValueError("unknown rank counts must be in 0..4")
    if torch.any(unknown_counts[:, 13:] > 1):
        raise ValueError("joker unknown counts cannot exceed one")
    total = unknown_counts.sum(dim=1)
    if torch.any((capacity_a < 0) | (capacity_a > total)):
        raise ValueError("container-A capacity must fit the unknown pool")


__all__ = (
    "BELIEF_RANK_COUNT",
    "BELIEF_SCHEMA_VERSION",
    "MAX_RANK_COPIES",
    "BeliefMarginals",
    "cardinality_marginals",
    "log_partition",
    "true_assignment_score",
    "validate_assignment",
)
