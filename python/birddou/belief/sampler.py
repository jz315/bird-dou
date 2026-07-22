"""Exact backward-DP sampler for capacity-preserving hidden hands."""

from __future__ import annotations

import torch
from torch import Tensor

from birddou.belief.cardinality_crf import (
    BELIEF_RANK_COUNT,
    MAX_RANK_COPIES,
    _dynamic_program,
    _validate_inputs,
)

_NEGATIVE_INFINITY = -1.0e30


def sample_hidden_allocations(
    scores: Tensor,
    unknown_counts: Tensor,
    capacity_a: Tensor,
    sample_count: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw `[B, S, 15]` exact CRF samples with zero conservation violations."""
    _validate_inputs(scores, unknown_counts, capacity_a)
    if sample_count <= 0:
        raise ValueError("belief sample_count must be positive")
    program = _dynamic_program(scores, unknown_counts, capacity_a)
    batch_size = scores.shape[0]
    count_values = torch.arange(MAX_RANK_COPIES + 1, device=scores.device)
    used = torch.zeros((batch_size, sample_count), dtype=torch.int64, device=scores.device)
    allocation: list[Tensor] = []
    for rank in range(BELIEF_RANK_COUNT):
        next_used = used[:, :, None] + count_values[None, None, :]
        in_capacity = next_used <= capacity_a[:, None, None]
        safe_index = next_used.clamp(max=program.backward[rank + 1].shape[1] - 1)
        suffix_table = program.backward[rank + 1][:, None, :].expand(-1, sample_count, -1)
        suffix = suffix_table.gather(2, safe_index)
        logits = scores[:, None, rank, :] + suffix
        valid = (count_values[None, None, :] <= unknown_counts[:, rank, None, None]) & in_capacity
        logits = torch.where(valid, logits, logits.new_full(logits.shape, _NEGATIVE_INFINITY))
        probability = torch.softmax(logits, dim=-1)
        selected = torch.multinomial(
            probability.reshape(batch_size * sample_count, -1),
            1,
            generator=generator,
        ).reshape(batch_size, sample_count)
        allocation.append(selected)
        used = used + selected
    result = torch.stack(allocation, dim=-1)
    if not torch.equal(used, capacity_a[:, None].expand_as(used)) or torch.any(
        result > unknown_counts[:, None, :]
    ):
        raise RuntimeError("belief sampler violated a cardinality constraint")
    return result


__all__ = ("sample_hidden_allocations",)
