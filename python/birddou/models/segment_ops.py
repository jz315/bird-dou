"""Differentiable non-empty ragged segment reductions without action padding."""

from __future__ import annotations

import torch
from torch import Tensor


def segment_state_index(offsets: Tensor) -> Tensor:
    """Return the flat-row to segment mapping declared by `[B+1]` offsets."""
    lengths = _validate_offsets(offsets)
    return torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.int64, device=offsets.device),
        lengths,
    )


def segment_sum(values: Tensor, offsets: Tensor) -> Tensor:
    """Sum all flat rows independently inside each segment."""
    state_index = _validate_values(values, offsets)
    output = torch.zeros(
        (offsets.shape[0] - 1, *values.shape[1:]),
        dtype=values.dtype,
        device=values.device,
    )
    return output.index_add(0, state_index, values)


def segment_mean(values: Tensor, offsets: Tensor) -> Tensor:
    """Mean all flat rows independently inside each segment."""
    lengths = _validate_offsets(offsets)
    summed = segment_sum(values, offsets)
    shape = (lengths.shape[0],) + (1,) * (values.ndim - 1)
    return summed / lengths.reshape(shape).to(values.dtype)


def segment_max(values: Tensor, offsets: Tensor) -> Tensor:
    """Maximum over flat rows for every trailing feature independently."""
    state_index = _validate_values(values, offsets)
    output = torch.full(
        (offsets.shape[0] - 1, *values.shape[1:]),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    index = state_index.reshape((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
    return output.scatter_reduce(0, index, values, reduce="amax", include_self=True)


def segment_logsumexp(values: Tensor, offsets: Tensor) -> Tensor:
    """Numerically stable log-sum-exp independently inside every segment."""
    state_index = _validate_values(values, offsets)
    maximum = segment_max(values, offsets)
    centered = values - maximum[state_index]
    summed = segment_sum(torch.exp(centered), offsets)
    return maximum + torch.log(summed)


def segment_softmax(values: Tensor, offsets: Tensor) -> Tensor:
    """Numerically stable softmax whose probabilities normalize per segment."""
    state_index = _validate_values(values, offsets)
    maximum = segment_max(values, offsets)
    exponentials = torch.exp(values - maximum[state_index])
    normalizer = segment_sum(exponentials, offsets)
    return exponentials / normalizer[state_index]


def _validate_values(values: Tensor, offsets: Tensor) -> Tensor:
    if not values.is_floating_point():
        raise ValueError("segment values must use a floating dtype")
    if values.ndim < 1:
        raise ValueError("segment values must have at least one dimension")
    if not torch.isfinite(values).all():
        raise ValueError("segment values contain NaN or infinity")
    if values.device != offsets.device:
        raise ValueError("segment values and offsets must share one device")
    lengths = _validate_offsets(offsets)
    if values.shape[0] != int(offsets[-1].item()):
        raise ValueError("segment offsets do not span the value rows")
    return torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.int64, device=offsets.device),
        lengths,
    )


def _validate_offsets(offsets: Tensor) -> Tensor:
    if offsets.dtype != torch.int64 or offsets.ndim != 1 or offsets.shape[0] < 2:
        raise ValueError("segment offsets must be int64 [B+1]")
    if offsets.device.type == "meta":
        raise ValueError("segment offsets cannot use the meta device")
    if int(offsets[0].item()) != 0:
        raise ValueError("segment offsets must start at zero")
    lengths = torch.diff(offsets)
    if torch.any(lengths <= 0):
        raise ValueError("segment offsets must define non-empty increasing segments")
    return lengths


__all__ = (
    "segment_logsumexp",
    "segment_max",
    "segment_mean",
    "segment_softmax",
    "segment_state_index",
    "segment_sum",
)
