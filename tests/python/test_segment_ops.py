"""Correctness, stability, validation, and gradient tests for E019 segment ops."""

from __future__ import annotations

import pytest
import torch

from birddou.models.segment_ops import (
    segment_logsumexp,
    segment_max,
    segment_mean,
    segment_softmax,
    segment_state_index,
    segment_sum,
)


def test_segment_reductions_match_independent_loop_oracles() -> None:
    """Every reduction respects non-uniform segment boundaries and trailing axes."""
    offsets = torch.tensor([0, 1, 4, 6], dtype=torch.int64)
    values = torch.tensor(
        [[1.0, -3.0], [2.0, 0.0], [-4.0, 5.0], [3.0, 1.0], [7.0, -2.0], [1.0, 4.0]]
    )
    slices = [values[offsets[index] : offsets[index + 1]] for index in range(3)]

    assert segment_state_index(offsets).tolist() == [0, 1, 1, 1, 2, 2]
    torch.testing.assert_close(
        segment_sum(values, offsets), torch.stack([x.sum(0) for x in slices])
    )
    torch.testing.assert_close(
        segment_mean(values, offsets), torch.stack([x.mean(0) for x in slices])
    )
    torch.testing.assert_close(
        segment_max(values, offsets), torch.stack([x.amax(0) for x in slices])
    )
    torch.testing.assert_close(
        segment_logsumexp(values, offsets),
        torch.stack([torch.logsumexp(x, 0) for x in slices]),
    )
    expected_softmax = torch.cat([torch.softmax(x, 0) for x in slices])
    torch.testing.assert_close(segment_softmax(values, offsets), expected_softmax)


def test_segment_softmax_is_stable_and_normalized_for_each_state() -> None:
    """Large logits, a singleton, and a long candidate segment remain finite."""
    offsets = torch.tensor([0, 1, 4, 10_004], dtype=torch.int64)
    values = torch.cat(
        (
            torch.tensor([[10_000.0]]),
            torch.tensor([[-10_000.0], [10_000.0], [9_999.0]]),
            torch.linspace(-1_000.0, 1_000.0, 10_000).unsqueeze(1),
        )
    )
    probabilities = segment_softmax(values, offsets)
    state_index = segment_state_index(offsets)

    assert probabilities.shape == values.shape
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        segment_sum(probabilities, offsets),
        torch.ones(3, 1),
        rtol=1e-5,
        atol=1e-6,
    )
    assert probabilities[0].item() == 1.0
    assert state_index.shape == (10_004,)


def test_all_differentiable_segment_ops_have_finite_gradients() -> None:
    """Reductions and normalized probabilities preserve autograd."""
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    values = torch.randn(5, 3, requires_grad=True)
    loss = (
        segment_mean(values, offsets).square().sum()
        + segment_max(values, offsets).square().sum()
        + segment_logsumexp(values, offsets).square().sum()
        + segment_softmax(values, offsets).square().sum()
    )
    torch.autograd.backward((loss,))

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize(
    ("offsets", "match"),
    [
        (torch.tensor([1, 2], dtype=torch.int64), "start at zero"),
        (torch.tensor([0, 1, 1], dtype=torch.int64), "non-empty increasing"),
        (torch.tensor([0.0, 1.0]), "int64"),
        (torch.tensor([[0, 1]], dtype=torch.int64), "int64"),
    ],
)
def test_invalid_offsets_are_rejected(offsets: torch.Tensor, match: str) -> None:
    """Malformed segmentation never silently produces padded or incomplete rows."""
    with pytest.raises(ValueError, match=match):
        segment_mean(torch.ones(1, 2), offsets)


def test_invalid_segment_values_are_rejected() -> None:
    """The public boundary requires a complete finite floating segment span."""
    offsets = torch.tensor([0, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="floating"):
        segment_sum(torch.ones(2, dtype=torch.int64), offsets)
    with pytest.raises(ValueError, match="span"):
        segment_sum(torch.ones(3), offsets)
    with pytest.raises(ValueError, match="NaN"):
        segment_sum(torch.tensor([1.0, float("nan")]), offsets)
