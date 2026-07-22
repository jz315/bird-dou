"""Numerically checked reinforcement-learning losses."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as functional

DmcLossName = Literal["mse", "huber"]


def dmc_value_loss(
    prediction: Tensor,
    target: Tensor,
    loss_name: DmcLossName,
    huber_delta: float = 1.0,
) -> Tensor:
    """Compute the scalar terminal-return regression loss."""
    values = prediction.squeeze(-1)
    if values.shape != target.shape:
        raise ValueError(f"prediction shape {values.shape} differs from target {target.shape}")
    if not torch.isfinite(values).all() or not torch.isfinite(target).all():
        raise ValueError("DMC loss received NaN or infinity")
    if loss_name == "mse":
        return functional.mse_loss(values, target)
    if loss_name == "huber":
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        return functional.huber_loss(values, target, delta=huber_delta)
    raise ValueError(f"unknown DMC loss: {loss_name}")


__all__ = ("DmcLossName", "dmc_value_loss")
