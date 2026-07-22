"""Shared-trunk landlord/farmer conditioning with compact seat adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class RoleAdapterConfig:
    """Dimensions for the three-seat residual adapter boundary."""

    d_model: int = 256
    bottleneck_dim: int = 64
    dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.bottleneck_dim <= 0:
            raise ValueError("role adapter dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("role adapter dropout must be in [0, 1)")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError("role adapter LayerNorm epsilon must be positive")


class BottleneckRoleAdapter(nn.Module):
    """Small seat-specific residual path around the shared state trunk."""

    def __init__(self, config: RoleAdapterConfig) -> None:
        super().__init__()
        self.down = nn.Linear(config.d_model, config.bottleneck_dim)
        self.up = nn.Linear(config.bottleneck_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        update = self.up(torch.nn.functional.silu(self.down(inputs)))
        return inputs + cast(Tensor, self.dropout(update))


class SeatLayerNorm(nn.Module):
    """Shared normalization statistics with landlord/down/up affine parameters."""

    def __init__(self, config: RoleAdapterConfig) -> None:
        super().__init__()
        self.width = config.d_model
        self.epsilon = config.layer_norm_epsilon
        self.weight = nn.Embedding(3, config.d_model)
        self.bias = nn.Embedding(3, config.d_model)
        nn.init.ones_(self.weight.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, inputs: Tensor, seat: Tensor) -> Tensor:
        normalized = functional.layer_norm(inputs, (self.width,), eps=self.epsilon)
        return cast(Tensor, normalized * self.weight(seat) + self.bias(seat))


class RoleSeatAdapter(nn.Module):
    """Add role/seat embeddings, then apply one small adapter per relative seat."""

    def __init__(self, config: RoleAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.role_embedding = nn.Embedding(2, config.d_model)
        self.seat_embedding = nn.Embedding(3, config.d_model)
        self.adapters = nn.ModuleList(BottleneckRoleAdapter(config) for _ in range(3))
        self.norm = SeatLayerNorm(config)

    def forward(self, state: Tensor, seat: Tensor) -> Tensor:
        _validate_state_and_seat(state, seat, self.config.d_model)
        role = (seat != 0).to(torch.int64)
        conditioned = state + self.role_embedding(role) + self.seat_embedding(seat)
        candidates = torch.stack(
            [cast(Tensor, adapter(conditioned)) for adapter in self.adapters],
            dim=1,
        )
        selected = candidates[
            torch.arange(state.shape[0], device=state.device),
            seat,
        ]
        return cast(Tensor, self.norm(selected, seat))


class RoleSpecificLinear(nn.Module):
    """Three compact output projections selected per flat action row."""

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        if input_width <= 0 or output_width <= 0:
            raise ValueError("role-specific linear dimensions must be positive")
        self.input_width = input_width
        self.weight = nn.Parameter(torch.empty(3, output_width, input_width))
        self.bias = nn.Parameter(torch.empty(3, output_width))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = input_width**-0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor, seat: Tensor) -> Tensor:
        _validate_state_and_seat(inputs, seat, self.input_width)
        weight = self.weight[seat]
        return torch.bmm(weight, inputs.unsqueeze(-1)).squeeze(-1) + self.bias[seat]


def _validate_state_and_seat(state: Tensor, seat: Tensor, width: int) -> None:
    if state.ndim != 2 or state.shape[1] != width or not state.is_floating_point():
        raise ValueError(f"role adapter state must be floating [B, {width}]")
    if not torch.isfinite(state).all():
        raise ValueError("role adapter state contains NaN or infinity")
    if seat.dtype != torch.int64 or seat.shape != (state.shape[0],):
        raise ValueError("role adapter seat must be int64 [B]")
    if seat.device != state.device:
        raise ValueError("role adapter state and seat must share one device")
    if torch.any((seat < 0) | (seat > 2)):
        raise ValueError("role adapter seat values must be in 0..2")


__all__ = (
    "BottleneckRoleAdapter",
    "RoleAdapterConfig",
    "RoleSeatAdapter",
    "RoleSpecificLinear",
    "SeatLayerNorm",
)
