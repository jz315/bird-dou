"""Checkpoint-compatible native PyTorch reproduction of the DouZero networks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final, Literal

import torch
from torch import Tensor, nn

DOUZERO_MODEL_SCHEMA_VERSION: Final = 1
DOUZERO_LSTM_INPUT: Final = 162
DOUZERO_LSTM_HIDDEN: Final = 128
DOUZERO_MLP_HIDDEN: Final = 512
DOUZERO_LANDLORD_INPUT: Final = 373
DOUZERO_FARMER_INPUT: Final = 484
DouZeroPosition = Literal["landlord", "landlord_down", "landlord_up"]


class _DouZeroLstmModel(nn.Module):
    """Shared implementation with role-specific flat-feature width."""

    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(DOUZERO_LSTM_INPUT, DOUZERO_LSTM_HIDDEN, batch_first=True)
        self.dense1 = nn.Linear(input_width + DOUZERO_LSTM_HIDDEN, DOUZERO_MLP_HIDDEN)
        self.dense2 = nn.Linear(DOUZERO_MLP_HIDDEN, DOUZERO_MLP_HIDDEN)
        self.dense3 = nn.Linear(DOUZERO_MLP_HIDDEN, DOUZERO_MLP_HIDDEN)
        self.dense4 = nn.Linear(DOUZERO_MLP_HIDDEN, DOUZERO_MLP_HIDDEN)
        self.dense5 = nn.Linear(DOUZERO_MLP_HIDDEN, DOUZERO_MLP_HIDDEN)
        self.dense6 = nn.Linear(DOUZERO_MLP_HIDDEN, 1)

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        *,
        return_value: bool = True,
    ) -> Mapping[str, Tensor]:
        """Score each candidate, or return the stable first argmax action."""
        sequence, _ = self.lstm(z)
        hidden = sequence[:, -1, :]
        value = torch.cat((hidden, x), dim=-1)
        value = torch.relu(self.dense1(value))
        value = torch.relu(self.dense2(value))
        value = torch.relu(self.dense3(value))
        value = torch.relu(self.dense4(value))
        value = torch.relu(self.dense5(value))
        value = self.dense6(value)
        if return_value:
            return {"values": value}
        return {"action": torch.argmax(value, dim=0)[0]}


class DouZeroLandlordModel(_DouZeroLstmModel):
    """Landlord 373-feature baseline network."""

    def __init__(self) -> None:
        super().__init__(DOUZERO_LANDLORD_INPUT)


class DouZeroFarmerModel(_DouZeroLstmModel):
    """Shared upstream/downstream farmer 484-feature baseline network."""

    def __init__(self) -> None:
        super().__init__(DOUZERO_FARMER_INPUT)


DOUZERO_MODEL_FACTORIES: Mapping[DouZeroPosition, Callable[[], _DouZeroLstmModel]] = {
    "landlord": DouZeroLandlordModel,
    "landlord_down": DouZeroFarmerModel,
    "landlord_up": DouZeroFarmerModel,
}


def create_douzero_model(position: DouZeroPosition) -> _DouZeroLstmModel:
    """Construct a role network with official checkpoint-compatible key names."""
    try:
        factory = DOUZERO_MODEL_FACTORIES[position]
    except KeyError as error:
        raise ValueError(f"unknown DouZero position: {position}") from error
    return factory()


__all__ = (
    "DOUZERO_FARMER_INPUT",
    "DOUZERO_LANDLORD_INPUT",
    "DOUZERO_LSTM_HIDDEN",
    "DOUZERO_LSTM_INPUT",
    "DOUZERO_MLP_HIDDEN",
    "DOUZERO_MODEL_FACTORIES",
    "DOUZERO_MODEL_SCHEMA_VERSION",
    "DouZeroFarmerModel",
    "DouZeroLandlordModel",
    "DouZeroPosition",
    "create_douzero_model",
)
