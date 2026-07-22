"""Native NumPy reproduction of the version-1 DouZero feature tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from birddou.env_types import Action, Observation, PlayGameAction, RankCounts
from birddou.eval.paired_deals import SeatRole, role_for_game_seat, role_for_seat

DOUZERO_FEATURE_SCHEMA_VERSION = 1
DOUZERO_CARD_WIDTH = 54
DOUZERO_HISTORY_ACTIONS = 15
DOUZERO_HISTORY_ROWS = 5
DOUZERO_HISTORY_WIDTH = 162
DOUZERO_LANDLORD_WIDTH = 373
DOUZERO_FARMER_WIDTH = 484


class DouZeroFeatureError(ValueError):
    """An observation cannot be represented by the baseline feature schema."""


@dataclass(frozen=True, slots=True)
class DouZeroFeatureBatch:
    """Candidate-aligned legacy features for one information set."""

    schema_version: int
    position: SeatRole
    x_batch: NDArray[np.float32]
    z_batch: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.schema_version != DOUZERO_FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported DouZero feature schema")
        expected_width = (
            DOUZERO_LANDLORD_WIDTH if self.position is SeatRole.LANDLORD else DOUZERO_FARMER_WIDTH
        )
        if self.x_batch.ndim != 2 or self.x_batch.shape[1] != expected_width:
            raise ValueError("x_batch has an invalid legacy feature shape")
        if self.z_batch.shape != (self.x_batch.shape[0], 5, 162):
            raise ValueError("z_batch has an invalid legacy history shape")


def encode_douzero_features(
    observation: Observation,
    legal_actions: Sequence[Action],
) -> DouZeroFeatureBatch:
    """Encode one safe BIRD-Dou observation without importing upstream code."""
    _validate_observation(observation, legal_actions)
    seat = observation["observer"]
    landlord_seat = observation["landlord"]
    if landlord_seat is None:
        raise DouZeroFeatureError("legacy DouZero features require a resolved landlord")
    role = role_for_game_seat(seat, landlord_seat)
    landlord_down_seat = (landlord_seat + 1) % 3
    landlord_up_seat = (landlord_seat + 2) % 3
    action_batch = np.stack(
        [rank_counts_to_douzero_array(_play_counts(action)) for action in legal_actions]
    )
    own = rank_counts_to_douzero_array(observation["own_hand"])
    unknown = rank_counts_to_douzero_array(observation["unknown_pool"])
    last = rank_counts_to_douzero_array(
        [0] * 15 if observation["last_non_pass"] is None else observation["last_non_pass"]["cards"]
    )
    played = tuple(rank_counts_to_douzero_array(counts) for counts in observation["public_played"])
    bomb = _one_hot(observation["bomb_count"], 15, "bomb_count", zero_based=True)
    count = len(legal_actions)
    fixed: tuple[NDArray[np.float32], ...]

    if role is SeatRole.LANDLORD:
        fixed = (
            own,
            unknown,
            last,
            played[landlord_up_seat],
            played[landlord_down_seat],
            _one_hot(
                observation["cards_left"][landlord_up_seat],
                17,
                "landlord_up cards_left",
            ),
            _one_hot(
                observation["cards_left"][landlord_down_seat],
                17,
                "landlord_down cards_left",
            ),
            bomb,
        )
    else:
        teammate_seat = landlord_down_seat if role is SeatRole.LANDLORD_UP else landlord_up_seat
        last_by_seat = _last_actions_by_seat(observation)
        fixed = (
            own,
            unknown,
            played[landlord_seat],
            played[teammate_seat],
            last,
            last_by_seat[landlord_seat],
            last_by_seat[teammate_seat],
            _one_hot(
                observation["cards_left"][landlord_seat],
                20,
                "landlord cards_left",
            ),
            _one_hot(
                observation["cards_left"][teammate_seat],
                17,
                "teammate cards_left",
            ),
            bomb,
        )

    fixed_batch = np.repeat(np.concatenate(fixed)[None, :], count, axis=0)
    x_batch = np.ascontiguousarray(np.hstack((fixed_batch, action_batch)), dtype=np.float32)
    history = _history_array(observation)
    z_batch = np.ascontiguousarray(np.repeat(history[None, :, :], count, axis=0))
    return DouZeroFeatureBatch(
        schema_version=DOUZERO_FEATURE_SCHEMA_VERSION,
        position=role,
        x_batch=x_batch,
        z_batch=z_batch,
    )


def rank_counts_to_douzero_array(counts: RankCounts) -> NDArray[np.float32]:
    """Encode 15 rank counts as the exact 54-bit DouZero plane layout."""
    _validate_rank_counts(counts)
    encoded = np.zeros(DOUZERO_CARD_WIDTH, dtype=np.float32)
    for rank, count in enumerate(counts[:13]):
        encoded[rank * 4 : rank * 4 + count] = 1.0
    encoded[52] = counts[13]
    encoded[53] = counts[14]
    return encoded


def _history_array(observation: Observation) -> NDArray[np.float32]:
    cardplay = [event for event in observation["history"] if "play" in event["action"]]
    recent = cardplay[-DOUZERO_HISTORY_ACTIONS:]
    history = np.zeros((DOUZERO_HISTORY_ACTIONS, DOUZERO_CARD_WIDTH), dtype=np.float32)
    start = DOUZERO_HISTORY_ACTIONS - len(recent)
    for row, event in enumerate(recent, start=start):
        history[row] = rank_counts_to_douzero_array(_play_counts(event["action"]))
    return np.ascontiguousarray(history.reshape(DOUZERO_HISTORY_ROWS, DOUZERO_HISTORY_WIDTH))


def _last_actions_by_seat(
    observation: Observation,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    empty: RankCounts = [0] * 15
    last: list[RankCounts] = [empty.copy(), empty.copy(), empty.copy()]
    for event in observation["history"]:
        if "play" not in event["action"]:
            continue
        actor = event["actor"]
        if actor not in (0, 1, 2):
            raise DouZeroFeatureError(f"history actor is outside 0..2: {actor}")
        last[actor] = _play_counts(event["action"])
    return (
        rank_counts_to_douzero_array(last[0]),
        rank_counts_to_douzero_array(last[1]),
        rank_counts_to_douzero_array(last[2]),
    )


def _play_counts(action: Action) -> RankCounts:
    if "play" not in action:
        raise DouZeroFeatureError("legacy post-bid features received a non-play action")
    return cast(PlayGameAction, action)["play"]["cards"]


def _one_hot(
    value: int,
    width: int,
    label: str,
    *,
    zero_based: bool = False,
) -> NDArray[np.float32]:
    lower = 0 if zero_based else 1
    upper = width - 1 if zero_based else width
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise DouZeroFeatureError(f"{label} must be in {lower}..{upper}, got {value!r}")
    encoded = np.zeros(width, dtype=np.float32)
    encoded[value if zero_based else value - 1] = 1.0
    return encoded


def _validate_observation(
    observation: Observation,
    legal_actions: Sequence[Action],
) -> None:
    if observation["phase"] != "card_play":
        raise DouZeroFeatureError(
            f"legacy DouZero features require card_play, got {observation['phase']}"
        )
    if observation["current_player"] != observation["observer"]:
        raise DouZeroFeatureError("features require the current player's observation")
    role_for_seat(observation["observer"])
    landlord = observation["landlord"]
    if landlord is None:
        raise DouZeroFeatureError("features require a resolved landlord")
    role_for_seat(landlord)
    if len(observation["public_played"]) != 3 or len(observation["cards_left"]) != 3:
        raise DouZeroFeatureError("observation must contain exactly three seats")
    if not legal_actions:
        raise DouZeroFeatureError("features require at least one legal action")


def _validate_rank_counts(counts: RankCounts) -> None:
    if len(counts) != 15:
        raise DouZeroFeatureError(f"rank counts must have length 15, got {len(counts)}")
    for rank, count in enumerate(counts):
        maximum = 1 if rank >= 13 else 4
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= maximum:
            raise DouZeroFeatureError(f"rank {rank} count must be in 0..{maximum}, got {count!r}")


__all__ = (
    "DOUZERO_CARD_WIDTH",
    "DOUZERO_FARMER_WIDTH",
    "DOUZERO_FEATURE_SCHEMA_VERSION",
    "DOUZERO_HISTORY_ACTIONS",
    "DOUZERO_HISTORY_ROWS",
    "DOUZERO_HISTORY_WIDTH",
    "DOUZERO_LANDLORD_WIDTH",
    "DouZeroFeatureBatch",
    "DouZeroFeatureError",
    "encode_douzero_features",
    "rank_counts_to_douzero_array",
)
