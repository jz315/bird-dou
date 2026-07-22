"""Version-1 ragged BIRD-Dou feature schema and deterministic encoders."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from birddou.env import minimum_play_groups
from birddou.env_types import (
    Action,
    DoubleGameAction,
    Move,
    Observation,
    PlayGameAction,
    RankCounts,
    RuleConfig,
)

FEATURE_SCHEMA_VERSION = 1
DEFAULT_HISTORY_MAX_LENGTH = 96
DECOMPOSITION_DISABLED_GROUPS = 21

RANK_CATEGORICAL_COLUMNS = (
    "rank_id",
    "own_count",
    "unknown_count",
    "played_relative_self",
    "played_relative_next",
    "played_relative_previous",
    "last_non_pass_count",
    "public_bottom_count",
    "is_straight_eligible",
)
RANK_NUMERIC_COLUMNS = (
    "own_fraction",
    "unknown_fraction",
    "public_played_fraction",
)
HISTORY_META_COLUMNS = (
    "phase",
    "relative_actor",
    "is_pass",
    "is_play",
    "is_bid",
    "is_double",
    "move_kind",
    "main_rank",
    "chain_len",
    "wing_kind",
    "total_cards",
    "cards_left_after",
    "multiplier_exp_after",
    "trick_index",
    "position_in_trick",
)
SCALAR_COLUMNS = (
    "observer",
    "role",
    "landlord_relative",
    "cards_left_self",
    "cards_left_next",
    "cards_left_previous",
    "consecutive_passes",
    "multiplier_exp",
    "bomb_count",
    "history_length",
    "history_truncated",
    "legal_action_count",
    "phase",
    "active_target_total_cards",
    "rule_config_id",
)
ACTION_META_COLUMNS = (
    "move_kind",
    "main_rank",
    "chain_len",
    "wing_kind",
    "total_cards",
    "is_pass",
    "is_bomb",
    "is_rocket",
    "empties_hand",
    "leaves_one_card",
    "breaks_bomb_count",
    "breaks_pair_count",
    "min_groups_after",
    "number_of_min_decompositions_capped",
)

MOVE_KIND_CODES = {
    "pass": 0,
    "single": 1,
    "pair": 2,
    "triple": 3,
    "triple_with_single": 4,
    "triple_with_pair": 5,
    "straight": 6,
    "pair_straight": 7,
    "triple_straight": 8,
    "airplane_with_singles": 9,
    "airplane_with_pairs": 10,
    "four_with_two_singles": 11,
    "four_with_two_pairs": 12,
    "bomb": 13,
    "rocket": 14,
}
NO_MOVE_KIND = 15
NO_MAIN_RANK = 16
PHASE_CODES = {"bidding": 0, "doubling": 1, "card_play": 2, "terminal": 3}


class FeatureEncodingError(ValueError):
    """Inputs violate the versioned feature-schema contract."""


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Ablation and bounded-history controls for schema version 1."""

    schema_version: int = FEATURE_SCHEMA_VERSION
    history_max_length: int = DEFAULT_HISTORY_MAX_LENGTH
    history_early_events: int = 8
    decomposition_features: bool = True
    min_decompositions_cap: int = 255

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported feature config schema")
        if self.history_max_length <= 0:
            raise ValueError("history_max_length must be positive")
        if not 0 <= self.history_early_events <= self.history_max_length:
            raise ValueError("history_early_events must fit the history window")
        if self.min_decompositions_cap <= 0:
            raise ValueError("min_decompositions_cap must be positive")


@dataclass(frozen=True, slots=True)
class RaggedBatch:
    """Contiguous tensor batch with a stable legal-action segment per state."""

    schema_version: int
    rank_categorical: Tensor
    rank_numeric: Tensor
    history_rank_counts: Tensor
    history_meta: Tensor
    history_mask: Tensor
    scalars: Tensor
    action_rank_counts: Tensor
    post_hand_counts: Tensor
    action_meta: Tensor
    action_state_index: Tensor
    action_offsets: Tensor
    chosen_action_flat_index: Tensor

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported RaggedBatch schema")
        batch_size = self.rank_categorical.shape[0]
        if self.rank_categorical.shape != (batch_size, 15, len(RANK_CATEGORICAL_COLUMNS)):
            raise ValueError("rank_categorical shape mismatch")
        if self.rank_numeric.shape != (batch_size, 15, len(RANK_NUMERIC_COLUMNS)):
            raise ValueError("rank_numeric shape mismatch")
        if self.history_rank_counts.ndim != 3 or self.history_rank_counts.shape[0] != batch_size:
            raise ValueError("history_rank_counts shape mismatch")
        history_length = self.history_rank_counts.shape[1]
        if self.history_rank_counts.shape[2] != 15:
            raise ValueError("history rank width must be 15")
        if self.history_meta.shape != (
            batch_size,
            history_length,
            len(HISTORY_META_COLUMNS),
        ):
            raise ValueError("history_meta shape mismatch")
        if self.history_mask.shape != (batch_size, history_length):
            raise ValueError("history_mask shape mismatch")
        if self.scalars.shape != (batch_size, len(SCALAR_COLUMNS)):
            raise ValueError("scalars shape mismatch")
        action_count = self.action_rank_counts.shape[0]
        if self.action_rank_counts.shape != (action_count, 15):
            raise ValueError("action_rank_counts shape mismatch")
        if self.post_hand_counts.shape != (action_count, 15):
            raise ValueError("post_hand_counts shape mismatch")
        if self.action_meta.shape != (action_count, len(ACTION_META_COLUMNS)):
            raise ValueError("action_meta shape mismatch")
        if self.action_state_index.shape != (action_count,):
            raise ValueError("action_state_index shape mismatch")
        if self.action_offsets.shape != (batch_size + 1,):
            raise ValueError("action_offsets shape mismatch")
        if self.chosen_action_flat_index.shape != (batch_size,):
            raise ValueError("chosen_action_flat_index shape mismatch")
        _validate_tensor_dtypes(self)
        offsets = self.action_offsets.detach().cpu().tolist()
        if offsets[0] != 0 or offsets[-1] != action_count:
            raise ValueError("action offsets must span exactly 0..M")
        if any(left >= right for left, right in zip(offsets, offsets[1:], strict=False)):
            raise ValueError("every state must own at least one action")
        expected_state = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device=self.action_state_index.device),
            torch.diff(self.action_offsets),
        )
        if not torch.equal(self.action_state_index, expected_state):
            raise ValueError("action_state_index differs from offsets")
        for state, chosen in enumerate(self.chosen_action_flat_index.detach().cpu().tolist()):
            if chosen != -1 and not offsets[state] <= chosen < offsets[state + 1]:
                raise ValueError("chosen action flat index lies outside its state segment")

    @property
    def batch_size(self) -> int:
        return self.rank_categorical.shape[0]

    @property
    def action_count(self) -> int:
        return self.action_rank_counts.shape[0]

    def to(self, device: str | torch.device) -> RaggedBatch:
        """Move every tensor together without changing schema or segmentation."""
        return RaggedBatch(
            schema_version=self.schema_version,
            rank_categorical=self.rank_categorical.to(device),
            rank_numeric=self.rank_numeric.to(device),
            history_rank_counts=self.history_rank_counts.to(device),
            history_meta=self.history_meta.to(device),
            history_mask=self.history_mask.to(device),
            scalars=self.scalars.to(device),
            action_rank_counts=self.action_rank_counts.to(device),
            post_hand_counts=self.post_hand_counts.to(device),
            action_meta=self.action_meta.to(device),
            action_state_index=self.action_state_index.to(device),
            action_offsets=self.action_offsets.to(device),
            chosen_action_flat_index=self.chosen_action_flat_index.to(device),
        )


def load_feature_config(path: Path) -> FeatureConfig:
    """Load the JSON-subset YAML feature configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FeatureEncodingError("feature config must be an object")
    values = cast(dict[str, object], raw)
    return FeatureConfig(
        schema_version=_config_int(values, "schema_version"),
        history_max_length=_config_int(values, "history_max_length"),
        history_early_events=_config_int(values, "history_early_events"),
        decomposition_features=_config_bool(values, "decomposition_features"),
        min_decompositions_cap=_config_int(values, "min_decompositions_cap"),
    )


def encode_ragged_batch(
    observations: Sequence[Observation],
    legal_actions: Sequence[Sequence[Action]],
    rules: RuleConfig,
    chosen_action_indices: Sequence[int] | None = None,
    config: FeatureConfig | None = None,
) -> RaggedBatch:
    """Encode decision states and all legal candidates into one ragged tensor batch."""
    settings = config if config is not None else FeatureConfig()
    if not observations:
        raise FeatureEncodingError("at least one observation is required")
    if len(observations) != len(legal_actions):
        raise FeatureEncodingError("observations and legal_actions must have equal length")
    if chosen_action_indices is not None and len(chosen_action_indices) != len(observations):
        raise FeatureEncodingError("chosen_action_indices must have one entry per state")

    rank_categorical: list[list[list[int]]] = []
    rank_numeric: list[list[list[float]]] = []
    history_counts: list[list[list[int]]] = []
    history_meta: list[list[list[int]]] = []
    history_mask: list[list[bool]] = []
    scalars: list[list[float]] = []
    action_counts: list[list[int]] = []
    post_counts: list[list[int]] = []
    action_meta: list[list[int]] = []
    state_indices: list[int] = []
    offsets = [0]
    chosen_flat: list[int] = []

    for state_index, (observation, actions) in enumerate(
        zip(observations, legal_actions, strict=True)
    ):
        if not actions:
            raise FeatureEncodingError(f"state {state_index} has no legal actions")
        _validate_observation(observation)
        if observation["phase"] != "card_play":
            raise FeatureEncodingError(
                "ragged card-play features require a card_play observation; "
                "use BidBatch for bidding"
            )
        categorical, numeric = _encode_rank_features(observation)
        rows, metas, mask, truncated = _encode_history(observation, settings, rules)
        rank_categorical.append(categorical)
        rank_numeric.append(numeric)
        history_counts.append(rows)
        history_meta.append(metas)
        history_mask.append(mask)
        scalars.append(_encode_scalars(observation, len(actions), sum(mask), truncated, rules))
        state_action_start = len(action_counts)
        for action in actions:
            counts, post, metadata = _encode_action(
                observation["own_hand"],
                action,
                settings,
            )
            action_counts.append(counts)
            post_counts.append(post)
            action_meta.append(metadata)
            state_indices.append(state_index)
        if settings.decomposition_features:
            summaries = minimum_play_groups(
                post_counts[state_action_start:],
                rules,
                settings.min_decompositions_cap,
            )
            if len(summaries) != len(actions):
                raise FeatureEncodingError("native decomposition result count mismatch")
            for row, summary in enumerate(summaries, start=state_action_start):
                groups = summary.get("min_groups")
                decompositions = summary.get("optimal_orderings_capped")
                if not isinstance(groups, int) or not isinstance(decompositions, int):
                    raise FeatureEncodingError("native decomposition result is malformed")
                action_meta[row][-2:] = [groups, decompositions]
        start = offsets[-1]
        offsets.append(start + len(actions))
        if chosen_action_indices is None:
            chosen_flat.append(-1)
        else:
            chosen = chosen_action_indices[state_index]
            if (
                isinstance(chosen, bool)
                or not isinstance(chosen, int)
                or not 0 <= chosen < len(actions)
            ):
                raise FeatureEncodingError(
                    f"chosen action {chosen!r} is outside state {state_index} segment"
                )
            chosen_flat.append(start + chosen)

    return RaggedBatch(
        schema_version=FEATURE_SCHEMA_VERSION,
        rank_categorical=torch.tensor(rank_categorical, dtype=torch.int64),
        rank_numeric=torch.tensor(rank_numeric, dtype=torch.float32),
        history_rank_counts=torch.tensor(history_counts, dtype=torch.int64),
        history_meta=torch.tensor(history_meta, dtype=torch.int64),
        history_mask=torch.tensor(history_mask, dtype=torch.bool),
        scalars=torch.tensor(scalars, dtype=torch.float32),
        action_rank_counts=torch.tensor(action_counts, dtype=torch.int64),
        post_hand_counts=torch.tensor(post_counts, dtype=torch.int64),
        action_meta=torch.tensor(action_meta, dtype=torch.int64),
        action_state_index=torch.tensor(state_indices, dtype=torch.int64),
        action_offsets=torch.tensor(offsets, dtype=torch.int64),
        chosen_action_flat_index=torch.tensor(chosen_flat, dtype=torch.int64),
    )


def _encode_rank_features(
    observation: Observation,
) -> tuple[list[list[int]], list[list[float]]]:
    observer = observation["observer"]
    last = (
        [0] * 15 if observation["last_non_pass"] is None else observation["last_non_pass"]["cards"]
    )
    categorical: list[list[int]] = []
    numeric: list[list[float]] = []
    for rank in range(15):
        capacity = 1 if rank >= 13 else 4
        relative_played = [
            observation["public_played"][(observer + relative) % 3][rank] for relative in range(3)
        ]
        categorical.append(
            [
                rank,
                observation["own_hand"][rank],
                observation["unknown_pool"][rank],
                *relative_played,
                last[rank],
                observation["public_bottom_cards"][rank],
                int(rank <= 11),
            ]
        )
        numeric.append(
            [
                observation["own_hand"][rank] / capacity,
                observation["unknown_pool"][rank] / capacity,
                sum(relative_played) / capacity,
            ]
        )
    return categorical, numeric


def _encode_history(
    observation: Observation,
    config: FeatureConfig,
    rules: RuleConfig,
) -> tuple[list[list[int]], list[list[int]], list[bool], int]:
    initial_cards = [
        observation["cards_left"][seat] + sum(observation["public_played"][seat])
        for seat in range(3)
    ]
    rows: list[tuple[list[int], list[int]]] = []
    for index, bid_event in enumerate(observation["bid_history"]):
        actor = bid_event["actor"]
        bid_action = bid_event["action"]
        rows.append(
            (
                [0] * 15,
                [
                    PHASE_CODES["bidding"],
                    (actor - observation["observer"]) % 3,
                    int(bid_action == "pass"),
                    0,
                    1,
                    0,
                    NO_MOVE_KIND,
                    NO_MAIN_RANK,
                    0,
                    0,
                    0,
                    initial_cards[actor],
                    0,
                    0,
                    index,
                ],
            )
        )

    multiplier = sum(event["action"] == "rob" for event in observation["bid_history"])
    cards_left = initial_cards.copy()
    trick_index = 0
    position_in_trick = 0
    consecutive_passes = 0
    for event in observation["history"]:
        actor = event["actor"]
        event_action = event["action"]
        if "bid" in event_action:
            continue
        if "double" in event_action:
            double_action = cast(DoubleGameAction, event_action)
            doubled = double_action["double"] == "double"
            multiplier += int(doubled)
            rows.append(
                (
                    [0] * 15,
                    [
                        PHASE_CODES["doubling"],
                        (actor - observation["observer"]) % 3,
                        int(not doubled),
                        0,
                        0,
                        1,
                        NO_MOVE_KIND,
                        NO_MAIN_RANK,
                        0,
                        0,
                        0,
                        cards_left[actor],
                        multiplier,
                        trick_index,
                        position_in_trick,
                    ],
                )
            )
            continue
        move = _play_move(event_action)
        cards = _rank_counts(move["cards"], "history action")
        kind = _move_kind(move)
        total = move["total_cards"]
        cards_left[actor] -= total
        if cards_left[actor] < 0:
            raise FeatureEncodingError("history reconstructs a negative hand size")
        if kind in ("bomb", "rocket"):
            factor = rules["rocket_multiplier"] if kind == "rocket" else rules["bomb_multiplier"]
            multiplier += int(math.log2(factor))
        rows.append(
            (
                cards,
                [
                    PHASE_CODES["card_play"],
                    (actor - observation["observer"]) % 3,
                    int(kind == "pass"),
                    1,
                    0,
                    0,
                    MOVE_KIND_CODES[kind],
                    move["main_rank"],
                    move["chain_len"],
                    _wing_kind(kind),
                    total,
                    cards_left[actor],
                    multiplier,
                    trick_index,
                    position_in_trick,
                ],
            )
        )
        if kind == "pass":
            consecutive_passes += 1
        else:
            consecutive_passes = 0
        if consecutive_passes == 2:
            trick_index += 1
            position_in_trick = 0
            consecutive_passes = 0
        else:
            position_in_trick += 1

    retained, truncated = _retain_history(rows, len(observation["bid_history"]), config)
    count_rows = [item[0] for item in retained]
    meta_rows = [item[1] for item in retained]
    mask = [True] * len(retained)
    padding = config.history_max_length - len(retained)
    count_rows.extend([[0] * 15 for _ in range(padding)])
    meta_rows.extend([[0] * len(HISTORY_META_COLUMNS) for _ in range(padding)])
    mask.extend([False] * padding)
    return count_rows, meta_rows, mask, truncated


def _retain_history(
    rows: Sequence[tuple[list[int], list[int]]],
    bid_count: int,
    config: FeatureConfig,
) -> tuple[list[tuple[list[int], list[int]]], int]:
    if len(rows) <= config.history_max_length:
        return list(rows), 0
    if bid_count > config.history_max_length:
        raise FeatureEncodingError("bidding history alone exceeds history_max_length")
    bids = list(rows[:bid_count])
    plays = list(rows[bid_count:])
    capacity = config.history_max_length - bid_count
    early_count = min(config.history_early_events, capacity)
    late_count = capacity - early_count
    early = plays[:early_count]
    late = plays[-late_count:] if late_count else []
    retained = bids + early + late
    return retained, len(rows) - len(retained)


def _encode_scalars(
    observation: Observation,
    legal_action_count: int,
    history_length: int,
    history_truncated: int,
    rules: RuleConfig,
) -> list[float]:
    observer = observation["observer"]
    landlord = observation["landlord"]
    target_total = (
        0 if observation["last_non_pass"] is None else observation["last_non_pass"]["total_cards"]
    )
    return [
        float(observer),
        float({"landlord": 0, "farmer": 1, "unassigned": 2}[observation["role"]]),
        float(3 if landlord is None else (landlord - observer) % 3),
        float(observation["cards_left"][observer]),
        float(observation["cards_left"][(observer + 1) % 3]),
        float(observation["cards_left"][(observer + 2) % 3]),
        float(observation["consecutive_passes"]),
        float(observation["multiplier_exp"]),
        float(observation["bomb_count"]),
        float(history_length),
        float(history_truncated),
        float(legal_action_count),
        float(PHASE_CODES[observation["phase"]]),
        float(target_total),
        float(rules["rule_config_id"]),
    ]


def _encode_action(
    hand: RankCounts,
    action: Action,
    config: FeatureConfig,
) -> tuple[list[int], list[int], list[int]]:
    move = _play_move(action)
    counts = _rank_counts(move["cards"], "legal action")
    post = [owned - used for owned, used in zip(hand, counts, strict=True)]
    if any(value < 0 for value in post):
        raise FeatureEncodingError("legal action consumes cards absent from own_hand")
    kind = _move_kind(move)
    if not config.decomposition_features:
        groups = DECOMPOSITION_DISABLED_GROUPS
        decompositions = config.min_decompositions_cap + 1
    else:
        groups = 0
        decompositions = 0
    remaining = sum(post)
    metadata = [
        MOVE_KIND_CODES[kind],
        move["main_rank"],
        move["chain_len"],
        _wing_kind(kind),
        move["total_cards"],
        int(kind == "pass"),
        int(kind == "bomb"),
        int(kind == "rocket"),
        int(remaining == 0),
        int(remaining == 1),
        sum(owned == 4 and 0 < used < 4 for owned, used in zip(hand, counts, strict=True)),
        sum(owned >= 2 and after < 2 for owned, after in zip(hand, post, strict=True)),
        groups,
        decompositions,
    ]
    return counts, post, metadata


def _validate_observation(observation: Observation) -> None:
    if observation["phase"] not in PHASE_CODES:
        raise FeatureEncodingError(f"unknown observation phase: {observation['phase']}")
    if observation["current_player"] != observation["observer"]:
        raise FeatureEncodingError("ragged features require current-player observations")
    if len(observation["public_played"]) != 3 or len(observation["cards_left"]) != 3:
        raise FeatureEncodingError("observation must contain exactly three seats")
    for label, counts in (
        ("own_hand", observation["own_hand"]),
        ("unknown_pool", observation["unknown_pool"]),
        ("public_bottom_cards", observation["public_bottom_cards"]),
    ):
        _rank_counts(counts, label)


def _rank_counts(value: Sequence[int], label: str) -> list[int]:
    if len(value) != 15:
        raise FeatureEncodingError(f"{label} must contain 15 rank counts")
    result = list(value)
    for rank, count in enumerate(result):
        maximum = 1 if rank >= 13 else 4
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= maximum:
            raise FeatureEncodingError(f"{label} rank {rank} count is invalid: {count!r}")
    return result


def _play_move(action: Action) -> Move:
    if "play" not in action:
        raise FeatureEncodingError("schema version 1 supports card-play actions only")
    return cast(PlayGameAction, action)["play"]


def _move_kind(move: Move) -> str:
    kind = move.get("kind")
    if not isinstance(kind, str) or kind not in MOVE_KIND_CODES:
        raise FeatureEncodingError(f"unknown move kind: {kind!r}")
    return kind


def _wing_kind(kind: str) -> int:
    if kind in ("triple_with_single", "airplane_with_singles", "four_with_two_singles"):
        return 1
    if kind in ("triple_with_pair", "airplane_with_pairs", "four_with_two_pairs"):
        return 2
    return 0


def _validate_tensor_dtypes(batch: RaggedBatch) -> None:
    integer_tensors = (
        batch.rank_categorical,
        batch.history_rank_counts,
        batch.history_meta,
        batch.action_rank_counts,
        batch.post_hand_counts,
        batch.action_meta,
        batch.action_state_index,
        batch.action_offsets,
        batch.chosen_action_flat_index,
    )
    if any(tensor.dtype != torch.int64 for tensor in integer_tensors):
        raise ValueError("categorical/count/index tensors must use int64")
    if batch.rank_numeric.dtype != torch.float32 or batch.scalars.dtype != torch.float32:
        raise ValueError("numeric/scalar tensors must use float32")
    if batch.history_mask.dtype != torch.bool:
        raise ValueError("history_mask must use bool")


def _config_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FeatureEncodingError(f"feature config {key} must be an integer")
    return value


def _config_bool(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise FeatureEncodingError(f"feature config {key} must be a boolean")
    return value


__all__ = (
    "ACTION_META_COLUMNS",
    "DEFAULT_HISTORY_MAX_LENGTH",
    "DECOMPOSITION_DISABLED_GROUPS",
    "FEATURE_SCHEMA_VERSION",
    "HISTORY_META_COLUMNS",
    "MOVE_KIND_CODES",
    "RANK_CATEGORICAL_COLUMNS",
    "RANK_NUMERIC_COLUMNS",
    "SCALAR_COLUMNS",
    "FeatureConfig",
    "FeatureEncodingError",
    "RaggedBatch",
    "encode_ragged_batch",
    "load_feature_config",
)
