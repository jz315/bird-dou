"""Typed contiguous-buffer protocol for the native batched environment."""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]
Int8Array = NDArray[np.int8]
UInt32Array = NDArray[np.uint32]
UInt64Array = NDArray[np.uint64]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]


class BatchObservation(TypedDict):
    """Structure-of-arrays current-player observations for `B` environments."""

    schema_version: int
    batch_size: int
    phase: UInt8Array
    observer: UInt8Array
    role: UInt8Array
    own_hand: UInt8Array
    public_played: UInt8Array
    public_bottom_cards: UInt8Array
    unknown_pool: UInt8Array
    cards_left: UInt8Array
    current_player: UInt8Array
    landlord: Int8Array
    last_non_pass_valid: UInt8Array
    last_non_pass_rank_counts: UInt8Array
    last_non_pass_kind: UInt8Array
    last_non_pass_main_rank: UInt8Array
    last_non_pass_chain_len: UInt8Array
    last_non_pass_total_cards: UInt8Array
    consecutive_passes: UInt8Array
    multiplier_exp: UInt8Array
    bomb_count: UInt8Array
    terminal: UInt8Array
    history_offsets: Int64Array
    history_sequence: UInt32Array
    history_actor: UInt8Array
    history_phase: UInt8Array
    history_action_code: UInt8Array
    history_rank_counts: UInt8Array
    history_kind: UInt8Array
    history_main_rank: UInt8Array
    history_chain_len: UInt8Array
    history_total_cards: UInt8Array


class PackedActions(TypedDict):
    """Ragged legal actions with per-environment `[offsets[i], offsets[i+1])` ranges."""

    schema_version: int
    batch_size: int
    offsets: Int64Array
    state_index: Int64Array
    phase: UInt8Array
    action_code: UInt8Array
    rank_counts: UInt8Array
    kind: UInt8Array
    main_rank: UInt8Array
    chain_len: UInt8Array
    total_cards: UInt8Array


class BatchStepResult(TypedDict):
    """Per-environment transition results and packed next observations."""

    schema_version: int
    batch_size: int
    acted: UInt8Array
    event_sequence: Int64Array
    event_actor: Int8Array
    action_rank_counts: UInt8Array
    action_phase: UInt8Array
    action_code: UInt8Array
    action_kind: UInt8Array
    action_main_rank: UInt8Array
    action_chain_len: UInt8Array
    action_total_cards: UInt8Array
    next_player: Int8Array
    terminal: UInt8Array
    raw_payoff: Int32Array
    objective_payoff: Int32Array
    observation: BatchObservation


__all__ = (
    "BatchObservation",
    "BatchStepResult",
    "Int8Array",
    "Int32Array",
    "Int64Array",
    "PackedActions",
    "UInt8Array",
    "UInt32Array",
    "UInt64Array",
)
