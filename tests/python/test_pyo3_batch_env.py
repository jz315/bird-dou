"""End-to-end acceptance tests for the E011 packed NumPy batch environment."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from birddou import (
    BATCH_SCHEMA_VERSION,
    BatchStepResult,
    PackedActions,
    PyBatchDdzEnv,
    PyDdzEnv,
    RuleConfig,
    load_rule_config,
)
from birddou.env_types import PlayGameAction

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
COMPLETE_RULE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rules" / "canonical_full.yaml"


def rules() -> RuleConfig:
    """Load a fresh authoritative rule dictionary."""
    return load_rule_config(RULE_CONFIG_PATH)


def complete_rules() -> RuleConfig:
    """Load the complete bidding/doubling/scoring profile."""
    return load_rule_config(COMPLETE_RULE_CONFIG_PATH)


def assert_contiguous_arrays(payload: object) -> None:
    """Recursively require every transport buffer to be a non-object C array."""
    for value in cast(dict[str, object], payload).values():
        if isinstance(value, dict):
            assert_contiguous_arrays(value)
        elif isinstance(value, np.ndarray):
            assert value.flags.c_contiguous
            assert value.dtype != np.dtype(object)


def assert_equal_array_payloads(first: object, second: object) -> None:
    """Compare corresponding top-level arrays in two packed dictionaries."""
    first_mapping = cast(dict[str, object], first)
    second_mapping = cast(dict[str, object], second)
    for key, value in first_mapping.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, second_mapping[key])


def test_reset_returns_compact_deterministic_contiguous_observations() -> None:
    """Reset packs complete information-set-safe observations without object arrays."""
    environment = PyBatchDdzEnv(rules())
    assert repr(environment) == "PyBatchDdzEnv(uninitialized)"
    assert environment.batch_size == 0
    assert not environment.is_initialized
    assert not environment.all_terminal
    with pytest.raises(RuntimeError, match="reset first"):
        environment.legal_actions_packed()

    seeds = np.array([7, 8, 9], dtype=np.uint64)
    observation = environment.reset(seeds)
    assert environment.batch_size == 3
    assert environment.is_initialized
    assert repr(environment) == "PyBatchDdzEnv(batch_size=3, all_terminal=false)"
    assert_contiguous_arrays(observation)
    assert BATCH_SCHEMA_VERSION == 1
    assert observation["schema_version"] == BATCH_SCHEMA_VERSION
    assert observation["batch_size"] == 3
    assert observation["phase"].dtype == np.uint8
    assert observation["phase"].tolist() == [2, 2, 2]
    assert observation["own_hand"].shape == (3, 15)
    assert observation["public_played"].shape == (3, 3, 15)
    assert observation["public_bottom_cards"].shape == (3, 15)
    assert observation["unknown_pool"].shape == (3, 15)
    assert observation["cards_left"].shape == (3, 3)
    assert observation["cards_left"].tolist() == [[20, 17, 17]] * 3
    assert observation["history_offsets"].dtype == np.int64
    assert observation["history_offsets"].tolist() == [0, 0, 0, 0]
    assert observation["history_rank_counts"].shape == (0, 15)
    assert observation["own_hand"][0].tolist() == [
        1,
        0,
        2,
        1,
        1,
        2,
        2,
        2,
        0,
        4,
        2,
        3,
        0,
        0,
        0,
    ]

    same = PyBatchDdzEnv(rules()).reset(seeds.copy())
    assert_equal_array_payloads(observation, same)


def test_numpy_inputs_are_strict_and_failed_calls_do_not_advance() -> None:
    """Dtypes, contiguity, batch length, and local indices are validated first."""
    environment = PyBatchDdzEnv(rules())
    with pytest.raises(ValueError, match="at least one seed"):
        environment.reset(np.array([], dtype=np.uint64))
    with pytest.raises(ValueError, match="C-contiguous"):
        environment.reset(np.arange(8, dtype=np.uint64)[::2])
    with pytest.raises(TypeError):
        environment.reset(np.array([1, 2], dtype=np.int64))

    environment.reset(np.array([10, 11, 12], dtype=np.uint64))
    before = environment.observe_packed()
    with pytest.raises(ValueError, match="does not match batch size"):
        environment.step_packed(np.array([0, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="outside environment 1"):
        environment.step_packed(np.array([0, 1_000_000, 0], dtype=np.int64))
    after = environment.observe_packed()
    assert_equal_array_payloads(before, after)


def test_packed_actions_and_steps_match_independent_single_environments() -> None:
    """Ragged ranges preserve each single environment's stable action order."""
    seeds = np.array([7, 8, 9], dtype=np.uint64)
    environment = PyBatchDdzEnv(rules())
    environment.reset(seeds)
    packed = environment.legal_actions_packed()
    assert_contiguous_arrays(packed)
    assert packed["schema_version"] == 1
    assert packed["batch_size"] == 3
    assert packed["offsets"].shape == (4,)
    assert packed["rank_counts"].shape == (len(packed["kind"]), 15)
    assert packed["state_index"].shape == packed["kind"].shape

    singles = [PyDdzEnv() for _ in seeds]
    indices = np.array([0, 1, 2], dtype=np.int64)
    for env_index, (single, seed) in enumerate(zip(singles, seeds, strict=True)):
        single.reset(int(seed), rules())
        actions = single.legal_actions()
        start = int(packed["offsets"][env_index])
        end = int(packed["offsets"][env_index + 1])
        assert end - start == len(actions)
        for local_index, action in enumerate(actions):
            move = cast(PlayGameAction, action)["play"]
            flat_index = start + local_index
            assert packed["rank_counts"][flat_index].tolist() == move["cards"]
            assert int(packed["main_rank"][flat_index]) == move["main_rank"]
            assert int(packed["chain_len"][flat_index]) == move["chain_len"]
            assert int(packed["total_cards"][flat_index]) == move["total_cards"]

    result = environment.step_packed(indices)
    assert_contiguous_arrays(result)
    assert result["acted"].tolist() == [1, 1, 1]
    assert result["event_sequence"].tolist() == [0, 0, 0]
    assert result["event_actor"].tolist() == [0, 0, 0]
    assert result["next_player"].tolist() == [1, 1, 1]
    assert result["raw_payoff"].shape == (3, 3)
    assert result["objective_payoff"].shape == (3, 3)
    assert result["observation"]["history_offsets"].tolist() == [0, 1, 2, 3]

    for env_index, (single, local_index) in enumerate(zip(singles, indices, strict=True)):
        expected = single.step(single.legal_actions()[int(local_index)])
        expected_action = cast(PlayGameAction, expected["event"]["action"])
        assert result["action_rank_counts"][env_index].tolist() == expected_action["play"]["cards"]
        assert result["terminal"][env_index] == expected["terminal"]
        assert result["raw_payoff"][env_index].tolist() == expected["raw_payoff"]


def test_asynchronous_games_reach_terminal_with_negative_one_noops() -> None:
    """Finished rows remain stable while other rows continue stepping."""
    batch_size = 8
    environment = PyBatchDdzEnv(rules())
    environment.reset(np.arange(1, batch_size + 1, dtype=np.uint64))
    result: BatchStepResult | None = None
    saw_partial_terminal = False

    for _ in range(500):
        if environment.all_terminal:
            break
        actions: PackedActions = environment.legal_actions_packed()
        indices = np.full(batch_size, -1, dtype=np.int64)
        for env_index in range(batch_size):
            start = int(actions["offsets"][env_index])
            end = int(actions["offsets"][env_index + 1])
            if start < end:
                indices[env_index] = int(np.argmax(actions["total_cards"][start:end]))
        result = environment.step_packed(indices)
        saw_partial_terminal |= bool(np.any(result["acted"] == 0) and np.any(result["acted"] == 1))

    assert environment.all_terminal
    assert saw_partial_terminal
    assert result is not None
    assert np.all(result["terminal"] == 1)
    assert np.all(result["raw_payoff"].sum(axis=1) == 0)
    assert_contiguous_arrays(result)

    terminal_noop = environment.step_packed(np.full(batch_size, -1, dtype=np.int64))
    assert np.all(terminal_noop["acted"] == 0)
    assert np.all(terminal_noop["event_sequence"] == -1)
    with pytest.raises(ValueError, match="terminal environment 0"):
        invalid = np.full(batch_size, -1, dtype=np.int64)
        invalid[0] = 0
        environment.step_packed(invalid)


def test_complete_batch_protocol_covers_bidding_and_doubling_before_cardplay() -> None:
    """Non-play actions use explicit phase/action codes and preserve ragged ranges."""
    environment = PyBatchDdzEnv(complete_rules())
    observation = environment.reset(np.array([101, 102], dtype=np.uint64))
    assert observation["phase"].tolist() == [0, 0]
    assert observation["role"].tolist() == [2, 2]
    assert observation["cards_left"].tolist() == [[17, 17, 17], [17, 17, 17]]
    assert observation["unknown_pool"].sum(axis=1).tolist() == [37, 37]

    bids = environment.legal_actions_packed()
    assert bids["offsets"].tolist() == [0, 4, 8]
    assert bids["phase"].tolist() == [0] * 8
    assert bids["action_code"].tolist() == [0, 1, 2, 3] * 2
    bid_result = environment.step_packed(np.array([3, 3], dtype=np.int64))
    assert bid_result["action_phase"].tolist() == [0, 0]
    assert bid_result["action_code"].tolist() == [3, 3]
    assert bid_result["observation"]["phase"].tolist() == [1, 1]
    assert bid_result["observation"]["history_phase"].tolist() == [0, 0]

    for _ in range(3):
        doubles = environment.legal_actions_packed()
        assert doubles["phase"].tolist() == [1] * 4
        assert doubles["action_code"].tolist() == [0, 1] * 2
        double_result = environment.step_packed(np.array([0, 0], dtype=np.int64))
    assert double_result["observation"]["phase"].tolist() == [2, 2]
    assert double_result["observation"]["role"].tolist() == [0, 0]
    assert double_result["observation"]["history_offsets"].tolist() == [0, 4, 8]
