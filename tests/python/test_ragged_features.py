"""Shape, semantics, privacy, and determinism tests for feature schema version 1."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import torch

from birddou import PyDdzEnv, generate_lead_actions, load_rule_config, minimum_play_groups
from birddou.env_types import Action, GameEvent, Move, Observation, RuleConfig
from birddou.features import (
    ACTION_META_COLUMNS,
    DECOMPOSITION_DISABLED_GROUPS,
    FEATURE_SCHEMA_VERSION,
    HISTORY_META_COLUMNS,
    RANK_CATEGORICAL_COLUMNS,
    RANK_NUMERIC_COLUMNS,
    SCALAR_COLUMNS,
    FeatureConfig,
    FeatureEncodingError,
    encode_candidate_actions,
    encode_observations,
    encode_public_history,
    encode_ragged_batch,
    load_feature_config,
)
from birddou.schemas import RAGGED_BATCH_SCHEMA

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
FEATURE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model" / "bird_dou_features_v1.yaml"


def rules() -> RuleConfig:
    return load_rule_config(RULES_PATH)


def test_actual_states_form_deterministic_ragged_segments_and_chosen_mapping() -> None:
    """Different candidate counts pack into exact offsets without object tensors."""
    environments = (PyDdzEnv(), PyDdzEnv())
    observations = [
        environment.reset(seed, rules())
        for environment, seed in zip(environments, (7, 8), strict=True)
    ]
    actions = [environment.legal_actions() for environment in environments]
    settings = FeatureConfig(decomposition_features=False)
    first = encode_ragged_batch(observations, actions, rules(), (0, len(actions[1]) - 1), settings)
    second = encode_ragged_batch(observations, actions, rules(), (0, len(actions[1]) - 1), settings)

    assert first.schema_version == FEATURE_SCHEMA_VERSION
    assert first.rank_categorical.shape == (2, 15, len(RANK_CATEGORICAL_COLUMNS))
    assert first.rank_numeric.shape == (2, 15, len(RANK_NUMERIC_COLUMNS))
    assert first.history_meta.shape == (2, 96, len(HISTORY_META_COLUMNS))
    assert first.scalars.shape == (2, len(SCALAR_COLUMNS))
    assert first.action_meta.shape == (sum(map(len, actions)), len(ACTION_META_COLUMNS))
    assert first.action_offsets.tolist() == [0, len(actions[0]), len(actions[0]) + len(actions[1])]
    assert first.chosen_action_flat_index.tolist() == [0, sum(map(len, actions)) - 1]
    assert first.action_state_index.tolist() == [0] * len(actions[0]) + [1] * len(actions[1])
    assert first.history_mask.dtype == torch.bool
    assert first.rank_categorical.dtype == torch.int64
    assert first.rank_numeric.dtype == torch.float32
    assert all(
        torch.equal(getattr(first, field), getattr(second, field))
        for field in first.__dataclass_fields__
        if field != "schema_version"
    )


def test_split_feature_modules_and_machine_readable_schema_match_canonical_batch() -> None:
    environment = PyDdzEnv()
    observation = environment.reset(70016, rules())
    actions = tuple(environment.legal_actions())
    config = FeatureConfig(decomposition_features=False)
    canonical = encode_ragged_batch((observation,), (actions,), rules(), config=config)
    combined = encode_observations((observation,), (actions,), rules(), config=config)
    history = encode_public_history((observation,), rules(), config=config)
    candidates = encode_candidate_actions((observation,), (actions,), rules(), config=config)

    assert torch.equal(combined.action_meta, canonical.action_meta)
    assert torch.equal(history.rank_counts, canonical.history_rank_counts)
    assert torch.equal(history.metadata, canonical.history_meta)
    assert torch.equal(history.mask, canonical.history_mask)
    assert torch.equal(candidates.rank_counts, canonical.action_rank_counts)
    assert torch.equal(candidates.metadata, canonical.action_meta)
    assert torch.equal(candidates.offsets, canonical.action_offsets)
    assert RAGGED_BATCH_SCHEMA.schema_version == FEATURE_SCHEMA_VERSION
    assert len(RAGGED_BATCH_SCHEMA.fingerprint()) == 64
    assert {field.name for field in RAGGED_BATCH_SCHEMA.fields} == set(
        canonical.__dataclass_fields__
    ) - {"schema_version"}


def test_relative_seats_pass_and_exact_decomposition_metadata() -> None:
    """Rank features rotate by observer while candidate metadata remains exact."""
    observation = _small_observation(observer=0)
    actions = (_pass(), _single(0), _single(2))
    batch = encode_ragged_batch((observation,), (actions,), rules(), (2,))
    meta = {name: index for index, name in enumerate(ACTION_META_COLUMNS)}

    assert batch.action_rank_counts[0].tolist() == [0] * 15
    assert batch.post_hand_counts[0].tolist() == observation["own_hand"]
    assert batch.action_meta[:, meta["min_groups_after"]].tolist() == [2, 1, 1]
    assert batch.action_meta[:, meta["number_of_min_decompositions_capped"]].tolist() == [2, 1, 1]
    assert batch.action_meta[:, meta["is_pass"]].tolist() == [1, 0, 0]
    assert batch.action_meta[:, meta["leaves_one_card"]].tolist() == [0, 1, 1]
    assert batch.chosen_action_flat_index.tolist() == [2]

    rotated = _small_observation(observer=1)
    rotated["public_played"] = [
        observation["public_played"][2],
        observation["public_played"][0],
        observation["public_played"][1],
    ]
    rotated_batch = encode_ragged_batch(
        (rotated,),
        (actions,),
        rules(),
        config=FeatureConfig(decomposition_features=False),
    )
    base_without_decomposition = encode_ragged_batch(
        (observation,),
        (actions,),
        rules(),
        config=FeatureConfig(decomposition_features=False),
    )
    assert torch.equal(
        rotated_batch.rank_categorical,
        base_without_decomposition.rank_categorical,
    )
    assert torch.equal(rotated_batch.rank_numeric, base_without_decomposition.rank_numeric)
    assert rotated_batch.action_meta[:, -2].tolist() == [DECOMPOSITION_DISABLED_GROUPS] * 3


def test_history_valid_rows_ignore_right_padding_and_track_trick_reset() -> None:
    """Valid event encodings are independent of the configured right-padding width."""
    observation = _small_observation(observer=0)
    observation["own_hand"] = _counts((2, 1))
    observation["cards_left"] = [1, 17, 17]
    observation["public_played"] = [[0] * 15 for _ in range(3)]
    observation["public_played"][0] = _counts((0, 1))
    observation["history"] = [
        _event(0, 0, _single(0)),
        _event(1, 1, _pass()),
        _event(2, 2, _pass()),
    ]
    action = _single(2)
    short = encode_ragged_batch(
        (observation,),
        ((action,),),
        rules(),
        config=FeatureConfig(
            history_max_length=5,
            history_early_events=2,
            decomposition_features=False,
        ),
    )
    long = encode_ragged_batch(
        (observation,),
        ((action,),),
        rules(),
        config=FeatureConfig(history_max_length=96, decomposition_features=False),
    )
    columns = {name: index for index, name in enumerate(HISTORY_META_COLUMNS)}

    assert short.history_mask.tolist() == [[True, True, True, False, False]]
    assert long.history_mask[0, :3].tolist() == [True, True, True]
    assert torch.equal(short.history_rank_counts[0, :3], long.history_rank_counts[0, :3])
    assert torch.equal(short.history_meta[0, :3], long.history_meta[0, :3])
    assert short.history_meta[0, :3, columns["cards_left_after"]].tolist() == [1, 17, 17]
    assert short.history_meta[0, :3, columns["trick_index"]].tolist() == [0, 0, 0]
    assert short.history_meta[0, :3, columns["position_in_trick"]].tolist() == [0, 1, 2]


def test_native_lead_and_minimum_group_helpers_share_authoritative_rules() -> None:
    """The Python boundary exposes canonical leads and capped exact decomposition."""
    unrelated = _counts((0, 1), (2, 1))
    pair = _counts((4, 2))

    assert len(generate_lead_actions(unrelated, rules())) == 2
    summaries = minimum_play_groups([unrelated, pair, [0] * 15], rules(), 255)
    assert summaries == [
        {"min_groups": 2, "optimal_orderings_capped": 2},
        {"min_groups": 1, "optimal_orderings_capped": 1},
        {"min_groups": 0, "optimal_orderings_capped": 1},
    ]


def test_long_history_retains_early_context_and_latest_events() -> None:
    """Bounded histories report truncation and preserve deterministic endpoints."""
    observation = _small_observation(observer=0)
    observation["own_hand"] = _counts((2, 1))
    observation["cards_left"] = [1, 17, 17]
    observation["public_played"] = [_counts((0, 1), (1, 1)), [0] * 15, [0] * 15]
    observation["history"] = [
        _event(0, 0, _single(0)),
        _event(1, 1, _pass()),
        _event(2, 2, _pass()),
        _event(3, 0, _single(1)),
        _event(4, 1, _pass()),
        _event(5, 2, _pass()),
    ]
    batch = encode_ragged_batch(
        (observation,),
        ((_single(2),),),
        rules(),
        config=FeatureConfig(
            history_max_length=4,
            history_early_events=1,
            decomposition_features=False,
        ),
    )
    scalar = {name: index for index, name in enumerate(SCALAR_COLUMNS)}
    sequence_counts = batch.history_rank_counts[0, :, :2].tolist()

    assert batch.history_mask.tolist() == [[True, True, True, True]]
    assert batch.scalars[0, scalar["history_truncated"]].item() == 2
    assert sequence_counts == [[1, 0], [0, 1], [0, 0], [0, 0]]


def test_feature_config_and_invalid_chosen_action_are_explicit() -> None:
    """Ablations are versioned and invalid local indices never silently wrap."""
    config = load_feature_config(FEATURE_CONFIG_PATH)
    assert config.schema_version == FEATURE_SCHEMA_VERSION
    assert config.decomposition_features
    assert config.history_max_length == 96

    with pytest.raises(FeatureEncodingError, match="outside"):
        encode_ragged_batch(
            (_small_observation(0),),
            ((_single(0),),),
            rules(),
            (1,),
            FeatureConfig(decomposition_features=False),
        )


def _small_observation(observer: int) -> Observation:
    public_played = [_counts((3, 1)), _counts((4, 1)), _counts((5, 1))]
    return cast(
        Observation,
        {
            "schema_version": 1,
            "phase": "card_play",
            "observer": observer,
            "role": "landlord" if observer == 0 else "farmer",
            "own_hand": _counts((0, 1), (2, 1)),
            "public_played": public_played,
            "public_bottom_cards": [0] * 15,
            "unknown_pool": [0] * 15,
            "cards_left": [2, 17, 17],
            "current_player": observer,
            "landlord": 0,
            "last_non_pass": None,
            "consecutive_passes": 0,
            "bid_history": [],
            "history": [],
            "multiplier_exp": 0,
            "bomb_count": 0,
        },
    )


def _counts(*values: tuple[int, int]) -> list[int]:
    result = [0] * 15
    for rank, count in values:
        result[rank] = count
    return result


def _pass() -> Action:
    return {
        "play": cast(
            Move,
            {"kind": "pass", "cards": [0] * 15, "main_rank": 15, "chain_len": 0, "total_cards": 0},
        )
    }


def _single(rank: int) -> Action:
    return {
        "play": cast(
            Move,
            {
                "kind": "single",
                "cards": _counts((rank, 1)),
                "main_rank": rank,
                "chain_len": 1,
                "total_cards": 1,
            },
        )
    }


def _event(sequence: int, actor: int, action: Action) -> GameEvent:
    return {"sequence": sequence, "actor": actor, "action": deepcopy(action)}
