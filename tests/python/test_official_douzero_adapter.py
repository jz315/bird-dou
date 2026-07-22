"""Official feature and checkpoint inference acceptance tests for E013."""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from birddou import PyDdzEnv, load_rule_config
from birddou.env_types import PlayGameAction, RuleConfig
from birddou.eval.arena import Arena
from birddou.eval.baselines import PolicyDecisionContext
from birddou.eval.douzero_differential import (
    Deal,
    Hands,
    OfficialDouZeroEngine,
    rank_counts_to_douzero_cards,
)
from birddou.eval.paired_deals import (
    SEAT_ROLES,
    ScheduledMatch,
    SeatAssignment,
    SeatRole,
    generate_paired_deals,
    role_for_game_seat,
    role_for_seat,
)
from birddou.features import (
    DOUZERO_FARMER_WIDTH,
    DOUZERO_FEATURE_SCHEMA_VERSION,
    DOUZERO_HISTORY_ACTIONS,
    DOUZERO_HISTORY_ROWS,
    DOUZERO_HISTORY_WIDTH,
    DOUZERO_LANDLORD_WIDTH,
    DouZeroFeatureError,
    encode_douzero_features,
    rank_counts_to_douzero_array,
)
from birddou.models.baseline_douzero import (
    DouZeroAdapterError,
    OfficialDouZeroPolicy,
    _import_official_module,
    encode_official_features,
    load_official_checkpoint_set,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "artifacts" / "baselines" / "douzero" / "manifest.toml"
SOURCE_PATH = MANIFEST_PATH.parent / "source"
WEIGHTS_PATH = MANIFEST_PATH.parent / "weights"
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
MODEL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model" / "douzero_baseline.yaml"
FULL_RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "canonical_full.yaml"


class ReferenceTensor(Protocol):
    """Small tensor surface used by the dynamically imported reference."""

    def detach(self) -> ReferenceTensor: ...

    def cpu(self) -> ReferenceTensor: ...

    def numpy(self) -> NDArray[np.float32]: ...


def rules() -> RuleConfig:
    """Load the exact post-bid profile used by official checkpoints."""
    return load_rule_config(RULES_PATH)


def require_source() -> None:
    """Skip only when the separately fetched Apache-2.0 reference is absent."""
    if not (SOURCE_PATH / "douzero" / "env" / "env.py").is_file():
        pytest.skip("run scripts/fetch_douzero_baseline.py for official-source tests")


def require_weights() -> None:
    """Skip only when external checkpoints or the model extra are absent."""
    require_source()
    if importlib.util.find_spec("torch") is None:
        pytest.skip("install bird-dou[model] for checkpoint tests")
    if not (WEIGHTS_PATH / "douzero_ADP" / "landlord.ckpt").is_file():
        pytest.skip("fetch the douzero_ADP and douzero_WP weight sets")


def test_bird_observations_reproduce_official_features_for_all_roles() -> None:
    """The safe adapter matches actual official infosets across a trajectory."""
    require_source()
    environment = PyDdzEnv()
    environment.reset(70013, rules())
    deal = _deal_from_native_state(environment.serialize())
    reference = OfficialDouZeroEngine(SOURCE_PATH)
    reference.reset(deal)
    feature_module = importlib.import_module("douzero.env.env")
    reference_encoder = cast(
        Callable[[object], Mapping[str, object]],
        feature_module.get_obs,
    )
    seen_roles: set[SeatRole] = set()

    for _decision_index in range(30):
        seat = environment.current_player
        observation = environment.observe(seat)
        legal_actions = environment.legal_actions()
        candidate = encode_official_features(observation, legal_actions, SOURCE_PATH)
        native = encode_douzero_features(observation, legal_actions)
        official = reference_encoder(reference._game.game_infoset)
        official_x = cast(NDArray[np.float32], official["x_batch"])
        official_z = cast(NDArray[np.float32], official["z_batch"])
        official_actions = cast(list[list[int]], official["legal_actions"])
        official_rows = {tuple(action): row for row, action in enumerate(official_actions)}

        assert candidate.position is role_for_seat(seat)
        assert set(candidate.legal_action_cards) == set(official_rows)
        for row, action in enumerate(candidate.legal_action_cards):
            np.testing.assert_array_equal(candidate.x_batch[row], official_x[official_rows[action]])
            np.testing.assert_array_equal(native.x_batch[row], official_x[official_rows[action]])
        np.testing.assert_array_equal(candidate.z_batch[0], official_z[0])
        np.testing.assert_array_equal(native.z_batch[0], official_z[0])
        seen_roles.add(candidate.position)

        selected = max(
            range(len(legal_actions)),
            key=lambda index: cast(PlayGameAction, legal_actions[index])["play"]["total_cards"],
        )
        move = cast(PlayGameAction, legal_actions[selected])["play"]
        normalized = tuple(move["cards"])
        reference.step(normalized)
        result = environment.step(legal_actions[selected])
        if result["terminal"]:
            break

    assert seen_roles == set(SeatRole)


def test_native_card_planes_are_exact_and_reject_impossible_counts() -> None:
    """The 15-rank representation maps exactly onto 52 suit bits and two jokers."""
    counts = [4, 3, 2, 1, 0, 4, 0, 1, 2, 3, 4, 1, 2, 1, 1]
    encoded = rank_counts_to_douzero_array(counts)

    assert encoded.shape == (54,)
    assert encoded.dtype == np.float32
    for rank, count in enumerate(counts[:13]):
        np.testing.assert_array_equal(
            encoded[rank * 4 : rank * 4 + 4],
            np.asarray([1] * count + [0] * (4 - count), dtype=np.float32),
        )
    np.testing.assert_array_equal(encoded[52:], np.ones(2, dtype=np.float32))

    with pytest.raises(DouZeroFeatureError, match="length 15"):
        rank_counts_to_douzero_array([0] * 14)
    with pytest.raises(DouZeroFeatureError, match="rank 13"):
        rank_counts_to_douzero_array([0] * 13 + [2, 0])


def test_native_features_map_roles_relative_to_a_dynamic_landlord() -> None:
    environment = PyDdzEnv()
    full_rules = load_rule_config(FULL_RULES_PATH)
    for seed in range(100):
        observation = environment.reset(seed, full_rules)
        if observation["current_player"] != 0:
            break
    else:
        raise AssertionError("test seeds did not produce a non-zero first bidder")
    while observation["phase"] != "card_play":
        actions = tuple(environment.legal_actions())
        desired = {"score": 3} if observation["phase"] == "bidding" else "decline"
        key = "bid" if observation["phase"] == "bidding" else "double"
        selected = next(index for index, action in enumerate(actions) if action.get(key) == desired)
        environment.step(actions[selected])
        observation = environment.observe(environment.current_player)
    landlord = observation["landlord"]
    assert landlord is not None and landlord != 0

    encoded = encode_douzero_features(observation, environment.legal_actions())

    assert encoded.position is role_for_game_seat(observation["observer"], landlord)
    assert encoded.x_batch.shape[1] in (DOUZERO_LANDLORD_WIDTH, DOUZERO_FARMER_WIDTH)


def test_model_config_locks_the_native_compatibility_contract() -> None:
    """The checked-in baseline switch cannot silently drift from encoder constants."""
    config = cast(dict[str, object], json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8")))

    assert config["schema_version"] == DOUZERO_FEATURE_SCHEMA_VERSION
    assert config["feature_encoder"] == "native"
    assert config["history_actions"] == DOUZERO_HISTORY_ACTIONS
    assert config["history_shape"] == [DOUZERO_HISTORY_ROWS, DOUZERO_HISTORY_WIDTH]
    assert config["landlord_input"] == DOUZERO_LANDLORD_WIDTH
    assert config["farmer_input"] == DOUZERO_FARMER_WIDTH


def test_official_checkpoints_match_reference_forward_and_run_new_environment() -> None:
    """Checksummed ADP/WP weights load and ADP decisions match official PyTorch."""
    require_weights()
    adp = load_official_checkpoint_set(MANIFEST_PATH, "douzero_ADP")
    wp = load_official_checkpoint_set(MANIFEST_PATH, "douzero_WP")
    assert [item.path.stat().st_size for item in adp.files] == [5835440, 6062768, 6062768]
    assert len({item.sha256 for item in (*adp.files, *wp.files)}) == 6

    absent_source = REPOSITORY_ROOT / "artifacts" / "absent-douzero-source"
    policy = OfficialDouZeroPolicy("official:adp", replace(adp, source=absent_source))
    wp_policy = OfficialDouZeroPolicy("official:wp", replace(wp, source=absent_source))
    assert wp_policy.checkpoint_set_name == "douzero_WP"
    _import_official_module(SOURCE_PATH, "douzero.dmc.models")
    deep_agent = importlib.import_module("douzero.evaluation.deep_agent")
    load_model = cast(Callable[[str, str], object], deep_agent._load_model)
    reference_models = {
        role: cast(
            Callable[..., Mapping[str, object]],
            load_model(role.value, str(adp.file_for_role(role).path)),
        )
        for role in SEAT_ROLES
    }
    torch = importlib.import_module("torch")
    environment = PyDdzEnv()
    environment.reset(17, rules())
    seen_roles: set[SeatRole] = set()
    for decision_index in range(30):
        seat = environment.current_player
        role = role_for_seat(seat)
        observation = environment.observe(seat)
        actions = environment.legal_actions()
        context = PolicyDecisionContext(
            deal_index=0,
            deal_seed=17,
            match_id="reference-forward",
            seat=seat,
            role=role,
            decision_index=decision_index,
        )
        scores = policy.score_actions(observation, actions, context)
        features = encode_official_features(observation, actions, SOURCE_PATH)
        with torch.inference_mode():
            output = reference_models[role](
                torch.from_numpy(features.z_batch),
                torch.from_numpy(features.x_batch),
                return_value=True,
            )
        values = cast(ReferenceTensor, output["values"])
        reference_scores = values.detach().cpu().numpy()[:, 0]
        np.testing.assert_array_equal(scores, reference_scores)
        assert policy.select_action(observation, actions, context) == int(
            np.argmax(reference_scores)
        )
        seen_roles.add(role)
        selected = max(
            range(len(actions)),
            key=lambda index: cast(PlayGameAction, actions[index])["play"]["total_cards"],
        )
        result = environment.step(actions[selected])
        if seen_roles == set(SeatRole) or result["terminal"]:
            break
    assert seen_roles == set(SeatRole)

    arena = Arena(rules(), (policy,))
    deal = generate_paired_deals(7, 1).deals[0]
    scheduled = ScheduledMatch(
        "official-adp-smoke",
        deal,
        SeatAssignment((policy.policy_id, policy.policy_id, policy.policy_id)),
    )
    first = arena.play_match(scheduled)
    second = arena.play_match(scheduled)
    assert first == second
    assert first.action_count == 56
    assert first.winner_role is SeatRole.LANDLORD_DOWN
    assert first.raw_payoff == (-2, 1, 1)
    assert first.terminal_state_sha256 == (
        "4466e8ed7b1fcc651734b61941c00e56b87f85a90e2aee054ac6be516aad64cf"
    )


def test_missing_checkpoint_set_is_an_explicit_error(tmp_path: Path) -> None:
    """A missing external artifact never falls back to random parameters."""
    manifest = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        'weights_directory = "weights"',
        'weights_directory = "missing-weights"',
    )
    temporary_manifest = tmp_path / "manifest.toml"
    temporary_manifest.write_text(manifest, encoding="utf-8")

    with pytest.raises(DouZeroAdapterError, match="checkpoint is absent"):
        load_official_checkpoint_set(temporary_manifest, "douzero_ADP")


def _deal_from_native_state(serialized: bytes) -> Deal:
    envelope = cast(dict[str, object], json.loads(serialized))
    state = cast(dict[str, object], envelope["state"])
    raw_hands = cast(list[list[int]], state["hands"])
    raw_bottom = cast(list[int], state["bottom_cards"])
    hands = cast(Hands, tuple(tuple(hand) for hand in raw_hands))
    bottom = tuple(raw_bottom)
    return Deal(
        hands=hands,
        bottom_cards=bottom,
        douzero_hands=cast(
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            tuple(tuple(rank_counts_to_douzero_cards(hand)) for hand in hands),
        ),
        douzero_bottom_cards=tuple(rank_counts_to_douzero_cards(bottom)),
    )
