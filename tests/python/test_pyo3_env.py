"""End-to-end acceptance tests for the E010 PyO3 single environment."""

import json
from pathlib import Path
from typing import cast

import pytest

from birddou import PyDdzEnv, RuleConfig, load_rule_config, parse_rule_config
from birddou.env_types import PlayGameAction

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"


def rules() -> RuleConfig:
    """Return a mutable rule dictionary for one test."""
    return load_rule_config(RULE_CONFIG_PATH)


def play_payload(action: object) -> dict[str, object]:
    """Narrow a post-bid action to its serialized move payload."""
    play = cast(PlayGameAction, action)["play"]
    return cast(dict[str, object], play)


def test_reset_is_seeded_reproducible_and_information_set_safe() -> None:
    """The native deal is deterministic and observations do not reveal allocations."""
    first = PyDdzEnv()
    second = PyDdzEnv()
    third = PyDdzEnv()

    observation = first.reset(7, rules())
    second.reset(7, rules())
    third.reset(8, rules())

    assert first.seed == 7
    assert first.is_initialized
    assert first.current_player == 0
    assert not first.terminal
    assert "seed=Some(7)" in repr(first)
    assert first.serialize() == second.serialize()
    assert first.serialize() != third.serialize()
    assert observation["schema_version"] == 1
    assert observation["phase"] == "card_play"
    assert observation["observer"] == 0
    assert observation["role"] == "landlord"
    assert sum(observation["own_hand"]) == 20
    assert sum(observation["unknown_pool"]) == 34
    assert sum(observation["public_bottom_cards"]) == 3
    assert observation["cards_left"] == [20, 17, 17]
    assert "hands" not in observation

    envelope = json.loads(first.serialize())
    assert envelope["schema_version"] == 1
    assert envelope["state"]["history"] == []
    assert envelope["state"]["hands"] == [
        [1, 0, 2, 1, 1, 2, 2, 2, 0, 4, 2, 3, 0, 0, 0],
        [3, 1, 1, 2, 1, 1, 2, 0, 3, 0, 1, 0, 1, 0, 1],
        [0, 3, 1, 1, 2, 1, 0, 2, 1, 0, 1, 1, 3, 1, 0],
    ]
    assert envelope["state"]["bottom_cards"] == [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def test_invalid_calls_are_rejected_without_replacing_a_valid_game() -> None:
    """User input failures are explicit and reset remains transactional."""
    environment = PyDdzEnv()
    assert repr(environment) == "PyDdzEnv(uninitialized)"
    assert environment.seed is None
    assert not environment.is_initialized
    with pytest.raises(RuntimeError, match="reset first"):
        environment.legal_actions()
    with pytest.raises(RuntimeError, match="reset first"):
        environment.serialize()

    environment.reset(11, rules())
    original = environment.serialize()
    invalid_rules = rules()
    invalid_rules["rule_config_id"] = 0
    with pytest.raises(ValueError, match="rule_config_id"):
        environment.reset(12, invalid_rules)
    assert environment.seed == 11
    assert environment.serialize() == original
    with pytest.raises(ValueError, match="unknown field"):
        parse_rule_config(RULE_CONFIG_PATH.read_text(encoding="utf-8") + "\nunknown: true\n")
    with pytest.raises(ValueError, match="outside 0..=2"):
        environment.observe(3)


def test_serialized_state_restore_is_exact_transactional_and_branchable() -> None:
    """Training rollouts can branch from validated native state without Python rules."""
    source = PyDdzEnv()
    source.reset(17, rules())
    source.step(source.legal_actions()[0])
    snapshot = source.serialize()

    restored = PyDdzEnv()
    observation = restored.restore(snapshot, rules())
    assert observation == source.observe(source.current_player)
    assert restored.seed is None
    assert restored.serialize() == snapshot
    assert restored.legal_actions() == source.legal_actions()

    before = restored.serialize()
    with pytest.raises(ValueError, match="serialized"):
        restored.restore(b"not a state envelope", rules())
    assert restored.serialize() == before

    alternative = restored.legal_actions()[-1]
    restored.step(alternative)
    assert restored.serialize() != before
    assert source.serialize() == snapshot


def test_legal_actions_and_step_use_the_canonical_rust_protocol() -> None:
    """Actions round-trip through Python without duplicating legality logic."""
    environment = PyDdzEnv()
    observation = environment.reset(19, rules())
    actions = environment.legal_actions()

    assert actions
    assert all("play" in action for action in actions)
    assert all(len(cast(list[int], play_payload(action)["cards"])) == 15 for action in actions)
    assert all(play_payload(action)["kind"] != "pass" for action in actions)

    original = environment.serialize()
    illegal_pass: PlayGameAction = {
        "play": {
            "kind": "pass",
            "cards": [0] * 15,
            "main_rank": 15,
            "chain_len": 0,
            "total_cards": 0,
        }
    }
    with pytest.raises(ValueError, match="illegal"):
        environment.step(illegal_pass)
    assert environment.serialize() == original

    result = environment.step(actions[0])
    assert result["event"]["sequence"] == 0
    assert result["event"]["actor"] == observation["current_player"]
    assert result["event"]["action"] == actions[0]
    assert not result["terminal"]
    assert result["next_player"] == 1
    assert environment.current_player == 1

    farmer_view = environment.observe(1)
    assert farmer_view["observer"] == 1
    assert farmer_view["role"] == "farmer"
    assert sum(farmer_view["own_hand"]) == 17
    assert sum(farmer_view["unknown_pool"]) == sum(farmer_view["cards_left"]) - 17
    assert len(farmer_view["history"]) == 1


def test_a_complete_python_driven_game_terminates_with_payoff() -> None:
    """Repeated Python calls can drive a complete authoritative Rust game."""
    environment = PyDdzEnv()
    environment.reset(23, rules())
    steps = 0
    result = None

    while not environment.terminal:
        actions = environment.legal_actions()
        assert actions
        action = max(actions, key=lambda item: cast(int, play_payload(item)["total_cards"]))
        result = environment.step(action)
        steps += 1
        assert steps < 500

    assert result is not None
    assert result["terminal"]
    assert result["next_player"] is None
    assert sum(result["raw_payoff"]) == 0
    assert any(payoff > 0 for payoff in result["objective_payoff"])
    assert any(payoff < 0 for payoff in result["objective_payoff"])
    assert environment.legal_actions() == []
    with pytest.raises(RuntimeError, match="terminal"):
        environment.step(result["event"]["action"])
