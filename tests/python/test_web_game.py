from pathlib import Path

import pytest

from birddou.eval.baselines import FixedBidPolicy
from birddou.models.baseline_douzero import OfficialDouZeroPolicy
from birddou.web.game import GameService, WebGameError

ROOT = Path(__file__).resolve().parents[2]


def test_web_game_exposes_only_information_safe_state() -> None:
    service = GameService(ROOT)
    state = service.create_game(seed=20260722, human_seat=0)

    assert state["schemaVersion"] == 1
    assert state["gameId"]
    hand = state["hand"]
    cards_left = state["cardsLeft"]
    assert isinstance(hand, list) and len(hand) == 15
    assert isinstance(cards_left, list) and len(cards_left) == 3
    assert "hands" not in state
    assert "unknownPool" not in state
    assert "seed" not in state


def test_human_can_drive_a_complete_browser_game() -> None:
    service = GameService(ROOT)
    state = service.create_game(seed=42, human_seat=0)
    steps = 0
    while not state["terminal"]:
        assert state["humanTurn"]
        actions = state["legalActions"]
        assert isinstance(actions, list) and actions
        action = max(actions, key=lambda item: item["totalCards"])
        state = service.play(str(state["gameId"]), int(action["index"]))
        steps += 1
        assert steps < 200

    result = state["result"]
    assert isinstance(result, dict)
    assert isinstance(result["message"], str)


def test_invalid_or_stale_web_action_is_rejected() -> None:
    service = GameService(ROOT)
    state = service.create_game(seed=19, human_seat=0)
    assert state["humanTurn"]
    with pytest.raises(WebGameError, match="失效"):
        service.play(str(state["gameId"]), 100_000)


def test_web_game_lists_checkpoint_mode_without_loading_it() -> None:
    service = GameService(ROOT)
    modes = service.available_modes()
    assert modes[0]["id"] == "heuristic"
    assert modes[0]["recommended"] is True


def test_web_game_loads_official_douzero_when_local_artifacts_exist() -> None:
    service = GameService(ROOT)
    if not service._douzero_weight_set_available("douzero_ADP"):
        pytest.skip("official DouZero artifacts are not available locally")

    modes = {str(mode["id"]) for mode in service.available_modes()}
    assert "douzero_adp" in modes
    if service._douzero_weight_set_available("douzero_WP"):
        assert "douzero_wp" in modes

    policy = service._policy("douzero_adp")
    assert isinstance(policy, FixedBidPolicy)
    assert isinstance(policy.cardplay, OfficialDouZeroPolicy)
    assert policy.cardplay.checkpoint_set_name == "douzero_ADP"
    assert service._policy("douzero_adp") is policy
