from typing import cast

import pytest

from birddou.web.game import WebGameError
from birddou.web.guandan import GuandanService


def test_guandan_web_state_is_information_safe() -> None:
    service = GuandanService()
    state = service.create_game(seed=20260723, human_seat=0)

    assert state["schemaVersion"] == 1
    assert len(cast(list[int], state["hand"])) == 27
    assert len(cast(list[int], state["cardsLeft"])) == 4
    assert "hands" not in state
    assert "seed" not in state
    assert state["humanTurn"] is True


def test_human_can_drive_a_complete_guandan_game() -> None:
    service = GuandanService()
    state = service.create_game(seed=42, human_seat=0)
    turns = 0

    while state["result"] is None:
        actions = state["legalActions"]
        assert isinstance(actions, list) and actions
        selected = max(actions, key=lambda action: action["totalCards"])
        cards = None if selected["kind"] == "pass" else selected["cards"]
        state = service.play(str(state["gameId"]), cards)
        turns += 1
        assert turns < 200

    result = state["result"]
    assert isinstance(result, dict)
    assert len(result["finishOrder"]) == 4


def test_guandan_rejects_stale_or_foreign_cards_transactionally() -> None:
    service = GuandanService()
    state = service.create_game(seed=99, human_seat=0)
    hand = set(cast(list[int], state["hand"]))
    foreign = next(card for card in range(108) if card not in hand)

    with pytest.raises(WebGameError, match="出牌不合法"):
        service.play(str(state["gameId"]), [foreign])

    restored = service.get_game(str(state["gameId"]))
    assert restored["hand"] == state["hand"]
    assert restored["currentPlayer"] == state["currentPlayer"]
