"""Public-information and process-boundary tests for the PerfectDou adapter."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from birddou import PyDdzEnv, load_rule_config
from birddou.eval import (
    PerfectDouPolicy,
    PerfectDouProcessBackend,
    PolicyDecisionContext,
    role_for_game_seat,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
FULL_RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "canonical_full.yaml"
ECHO_WORKER = """
import json
import sys
for line in sys.stdin:
    request = json.loads(line)
    legal = request["public_infoset"]["legal_actions"]
    print(json.dumps({"action": legal[0]}), flush=True)
"""


class _CapturingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    def select_action(self, role: str, public_infoset: Mapping[str, object]) -> list[int]:
        self.calls.append((role, public_infoset))
        legal = cast(list[list[int]], public_infoset["legal_actions"])
        return legal[0]


def test_perfectdou_policy_exposes_only_public_and_local_information() -> None:
    environment = PyDdzEnv()
    observation = environment.reset(501, load_rule_config(RULES_PATH))
    legal_actions = tuple(environment.legal_actions())
    backend = _CapturingBackend()
    policy = PerfectDouPolicy("perfectdou-test", backend)

    selected = policy.select_action(
        observation,
        legal_actions,
        PolicyDecisionContext(
            0,
            501,
            "perfectdou-public-boundary",
            0,
            role_for_game_seat(0, 0),
            0,
        ),
    )

    assert selected == 0
    assert len(backend.calls) == 1
    role, infoset = backend.calls[0]
    assert role == "landlord"
    assert set(infoset) == {
        "player_position",
        "player_hand_cards",
        "other_hand_cards",
        "num_cards_left_dict",
        "three_landlord_cards",
        "card_play_action_seq",
        "legal_actions",
        "last_move",
        "last_move_dict",
        "played_cards",
        "bomb_num",
        "last_pid",
    }
    assert len(cast(list[int], infoset["player_hand_cards"])) == 20
    assert len(cast(list[int], infoset["other_hand_cards"])) == 34
    assert "hands" not in infoset
    assert "all_handcards" not in infoset


def test_perfectdou_process_backend_uses_versioned_jsonl_protocol() -> None:
    backend = PerfectDouProcessBackend(
        (sys.executable, "-u", "-c", ECHO_WORKER),
        timeout_seconds=5.0,
    )
    try:
        assert backend.select_action(
            "landlord",
            {"legal_actions": [[3, 3], [4]], "player_hand_cards": [3, 3, 4]},
        ) == [3, 3]
        assert (
            backend.select_action(
                "landlord_up",
                {"legal_actions": [[], [17]], "player_hand_cards": [17]},
            )
            == []
        )
    finally:
        backend.close()


def test_perfectdou_policy_maps_dynamic_landlord_roles_for_complete_games() -> None:
    environment = PyDdzEnv()
    rules = load_rule_config(FULL_RULES_PATH)
    for seed in range(100):
        observation = environment.reset(seed, rules)
        if observation["current_player"] != 0:
            break
    else:
        raise AssertionError("test seeds did not produce a non-zero first bidder")
    while observation["phase"] != "card_play":
        actions = tuple(environment.legal_actions())
        if observation["phase"] == "bidding":
            selected = next(
                index for index, action in enumerate(actions) if action.get("bid") == {"score": 3}
            )
        else:
            selected = next(
                index for index, action in enumerate(actions) if action.get("double") == "decline"
            )
        environment.step(actions[selected])
        observation = environment.observe(environment.current_player)
    landlord = observation["landlord"]
    assert landlord is not None and landlord != 0
    seat = observation["observer"]
    role = role_for_game_seat(seat, landlord)
    backend = _CapturingBackend()

    selected = PerfectDouPolicy("dynamic-perfectdou", backend).select_action(
        observation,
        tuple(environment.legal_actions()),
        PolicyDecisionContext(0, seed, "dynamic-perfectdou", seat, role, 0),
    )

    assert selected == 0
    assert backend.calls[0][0] == role.value
    counts = cast(dict[str, int], backend.calls[0][1]["num_cards_left_dict"])
    assert counts["landlord"] == 20
    assert counts["landlord_down"] == 17
    assert counts["landlord_up"] == 17
