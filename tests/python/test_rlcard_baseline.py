"""Public-state conversion and deterministic RLCard rule-baseline adapter tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from birddou import PyDdzEnv, load_rule_config
from birddou.eval import PolicyDecisionContext, RlcardRulePolicy

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"


class _CapturingRuleAgent:
    def __init__(self) -> None:
        self.raw_states: list[Mapping[str, object]] = []

    def step(self, state: Mapping[str, object]) -> str:
        raw = cast(Mapping[str, object], state["raw_obs"])
        self.raw_states.append(raw)
        actions = cast(list[str], raw["actions"])
        return str(np.random.choice(actions))


def test_rlcard_adapter_converts_only_public_information_and_restores_numpy_rng() -> None:
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    observation = environment.reset(77, rules)
    actions = tuple(environment.legal_actions())
    agent = _CapturingRuleAgent()
    policy = RlcardRulePolicy("rlcard", seed=9, agent=agent)
    context = PolicyDecisionContext(0, 77, "rlcard-test", 0, None, 0)

    np.random.seed(123)
    expected_next = float(np.random.random())
    np.random.seed(123)
    first = policy.select_action(observation, actions, context)
    actual_next = float(np.random.random())
    second = policy.select_action(observation, actions, context)

    assert first == second
    assert actual_next == expected_next
    assert 0 <= first < len(actions)
    raw = agent.raw_states[0]
    assert raw["self"] == observation["observer"]
    assert raw["current_hand"] != raw["others_hand"]
    assert len(cast(str, raw["current_hand"])) == sum(observation["own_hand"])
    assert len(cast(str, raw["others_hand"])) == sum(observation["unknown_pool"])
    assert "hands" not in raw


def test_pinned_official_rlcard_rule_agent_selects_a_native_legal_action_when_installed() -> None:
    pytest.importorskip("rlcard")
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    observation = environment.reset(78, rules)
    actions = tuple(environment.legal_actions())
    selected = RlcardRulePolicy("official", seed=10).select_action(
        observation,
        actions,
        PolicyDecisionContext(0, 78, "official-rlcard", 0, None, 0),
    )
    assert 0 <= selected < len(actions)
