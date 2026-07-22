"""Native exact solving and root-consistent hidden-state rollout tests."""

from pathlib import Path
from typing import cast

import pytest

from birddou import Action, PyDdzEnv, RuleConfig, load_rule_config, solve_endgame
from birddou.env_types import PlayGameAction
from birddou.eval.baselines import LongestMovePolicy
from birddou.search.endgame import (
    BeliefRolloutConfig,
    SearchPipelineConfig,
    SearchTriggerConfig,
    SearchValidationMetrics,
    evaluate_search_acceptance,
    evaluate_search_trigger,
    load_search_pipeline_config,
    materialize_hidden_states,
    root_consistent_belief_rollout,
    triggered_belief_rollout,
)

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs" / "rules" / "canonical_full.yaml"
SEARCH_CONFIG_PATH = ROOT / "configs" / "train" / "endgame_search.yaml"


def _rules() -> RuleConfig:
    return load_rule_config(RULES_PATH)


def _counts(cards: list[int]) -> list[int]:
    return [cards.count(rank) for rank in range(15)]


def _total_cards(action: Action) -> int:
    if "play" not in action:
        return 0
    return cast(PlayGameAction, action)["play"]["total_cards"]


def _late_root() -> tuple[PyDdzEnv, list[list[int]]]:
    rules = _rules()
    deck = [rank for rank in range(13) for _ in range(4)] + [13, 14]
    hands = [_counts(deck[:17]), _counts(deck[17:34]), _counts(deck[34:51])]
    bottom = _counts(deck[51:])
    current_hands = [hand.copy() for hand in hands]
    current_hands[0] = [
        owned + added for owned, added in zip(current_hands[0], bottom, strict=True)
    ]
    environment = PyDdzEnv()
    environment.reset_complete_deal(hands, bottom, 0, rules)
    environment.step(environment.legal_actions()[-1])
    for _ in range(3):
        environment.step(environment.legal_actions()[0])
    while (
        not environment.terminal
        and sum(environment.observe(environment.current_player)["cards_left"]) > 18
    ):
        action = max(
            environment.legal_actions(),
            key=_total_cards,
        )
        result = environment.step(action)
        event_action = result["event"]["action"]
        if "play" in event_action:
            played = cast(PlayGameAction, event_action)["play"]
            actor = result["event"]["actor"]
            for rank, used in enumerate(played["cards"]):
                current_hands[actor][rank] -= used
    assert not environment.terminal
    return environment, current_hands


def test_native_exact_solver_proves_endgame_without_mutating_serialized_root() -> None:
    environment, _ = _late_root()
    root = environment.serialize()
    total = sum(environment.observe(environment.current_player)["cards_left"])
    result = solve_endgame(root, _rules(), total, 1_000_000)

    assert result["best_action"] in environment.legal_actions()
    assert result["plies_to_terminal"] > 0
    assert result["nodes"] > 0
    assert environment.serialize() == root
    with pytest.raises(ValueError, match="above configured maximum"):
        solve_endgame(root, _rules(), total - 1, 1_000_000)


def test_materialized_samples_keep_root_actions_identical_and_force_all_comparisons() -> None:
    environment, hands = _late_root()
    rules = _rules()
    observer = environment.current_player
    root = environment.serialize()
    assignment_a = hands[(observer + 1) % 3]
    samples = materialize_hidden_states(root, rules, observer, (assignment_a, assignment_a))
    root_actions = tuple(environment.legal_actions())
    result = root_consistent_belief_rollout(
        samples,
        rules,
        observer,
        root_actions,
        LongestMovePolicy("rollout"),
        BeliefRolloutConfig(exact_max_total_cards=18),
    )

    assert samples == (root, root)
    assert result.sample_count == 2
    assert result.root_action_count == len(root_actions)
    assert len(result.values) == len(root_actions)
    assert all(value.exact_sample_count == 2 for value in result.values)
    assert 0 <= result.selected_action_index < len(root_actions)

    invalid = assignment_a.copy()
    invalid[0] += 1
    with pytest.raises(ValueError, match="cards"):
        materialize_hidden_states(root, rules, observer, (invalid,))


def test_search_trigger_is_public_bounded_and_disabled_outside_cardplay() -> None:
    environment, _ = _late_root()
    observation = environment.observe(environment.current_player)
    trigger = evaluate_search_trigger(
        observation,
        environment.legal_actions(),
        belief_entropy=1.0,
        config=SearchTriggerConfig(total_cards_threshold=18),
    )
    assert trigger.enabled and "total_cards" in trigger.reasons

    bidding = PyDdzEnv()
    bid_observation = bidding.reset(7, _rules())
    assert not evaluate_search_trigger(
        bid_observation, bidding.legal_actions(), belief_entropy=0.0
    ).enabled


def test_guarded_entry_point_never_rolls_out_outside_public_trigger() -> None:
    rules = _rules()
    environment = PyDdzEnv()
    observation = environment.reset(71, rules)
    config = SearchPipelineConfig(
        schema_version=1,
        enabled=True,
        trigger=SearchTriggerConfig(
            total_cards_threshold=1,
            player_cards_threshold=1,
            belief_entropy_threshold=0.0,
            bomb_decision_enabled=False,
        ),
        rollout=BeliefRolloutConfig(),
    )
    result = triggered_belief_rollout(
        observation,
        environment.legal_actions(),
        1.0,
        (),
        rules,
        observation["observer"],
        LongestMovePolicy("must-not-run"),
        config,
    )
    assert not result.trigger.enabled
    assert result.search is None


def test_search_config_and_acceptance_gate_require_positive_paired_evidence() -> None:
    config = load_search_pipeline_config(SEARCH_CONFIG_PATH)
    assert config.enabled
    assert config.trigger.player_cards_threshold == 5
    accepted = evaluate_search_acceptance(SearchValidationMetrics(100, 20, 20, 0, 0.01))
    rejected = evaluate_search_acceptance(SearchValidationMetrics(100, 20, 5, 1, 0.0))
    assert accepted.accepted and accepted.triggered_fraction == pytest.approx(0.2)
    assert not rejected.accepted and len(rejected.reasons) == 2
