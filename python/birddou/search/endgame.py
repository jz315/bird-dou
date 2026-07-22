"""Triggered root-consistent belief rollout over replay-valid native samples."""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from birddou.env import PyDdzEnv, solve_endgame
from birddou.env_types import Action, Observation, PlayGameAction, RuleConfig, StepResult
from birddou.eval.baselines import Policy, PolicyDecisionContext
from birddou.eval.paired_deals import role_for_game_seat

SEARCH_PIPELINE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SearchTriggerConfig:
    """Public-information conditions under which search is allowed to run."""

    total_cards_threshold: int = 18
    player_cards_threshold: int = 5
    belief_entropy_threshold: float = 0.25
    bomb_decision_enabled: bool = True

    def __post_init__(self) -> None:
        if self.total_cards_threshold <= 0 or self.player_cards_threshold <= 0:
            raise ValueError("search card thresholds must be positive")
        if not math.isfinite(self.belief_entropy_threshold) or self.belief_entropy_threshold < 0:
            raise ValueError("search belief entropy threshold must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BeliefRolloutConfig:
    """Bounded rollout, exact-solve, and risk aggregation controls."""

    max_actions: int = 500
    exact_max_total_cards: int = 12
    exact_max_nodes: int = 1_000_000
    risk_coefficient: float = 0.25
    score_scale: float = 16.0

    def __post_init__(self) -> None:
        if self.max_actions <= 0 or self.exact_max_total_cards <= 0 or self.exact_max_nodes <= 0:
            raise ValueError("belief rollout limits must be positive")
        if not math.isfinite(self.risk_coefficient) or self.risk_coefficient < 0.0:
            raise ValueError("belief rollout risk coefficient must be finite and non-negative")
        if not math.isfinite(self.score_scale) or self.score_scale <= 0.0:
            raise ValueError("belief rollout score scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class SearchPipelineConfig:
    """Versioned switch and controls for optional online endgame search."""

    schema_version: int
    enabled: bool
    trigger: SearchTriggerConfig
    rollout: BeliefRolloutConfig

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_PIPELINE_SCHEMA_VERSION:
            raise ValueError("unsupported search pipeline schema")


@dataclass(frozen=True, slots=True)
class SearchTrigger:
    """Whether and why this public decision qualifies for optional search."""

    enabled: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RootActionSearchValue:
    """One root action aggregated across the identical hidden-state sample set."""

    action_index: int
    expected_value: float
    win_probability: float
    standard_deviation: float
    risk_adjusted_value: float
    exact_sample_count: int


@dataclass(frozen=True, slots=True)
class RootConsistentSearchResult:
    """Search selection with all root candidates evaluated on all samples."""

    selected_action_index: int
    values: tuple[RootActionSearchValue, ...]
    sample_count: int
    root_action_count: int

    def __post_init__(self) -> None:
        if self.sample_count <= 0 or self.root_action_count <= 0:
            raise ValueError("root-consistent search counts must be positive")
        if len(self.values) != self.root_action_count:
            raise ValueError("root-consistent search omitted a root action")
        if not 0 <= self.selected_action_index < self.root_action_count:
            raise ValueError("root-consistent selected action is invalid")


@dataclass(frozen=True, slots=True)
class TriggeredSearchResult:
    """Trigger audit plus an optional search result from the guarded entry point."""

    trigger: SearchTrigger
    search: RootConsistentSearchResult | None

    def __post_init__(self) -> None:
        if self.trigger.enabled != (self.search is not None):
            raise ValueError("triggered search result disagrees with its trigger")


@dataclass(frozen=True, slots=True)
class SearchValidationMetrics:
    """Paired-strength and trigger-boundary observations from an evaluation run."""

    evaluated_states: int
    triggered_states: int
    searched_states: int
    searched_outside_trigger: int
    paired_strength_delta_ci_lower: float

    def __post_init__(self) -> None:
        counts = (
            self.evaluated_states,
            self.triggered_states,
            self.searched_states,
            self.searched_outside_trigger,
        )
        if any(value < 0 for value in counts):
            raise ValueError("search validation counts must be non-negative")
        if self.triggered_states > self.evaluated_states:
            raise ValueError("triggered search states exceed evaluated states")
        if self.searched_states > self.evaluated_states:
            raise ValueError("searched states exceed evaluated states")
        if self.searched_outside_trigger > self.searched_states:
            raise ValueError("outside-trigger searches exceed all searches")
        if not math.isfinite(self.paired_strength_delta_ci_lower):
            raise ValueError("search paired lower confidence bound must be finite")


@dataclass(frozen=True, slots=True)
class SearchAcceptanceReport:
    """Whether search is trigger-safe and statistically better than the pure network."""

    accepted: bool
    triggered_fraction: float
    paired_strength_delta_ci_lower: float
    reasons: tuple[str, ...]


def load_search_pipeline_config(path: Path) -> SearchPipelineConfig:
    """Load the JSON-subset YAML switch, trigger thresholds, and rollout limits."""
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "search config")
    trigger = _mapping(root.get("trigger"), "search trigger config")
    rollout = _mapping(root.get("rollout"), "search rollout config")
    return SearchPipelineConfig(
        schema_version=_integer(root, "schema_version"),
        enabled=_boolean(root, "enabled"),
        trigger=SearchTriggerConfig(
            total_cards_threshold=_integer(trigger, "total_cards_threshold"),
            player_cards_threshold=_integer(trigger, "player_cards_threshold"),
            belief_entropy_threshold=_number(trigger, "belief_entropy_threshold"),
            bomb_decision_enabled=_boolean(trigger, "bomb_decision_enabled"),
        ),
        rollout=BeliefRolloutConfig(
            max_actions=_integer(rollout, "max_actions"),
            exact_max_total_cards=_integer(rollout, "exact_max_total_cards"),
            exact_max_nodes=_integer(rollout, "exact_max_nodes"),
            risk_coefficient=_number(rollout, "risk_coefficient"),
            score_scale=_number(rollout, "score_scale"),
        ),
    )


def evaluate_search_trigger(
    observation: Observation,
    legal_actions: Sequence[Action],
    belief_entropy: float,
    config: SearchTriggerConfig | None = None,
) -> SearchTrigger:
    """Apply only public card-count, bomb-decision, and belief-entropy triggers."""
    settings = config if config is not None else SearchTriggerConfig()
    if observation["phase"] != "card_play" or observation["landlord"] is None:
        return SearchTrigger(False, ())
    if not math.isfinite(belief_entropy) or belief_entropy < 0.0:
        raise ValueError("belief entropy must be finite and non-negative")
    reasons: list[str] = []
    if sum(observation["cards_left"]) <= settings.total_cards_threshold:
        reasons.append("total_cards")
    if min(observation["cards_left"]) <= settings.player_cards_threshold:
        reasons.append("player_cards")
    if settings.bomb_decision_enabled and any(_is_bomb_action(action) for action in legal_actions):
        reasons.append("bomb_or_rocket")
    if belief_entropy <= settings.belief_entropy_threshold:
        reasons.append("low_belief_entropy")
    return SearchTrigger(bool(reasons), tuple(reasons))


def materialize_hidden_states(
    serialized_root: bytes,
    rules: RuleConfig,
    observer: int,
    assignments_a: Sequence[Sequence[int]],
) -> tuple[bytes, ...]:
    """Turn constrained container-A samples into replay-valid full native roots."""
    if not assignments_a:
        raise ValueError("belief state materialization requires at least one assignment")
    states: list[bytes] = []
    environment = PyDdzEnv()
    for assignment in assignments_a:
        environment.restore_with_hidden_sample(
            serialized_root,
            rules,
            observer,
            list(assignment),
        )
        states.append(environment.serialize())
    return tuple(states)


def root_consistent_belief_rollout(
    sampled_states: Sequence[bytes],
    rules: RuleConfig,
    observer: int,
    root_actions: Sequence[Action],
    continuation_policy: Policy,
    config: BeliefRolloutConfig | None = None,
) -> RootConsistentSearchResult:
    """Force every root action through every hidden sample, then aggregate outcomes."""
    settings = config if config is not None else BeliefRolloutConfig()
    if not sampled_states or not root_actions:
        raise ValueError("belief rollout requires samples and root actions")
    values_by_action: list[list[float]] = [[] for _ in root_actions]
    wins_by_action: list[list[float]] = [[] for _ in root_actions]
    exact_by_action = [0] * len(root_actions)
    environment = PyDdzEnv()
    for sample_index, state in enumerate(sampled_states):
        observation = environment.restore(state, rules)
        if observation["observer"] != observer or observation["current_player"] != observer:
            raise ValueError("sample root does not belong to the requested acting observer")
        sample_actions = tuple(environment.legal_actions())
        if sample_actions != tuple(root_actions):
            raise ValueError("hidden samples do not share one identical legal root action set")
        for action_index, root_action in enumerate(root_actions):
            environment.restore(state, rules)
            result = environment.step(root_action)
            total_cards = sum(environment.observe(environment.current_player)["cards_left"])
            if not environment.terminal and total_cards <= settings.exact_max_total_cards:
                exact = solve_endgame(
                    environment.serialize(),
                    rules,
                    settings.exact_max_total_cards,
                    settings.exact_max_nodes,
                )
                landlord = environment.observe(environment.current_player)["landlord"]
                if landlord is None:
                    raise RuntimeError("exact belief rollout has no resolved landlord")
                observer_landlord = observer == landlord
                observer_win = exact["landlord_forced_win"] == observer_landlord
                value = 1.0 if observer_win else -1.0
                win = float(observer_win)
                exact_by_action[action_index] += 1
            else:
                result = _roll_to_terminal(
                    environment,
                    rules,
                    continuation_policy,
                    observer,
                    sample_index,
                    action_index,
                    result,
                    settings,
                )
                payoff = result["raw_payoff"][observer]
                value = math.tanh(payoff / settings.score_scale)
                win = float(payoff > 0)
            values_by_action[action_index].append(value)
            wins_by_action[action_index].append(win)
    aggregated: list[RootActionSearchValue] = []
    for action_index, (action_values, action_wins) in enumerate(
        zip(values_by_action, wins_by_action, strict=True)
    ):
        mean = sum(action_values) / len(action_values)
        variance = sum((value - mean) ** 2 for value in action_values) / len(action_values)
        deviation = math.sqrt(variance)
        aggregated.append(
            RootActionSearchValue(
                action_index=action_index,
                expected_value=mean,
                win_probability=sum(action_wins) / len(action_wins),
                standard_deviation=deviation,
                risk_adjusted_value=mean - settings.risk_coefficient * deviation,
                exact_sample_count=exact_by_action[action_index],
            )
        )
    selected = max(
        range(len(aggregated)),
        key=lambda index: (aggregated[index].risk_adjusted_value, -index),
    )
    return RootConsistentSearchResult(
        selected_action_index=selected,
        values=tuple(aggregated),
        sample_count=len(sampled_states),
        root_action_count=len(root_actions),
    )


def triggered_belief_rollout(
    observation: Observation,
    legal_actions: Sequence[Action],
    belief_entropy: float,
    sampled_states: Sequence[bytes],
    rules: RuleConfig,
    observer: int,
    continuation_policy: Policy,
    config: SearchPipelineConfig,
) -> TriggeredSearchResult:
    """Run rollout only when the enabled pipeline's public trigger allows it."""
    if not config.enabled:
        return TriggeredSearchResult(SearchTrigger(False, ()), None)
    trigger = evaluate_search_trigger(
        observation,
        legal_actions,
        belief_entropy,
        config.trigger,
    )
    if not trigger.enabled:
        return TriggeredSearchResult(trigger, None)
    result = root_consistent_belief_rollout(
        sampled_states,
        rules,
        observer,
        legal_actions,
        continuation_policy,
        config.rollout,
    )
    return TriggeredSearchResult(trigger, result)


def evaluate_search_acceptance(
    metrics: SearchValidationMetrics,
) -> SearchAcceptanceReport:
    """Require trigger-only use and a positive paired 95% CI lower bound."""
    reasons: list[str] = []
    if metrics.searched_states == 0:
        reasons.append("evaluation observed no searched states")
    if metrics.searched_outside_trigger != 0:
        reasons.append("search ran outside its public trigger boundary")
    if metrics.paired_strength_delta_ci_lower <= 0.0:
        reasons.append("search did not beat the pure network with a positive paired lower bound")
    fraction = (
        0.0
        if metrics.evaluated_states == 0
        else metrics.triggered_states / metrics.evaluated_states
    )
    return SearchAcceptanceReport(
        accepted=not reasons,
        triggered_fraction=fraction,
        paired_strength_delta_ci_lower=metrics.paired_strength_delta_ci_lower,
        reasons=tuple(reasons),
    )


def _roll_to_terminal(
    environment: PyDdzEnv,
    rules: RuleConfig,
    policy: Policy,
    observer: int,
    sample_index: int,
    action_index: int,
    result: StepResult,
    config: BeliefRolloutConfig,
) -> StepResult:
    decision_counts = [0, 0, 0]
    steps = 1
    while not environment.terminal:
        if steps >= config.max_actions:
            raise RuntimeError("belief rollout exceeded max_actions")
        seat = environment.current_player
        observation = environment.observe(seat)
        actions = tuple(environment.legal_actions())
        landlord = observation["landlord"]
        if landlord is None:
            raise RuntimeError("card-play belief rollout lost landlord assignment")
        context = PolicyDecisionContext(
            deal_index=sample_index,
            deal_seed=sample_index,
            match_id=f"belief-search-{observer}-{sample_index}-{action_index}",
            seat=seat,
            role=role_for_game_seat(seat, landlord),
            decision_index=decision_counts[seat],
        )
        selected = policy.select_action(observation, actions, context)
        if isinstance(selected, bool):
            raise RuntimeError("belief rollout policy returned a boolean")
        try:
            local_index = operator.index(selected)
        except TypeError as error:
            raise RuntimeError("belief rollout policy returned a non-integer") from error
        if not 0 <= local_index < len(actions):
            raise RuntimeError("belief rollout policy returned an invalid action index")
        result = environment.step(actions[local_index])
        decision_counts[seat] += 1
        steps += 1
    return result


def _is_bomb_action(action: Action) -> bool:
    if "play" not in action:
        return False
    return cast(PlayGameAction, action)["play"]["kind"] in ("bomb", "rocket")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"search config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"search config {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"search config {key} must be finite")
    return number


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"search config {key} must be boolean")
    return value


__all__ = (
    "SEARCH_PIPELINE_SCHEMA_VERSION",
    "BeliefRolloutConfig",
    "RootActionSearchValue",
    "RootConsistentSearchResult",
    "SearchAcceptanceReport",
    "SearchPipelineConfig",
    "SearchTrigger",
    "SearchTriggerConfig",
    "SearchValidationMetrics",
    "TriggeredSearchResult",
    "evaluate_search_acceptance",
    "evaluate_search_trigger",
    "load_search_pipeline_config",
    "materialize_hidden_states",
    "root_consistent_belief_rollout",
    "triggered_belief_rollout",
)
