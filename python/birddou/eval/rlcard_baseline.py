"""Information-set-safe adapter for RLCard's official doudizhu-rule-v1 model."""

from __future__ import annotations

import operator
import threading
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Protocol, cast

import numpy as np

from birddou.env_types import Action, Observation, PlayGameAction
from birddou.eval.baselines import PolicyDecisionContext
from birddou.eval.paired_deals import splitmix64

RLCARD_BASELINE_ID = "rlcard_rule_v1"
RLCARD_PINNED_VERSION = "1.0.7"
_RANK_SYMBOLS = "3456789TJQKA2BR"
_NUMPY_RANDOM_LOCK = threading.Lock()


class RlcardBaselineUnavailable(ImportError):
    """The pinned optional RLCard package is not installed or is incompatible."""


class RlcardRuleAgent(Protocol):
    """Minimal raw-state surface of RLCard's DouDizhuRuleAgentV1."""

    def step(self, state: Mapping[str, object]) -> str: ...


class RlcardRulePolicy:
    """Run the official rule agent on a converted public BIRD-Dou observation."""

    def __init__(
        self,
        policy_id: str,
        seed: int = 0,
        *,
        agent: RlcardRuleAgent | None = None,
    ) -> None:
        if not policy_id:
            raise ValueError("RLCard policy_id must be non-empty")
        if not 0 <= seed < 1 << 64:
            raise ValueError("RLCard policy seed must fit uint64")
        self._policy_id = policy_id
        self._seed = seed
        self._agent = _load_agent() if agent is None else agent

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        """Convert only public/local fields, then map the selected raw action exactly."""
        if observation["phase"] != "card_play":
            raise ValueError("RLCard rule baseline supports post-bid card play only")
        if observation["observer"] != context.seat:
            raise ValueError("RLCard observation seat differs from decision context")
        if observation["landlord"] is None:
            raise ValueError("RLCard rule baseline requires a resolved landlord")
        action_strings = tuple(_action_string(action) for action in legal_actions)
        if len(set(action_strings)) != len(action_strings):
            raise ValueError("RLCard action conversion is not one-to-one")
        raw_state: dict[str, object] = {
            "seen_cards": _counts_string(observation["public_bottom_cards"]),
            "landlord": observation["landlord"],
            "trace": [
                (event["actor"], _action_string(event["action"]))
                for event in observation["history"]
                if "play" in event["action"]
            ],
            "played_cards": [_counts_string(counts) for counts in observation["public_played"]],
            "self": observation["observer"],
            "current_hand": _counts_string(observation["own_hand"]),
            "others_hand": _counts_string(observation["unknown_pool"]),
            "num_cards_left": list(observation["cards_left"]),
            "actions": list(action_strings),
        }
        wrapper: Mapping[str, object] = {"raw_obs": raw_state}
        random_seed = splitmix64(
            self._seed ^ context.deal_seed ^ (context.seat << 16) ^ context.decision_index
        )
        # RLCard v1 uses NumPy's module-global RNG only in its final fallback.
        # Serialize and restore that state so evaluation stays seeded and side-effect free.
        with _NUMPY_RANDOM_LOCK:
            numpy_state = np.random.get_state()
            try:
                np.random.seed(random_seed & 0xFFFF_FFFF)
                selected = self._agent.step(wrapper)
            finally:
                np.random.set_state(numpy_state)
        try:
            return action_strings.index(selected)
        except ValueError as error:
            raise RuntimeError(
                f"RLCard rule agent returned action outside canonical legal set: {selected!r}"
            ) from error


def _load_agent() -> RlcardRuleAgent:
    try:
        rlcard = import_module("rlcard")
        rule_models = import_module("rlcard.models.doudizhu_rule_models")
    except ImportError as error:
        raise RlcardBaselineUnavailable(
            'RLCard rule baseline requires `pip install "bird-dou[rlcard]"`'
        ) from error
    version = getattr(rlcard, "__version__", None)
    if version is not None and version != RLCARD_PINNED_VERSION:
        raise RlcardBaselineUnavailable(
            f"RLCard version {version!r} differs from pinned {RLCARD_PINNED_VERSION}"
        )
    factory = cast(Callable[[], object], operator.attrgetter("DouDizhuRuleAgentV1")(rule_models))
    return cast(RlcardRuleAgent, factory())


def _action_string(action: Action) -> str:
    if "play" not in action:
        raise ValueError("RLCard rule baseline received a non-cardplay action")
    play = cast(PlayGameAction, action)["play"]
    return "pass" if play["kind"] == "pass" else _counts_string(play["cards"])


def _counts_string(counts: Sequence[int]) -> str:
    if len(counts) != 15:
        raise ValueError("RLCard rank counts must have width 15")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("RLCard rank counts must be non-negative integers")
    return "".join(symbol * count for symbol, count in zip(_RANK_SYMBOLS, counts, strict=True))


__all__ = (
    "RLCARD_BASELINE_ID",
    "RLCARD_PINNED_VERSION",
    "RlcardBaselineUnavailable",
    "RlcardRuleAgent",
    "RlcardRulePolicy",
)
