"""Information-safe state adapter for a human-versus-AI browser game."""

from __future__ import annotations

import math
import secrets
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from birddou import PyDdzEnv, load_rule_config
from birddou.cli.policy_artifacts import load_full_game_checkpoint_policy
from birddou.env_types import (
    Action,
    BidGameAction,
    Observation,
    PlayGameAction,
    RuleConfig,
    StepResult,
)
from birddou.eval.baselines import Policy, PolicyDecisionContext
from birddou.eval.paired_deals import role_for_game_seat

RANK_LABELS = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "小王", "大王")
MOVE_LABELS = {
    "pass": "不出",
    "single": "单张",
    "pair": "对子",
    "triple": "三张",
    "triple_with_single": "三带一",
    "triple_with_pair": "三带二",
    "straight": "顺子",
    "pair_straight": "连对",
    "airplane": "飞机",
    "airplane_with_singles": "飞机带单",
    "airplane_with_pairs": "飞机带对",
    "four_with_two_singles": "四带二",
    "four_with_two_pairs": "四带两对",
    "bomb": "炸弹",
    "rocket": "王炸",
}


class WebGameError(RuntimeError):
    """A browser game request violated the public game contract."""


class FriendlyRulePolicy:
    """Fast deterministic opponent that makes legal, reasonably human-like moves."""

    def __init__(self, policy_id: str = "friendly_rule_v1") -> None:
        self._policy_id = policy_id

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        del context
        if not legal_actions:
            raise ValueError("rule opponent received no legal actions")
        phase = observation["phase"]
        if phase == "bidding":
            return self._select_bid(observation, legal_actions)
        if phase == "doubling":
            return self._select_double(observation, legal_actions)
        if phase != "card_play":
            raise ValueError(f"rule opponent cannot act in phase {phase}")
        return self._select_play(observation, legal_actions)

    @staticmethod
    def _hand_strength(hand: Sequence[int]) -> float:
        bombs = sum(count == 4 for count in hand[:13])
        rocket = int(hand[13] > 0 and hand[14] > 0)
        high_cards = hand[11] + 2 * hand[12] + 2 * hand[13] + 3 * hand[14]
        triples = sum(count >= 3 for count in hand[:13])
        return high_cards + 3.0 * bombs + 4.0 * rocket + 0.6 * triples

    def _select_bid(self, observation: Observation, legal_actions: Sequence[Action]) -> int:
        strength = self._hand_strength(observation["own_hand"])
        target = 3 if strength >= 10.0 else 2 if strength >= 7.0 else 1 if strength >= 4.5 else 0
        best_pass = 0
        candidates: list[tuple[int, int]] = []
        for index, action in enumerate(legal_actions):
            bid = action.get("bid")
            if bid == "pass":
                best_pass = index
            elif isinstance(bid, Mapping):
                score = bid.get("score")
                if isinstance(score, int) and score <= target:
                    candidates.append((score, index))
            elif bid in ("call", "rob") and target > 0:
                candidates.append((1 if bid == "call" else 2, index))
        return max(candidates, default=(0, best_pass))[1]

    def _select_double(self, observation: Observation, legal_actions: Sequence[Action]) -> int:
        wants_double = self._hand_strength(observation["own_hand"]) >= 8.0
        desired = "double" if wants_double else "decline"
        for index, action in enumerate(legal_actions):
            if action.get("double") == desired:
                return index
        return 0

    @staticmethod
    def _select_play(observation: Observation, legal_actions: Sequence[Action]) -> int:
        plays = [
            (index, cast(PlayGameAction, action)["play"])
            for index, action in enumerate(legal_actions)
        ]
        non_pass = [(index, move) for index, move in plays if move["kind"] != "pass"]
        if not non_pass:
            return plays[0][0]

        last_actor = _last_non_pass_actor(observation)
        landlord = observation["landlord"]
        actor = observation["observer"]
        teammate_led = (
            landlord is not None
            and actor != landlord
            and last_actor is not None
            and last_actor != landlord
            and last_actor != actor
        )
        pass_index = next((index for index, move in plays if move["kind"] == "pass"), None)
        if teammate_led and pass_index is not None:
            return pass_index

        leading = pass_index is None
        if leading:
            return max(
                non_pass,
                key=lambda item: (
                    item[1]["total_cards"],
                    -int(item[1]["kind"] in ("bomb", "rocket")),
                    -item[1]["main_rank"],
                ),
            )[0]
        return min(
            non_pass,
            key=lambda item: (
                int(item[1]["kind"] in ("bomb", "rocket")),
                item[1]["total_cards"],
                item[1]["main_rank"],
            ),
        )[0]


@dataclass(slots=True)
class GameSession:
    """One isolated native game with automatic opponent turns."""

    game_id: str
    seed: int
    human_seat: int
    ai_mode: str
    rules: RuleConfig
    ai_policy: Policy
    environment: PyDdzEnv = field(default_factory=PyDdzEnv)
    decision_index: int = 0
    terminal_result: StepResult | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def start(self) -> dict[str, object]:
        with self.lock:
            self.environment.reset(self.seed, self.rules)
            self._run_ai_turns()
            return self.public_state()

    def play(self, action_index: int) -> dict[str, object]:
        with self.lock:
            if self.environment.terminal:
                raise WebGameError("本局已经结束，请开始新游戏")
            if self.environment.current_player != self.human_seat:
                raise WebGameError("现在不是你的回合")
            legal_actions = self.environment.legal_actions()
            if not 0 <= action_index < len(legal_actions):
                raise WebGameError("所选动作已经失效，请重新选择")
            self.terminal_result = self.environment.step(legal_actions[action_index])
            self.decision_index += 1
            self._run_ai_turns()
            return self.public_state()

    def public_state(self) -> dict[str, object]:
        """Return only the human player's information set and public history."""
        observation = self.environment.observe(self.human_seat)
        human_turn = not self.environment.terminal and (
            self.environment.current_player == self.human_seat
        )
        actions = self.environment.legal_actions() if human_turn else []
        landlord = observation["landlord"]
        multiplier = 1 << observation["multiplier_exp"]
        return {
            "schemaVersion": 1,
            "gameId": self.game_id,
            "seed": self.seed,
            "humanSeat": self.human_seat,
            "aiMode": self.ai_mode,
            "phase": observation["phase"],
            "humanTurn": human_turn,
            "terminal": self.environment.terminal,
            "currentPlayer": None if self.environment.terminal else self.environment.current_player,
            "landlord": landlord,
            "role": observation["role"],
            "hand": list(observation["own_hand"]),
            "cardsLeft": list(observation["cards_left"]),
            "bottomCards": list(observation["public_bottom_cards"]),
            "multiplier": multiplier,
            "bombCount": observation["bomb_count"],
            "lastNonPass": (
                None
                if observation["last_non_pass"] is None
                else _move_summary(observation["last_non_pass"])
            ),
            "legalActions": [
                _action_summary(action, index) for index, action in enumerate(actions)
            ],
            "recentActions": [_history_summary(event) for event in observation["history"][-12:]],
            "result": self._result_payload(landlord),
        }

    def _run_ai_turns(self) -> None:
        safety = 0
        while not self.environment.terminal and self.environment.current_player != self.human_seat:
            seat = self.environment.current_player
            observation = self.environment.observe(seat)
            actions = self.environment.legal_actions()
            landlord = observation["landlord"]
            context = PolicyDecisionContext(
                deal_index=0,
                deal_seed=self.seed,
                match_id=self.game_id,
                seat=seat,
                role=None if landlord is None else role_for_game_seat(seat, landlord),
                decision_index=self.decision_index,
            )
            selected = self.ai_policy.select_action(observation, actions, context)
            if not 0 <= selected < len(actions):
                raise WebGameError("AI 返回了无效动作")
            self.terminal_result = self.environment.step(actions[selected])
            self.decision_index += 1
            safety += 1
            if safety > 500:
                raise WebGameError("AI 自动回合超过安全上限")

    def _result_payload(self, landlord: int | None) -> dict[str, object] | None:
        if not self.environment.terminal or self.terminal_result is None:
            return None
        raw = list(self.terminal_result["raw_payoff"])
        all_pass = landlord is None
        human_won = None if all_pass else raw[self.human_seat] > 0
        winner_seat = None if all_pass else self.terminal_result["event"]["actor"]
        if all_pass:
            message = "三家都没有叫地主，本局流局"
        elif human_won:
            message = "漂亮！这一局你赢了"
        else:
            message = "这一局 AI 获胜，再来一把"
        return {
            "allPass": all_pass,
            "humanWon": human_won,
            "winnerSeat": winner_seat,
            "rawPayoff": raw,
            "message": message,
        }


class GameService:
    """Thread-safe in-memory game registry and lazy AI checkpoint loader."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.rules = load_rule_config(
            self.repository_root / "configs" / "rules" / "canonical_full.yaml"
        )
        self._games: dict[str, GameSession] = {}
        self._policies: dict[str, Policy] = {"heuristic": FriendlyRulePolicy()}
        self._lock = threading.RLock()

    def available_modes(self) -> list[dict[str, object]]:
        checkpoint = self._smoke_checkpoint()
        modes: list[dict[str, object]] = [
            {
                "id": "heuristic",
                "label": "快速规则 AI",
                "description": "响应快，适合先玩一局",
                "recommended": True,
            }
        ]
        if checkpoint.is_file():
            modes.append(
                {
                    "id": "bird_dou_smoke",
                    "label": "Bird-Dou 冒烟模型",
                    "description": "真实神经网络推理，但尚未完成大规模训练",
                    "recommended": False,
                }
            )
        return modes

    def create_game(
        self,
        *,
        seed: int | None = None,
        human_seat: int = 0,
        ai_mode: str = "heuristic",
    ) -> dict[str, object]:
        if not 0 <= human_seat <= 2:
            raise WebGameError("玩家座位必须是 0、1 或 2")
        active_seed = secrets.randbits(64) if seed is None else seed
        if not 0 <= active_seed < 1 << 64:
            raise WebGameError("牌局种子必须是 uint64")
        policy = self._policy(ai_mode)
        game_id = uuid.uuid4().hex
        game = GameSession(game_id, active_seed, human_seat, ai_mode, self.rules, policy)
        with self._lock:
            self._games[game_id] = game
            self._prune_games()
        state = game.start()
        state["availableAiModes"] = self.available_modes()
        return state

    def get_game(self, game_id: str) -> dict[str, object]:
        game = self._get_session(game_id)
        state = game.public_state()
        state["availableAiModes"] = self.available_modes()
        return state

    def play(self, game_id: str, action_index: int) -> dict[str, object]:
        game = self._get_session(game_id)
        state = game.play(action_index)
        state["availableAiModes"] = self.available_modes()
        return state

    def _get_session(self, game_id: str) -> GameSession:
        with self._lock:
            game = self._games.get(game_id)
        if game is None:
            raise WebGameError("找不到这局游戏，请重新开局")
        return game

    def _policy(self, mode: str) -> Policy:
        with self._lock:
            cached = self._policies.get(mode)
            if cached is not None:
                return cached
            if mode != "bird_dou_smoke":
                raise WebGameError(f"未知 AI 模式：{mode}")
            checkpoint = self._smoke_checkpoint()
            if not checkpoint.is_file():
                raise WebGameError("Bird-Dou 冒烟模型不存在")
            try:
                policy = load_full_game_checkpoint_policy(
                    "bird-dou-web-smoke",
                    checkpoint,
                    self.repository_root / "configs" / "model" / "bid_head_v2.yaml",
                    self.repository_root / "configs" / "model" / "bird_dou_v1.yaml",
                    self.repository_root / "configs" / "model" / "bird_dou_features_v1.yaml",
                    self.rules,
                    "cpu",
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise WebGameError(f"无法加载 Bird-Dou 冒烟模型：{error}") from error
            self._policies[mode] = policy
            return policy

    def _smoke_checkpoint(self) -> Path:
        return self.repository_root / "artifacts" / "train" / "full_game_smoke" / "checkpoint.pt"

    def _prune_games(self) -> None:
        while len(self._games) > 64:
            oldest = next(iter(self._games))
            del self._games[oldest]


def _last_non_pass_actor(observation: Observation) -> int | None:
    for event in reversed(observation["history"]):
        action = event["action"]
        if "play" in action and cast(PlayGameAction, action)["play"]["kind"] != "pass":
            return event["actor"]
    return None


def _action_summary(action: Action, index: int) -> dict[str, object]:
    if "play" in action:
        summary = _move_summary(cast(PlayGameAction, action)["play"])
        summary["index"] = index
        summary["phase"] = "card_play"
        return summary
    if "bid" in action:
        bid = cast(BidGameAction, action)["bid"]
        if isinstance(bid, Mapping):
            score = bid.get("score")
            label = f"叫 {score} 分"
            kind = f"score_{score}"
        else:
            translations = {"pass": "不叫", "call": "叫地主", "rob": "抢地主"}
            label = translations.get(bid, str(bid))
            kind = str(bid)
        return {
            "index": index,
            "phase": "bidding",
            "kind": kind,
            "label": label,
            "cards": [0] * 15,
            "totalCards": 0,
        }
    double = action["double"]
    return {
        "index": index,
        "phase": "doubling",
        "kind": double,
        "label": "加倍" if double == "double" else "不加倍",
        "cards": [0] * 15,
        "totalCards": 0,
    }


def _move_summary(move: Mapping[str, object]) -> dict[str, object]:
    kind = cast(str, move["kind"])
    cards = cast(list[int], move["cards"])
    card_text = _format_cards(cards)
    move_name = MOVE_LABELS.get(kind, kind)
    return {
        "kind": kind,
        "label": move_name if not card_text else f"{move_name} · {card_text}",
        "cards": list(cards),
        "totalCards": cast(int, move["total_cards"]),
    }


def _history_summary(event: Mapping[str, object]) -> dict[str, object]:
    action = cast(Action, event["action"])
    summary = _action_summary(action, cast(int, event["sequence"]))
    summary["actor"] = cast(int, event["actor"])
    summary["sequence"] = cast(int, event["sequence"])
    return summary


def _format_cards(counts: Sequence[int]) -> str:
    if len(counts) != 15 or any(count < 0 for count in counts):
        raise WebGameError("牌面计数无效")
    labels = [label for label, count in zip(RANK_LABELS, counts, strict=True) for _ in range(count)]
    return " ".join(labels)


def _finite_number(value: object) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise WebGameError("请求中包含无效数字")
    return float(value)


__all__ = ("FriendlyRulePolicy", "GameService", "GameSession", "WebGameError")
