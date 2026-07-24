"""One information-safe Guandan browser session."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from birddou._guandan_native import PyGuandanGame

from ..game import WebGameError
from .policy import choose_action


@dataclass(slots=True)
class GuandanSession:
    game_id: str
    seed: int
    human_seat: int
    level: int
    ai_mode: str
    native: PyGuandanGame = field(init=False)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.native = PyGuandanGame(self.seed, self.level, self.human_seat)

    def start(self) -> dict[str, object]:
        with self.lock:
            self._run_ai_turns()
            return self.public_state()

    def play(self, card_ids: list[int] | None) -> dict[str, object]:
        with self.lock:
            if self.native.terminal:
                raise WebGameError("本局已经结束，请开始新游戏")
            if self.native.current_player != self.human_seat:
                raise WebGameError("现在不是你的回合")
            try:
                self.native.play_cards(self.human_seat, card_ids)
                self._run_ai_turns()
            except (RuntimeError, ValueError) as error:
                raise WebGameError(f"出牌不合法：{error}") from error
            return self.public_state()

    def public_state(self) -> dict[str, object]:
        state = cast(dict[str, object], self.native.state(self.human_seat))
        state["gameId"] = self.game_id
        state["aiMode"] = self.ai_mode
        return state

    def _run_ai_turns(self) -> None:
        safety = 0
        while not self.native.terminal and self.native.current_player != self.human_seat:
            seat = self.native.current_player
            if seat is None:
                break
            state = self.native.state(seat)
            raw_actions = state.get("legalActions")
            if not isinstance(raw_actions, list) or not raw_actions:
                raise WebGameError("Rust 规则引擎没有返回合法动作")
            actions = [cast(Mapping[str, object], action) for action in raw_actions]
            selected = choose_action(actions, leading=state.get("target") is None)
            self.native.step_action(seat, selected)
            safety += 1
            if safety > 600:
                raise WebGameError("AI 自动回合超过安全上限")


def validate_card_ids(value: object) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(card, int) and not isinstance(card, bool) and 0 <= card < 108
        for card in value
    ):
        raise WebGameError("cardIds 必须是 0..107 的实体牌编号数组或 null")
    return cast(list[int], value)


__all__ = ("GuandanSession", "validate_card_ids")
