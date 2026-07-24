"""Thread-safe registry for independent Guandan sessions."""

from __future__ import annotations

import secrets
import threading
import uuid

from ..game import WebGameError
from .session import GuandanSession


class GuandanService:
    def __init__(self) -> None:
        self._games: dict[str, GuandanSession] = {}
        self._lock = threading.RLock()

    @staticmethod
    def available_modes() -> list[dict[str, object]]:
        return [
            {
                "id": "heuristic",
                "label": "快速规则 AI",
                "description": "四位玩家、两组对家，由 Rust 规则引擎生成合法动作",
                "recommended": True,
            }
        ]

    def create_game(
        self,
        *,
        seed: int | None = None,
        human_seat: int = 0,
        level: int = 0,
        ai_mode: str = "heuristic",
    ) -> dict[str, object]:
        if not 0 <= human_seat < 4:
            raise WebGameError("玩家座位必须在 0..3")
        if not 0 <= level < 13:
            raise WebGameError("级牌必须在 2..A")
        if ai_mode != "heuristic":
            raise WebGameError(f"未知掼蛋 AI 模式：{ai_mode}")
        active_seed = secrets.randbits(64) if seed is None else seed
        if not 0 <= active_seed < 1 << 64:
            raise WebGameError("牌局种子必须是 uint64")

        game_id = uuid.uuid4().hex
        game = GuandanSession(game_id, active_seed, human_seat, level, ai_mode)
        with self._lock:
            self._games[game_id] = game
            while len(self._games) > 64:
                del self._games[next(iter(self._games))]
        return self._with_modes(game.start())

    def get_game(self, game_id: str) -> dict[str, object]:
        return self._with_modes(self._get(game_id).public_state())

    def play(self, game_id: str, card_ids: list[int] | None) -> dict[str, object]:
        return self._with_modes(self._get(game_id).play(card_ids))

    def _get(self, game_id: str) -> GuandanSession:
        with self._lock:
            game = self._games.get(game_id)
        if game is None:
            raise WebGameError("找不到这局掼蛋，请重新开局")
        return game

    def _with_modes(self, state: dict[str, object]) -> dict[str, object]:
        state["availableAiModes"] = self.available_modes()
        return state

