"""Public JSONL bridge for the official Python-3.7/Linux PerfectDou release."""

from __future__ import annotations

import json
import math
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from birddou.env_types import Action, Observation, PlayGameAction
from birddou.eval.baselines import PolicyDecisionContext
from birddou.eval.paired_deals import role_for_game_seat

PERFECTDOU_BASELINE_ID = "perfectdou"
PERFECTDOU_PROTOCOL_VERSION = 1
_ENV_RANKS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30)


class PerfectDouBaselineUnavailable(RuntimeError):
    """No compatible external PerfectDou runtime was configured."""


class PerfectDouBackend(Protocol):
    """Execution boundary kept outside the Python-3.11 BIRD-Dou process."""

    def select_action(self, role: str, public_infoset: Mapping[str, object]) -> list[int]: ...


class PerfectDouProcessBackend:
    """Persistent, timeout-bounded JSONL subprocess backend for official weights."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command or any(not item for item in command):
            raise ValueError("PerfectDou backend command must be non-empty")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("PerfectDou backend timeout must be finite and positive")
        self.command = tuple(command)
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def select_action(self, role: str, public_infoset: Mapping[str, object]) -> list[int]:
        request = json.dumps(
            {
                "protocol_version": PERFECTDOU_PROTOCOL_VERSION,
                "role": role,
                "public_infoset": public_infoset,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            process = self._ensure_process()
            if process.stdin is None or process.stdout is None:
                raise PerfectDouBaselineUnavailable("PerfectDou process has no JSONL pipes")
            stdout = process.stdout
            try:
                process.stdin.write(request + "\n")
                process.stdin.flush()
            except OSError as error:
                self._terminate()
                raise PerfectDouBaselineUnavailable("PerfectDou request pipe failed") from error
            lines: list[str] = []

            def read_reply() -> None:
                lines.append(stdout.readline())

            reader = threading.Thread(target=read_reply, name="perfectdou-jsonl-reply", daemon=True)
            reader.start()
            reader.join(self.timeout_seconds)
            if reader.is_alive():
                self._terminate()
                reader.join(1.0)
                raise PerfectDouBaselineUnavailable("PerfectDou response timed out")
            if not lines or not lines[0]:
                return_code = process.poll()
                self._terminate()
                raise PerfectDouBaselineUnavailable(
                    f"PerfectDou worker exited without a response (code={return_code})"
                )
            try:
                response = json.loads(lines[0])
            except json.JSONDecodeError as error:
                raise PerfectDouBaselineUnavailable("PerfectDou returned invalid JSON") from error
            values = _mapping(response, "PerfectDou response")
            remote_error = values.get("error")
            if remote_error is not None:
                raise PerfectDouBaselineUnavailable(f"PerfectDou worker error: {remote_error}")
            action = values.get("action")
            if not isinstance(action, list) or any(
                not isinstance(card, int) or isinstance(card, bool) for card in action
            ):
                raise PerfectDouBaselineUnavailable("PerfectDou response action is invalid")
            return cast(list[int], action)

    def close(self) -> None:
        """Terminate the optional external runtime without an unbounded wait."""
        with self._lock:
            self._terminate()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as error:
            raise PerfectDouBaselineUnavailable(
                f"cannot launch PerfectDou backend {self.command!r}"
            ) from error
        return self._process

    def _terminate(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


class PerfectDouPolicy:
    """Convert a BIRD-Dou public information set for the official deployment model."""

    def __init__(self, policy_id: str, backend: PerfectDouBackend) -> None:
        if not policy_id:
            raise ValueError("PerfectDou policy_id must be non-empty")
        self._policy_id = policy_id
        self._backend = backend

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        """Call PerfectDou without materializing either hidden-hand allocation."""
        if observation["phase"] != "card_play" or observation["landlord"] is None:
            raise ValueError("official PerfectDou requires card play with a resolved landlord")
        if observation["observer"] != context.seat:
            raise ValueError("PerfectDou observation seat differs from decision context")
        landlord = observation["landlord"]
        resolved_role = role_for_game_seat(context.seat, landlord)
        if context.role is not resolved_role:
            raise ValueError("PerfectDou context role differs from resolved landlord")
        role = resolved_role.value
        seat_roles = tuple(role_for_game_seat(seat, landlord).value for seat in range(3))
        legal_cards = tuple(_action_cards(action) for action in legal_actions)
        if len({tuple(cards) for cards in legal_cards}) != len(legal_cards):
            raise ValueError("PerfectDou action conversion is not one-to-one")
        last_move_by_seat: list[list[int]] = [[], [], []]
        action_sequence: list[list[int]] = []
        last_non_pass_actor = 0
        for event in observation["history"]:
            if "play" not in event["action"]:
                continue
            cards = _action_cards(event["action"])
            action_sequence.append(cards)
            last_move_by_seat[event["actor"]] = cards
            if cards:
                last_non_pass_actor = event["actor"]
        public_infoset: Mapping[str, object] = {
            "player_position": role,
            "player_hand_cards": _counts_cards(observation["own_hand"]),
            "other_hand_cards": _counts_cards(observation["unknown_pool"]),
            "num_cards_left_dict": {
                seat_roles[seat]: observation["cards_left"][seat] for seat in range(3)
            },
            "three_landlord_cards": _counts_cards(observation["public_bottom_cards"]),
            "card_play_action_seq": action_sequence,
            "legal_actions": list(legal_cards),
            "last_move": (
                []
                if observation["last_non_pass"] is None
                else _counts_cards(observation["last_non_pass"]["cards"])
            ),
            "last_move_dict": {seat_roles[seat]: last_move_by_seat[seat] for seat in range(3)},
            "played_cards": {
                seat_roles[seat]: _counts_cards(observation["public_played"][seat])
                for seat in range(3)
            },
            "bomb_num": observation["bomb_count"],
            "last_pid": seat_roles[last_non_pass_actor],
        }
        selected = self._backend.select_action(role, public_infoset)
        try:
            return legal_cards.index(selected)
        except ValueError as error:
            raise RuntimeError(
                f"PerfectDou returned action outside canonical legal set: {selected!r}"
            ) from error


def _action_cards(action: Action) -> list[int]:
    if "play" not in action:
        raise ValueError("PerfectDou received a non-cardplay action")
    move = cast(PlayGameAction, action)["play"]
    return [] if move["kind"] == "pass" else _counts_cards(move["cards"])


def _counts_cards(counts: Sequence[int]) -> list[int]:
    if len(counts) != 15:
        raise ValueError("PerfectDou rank counts must have width 15")
    cards: list[int] = []
    for rank, count in zip(_ENV_RANKS, counts, strict=True):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("PerfectDou rank counts must be non-negative integers")
        cards.extend([rank] * count)
    return cards


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PerfectDouBaselineUnavailable(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


__all__ = (
    "PERFECTDOU_BASELINE_ID",
    "PERFECTDOU_PROTOCOL_VERSION",
    "PerfectDouBackend",
    "PerfectDouBaselineUnavailable",
    "PerfectDouPolicy",
    "PerfectDouProcessBackend",
)
