"""Simple presentation-layer opponent policy over Rust-generated legal actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_EXPENSIVE_KINDS = {"bomb", "straight_flush", "four_jokers"}


def choose_action(actions: Sequence[Mapping[str, object]], *, leading: bool) -> int:
    """Choose one action without interpreting or recreating card legality."""
    playable = [action for action in actions if action.get("kind") != "pass"]
    if not playable:
        return _index(actions[0])
    if leading:
        selected = max(
            playable,
            key=lambda action: (
                _count(action),
                action.get("kind") not in _EXPENSIVE_KINDS,
                -_index(action),
            ),
        )
    else:
        selected = min(
            playable,
            key=lambda action: (
                action.get("kind") in _EXPENSIVE_KINDS,
                _count(action),
                _index(action),
            ),
        )
    return _index(selected)


def _index(action: Mapping[str, object]) -> int:
    value = action.get("index")
    if not isinstance(value, int):
        raise ValueError("native Guandan action has no integer index")
    return value


def _count(action: Mapping[str, object]) -> int:
    value = action.get("totalCards")
    if not isinstance(value, int):
        raise ValueError("native Guandan action has no card count")
    return value

