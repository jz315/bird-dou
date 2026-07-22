"""Local browser game backed by the authoritative native environment."""

from .game import GameService, GameSession, WebGameError

__all__ = ("GameService", "GameSession", "WebGameError")
