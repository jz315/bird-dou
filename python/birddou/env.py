"""Public single-environment API implemented by the PyO3 extension."""

from os import PathLike
from pathlib import Path

from ._native import (
    API_SCHEMA_VERSION,
    SHUFFLE_ALGORITHM,
    PyDdzEnv,
    generate_lead_actions,
    minimum_play_groups,
    parse_rule_config,
    solve_endgame,
)
from .env_types import Action, ExactSolveResult, Move, Observation, RuleConfig, StepResult


def load_rule_config(path: str | PathLike[str]) -> RuleConfig:
    """Load and validate a YAML rule profile through the authoritative Rust parser."""
    return parse_rule_config(Path(path).read_text(encoding="utf-8"))


__all__ = (
    "API_SCHEMA_VERSION",
    "SHUFFLE_ALGORITHM",
    "Action",
    "Move",
    "ExactSolveResult",
    "Observation",
    "PyDdzEnv",
    "RuleConfig",
    "StepResult",
    "load_rule_config",
    "generate_lead_actions",
    "minimum_play_groups",
    "parse_rule_config",
    "solve_endgame",
)
