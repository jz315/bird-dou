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
    parse_versioned_rule_config,
    rule_config_hash,
    solve_endgame,
)
from .env_types import (
    Action,
    ExactSolveResult,
    Move,
    Observation,
    RuleConfig,
    StepResult,
    VersionedRuleConfig,
)


def load_rule_config(path: str | PathLike[str]) -> RuleConfig:
    """Load a legacy v1 YAML profile executable by :class:`PyDdzEnv`."""
    return parse_rule_config(Path(path).read_text(encoding="utf-8"))


def load_versioned_rule_config(path: str | PathLike[str]) -> VersionedRuleConfig:
    """Load and validate either rule schema without selecting an engine implementation."""
    return parse_versioned_rule_config(Path(path).read_text(encoding="utf-8"))


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
    "load_versioned_rule_config",
    "generate_lead_actions",
    "minimum_play_groups",
    "parse_rule_config",
    "parse_versioned_rule_config",
    "rule_config_hash",
    "solve_endgame",
)
