"""Public package metadata for BIRD-Dou."""

from typing import Final

from .batch_env import (
    BATCH_SCHEMA_VERSION,
    BatchObservation,
    BatchStepResult,
    PackedActions,
    PyBatchDdzEnv,
)
from .env import (
    API_SCHEMA_VERSION,
    SHUFFLE_ALGORITHM,
    PyDdzEnv,
    generate_lead_actions,
    load_rule_config,
    load_versioned_rule_config,
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
    RuleConfigV2,
    StepResult,
    VersionedRuleConfig,
)

__version__: Final = "0.1.0"

__all__ = (
    "API_SCHEMA_VERSION",
    "BATCH_SCHEMA_VERSION",
    "SHUFFLE_ALGORITHM",
    "Action",
    "BatchObservation",
    "BatchStepResult",
    "ExactSolveResult",
    "Move",
    "Observation",
    "PyDdzEnv",
    "PyBatchDdzEnv",
    "PackedActions",
    "RuleConfig",
    "RuleConfigV2",
    "StepResult",
    "VersionedRuleConfig",
    "__version__",
    "load_rule_config",
    "load_versioned_rule_config",
    "generate_lead_actions",
    "minimum_play_groups",
    "parse_rule_config",
    "parse_versioned_rule_config",
    "rule_config_hash",
    "solve_endgame",
)
