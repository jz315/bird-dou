"""Cardinality-constrained hidden-card belief components."""

from .cardinality_crf import (
    BELIEF_RANK_COUNT,
    BELIEF_SCHEMA_VERSION,
    MAX_RANK_COPIES,
    BeliefMarginals,
    cardinality_marginals,
    log_partition,
    true_assignment_score,
    validate_assignment,
)
from .losses import (
    CalibrationBin,
    CalibrationReport,
    belief_nll,
    calibration_report,
    uniform_belief_nll,
)
from .sampler import sample_hidden_allocations
from .three_container_crf import (
    THREE_CONTAINER_SCHEMA_VERSION,
    ThreeContainerMarginals,
    sample_three_container_allocations,
    three_container_log_partition,
    three_container_marginals,
    three_container_nll,
    three_container_true_score,
    validate_three_container_assignment,
)

__all__ = (
    "BELIEF_RANK_COUNT",
    "BELIEF_SCHEMA_VERSION",
    "MAX_RANK_COPIES",
    "THREE_CONTAINER_SCHEMA_VERSION",
    "BeliefMarginals",
    "CalibrationBin",
    "CalibrationReport",
    "ThreeContainerMarginals",
    "belief_nll",
    "calibration_report",
    "cardinality_marginals",
    "log_partition",
    "sample_hidden_allocations",
    "sample_three_container_allocations",
    "three_container_log_partition",
    "three_container_marginals",
    "three_container_nll",
    "three_container_true_score",
    "true_assignment_score",
    "uniform_belief_nll",
    "validate_assignment",
    "validate_three_container_assignment",
)
