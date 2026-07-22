"""Fair experiment contracts shared by DMC, V-trace, and Hybrid learners."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor

from birddou.features.ragged import RaggedBatch
from birddou.models.segment_ops import segment_sum
from birddou.rl.hybrid import HybridLossConfig, HybridLossOutput, TrainerMode, hybrid_loss
from birddou.rl.vtrace import (
    PolicyLagMonitor,
    VTraceConfig,
    VTraceReturns,
    vtrace_from_log_probabilities,
)

ALGORITHM_COMPARISON_SCHEMA_VERSION = 1


class LearnerPolicyOutput(Protocol):
    """Student action fields consumed by the shared learner step."""

    @property
    def policy_log_probability(self) -> Tensor: ...

    @property
    def policy_probability(self) -> Tensor: ...

    @property
    def mc_q(self) -> Tensor: ...

    @property
    def win_logit(self) -> Tensor: ...

    @property
    def expected_score(self) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class LearnerTrajectoryBatch:
    """Time-major actor data with raw and transformed rewards kept separately."""

    behavior_log_probability: Tensor
    actor_policy_version: Tensor
    raw_reward: Tensor
    training_reward: Tensor
    done: Tensor
    terminal_target: Tensor
    win_target: Tensor
    score_target: Tensor
    bootstrap_value: Tensor

    def __post_init__(self) -> None:
        shape = self.behavior_log_probability.shape
        if self.behavior_log_probability.ndim != 2:
            raise ValueError("learner trajectories must be time-major [T, B]")
        vectors = (
            self.raw_reward,
            self.training_reward,
            self.terminal_target,
            self.win_target,
            self.score_target,
        )
        if any(value.shape != shape for value in (*vectors, self.actor_policy_version, self.done)):
            raise ValueError("learner trajectory tensors must share [T, B]")
        floating = (self.behavior_log_probability, *vectors, self.bootstrap_value)
        if any(
            not value.is_floating_point() or not torch.isfinite(value).all() for value in floating
        ):
            raise ValueError("learner floating trajectory fields must be finite")
        if self.actor_policy_version.dtype != torch.int64:
            raise ValueError("actor policy versions must use int64")
        if self.done.dtype != torch.bool:
            raise ValueError("learner done flags must use bool")
        if self.bootstrap_value.shape != shape[1:]:
            raise ValueError("learner bootstrap value must match the batch axis")
        if len({value.device for value in (*floating, self.actor_policy_version, self.done)}) != 1:
            raise ValueError("learner trajectory tensors must share one device")
        if torch.any(self.behavior_log_probability > 0.0):
            raise ValueError("behavior log probabilities cannot exceed zero")
        if torch.any(self.actor_policy_version < 0):
            raise ValueError("actor policy versions must be non-negative")
        if torch.any((self.win_target < 0.0) | (self.win_target > 1.0)):
            raise ValueError("win targets must be probabilities in 0..1")


@dataclass(frozen=True, slots=True)
class LearnerStepOutput:
    """One switchable learner result plus off-policy diagnostics."""

    losses: HybridLossOutput
    vtrace: VTraceReturns
    state_value_prediction: Tensor
    entropy: Tensor
    chosen_action_flat_index: Tensor


def bird_dou_learner_step(
    output: LearnerPolicyOutput,
    ragged_batch: RaggedBatch,
    trajectory: LearnerTrajectoryBatch,
    loss_config: HybridLossConfig,
    vtrace_config: VTraceConfig,
    *,
    learner_policy_version: int,
    lag_monitor: PolicyLagMonitor | None = None,
    belief_loss: Tensor | None = None,
    kd_loss: Tensor | None = None,
    auxiliary_loss: Tensor | None = None,
) -> LearnerStepOutput:
    """Compute an actual DMC/V-trace/Hybrid update from flattened legal actions."""
    if learner_policy_version < 0:
        raise ValueError("learner policy version must be non-negative")
    time_steps, parallel_trajectories = trajectory.behavior_log_probability.shape
    state_count = time_steps * parallel_trajectories
    if ragged_batch.batch_size != state_count:
        raise ValueError("ragged state count does not match time-major trajectory")
    chosen = ragged_batch.chosen_action_flat_index
    if chosen.shape != (state_count,) or torch.any(chosen < 0):
        raise ValueError("learner requires one chosen legal action per state")
    action_count = ragged_batch.action_count
    action_fields = (
        output.policy_log_probability,
        output.policy_probability,
        output.mc_q,
        output.win_logit,
        output.expected_score,
    )
    if any(value.shape != (action_count,) for value in action_fields):
        raise ValueError("learner policy output does not match legal action count")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all() for value in action_fields
    ):
        raise ValueError("learner policy output must be finite and floating")
    devices = {value.device for value in (*action_fields, chosen, trajectory.training_reward)}
    if len(devices) != 1:
        raise ValueError("learner policy, features, and trajectory must share one device")

    state_value = segment_sum(
        output.policy_probability.detach() * output.mc_q,
        ragged_batch.action_offsets,
    ).reshape(time_steps, parallel_trajectories)
    entropy = -segment_sum(
        output.policy_probability * output.policy_log_probability,
        ragged_batch.action_offsets,
    ).reshape(time_steps, parallel_trajectories)
    target_log_probability = output.policy_log_probability[chosen].reshape(
        time_steps, parallel_trajectories
    )
    vtrace = vtrace_from_log_probabilities(
        trajectory.behavior_log_probability,
        target_log_probability,
        trajectory.training_reward,
        state_value,
        trajectory.bootstrap_value,
        trajectory.done,
        vtrace_config,
    )
    if lag_monitor is not None:
        lag_monitor.observe(
            learner_policy_version,
            trajectory.actor_policy_version,
            vtrace.importance_weights.detach(),
        )
    chosen_log_probability = target_log_probability
    losses = hybrid_loss(
        loss_config,
        chosen_log_probability=chosen_log_probability,
        entropy=entropy,
        value_prediction=state_value,
        vtrace=vtrace,
        mc_q_prediction=output.mc_q[chosen].reshape(time_steps, parallel_trajectories),
        terminal_target=trajectory.terminal_target,
        win_logit=output.win_logit[chosen].reshape(time_steps, parallel_trajectories),
        win_target=trajectory.win_target,
        score_prediction=output.expected_score[chosen].reshape(time_steps, parallel_trajectories),
        score_target=trajectory.score_target,
        belief_loss=belief_loss,
        kd_loss=kd_loss,
        auxiliary_loss=auxiliary_loss,
    )
    return LearnerStepOutput(losses, vtrace, state_value, entropy, chosen)


@dataclass(frozen=True, slots=True)
class FairTrainingBudget:
    """All resources and inputs that must be equal in an algorithm comparison."""

    environment_frames: int
    learner_updates: int
    seeds: tuple[int, ...]
    model_config: str
    rules_config: str
    actor_processes: int
    envs_per_actor: int
    unroll_length: int
    max_inference_states: int
    max_inference_actions: int
    device: str

    def __post_init__(self) -> None:
        counts = (
            self.environment_frames,
            self.learner_updates,
            self.actor_processes,
            self.envs_per_actor,
            self.unroll_length,
            self.max_inference_states,
            self.max_inference_actions,
        )
        if min(counts) <= 0:
            raise ValueError("fair comparison resource counts must be positive")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("fair comparison seeds must be non-empty and non-negative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("fair comparison seeds must be unique")
        if not self.model_config or not self.rules_config or not self.device:
            raise ValueError("fair comparison model, rules, and device cannot be empty")


@dataclass(frozen=True, slots=True)
class FairComparisonConfig:
    """Versioned three-mode comparison plan loaded from repository config."""

    schema_version: int
    modes: tuple[TrainerMode, ...]
    budget: FairTrainingBudget

    def __post_init__(self) -> None:
        if self.schema_version != ALGORITHM_COMPARISON_SCHEMA_VERSION:
            raise ValueError("unsupported fair comparison schema")
        if len(self.modes) != 3 or set(self.modes) != set(TrainerMode):
            raise ValueError("fair comparison must contain DMC, V-trace, and Hybrid once")


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    """Completed run record used to prove rather than assume a fair result."""

    mode: TrainerMode
    budget: FairTrainingBudget
    completed_environment_frames: int
    completed_learner_updates: int
    metrics: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.completed_environment_frames < 0 or self.completed_learner_updates < 0:
            raise ValueError("completed comparison work cannot be negative")
        names = [name for name, _ in self.metrics]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("comparison metric names must be non-empty and unique")
        if any(not math.isfinite(value) for _, value in self.metrics):
            raise ValueError("comparison metrics must be finite")


@dataclass(frozen=True, slots=True)
class FairComparisonReport:
    """Validated neutral table; it deliberately does not declare a winner."""

    modes: tuple[TrainerMode, ...]
    metric_names: tuple[str, ...]
    metric_rows: tuple[tuple[float, ...], ...]
    environment_frames_per_mode: int
    learner_updates_per_mode: int


def load_fair_comparison_config(path: Path) -> FairComparisonConfig:
    """Load the JSON-subset YAML fair-comparison contract."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "fair comparison config")
    modes_value = root.get("modes")
    seeds_value = root.get("seeds")
    if not isinstance(modes_value, list) or not all(
        isinstance(value, str) for value in modes_value
    ):
        raise ValueError("fair comparison modes must be a string list")
    if not isinstance(seeds_value, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in seeds_value
    ):
        raise ValueError("fair comparison seeds must be an integer list")
    budget = FairTrainingBudget(
        environment_frames=_integer(root, "environment_frames"),
        learner_updates=_integer(root, "learner_updates"),
        seeds=tuple(cast(list[int], seeds_value)),
        model_config=_string(root, "model_config"),
        rules_config=_string(root, "rules_config"),
        actor_processes=_integer(root, "actor_processes"),
        envs_per_actor=_integer(root, "envs_per_actor"),
        unroll_length=_integer(root, "unroll_length"),
        max_inference_states=_integer(root, "max_inference_states"),
        max_inference_actions=_integer(root, "max_inference_actions"),
        device=_string(root, "device"),
    )
    return FairComparisonConfig(
        schema_version=_integer(root, "schema_version"),
        modes=tuple(TrainerMode(value) for value in cast(list[str], modes_value)),
        budget=budget,
    )


def validate_fair_comparison(runs: Sequence[ComparisonRun]) -> FairComparisonReport:
    """Reject unequal budgets, missing work, modes, or incomparable metrics."""
    if len(runs) != 3 or {run.mode for run in runs} != set(TrainerMode):
        raise ValueError("comparison requires exactly DMC, V-trace, and Hybrid")
    reference = runs[0].budget
    if any(run.budget != reference for run in runs[1:]):
        raise ValueError("comparison runs do not share the same budget and inputs")
    if any(
        run.completed_environment_frames != reference.environment_frames
        or run.completed_learner_updates != reference.learner_updates
        for run in runs
    ):
        raise ValueError("comparison run has not completed its declared fair budget")
    metric_names = tuple(name for name, _ in runs[0].metrics)
    if not metric_names:
        raise ValueError("comparison requires at least one evaluation metric")
    if any(tuple(name for name, _ in run.metrics) != metric_names for run in runs[1:]):
        raise ValueError("comparison runs must report the same ordered metrics")
    ordered = tuple(sorted(runs, key=lambda run: tuple(TrainerMode).index(run.mode)))
    return FairComparisonReport(
        modes=tuple(run.mode for run in ordered),
        metric_names=metric_names,
        metric_rows=tuple(tuple(value for _, value in run.metrics) for run in ordered),
        environment_frames_per_mode=reference.environment_frames,
        learner_updates_per_mode=reference.learner_updates,
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"fair comparison {key} must be an integer")
    return value


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"fair comparison {key} must be a non-empty string")
    return value


__all__ = (
    "ALGORITHM_COMPARISON_SCHEMA_VERSION",
    "ComparisonRun",
    "FairComparisonConfig",
    "FairComparisonReport",
    "FairTrainingBudget",
    "LearnerPolicyOutput",
    "LearnerStepOutput",
    "LearnerTrajectoryBatch",
    "bird_dou_learner_step",
    "load_fair_comparison_config",
    "validate_fair_comparison",
)
