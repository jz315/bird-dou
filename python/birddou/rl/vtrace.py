"""Numerically bounded IMPALA V-trace targets and policy-lag diagnostics."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

VTRACE_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VTraceConfig:
    schema_version: int = VTRACE_CONFIG_SCHEMA_VERSION
    gamma: float = 1.0
    rho_bar: float = 1.0
    c_bar: float = 1.0
    policy_gradient_rho_bar: float = 1.0
    max_log_importance_weight: float = 20.0

    def __post_init__(self) -> None:
        if self.schema_version != VTRACE_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported V-trace config schema")
        values = (
            self.gamma,
            self.rho_bar,
            self.c_bar,
            self.policy_gradient_rho_bar,
            self.max_log_importance_weight,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("V-trace settings must be finite")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("V-trace gamma must be in 0..1")
        if (
            min(
                self.rho_bar,
                self.c_bar,
                self.policy_gradient_rho_bar,
                self.max_log_importance_weight,
            )
            <= 0.0
        ):
            raise ValueError("V-trace clipping settings must be positive")


def load_vtrace_config(path: Path) -> VTraceConfig:
    """Load a versioned JSON-subset YAML V-trace configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ValueError("V-trace config must be a string-keyed mapping")
    values = cast(Mapping[str, object], raw)
    return VTraceConfig(
        schema_version=_integer(values, "schema_version"),
        gamma=_number(values, "gamma"),
        rho_bar=_number(values, "rho_bar"),
        c_bar=_number(values, "c_bar"),
        policy_gradient_rho_bar=_number(values, "policy_gradient_rho_bar"),
        max_log_importance_weight=_number(values, "max_log_importance_weight"),
    )


@dataclass(frozen=True, slots=True)
class VTraceReturns:
    value_targets: Tensor
    policy_advantages: Tensor
    importance_weights: Tensor
    clipped_rhos: Tensor
    clipped_cs: Tensor
    log_importance_weights: Tensor


def vtrace_from_log_probabilities(
    behavior_log_probability: Tensor,
    target_log_probability: Tensor,
    rewards: Tensor,
    values: Tensor,
    bootstrap_value: Tensor,
    done: Tensor,
    config: VTraceConfig | None = None,
) -> VTraceReturns:
    """Compute standard time-major V-trace targets for `[T, ...]` trajectories."""
    settings = config if config is not None else VTraceConfig()
    tensors = (target_log_probability, rewards, values, done)
    if behavior_log_probability.ndim < 1 or any(
        tensor.shape != behavior_log_probability.shape for tensor in tensors
    ):
        raise ValueError("V-trace trajectory tensors must share a non-scalar shape")
    if bootstrap_value.shape != behavior_log_probability.shape[1:]:
        raise ValueError("V-trace bootstrap value shape must match trailing batch axes")
    if done.dtype != torch.bool:
        raise ValueError("V-trace done flags must use bool")
    floating = (behavior_log_probability, target_log_probability, rewards, values, bootstrap_value)
    if any(not tensor.is_floating_point() for tensor in floating):
        raise ValueError("V-trace probabilities, rewards, and values must be floating")
    if len({tensor.device for tensor in (*floating, done)}) != 1:
        raise ValueError("V-trace tensors must share one device")
    if any(not torch.isfinite(tensor).all() for tensor in floating):
        raise ValueError("V-trace tensors contain NaN or infinity")

    log_rhos = target_log_probability - behavior_log_probability
    bounded_log_rhos = log_rhos.clamp(
        -settings.max_log_importance_weight,
        settings.max_log_importance_weight,
    )
    importance = torch.exp(bounded_log_rhos)
    clipped_rhos = importance.clamp(max=settings.rho_bar)
    clipped_cs = importance.clamp(max=settings.c_bar)
    pg_rhos = importance.clamp(max=settings.policy_gradient_rho_bar)
    discounts = (~done).to(rewards.dtype) * settings.gamma
    next_values = torch.cat((values[1:], bootstrap_value.unsqueeze(0)), dim=0)
    deltas = clipped_rhos * (rewards + discounts * next_values - values)

    accumulator = torch.zeros_like(bootstrap_value)
    corrections: list[Tensor] = []
    for time_index in range(values.shape[0] - 1, -1, -1):
        accumulator = (
            deltas[time_index] + discounts[time_index] * clipped_cs[time_index] * accumulator
        )
        corrections.append(accumulator)
    corrections.reverse()
    value_targets = values + torch.stack(corrections, dim=0)
    next_targets = torch.cat((value_targets[1:], bootstrap_value.unsqueeze(0)), dim=0)
    policy_advantages = pg_rhos * (rewards + discounts * next_targets - values)
    result = VTraceReturns(
        value_targets=value_targets,
        policy_advantages=policy_advantages,
        importance_weights=importance,
        clipped_rhos=clipped_rhos,
        clipped_cs=clipped_cs,
        log_importance_weights=log_rhos,
    )
    if any(not getattr(result, field).isfinite().all() for field in result.__dataclass_fields__):
        raise RuntimeError("V-trace produced a non-finite target")
    return result


@dataclass(frozen=True, slots=True)
class PolicyLagStats:
    sample_count: int
    mean_lag: float
    maximum_lag: int
    p95_lag: float
    stale_fraction: float


class PolicyLagMonitor:
    """Bounded rolling policy-version and importance-weight monitor."""

    def __init__(self, stale_after: int = 8, capacity: int = 100_000) -> None:
        if stale_after < 0 or capacity <= 0:
            raise ValueError("policy-lag thresholds are invalid")
        self.stale_after = stale_after
        self._lags: deque[int] = deque(maxlen=capacity)
        self._importance: deque[float] = deque(maxlen=capacity)

    def observe(
        self,
        learner_version: int,
        actor_versions: Tensor,
        importance_weights: Tensor,
    ) -> None:
        if learner_version < 0:
            raise ValueError("learner policy version must be non-negative")
        if actor_versions.dtype != torch.int64 or actor_versions.shape != importance_weights.shape:
            raise ValueError("actor versions and importance weights must have matching shape")
        if (
            not importance_weights.is_floating_point()
            or not torch.isfinite(importance_weights).all()
            or torch.any(importance_weights <= 0.0)
        ):
            raise ValueError("importance weights must be finite, floating, and positive")
        lags = learner_version - actor_versions
        if torch.any(lags < 0):
            raise ValueError("actor policy version cannot be newer than learner")
        self._lags.extend(int(value) for value in lags.reshape(-1).cpu().tolist())
        self._importance.extend(
            float(value) for value in importance_weights.reshape(-1).cpu().tolist()
        )

    def stats(self) -> PolicyLagStats:
        if not self._lags:
            return PolicyLagStats(0, 0.0, 0, 0.0, 0.0)
        values = torch.tensor(tuple(self._lags), dtype=torch.float32)
        return PolicyLagStats(
            sample_count=len(self._lags),
            mean_lag=float(values.mean().item()),
            maximum_lag=int(values.max().item()),
            p95_lag=float(torch.quantile(values, 0.95).item()),
            stale_fraction=float((values > self.stale_after).float().mean().item()),
        )

    @property
    def importance_weight_range(self) -> tuple[float, float]:
        if not self._importance:
            return 0.0, 0.0
        return min(self._importance), max(self._importance)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"V-trace config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"V-trace config {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"V-trace config {key} must be finite")
    return numeric


__all__ = (
    "PolicyLagMonitor",
    "PolicyLagStats",
    "VTRACE_CONFIG_SCHEMA_VERSION",
    "VTraceConfig",
    "VTraceReturns",
    "load_vtrace_config",
    "vtrace_from_log_probabilities",
)
