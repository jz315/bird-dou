"""Switchable DMC, V-trace, and Hybrid loss composition."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional

from birddou.rl.vtrace import VTraceReturns

HYBRID_CONFIG_SCHEMA_VERSION = 1


class TrainerMode(StrEnum):
    DMC = "dmc"
    VTRACE = "vtrace"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class HybridLossConfig:
    schema_version: int = HYBRID_CONFIG_SCHEMA_VERSION
    mode: TrainerMode = TrainerMode.HYBRID
    policy_coef: float = 1.0
    value_coef: float = 0.5
    mc_q_coef: float = 0.25
    win_coef: float = 0.25
    score_coef: float = 0.10
    belief_coef: float = 0.20
    kd_coef: float = 0.0
    entropy_coef: float = 0.01
    aux_coef: float = 0.05

    def __post_init__(self) -> None:
        if self.schema_version != HYBRID_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported Hybrid loss schema")
        coefficients = (
            self.policy_coef,
            self.value_coef,
            self.mc_q_coef,
            self.win_coef,
            self.score_coef,
            self.belief_coef,
            self.kd_coef,
            self.entropy_coef,
            self.aux_coef,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in coefficients):
            raise ValueError("Hybrid loss coefficients must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class HybridLossOutput:
    total: Tensor
    policy: Tensor
    value: Tensor
    mc_q: Tensor
    win: Tensor
    score: Tensor
    belief: Tensor
    kd: Tensor
    entropy: Tensor
    auxiliary: Tensor


def load_hybrid_loss_config(path: Path) -> HybridLossConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = _mapping(raw, "Hybrid config")
    return HybridLossConfig(
        schema_version=_integer(values, "schema_version"),
        mode=TrainerMode(_string(values, "mode")),
        policy_coef=_number(values, "policy_coef"),
        value_coef=_number(values, "value_coef"),
        mc_q_coef=_number(values, "mc_q_coef"),
        win_coef=_number(values, "win_coef"),
        score_coef=_number(values, "score_coef"),
        belief_coef=_number(values, "belief_coef"),
        kd_coef=_number(values, "kd_coef"),
        entropy_coef=_number(values, "entropy_coef"),
        aux_coef=_number(values, "aux_coef"),
    )


def hybrid_loss(
    config: HybridLossConfig,
    *,
    chosen_log_probability: Tensor,
    entropy: Tensor,
    value_prediction: Tensor,
    vtrace: VTraceReturns,
    mc_q_prediction: Tensor,
    terminal_target: Tensor,
    win_logit: Tensor,
    win_target: Tensor,
    score_prediction: Tensor,
    score_target: Tensor,
    belief_loss: Tensor | None = None,
    kd_loss: Tensor | None = None,
    auxiliary_loss: Tensor | None = None,
) -> HybridLossOutput:
    """Compose independently switchable objectives for one fair trainer mode."""
    shape = chosen_log_probability.shape
    vectors = (
        chosen_log_probability,
        entropy,
        value_prediction,
        vtrace.value_targets,
        vtrace.policy_advantages,
        mc_q_prediction,
        terminal_target,
        win_logit,
        win_target,
        score_prediction,
        score_target,
    )
    if any(value.shape != shape for value in vectors):
        raise ValueError("Hybrid loss vectors must share one shape")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in vectors):
        raise ValueError("Hybrid loss vectors must be finite and floating")
    zero = chosen_log_probability.new_zeros(())
    policy = -(chosen_log_probability * vtrace.policy_advantages.detach()).mean()
    value = functional.huber_loss(value_prediction, vtrace.value_targets.detach())
    mc_q = functional.huber_loss(mc_q_prediction, terminal_target)
    win = functional.binary_cross_entropy_with_logits(win_logit, win_target)
    score = functional.huber_loss(score_prediction, score_target)
    entropy_loss = -entropy.mean()
    belief = zero if belief_loss is None else _scalar_loss(belief_loss, "Belief")
    kd = zero if kd_loss is None else _scalar_loss(kd_loss, "KD")
    auxiliary = zero if auxiliary_loss is None else _scalar_loss(auxiliary_loss, "auxiliary")
    if config.mode is TrainerMode.DMC:
        total = config.mc_q_coef * mc_q + config.win_coef * win + config.score_coef * score
    elif config.mode is TrainerMode.VTRACE:
        total = (
            config.policy_coef * policy
            + config.value_coef * value
            + config.win_coef * win
            + config.score_coef * score
            + config.entropy_coef * entropy_loss
        )
    else:
        total = (
            config.policy_coef * policy
            + config.value_coef * value
            + config.mc_q_coef * mc_q
            + config.win_coef * win
            + config.score_coef * score
            + config.belief_coef * belief
            + config.kd_coef * kd
            + config.entropy_coef * entropy_loss
            + config.aux_coef * auxiliary
        )
    output = HybridLossOutput(
        total, policy, value, mc_q, win, score, belief, kd, entropy_loss, auxiliary
    )
    if any(not getattr(output, field).isfinite() for field in output.__dataclass_fields__):
        raise RuntimeError("Hybrid loss produced a non-finite component")
    return output


def score_train_reward(raw_score: Tensor) -> Tensor:
    """Log-transform platform score so extreme bomb multipliers cannot dominate."""
    if not raw_score.is_floating_point() or not torch.isfinite(raw_score).all():
        raise ValueError("raw score must be finite and floating")
    return torch.sign(raw_score) * torch.log2(1.0 + raw_score.abs())


def blend_win_score_reward(
    win_reward: Tensor,
    raw_score: Tensor,
    score_weight: float,
) -> Tensor:
    """Blend win and stable-score objectives for metric-gated curriculum stages."""
    if not math.isfinite(score_weight) or not 0.0 <= score_weight <= 1.0:
        raise ValueError("score_weight must be finite and in 0..1")
    if win_reward.shape != raw_score.shape:
        raise ValueError("win and score rewards must share shape")
    return (1.0 - score_weight) * win_reward + score_weight * score_train_reward(raw_score)


def _scalar_loss(value: Tensor, label: str) -> Tensor:
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError(f"{label} loss must be a finite scalar")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Hybrid config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Hybrid config {key} must be finite")
    return float(value)


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hybrid config {key} must be a non-empty string")
    return value


__all__ = (
    "HYBRID_CONFIG_SCHEMA_VERSION",
    "HybridLossConfig",
    "HybridLossOutput",
    "TrainerMode",
    "blend_win_score_reward",
    "hybrid_loss",
    "load_hybrid_loss_config",
    "score_train_reward",
)
