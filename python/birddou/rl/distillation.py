"""Information-set-consistent Teacher distillation over Belief samples."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional

from birddou.belief.sampler import sample_hidden_allocations
from birddou.features.ragged import RaggedBatch
from birddou.models.belief_bird_dou import (
    BeliefBirdDouModel,
    BeliefBirdDouOutput,
    belief_constraints_from_batch,
)
from birddou.models.privileged_teacher import PrivilegedTeacher
from birddou.models.segment_ops import segment_softmax, segment_sum

IS_KD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class InformationSetDistillationConfig:
    """Sampling, temperature, and loss weights for IS-KD."""

    schema_version: int = IS_KD_SCHEMA_VERSION
    belief_samples_k: int = 4
    teacher_temperature: float = 0.5
    value_coefficient: float = 0.5
    stop_gradient_through_belief_for_kd: bool = True
    include_true_state: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != IS_KD_SCHEMA_VERSION:
            raise ValueError("unsupported IS-KD schema")
        if self.belief_samples_k <= 0:
            raise ValueError("IS-KD belief_samples_k must be positive")
        if not math.isfinite(self.teacher_temperature) or self.teacher_temperature <= 0.0:
            raise ValueError("IS-KD teacher_temperature must be positive and finite")
        if not math.isfinite(self.value_coefficient) or self.value_coefficient < 0.0:
            raise ValueError("IS-KD value_coefficient must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class InformationSetDistillationOutput:
    """Student output, sampled states, averaged Teacher targets, and losses."""

    loss: Tensor
    policy_kl: Tensor
    value_loss: Tensor
    q_bar: Tensor
    teacher_probability: Tensor
    hidden_samples_a: Tensor
    student: BeliefBirdDouOutput


def load_is_kd_config(path: Path) -> InformationSetDistillationConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = _mapping(raw, "IS-KD config")
    return InformationSetDistillationConfig(
        schema_version=_integer(values, "schema_version"),
        belief_samples_k=_integer(values, "belief_samples_k"),
        teacher_temperature=_number(values, "teacher_temperature"),
        value_coefficient=_number(values, "value_coefficient"),
        stop_gradient_through_belief_for_kd=_boolean(values, "stop_gradient_through_belief_for_kd"),
        include_true_state=_boolean(values, "include_true_state"),
    )


def information_set_distillation_loss(
    student: BeliefBirdDouModel,
    teacher: PrivilegedTeacher,
    batch: RaggedBatch,
    true_assignment_a: Tensor | None,
    config: InformationSetDistillationConfig,
    *,
    generator: torch.Generator | None = None,
) -> InformationSetDistillationOutput:
    """Average Teacher Q over legal hidden states before forming the soft target."""
    student_output = student(batch)
    unknown, capacity_a, _ = belief_constraints_from_batch(batch)
    sample_scores = student_output.scores
    if config.stop_gradient_through_belief_for_kd:
        sample_scores = sample_scores.detach()
    samples = sample_hidden_allocations(
        sample_scores.float(),
        unknown,
        capacity_a,
        config.belief_samples_k,
        generator=generator,
    )
    if config.include_true_state:
        if true_assignment_a is None:
            raise ValueError("true-state KD ablation requires a true hidden assignment")
        samples = torch.cat((samples, true_assignment_a[:, None]), dim=1)
    q_values: list[Tensor] = []
    teacher.eval()
    with torch.no_grad():
        for sample_index in range(samples.shape[1]):
            q_values.append(teacher(batch, samples[:, sample_index]).policy.mc_q)
    q_bar = torch.stack(q_values, dim=0).mean(dim=0)
    teacher_probability = segment_softmax(
        q_bar / config.teacher_temperature,
        batch.action_offsets,
    )
    log_teacher = torch.log(
        teacher_probability.clamp_min(torch.finfo(teacher_probability.dtype).tiny)
    )
    terms = teacher_probability * (log_teacher - student_output.policy.policy_log_probability)
    policy_kl = segment_sum(terms, batch.action_offsets).mean()
    value_loss = per_state_action_huber_loss(
        student_output.policy.mc_q,
        q_bar,
        batch.action_offsets,
    )
    loss = policy_kl + config.value_coefficient * value_loss
    if not torch.isfinite(loss):
        raise RuntimeError("IS-KD produced a non-finite loss")
    return InformationSetDistillationOutput(
        loss,
        policy_kl,
        value_loss,
        q_bar,
        teacher_probability,
        samples,
        student_output,
    )


def direct_state_distillation_loss(
    student: BeliefBirdDouModel,
    teacher: PrivilegedTeacher,
    batch: RaggedBatch,
    true_assignment_a: Tensor,
    config: InformationSetDistillationConfig,
) -> InformationSetDistillationOutput:
    """True-state-only ablation whose target is not information-set consistent."""
    student_output = student(batch)
    teacher.eval()
    with torch.no_grad():
        q_bar = teacher(batch, true_assignment_a).policy.mc_q
    teacher_probability = segment_softmax(
        q_bar / config.teacher_temperature,
        batch.action_offsets,
    )
    log_teacher = torch.log(
        teacher_probability.clamp_min(torch.finfo(teacher_probability.dtype).tiny)
    )
    terms = teacher_probability * (log_teacher - student_output.policy.policy_log_probability)
    policy_kl = segment_sum(terms, batch.action_offsets).mean()
    value_loss = per_state_action_huber_loss(
        student_output.policy.mc_q,
        q_bar,
        batch.action_offsets,
    )
    loss = policy_kl + config.value_coefficient * value_loss
    return InformationSetDistillationOutput(
        loss,
        policy_kl,
        value_loss,
        q_bar,
        teacher_probability,
        true_assignment_a[:, None],
        student_output,
    )


def privileged_critic_loss(
    teacher: PrivilegedTeacher,
    batch: RaggedBatch,
    true_assignment_a: Tensor,
    terminal_target: Tensor,
) -> Tensor:
    """Regress chosen privileged MC-Q values to true team terminal returns."""
    chosen = batch.chosen_action_flat_index
    if torch.any(chosen < 0):
        raise ValueError("privileged critic loss requires chosen actions")
    if terminal_target.shape != (batch.batch_size,) or not torch.isfinite(terminal_target).all():
        raise ValueError("privileged critic targets must be finite [B]")
    prediction = teacher(batch, true_assignment_a).policy.mc_q[chosen]
    return functional.huber_loss(prediction, terminal_target)


def per_state_action_huber_loss(
    prediction: Tensor,
    target: Tensor,
    action_offsets: Tensor,
) -> Tensor:
    """Average legal-action value errors within states, then across states."""
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("KD value prediction and target must be matching action vectors")
    if action_offsets.dtype != torch.int64 or action_offsets.ndim != 1:
        raise ValueError("KD action offsets must be a one-dimensional int64 tensor")
    if action_offsets.numel() < 2 or int(action_offsets[0]) != 0:
        raise ValueError("KD action offsets must begin at zero and contain a state")
    if int(action_offsets[-1]) != prediction.numel():
        raise ValueError("KD action offsets must span every legal action")
    lengths = action_offsets[1:] - action_offsets[:-1]
    if torch.any(lengths <= 0):
        raise ValueError("KD states must each contain at least one legal action")
    per_action = functional.huber_loss(prediction, target, reduction="none")
    per_state = segment_sum(per_action, action_offsets) / lengths.to(per_action)
    return per_state.mean()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"IS-KD config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"IS-KD config {key} must be a finite number")
    return float(value)


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"IS-KD config {key} must be boolean")
    return value


__all__ = (
    "IS_KD_SCHEMA_VERSION",
    "InformationSetDistillationConfig",
    "InformationSetDistillationOutput",
    "direct_state_distillation_loss",
    "information_set_distillation_loss",
    "load_is_kd_config",
    "per_state_action_huber_loss",
    "privileged_critic_loss",
)
