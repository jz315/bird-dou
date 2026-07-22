"""Cheap legal-action proposal ranking with non-negotiable safety protection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from birddou.features.ragged import ACTION_META_COLUMNS, RaggedBatch
from birddou.models.rank_mixer import RmsNorm

PROPOSAL_SCHEMA_VERSION = 1
PROPOSAL_ARCHITECTURE = "bird_dou_proposal_v1"
_U64_MASK = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class ProposalConfig:
    """Versioned cheap-network dimensions and dynamic Top-K policy."""

    schema_version: int
    architecture: str
    hidden_dim: int
    hidden_layers: int
    dropout: float
    min_k: int
    max_k: int
    uncertainty_scale: float
    full_action_fraction: float
    exploration_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != PROPOSAL_SCHEMA_VERSION:
            raise ValueError("unsupported Proposal schema")
        if self.architecture != PROPOSAL_ARCHITECTURE:
            raise ValueError("unsupported Proposal architecture")
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("Proposal dimensions and layers must be positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("Proposal dropout must be in [0, 1)")
        if self.min_k <= 0 or self.max_k < self.min_k:
            raise ValueError("Proposal Top-K range is invalid")
        if not math.isfinite(self.uncertainty_scale) or self.uncertainty_scale < 0.0:
            raise ValueError("Proposal uncertainty scale must be finite and non-negative")
        if not 0.0 <= self.full_action_fraction <= 1.0:
            raise ValueError("Proposal full-action fraction must be in [0, 1]")
        if not 0 <= self.exploration_seed <= _U64_MASK:
            raise ValueError("Proposal exploration seed must fit uint64")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ProposalOutput:
    """One cheap score per legal action."""

    score: Tensor


@dataclass(frozen=True, slots=True)
class ProposalSelection:
    """Protected, original-order action subset and per-state dynamic K."""

    selected_mask: Tensor
    selected_flat_index: Tensor
    selected_offsets: Tensor
    dynamic_k: Tensor
    protected_mask: Tensor

    def __post_init__(self) -> None:
        if self.selected_mask.dtype != torch.bool or self.protected_mask.dtype != torch.bool:
            raise ValueError("Proposal selection masks must be bool")
        if self.selected_mask.shape != self.protected_mask.shape:
            raise ValueError("Proposal selected/protected masks differ in shape")
        if (
            self.selected_flat_index.dtype != torch.int64
            or self.selected_offsets.dtype != torch.int64
        ):
            raise ValueError("Proposal indices and offsets must be int64")
        if self.dynamic_k.dtype != torch.int64:
            raise ValueError("Proposal dynamic K must be int64")
        if not torch.all(self.selected_mask | ~self.protected_mask):
            raise ValueError("Proposal dropped a protected action")


@dataclass(frozen=True, slots=True)
class ProposalGateThresholds:
    """Predeclared requirements before hard pruning may be enabled."""

    min_teacher_recall: float
    min_throughput_ratio: float
    min_paired_delta_ci_lower: float = 0.0
    min_full_action_fraction: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_teacher_recall <= 1.0:
            raise ValueError("Proposal recall threshold must be in [0, 1]")
        if self.min_throughput_ratio <= 1.0:
            raise ValueError("Proposal throughput threshold must exceed one")
        if self.min_paired_delta_ci_lower > 0.0:
            raise ValueError("Proposal non-regression lower-bound threshold cannot be positive")
        if not 0.0 < self.min_full_action_fraction <= 1.0:
            raise ValueError("Proposal full-action validation threshold must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ProposalValidationMetrics:
    """Independent-set recall, safety, throughput, and paired-strength measurements."""

    teacher_recall: float
    finish_recall: float
    bomb_recall: float
    throughput_ratio: float
    paired_strength_delta_ci_lower: float
    observed_full_action_fraction: float

    def __post_init__(self) -> None:
        values = (
            self.teacher_recall,
            self.finish_recall,
            self.bomb_recall,
            self.throughput_ratio,
            self.paired_strength_delta_ci_lower,
            self.observed_full_action_fraction,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Proposal validation metrics must be finite")
        if any(
            not 0.0 <= value <= 1.0
            for value in (
                self.teacher_recall,
                self.finish_recall,
                self.bomb_recall,
                self.observed_full_action_fraction,
            )
        ):
            raise ValueError("Proposal recalls and fractions must be in [0, 1]")
        if self.throughput_ratio <= 0.0:
            raise ValueError("Proposal throughput ratio must be positive")


@dataclass(frozen=True, slots=True)
class ProposalGateReport:
    """Hard-pruning gate result with explicit failure reasons."""

    accepted: bool
    reasons: tuple[str, ...]


def load_proposal_config(path: Path) -> ProposalConfig:
    """Load a JSON-subset YAML Proposal configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "Proposal config")
    return ProposalConfig(
        schema_version=_integer(root, "schema_version"),
        architecture=_string(root, "architecture"),
        hidden_dim=_integer(root, "hidden_dim"),
        hidden_layers=_integer(root, "hidden_layers"),
        dropout=_number(root, "dropout"),
        min_k=_integer(root, "min_k"),
        max_k=_integer(root, "max_k"),
        uncertainty_scale=_number(root, "uncertainty_scale"),
        full_action_fraction=_number(root, "full_action_fraction"),
        exploration_seed=_integer(root, "exploration_seed"),
    )


class ProposalNetwork(nn.Module):
    """Rank candidates from cheap public summaries and existing action metadata."""

    def __init__(self, config: ProposalConfig) -> None:
        super().__init__()
        self.config = config
        state_width = 15 * 2 + 8
        action_width = 15 * 2 + len(ACTION_META_COLUMNS)
        self.state_encoder = _mlp(state_width, config.hidden_dim, config)
        self.action_encoder = _mlp(action_width, config.hidden_dim, config)
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, batch: RaggedBatch) -> ProposalOutput:
        """Score every legal action without invoking the expensive policy trunk."""
        own = batch.rank_categorical[..., 1].to(torch.float32) / 4.0
        unknown = batch.rank_categorical[..., 2].to(torch.float32) / 4.0
        scalar = torch.stack(
            (
                batch.scalars[:, 3] / 20.0,
                batch.scalars[:, 4] / 20.0,
                batch.scalars[:, 5] / 20.0,
                batch.scalars[:, 6] / 2.0,
                batch.scalars[:, 7] / 8.0,
                batch.scalars[:, 8] / 8.0,
                batch.scalars[:, 11] / 128.0,
                batch.scalars[:, 13] / 20.0,
            ),
            dim=-1,
        )
        state = self.state_encoder(torch.cat((own, unknown, scalar), dim=-1))
        action = self.action_encoder(
            torch.cat(
                (
                    batch.action_rank_counts.to(torch.float32) / 4.0,
                    batch.post_hand_counts.to(torch.float32) / 4.0,
                    _normalized_action_meta(batch.action_meta),
                ),
                dim=-1,
            )
        )
        score = self.score_head(
            torch.cat((state[batch.action_state_index], action), dim=-1)
        ).squeeze(-1)
        if score.shape != (batch.action_count,) or not torch.isfinite(score).all():
            raise RuntimeError("Proposal network produced invalid scores")
        return ProposalOutput(score)


def proposal_protected_mask(
    batch: RaggedBatch,
    *,
    blocks_immediate_loss: Tensor | None = None,
    teacher_high_value: Tensor | None = None,
    exploration_flat_index: Tensor | None = None,
) -> Tensor:
    """Protect pass, bombs, rocket, finish, blocker, exploration, and Teacher actions."""
    columns = {name: ACTION_META_COLUMNS.index(name) for name in ACTION_META_COLUMNS}
    protected = (
        batch.action_meta[:, columns["is_pass"]].bool()
        | batch.action_meta[:, columns["is_bomb"]].bool()
        | batch.action_meta[:, columns["is_rocket"]].bool()
        | batch.action_meta[:, columns["empties_hand"]].bool()
    )
    for mask, label in (
        (blocks_immediate_loss, "blocks_immediate_loss"),
        (teacher_high_value, "teacher_high_value"),
    ):
        if mask is not None:
            if (
                mask.dtype != torch.bool
                or mask.shape != protected.shape
                or mask.device != protected.device
            ):
                raise ValueError(f"Proposal {label} mask must be bool [M] on the batch device")
            protected = protected | mask
    if exploration_flat_index is not None:
        if (
            exploration_flat_index.dtype != torch.int64
            or exploration_flat_index.shape != (batch.batch_size,)
            or exploration_flat_index.device != protected.device
        ):
            raise ValueError("Proposal exploration indices must be int64 [B] on the batch device")
        for state, flat_index in enumerate(exploration_flat_index.tolist()):
            start = int(batch.action_offsets[state].item())
            end = int(batch.action_offsets[state + 1].item())
            if not start <= flat_index < end:
                raise ValueError("Proposal exploration index lies outside its state segment")
        protected = protected.clone()
        protected[exploration_flat_index] = True
    return protected


def select_proposals(
    batch: RaggedBatch,
    scores: Tensor,
    config: ProposalConfig,
    uncertainty: Tensor,
    *,
    protected_mask: Tensor | None = None,
) -> ProposalSelection:
    """Select dynamic Top-K plus every protected action, preserving canonical order."""
    if scores.shape != (batch.action_count,) or not scores.is_floating_point():
        raise ValueError("Proposal scores must be floating [M]")
    if uncertainty.shape != (batch.batch_size,) or not uncertainty.is_floating_point():
        raise ValueError("Proposal uncertainty must be floating [B]")
    if scores.device != batch.action_offsets.device or uncertainty.device != scores.device:
        raise ValueError("Proposal scores, uncertainty, and batch must share one device")
    if not torch.isfinite(scores).all() or not torch.isfinite(uncertainty).all():
        raise ValueError("Proposal score/uncertainty contains NaN or infinity")
    if torch.any((uncertainty < 0.0) | (uncertainty > 1.0)):
        raise ValueError("Proposal uncertainty must be in [0, 1]")
    if protected_mask is None:
        protected = proposal_protected_mask(batch)
    else:
        if protected_mask.dtype != torch.bool or protected_mask.shape != scores.shape:
            raise ValueError("Proposal protected mask must be bool [M]")
        protected = protected_mask
    requested = config.min_k + torch.round(
        uncertainty * config.uncertainty_scale * (config.max_k - config.min_k)
    ).to(torch.int64)
    requested = requested.clamp(min=config.min_k, max=config.max_k)
    selected = protected.clone()
    actual_k: list[int] = []
    for state in range(batch.batch_size):
        start = int(batch.action_offsets[state].item())
        end = int(batch.action_offsets[state + 1].item())
        state_k = min(int(requested[state].item()), end - start)
        top = torch.topk(scores[start:end], state_k, sorted=False).indices + start
        selected[top] = True
        actual_k.append(state_k)
    selected_index = torch.nonzero(selected, as_tuple=False).squeeze(-1)
    selected_counts = torch.zeros(
        batch.batch_size, dtype=torch.int64, device=selected.device
    ).index_add(
        0,
        batch.action_state_index[selected_index],
        torch.ones_like(selected_index),
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=selected.device),
            torch.cumsum(selected_counts, dim=0),
        )
    )
    return ProposalSelection(
        selected_mask=selected,
        selected_flat_index=selected_index,
        selected_offsets=offsets,
        dynamic_k=torch.tensor(actual_k, dtype=torch.int64, device=selected.device),
        protected_mask=protected,
    )


def should_use_full_action_set(global_step: int, state_id: int, config: ProposalConfig) -> bool:
    """Deterministically reserve the configured fraction of training states unpruned."""
    if global_step < 0 or state_id < 0:
        raise ValueError("Proposal step/state IDs must be non-negative")
    key = config.exploration_seed ^ global_step ^ ((state_id << 1) & _U64_MASK)
    uniform = _splitmix64(key & _U64_MASK) / float(1 << 64)
    return uniform < config.full_action_fraction


def subset_ragged_batch(batch: RaggedBatch, selection: ProposalSelection) -> RaggedBatch:
    """Build a canonical-order RaggedBatch containing exactly the selected actions."""
    indices = selection.selected_flat_index
    return RaggedBatch(
        schema_version=batch.schema_version,
        rank_categorical=batch.rank_categorical,
        rank_numeric=batch.rank_numeric,
        history_rank_counts=batch.history_rank_counts,
        history_meta=batch.history_meta,
        history_mask=batch.history_mask,
        scalars=batch.scalars,
        action_rank_counts=batch.action_rank_counts[indices],
        post_hand_counts=batch.post_hand_counts[indices],
        action_meta=batch.action_meta[indices],
        action_state_index=batch.action_state_index[indices],
        action_offsets=selection.selected_offsets,
        chosen_action_flat_index=torch.full(
            (batch.batch_size,),
            -1,
            dtype=torch.int64,
            device=batch.chosen_action_flat_index.device,
        ),
    )


def evaluate_proposal_gate(
    metrics: ProposalValidationMetrics,
    thresholds: ProposalGateThresholds,
) -> ProposalGateReport:
    """Permit hard pruning only after recall, safety, speed, and strength gates pass."""
    reasons: list[str] = []
    if metrics.teacher_recall < thresholds.min_teacher_recall:
        reasons.append("Teacher-best Top-K recall is below threshold")
    if metrics.finish_recall != 1.0:
        reasons.append("direct-finish recall is not 100%")
    if metrics.bomb_recall != 1.0:
        reasons.append("bomb/rocket recall is not 100%")
    if metrics.throughput_ratio < thresholds.min_throughput_ratio:
        reasons.append("measured throughput ratio is below threshold")
    if metrics.paired_strength_delta_ci_lower < thresholds.min_paired_delta_ci_lower:
        reasons.append("paired strength lower bound crosses the non-regression gate")
    if metrics.observed_full_action_fraction < thresholds.min_full_action_fraction:
        reasons.append("training retained too few full-action control states")
    return ProposalGateReport(not reasons, tuple(reasons))


def _mlp(input_dim: int, hidden_dim: int, config: ProposalConfig) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
    for _ in range(config.hidden_layers - 1):
        layers.extend((nn.Dropout(config.dropout), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
    layers.append(RmsNorm(hidden_dim))
    return nn.Sequential(*layers)


def _normalized_action_meta(metadata: Tensor) -> Tensor:
    scale = metadata.new_tensor([15, 15, 12, 2, 20, 1, 1, 1, 1, 1, 4, 4, 20, 256]).to(torch.float32)
    return metadata.to(torch.float32) / scale


def _splitmix64(value: int) -> int:
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (value ^ (value >> 31)) & _U64_MASK


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Proposal config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Proposal config {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Proposal config {key} must be finite")
    return numeric


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Proposal config {key} must be a non-empty string")
    return value


__all__ = (
    "PROPOSAL_ARCHITECTURE",
    "PROPOSAL_SCHEMA_VERSION",
    "ProposalConfig",
    "ProposalGateReport",
    "ProposalGateThresholds",
    "ProposalNetwork",
    "ProposalOutput",
    "ProposalSelection",
    "ProposalValidationMetrics",
    "evaluate_proposal_gate",
    "load_proposal_config",
    "proposal_protected_mask",
    "select_proposals",
    "should_use_full_action_set",
    "subset_ragged_batch",
)
