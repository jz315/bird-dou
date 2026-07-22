"""Offline Belief pretraining and joint-finetuning loss utilities."""

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

from birddou.belief.data import BeliefDataset
from birddou.belief.losses import CalibrationReport, belief_nll
from birddou.features import RaggedBatch
from birddou.models.belief_bird_dou import (
    BeliefBirdDouModel,
    belief_constraints_from_batch,
)

BELIEF_PRETRAIN_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class BeliefBaseCheckpointIdentity:
    """Pinned E020 no-Belief artifact required before offline Belief training."""

    path: Path
    sha256: str
    policy_version: int
    model_fingerprint: str
    feature_fingerprint: str
    rules_hash: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.sha256, "checkpoint SHA-256"),
            (self.model_fingerprint, "model fingerprint"),
            (self.feature_fingerprint, "feature fingerprint"),
            (self.rules_hash, "rules hash"),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"Belief base {label} must contain 64 lowercase hex characters")
        if self.policy_version < 0:
            raise ValueError("Belief base policy_version must be non-negative")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True, slots=True)
class BeliefWarmStartReport:
    """Auditable proof that the enabled Belief branch preserves its E020 prior."""

    base: BeliefBaseCheckpointIdentity
    belief_scale: float
    policy_logit_max_abs_error: float
    mc_q_max_abs_error: float
    policy_logit_exact: bool
    mc_q_exact: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["base"] = self.base.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class BeliefPretrainConfig:
    """Deterministic offline-supervision configuration."""

    schema_version: int = BELIEF_PRETRAIN_SCHEMA_VERSION
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 10.0
    device: str = "cpu"
    freeze_public_encoder: bool = True
    seed: int = 5005

    def __post_init__(self) -> None:
        if self.schema_version != BELIEF_PRETRAIN_SCHEMA_VERSION:
            raise ValueError("unsupported Belief pretrain schema")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Belief pretrain epochs and batch_size must be positive")
        if (
            not math.isfinite(self.learning_rate)
            or not math.isfinite(self.weight_decay)
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
        ):
            raise ValueError("Belief pretrain optimizer settings are invalid")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("Belief pretrain max_grad_norm must be positive")
        if not 0 <= self.seed < 1 << 64:
            raise ValueError("Belief pretrain seed must fit uint64")


@dataclass(frozen=True, slots=True)
class BeliefPretrainResult:
    losses: tuple[float, ...]
    update_count: int
    checkpoint_path: Path | None


class BeliefOfflineTrainer:
    """Freeze the public policy encoder and optimize exact hidden-hand NLL."""

    def __init__(
        self,
        model: BeliefBirdDouModel,
        config: BeliefPretrainConfig,
        *,
        warm_start: BeliefWarmStartReport,
    ) -> None:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"requested unavailable CUDA device: {config.device}")
        if (
            warm_start.base.model_fingerprint != model.config.base.fingerprint()
            or warm_start.belief_scale != 0.0
            or not warm_start.policy_logit_exact
            or not warm_start.mc_q_exact
        ):
            raise ValueError("Belief pretraining requires a verified zero-gate E020 warm-start")
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.model = model.to(config.device)
        self.config = config
        self.model.set_public_encoder_frozen(config.freeze_public_encoder)
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.generator = torch.Generator().manual_seed(config.seed)
        self.warm_start = warm_start
        self.losses: list[float] = []
        self.update_count = 0

    def train(
        self,
        dataset: BeliefDataset,
        checkpoint_path: Path | None = None,
    ) -> BeliefPretrainResult:
        """Run shuffled mini-batch NLL updates and optionally save a weights checkpoint."""
        for _ in range(self.config.epochs):
            permutation = torch.randperm(dataset.state_count, generator=self.generator)
            for start in range(0, dataset.state_count, self.config.batch_size):
                selected = dataset.select(permutation[start : start + self.config.batch_size])
                batch = selected.batch.to(self.config.device)
                label = selected.true_assignment_a.to(self.config.device)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                encoding = self.model.encode_belief(batch)
                unknown, capacity_a, _ = belief_constraints_from_batch(batch)
                loss = belief_nll(
                    encoding.scores.float(),
                    unknown,
                    capacity_a,
                    label,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Belief pretraining produced a non-finite loss")
                torch.autograd.backward((loss,))
                gradient_norm = nn.utils.clip_grad_norm_(
                    (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                    self.config.max_grad_norm,
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Belief pretraining produced non-finite gradients")
                self.optimizer.step()
                self.losses.append(float(loss.detach().cpu().item()))
                self.update_count += 1
        resolved_checkpoint: Path | None = None
        if checkpoint_path is not None:
            resolved_checkpoint = checkpoint_path.resolve()
            resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": BELIEF_PRETRAIN_SCHEMA_VERSION,
                    "model_fingerprint": self.model.config.fingerprint(),
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": asdict(self.config),
                    "losses": self.losses,
                    "update_count": self.update_count,
                    "generator_state": self.generator.get_state(),
                    "base_warm_start": self.warm_start.to_dict(),
                },
                resolved_checkpoint,
            )
        return BeliefPretrainResult(tuple(self.losses), self.update_count, resolved_checkpoint)

    def unfreeze_for_behavior_anchored_finetuning(self) -> None:
        """Enable gradients before the explicitly behavior-cloned anchoring stage."""
        self.model.set_public_encoder_frozen(False)
        existing = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        newly_trainable = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in existing
        ]
        if newly_trainable:
            self.optimizer.add_param_group({"params": newly_trainable})

    def behavior_anchored_belief_finetune(
        self,
        dataset: BeliefDataset,
        *,
        epochs: int = 1,
        belief_coefficient: float = 0.2,
    ) -> tuple[float, ...]:
        """Imitate recorded actions plus Belief NLL; this is not an RL objective."""
        if epochs <= 0:
            raise ValueError("joint-finetune epochs must be positive")
        self.unfreeze_for_behavior_anchored_finetuning()
        losses: list[float] = []
        for _ in range(epochs):
            permutation = torch.randperm(dataset.state_count, generator=self.generator)
            for start in range(0, dataset.state_count, self.config.batch_size):
                selected = dataset.select(permutation[start : start + self.config.batch_size])
                batch = selected.batch.to(self.config.device)
                if torch.any(batch.chosen_action_flat_index < 0):
                    raise ValueError("joint fine-tuning requires chosen behavior actions")
                labels = selected.true_assignment_a.to(self.config.device)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(batch)
                unknown, capacity_a, _ = belief_constraints_from_batch(batch)
                hidden_loss = belief_nll(output.scores.float(), unknown, capacity_a, labels)
                policy_loss = -output.policy.policy_log_probability[
                    batch.chosen_action_flat_index
                ].mean()
                total = behavior_anchored_belief_loss(
                    policy_loss, hidden_loss, belief_coefficient
                )
                torch.autograd.backward((total,))
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("joint Belief fine-tuning produced non-finite gradients")
                self.optimizer.step()
                losses.append(float(total.detach().cpu().item()))
                self.update_count += 1
        return tuple(losses)


def behavior_anchored_belief_loss(
    policy_loss: Tensor,
    belief_loss: Tensor,
    belief_coefficient: float,
) -> Tensor:
    """Combine behavior-action NLL with exact Belief NLL for representation anchoring."""
    if not math.isfinite(belief_coefficient) or belief_coefficient < 0.0:
        raise ValueError("belief_coefficient must be finite and non-negative")
    if policy_loss.ndim != 0 or belief_loss.ndim != 0:
        raise ValueError("joint policy and Belief losses must be scalars")
    if not torch.isfinite(policy_loss) or not torch.isfinite(belief_loss):
        raise ValueError("joint policy and Belief losses must be finite")
    return policy_loss + belief_coefficient * belief_loss


def warm_start_belief_from_base_checkpoint(
    model: BeliefBirdDouModel,
    verification_batch: RaggedBatch,
    identity: BeliefBaseCheckpointIdentity,
    *,
    device: str = "cpu",
) -> BeliefWarmStartReport:
    """Strictly load E020 into ``model.base`` and prove the zero-gate invariant."""
    path = identity.path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Belief base checkpoint does not exist: {path}")
    if _sha256_file(path) != identity.sha256:
        raise RuntimeError("Belief base checkpoint SHA-256 mismatch")
    if model.config.base.fingerprint() != identity.model_fingerprint:
        raise RuntimeError("Belief base model config fingerprint mismatch")
    checkpoint = _mapping(
        torch.load(path, map_location=device, weights_only=True),
        "Belief base checkpoint",
    )
    expected = {
        "trainer_mode": "bird_dou_dmc",
        "model_fingerprint": identity.model_fingerprint,
        "feature_fingerprint": identity.feature_fingerprint,
        "rules_hash": identity.rules_hash,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Belief base checkpoint {key} mismatch")
    state = _mapping(checkpoint.get("state"), "Belief base training state")
    if _integer(state, "policy_version") != identity.policy_version:
        raise RuntimeError("Belief base checkpoint policy_version mismatch")
    weights = _mapping(checkpoint.get("model"), "Belief base model state")
    model.base.load_state_dict(dict(weights), strict=True)
    model.to(device)
    if model.belief_scale.detach().item() != 0.0:
        raise RuntimeError("Belief residual scale must be exactly zero before pretraining")

    batch = verification_batch.to(device)
    model.eval()
    with torch.inference_mode():
        base_output = model.base(batch)
        belief_output = model(batch).policy
    policy_error = _max_abs_error(base_output.policy_logit, belief_output.policy_logit)
    q_error = _max_abs_error(base_output.mc_q, belief_output.mc_q)
    policy_exact = torch.equal(base_output.policy_logit, belief_output.policy_logit)
    q_exact = torch.equal(base_output.mc_q, belief_output.mc_q)
    if not policy_exact or not q_exact:
        raise RuntimeError(
            "zero-gated Belief warm-start changed policy_logit or mc_q "
            f"(max errors: policy={policy_error}, mc_q={q_error})"
        )
    return BeliefWarmStartReport(
        identity,
        float(model.belief_scale.detach().cpu().item()),
        policy_error,
        q_error,
        policy_exact,
        q_exact,
    )


def _max_abs_error(left: Tensor, right: Tensor) -> float:
    if (
        left.shape != right.shape
        or not torch.isfinite(left).all()
        or not torch.isfinite(right).all()
    ):
        raise RuntimeError("Belief warm-start parity tensors are invalid or non-finite")
    return float((left.float() - right.float()).abs().max().cpu().item())


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{key} must be an integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_calibration_json(report: CalibrationReport, path: Path) -> None:
    """Write a dataclass-compatible calibration record for evaluation artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = (
    "BELIEF_PRETRAIN_SCHEMA_VERSION",
    "BeliefBaseCheckpointIdentity",
    "BeliefOfflineTrainer",
    "BeliefPretrainConfig",
    "BeliefPretrainResult",
    "BeliefWarmStartReport",
    "behavior_anchored_belief_loss",
    "save_calibration_json",
    "warm_start_belief_from_base_checkpoint",
)
