"""Offline Belief pretraining and joint-finetuning loss utilities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from birddou.belief.data import BeliefDataset
from birddou.belief.losses import CalibrationReport, belief_nll
from birddou.models.belief_bird_dou import (
    BeliefBirdDouModel,
    belief_constraints_from_batch,
)

BELIEF_PRETRAIN_SCHEMA_VERSION = 1


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

    def __init__(self, model: BeliefBirdDouModel, config: BeliefPretrainConfig) -> None:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"requested unavailable CUDA device: {config.device}")
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
                },
                resolved_checkpoint,
            )
        return BeliefPretrainResult(tuple(self.losses), self.update_count, resolved_checkpoint)

    def unfreeze_for_joint_finetuning(self) -> None:
        """Enable gradients through the public encoder after offline NLL pretraining."""
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

    def joint_finetune(
        self,
        dataset: BeliefDataset,
        *,
        epochs: int = 1,
        belief_coefficient: float = 0.2,
    ) -> tuple[float, ...]:
        """Unfreeze the policy and jointly imitate behavior with exact Belief NLL."""
        if epochs <= 0:
            raise ValueError("joint-finetune epochs must be positive")
        self.unfreeze_for_joint_finetuning()
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
                total = joint_belief_policy_loss(policy_loss, hidden_loss, belief_coefficient)
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


def joint_belief_policy_loss(
    policy_loss: Tensor,
    belief_loss: Tensor,
    belief_coefficient: float,
) -> Tensor:
    """Combine an on-policy objective with exact Belief NLL for joint fine-tuning."""
    if not math.isfinite(belief_coefficient) or belief_coefficient < 0.0:
        raise ValueError("belief_coefficient must be finite and non-negative")
    if policy_loss.ndim != 0 or belief_loss.ndim != 0:
        raise ValueError("joint policy and Belief losses must be scalars")
    if not torch.isfinite(policy_loss) or not torch.isfinite(belief_loss):
        raise ValueError("joint policy and Belief losses must be finite")
    return policy_loss + belief_coefficient * belief_loss


def save_calibration_json(report: CalibrationReport, path: Path) -> None:
    """Write a dataclass-compatible calibration record for evaluation artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = (
    "BELIEF_PRETRAIN_SCHEMA_VERSION",
    "BeliefOfflineTrainer",
    "BeliefPretrainConfig",
    "BeliefPretrainResult",
    "joint_belief_policy_loss",
    "save_calibration_json",
)
