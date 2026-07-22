"""BIRD-Dou with an exact two-container hidden-hand Belief CRF."""

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

from birddou.belief.cardinality_crf import BeliefMarginals, cardinality_marginals
from birddou.features.ragged import FEATURE_SCHEMA_VERSION, RaggedBatch
from birddou.models.action_encoder import SwiGluMlp, load_action_encoder_config
from birddou.models.bird_dou import (
    BIRD_DOU_ARCHITECTURE,
    BirdDouConfig,
    BirdDouModel,
    BirdDouOutput,
    PublicStateEncoding,
)
from birddou.models.history_encoder import load_history_encoder_config
from birddou.models.rank_mixer import RmsNorm, load_rank_mixer_config

BELIEF_BIRD_DOU_ARCHITECTURE = "bird_dou_belief_v1"
BELIEF_BIRD_DOU_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BeliefBirdDouConfig:
    """Versioned Belief extension around a no-Belief-compatible shared trunk."""

    schema_version: int
    architecture: str
    feature_schema_version: int
    base: BirdDouConfig
    count_embedding_dim: int
    hidden_multiplier: int
    dropout: float
    enabled: bool

    def __post_init__(self) -> None:
        if self.schema_version != BELIEF_BIRD_DOU_SCHEMA_VERSION:
            raise ValueError("unsupported Belief BIRD-Dou schema")
        if self.architecture != BELIEF_BIRD_DOU_ARCHITECTURE:
            raise ValueError("unsupported Belief BIRD-Dou architecture")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("Belief BIRD-Dou feature schema mismatch")
        if self.base.feature_schema_version != self.feature_schema_version:
            raise ValueError("base and Belief feature schemas differ")
        if self.count_embedding_dim <= 0 or self.hidden_multiplier <= 0:
            raise ValueError("Belief network dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Belief dropout must be in [0, 1)")
        if not self.enabled:
            raise ValueError("bird_dou_belief_v1 requires Belief enabled")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class BeliefBirdDouOutput:
    """Policy output plus exact hidden-card distribution diagnostics."""

    policy: BirdDouOutput
    scores: Tensor
    marginals: BeliefMarginals
    belief_pool: Tensor
    fused_state: Tensor


@dataclass(frozen=True, slots=True)
class BeliefStateEncoding:
    """Belief-only result used by offline pretraining without action-head work."""

    public: PublicStateEncoding
    scores: Tensor
    marginals: BeliefMarginals
    belief_pool: Tensor
    fused_state: Tensor


def load_belief_bird_dou_config(path: Path) -> BeliefBirdDouConfig:
    """Load the full Belief architecture while retaining the E020 base dimensions."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "model config")
    output = _mapping(root.get("output"), "output")
    belief = _mapping(root.get("belief"), "belief")
    schema_version = _integer(root, "schema_version")
    feature_schema_version = _integer(root, "feature_schema_version")
    base = BirdDouConfig(
        schema_version=schema_version,
        architecture=BIRD_DOU_ARCHITECTURE,
        feature_schema_version=feature_schema_version,
        d_model=_integer(root, "d_model"),
        rank_mixer=load_rank_mixer_config(path),
        history=load_history_encoder_config(path),
        action=load_action_encoder_config(path),
        role_adapter_dim=_integer(root, "role_adapter_dim"),
        score_quantiles=_integer(root, "score_quantiles"),
        output_hidden_multiplier=_integer(output, "hidden_multiplier"),
        output_hidden_layers=_integer(output, "hidden_layers"),
        belief_enabled=False,
    )
    return BeliefBirdDouConfig(
        schema_version=schema_version,
        architecture=_string(root, "architecture"),
        feature_schema_version=feature_schema_version,
        base=base,
        count_embedding_dim=_integer(belief, "count_embedding_dim"),
        hidden_multiplier=_integer(belief, "hidden_multiplier"),
        dropout=_number(belief, "dropout"),
        enabled=_boolean(root, "belief_enabled"),
    )


class BeliefScoreNetwork(nn.Module):
    """Score assigning 0..4 cards of each unknown rank to the next player."""

    def __init__(self, config: BeliefBirdDouConfig) -> None:
        super().__init__()
        width = config.base.d_model
        hidden = width * config.hidden_multiplier
        self.width = width
        self.count_embedding = nn.Embedding(5, config.count_embedding_dim)
        self.seat_embedding = nn.Embedding(3, config.count_embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(width * 2 + config.count_embedding_dim * 2 + 2, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 5),
        )

    def forward(
        self,
        public_state: Tensor,
        rank_tokens: Tensor,
        unknown_counts: Tensor,
        capacity_a: Tensor,
        capacity_b: Tensor,
        seat: Tensor,
    ) -> Tensor:
        batch_size = public_state.shape[0]
        if public_state.shape != (batch_size, self.width):
            raise ValueError("Belief public state shape mismatch")
        if rank_tokens.shape != (batch_size, 15, self.width):
            raise ValueError("Belief rank token shape mismatch")
        if unknown_counts.dtype != torch.int64 or unknown_counts.shape != (batch_size, 15):
            raise ValueError("Belief unknown counts must be int64 [B, 15]")
        total_scale = torch.clamp((capacity_a + capacity_b).to(torch.float32), min=1.0)
        capacity_features = torch.stack(
            (
                capacity_a.to(torch.float32) / total_scale,
                capacity_b.to(torch.float32) / total_scale,
            ),
            dim=-1,
        )
        state = public_state[:, None].expand(-1, 15, -1)
        capacities = capacity_features[:, None].expand(-1, 15, -1)
        seats = self.seat_embedding(seat)[:, None].expand(-1, 15, -1)
        inputs = torch.cat(
            (
                state,
                rank_tokens,
                self.count_embedding(unknown_counts),
                seats,
                capacities,
            ),
            dim=-1,
        )
        return cast(Tensor, self.network(inputs))


class BeliefFeaturePool(nn.Module):
    """Pool exact marginal moments and key-card probabilities into one state."""

    def __init__(self, config: BeliefBirdDouConfig) -> None:
        super().__init__()
        width = config.base.d_model
        self.rank_projection = nn.Sequential(nn.Linear(6, width), nn.SiLU(), RmsNorm(width))
        self.key_projection = nn.Sequential(nn.Linear(8, width), nn.SiLU(), RmsNorm(width))
        self.fusion = SwiGluMlp(width * 3, width * 2, width, config.dropout)

    def forward(self, marginals: BeliefMarginals) -> Tensor:
        rank_features = torch.stack(
            (
                marginals.expected_a / 4.0,
                marginals.variance_a / 4.0,
                marginals.entropy_a / math.log(5.0),
                marginals.expected_b / 4.0,
                marginals.variance_b / 4.0,
                marginals.entropy_b / math.log(5.0),
            ),
            dim=-1,
        )
        rank_state = self.rank_projection(rank_features)
        keys = self.key_projection(
            torch.cat((marginals.key_probability_a, marginals.key_probability_b), dim=-1)
        )
        return cast(
            Tensor,
            self.fusion(torch.cat((rank_state.mean(dim=1), rank_state.amax(dim=1), keys), dim=-1)),
        )


class BeliefBirdDouModel(nn.Module):
    """Student policy whose only hidden-card input is a constrained distribution."""

    def __init__(self, config: BeliefBirdDouConfig) -> None:
        super().__init__()
        self.config = config
        self.base = BirdDouModel(config.base)
        self.belief_scores = BeliefScoreNetwork(config)
        self.belief_pool = BeliefFeaturePool(config)
        width = config.base.d_model
        self.state_fusion = SwiGluMlp(width * 2, width * 2, width, config.dropout)
        self.belief_scale = nn.Parameter(torch.zeros(()))

    def forward(self, batch: RaggedBatch) -> BeliefBirdDouOutput:
        belief = self.encode_belief(batch)
        policy = self.base.forward_from_state(batch, belief.public, belief.fused_state)
        return BeliefBirdDouOutput(
            policy,
            belief.scores,
            belief.marginals,
            belief.belief_pool,
            belief.fused_state,
        )

    def encode_belief(self, batch: RaggedBatch) -> BeliefStateEncoding:
        """Run public encoding, constrained CRF, and state fusion only."""
        public = self.base.encode_public_state(batch)
        unknown, capacity_a, capacity_b = belief_constraints_from_batch(batch)
        scores = self.belief_scores(
            public.pre_belief_state,
            public.rank_tokens,
            unknown,
            capacity_a,
            capacity_b,
            public.seat,
        )
        # The constrained log-sum-exp path remains FP32 under mixed precision.
        marginals = cardinality_marginals(scores.float(), unknown, capacity_a)
        belief_pool = self.belief_pool(marginals)
        belief_update = self.state_fusion(torch.cat((public.pre_belief_state, belief_pool), dim=-1))
        fused = public.pre_belief_state + torch.tanh(self.belief_scale) * belief_update
        return BeliefStateEncoding(public, scores, marginals, belief_pool, fused)

    def set_public_encoder_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the information-set encoder for offline Belief pretraining."""
        modules = (
            self.base.rank_token_encoder,
            self.base.rank_mixer,
            self.base.history_encoder,
            self.base.state_fusion,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)


def belief_constraints_from_batch(batch: RaggedBatch) -> tuple[Tensor, Tensor, Tensor]:
    """Extract next/previous hidden capacities from public schema fields only."""
    unknown = batch.rank_categorical[..., 2]
    capacity_values = batch.scalars[:, 4:6]
    capacity = capacity_values.to(torch.int64)
    if not torch.equal(capacity_values, capacity.to(torch.float32)):
        raise ValueError("hidden-player capacities must be integer scalar values")
    capacity_a, capacity_b = capacity.unbind(dim=-1)
    if not torch.equal(unknown.sum(dim=1), capacity_a + capacity_b):
        raise ValueError("unknown pool does not match the two hidden capacities")
    return unknown, capacity_a, capacity_b


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"model config {key} must be a finite number")
    return float(value)


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"model config {key} must be a non-empty string")
    return value


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"model config {key} must be a boolean")
    return value


__all__ = (
    "BELIEF_BIRD_DOU_ARCHITECTURE",
    "BELIEF_BIRD_DOU_SCHEMA_VERSION",
    "BeliefBirdDouConfig",
    "BeliefBirdDouModel",
    "BeliefBirdDouOutput",
    "BeliefFeaturePool",
    "BeliefScoreNetwork",
    "BeliefStateEncoding",
    "belief_constraints_from_batch",
    "load_belief_bird_dou_config",
)
