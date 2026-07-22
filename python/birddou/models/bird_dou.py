"""Complete information-set BIRD-Dou model without the optional Belief CRF."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from birddou.features.ragged import FEATURE_SCHEMA_VERSION, RaggedBatch
from birddou.models.action_encoder import (
    ActionEncoderConfig,
    ActionEncoding,
    RaggedActionEncoder,
    SwiGluMlp,
    load_action_encoder_config,
)
from birddou.models.history_encoder import (
    HistoryEncoderConfig,
    HistoryEncoding,
    RoleGatedHistoryEncoder,
    load_history_encoder_config,
)
from birddou.models.rank_mixer import (
    RankMixer,
    RankMixerConfig,
    RankTokenEncoder,
    RmsNorm,
    load_rank_mixer_config,
)
from birddou.models.role_adapters import RoleAdapterConfig, RoleSeatAdapter
from birddou.models.segment_ops import segment_logsumexp, segment_softmax

BIRD_DOU_MODEL_SCHEMA_VERSION = 1
BIRD_DOU_ARCHITECTURE = "bird_dou_no_belief_v1"
DecisionMode = Literal["policy", "wp", "score", "mc_q", "risk"]


@dataclass(frozen=True, slots=True)
class BirdDouConfig:
    """Complete versioned model configuration with aligned submodule widths."""

    schema_version: int = BIRD_DOU_MODEL_SCHEMA_VERSION
    architecture: str = BIRD_DOU_ARCHITECTURE
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    d_model: int = 256
    rank_mixer: RankMixerConfig = RankMixerConfig()
    history: HistoryEncoderConfig = HistoryEncoderConfig()
    action: ActionEncoderConfig = ActionEncoderConfig()
    role_adapter_dim: int = 64
    score_quantiles: int = 11
    output_hidden_multiplier: int = 4
    output_hidden_layers: int = 3
    belief_enabled: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != BIRD_DOU_MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported BIRD-Dou model schema")
        if self.architecture != BIRD_DOU_ARCHITECTURE:
            raise ValueError("unsupported BIRD-Dou architecture")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("BIRD-Dou feature schema mismatch")
        if self.d_model <= 0 or self.role_adapter_dim <= 0:
            raise ValueError("BIRD-Dou model and adapter dimensions must be positive")
        if self.score_quantiles <= 0:
            raise ValueError("score_quantiles must be positive")
        if self.output_hidden_multiplier <= 0 or self.output_hidden_layers <= 0:
            raise ValueError("output hidden dimensions must be positive")
        if self.belief_enabled:
            raise ValueError("bird_dou_no_belief_v1 cannot enable Belief")
        widths = (self.rank_mixer.d_model, self.history.d_model, self.action.d_model)
        if any(width != self.d_model for width in widths):
            raise ValueError("all BIRD-Dou submodules must share d_model")

    def fingerprint(self) -> str:
        """Return a stable hash for checkpoint compatibility checks."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class BirdDouOutput:
    """Policy, conditional score, DMC, auxiliary, and diagnostic outputs."""

    policy_logit: Tensor
    policy_log_probability: Tensor
    policy_probability: Tensor
    policy_log_normalizer: Tensor
    win_logit: Tensor
    score_if_win: Tensor
    score_if_loss: Tensor
    expected_score: Tensor
    mc_q: Tensor
    turns_to_finish: Tensor
    score_win_quantiles: Tensor
    score_loss_quantiles: Tensor
    state: Tensor
    seat: Tensor
    rank_tokens: Tensor
    history: HistoryEncoding
    actions: ActionEncoding


@dataclass(frozen=True, slots=True)
class PublicStateEncoding:
    """Information-set state before optional Belief fusion and role adaptation."""

    seat: Tensor
    rank_tokens: Tensor
    history: HistoryEncoding
    pre_belief_state: Tensor


def load_bird_dou_config(path: Path) -> BirdDouConfig:
    """Load and cross-check the complete JSON-subset YAML model configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "model config")
    output = _mapping(root.get("output"), "output")
    return BirdDouConfig(
        schema_version=_integer(root, "schema_version"),
        architecture=_string(root, "architecture"),
        feature_schema_version=_integer(root, "feature_schema_version"),
        d_model=_integer(root, "d_model"),
        rank_mixer=load_rank_mixer_config(path),
        history=load_history_encoder_config(path),
        action=load_action_encoder_config(path),
        role_adapter_dim=_integer(root, "role_adapter_dim"),
        score_quantiles=_integer(root, "score_quantiles"),
        output_hidden_multiplier=_integer(output, "hidden_multiplier"),
        output_hidden_layers=_integer(output, "hidden_layers"),
        belief_enabled=_boolean(root, "belief_enabled"),
    )


class RoleActionOutputHead(nn.Module):
    """One role/seat-specific deep head over the shared action representation."""

    def __init__(self, config: BirdDouConfig) -> None:
        super().__init__()
        hidden = config.d_model * config.output_hidden_multiplier
        layers: list[nn.Module] = [
            RmsNorm(config.d_model),
            nn.Linear(config.d_model, hidden),
            nn.SiLU(),
        ]
        for _ in range(config.output_hidden_layers - 1):
            layers.extend((nn.Linear(hidden, hidden), nn.SiLU()))
        layers.append(nn.Linear(hidden, 6 + 2 * config.score_quantiles))
        self.network = nn.Sequential(*layers)

    def forward(self, actions: Tensor) -> Tensor:
        return cast(Tensor, self.network(actions))


class RoleActionOutputHeads(nn.Module):
    """Independent landlord/down/up heads evaluated only for their own rows."""

    def __init__(self, config: BirdDouConfig) -> None:
        super().__init__()
        self.input_width = config.d_model
        self.output_width = 6 + 2 * config.score_quantiles
        self.heads = nn.ModuleList(RoleActionOutputHead(config) for _ in range(3))

    def forward(self, actions: Tensor, seat: Tensor) -> Tensor:
        if (
            actions.ndim != 2
            or actions.shape[1] != self.input_width
            or not actions.is_floating_point()
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("action output inputs must be finite floating [M, d_model]")
        if seat.dtype != torch.int64 or seat.shape != (actions.shape[0],):
            raise ValueError("action output seat must be int64 [M]")
        if seat.device != actions.device or torch.any((seat < 0) | (seat > 2)):
            raise ValueError("action output seat is invalid")
        output = torch.zeros(
            (actions.shape[0], self.output_width),
            dtype=actions.dtype,
            device=actions.device,
        )
        for seat_index, head in enumerate(self.heads):
            indices = torch.nonzero(seat == seat_index, as_tuple=False).squeeze(-1)
            if indices.numel() > 0:
                selected = head(actions.index_select(0, indices))
                output = output.index_copy(0, indices, selected)
        return output


class BirdDouModel(nn.Module):
    """Rank, history, role, ragged-action, and multi-head BIRD-Dou network."""

    def __init__(self, config: BirdDouConfig) -> None:
        super().__init__()
        self.config = config
        self.rank_token_encoder = RankTokenEncoder(config.rank_mixer)
        self.rank_mixer = RankMixer(config.rank_mixer)
        self.history_encoder = RoleGatedHistoryEncoder(config.history)
        self.state_fusion = SwiGluMlp(
            config.d_model * 4,
            config.d_model * 2,
            config.d_model,
            config.history.dropout,
        )
        self.role_adapter = RoleSeatAdapter(
            RoleAdapterConfig(
                d_model=config.d_model,
                bottleneck_dim=config.role_adapter_dim,
                dropout=config.history.dropout,
            )
        )
        self.action_encoder = RaggedActionEncoder(config.action)
        self.output_heads = RoleActionOutputHeads(config)

    def forward(self, batch: RaggedBatch) -> BirdDouOutput:
        public = self.encode_public_state(batch)
        return self.forward_from_state(batch, public, public.pre_belief_state)

    def encode_public_state(self, batch: RaggedBatch) -> PublicStateEncoding:
        """Encode only legal information, stopping at the optional Belief boundary."""
        if batch.schema_version != self.config.feature_schema_version:
            raise ValueError("RaggedBatch schema differs from BIRD-Dou config")
        seat = seat_from_scalars(batch.scalars)
        rank_tokens = self.rank_mixer(
            self.rank_token_encoder(batch.rank_categorical, batch.rank_numeric)
        )
        history = self.history_encoder(
            batch.history_rank_counts,
            batch.history_meta,
            batch.history_mask,
            batch.scalars,
            seat,
        )
        pre_belief_state = self.state_fusion(
            torch.cat(
                (
                    rank_tokens.mean(dim=1),
                    rank_tokens.amax(dim=1),
                    history.fused,
                    history.scalar,
                ),
                dim=-1,
            )
        )
        return PublicStateEncoding(seat, rank_tokens, history, pre_belief_state)

    def forward_from_state(
        self,
        batch: RaggedBatch,
        public: PublicStateEncoding,
        state_before_role: Tensor,
    ) -> BirdDouOutput:
        """Run role adaptation, ragged actions, and heads from a fused state."""
        if state_before_role.shape != (batch.batch_size, self.config.d_model):
            raise ValueError("BIRD-Dou fused state must be [B, d_model]")
        if not state_before_role.is_floating_point() or not torch.isfinite(state_before_role).all():
            raise ValueError("BIRD-Dou fused state must be finite and floating")
        state = self.role_adapter(state_before_role, public.seat)
        actions = self.action_encoder(batch, state, public.rank_tokens)
        action_seat = public.seat[batch.action_state_index]
        raw = self.output_heads(actions.action, action_seat)
        policy_logit = raw[:, 0]
        policy_log_normalizer = segment_logsumexp(policy_logit, batch.action_offsets)
        policy_log_probability = policy_logit - policy_log_normalizer[batch.action_state_index]
        policy_probability = segment_softmax(policy_logit, batch.action_offsets)
        win_logit = raw[:, 1]
        score_if_win = functional.softplus(raw[:, 2])
        score_if_loss = -functional.softplus(raw[:, 3])
        win_probability = torch.sigmoid(win_logit)
        expected_score = win_probability * score_if_win + (1.0 - win_probability) * score_if_loss
        mc_q = raw[:, 4]
        turns_to_finish = functional.softplus(raw[:, 5])
        quantile_count = self.config.score_quantiles
        score_win_quantiles = torch.sort(
            functional.softplus(raw[:, 6 : 6 + quantile_count]),
            dim=-1,
        ).values
        score_loss_quantiles = torch.sort(
            -functional.softplus(raw[:, 6 + quantile_count :]),
            dim=-1,
        ).values
        return BirdDouOutput(
            policy_logit=policy_logit,
            policy_log_probability=policy_log_probability,
            policy_probability=policy_probability,
            policy_log_normalizer=policy_log_normalizer,
            win_logit=win_logit,
            score_if_win=score_if_win,
            score_if_loss=score_if_loss,
            expected_score=expected_score,
            mc_q=mc_q,
            turns_to_finish=turns_to_finish,
            score_win_quantiles=score_win_quantiles,
            score_loss_quantiles=score_loss_quantiles,
            state=state,
            seat=public.seat,
            rank_tokens=public.rank_tokens,
            history=public.history,
            actions=actions,
        )


def seat_from_scalars(scalars: Tensor) -> Tensor:
    """Map landlord-relative scalar codes to landlord/downstream/upstream IDs."""
    if scalars.dtype != torch.float32 or scalars.ndim != 2 or scalars.shape[1] < 3:
        raise ValueError("seat derivation requires float32 scalar rows")
    relative_float = scalars[:, 2]
    relative = relative_float.to(torch.int64)
    if not torch.equal(relative_float, relative.to(torch.float32)):
        raise ValueError("landlord-relative scalar must contain integer codes")
    if torch.any((relative < 0) | (relative > 2)):
        raise ValueError("landlord-relative scalar must be in 0..2")
    return torch.remainder(-relative, 3)


def decision_values(
    output: BirdDouOutput,
    mode: DecisionMode,
    risk_aversion: float = 0.0,
) -> Tensor:
    """Return one documented action-selection utility per flat candidate."""
    if not math.isfinite(risk_aversion) or risk_aversion < 0.0:
        raise ValueError("risk_aversion must be finite and non-negative")
    if mode == "policy":
        return output.policy_logit
    if mode == "wp":
        return torch.sigmoid(output.win_logit)
    if mode == "score":
        return output.expected_score
    if mode == "mc_q":
        return output.mc_q
    if mode == "risk":
        downside = torch.relu(output.expected_score - output.score_loss_quantiles[:, 0])
        return output.expected_score - risk_aversion * downside
    raise ValueError(f"unknown BIRD-Dou decision mode: {mode}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model config {key} must be an integer")
    return value


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
    "BIRD_DOU_ARCHITECTURE",
    "BIRD_DOU_MODEL_SCHEMA_VERSION",
    "BirdDouConfig",
    "BirdDouModel",
    "BirdDouOutput",
    "DecisionMode",
    "PublicStateEncoding",
    "RoleActionOutputHead",
    "RoleActionOutputHeads",
    "decision_values",
    "load_bird_dou_config",
    "seat_from_scalars",
)
