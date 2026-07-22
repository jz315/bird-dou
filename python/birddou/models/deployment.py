"""Compact observation-only policy and reproducible deployment bundle export."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from birddou.env_types import Action, Observation, RuleConfig
from birddou.eval.baselines import PolicyDecisionContext
from birddou.features.ragged import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.bid_head import BidHead, BidHeadConfig, encode_bid_batch
from birddou.models.proposal import ProposalConfig, ProposalNetwork
from birddou.models.segment_ops import segment_max, segment_mean

DEPLOYMENT_SCHEMA_VERSION = 1
COMPACT_POLICY_ARCHITECTURE = "bird_dou_compact_policy_v1"


@dataclass(frozen=True, slots=True)
class CompactPolicyConfig:
    """Versioned compact actor dimensions around the cheap Proposal encoder."""

    schema_version: int
    architecture: str
    proposal: ProposalConfig
    value_hidden_dim: int
    dropout: float

    def __post_init__(self) -> None:
        if self.schema_version != DEPLOYMENT_SCHEMA_VERSION:
            raise ValueError("unsupported compact policy schema")
        if self.architecture != COMPACT_POLICY_ARCHITECTURE:
            raise ValueError("unsupported compact policy architecture")
        if self.value_hidden_dim <= 0:
            raise ValueError("compact value hidden dimension must be positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("compact policy dropout must be in [0, 1)")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CompactPolicyOutput:
    """Flat legal-action logits and per-state outcome summaries."""

    policy_logits: Tensor
    state_value: Tensor
    win_logit: Tensor
    win_probability: Tensor
    expected_score: Tensor


@dataclass(frozen=True, slots=True)
class DeploymentManifest:
    """Auditable observation-only deployment artifact identity."""

    schema_version: int
    architecture: str
    compact_fingerprint: str
    bid_fingerprint: str | None
    rules_sha256: str
    weights_sha256: str
    input_contract: str

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def load_compact_policy_config(path: Path) -> CompactPolicyConfig:
    """Load compact and nested Proposal configuration from JSON-subset YAML."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "compact policy config")
    proposal = _mapping(root.get("proposal"), "compact proposal")
    proposal_config = ProposalConfig(
        schema_version=_integer(proposal, "schema_version"),
        architecture=_string(proposal, "architecture"),
        hidden_dim=_integer(proposal, "hidden_dim"),
        hidden_layers=_integer(proposal, "hidden_layers"),
        dropout=_number(proposal, "dropout"),
        min_k=_integer(proposal, "min_k"),
        max_k=_integer(proposal, "max_k"),
        uncertainty_scale=_number(proposal, "uncertainty_scale"),
        full_action_fraction=_number(proposal, "full_action_fraction"),
        exploration_seed=_integer(proposal, "exploration_seed"),
    )
    return CompactPolicyConfig(
        schema_version=_integer(root, "schema_version"),
        architecture=_string(root, "architecture"),
        proposal=proposal_config,
        value_hidden_dim=_integer(root, "value_hidden_dim"),
        dropout=_number(root, "dropout"),
    )


class CompactPolicyModel(nn.Module):
    """Small distilled actor reusing the cheap Proposal action scorer."""

    def __init__(self, config: CompactPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.proposal = ProposalNetwork(config.proposal)
        self.value_head = nn.Sequential(
            nn.Linear(2, config.value_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_hidden_dim, 3),
        )

    def forward(self, batch: RaggedBatch) -> CompactPolicyOutput:
        """Score a validated RaggedBatch while keeping the export model small."""
        policy_logits = self.proposal(batch).score
        state_summary = torch.stack(
            (
                segment_mean(policy_logits, batch.action_offsets),
                segment_max(policy_logits, batch.action_offsets),
            ),
            dim=-1,
        )
        state_value, win_logit, expected_score = self.value_head(state_summary).unbind(dim=-1)
        return CompactPolicyOutput(
            policy_logits,
            state_value,
            win_logit,
            torch.sigmoid(win_logit),
            expected_score,
        )


class DeploymentPolicy:
    """Complete-game policy whose call boundary is only Observation + legal actions."""

    def __init__(
        self,
        policy_id: str,
        cardplay: CompactPolicyModel,
        rules: RuleConfig,
        *,
        bid_head: BidHead | None = None,
        feature_config: FeatureConfig | None = None,
        double: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        if not policy_id:
            raise ValueError("deployment policy_id must be non-empty")
        self._policy_id = policy_id
        self._cardplay = cardplay.to(device).eval()
        self._bid_head = None if bid_head is None else bid_head.to(device).eval()
        self._rules = rules
        self._features = feature_config if feature_config is not None else FeatureConfig()
        self._double = double
        self._device = device

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        if observation["observer"] != context.seat or not legal_actions:
            raise ValueError("deployment policy received an invalid public decision")
        if observation["phase"] == "doubling":
            desired = "double" if self._double else "decline"
            selected = next(
                (
                    index
                    for index, action in enumerate(legal_actions)
                    if action.get("double") == desired
                ),
                -1,
            )
            if selected < 0:
                raise ValueError("deployment doubling action is absent from legal actions")
            return selected
        if observation["phase"] == "bidding":
            if self._bid_head is None:
                raise ValueError("complete deployment requires a Bid Head")
            bid_batch = encode_bid_batch(
                (observation,),
                (legal_actions,),
                self._rules,
                history_max_length=self._bid_head.config.history_max_length,
            ).to(self._device)
            with torch.inference_mode():
                return int(torch.argmax(self._bid_head(bid_batch).mc_q).item())
        if observation["phase"] != "card_play":
            raise ValueError(f"deployment cannot act in phase {observation['phase']}")
        cardplay_batch = encode_ragged_batch(
            (observation,),
            (legal_actions,),
            self._rules,
            config=self._features,
        ).to(self._device)
        with torch.inference_mode():
            return int(torch.argmax(self._cardplay(cardplay_batch).policy_logits).item())


def export_deployment_bundle(
    destination: Path,
    cardplay: CompactPolicyModel,
    rules: RuleConfig,
    *,
    bid_head: BidHead | None = None,
) -> DeploymentManifest:
    """Atomically export weights plus a hash-bound observation-only manifest."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload: dict[str, object] = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "compact_config": asdict(cardplay.config),
        "compact_state_dict": cardplay.state_dict(),
        "bid_config": None if bid_head is None else asdict(bid_head.config),
        "bid_state_dict": None if bid_head is None else bid_head.state_dict(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    weights_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
    rules_payload = json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()
    manifest = DeploymentManifest(
        schema_version=DEPLOYMENT_SCHEMA_VERSION,
        architecture=COMPACT_POLICY_ARCHITECTURE,
        compact_fingerprint=cardplay.config.fingerprint(),
        bid_fingerprint=None if bid_head is None else bid_head.config.fingerprint(),
        rules_sha256=hashlib.sha256(rules_payload).hexdigest(),
        weights_sha256=weights_sha256,
        input_contract="Observation+legal_actions:v1",
    )
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_temporary, manifest_path)
    return manifest


def load_deployment_bundle(
    source: Path,
    rules: RuleConfig,
) -> tuple[CompactPolicyModel, BidHead | None, DeploymentManifest]:
    """Verify hashes and restore a strict deployment bundle with weights-only loading."""
    source = source.resolve()
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    raw_manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    manifest = DeploymentManifest(
        schema_version=_integer(raw_manifest, "schema_version"),
        architecture=_string(raw_manifest, "architecture"),
        compact_fingerprint=_string(raw_manifest, "compact_fingerprint"),
        bid_fingerprint=_optional_string(raw_manifest, "bid_fingerprint"),
        rules_sha256=_string(raw_manifest, "rules_sha256"),
        weights_sha256=_string(raw_manifest, "weights_sha256"),
        input_contract=_string(raw_manifest, "input_contract"),
    )
    if manifest.schema_version != DEPLOYMENT_SCHEMA_VERSION:
        raise ValueError("unsupported deployment manifest schema")
    if manifest.architecture != COMPACT_POLICY_ARCHITECTURE:
        raise ValueError("deployment manifest architecture mismatch")
    if manifest.input_contract != "Observation+legal_actions:v1":
        raise ValueError("deployment input contract is not observation-only v1")
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest.weights_sha256:
        raise ValueError("deployment weights checksum mismatch")
    rules_payload = json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(rules_payload).hexdigest() != manifest.rules_sha256:
        raise ValueError("deployment rule configuration checksum mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    root = _mapping(payload, "deployment weights")
    compact_config = _compact_config_from_mapping(
        _mapping(root.get("compact_config"), "compact config")
    )
    if compact_config.fingerprint() != manifest.compact_fingerprint:
        raise ValueError("deployment compact configuration fingerprint mismatch")
    cardplay = CompactPolicyModel(compact_config)
    compact_state = root.get("compact_state_dict")
    if not isinstance(compact_state, Mapping):
        raise ValueError("deployment compact state_dict is missing")
    cardplay.load_state_dict(compact_state, strict=True)
    raw_bid_config = root.get("bid_config")
    raw_bid_state = root.get("bid_state_dict")
    if raw_bid_config is None and raw_bid_state is None and manifest.bid_fingerprint is None:
        bid_head = None
    elif raw_bid_config is not None and isinstance(raw_bid_state, Mapping):
        bid_values = _mapping(raw_bid_config, "bid config")
        bid_config = BidHeadConfig(
            schema_version=_integer(bid_values, "schema_version"),
            architecture=_string(bid_values, "architecture"),
            d_model=_integer(bid_values, "d_model"),
            rank_layers=_integer(bid_values, "rank_layers"),
            history_layers=_integer(bid_values, "history_layers"),
            attention_heads=_integer(bid_values, "attention_heads"),
            hidden_multiplier=_integer(bid_values, "hidden_multiplier"),
            history_max_length=_integer(bid_values, "history_max_length"),
            dropout=_number(bid_values, "dropout"),
        )
        if bid_config.fingerprint() != manifest.bid_fingerprint:
            raise ValueError("deployment Bid Head fingerprint mismatch")
        bid_head = BidHead(bid_config)
        bid_head.load_state_dict(raw_bid_state, strict=True)
    else:
        raise ValueError("deployment Bid Head config/state/manifest disagree")
    return cardplay, bid_head, manifest


def _compact_config_from_mapping(root: Mapping[str, object]) -> CompactPolicyConfig:
    proposal = _mapping(root.get("proposal"), "compact proposal")
    return CompactPolicyConfig(
        schema_version=_integer(root, "schema_version"),
        architecture=_string(root, "architecture"),
        proposal=ProposalConfig(
            schema_version=_integer(proposal, "schema_version"),
            architecture=_string(proposal, "architecture"),
            hidden_dim=_integer(proposal, "hidden_dim"),
            hidden_layers=_integer(proposal, "hidden_layers"),
            dropout=_number(proposal, "dropout"),
            min_k=_integer(proposal, "min_k"),
            max_k=_integer(proposal, "max_k"),
            uncertainty_scale=_number(proposal, "uncertainty_scale"),
            full_action_fraction=_number(proposal, "full_action_fraction"),
            exploration_seed=_integer(proposal, "exploration_seed"),
        ),
        value_hidden_dim=_integer(root, "value_hidden_dim"),
        dropout=_number(root, "dropout"),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"compact policy config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"compact policy config {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"compact policy config {key} must be finite")
    return numeric


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"compact policy config {key} must be a non-empty string")
    return value


def _optional_string(values: Mapping[str, object], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"compact policy config {key} must be null or a non-empty string")
    return value


__all__ = (
    "COMPACT_POLICY_ARCHITECTURE",
    "DEPLOYMENT_SCHEMA_VERSION",
    "CompactPolicyConfig",
    "CompactPolicyModel",
    "CompactPolicyOutput",
    "DeploymentManifest",
    "DeploymentPolicy",
    "export_deployment_bundle",
    "load_deployment_bundle",
    "load_compact_policy_config",
)
