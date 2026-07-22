"""Strict loaders for named BIRD-Dou checkpoints used by evaluation CLIs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from birddou.env_types import RuleConfig
from birddou.eval.baselines import Policy, StagedPolicy

if TYPE_CHECKING:
    from torch import Tensor

    from birddou.features import FeatureConfig


def parse_named_checkpoints(values: Sequence[str], label: str) -> dict[str, Path]:
    """Parse repeatable ``NAME=PATH`` options with no silent replacement."""
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path_value = raw.partition("=")
        name = name.strip()
        path_value = path_value.strip()
        if separator != "=" or not name or not path_value:
            raise ValueError(f"{label} must use NAME=PATH, got {raw!r}")
        if name in result:
            raise ValueError(f"duplicate {label} name: {name}")
        result[name] = Path(path_value).resolve()
    return result


def load_cardplay_checkpoint_policy(
    policy_id: str,
    checkpoint_path: Path,
    model_config_path: Path,
    feature_config_path: Path,
    rules: RuleConfig,
    device: str,
) -> Policy:
    """Load a raw, DMC, or full-game card-play checkpoint with schema checks."""
    import torch

    from birddou.models.bird_dou import BirdDouModel, load_bird_dou_config
    from birddou.rl.bird_dou_dmc import BirdDouPolicy

    checkpoint = _mapping(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        "BIRD-Dou checkpoint",
    )
    model_config = load_bird_dou_config(model_config_path)
    feature_config = _feature_config_for_checkpoint(feature_config_path, checkpoint)
    _validate_model_features(model_config, feature_config)
    if "cardplay_model" in checkpoint:
        state = _tensor_mapping(checkpoint.get("cardplay_model"), "cardplay_model")
        _validate_optional_metadata(checkpoint, model_config.fingerprint(), feature_config)
    elif "model" in checkpoint:
        state = _tensor_mapping(checkpoint.get("model"), "model")
        _validate_optional_metadata(checkpoint, model_config.fingerprint(), feature_config)
    else:
        state = _tensor_mapping(checkpoint, "raw cardplay state")
    model = BirdDouModel(model_config)
    model.load_state_dict(state, strict=True)
    return BirdDouPolicy(policy_id, model, rules, feature_config, device=device)


def load_full_game_checkpoint_policy(
    policy_id: str,
    checkpoint_path: Path,
    bid_model_config_path: Path,
    cardplay_model_config_path: Path,
    feature_config_path: Path,
    rules: RuleConfig,
    device: str,
) -> Policy:
    """Load one joint Bid Head/Cardplay checkpoint as a phase-dispatching policy."""
    import torch

    from birddou.models.bid_head import BidHead, load_bid_head_config
    from birddou.models.bird_dou import BirdDouModel, load_bird_dou_config
    from birddou.rl.bidding import BidHeadPolicy
    from birddou.rl.bird_dou_dmc import BirdDouPolicy
    from birddou.rl.full_game import FULL_GAME_CHECKPOINT_SCHEMA_VERSION

    if rules["profile"] != "canonical_full":
        raise ValueError("full-game checkpoint policies require canonical_full rules")
    bid_config = load_bid_head_config(bid_model_config_path)
    cardplay_config = load_bird_dou_config(cardplay_model_config_path)
    checkpoint = _mapping(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        "full-game checkpoint",
    )
    feature_config = _feature_config_for_checkpoint(feature_config_path, checkpoint)
    _validate_model_features(cardplay_config, feature_config)
    required_metadata = {
        "checkpoint_schema_version": FULL_GAME_CHECKPOINT_SCHEMA_VERSION,
        "trainer_mode": "full_game_joint",
        "rules_hash": _stable_hash(rules),
        "feature_fingerprint": _stable_hash(asdict(feature_config)),
        "bid_model_fingerprint": bid_config.fingerprint(),
        "cardplay_model_fingerprint": cardplay_config.fingerprint(),
    }
    for key, expected in required_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"full-game checkpoint {key} mismatch")
    bid_model = BidHead(bid_config)
    bid_model.load_state_dict(_tensor_mapping(checkpoint.get("bid_model"), "bid_model"), True)
    cardplay_model = BirdDouModel(cardplay_config)
    cardplay_model.load_state_dict(
        _tensor_mapping(checkpoint.get("cardplay_model"), "cardplay_model"),
        strict=True,
    )
    bidding = BidHeadPolicy(f"{policy_id}:bidding", bid_model, rules, device)
    cardplay = BirdDouPolicy(
        f"{policy_id}:cardplay",
        cardplay_model,
        rules,
        feature_config,
        device=device,
    )
    return StagedPolicy(policy_id, bidding, cardplay)


def load_dmc_checkpoint_policy(
    policy_id: str,
    checkpoint_path: Path,
    device: str,
) -> Policy:
    """Load the three-role exact-DouZero DMC checkpoint for formal Arena use."""
    import torch

    from birddou.eval.paired_deals import SEAT_ROLES
    from birddou.features import DOUZERO_FEATURE_SCHEMA_VERSION
    from birddou.models.douzero_model import (
        DOUZERO_MODEL_SCHEMA_VERSION,
        create_douzero_model,
    )
    from birddou.rl.dmc import DMC_CHECKPOINT_SCHEMA_VERSION, DmcGreedyPolicy

    checkpoint = _mapping(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        "DMC checkpoint",
    )
    expected = {
        "checkpoint_schema_version": DMC_CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": DOUZERO_FEATURE_SCHEMA_VERSION,
        "model_schema_version": DOUZERO_MODEL_SCHEMA_VERSION,
        "trainer_mode": "dmc",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"DMC checkpoint {key} mismatch")
    raw_models = _mapping(checkpoint.get("models"), "DMC models")
    models = {}
    for role in SEAT_ROLES:
        model = create_douzero_model(role.value).to(device)
        model.load_state_dict(_tensor_mapping(raw_models.get(role.value), role.value), strict=True)
        models[role] = model
    return DmcGreedyPolicy(policy_id, models, device)


def _validate_model_features(model_config: object, feature_config: FeatureConfig) -> None:
    feature_schema_version = getattr(model_config, "feature_schema_version", None)
    history = getattr(model_config, "history", None)
    action = getattr(model_config, "action", None)
    if feature_schema_version != feature_config.schema_version:
        raise ValueError("BIRD-Dou model and feature schema versions differ")
    if getattr(history, "max_length", None) != feature_config.history_max_length:
        raise ValueError("BIRD-Dou model and feature history lengths differ")
    if getattr(action, "decomposition_count_cap", None) != feature_config.min_decompositions_cap:
        raise ValueError("BIRD-Dou model and feature decomposition caps differ")


def _validate_optional_metadata(
    checkpoint: Mapping[str, object],
    model_fingerprint: str,
    feature_config: FeatureConfig,
) -> None:
    expected = {
        "feature_fingerprint": _stable_hash(asdict(feature_config)),
    }
    if "model_fingerprint" in checkpoint:
        expected["model_fingerprint"] = model_fingerprint
    if "cardplay_model_fingerprint" in checkpoint:
        expected["cardplay_model_fingerprint"] = model_fingerprint
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"BIRD-Dou checkpoint {key} mismatch")


def _feature_config_for_checkpoint(
    path: Path,
    checkpoint: Mapping[str, object],
) -> FeatureConfig:
    """Resolve the one runtime ablation switch from a checkpoint fingerprint."""
    from birddou.features import load_feature_config

    configured = load_feature_config(path)
    expected = checkpoint.get("feature_fingerprint")
    if expected is None or expected == _stable_hash(asdict(configured)):
        return configured
    alternate = replace(
        configured,
        decomposition_features=not configured.decomposition_features,
    )
    if expected == _stable_hash(asdict(alternate)):
        return alternate
    raise ValueError("BIRD-Dou checkpoint feature_fingerprint mismatch")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _tensor_mapping(value: object, label: str) -> Mapping[str, Tensor]:
    import torch

    result = _mapping(value, label)
    if not result or any(not isinstance(item, torch.Tensor) for item in result.values()):
        raise ValueError(f"{label} must contain only tensors")
    return cast(Mapping[str, torch.Tensor], result)


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = (
    "load_cardplay_checkpoint_policy",
    "load_dmc_checkpoint_policy",
    "load_full_game_checkpoint_policy",
    "parse_named_checkpoints",
)
