"""Resumable complete-game Bid Head/Cardplay curriculum smoke tests."""

import gc
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from birddou import load_rule_config
from birddou.cli.policy_artifacts import load_full_game_checkpoint_policy
from birddou.features import load_feature_config
from birddou.models.bird_dou import BirdDouModel, load_bird_dou_config
from birddou.rl import (
    FullGameConfig,
    FullGameTrainer,
    FullGameTrainingError,
    load_full_game_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "train" / "full_game_smoke.yaml"


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_cardplay_warm_start(path: Path, base: FullGameConfig) -> tuple[str, str, torch.Tensor]:
    model_config = load_bird_dou_config(base.cardplay_model_path)
    feature_config = replace(
        load_feature_config(base.feature_path),
        decomposition_features=base.decomposition_features,
    )
    model = BirdDouModel(model_config)
    state = model.state_dict()
    key = next(iter(state))
    expected = state[key].detach().clone()
    torch.save(
        {
            "model_fingerprint": model_config.fingerprint(),
            "feature_fingerprint": _stable_hash(asdict(feature_config)),
            "model": state,
            "state": {"policy_version": 17},
        },
        path,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    del model, state
    gc.collect()
    return digest, key, expected


def test_full_game_trainer_updates_checkpoints_and_resumes(tmp_path: Path) -> None:
    base = load_full_game_config(CONFIG_PATH)
    first_config = replace(base, output_directory=tmp_path, episodes=1)
    first = FullGameTrainer(first_config)
    result = first.train()

    assert result.state.episodes == 1
    assert result.state.learner_updates == 2
    assert result.state.bid_pretraining_updates == 1
    assert result.state.policy_version == 2
    assert result.state.stage == "bid_win_frozen"
    assert result.state.frames > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in result.losses.values())
    for name in (
        "checkpoint.pt",
        "manifest.json",
        "metrics.jsonl",
        "bid_pretraining_metrics.jsonl",
        "league.json",
        "bid_head.ckpt",
        "cardplay.ckpt",
    ):
        assert (tmp_path / name).is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trainer_mode"] == "full_game_joint"
    assert manifest["training_phase"] == "bid_win_frozen"
    assert manifest["feature_schema_version"] == 1
    assert manifest["model_arch_version"] == ("bird_dou_bid_head_v1+bird_dou_no_belief_v1")
    assert manifest["bid_model_fingerprint"]
    assert manifest["cardplay_model_fingerprint"]
    assert manifest["continuation_policy_hash"]
    assert manifest["continuation_model_architecture"] == "longest_move_smoke_only"
    assert manifest["continuation_decision_mode"] == "longest_move"
    assert manifest["optimizer_state"]
    assert manifest["rng_state"]
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert checkpoint["league_snapshot"]["schedule_cursor"] == 1
    assert len(checkpoint["pretraining_history"]) == 1
    assert checkpoint["pretraining_history"][0]["continuation_policy_hash"]

    del first
    gc.collect()
    loaded_policy = load_full_game_checkpoint_policy(
        "loaded-full-game",
        tmp_path / "checkpoint.pt",
        first_config.bid_model_path,
        first_config.cardplay_model_path,
        first_config.feature_path,
        load_rule_config(first_config.rules_path),
        "cpu",
    )
    assert loaded_policy.policy_id == "loaded-full-game"
    del loaded_policy
    gc.collect()

    resumed = FullGameTrainer(replace(first_config, episodes=2))
    resumed.load_checkpoint()
    resumed_result = resumed.train()
    assert resumed_result.state.episodes == 2
    assert resumed_result.state.learner_updates == 3
    assert resumed_result.state.bid_pretraining_updates == 1
    assert resumed_result.state.policy_version == 3
    assert len(resumed_result.metrics_history) == 2


def test_full_game_loads_verified_cardplay_and_reuses_it_as_continuation(
    tmp_path: Path,
) -> None:
    base = load_full_game_config(CONFIG_PATH)
    source = tmp_path / "strong-cardplay.pt"
    digest, key, expected = _write_cardplay_warm_start(source, base)
    config = replace(
        base,
        output_directory=tmp_path / "joint",
        bid_pretraining_batches=0,
        cardplay_checkpoint_path=source,
        cardplay_checkpoint_sha256=digest,
        cardplay_policy_version=17,
        allow_random_cardplay_smoke=False,
    )
    trainer = FullGameTrainer(config)

    torch.testing.assert_close(trainer.cardplay_model.state_dict()[key], expected)
    assert trainer.continuation_policy is trainer.cardplay_policy
    assert trainer.continuation_policy_hash == digest
    assert trainer.continuation_policy_version == 17
    assert trainer.continuation_model_architecture == "bird_dou_no_belief_v1"
    assert trainer.continuation_decision_mode == "mc_q"
    assert (
        config.fingerprint()
        == replace(
            config, cardplay_checkpoint_path=tmp_path / "relocated-cardplay.pt"
        ).fingerprint()
    )

    with pytest.raises(ValueError, match="SHA-256"):
        replace(config, cardplay_checkpoint_sha256="bad")

    del trainer
    gc.collect()
    with pytest.raises(FullGameTrainingError, match="SHA-256 mismatch"):
        FullGameTrainer(replace(config, cardplay_checkpoint_sha256="0" * 64))
