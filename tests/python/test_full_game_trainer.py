"""Resumable complete-game Bid Head/Cardplay curriculum smoke tests."""

import gc
import json
from dataclasses import replace
from pathlib import Path

import torch

from birddou import load_rule_config
from birddou.cli.policy_artifacts import load_full_game_checkpoint_policy
from birddou.rl import FullGameTrainer, load_full_game_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "train" / "full_game_smoke.yaml"


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
    assert manifest["optimizer_state"]
    assert manifest["rng_state"]
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert checkpoint["league_snapshot"]["schedule_cursor"] == 1
    assert len(checkpoint["pretraining_history"]) == 1

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
