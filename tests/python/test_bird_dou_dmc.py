"""Shared-model collection, multi-head loss, checkpoint, and resume tests for E020."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.eval.baselines import PolicyDecisionContext
from birddou.eval.paired_deals import SeatRole
from birddou.features import FeatureConfig, encode_ragged_batch
from birddou.models.action_encoder import ActionEncoderConfig
from birddou.models.bird_dou import BirdDouConfig, BirdDouModel
from birddou.models.history_encoder import HistoryEncoderConfig
from birddou.models.rank_mixer import RankMixerConfig
from birddou.rl.bird_dou_dmc import (
    BirdDouDmcTrainer,
    BirdDouPolicy,
    BirdDouTransition,
    bird_dou_dmc_loss,
    collate_bird_dou_transitions,
    load_bird_dou_dmc_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "train" / "bird_dou_dmc_smoke.yaml"


def tiny_config() -> BirdDouConfig:
    d_model = 16
    return BirdDouConfig(
        d_model=d_model,
        rank_mixer=RankMixerConfig(
            d_model=d_model,
            blocks=1,
            attention_every=1,
            attention_heads=2,
            rank_embedding_dim=4,
            count_embedding_dim=2,
            flag_embedding_dim=2,
            swiglu_multiplier=1,
            dropout=0.0,
            drop_path=0.0,
        ),
        history=HistoryEncoderConfig(
            d_model=d_model,
            max_length=96,
            count_embedding_dim=2,
            categorical_embedding_dim=2,
            gru_layers=1,
            transformer_layers=1,
            attention_heads=2,
            feedforward_multiplier=1,
            dropout=0.0,
        ),
        action=ActionEncoderConfig(
            d_model=d_model,
            rank_blocks=0,
            attention_heads=2,
            count_embedding_dim=2,
            meta_embedding_dim=2,
            swiglu_multiplier=1,
            dropout=0.0,
        ),
        role_adapter_dim=4,
        score_quantiles=3,
        output_hidden_multiplier=1,
        output_hidden_layers=1,
    )


def test_collation_multi_head_loss_and_policy_use_complete_action_segments() -> None:
    """Chosen indices shift correctly and every loss is finite over cached states."""
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    observation = environment.reset(20023, rules)
    legal_actions = tuple(environment.legal_actions())
    feature_config = FeatureConfig(decomposition_features=False)
    batch = encode_ragged_batch(
        (observation,),
        (legal_actions,),
        rules,
        config=feature_config,
    )
    transitions = tuple(
        BirdDouTransition(
            serialized_state=environment.serialize(),
            seat=0,
            role=SeatRole.LANDLORD,
            batch=batch,
            chosen_action_index=index,
            behavior_logprob=0.0,
            policy_version=0,
            target=target,
            raw_score=target * 2.0,
            win_target=float(target > 0.0),
            turns_to_finish=float(2 - index),
        )
        for index, target in enumerate((1.0, -1.0))
    )
    combined = collate_bird_dou_transitions(transitions)
    assert combined.action_offsets.tolist() == [0, batch.action_count, 2 * batch.action_count]
    assert combined.chosen_action_flat_index.tolist() == [0, batch.action_count + 1]

    model = BirdDouModel(tiny_config())
    output = model(combined)
    config = load_bird_dou_dmc_config(CONFIG_PATH)
    losses = bird_dou_dmc_loss(
        output,
        combined,
        torch.tensor([1.0, -1.0]),
        torch.tensor([2.0, -2.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([2.0, 1.0]),
        config,
    )
    torch.autograd.backward((losses.total,))
    assert all(np.isfinite(value) for value in losses.detached().values())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    policy = BirdDouPolicy(
        "bird-dou:test",
        model,
        rules,
        feature_config,
    )
    selected = policy.select_action(
        observation,
        legal_actions,
        PolicyDecisionContext(0, 20023, "smoke", 0, SeatRole.LANDLORD, 0),
    )
    assert 0 <= selected < len(legal_actions)


def test_one_episode_updates_all_roles_and_restores_checkpoint(tmp_path: Path) -> None:
    """A complete game gives every seat terminal targets and produces an exact checkpoint."""
    config = replace(
        load_bird_dou_dmc_config(CONFIG_PATH),
        episodes=1,
        output_directory=tmp_path / "bird-dou",
        evaluation_deals=1,
        bootstrap_resamples=20,
    )
    trainer = BirdDouDmcTrainer(
        config,
        model_config=tiny_config(),
        feature_config=FeatureConfig(decomposition_features=False),
    )
    result = trainer.train()

    assert result.state.episodes == 1
    assert result.state.learner_updates == 1
    assert result.state.frames > 0
    assert result.state.policy_version == 1
    assert result.state.landlord_frames > 0
    assert result.state.landlord_down_frames > 0
    assert result.state.landlord_up_frames > 0
    assert (
        sum(
            (
                result.state.landlord_frames,
                result.state.landlord_down_frames,
                result.state.landlord_up_frames,
            )
        )
        == result.state.frames
    )
    assert all(np.isfinite(value) for value in result.losses.values())
    assert result.checkpoint_path.is_file()
    assert result.manifest_path.is_file()
    assert (config.output_directory / "bird_dou.ckpt").is_file()
    manifest = cast(dict[str, object], json.loads(result.manifest_path.read_text(encoding="utf-8")))
    assert manifest["trainer_mode"] == "bird_dou_dmc"
    assert manifest["training_phase"] == "bird_dou_no_belief_dmc"
    assert manifest["episodes"] == 1
    assert manifest["optimizer_state"] is True
    assert manifest["rng_state"] is True
    assert isinstance(manifest["league_snapshot"], str)
    assert (config.output_directory / "league.json").is_file()

    restored = BirdDouDmcTrainer(
        config,
        model_config=tiny_config(),
        feature_config=FeatureConfig(decomposition_features=False),
    )
    restored.load_checkpoint(result.checkpoint_path)
    assert asdict(restored.state) == asdict(trainer.state)
    assert restored.losses == trainer.losses
    assert restored.metrics_history == trainer.metrics_history
    assert restored.league.to_dict() == trainer.league.to_dict()
    assert restored.rng.bit_generator.state == trainer.rng.bit_generator.state
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_checked_in_config_declares_structured_dmc_without_expensive_decomposition() -> None:
    """The CPU smoke gate is explicit about its model, feature ablation, and objectives."""
    config = load_bird_dou_dmc_config(CONFIG_PATH)
    assert config.trainer_mode == "bird_dou_dmc"
    assert config.episodes == 1
    assert config.decision_mode == "mc_q"
    assert not config.decomposition_features
    assert config.mc_q_weight == 1.0
    assert config.policy_weight > 0.0
    assert config.win_weight > 0.0
