"""Privileged-label isolation, artifact, offline pretrain, and joint-loss tests for M5."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.belief.data import (
    BeliefDataset,
    extract_hidden_assignment,
    generate_belief_dataset,
    load_belief_dataset,
    save_belief_dataset,
)
from birddou.belief.training import (
    BeliefOfflineTrainer,
    BeliefPretrainConfig,
    joint_belief_policy_loss,
)
from birddou.eval.baselines import LongestMovePolicy, SeededRandomPolicy
from birddou.features import FeatureConfig
from birddou.models.action_encoder import ActionEncoderConfig
from birddou.models.belief_bird_dou import (
    BELIEF_BIRD_DOU_ARCHITECTURE,
    BeliefBirdDouConfig,
    BeliefBirdDouModel,
)
from birddou.models.bird_dou import BirdDouConfig
from birddou.models.history_encoder import HistoryEncoderConfig
from birddou.models.rank_mixer import RankMixerConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"


def tiny_config() -> BeliefBirdDouConfig:
    width = 8
    base = BirdDouConfig(
        d_model=width,
        rank_mixer=RankMixerConfig(
            d_model=width,
            blocks=1,
            attention_every=1,
            attention_heads=1,
            rank_embedding_dim=2,
            count_embedding_dim=2,
            flag_embedding_dim=2,
            swiglu_multiplier=1,
            dropout=0.0,
            drop_path=0.0,
        ),
        history=HistoryEncoderConfig(
            d_model=width,
            max_length=96,
            count_embedding_dim=2,
            categorical_embedding_dim=2,
            gru_layers=1,
            transformer_layers=1,
            attention_heads=1,
            feedforward_multiplier=1,
            dropout=0.0,
        ),
        action=ActionEncoderConfig(
            d_model=width,
            rank_blocks=0,
            attention_heads=1,
            count_embedding_dim=2,
            meta_embedding_dim=2,
            swiglu_multiplier=1,
            dropout=0.0,
        ),
        role_adapter_dim=2,
        score_quantiles=3,
        output_hidden_multiplier=1,
        output_hidden_layers=1,
    )
    return BeliefBirdDouConfig(
        schema_version=1,
        architecture=BELIEF_BIRD_DOU_ARCHITECTURE,
        feature_schema_version=1,
        base=base,
        count_embedding_dim=2,
        hidden_multiplier=1,
        dropout=0.0,
        enabled=True,
    )


def small_dataset() -> BeliefDataset:
    rules = load_rule_config(RULES_PATH)
    return generate_belief_dataset(
        1,
        5009,
        rules,
        (
            SeededRandomPolicy("dataset:random", 5009),
            LongestMovePolicy("dataset:longest"),
        ),
        FeatureConfig(decomposition_features=False),
    )


def test_mixed_policy_dataset_labels_and_npz_roundtrip(tmp_path: Path) -> None:
    """Only the generator reads full state; persisted labels exactly satisfy public counts."""
    dataset = small_dataset()
    assert dataset.state_count > 0
    assert dataset.policy_ids == ("dataset:random", "dataset:longest")
    assert set(dataset.policy_index.tolist()) == {0, 1}
    assert torch.all(dataset.batch.chosen_action_flat_index >= 0)

    selected = dataset.select(torch.tensor([dataset.state_count - 1, 0]))
    assert selected.state_count == 2
    assert selected.batch.action_offsets[0].item() == 0
    assert selected.batch.action_offsets[-1].item() == selected.batch.action_count
    artifact = save_belief_dataset(
        dataset,
        tmp_path / "belief.npz",
        game_count=1,
        master_seed=5009,
    )
    assert artifact.dataset_path.is_file()
    assert artifact.manifest_path.is_file()
    assert artifact.sha256 == hashlib.sha256(artifact.dataset_path.read_bytes()).hexdigest()
    restored = load_belief_dataset(artifact.dataset_path)
    assert restored.policy_ids == dataset.policy_ids
    assert torch.equal(restored.true_assignment_a, dataset.true_assignment_a)
    assert torch.equal(restored.policy_index, dataset.policy_index)
    for field in dataset.batch.__dataclass_fields__:
        if field != "schema_version":
            assert torch.equal(getattr(restored.batch, field), getattr(dataset.batch, field))


def test_oracle_extraction_is_separate_from_the_public_observation() -> None:
    """The training label reconstructs unknown_pool but is absent from Observation."""
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    observation = environment.reset(5010, rules)
    label = extract_hidden_assignment(environment.serialize(), observation)
    assert len(label) == 15
    assert sum(label) == observation["cards_left"][1]
    assert "hands" not in observation
    assert label != observation["unknown_pool"]


def test_frozen_offline_pretrain_then_joint_unfreeze(tmp_path: Path) -> None:
    """Offline NLL leaves the public encoder frozen, then joint loss reaches both paths."""
    dataset = small_dataset()
    subset = dataset.select(torch.arange(min(8, dataset.state_count), dtype=torch.int64))
    model = BeliefBirdDouModel(tiny_config())
    public_before = {
        key: value.detach().clone()
        for key, value in model.base.rank_token_encoder.state_dict().items()
    }
    scorer_before = {
        key: value.detach().clone() for key, value in model.belief_scores.state_dict().items()
    }
    trainer = BeliefOfflineTrainer(
        model,
        BeliefPretrainConfig(
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            freeze_public_encoder=True,
        ),
    )
    result = trainer.train(subset, tmp_path / "belief-pretrain.pt")
    assert result.update_count == 2
    assert result.checkpoint_path is not None and result.checkpoint_path.is_file()
    assert all(np.isfinite(result.losses))
    assert all(
        torch.equal(value, model.base.rank_token_encoder.state_dict()[key])
        for key, value in public_before.items()
    )
    assert any(
        not torch.equal(value, model.belief_scores.state_dict()[key])
        for key, value in scorer_before.items()
    )

    joint_losses = trainer.joint_finetune(subset, epochs=1, belief_coefficient=0.2)
    assert len(joint_losses) == 2 and all(np.isfinite(joint_losses))
    assert model.base.rank_token_encoder.projection.weight.grad is not None
    assert model.belief_scores.network[0].weight.grad is not None


def test_joint_loss_rejects_invalid_coefficient() -> None:
    with np.testing.assert_raises_regex(ValueError, "coefficient"):
        joint_belief_policy_loss(torch.tensor(1.0), torch.tensor(2.0), -0.1)
