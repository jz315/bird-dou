"""Belief-state fusion, privacy, gradients, and serialization tests for M5."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.belief import belief_nll
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.action_encoder import ActionEncoderConfig
from birddou.models.belief_bird_dou import (
    BELIEF_BIRD_DOU_ARCHITECTURE,
    BeliefBirdDouConfig,
    BeliefBirdDouModel,
    belief_constraints_from_batch,
    load_belief_bird_dou_config,
)
from birddou.models.bird_dou import BirdDouConfig, BirdDouModel
from birddou.models.history_encoder import HistoryEncoderConfig
from birddou.models.rank_mixer import RankMixerConfig
from birddou.models.segment_ops import segment_sum

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
MODEL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model" / "bird_dou_belief_v1.yaml"


def tiny_config() -> BeliefBirdDouConfig:
    width = 16
    base = BirdDouConfig(
        d_model=width,
        rank_mixer=RankMixerConfig(
            d_model=width,
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
            d_model=width,
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
            d_model=width,
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
    return BeliefBirdDouConfig(
        schema_version=1,
        architecture=BELIEF_BIRD_DOU_ARCHITECTURE,
        feature_schema_version=1,
        base=base,
        count_embedding_dim=4,
        hidden_multiplier=1,
        dropout=0.0,
        enabled=True,
    )


def real_batch_and_labels() -> tuple[RaggedBatch, torch.Tensor]:
    rules = load_rule_config(RULES_PATH)
    environments = [PyDdzEnv(), PyDdzEnv()]
    environments[0].reset(5007, rules)
    environments[1].reset(5008, rules)
    environments[1].step(environments[1].legal_actions()[0])
    observations = tuple(env.observe(env.current_player) for env in environments)
    batch = encode_ragged_batch(
        observations,
        tuple(env.legal_actions() for env in environments),
        rules,
        config=FeatureConfig(decomposition_features=False),
    )
    labels = []
    for environment in environments:
        state = cast(dict[str, object], json.loads(environment.serialize()))["state"]
        hands = cast(dict[str, object], state)["hands"]
        observer = environment.current_player
        labels.append(cast(list[list[int]], hands)[(observer + 1) % 3])
    return batch, torch.tensor(labels, dtype=torch.int64)


def test_belief_model_conserves_cards_trains_jointly_and_normalizes_policy() -> None:
    """Exact marginals influence actions while supervised and policy gradients coexist."""
    torch.manual_seed(5007)
    batch, labels = real_batch_and_labels()
    model = BeliefBirdDouModel(tiny_config())
    output = model(batch)
    unknown, capacity_a, _ = belief_constraints_from_batch(batch)

    assert output.scores.shape == (batch.batch_size, 15, 5)
    assert output.belief_pool.shape == (batch.batch_size, 16)
    assert output.fused_state.shape == output.belief_pool.shape
    torch.testing.assert_close(output.marginals.expected_a.sum(dim=1), capacity_a.to(torch.float32))
    torch.testing.assert_close(
        output.marginals.expected_a + output.marginals.expected_b,
        unknown.to(torch.float32),
    )
    torch.testing.assert_close(
        segment_sum(output.policy.policy_probability, batch.action_offsets),
        torch.ones(batch.batch_size),
    )
    loss = (
        belief_nll(output.scores.float(), unknown, capacity_a, labels)
        + output.policy.mc_q.square().mean()
        + output.policy.policy_logit.square().mean()
    )
    torch.autograd.backward((loss,))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert all(parameter.grad is not None for parameter in model.belief_scores.parameters())


def test_student_output_never_reads_true_hidden_labels_and_uses_belief() -> None:
    """Changing an external oracle label is inert, while changing CRF scores is observable."""
    batch, labels = real_batch_and_labels()
    model = BeliefBirdDouModel(tiny_config()).eval()
    with torch.no_grad():
        model.belief_scale.fill_(1.0)
    first = model(batch).policy.mc_q.detach().clone()
    shuffled_labels = labels.flip(0)
    assert not torch.equal(labels, shuffled_labels)
    second = model(batch).policy.mc_q.detach().clone()
    assert torch.equal(first, second)

    with torch.no_grad():
        for parameter in model.belief_scores.parameters():
            parameter.zero_()
    uniform_belief = model(batch).policy.mc_q.detach()
    assert not torch.equal(first, uniform_belief)


def test_zero_initialized_belief_residual_cannot_degrade_base_policy() -> None:
    """A copied no-Belief checkpoint is bit-exact until joint training opens the gate."""
    batch, _ = real_batch_and_labels()
    config = tiny_config()
    base = BirdDouModel(config.base).eval()
    belief = BeliefBirdDouModel(config).eval()
    belief.base.load_state_dict(base.state_dict(), strict=True)

    expected = base(batch)
    actual = belief(batch).policy
    assert belief.belief_scale.item() == 0.0
    assert torch.equal(expected.policy_logit, actual.policy_logit)
    assert torch.equal(expected.mc_q, actual.mc_q)


def test_config_checkpoint_and_public_capacity_validation(tmp_path: Path) -> None:
    """The Belief architecture is versioned and rejects inconsistent public constraints."""
    loaded = load_belief_bird_dou_config(MODEL_CONFIG_PATH)
    assert loaded.architecture == BELIEF_BIRD_DOU_ARCHITECTURE
    assert loaded.enabled
    assert not loaded.base.belief_enabled
    assert loaded.base.d_model == 256

    batch, _ = real_batch_and_labels()
    config = tiny_config()
    original = BeliefBirdDouModel(config).eval()
    expected = original(batch).policy.mc_q
    path = tmp_path / "belief.pt"
    torch.save(original.state_dict(), path)
    restored = BeliefBirdDouModel(config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(expected, restored(batch).policy.mc_q)
    assert config.fingerprint() == tiny_config().fingerprint()

    scalars = batch.scalars.clone()
    scalars[0, 4] -= 1.0
    invalid = replace(batch, scalars=scalars)
    with torch.no_grad(), pytest.raises(ValueError, match="unknown pool"):
        original(invalid)
