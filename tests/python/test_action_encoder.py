"""Ragged action encoding, ablation, serialization, and scale tests for E019."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.action_encoder import (
    ActionEncoderConfig,
    RaggedActionEncoder,
    load_action_encoder_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
MODEL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model" / "bird_dou_v1.yaml"


def small_config(
    *,
    rank_blocks: int = 2,
    post_hand_enabled: bool = True,
    cross_attention_enabled: bool = True,
    set_context_enabled: bool = True,
) -> ActionEncoderConfig:
    return ActionEncoderConfig(
        d_model=32,
        rank_blocks=rank_blocks,
        attention_heads=4,
        count_embedding_dim=4,
        meta_embedding_dim=4,
        swiglu_multiplier=2,
        dropout=0.0,
        post_hand_enabled=post_hand_enabled,
        cross_attention_enabled=cross_attention_enabled,
        set_context_enabled=set_context_enabled,
    )


def real_action_batch() -> RaggedBatch:
    rules = load_rule_config(RULES_PATH)
    environments = (PyDdzEnv(), PyDdzEnv())
    observations = tuple(
        environment.reset(seed, rules)
        for environment, seed in zip(environments, (19019, 29019), strict=True)
    )
    return encode_ragged_batch(
        observations,
        tuple(environment.legal_actions() for environment in environments),
        rules,
        config=FeatureConfig(decomposition_features=False),
    )


def repeated_single_state_batch(batch: RaggedBatch, action_count: int) -> RaggedBatch:
    """Build a schema-valid stress batch from one real encoded legal action."""
    return replace(
        batch,
        rank_categorical=batch.rank_categorical[:1],
        rank_numeric=batch.rank_numeric[:1],
        history_rank_counts=batch.history_rank_counts[:1],
        history_meta=batch.history_meta[:1],
        history_mask=batch.history_mask[:1],
        scalars=batch.scalars[:1],
        action_rank_counts=batch.action_rank_counts[:1].repeat(action_count, 1),
        post_hand_counts=batch.post_hand_counts[:1].repeat(action_count, 1),
        action_meta=batch.action_meta[:1].repeat(action_count, 1),
        action_state_index=torch.zeros(action_count, dtype=torch.int64),
        action_offsets=torch.tensor([0, action_count], dtype=torch.int64),
        chosen_action_flat_index=torch.tensor([-1], dtype=torch.int64),
    )


def test_ragged_forward_cross_attention_and_gradients_are_finite() -> None:
    """All actions remain flat while attention is confined to 15 state rank tokens."""
    torch.manual_seed(19019)
    batch = real_action_batch()
    config = small_config()
    model = RaggedActionEncoder(config)
    state = torch.randn(batch.batch_size, config.d_model, requires_grad=True)
    rank_tokens = torch.randn(batch.batch_size, 15, config.d_model, requires_grad=True)
    encoding = model(batch, state, rank_tokens)
    loss = encoding.action.square().mean() + encoding.attention_weights.square().mean()
    loss.backward()

    assert encoding.action.shape == (batch.action_count, config.d_model)
    assert encoding.query.shape == encoding.action.shape
    assert encoding.rank_context.shape == encoding.action.shape
    assert encoding.attention_weights.shape == (batch.action_count, config.attention_heads, 15)
    assert encoding.set_mean.shape == (batch.batch_size, config.d_model)
    assert encoding.set_max.shape == encoding.set_mean.shape
    torch.testing.assert_close(
        encoding.attention_weights.sum(dim=-1),
        torch.ones(batch.action_count, config.attention_heads),
    )
    assert torch.isfinite(encoding.action).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert rank_tokens.grad is not None and torch.isfinite(rank_tokens.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_single_and_large_candidate_sets_need_no_padding() -> None:
    """The same forward handles one action and thousands with storage linear in M."""
    source = real_action_batch()
    config = small_config(rank_blocks=0)
    model = RaggedActionEncoder(config).eval()

    singleton = repeated_single_state_batch(source, 1)
    singleton_encoding = model(
        singleton,
        torch.randn(1, config.d_model),
        torch.randn(1, 15, config.d_model),
    )
    assert singleton_encoding.action.shape == (1, config.d_model)
    torch.testing.assert_close(singleton_encoding.set_mean[0], singleton_encoding.base_action[0])
    torch.testing.assert_close(singleton_encoding.set_max[0], singleton_encoding.base_action[0])

    large = repeated_single_state_batch(source, 4_096)
    large_encoding = model(
        large,
        torch.randn(1, config.d_model),
        torch.randn(1, 15, config.d_model),
    )
    assert large_encoding.action.shape == (4_096, config.d_model)
    assert large_encoding.attention_weights.shape == (4_096, 4, 15)
    assert torch.isfinite(large_encoding.action).all()


def test_action_ablation_switches_remove_only_their_declared_inputs() -> None:
    """Post-hand, rank attention, and set-context switches have observable contracts."""
    torch.manual_seed(19020)
    batch = real_action_batch()
    state = torch.randn(batch.batch_size, 32)
    rank_tokens = torch.randn(batch.batch_size, 15, 32)

    no_post = RaggedActionEncoder(small_config(post_hand_enabled=False)).eval()
    changed_post = replace(batch, post_hand_counts=torch.zeros_like(batch.post_hand_counts))
    torch.testing.assert_close(
        no_post(batch, state, rank_tokens).action,
        no_post(changed_post, state, rank_tokens).action,
        rtol=0.0,
        atol=0.0,
    )

    no_cross = RaggedActionEncoder(small_config(cross_attention_enabled=False)).eval()
    cross_encoding = no_cross(batch, state, rank_tokens)
    assert torch.equal(cross_encoding.rank_context, torch.zeros_like(cross_encoding.rank_context))
    assert torch.equal(
        cross_encoding.attention_weights,
        torch.zeros_like(cross_encoding.attention_weights),
    )

    no_set = RaggedActionEncoder(small_config(set_context_enabled=False)).eval()
    set_encoding = no_set(batch, state, rank_tokens)
    assert torch.equal(set_encoding.action, set_encoding.base_action)


def test_action_config_roundtrip_and_invalid_context_are_explicit(tmp_path: Path) -> None:
    """Default architecture is locked and strict state dictionaries reproduce output."""
    loaded = load_action_encoder_config(MODEL_CONFIG_PATH)
    assert loaded.d_model == 256
    assert loaded.rank_blocks == 2
    assert loaded.attention_heads == 8
    assert loaded.decomposition_count_cap == 255

    batch = real_action_batch()
    config = small_config(rank_blocks=0)
    state = torch.randn(batch.batch_size, config.d_model)
    rank_tokens = torch.randn(batch.batch_size, 15, config.d_model)
    original = RaggedActionEncoder(config).eval()
    expected = original(batch, state, rank_tokens).action
    path = tmp_path / "action.pt"
    torch.save(original.state_dict(), path)
    restored = RaggedActionEncoder(config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    actual = restored(batch, state, rank_tokens).action
    assert torch.equal(expected, actual)

    with pytest.raises(ValueError, match="state shape"):
        original(batch, state[:, :-1], rank_tokens)
    with pytest.raises(ValueError, match="rank token shape"):
        original(batch, state, rank_tokens[:, :-1])
    with pytest.raises(ValueError, match="NaN"):
        invalid_state = state.clone()
        invalid_state[0, 0] = float("nan")
        original(batch, invalid_state, rank_tokens)
