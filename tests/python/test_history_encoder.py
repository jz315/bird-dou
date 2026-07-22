"""Causality, padding, gating, ablation, and gradient tests for E018."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.history_encoder import (
    CausalHistoryTransformer,
    HistoryEncoderConfig,
    HistoryEventEncoder,
    RoleGatedHistoryEncoder,
    RoleHistoryGate,
    load_history_encoder_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
MODEL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model" / "bird_dou_v1.yaml"


def small_config(
    *,
    gru_enabled: bool = True,
    transformer_enabled: bool = True,
    role_gate_enabled: bool = True,
) -> HistoryEncoderConfig:
    return HistoryEncoderConfig(
        d_model=32,
        max_length=96,
        count_embedding_dim=4,
        categorical_embedding_dim=4,
        gru_layers=2,
        transformer_layers=2,
        attention_heads=4,
        feedforward_multiplier=2,
        dropout=0.0,
        gru_enabled=gru_enabled,
        transformer_enabled=transformer_enabled,
        role_gate_enabled=role_gate_enabled,
    )


def real_history_batch() -> RaggedBatch:
    rules = load_rule_config(RULES_PATH)
    initial_environment = PyDdzEnv()
    initial = initial_environment.reset(18018, rules)
    progressed_environment = PyDdzEnv()
    progressed_environment.reset(18019, rules)
    progressed_environment.step(progressed_environment.legal_actions()[0])
    progressed = progressed_environment.observe(progressed_environment.current_player)
    return encode_ragged_batch(
        (initial, progressed),
        (initial_environment.legal_actions(), progressed_environment.legal_actions()),
        rules,
        config=FeatureConfig(decomposition_features=False),
    )


def test_dual_encoder_handles_zero_history_and_backpropagates_finite_gradients() -> None:
    """Real padded histories flow through both branches and the seat gate safely."""
    batch = real_history_batch()
    config = small_config()
    model = RoleGatedHistoryEncoder(config)
    seat = batch.scalars[:, 0].to(torch.int64)
    encoding = model(
        batch.history_rank_counts,
        batch.history_meta,
        batch.history_mask,
        batch.scalars,
        seat,
    )
    loss = encoding.fused.square().mean() + encoding.scalar.square().mean()
    torch.autograd.backward((loss,))

    assert encoding.fused.shape == (2, 32)
    assert encoding.gru.shape == encoding.fused.shape
    assert encoding.transformer.shape == encoding.fused.shape
    assert encoding.gate.shape == encoding.fused.shape
    assert torch.equal(encoding.fused[0], torch.zeros(32))
    assert torch.isfinite(encoding.fused).all()
    assert torch.all((encoding.gate >= 0.0) & (encoding.gate <= 1.0))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_padding_values_do_not_change_valid_history_summary() -> None:
    """Masking, not arbitrary right-padding contents, defines the valid prefix."""
    batch = real_history_batch()
    config = small_config()
    model = RoleGatedHistoryEncoder(config).eval()
    seat = batch.scalars[:, 0].to(torch.int64)
    expected = model(
        batch.history_rank_counts,
        batch.history_meta,
        batch.history_mask,
        batch.scalars,
        seat,
    ).fused
    counts = batch.history_rank_counts.clone()
    metadata = batch.history_meta.clone()
    padding = ~batch.history_mask
    counts[padding] = 3
    metadata[padding] = 0
    actual = model(counts, metadata, batch.history_mask, batch.scalars, seat).fused

    assert torch.equal(expected, actual)


def test_transformer_is_causal_at_every_prefix() -> None:
    """Changing future events cannot alter any earlier encoded event."""
    torch.manual_seed(18018)
    config = small_config()
    transformer = CausalHistoryTransformer(config).eval()
    events = torch.randn(1, 6, config.d_model)
    mask = torch.ones(1, 6, dtype=torch.bool)
    baseline = transformer.forward_sequence(events, mask)
    changed = events.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10.0
    candidate = transformer.forward_sequence(changed, mask)

    torch.testing.assert_close(baseline[:, :4], candidate[:, :4], rtol=0.0, atol=0.0)
    assert not torch.equal(baseline[:, 4:], candidate[:, 4:])


def test_role_gate_and_branch_ablation_contracts_are_exact() -> None:
    """Seat bias changes mixtures, while a single enabled branch bypasses gating."""
    config = small_config()
    gate = RoleHistoryGate(config)
    for parameter in gate.gate.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        gate.seat_bias.weight[0].fill_(-2.0)
        gate.seat_bias.weight[1].fill_(2.0)
    gru = torch.ones(2, config.d_model)
    transformer = torch.zeros_like(gru)
    scalar = torch.zeros_like(gru)
    fused, weights = gate(gru, transformer, scalar, torch.tensor([0, 1]))
    assert torch.all(weights[0] < weights[1])
    assert torch.all(fused[0] < fused[1])

    batch = real_history_batch()
    seat = batch.scalars[:, 0].to(torch.int64)
    gru_only = RoleGatedHistoryEncoder(small_config(transformer_enabled=False))(
        batch.history_rank_counts,
        batch.history_meta,
        batch.history_mask,
        batch.scalars,
        seat,
    )
    transformer_only = RoleGatedHistoryEncoder(small_config(gru_enabled=False))(
        batch.history_rank_counts,
        batch.history_meta,
        batch.history_mask,
        batch.scalars,
        seat,
    )
    assert torch.equal(gru_only.fused, gru_only.gru)
    assert torch.equal(transformer_only.fused, transformer_only.transformer)
    assert torch.equal(gru_only.gate, torch.ones_like(gru_only.gate))
    assert torch.equal(transformer_only.gate, torch.zeros_like(transformer_only.gate))


def test_config_serialization_and_invalid_masks_are_rejected(tmp_path: Path) -> None:
    """Default dimensions are locked and state dictionaries restore exact output."""
    loaded = load_history_encoder_config(MODEL_CONFIG_PATH)
    assert loaded.d_model == 256
    assert loaded.max_length == 96
    assert loaded.gru_layers == 2
    assert loaded.transformer_layers == 3
    assert loaded.attention_heads == 8

    batch = real_history_batch()
    config = small_config()
    original = RoleGatedHistoryEncoder(config).eval()
    seat = batch.scalars[:, 0].to(torch.int64)
    expected = original(
        batch.history_rank_counts,
        batch.history_meta,
        batch.history_mask,
        batch.scalars,
        seat,
    ).fused
    path = tmp_path / "history.pt"
    torch.save(original.state_dict(), path)
    restored = RoleGatedHistoryEncoder(config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    actual = restored(
        batch.history_rank_counts,
        batch.history_meta,
        batch.history_mask,
        batch.scalars,
        seat,
    ).fused
    assert torch.equal(expected, actual)

    invalid_mask = batch.history_mask.clone()
    invalid_mask[0, 1] = True
    with pytest.raises(ValueError, match="valid prefix"):
        HistoryEventEncoder(config)(
            batch.history_rank_counts,
            batch.history_meta,
            invalid_mask,
        )
    with pytest.raises(ValueError, match="at least one"):
        small_config(gru_enabled=False, transformer_enabled=False)
