"""End-to-end no-Belief BIRD-Dou model tests for E020."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.action_encoder import ActionEncoderConfig
from birddou.models.bird_dou import (
    BIRD_DOU_ARCHITECTURE,
    BirdDouConfig,
    BirdDouModel,
    decision_values,
    load_bird_dou_config,
    seat_from_scalars,
)
from birddou.models.history_encoder import HistoryEncoderConfig
from birddou.models.rank_mixer import RankMixerConfig
from birddou.models.role_adapters import RoleAdapterConfig, RoleSeatAdapter, RoleSpecificLinear
from birddou.models.segment_ops import segment_sum

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
MODEL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model" / "bird_dou_v1.yaml"


def tiny_config() -> BirdDouConfig:
    d_model = 32
    return BirdDouConfig(
        d_model=d_model,
        rank_mixer=RankMixerConfig(
            d_model=d_model,
            blocks=2,
            attention_every=1,
            attention_heads=4,
            rank_embedding_dim=8,
            count_embedding_dim=4,
            flag_embedding_dim=4,
            swiglu_multiplier=2,
            dropout=0.0,
            drop_path=0.0,
        ),
        history=HistoryEncoderConfig(
            d_model=d_model,
            max_length=96,
            count_embedding_dim=4,
            categorical_embedding_dim=4,
            gru_layers=1,
            transformer_layers=1,
            attention_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
        action=ActionEncoderConfig(
            d_model=d_model,
            rank_blocks=1,
            attention_heads=4,
            count_embedding_dim=4,
            meta_embedding_dim=4,
            swiglu_multiplier=2,
            dropout=0.0,
        ),
        role_adapter_dim=8,
        score_quantiles=5,
        output_hidden_multiplier=2,
        output_hidden_layers=1,
    )


def all_role_batch() -> RaggedBatch:
    """Return current-player states for landlord, downstream, and upstream seats."""
    rules = load_rule_config(RULES_PATH)
    environments = [PyDdzEnv(), PyDdzEnv(), PyDdzEnv()]
    environments[0].reset(20020, rules)
    environments[1].reset(20021, rules)
    environments[1].step(environments[1].legal_actions()[0])
    environments[2].reset(20022, rules)
    environments[2].step(environments[2].legal_actions()[0])
    environments[2].step(environments[2].legal_actions()[0])
    observations = tuple(env.observe(env.current_player) for env in environments)
    actions = tuple(env.legal_actions() for env in environments)
    return encode_ragged_batch(
        observations,
        actions,
        rules,
        config=FeatureConfig(decomposition_features=False),
    )


def test_complete_forward_normalizes_every_segment_and_backpropagates() -> None:
    """All required heads operate on a mixed-role ragged batch with finite gradients."""
    torch.manual_seed(20020)
    batch = all_role_batch()
    config = tiny_config()
    model = BirdDouModel(config)
    output = model(batch)

    assert output.seat.tolist() == [0, 1, 2]
    assert output.policy_logit.shape == (batch.action_count,)
    assert output.policy_probability.shape == output.policy_logit.shape
    assert output.policy_log_normalizer.shape == (batch.batch_size,)
    assert output.win_logit.shape == output.policy_logit.shape
    assert output.score_if_win.shape == output.policy_logit.shape
    assert output.score_if_loss.shape == output.policy_logit.shape
    assert output.expected_score.shape == output.policy_logit.shape
    assert output.mc_q.shape == output.policy_logit.shape
    assert output.turns_to_finish.shape == output.policy_logit.shape
    assert output.score_win_quantiles.shape == (batch.action_count, config.score_quantiles)
    assert output.score_loss_quantiles.shape == output.score_win_quantiles.shape
    torch.testing.assert_close(
        segment_sum(output.policy_probability, batch.action_offsets),
        torch.ones(batch.batch_size),
    )
    torch.testing.assert_close(
        output.policy_log_probability.exp(),
        output.policy_probability,
    )
    assert torch.all(output.score_if_win >= 0.0)
    assert torch.all(output.score_if_loss <= 0.0)
    assert torch.all(output.turns_to_finish >= 0.0)
    assert torch.all(torch.diff(output.score_win_quantiles, dim=-1) >= 0.0)
    assert torch.all(torch.diff(output.score_loss_quantiles, dim=-1) >= 0.0)

    tensors = (
        output.policy_logit,
        output.win_logit,
        output.score_if_win,
        output.score_if_loss,
        output.mc_q,
        output.turns_to_finish,
        output.score_win_quantiles,
        output.score_loss_quantiles,
    )
    loss = sum(tensor.square().mean() for tensor in tensors)
    torch.autograd.backward((loss,))
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cpu_bfloat16_autocast_forward_backward_keeps_reductions_fp32() -> None:
    torch.manual_seed(20024)
    batch = all_role_batch()
    model = BirdDouModel(tiny_config())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        loss = output.mc_q.float().square().mean() - output.policy_log_probability.mean()
    torch.autograd.backward((loss,))

    assert output.mc_q.dtype == torch.bfloat16
    assert output.policy_probability.dtype == torch.float32
    assert output.policy_log_probability.dtype == torch.float32
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_default_config_is_medium_sized_and_roundtrips_exactly(tmp_path: Path) -> None:
    """The frozen model stays in its 20M-35M budget and restores weights exactly."""
    loaded = load_bird_dou_config(MODEL_CONFIG_PATH)
    assert loaded.architecture == BIRD_DOU_ARCHITECTURE
    assert loaded.d_model == 256
    assert loaded.score_quantiles == 11
    assert not loaded.belief_enabled
    parameter_count = sum(parameter.numel() for parameter in BirdDouModel(loaded).parameters())
    assert 20_000_000 <= parameter_count <= 35_000_000

    batch = all_role_batch()
    config = tiny_config()
    original = BirdDouModel(config).eval()
    expected = original(batch).mc_q
    path = tmp_path / "bird-dou.pt"
    torch.save(original.state_dict(), path)
    restored = BirdDouModel(config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(expected, restored(batch).mc_q)
    assert config.fingerprint() == tiny_config().fingerprint()


def test_role_conditioning_and_decision_modes_have_explicit_contracts() -> None:
    """Seat-specific adapters/heads differ while public utility modes remain finite."""
    torch.manual_seed(20021)
    adapter = RoleSeatAdapter(RoleAdapterConfig(d_model=8, bottleneck_dim=2)).eval()
    state = torch.zeros(3, 8)
    seat = torch.tensor([0, 1, 2], dtype=torch.int64)
    adapted = adapter(state, seat)
    assert adapted.shape == state.shape
    assert not torch.equal(adapted[0], adapted[1])
    assert not torch.equal(adapted[1], adapted[2])

    linear = RoleSpecificLinear(8, 2)
    projected = linear(adapted, seat)
    assert projected.shape == (3, 2)
    assert torch.isfinite(projected).all()

    output = BirdDouModel(tiny_config()).eval()(all_role_batch())
    for mode in ("policy", "wp", "score", "mc_q", "risk"):
        values = decision_values(output, mode, risk_aversion=0.25)
        assert values.shape == output.mc_q.shape
        assert torch.isfinite(values).all()
    with pytest.raises(ValueError, match="risk_aversion"):
        decision_values(output, "risk", risk_aversion=-1.0)


def test_seat_derivation_and_no_belief_schema_reject_invalid_inputs() -> None:
    """Relative roles survive landlord rotation and no-Belief is enforced by config."""
    scalars = torch.zeros(3, 15, dtype=torch.float32)
    scalars[:, 2] = torch.tensor([0.0, 2.0, 1.0])
    assert seat_from_scalars(scalars).tolist() == [0, 1, 2]
    scalars[0, 2] = 0.5
    with pytest.raises(ValueError, match="integer"):
        seat_from_scalars(scalars)
    with pytest.raises(ValueError, match="cannot enable Belief"):
        BirdDouConfig(belief_enabled=True)
