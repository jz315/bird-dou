"""Independent shape, ablation, serialization, and gradient tests for E017."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.features import FeatureConfig, encode_ragged_batch
from birddou.models.rank_mixer import (
    DropPath,
    RankMixer,
    RankMixerConfig,
    RankTokenEncoder,
    RelativeRankAttention,
    RmsNorm,
    load_rank_mixer_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
MODEL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model" / "bird_dou_v1.yaml"


def small_config(
    *,
    convolution_enabled: bool = True,
    attention_enabled: bool = True,
    numeric_enabled: bool = True,
    drop_path: float = 0.0,
) -> RankMixerConfig:
    return RankMixerConfig(
        d_model=32,
        blocks=4,
        attention_every=2,
        attention_heads=4,
        rank_embedding_dim=8,
        count_embedding_dim=4,
        flag_embedding_dim=3,
        swiglu_multiplier=2,
        dropout=0.0,
        drop_path=drop_path,
        convolution_enabled=convolution_enabled,
        attention_enabled=attention_enabled,
        numeric_enabled=numeric_enabled,
    )


def test_real_rank_batch_has_expected_shape_and_finite_end_to_end_gradients() -> None:
    """Ragged rank fields flow through every local/global block with finite grads."""
    rules = load_rule_config(RULES_PATH)
    environments = (PyDdzEnv(), PyDdzEnv())
    observations = [
        environment.reset(seed, rules)
        for environment, seed in zip(environments, (17, 18), strict=True)
    ]
    actions = [environment.legal_actions() for environment in environments]
    batch = encode_ragged_batch(
        observations,
        actions,
        rules,
        config=FeatureConfig(decomposition_features=False),
    )
    config = small_config()
    encoder = RankTokenEncoder(config)
    mixer = RankMixer(config)

    tokens = encoder(batch.rank_categorical, batch.rank_numeric)
    output = mixer(tokens)
    loss = output.square().mean()
    torch.autograd.backward((loss,))

    assert tokens.shape == (2, 15, 32)
    assert output.shape == tokens.shape
    assert torch.isfinite(output).all()
    assert mixer.attention_block_count == 2
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in (*encoder.parameters(), *mixer.parameters())
    )
    attention_biases = [
        module.relative_bias
        for module in mixer.attention_blocks
        if isinstance(module, RelativeRankAttention)
    ]
    assert len(attention_biases) == 2
    assert all(parameter.grad is not None for parameter in attention_biases)


def test_local_and_global_ablation_switches_are_exact_identity() -> None:
    """Disabling both mixer paths gives a true fallback, not another hidden layer."""
    config = small_config(convolution_enabled=False, attention_enabled=False)
    mixer = RankMixer(config)
    inputs = torch.randn(3, 15, config.d_model)

    assert mixer.attention_block_count == 0
    assert torch.equal(mixer(inputs), inputs)


def test_numeric_ablation_ignores_numeric_values_and_categorical_range_is_checked() -> None:
    """The numeric rank channel is independently removable for controlled ablation."""
    config = small_config(numeric_enabled=False)
    encoder = RankTokenEncoder(config)
    categorical = torch.zeros((2, 15, 9), dtype=torch.int64)
    categorical[..., 0] = torch.arange(15)
    numeric_a = torch.zeros((2, 15, 3), dtype=torch.float32)
    numeric_b = torch.randn((2, 15, 3), dtype=torch.float32)

    assert torch.equal(encoder(categorical, numeric_a), encoder(categorical, numeric_b))
    categorical[..., 1] = 5
    with pytest.raises(ValueError, match="0..4"):
        encoder(categorical, numeric_a)


def test_eval_output_survives_weights_only_save_and_load(tmp_path: Path) -> None:
    """The independent module has stable checkpoint names and exact restored output."""
    torch.manual_seed(17017)
    config = small_config(drop_path=0.2)
    original = RankMixer(config).eval()
    inputs = torch.randn(2, 15, config.d_model)
    expected = original(inputs)
    path = tmp_path / "rank-mixer.pt"
    torch.save(original.state_dict(), path)
    restored = RankMixer(config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)

    assert torch.equal(expected, restored(inputs))
    assert torch.equal(restored(inputs), restored(inputs))


def test_norm_drop_path_and_config_validation_are_explicit() -> None:
    """Primitive contracts reject invalid widths/probabilities and preserve eval data."""
    inputs = torch.randn(2, 15, 8)
    assert torch.equal(DropPath(0.5).eval()(inputs), inputs)
    normalized = RmsNorm(8)(inputs)
    assert normalized.shape == inputs.shape
    assert torch.isfinite(normalized).all()

    loaded = load_rank_mixer_config(MODEL_CONFIG_PATH)
    assert loaded.d_model == 256
    assert loaded.blocks == 4
    assert loaded.attention_every == 2
    assert loaded.attention_heads == 8
    assert loaded.convolution_enabled and loaded.attention_enabled

    with pytest.raises(ValueError, match="divisible"):
        RankMixerConfig(d_model=30, attention_heads=8)
    with pytest.raises(ValueError, match="expects"):
        RankMixer(small_config())(torch.zeros(2, 14, 32))
