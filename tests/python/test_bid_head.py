"""Bidding feature boundary, constrained belief, and candidate-head tests."""

from pathlib import Path

import pytest
import torch

from birddou import PyDdzEnv, RuleConfig, load_rule_config
from birddou.features.ragged import FeatureEncodingError, encode_ragged_batch
from birddou.models.bid_head import (
    BidHead,
    BidHeadConfig,
    encode_bid_batch,
    load_bid_head_config,
)

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs" / "rules" / "canonical_full.yaml"
MODEL_PATH = ROOT / "configs" / "model" / "bid_head_v2.yaml"


def _rules() -> RuleConfig:
    return load_rule_config(RULES_PATH)


def _small_config() -> BidHeadConfig:
    return BidHeadConfig(2, "bird_dou_bid_head_v2", 32, 1, 1, 4, 2, 3, 0.0)


def test_native_bidding_observation_encodes_only_public_information() -> None:
    rules = _rules()
    environment = PyDdzEnv()
    observation = environment.reset(20260722, rules)
    actions = tuple(environment.legal_actions())
    batch = encode_bid_batch((observation,), (actions,), rules)

    assert observation["phase"] == "bidding"
    assert observation["role"] == "unassigned"
    assert observation["landlord"] is None
    assert sum(observation["own_hand"]) == 17
    assert sum(observation["unknown_pool"]) == 37
    assert sum(observation["public_bottom_cards"]) == 0
    assert batch.capacity_a.tolist() == [17]
    assert batch.capacity_b.tolist() == [17]
    assert batch.legal_action_code.tolist() == [0, 1, 2, 3]
    with pytest.raises(FeatureEncodingError, match="BidBatch"):
        encode_ragged_batch((observation,), (actions,), rules)


def test_bid_head_scores_ragged_actions_and_preserves_three_capacities() -> None:
    rules = _rules()
    environments = [PyDdzEnv(), PyDdzEnv()]
    observations = []
    legal = []
    for index, environment in enumerate(environments):
        observation = environment.reset(41 + index, rules)
        actions = tuple(environment.legal_actions())
        if index == 1:
            environment.step(actions[0])
            observation = environment.observe(environment.current_player)
            actions = tuple(environment.legal_actions())
        observations.append(observation)
        legal.append(actions)
    batch = encode_bid_batch(observations, legal, rules)
    output = BidHead(_small_config()).eval()(batch)

    assert output.policy_logits.shape == (sum(map(len, legal)),)
    assert output.mc_q.shape == output.policy_logits.shape
    assert output.mc_q is output.policy_logits
    assert output.win_probability.shape == output.policy_logits.shape
    assert output.expected_score.shape == output.policy_logits.shape
    for start, end in zip(batch.action_offsets[:-1], batch.action_offsets[1:], strict=True):
        segment = output.policy_probability[int(start) : int(end)]
        assert segment.sum().item() == pytest.approx(1.0)
    assert torch.allclose(
        output.belief.expected.sum(dim=1), output.belief.capacities.to(torch.float32)
    )
    assert output.belief.capacities[:, 2].tolist() == [3, 3]
    assert torch.isfinite(output.belief.log_partition).all()


def test_bid_head_config_is_versioned_and_default_model_runs() -> None:
    config = load_bid_head_config(MODEL_PATH)
    assert config.fingerprint() == load_bid_head_config(MODEL_PATH).fingerprint()
    rules = _rules()
    environment = PyDdzEnv()
    observation = environment.reset(7, rules)
    actions = tuple(environment.legal_actions())
    output = BidHead(config).eval()(encode_bid_batch((observation,), (actions,), rules))
    assert output.policy_probability.shape == (len(actions),)
    assert torch.isfinite(output.policy_probability).all()


def test_bid_head_cpu_bfloat16_autocast_preserves_fp32_crf_and_softmax() -> None:
    rules = _rules()
    environment = PyDdzEnv()
    observation = environment.reset(17, rules)
    actions = tuple(environment.legal_actions())
    batch = encode_bid_batch((observation,), (actions,), rules)
    model = BidHead(_small_config())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        loss = output.mc_q.float().square().mean() + output.belief.log_partition.mean()
    torch.autograd.backward((loss,))

    assert output.mc_q.dtype == torch.bfloat16
    assert output.policy_probability.dtype == torch.float32
    assert output.belief.log_partition.dtype == torch.float32
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("history_events", [0, 1, 2])
def test_bid_history_state_is_invariant_to_right_padding(history_events: int) -> None:
    rules = _rules()
    environment = PyDdzEnv()
    observation = environment.reset(800 + history_events, rules)
    for _ in range(history_events):
        environment.step(environment.legal_actions()[0])
        observation = environment.observe(environment.current_player)
    actions = tuple(environment.legal_actions())
    short = encode_bid_batch(
        (observation,), (actions,), rules, history_max_length=max(3, history_events)
    )
    long = encode_bid_batch((observation,), (actions,), rules, history_max_length=6)
    model = BidHead(_small_config()).eval()
    with torch.inference_mode():
        short_output = model(short)
        long_output = model(long)
    torch.testing.assert_close(short_output.state, long_output.state, rtol=0.0, atol=0.0)
    torch.testing.assert_close(short_output.mc_q, long_output.mc_q, rtol=0.0, atol=0.0)
