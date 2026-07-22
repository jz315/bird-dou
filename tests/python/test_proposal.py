"""Cheap Proposal network, permanent protection, dynamic Top-K, and gate tests."""

from pathlib import Path

import pytest
import torch

from birddou import PyDdzEnv, RuleConfig, load_rule_config
from birddou.features.ragged import (
    ACTION_META_COLUMNS,
    DECOMPOSITION_DISABLED_GROUPS,
    FeatureConfig,
    RaggedBatch,
    encode_ragged_batch,
)
from birddou.models.proposal import (
    ProposalGateThresholds,
    ProposalNetwork,
    ProposalValidationMetrics,
    evaluate_proposal_gate,
    load_proposal_config,
    proposal_protected_mask,
    select_proposals,
    should_use_full_action_set,
    subset_ragged_batch,
)

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
CONFIG_PATH = ROOT / "configs" / "model" / "proposal_v1.yaml"


def _rules() -> RuleConfig:
    return load_rule_config(RULES_PATH)


def _batch() -> RaggedBatch:
    rules = _rules()
    environments = [PyDdzEnv(), PyDdzEnv()]
    observations = []
    actions = []
    for index, environment in enumerate(environments):
        environment.reset(70 + index, rules)
        environment.step(environment.legal_actions()[-1])
        observations.append(environment.observe(environment.current_player))
        actions.append(tuple(environment.legal_actions()))
    return encode_ragged_batch(
        observations,
        actions,
        rules,
        config=FeatureConfig(decomposition_features=True),
    )


def test_proposal_is_ragged_cheap_and_protection_survives_low_scores() -> None:
    batch = _batch()
    minimum_groups = batch.action_meta[:, ACTION_META_COLUMNS.index("min_groups_after")]
    assert torch.all(minimum_groups != DECOMPOSITION_DISABLED_GROUPS)
    config = load_proposal_config(CONFIG_PATH)
    output = ProposalNetwork(config).eval()(batch)
    blockers = torch.zeros(batch.action_count, dtype=torch.bool)
    teachers = torch.zeros_like(blockers)
    blockers[int(batch.action_offsets[0])] = True
    teachers[int(batch.action_offsets[1])] = True
    exploration = batch.action_offsets[:-1].clone()
    protected = proposal_protected_mask(
        batch,
        blocks_immediate_loss=blockers,
        teacher_high_value=teachers,
        exploration_flat_index=exploration,
    )
    scores = output.score.detach().clone()
    scores[protected] = -1_000.0
    selection = select_proposals(
        batch,
        scores,
        config,
        torch.tensor([0.0, 1.0]),
        protected_mask=protected,
    )

    assert output.score.shape == (batch.action_count,)
    assert torch.all(selection.selected_mask[protected])
    assert selection.dynamic_k[0] <= selection.dynamic_k[1]
    assert selection.selected_offsets[-1].item() == selection.selected_flat_index.numel()
    assert torch.equal(
        selection.selected_flat_index,
        torch.sort(selection.selected_flat_index).values,
    )
    subset = subset_ragged_batch(batch, selection)
    assert subset.action_count == selection.selected_flat_index.numel()
    assert subset.action_count <= batch.action_count
    assert torch.equal(subset.action_offsets, selection.selected_offsets)
    for name in ("is_pass", "is_bomb", "is_rocket", "empties_hand"):
        required = batch.action_meta[:, ACTION_META_COLUMNS.index(name)].bool()
        assert torch.all(selection.selected_mask[required])


def test_full_action_controls_are_seeded_and_hard_pruning_gate_is_strict() -> None:
    config = load_proposal_config(CONFIG_PATH)
    first = [should_use_full_action_set(step, 9, config) for step in range(100)]
    second = [should_use_full_action_set(step, 9, config) for step in range(100)]
    assert first == second
    assert 0 < sum(first) < len(first)

    thresholds = ProposalGateThresholds(0.995, 1.2, 0.0)
    accepted = evaluate_proposal_gate(
        ProposalValidationMetrics(0.999, 1.0, 1.0, 1.5, 0.01, 0.1), thresholds
    )
    rejected = evaluate_proposal_gate(
        ProposalValidationMetrics(0.99, 1.0, 0.99, 1.1, -0.01, 0.0), thresholds
    )
    assert accepted.accepted
    assert not rejected.accepted
    assert len(rejected.reasons) == 5


def test_proposal_config_has_stable_identity_and_rejects_no_speedup_gate() -> None:
    config = load_proposal_config(CONFIG_PATH)
    assert config.fingerprint() == load_proposal_config(CONFIG_PATH).fingerprint()
    with pytest.raises(ValueError, match="exceed"):
        ProposalGateThresholds(0.99, 1.0)
    with pytest.raises(ValueError, match="finite"):
        ProposalValidationMetrics(float("nan"), 1.0, 1.0, 2.0, 0.0, 0.1)
