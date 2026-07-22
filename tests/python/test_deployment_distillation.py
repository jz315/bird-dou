"""Search distillation, compact actor, export hashes, and observation-only policy tests."""

from pathlib import Path

import pytest
import torch

from birddou import PyDdzEnv, RuleConfig, load_rule_config
from birddou.cli.export_deployment import main as export_deployment_main
from birddou.env_types import Action, Observation
from birddou.eval.baselines import PolicyDecisionContext
from birddou.eval.paired_deals import SeatRole
from birddou.features.ragged import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.deployment import (
    CompactPolicyModel,
    DeploymentPolicy,
    export_deployment_bundle,
    load_compact_policy_config,
    load_deployment_bundle,
)
from birddou.models.segment_ops import segment_softmax
from birddou.rl.search_distillation import (
    SearchDistillationBatch,
    compact_policy_distillation_loss,
    evaluate_distillation_retention,
    load_search_distillation_config,
    search_distillation_loss,
)

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
CONFIG_PATH = ROOT / "configs" / "model" / "compact_policy_v1.yaml"
DISTILLATION_CONFIG_PATH = ROOT / "configs" / "train" / "search_distillation.yaml"


def _rules() -> RuleConfig:
    return load_rule_config(RULES_PATH)


def _decision() -> tuple[Observation, tuple[Action, ...], RaggedBatch]:
    rules = _rules()
    environment = PyDdzEnv()
    observation = environment.reset(117, rules)
    actions = tuple(environment.legal_actions())
    batch = encode_ragged_batch(
        (observation,),
        (actions,),
        rules,
        config=FeatureConfig(decomposition_features=False),
    )
    return observation, actions, batch


def test_search_and_compact_distillation_losses_are_segment_correct() -> None:
    _, _, batch = _decision()
    config = load_compact_policy_config(CONFIG_PATH)
    model = CompactPolicyModel(config)
    student = model(batch)
    teacher_logits = torch.linspace(-1.0, 1.0, batch.action_count)
    teacher_value = torch.tensor([0.75])
    visits = segment_softmax(teacher_logits, batch.action_offsets)
    targets = SearchDistillationBatch(
        batch,
        visits,
        teacher_value,
        torch.tensor([[0.2, 0.3, 0.5]]),
    )
    search_loss = search_distillation_loss(student.policy_logits, student.state_value, targets)
    compact_loss = compact_policy_distillation_loss(student, teacher_logits, teacher_value, batch)
    gradients = torch.autograd.grad(
        compact_loss.total, tuple(model.parameters()), allow_unused=True
    )

    assert torch.isfinite(search_loss.total)
    assert torch.isfinite(compact_loss.total)
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_deployment_bundle_round_trips_hashes_and_policy_needs_only_public_inputs(
    tmp_path: Path,
) -> None:
    observation, actions, batch = _decision()
    rules = _rules()
    model = CompactPolicyModel(load_compact_policy_config(CONFIG_PATH)).eval()
    before = model(batch)
    destination = tmp_path / "compact.pt"
    manifest = export_deployment_bundle(destination, model, rules)
    restored, bid_head, restored_manifest = load_deployment_bundle(destination, rules)
    after = restored.eval()(batch)

    assert manifest == restored_manifest
    assert bid_head is None
    assert manifest.input_contract == "Observation+legal_actions:v1"
    assert torch.equal(before.policy_logits, after.policy_logits)
    assert destination.with_suffix(".pt.manifest.json").is_file()

    policy = DeploymentPolicy(
        "compact",
        restored,
        rules,
        feature_config=FeatureConfig(decomposition_features=False),
    )
    selected = policy.select_action(
        observation,
        actions,
        PolicyDecisionContext(0, 117, "deploy", 0, SeatRole.LANDLORD, 0),
    )
    assert 0 <= selected < len(actions)

    destination.write_bytes(destination.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_deployment_bundle(destination, rules)


def test_deployment_export_cli_accepts_strict_state_dictionary(tmp_path: Path) -> None:
    rules = _rules()
    model = CompactPolicyModel(load_compact_policy_config(CONFIG_PATH))
    weights = tmp_path / "compact-state.pt"
    destination = tmp_path / "service-bundle.pt"
    torch.save(model.state_dict(), weights)

    result = export_deployment_main(
        (
            "--compact-config",
            str(CONFIG_PATH),
            "--compact-weights",
            str(weights),
            "--rules",
            str(RULES_PATH),
            "--output",
            str(destination),
        )
    )

    assert result == 0
    _, bid_head, manifest = load_deployment_bundle(destination, rules)
    assert bid_head is None
    assert manifest.input_contract == "Observation+legal_actions:v1"


def test_compact_retention_gate_requires_positive_paired_evidence() -> None:
    config = load_search_distillation_config(DISTILLATION_CONFIG_PATH)
    accepted = evaluate_distillation_retention(0.1, 0.09, 0.01)
    rejected = evaluate_distillation_retention(0.1, 0.05, 0.0)
    assert accepted.accepted and accepted.retained_fraction == pytest.approx(0.9)
    assert not rejected.accepted and len(rejected.reasons) == 2
    assert config.minimum_retained_fraction == pytest.approx(0.8)
