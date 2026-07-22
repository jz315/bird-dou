"""Full-state critic, COMA, specialization, rollout, and gate tests for M8."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.belief.data import extract_hidden_assignment
from birddou.env_types import Action, Observation
from birddou.eval.bootstrap import BootstrapCI
from birddou.eval.metrics import ArenaReport, PairedEstimate, RoleReport
from birddou.eval.paired_deals import generate_paired_deals
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.action_encoder import ActionEncoderConfig
from birddou.models.bird_dou import BirdDouConfig, BirdDouModel
from birddou.models.farmer_team_critic import (
    FARMER_TEAM_CRITIC_ARCHITECTURE,
    FarmerTeamCritic,
    FarmerTeamCriticConfig,
    load_farmer_team_critic_config,
)
from birddou.models.history_encoder import HistoryEncoderConfig
from birddou.models.privileged_teacher import (
    PRIVILEGED_TEACHER_ARCHITECTURE,
    PrivilegedTeacherConfig,
)
from birddou.models.rank_mixer import RankMixerConfig
from birddou.models.segment_ops import segment_softmax
from birddou.rl.farmer_coordination import (
    CounterfactualRolloutBatch,
    FarmerAcceptanceThresholds,
    FarmerCoordinationConfig,
    FarmerExploiterSpec,
    FarmerSpecialistOptimizer,
    counterfactual_advantage,
    evaluate_farmer_acceptance,
    farmer_coordination_loss,
    generate_counterfactual_rollouts,
    generate_farmer_exploiter_schedule,
    load_farmer_coordination_config,
    select_high_value_farmer_states,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"


def _tiny_base_config() -> BirdDouConfig:
    width = 16
    return BirdDouConfig(
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


def _tiny_critic_config() -> FarmerTeamCriticConfig:
    base = _tiny_base_config()
    teacher = PrivilegedTeacherConfig(
        schema_version=1,
        architecture=PRIVILEGED_TEACHER_ARCHITECTURE,
        feature_schema_version=1,
        base=base,
        transformer_layers=1,
        attention_heads=2,
        feedforward_multiplier=1,
        count_embedding_dim=2,
        dropout=0.0,
        oracle_dropout=0.0,
    )
    return FarmerTeamCriticConfig(
        schema_version=1,
        architecture=FARMER_TEAM_CRITIC_ARCHITECTURE,
        feature_schema_version=1,
        teacher=teacher,
        hidden_multiplier=1,
        dropout=0.0,
    )


def _decision_batch(seats: tuple[int, ...]) -> tuple[RaggedBatch, torch.Tensor]:
    rules = load_rule_config(RULES_PATH)
    observations = []
    actions = []
    labels = []
    for row, target_seat in enumerate(seats):
        environment = PyDdzEnv()
        environment.reset(8_100 + row, rules)
        while environment.current_player != target_seat:
            environment.step(environment.legal_actions()[0])
        observation = environment.observe(environment.current_player)
        observations.append(observation)
        actions.append(environment.legal_actions())
        labels.append(extract_hidden_assignment(environment.serialize(), observation))
    return (
        encode_ragged_batch(
            tuple(observations),
            tuple(actions),
            rules,
            config=FeatureConfig(decomposition_features=False),
        ),
        torch.tensor(labels, dtype=torch.int64),
    )


def test_full_state_team_critic_is_farmer_only_and_has_finite_gradients() -> None:
    torch.manual_seed(8_001)
    batch, labels = _decision_batch((1, 2))
    critic = FarmerTeamCritic(_tiny_critic_config())
    output = critic(batch, labels)
    assert output.team_q.shape == (batch.action_count,)
    assert output.state_seat.tolist() == [1, 2]
    assert set(output.action_seat.tolist()) == {1, 2}
    loss = output.team_q.square().mean()
    torch.autograd.backward((loss,))
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in critic.team_head.parameters()
    )

    landlord_batch, landlord_labels = _decision_batch((0,))
    try:
        critic(landlord_batch, landlord_labels)
    except ValueError as error:
        assert "farmer" in str(error)
    else:
        raise AssertionError("centralized farmer Critic accepted a landlord state")


def test_counterfactual_baseline_loss_and_sparse_rollout_supervision() -> None:
    team_q = torch.tensor([1.0, 3.0, 2.0, 4.0], requires_grad=True)
    actor_logits = torch.tensor([0.0, 1.0, -1.0, 1.0], requires_grad=True)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    probability = segment_softmax(actor_logits, offsets)
    log_probability = probability.log()
    chosen = torch.tensor([0, 3], dtype=torch.int64)
    counterfactual = counterfactual_advantage(team_q, probability, offsets, chosen)
    expected_baseline = torch.stack(
        (
            probability[0] * team_q[0] + probability[1] * team_q[1],
            probability[2] * team_q[2] + probability[3] * team_q[3],
        )
    )
    torch.testing.assert_close(counterfactual.baseline, expected_baseline)
    torch.testing.assert_close(
        counterfactual.advantage,
        torch.stack((team_q[0], team_q[3])) - expected_baseline,
    )

    output = farmer_coordination_loss(
        FarmerCoordinationConfig(),
        team_q=team_q,
        actor_probability=probability,
        actor_log_probability=log_probability,
        action_offsets=offsets,
        chosen_action_flat_index=chosen,
        terminal_team_target=torch.tensor([-1.0, 1.0]),
        rollout_action_flat_index=torch.tensor([1, 2], dtype=torch.int64),
        rollout_team_target=torch.tensor([1.0, -1.0]),
    )
    torch.autograd.backward((output.total,))
    assert team_q.grad is not None and torch.isfinite(team_q.grad).all()
    assert actor_logits.grad is not None and torch.isfinite(actor_logits.grad).all()
    assert output.rollout.item() > 0.0


def test_farmer_specialist_update_keeps_landlord_output_bit_exact() -> None:
    torch.manual_seed(8_002)
    model = BirdDouModel(_tiny_base_config()).eval()
    landlord_batch, _ = _decision_batch((0,))
    farmer_batch, _ = _decision_batch((1, 2))
    before = model(landlord_batch).policy_logit.detach().clone()
    farmer_parameter = dict(model.named_parameters())["role_adapter.adapters.1.down.weight"]
    farmer_two_parameter = dict(model.named_parameters())["role_adapter.adapters.2.down.weight"]
    farmer_weight = farmer_parameter.detach().clone()
    farmer_two_weight = farmer_two_parameter.detach().clone()
    config = replace(FarmerCoordinationConfig(), specialist_learning_rate=1e-2)
    optimizer = FarmerSpecialistOptimizer(model, config)
    assert not next(model.rank_mixer.parameters()).requires_grad
    optimizer.zero_grad()
    farmer_loss = model(farmer_batch).policy_logit.square().mean()
    torch.autograd.backward((farmer_loss,))
    optimizer.step()
    after = model(landlord_batch).policy_logit.detach()
    assert torch.equal(before, after)
    assert not torch.equal(farmer_weight, farmer_parameter)
    assert not torch.equal(farmer_two_weight, farmer_two_parameter)
    assert all("privileged" not in key for key in model.state_dict())

    unsafe_model = BirdDouModel(_tiny_base_config()).eval()
    unsafe_before = unsafe_model(landlord_batch).policy_logit.detach().clone()
    unsafe_optimizer = FarmerSpecialistOptimizer(unsafe_model, config)
    unsafe_optimizer.zero_grad()
    torch.autograd.backward((unsafe_model(landlord_batch).policy_logit.mean(),))
    with pytest.raises(RuntimeError, match="landlord gradient"):
        unsafe_optimizer.step()
    assert torch.equal(unsafe_before, unsafe_model(landlord_batch).policy_logit)


class _SeededContinuation:
    def select_action(
        self,
        observation: Observation,
        legal_actions: tuple[Action, ...],
        seed: int,
    ) -> int:
        del observation
        return seed % len(legal_actions)


def test_top_n_counterfactual_rollout_restores_native_state_deterministically() -> None:
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    environment.reset(8_003, rules)
    environment.step(environment.legal_actions()[0])
    assert environment.current_player == 1
    snapshot = environment.serialize()
    scores = tuple(float(index) for index in range(len(environment.legal_actions())))
    config = replace(FarmerCoordinationConfig(), rollout_top_n=2)
    policy = _SeededContinuation()
    first = generate_counterfactual_rollouts(
        snapshot,
        rules,
        scores,
        policy,
        config,
        seed=88,
    )
    second = generate_counterfactual_rollouts(
        snapshot,
        rules,
        scores,
        policy,
        config,
        seed=88,
    )
    assert first == second
    assert isinstance(first, CounterfactualRolloutBatch)
    assert len(first.targets) == 2
    assert tuple(target.action_index for target in first.targets) == (
        len(scores) - 1,
        len(scores) - 2,
    )
    assert environment.serialize() == snapshot
    assert all(target.rollout_actions > 0 for target in first.targets)

    selected = select_high_value_farmer_states(
        torch.tensor([1.0, 3.0, 3.0, 2.0]),
        torch.tensor([1, 2, 1, 2]),
        replace(config, max_rollout_states_per_batch=2),
    )
    assert selected.tolist() == [1, 2]


def test_farmer_config_exploiter_schedule_and_empirical_gate_are_explicit() -> None:
    config = load_farmer_coordination_config(
        REPOSITORY_ROOT / "configs" / "train" / "farmer_coordination.yaml"
    )
    assert config == FarmerCoordinationConfig()
    with pytest.raises(ValueError, match="forbidden"):
        FarmerCoordinationConfig(handcrafted_cooperation_rewards=True)
    model_config = load_farmer_team_critic_config(
        REPOSITORY_ROOT / "configs" / "model" / "farmer_team_critic_v1.yaml"
    )
    assert model_config.architecture == FARMER_TEAM_CRITIC_ARCHITECTURE

    deals = generate_paired_deals(8_004, 2)
    schedule = generate_farmer_exploiter_schedule(
        deals,
        FarmerExploiterSpec("strong-landlord", "farmer-champion", "farmer-exploiter"),
    )
    assert len(schedule) == 4
    assert all(match.assignment.policy_ids[0] == "strong-landlord" for match in schedule)
    assert all(
        match.assignment.policy_ids[1] == match.assignment.policy_ids[2] for match in schedule
    )

    report = _acceptance_arena_report()
    accepted = evaluate_farmer_acceptance(
        report,
        landlord_parameters_unchanged=True,
        thresholds=FarmerAcceptanceThresholds(
            minimum_team_win_delta=0.0,
            maximum_seat_win_regression=0.02,
            require_team_ci_above_threshold=True,
        ),
    )
    assert accepted.passed
    rejected = evaluate_farmer_acceptance(report, landlord_parameters_unchanged=False)
    assert not rejected.passed
    assert "landlord execution parameters changed" in rejected.failures


def _estimate(delta: float, lower: float) -> PairedEstimate:
    interval = BootstrapCI(1, 10, delta, 0.01, lower, delta + 0.02, 0.95, 100, 1)
    return PairedEstimate(10, 0.5 + delta, 0.5, delta, interval)


def _role(name: str, delta: float, lower: float) -> RoleReport:
    estimate = _estimate(delta, lower)
    return RoleReport(name, 10, estimate, estimate, estimate)


def _acceptance_arena_report() -> ArenaReport:
    deals = generate_paired_deals(9, 1)
    landlord = _role("landlord", 0.0, -0.01)
    down = _role("landlord_down", -0.01, -0.02)
    up = _role("landlord_up", 0.01, 0.0)
    team = _role("farmer_team", 0.03, 0.01)
    overall = _role("overall", 0.01, 0.0)
    return ArenaReport(
        1,
        "rules",
        "candidate",
        "baseline",
        deals,
        3,
        6,
        landlord,
        down,
        up,
        team,
        overall,
    )
