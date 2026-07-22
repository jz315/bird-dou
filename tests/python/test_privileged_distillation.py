"""Full-state Teacher, Oracle Dropout, IS-KD, critic, and leakage tests for M6."""

from __future__ import annotations

from pathlib import Path

import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.belief.data import extract_hidden_assignment
from birddou.belief.sampler import sample_hidden_allocations
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.action_encoder import ActionEncoderConfig
from birddou.models.belief_bird_dou import (
    BELIEF_BIRD_DOU_ARCHITECTURE,
    BeliefBirdDouConfig,
    BeliefBirdDouModel,
    belief_constraints_from_batch,
)
from birddou.models.bird_dou import BirdDouConfig
from birddou.models.history_encoder import HistoryEncoderConfig
from birddou.models.privileged_teacher import (
    PRIVILEGED_TEACHER_ARCHITECTURE,
    PrivilegedCritic,
    PrivilegedTeacher,
    PrivilegedTeacherConfig,
    load_privileged_teacher_config,
)
from birddou.models.rank_mixer import RankMixerConfig
from birddou.models.segment_ops import segment_sum
from birddou.rl.distillation import (
    InformationSetDistillationConfig,
    direct_state_distillation_loss,
    information_set_distillation_loss,
    load_is_kd_config,
    per_state_action_huber_loss,
    privileged_critic_loss,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"
TEACHER_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "model" / "privileged_teacher_v1.yaml"
IS_KD_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "train" / "is_kd.yaml"


def base_config() -> BirdDouConfig:
    width = 8
    return BirdDouConfig(
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


def teacher_config() -> PrivilegedTeacherConfig:
    return PrivilegedTeacherConfig(
        schema_version=1,
        architecture=PRIVILEGED_TEACHER_ARCHITECTURE,
        feature_schema_version=1,
        base=base_config(),
        transformer_layers=1,
        attention_heads=1,
        feedforward_multiplier=1,
        count_embedding_dim=2,
        dropout=0.0,
        oracle_dropout=0.0,
    )


def student_config() -> BeliefBirdDouConfig:
    return BeliefBirdDouConfig(
        schema_version=1,
        architecture=BELIEF_BIRD_DOU_ARCHITECTURE,
        feature_schema_version=1,
        base=base_config(),
        count_embedding_dim=2,
        hidden_multiplier=1,
        dropout=0.0,
        enabled=True,
    )


def state_batch() -> tuple[RaggedBatch, torch.Tensor, torch.Tensor]:
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    observation = environment.reset(6006, rules)
    actions = tuple(environment.legal_actions())
    batch = encode_ragged_batch(
        (observation,),
        (actions,),
        rules,
        chosen_action_indices=(0,),
        config=FeatureConfig(decomposition_features=False),
    )
    true_assignment = torch.tensor(
        [extract_hidden_assignment(environment.serialize(), observation)],
        dtype=torch.int64,
    )
    unknown, capacity_a, _ = belief_constraints_from_batch(batch)
    samples = sample_hidden_allocations(
        torch.zeros(1, 15, 5),
        unknown,
        capacity_a,
        32,
        generator=torch.Generator().manual_seed(6006),
    )
    alternatives = samples[0][torch.any(samples[0] != true_assignment[0], dim=1)]
    assert alternatives.shape[0] > 0
    return batch, true_assignment, alternatives[:1]


def test_teacher_uses_oracle_state_and_dropout_removes_that_dependency() -> None:
    """Full-state changes are visible only while the hidden Oracle tokens are retained."""
    torch.manual_seed(6006)
    batch, true_assignment, alternative = state_batch()
    teacher = PrivilegedTeacher(teacher_config()).eval()
    true_output = teacher(batch, true_assignment, oracle_dropout=0.0)
    alternative_output = teacher(batch, alternative, oracle_dropout=0.0)
    assert true_output.policy.mc_q.shape == (batch.action_count,)
    assert true_output.hand_tokens.shape == (1, 3, 15, 8)
    assert not torch.equal(true_output.full_state, alternative_output.full_state)

    masked_true = teacher(batch, true_assignment, oracle_dropout=1.0)
    masked_alternative = teacher(batch, alternative, oracle_dropout=1.0)
    assert torch.all(masked_true.oracle_keep_mask[:, 0])
    assert not torch.any(masked_true.oracle_keep_mask[:, 1:])
    assert torch.equal(masked_true.full_state, masked_alternative.full_state)
    assert torch.equal(masked_true.policy.mc_q, masked_alternative.policy.mc_q)

    critic = PrivilegedCritic(teacher)
    assert critic(batch, true_assignment).shape == (batch.action_count,)


def test_information_set_kd_averages_legal_samples_and_trains_student_only() -> None:
    """The explicit true-state ablation remains legal and trains Student only."""
    torch.manual_seed(6007)
    batch, true_assignment, _ = state_batch()
    student = BeliefBirdDouModel(student_config())
    teacher = PrivilegedTeacher(teacher_config())
    config = InformationSetDistillationConfig(
        belief_samples_k=3,
        teacher_temperature=0.5,
        value_coefficient=0.5,
        include_true_state=True,
    )
    result = information_set_distillation_loss(
        student,
        teacher,
        batch,
        true_assignment,
        config,
        generator=torch.Generator().manual_seed(6007),
    )
    unknown, capacity_a, _ = belief_constraints_from_batch(batch)

    assert result.hidden_samples_a.shape == (1, 4, 15)
    assert torch.equal(result.hidden_samples_a.sum(dim=-1), capacity_a[:, None].expand(1, 4))
    assert torch.all(result.hidden_samples_a <= unknown[:, None])
    torch.testing.assert_close(
        segment_sum(result.teacher_probability, batch.action_offsets), torch.ones(1)
    )
    assert torch.isfinite(result.loss)
    torch.autograd.backward((result.loss,))
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in student.parameters()
    )
    assert all(parameter.grad is None for parameter in teacher.parameters())

    direct = direct_state_distillation_loss(student, teacher, batch, true_assignment, config)
    assert direct.hidden_samples_a.shape == (1, 1, 15)
    assert not torch.equal(result.q_bar, direct.q_bar)


def test_strict_is_kd_target_is_independent_of_example_true_hidden_state() -> None:
    """Strict IS-KD uses only Belief samples for a fixed public information set."""
    torch.manual_seed(6010)
    batch, true_assignment, alternative = state_batch()
    student = BeliefBirdDouModel(student_config()).eval()
    teacher = PrivilegedTeacher(teacher_config()).eval()
    config = InformationSetDistillationConfig(belief_samples_k=3)
    assert not config.include_true_state

    first = information_set_distillation_loss(
        student,
        teacher,
        batch,
        true_assignment,
        config,
        generator=torch.Generator().manual_seed(44),
    )
    second = information_set_distillation_loss(
        student,
        teacher,
        batch,
        alternative,
        config,
        generator=torch.Generator().manual_seed(44),
    )
    torch.testing.assert_close(first.hidden_samples_a, second.hidden_samples_a)
    torch.testing.assert_close(first.q_bar, second.q_bar)
    torch.testing.assert_close(first.teacher_probability, second.teacher_probability)


def test_kd_value_loss_weights_states_equally_not_actions_equally() -> None:
    prediction = torch.zeros(4)
    target = torch.tensor([1.0, 1.0, 1.0, 3.0])
    offsets = torch.tensor([0, 1, 4], dtype=torch.int64)
    loss = per_state_action_huber_loss(prediction, target, offsets)
    per_action = torch.nn.functional.huber_loss(prediction, target, reduction="none")
    expected = (per_action[:1].mean() + per_action[1:].mean()) / 2.0

    torch.testing.assert_close(loss, expected)
    assert loss != per_action.mean()


def test_privileged_critic_loss_and_student_checkpoint_leakage_boundary() -> None:
    """Centralized supervision backpropagates, while Student weights expose no Oracle module."""
    batch, true_assignment, _ = state_batch()
    teacher = PrivilegedTeacher(teacher_config())
    loss = privileged_critic_loss(
        teacher,
        batch,
        true_assignment,
        torch.ones(batch.batch_size),
    )
    torch.autograd.backward((loss,))
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in teacher.parameters())

    student_keys = tuple(BeliefBirdDouModel(student_config()).state_dict())
    forbidden = ("teacher", "oracle", "full_hand", "true_assignment", "privileged")
    assert not any(any(word in key for word in forbidden) for key in student_keys)


def test_versioned_teacher_and_is_kd_configs_load() -> None:
    teacher = load_privileged_teacher_config(TEACHER_CONFIG_PATH)
    distillation = load_is_kd_config(IS_KD_CONFIG_PATH)
    assert teacher.architecture == PRIVILEGED_TEACHER_ARCHITECTURE
    assert teacher.base.d_model == 256
    assert teacher.transformer_layers == 2
    assert distillation.belief_samples_k == 4
    assert distillation.teacher_temperature == 0.5
    assert distillation.stop_gradient_through_belief_for_kd
    assert not distillation.include_true_state
