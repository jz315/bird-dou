"""Numerical and configuration tests for M7 V-trace and Hybrid training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from birddou.rl.hybrid import (
    HybridLossConfig,
    TrainerMode,
    blend_win_score_reward,
    hybrid_loss,
    load_hybrid_loss_config,
    score_train_reward,
)
from birddou.rl.vtrace import (
    PolicyLagMonitor,
    VTraceConfig,
    VTraceReturns,
    load_vtrace_config,
    vtrace_from_log_probabilities,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_on_policy_terminal_vtrace_equals_undiscounted_return() -> None:
    """Gamma-one on-policy targets match the exact terminal Monte Carlo return."""
    zeros = torch.zeros(3, 2)
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    done = torch.tensor([[False, True], [False, False], [True, True]])
    result = vtrace_from_log_probabilities(
        zeros,
        zeros,
        rewards,
        zeros,
        torch.zeros(2),
        done,
    )
    expected = torch.tensor([[1.0, 1.0], [1.0, -1.0], [1.0, -1.0]])
    torch.testing.assert_close(result.value_targets, expected)
    torch.testing.assert_close(result.policy_advantages, expected)
    torch.testing.assert_close(result.importance_weights, torch.ones_like(zeros))


def test_vtrace_matches_manual_off_policy_recursion_and_is_stable() -> None:
    """Clipped recursion matches a hand oracle and extreme ratios stay finite."""
    behavior = torch.log(torch.tensor([0.5, 0.25]))
    target = torch.log(torch.tensor([0.25, 1.0]))
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    result = vtrace_from_log_probabilities(
        behavior,
        target,
        rewards,
        values,
        torch.tensor(0.0),
        torch.tensor([False, True]),
    )
    # rho=[0.5,1], c=[0.5,1], deltas=[0.75,1], corrections=[1.25,1].
    torch.testing.assert_close(result.value_targets, torch.tensor([1.75, 2.0]))
    torch.testing.assert_close(result.policy_advantages, torch.tensor([1.25, 1.0]))

    extreme_target = torch.tensor([1_000.0, -1_000.0], requires_grad=True)
    stable = vtrace_from_log_probabilities(
        torch.zeros(2),
        extreme_target,
        torch.tensor([0.0, 1.0]),
        torch.zeros(2, requires_grad=True),
        torch.tensor(0.0),
        torch.tensor([False, True]),
    )
    loss = stable.value_targets.sum() + stable.policy_advantages.sum()
    torch.autograd.backward((loss,))
    assert all(
        torch.isfinite(value).all()
        for value in (
            stable.value_targets,
            stable.policy_advantages,
            stable.importance_weights,
            extreme_target.grad,
        )
        if value is not None
    )


def test_vtrace_config_and_bounded_policy_lag_monitor() -> None:
    config = load_vtrace_config(REPOSITORY_ROOT / "configs" / "train" / "vtrace.yaml")
    assert config == VTraceConfig()
    with pytest.raises(ValueError, match="gamma"):
        VTraceConfig(gamma=1.1)

    monitor = PolicyLagMonitor(stale_after=1, capacity=3)
    monitor.observe(10, torch.tensor([10, 9]), torch.tensor([1.0, 0.5]))
    monitor.observe(10, torch.tensor([8, 7]), torch.tensor([0.25, 2.0]))
    stats = monitor.stats()
    assert stats.sample_count == 3
    assert stats.maximum_lag == 3
    assert stats.stale_fraction == pytest.approx(2.0 / 3.0)
    assert monitor.importance_weight_range == (0.25, 2.0)
    with pytest.raises(ValueError, match="newer"):
        monitor.observe(2, torch.tensor([3]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="positive"):
        monitor.observe(2, torch.tensor([2]), torch.tensor([0.0]))


def _loss_inputs() -> dict[str, torch.Tensor]:
    return {
        "chosen_log_probability": torch.tensor([-0.2, -0.4], requires_grad=True),
        "entropy": torch.tensor([0.5, 0.25], requires_grad=True),
        "value_prediction": torch.tensor([0.1, -0.1], requires_grad=True),
        "mc_q_prediction": torch.tensor([0.2, -0.2], requires_grad=True),
        "terminal_target": torch.tensor([1.0, -1.0]),
        "win_logit": torch.tensor([0.3, -0.3], requires_grad=True),
        "win_target": torch.tensor([1.0, 0.0]),
        "score_prediction": torch.tensor([0.4, -0.4], requires_grad=True),
        "score_target": torch.tensor([2.0, -2.0]),
    }


def _vtrace_for_loss() -> VTraceReturns:
    zeros = torch.zeros(2)
    return vtrace_from_log_probabilities(
        zeros,
        zeros,
        torch.tensor([0.0, 1.0]),
        zeros,
        torch.tensor(0.0),
        torch.tensor([False, True]),
    )


def test_hybrid_loss_modes_coefficients_and_gradients_are_independent() -> None:
    inputs = _loss_inputs()
    vtrace = _vtrace_for_loss()
    belief = torch.tensor(0.7, requires_grad=True)
    kd = torch.tensor(0.8, requires_grad=True)
    auxiliary = torch.tensor(0.9, requires_grad=True)
    config = HybridLossConfig(kd_coef=0.3)
    output = hybrid_loss(
        config,
        **inputs,
        vtrace=vtrace,
        belief_loss=belief,
        kd_loss=kd,
        auxiliary_loss=auxiliary,
    )
    expected = (
        config.policy_coef * output.policy
        + config.value_coef * output.value
        + config.mc_q_coef * output.mc_q
        + config.win_coef * output.win
        + config.score_coef * output.score
        + config.belief_coef * output.belief
        + config.kd_coef * output.kd
        + config.entropy_coef * output.entropy
        + config.aux_coef * output.auxiliary
    )
    torch.testing.assert_close(output.total, expected)
    torch.autograd.backward((output.total,))
    assert belief.grad is not None and belief.grad.item() == pytest.approx(config.belief_coef)
    assert kd.grad is not None and kd.grad.item() == pytest.approx(config.kd_coef)
    assert auxiliary.grad is not None and auxiliary.grad.item() == pytest.approx(config.aux_coef)

    dmc = hybrid_loss(
        replace(config, mode=TrainerMode.DMC),
        **_loss_inputs(),
        vtrace=vtrace,
    )
    torch.testing.assert_close(
        dmc.total,
        config.mc_q_coef * dmc.mc_q + config.win_coef * dmc.win + config.score_coef * dmc.score,
    )
    vtrace_only = hybrid_loss(
        replace(config, mode=TrainerMode.VTRACE),
        **_loss_inputs(),
        vtrace=vtrace,
    )
    torch.testing.assert_close(
        vtrace_only.total,
        config.policy_coef * vtrace_only.policy
        + config.value_coef * vtrace_only.value
        + config.win_coef * vtrace_only.win
        + config.score_coef * vtrace_only.score
        + config.entropy_coef * vtrace_only.entropy,
    )


def test_hybrid_defaults_and_reward_transform_follow_specification() -> None:
    config = load_hybrid_loss_config(REPOSITORY_ROOT / "configs" / "train" / "hybrid.yaml")
    assert config == HybridLossConfig()
    raw = torch.tensor([-7.0, -1.0, 0.0, 1.0, 7.0])
    torch.testing.assert_close(
        score_train_reward(raw),
        torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]),
    )
    win = torch.tensor([-1.0, 1.0])
    score = torch.tensor([-3.0, 3.0])
    torch.testing.assert_close(blend_win_score_reward(win, score, 0.0), win)
    torch.testing.assert_close(blend_win_score_reward(win, score, 1.0), score_train_reward(score))
