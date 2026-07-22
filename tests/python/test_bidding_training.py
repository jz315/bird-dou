"""Monte Carlo initialization, joint loss, curriculum, and complete-Arena tests."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from birddou import Action, Observation, PyDdzEnv, RuleConfig, load_rule_config
from birddou.eval.arena import Arena
from birddou.eval.baselines import (
    FirstLegalPolicy,
    FixedBidPolicy,
    LongestMovePolicy,
    PolicyDecisionContext,
)
from birddou.eval.paired_deals import ScheduledMatch, SeatAssignment, generate_paired_deals
from birddou.models.bid_head import BidHead, BidHeadConfig, encode_bid_batch
from birddou.models.segment_ops import segment_softmax, segment_sum
from birddou.rl.bidding import (
    BiddingAcceptanceThresholds,
    BiddingDistributionMonitor,
    BiddingEpisodeSummary,
    BidHeadPolicy,
    BidLossConfig,
    CompleteDealSample,
    CurriculumMetrics,
    CurriculumThresholds,
    MonteCarloConfig,
    WinScoreCurriculum,
    bid_supervised_loss,
    binary_calibration_error,
    build_joint_bid_batch,
    collect_complete_episode,
    combine_joint_training_loss,
    evaluate_bidding_acceptance,
    generate_initial_bid_mc_labels,
    joint_bid_loss,
    load_bidding_training_config,
    sample_initial_bid_deals,
    set_cardplay_frozen,
)

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs" / "rules" / "canonical_full.yaml"
TRAIN_PATH = ROOT / "configs" / "train" / "bidding.yaml"


class _RecordingContinuation:
    policy_id = "recording-mc-continuation"

    def __init__(self, inner: FixedBidPolicy) -> None:
        self.inner = inner
        self.phases: list[str] = []

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        self.phases.append(observation["phase"])
        return self.inner.select_action(observation, legal_actions, context)


def _rules() -> RuleConfig:
    return load_rule_config(RULES_PATH)


def _counts(cards: list[int]) -> tuple[int, ...]:
    counts = [0] * 15
    for rank in cards:
        counts[rank] += 1
    return tuple(counts)


def _sample_deals() -> tuple[CompleteDealSample, CompleteDealSample]:
    deck = [rank for rank in range(13) for _ in range(4)] + [13, 14]
    own = deck[:17]
    unknown = deck[17:]
    samples = []
    for sample_id, rotation in enumerate((0, 5)):
        rotated = unknown[rotation:] + unknown[:rotation]
        hands = (_counts(own), _counts(rotated[:17]), _counts(rotated[17:34]))
        samples.append(CompleteDealSample(hands, _counts(rotated[34:]), 0, sample_id))
    return samples[0], samples[1]


def _small_model() -> BidHead:
    config = BidHeadConfig(2, "bird_dou_bid_head_v2", 32, 1, 1, 4, 2, 3, 0.0)
    return BidHead(config)


def test_initial_bid_sampler_preserves_information_set_and_card_conservation() -> None:
    samples = sample_initial_bid_deals(611, _rules(), 4)
    bidder = samples[0].first_bidder
    own = samples[0].hands[bidder]

    assert len(samples) == 4
    assert all(sample.first_bidder == bidder for sample in samples)
    assert all(sample.hands[bidder] == own for sample in samples)
    assert len({(sample.hands, sample.bottom_cards) for sample in samples}) == 4
    expected = (4,) * 13 + (1, 1)
    for sample in samples:
        total = tuple(
            sum((*sample.hands, sample.bottom_cards)[row][rank] for row in range(4))
            for rank in range(15)
        )
        assert total == expected


def test_mc_initialization_branches_every_bid_and_fits_outcome_heads() -> None:
    rules = _rules()
    samples = _sample_deals()
    continuation = _RecordingContinuation(
        FixedBidPolicy(
            "fixed-mc-continuation",
            LongestMovePolicy("frozen-cardplay"),
            score_bid=1,
            double=False,
        )
    )
    labels = generate_initial_bid_mc_labels(
        samples, rules, continuation, MonteCarloConfig(1000, 0.5)
    )
    environment_hand = [list(hand) for hand in samples[0].hands]
    environment = PyDdzEnv()
    observation = environment.reset_complete_deal(
        environment_hand, list(samples[0].bottom_cards), 0, rules
    )
    actions = tuple(environment.legal_actions())
    batch = encode_bid_batch((observation,), (actions,), rules)
    output = _small_model()(batch)
    config = load_bidding_training_config(TRAIN_PATH).loss
    loss = bid_supervised_loss(output, labels, batch.action_offsets, config)
    (gradient,) = torch.autograd.grad(loss.total, output.mc_q, retain_graph=True)

    assert labels.sample_count.tolist() == [2, 2, 2, 2]
    assert torch.all((labels.win_target >= 0.0) & (labels.win_target <= 1.0))
    assert torch.isfinite(labels.score_target).all()
    assert {"bidding", "doubling", "card_play"} <= set(continuation.phases)
    assert torch.isfinite(loss.total)
    assert torch.isfinite(gradient).all()

    chosen = torch.tensor([0], dtype=torch.int64)
    joint = joint_bid_loss(
        output,
        chosen,
        torch.tensor([1.0]),
        torch.tensor([4.0]),
        batch.action_offsets,
        config,
    )
    assert torch.isfinite(joint.total)


def test_curriculum_monitor_calibration_and_acceptance_are_metric_gated() -> None:
    threshold = CurriculumThresholds(0.1, 0.1, 0.95, 0.35, 10)
    curriculum = WinScoreCurriculum(
        threshold,
        score_loss_coef=0.25,
        utility_score_coef=0.5,
    )
    assert curriculum.state.cardplay_frozen
    assert not curriculum.maybe_advance(CurriculumMetrics(9, 0.01, 0.5, 0.1))
    metrics = CurriculumMetrics(10, 0.05, 0.5, 0.1)
    assert curriculum.maybe_advance(metrics)
    assert not curriculum.state.cardplay_frozen
    assert curriculum.state.score_loss_coef == 0.0
    assert curriculum.state.utility_score_coef == 0.0
    assert curriculum.maybe_advance(metrics)
    assert curriculum.state.score_loss_coef == pytest.approx(0.25)
    assert curriculum.state.utility_score_coef == pytest.approx(0.5)

    monitor = BiddingDistributionMonitor(20)
    for index in range(12):
        monitor.add(
            BiddingEpisodeSummary(
                landlord_strength=float(index % 4),
                winning_bid=index % 3 + 1,
                redeal_count=index % 2,
                bid_action_count=3,
                positive_bid_count=1,
                landlord_won=index % 2 == 0,
                landlord_score=2.0 if index % 2 == 0 else -2.0,
            )
        )
    report = monitor.report(0.1, 0.95)
    assert not report.degenerate
    assert report.bid_ratio == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert binary_calibration_error((0.1, 0.9), (False, True), bins=2) == pytest.approx(0.1)
    gates = BiddingAcceptanceThresholds(0.1, 0.1, 0.95, 0.35, 0.2, 2.0)
    accepted = evaluate_bidding_acceptance(report, report, 0.05, 0.01, gates)
    rejected = evaluate_bidding_acceptance(report, report, 0.05, 0.0, gates)
    assert accepted.accepted
    assert not rejected.accepted and not rejected.beats_fixed_bidder


def test_rob_mode_monitor_does_not_apply_score_bucket_degeneracy() -> None:
    monitor = BiddingDistributionMonitor(20)
    for index in range(10):
        monitor.add(
            BiddingEpisodeSummary(
                landlord_strength=0.5,
                winning_bid=1,
                redeal_count=0,
                bid_action_count=3,
                positive_bid_count=1 + index % 2,
                landlord_won=index % 2 == 0,
                landlord_score=2.0 if index % 2 == 0 else -2.0,
                bidding_mode="rob",
                call_count=1,
                rob_count=index % 2,
                landlord_change_count=index % 2,
            )
        )
    report = monitor.report(0.1, 0.95)
    assert report.bidding_mode == "rob"
    assert report.bid_ratio == (1.0, 0.0, 0.0)
    assert not report.degenerate
    assert report.call_action_rate == pytest.approx(1 / 3)
    assert report.rob_action_rate == pytest.approx(1 / 6)
    assert report.mean_rob_count == pytest.approx(0.5)
    assert report.mean_landlord_changes == pytest.approx(0.5)


def test_complete_scoring_arena_runs_fixed_bidder_cardplay_composition() -> None:
    rules = _rules()
    policy = FixedBidPolicy(
        policy_id="fixed-complete",
        cardplay=LongestMovePolicy("inner-cardplay"),
        score_bid=1,
        double=False,
    )
    arena = Arena(rules, (policy,))
    deal = generate_paired_deals(20260722, 1).deals[0]
    result = arena.play_match(
        ScheduledMatch(
            "complete-score",
            deal,
            SeatAssignment((policy.policy_id, policy.policy_id, policy.policy_id)),
        )
    )

    assert result.landlord_seat in (0, 1, 2)
    assert result.final_deal_seed == result.deal_seed
    assert result.redeal_count == 0
    assert result.bidding_record_json != "[]"
    assert sum(result.raw_payoff) == 0
    assert any(abs(value) >= 2 for value in result.raw_payoff)


def test_complete_collector_attaches_one_terminal_return_to_both_stages() -> None:
    rules = _rules()
    fixed = FixedBidPolicy("bid", FirstLegalPolicy("unused"), score_bid=1)
    episode = collect_complete_episode(
        81,
        rules,
        fixed,
        LongestMovePolicy("cardplay"),
    )
    joint_batch = build_joint_bid_batch(episode, rules)
    model = _small_model()
    output = model(joint_batch.batch)
    config = load_bidding_training_config(TRAIN_PATH)
    bid_loss = joint_bid_loss(
        output,
        joint_batch.chosen_flat_index,
        joint_batch.terminal_win,
        joint_batch.terminal_score,
        joint_batch.batch.action_offsets,
        config.loss,
    )
    curriculum = WinScoreCurriculum(
        config.curriculum,
        config.loss.score_loss_coef,
        config.loss.utility_score_coef,
    )
    cardplay_loss = torch.tensor(2.0, requires_grad=True)
    frozen = combine_joint_training_loss(bid_loss, cardplay_loss, curriculum.state)
    set_cardplay_frozen(model, curriculum.state.cardplay_frozen)

    assert episode.bidding and episode.cardplay
    assert episode.landlord is not None and episode.winning_bid == 1
    assert joint_batch.terminal_score.shape == (len(episode.bidding),)
    assert not frozen.cardplay_included
    assert all(not parameter.requires_grad for parameter in model.parameters())

    all_pass = collect_complete_episode(
        82,
        rules,
        FirstLegalPolicy("always-pass"),
        FirstLegalPolicy("never-used"),
    )
    assert all_pass.all_pass and all_pass.terminal_payoff == (0, 0, 0)


def test_bid_head_policy_adapter_uses_only_bidding_boundary() -> None:
    rules = _rules()
    environment = PyDdzEnv()
    observation = environment.reset(99, rules)
    actions = tuple(environment.legal_actions())
    model = _small_model()
    policy = BidHeadPolicy("learned-bid", model, rules)
    model.train()
    selected = policy.select_action(
        observation,
        actions,
        PolicyDecisionContext(0, 99, "bid-policy", observation["observer"], None, 0),
    )
    assert 0 <= selected < len(actions)
    assert not model.training

    for parameter in model.outcome_head[-1].parameters():
        parameter.data[:] = torch.nan
    with pytest.raises(RuntimeError, match="non-finite mc_q"):
        policy.select_action(
            observation,
            actions,
            PolicyDecisionContext(0, 99, "bad-bid-policy", observation["observer"], None, 0),
        )


def test_joint_bid_loss_is_selected_action_dmc_with_per_state_entropy() -> None:
    rules = _rules()
    first = PyDdzEnv()
    second = PyDdzEnv()
    first_observation = first.reset(101, rules)
    second.reset(102, rules)
    second.step(second.legal_actions()[0])
    observations = (first_observation, second.observe(second.current_player))
    legal = (tuple(first.legal_actions()), tuple(second.legal_actions()))
    batch = encode_bid_batch(observations, legal, rules)
    output = _small_model()(batch)
    config = replace(load_bidding_training_config(TRAIN_PATH).loss, entropy_coef=0.0)
    chosen = batch.action_offsets[:-1]
    targets_win = torch.tensor([1.0, 0.0])
    targets_score = torch.tensor([2.0, -2.0])
    loss = joint_bid_loss(
        output,
        chosen,
        targets_win,
        targets_score,
        batch.action_offsets,
        config,
    )
    (gradient,) = torch.autograd.grad(loss.total, output.mc_q, retain_graph=True)
    non_chosen = torch.ones_like(gradient, dtype=torch.bool)
    non_chosen[chosen] = False
    assert torch.count_nonzero(gradient[non_chosen]) == 0
    assert torch.count_nonzero(gradient[chosen]) == chosen.numel()

    entropy_config = replace(config, entropy_coef=0.001)
    entropy_loss = joint_bid_loss(
        output,
        chosen,
        targets_win,
        targets_score,
        batch.action_offsets,
        entropy_config,
    )
    probability = segment_softmax(
        output.mc_q.float() / entropy_config.policy_temperature,
        batch.action_offsets,
    )
    expected_entropy = -segment_sum(
        probability * probability.clamp_min(torch.finfo(torch.float32).tiny).log(),
        batch.action_offsets,
    ).mean()
    torch.testing.assert_close(entropy_loss.entropy, expected_entropy)


def test_training_config_rejects_invalid_loss_switches() -> None:
    config = load_bidding_training_config(TRAIN_PATH)
    assert config.schema_version == 3
    assert config.collection_epsilon == pytest.approx(0.05)
    assert config.loss.entropy_coef == 0.0
    with pytest.raises(ValueError, match="weights"):
        BidLossConfig(-1.0, 1.0, 0.0, 0.0, 1.0, 16.0, 0.0)
