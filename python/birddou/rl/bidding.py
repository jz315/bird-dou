"""Monte Carlo initialization, staged joint losses, and bidding acceptance gates."""

from __future__ import annotations

import json
import math
import operator
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor

from birddou.env import PyDdzEnv
from birddou.env_types import Action, Observation, RuleConfig, StepResult
from birddou.eval.baselines import Policy, PolicyDecisionContext
from birddou.eval.paired_deals import role_for_game_seat, splitmix64
from birddou.models.bid_head import BidBatch, BidHead, BidHeadOutput, encode_bid_batch
from birddou.models.segment_ops import segment_softmax, segment_sum

BIDDING_TRAINING_SCHEMA_VERSION = 1


class BiddingStage(StrEnum):
    """Ordered curriculum stages; transitions require explicit metric gates."""

    BID_WIN_FROZEN = "bid_win_frozen"
    JOINT_WIN = "joint_win"
    JOINT_SCORE = "joint_score"


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Bounded deterministic rollout controls for Bid Head initialization."""

    max_actions: int
    all_pass_win_target: float

    def __post_init__(self) -> None:
        if self.max_actions <= 0:
            raise ValueError("Monte Carlo max_actions must be positive")
        if not 0.0 <= self.all_pass_win_target <= 1.0:
            raise ValueError("all-pass win target must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class BidLossConfig:
    """Weights for MC supervision and on-policy joint outcome learning."""

    policy_weight: float
    win_weight: float
    score_weight: float
    policy_temperature: float
    score_scale: float
    entropy_weight: float

    def __post_init__(self) -> None:
        weights = (self.policy_weight, self.win_weight, self.score_weight, self.entropy_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Bid loss weights must be finite and non-negative")
        if self.policy_temperature <= 0.0 or self.score_scale <= 0.0:
            raise ValueError("Bid loss temperature and score scale must be positive")


@dataclass(frozen=True, slots=True)
class CurriculumThresholds:
    """Observable gates required before unfreezing Cardplay or enabling score loss."""

    max_calibration_error: float
    min_call_rate: float
    max_call_rate: float
    max_redeal_rate: float
    min_complete_games: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_calibration_error <= 1.0:
            raise ValueError("calibration threshold must be in [0, 1]")
        if not 0.0 <= self.min_call_rate < self.max_call_rate <= 1.0:
            raise ValueError("call-rate gate must be a non-empty interval in [0, 1]")
        if not 0.0 <= self.max_redeal_rate <= 1.0:
            raise ValueError("redeal threshold must be in [0, 1]")
        if self.min_complete_games <= 0:
            raise ValueError("curriculum minimum game count must be positive")


@dataclass(frozen=True, slots=True)
class BiddingTrainingConfig:
    """Versioned M9 training configuration."""

    schema_version: int
    monte_carlo: MonteCarloConfig
    loss: BidLossConfig
    curriculum: CurriculumThresholds

    def __post_init__(self) -> None:
        if self.schema_version != BIDDING_TRAINING_SCHEMA_VERSION:
            raise ValueError("unsupported bidding training schema")


@dataclass(frozen=True, slots=True)
class CompleteDealSample:
    """One privileged, information-set-consistent deal used only for MC labels."""

    hands: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    bottom_cards: tuple[int, ...]
    first_bidder: int
    sample_id: int

    def __post_init__(self) -> None:
        if any(len(hand) != 15 for hand in self.hands) or len(self.bottom_cards) != 15:
            raise ValueError("complete deal rank-count rows must have width 15")
        if any(sum(hand) != 17 for hand in self.hands) or sum(self.bottom_cards) != 3:
            raise ValueError("complete deal must contain three 17-card hands and a 3-card bottom")
        if not 0 <= self.first_bidder < 3 or self.sample_id < 0:
            raise ValueError("first bidder must be a seat and sample_id must be non-negative")


@dataclass(frozen=True, slots=True)
class MonteCarloBidLabels:
    """Per-legal-action final outcome targets under one frozen continuation policy."""

    legal_actions: tuple[Action, ...]
    win_target: Tensor
    score_target: Tensor
    sample_count: Tensor


def sample_initial_bid_deals(
    seed: int,
    rules: RuleConfig,
    sample_count: int,
) -> tuple[CompleteDealSample, ...]:
    """Resample hidden opponents/bottom while preserving one initial information set."""
    if rules["profile"] != "canonical_full":
        raise ValueError("initial bidding samples require canonical_full rules")
    if not 0 <= seed < 1 << 64 or sample_count <= 0:
        raise ValueError("initial bidding sample seed/count is invalid")
    environment = PyDdzEnv()
    observation = environment.reset(seed, rules)
    first_bidder = observation["current_player"]
    envelope = _mapping(json.loads(environment.serialize()), "serialized environment")
    state = _mapping(envelope.get("state"), "serialized game state")
    hands = _rank_rows(state.get("hands"), 3, 17, "hands")
    bottom = _rank_rows([state.get("bottom_cards")], 1, 3, "bottom_cards")[0]
    own_hand = hands[first_bidder]
    hidden_cards: list[int] = []
    for seat, hand in enumerate(hands):
        if seat != first_bidder:
            hidden_cards.extend(_expand_counts(hand))
    hidden_cards.extend(_expand_counts(bottom))
    other_seats = tuple(seat for seat in range(3) if seat != first_bidder)
    samples: list[CompleteDealSample] = []
    for sample_id in range(sample_count):
        cards = list(hidden_cards)
        random.Random(splitmix64((seed + sample_id) & ((1 << 64) - 1))).shuffle(cards)
        sampled_hands = [own_hand, own_hand, own_hand]
        sampled_hands[first_bidder] = own_hand
        sampled_hands[other_seats[0]] = _counts(cards[:17])
        sampled_hands[other_seats[1]] = _counts(cards[17:34])
        samples.append(
            CompleteDealSample(
                cast(
                    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
                    tuple(sampled_hands),
                ),
                _counts(cards[34:]),
                first_bidder,
                sample_id,
            )
        )
    return tuple(samples)

    def __post_init__(self) -> None:
        action_count = len(self.legal_actions)
        if action_count == 0:
            raise ValueError("Monte Carlo labels require legal actions")
        if self.win_target.shape != (action_count,) or not self.win_target.is_floating_point():
            raise ValueError("win targets must be floating [M]")
        if self.score_target.shape != (action_count,) or not self.score_target.is_floating_point():
            raise ValueError("score targets must be floating [M]")
        if self.sample_count.dtype != torch.int64 or self.sample_count.shape != (action_count,):
            raise ValueError("sample counts must be int64 [M]")
        if torch.any(self.sample_count <= 0):
            raise ValueError("every legal bid must have at least one rollout")


@dataclass(frozen=True, slots=True)
class CompleteDecision:
    """One legal information-set decision retained for terminal-return training."""

    observation: Observation
    legal_actions: tuple[Action, ...]
    selected_index: int

    def __post_init__(self) -> None:
        if not self.legal_actions or not 0 <= self.selected_index < len(self.legal_actions):
            raise ValueError("complete-game decision selected index is invalid")


@dataclass(frozen=True, slots=True)
class CompleteEpisode:
    """One full bidding/doubling/cardplay trajectory with one shared terminal payoff."""

    seed: int
    bidding: tuple[CompleteDecision, ...]
    cardplay: tuple[CompleteDecision, ...]
    terminal_payoff: tuple[int, int, int]
    landlord: int | None
    winning_bid: int
    all_pass: bool
    action_count: int
    landlord_strength: float

    def __post_init__(self) -> None:
        if self.action_count <= 0 or self.action_count < len(self.bidding) + len(self.cardplay):
            raise ValueError("complete episode action count is invalid")
        if sum(self.terminal_payoff) != 0:
            raise ValueError("complete episode terminal payoff must be zero-sum")
        if self.all_pass != (self.landlord is None):
            raise ValueError("complete episode all-pass/landlord fields disagree")
        if self.all_pass and self.winning_bid != 0:
            raise ValueError("all-pass episodes cannot have a winning bid")
        if not self.all_pass and not 1 <= self.winning_bid <= 3:
            raise ValueError("resolved complete episode must have a winning bid")


@dataclass(frozen=True, slots=True)
class JointBidBatch:
    """Batched bidding decisions plus the same episode terminal outcomes."""

    batch: BidBatch
    chosen_flat_index: Tensor
    terminal_win: Tensor
    terminal_score: Tensor


@dataclass(frozen=True, slots=True)
class JointTrainingLoss:
    """Combined bid/cardplay loss honoring the curriculum's freeze switch."""

    total: Tensor
    bid: Tensor
    cardplay: Tensor
    cardplay_included: bool


@dataclass(frozen=True, slots=True)
class BidLossOutput:
    """Auditable components of one Bid Head training objective."""

    total: Tensor
    policy: Tensor
    win: Tensor
    score: Tensor
    entropy: Tensor


@dataclass(frozen=True, slots=True)
class BiddingEpisodeSummary:
    """One complete-game row for detecting data-distribution degeneration."""

    landlord_strength: float
    winning_bid: int
    redeal_count: int
    bid_action_count: int
    positive_bid_count: int
    landlord_won: bool
    landlord_score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.landlord_strength) or not math.isfinite(self.landlord_score):
            raise ValueError("bidding episode metrics must be finite")
        if not 1 <= self.winning_bid <= 3 or self.redeal_count < 0:
            raise ValueError("winning bid/redeal count is invalid")
        if self.bid_action_count <= 0 or not 0 <= self.positive_bid_count <= self.bid_action_count:
            raise ValueError("bidding action counts are invalid")


@dataclass(frozen=True, slots=True)
class BiddingDistributionReport:
    """Windowed bidding choices, role strength, and conditional outcome statistics."""

    game_count: int
    landlord_strength_mean: float
    landlord_strength_std: float
    bid_ratio: tuple[float, float, float]
    call_rate: float
    redeal_rate: float
    win_rate_by_bid: tuple[float, float, float]
    mean_score_by_bid: tuple[float, float, float]
    degenerate: bool


@dataclass(frozen=True, slots=True)
class CurriculumMetrics:
    """Metrics consumed by the staged training controller."""

    game_count: int
    calibration_error: float
    call_rate: float
    redeal_rate: float


@dataclass(frozen=True, slots=True)
class CurriculumState:
    """Current training switches derived from the stage, never from wall-clock steps."""

    stage: BiddingStage
    cardplay_frozen: bool
    win_weight: float
    score_weight: float


@dataclass(frozen=True, slots=True)
class BiddingAcceptanceThresholds:
    """Formal gates for claiming an empirical complete-bidding improvement."""

    max_calibration_error: float
    min_call_rate: float
    max_call_rate: float
    max_redeal_rate: float
    max_strength_mean_drift: float
    max_strength_std_ratio: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_calibration_error <= 1.0:
            raise ValueError("acceptance calibration threshold must be in [0, 1]")
        if not 0.0 <= self.min_call_rate < self.max_call_rate <= 1.0:
            raise ValueError("acceptance call-rate interval is invalid")
        if not 0.0 <= self.max_redeal_rate <= 1.0:
            raise ValueError("acceptance redeal threshold must be in [0, 1]")
        if self.max_strength_mean_drift < 0.0 or self.max_strength_std_ratio < 1.0:
            raise ValueError("acceptance distribution thresholds are invalid")


@dataclass(frozen=True, slots=True)
class BiddingAcceptanceReport:
    """Gate result; `accepted` is true only with a positive paired lower bound."""

    accepted: bool
    nondegenerate: bool
    calibrated: bool
    distribution_stable: bool
    beats_fixed_bidder: bool
    reasons: tuple[str, ...]


def load_bidding_training_config(path: Path) -> BiddingTrainingConfig:
    """Load all M9 training switches from one JSON-subset YAML file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "bidding training config")
    monte_carlo = _mapping(root.get("monte_carlo"), "monte_carlo")
    loss = _mapping(root.get("loss"), "loss")
    curriculum = _mapping(root.get("curriculum"), "curriculum")
    return BiddingTrainingConfig(
        schema_version=_integer(root, "schema_version"),
        monte_carlo=MonteCarloConfig(
            max_actions=_integer(monte_carlo, "max_actions"),
            all_pass_win_target=_number(monte_carlo, "all_pass_win_target"),
        ),
        loss=BidLossConfig(
            policy_weight=_number(loss, "policy_weight"),
            win_weight=_number(loss, "win_weight"),
            score_weight=_number(loss, "score_weight"),
            policy_temperature=_number(loss, "policy_temperature"),
            score_scale=_number(loss, "score_scale"),
            entropy_weight=_number(loss, "entropy_weight"),
        ),
        curriculum=CurriculumThresholds(
            max_calibration_error=_number(curriculum, "max_calibration_error"),
            min_call_rate=_number(curriculum, "min_call_rate"),
            max_call_rate=_number(curriculum, "max_call_rate"),
            max_redeal_rate=_number(curriculum, "max_redeal_rate"),
            min_complete_games=_integer(curriculum, "min_complete_games"),
        ),
    )


def generate_initial_bid_mc_labels(
    samples: Sequence[CompleteDealSample],
    rules: RuleConfig,
    continuation_policy: Policy,
    config: MonteCarloConfig,
) -> MonteCarloBidLabels:
    """Evaluate every initial bid across sampled opponent hands and bottom cards."""
    if not samples:
        raise ValueError("Monte Carlo initialization requires at least one complete deal sample")
    if rules["profile"] != "canonical_full":
        raise ValueError("Monte Carlo bidding labels require canonical_full rules")
    reference_hand = samples[0].hands[samples[0].first_bidder]
    reference_bidder = samples[0].first_bidder
    environment = PyDdzEnv()
    first_observation = environment.reset_complete_deal(
        [list(hand) for hand in samples[0].hands],
        list(samples[0].bottom_cards),
        reference_bidder,
        rules,
    )
    legal_actions = tuple(environment.legal_actions())
    if first_observation["phase"] != "bidding":
        raise RuntimeError("complete deal did not initialize in bidding")
    wins = torch.zeros(len(legal_actions), dtype=torch.float64)
    scores = torch.zeros(len(legal_actions), dtype=torch.float64)
    counts = torch.zeros(len(legal_actions), dtype=torch.int64)
    for sample in samples:
        if (
            sample.first_bidder != reference_bidder
            or sample.hands[reference_bidder] != reference_hand
        ):
            raise ValueError(
                "MC samples must preserve the acting seat and its 17-card information set"
            )
        for action_index, forced_action in enumerate(legal_actions):
            environment.reset_complete_deal(
                [list(hand) for hand in sample.hands],
                list(sample.bottom_cards),
                sample.first_bidder,
                rules,
            )
            branch_actions = tuple(environment.legal_actions())
            if forced_action not in branch_actions:
                raise RuntimeError("sampled deal changed the legal initial bid set")
            result = environment.step(forced_action)
            action_count = 1
            decision_counts = [0, 0, 0]
            decision_counts[reference_bidder] = 1
            while not environment.terminal:
                if action_count >= config.max_actions:
                    raise RuntimeError("Monte Carlo bid rollout exceeded max_actions")
                seat = environment.current_player
                observation = environment.observe(seat)
                actions = tuple(environment.legal_actions())
                landlord = observation["landlord"]
                context = PolicyDecisionContext(
                    deal_index=sample.sample_id,
                    deal_seed=sample.sample_id,
                    match_id=f"bid-mc-{sample.sample_id}-{action_index}",
                    seat=seat,
                    role=None if landlord is None else role_for_game_seat(seat, landlord),
                    decision_index=decision_counts[seat],
                )
                selected = continuation_policy.select_action(observation, actions, context)
                if isinstance(selected, bool):
                    raise RuntimeError("Monte Carlo continuation returned a boolean action")
                try:
                    selected_index = operator.index(selected)
                except TypeError as error:
                    raise RuntimeError("Monte Carlo continuation returned a non-integer") from error
                if not 0 <= selected_index < len(actions):
                    raise RuntimeError("Monte Carlo continuation returned an invalid action index")
                result = environment.step(actions[selected_index])
                decision_counts[seat] += 1
                action_count += 1
            terminal = environment.observe(environment.current_player)
            if terminal["landlord"] is None:
                win_value = config.all_pass_win_target
                score_value = 0.0
            else:
                payoff = result["raw_payoff"][reference_bidder]
                win_value = float(payoff > 0)
                score_value = float(payoff)
            wins[action_index] += win_value
            scores[action_index] += score_value
            counts[action_index] += 1
    denominator = counts.to(torch.float64)
    return MonteCarloBidLabels(legal_actions, wins / denominator, scores / denominator, counts)


class BidHeadPolicy:
    """Deterministic inference adapter that accepts bidding observations only."""

    def __init__(
        self,
        policy_id: str,
        model: BidHead,
        rules: RuleConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        if not policy_id:
            raise ValueError("Bid Head policy_id must be non-empty")
        self._policy_id = policy_id
        self._model = model.to(device).eval()
        self._rules = rules
        self._device = device

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        if observation["observer"] != context.seat or observation["phase"] != "bidding":
            raise ValueError("BidHeadPolicy accepts only the acting seat's bidding observation")
        batch = encode_bid_batch(
            (observation,),
            (legal_actions,),
            self._rules,
            history_max_length=self._model.config.history_max_length,
        ).to(self._device)
        with torch.inference_mode():
            output = self._model(batch)
        return int(torch.argmax(output.policy_logits).item())


def collect_complete_episode(
    seed: int,
    rules: RuleConfig,
    bidding_policy: Policy,
    cardplay_policy: Policy,
    *,
    double: bool = False,
    max_actions: int = 1_000,
) -> CompleteEpisode:
    """Run deal→bid→double→cardplay and retain both stages under one terminal return."""
    if rules["profile"] != "canonical_full":
        raise ValueError("complete episode collection requires canonical_full rules")
    if max_actions <= 0:
        raise ValueError("complete collector max_actions must be positive")
    environment = PyDdzEnv()
    environment.reset(seed, rules)
    bidding: list[CompleteDecision] = []
    cardplay: list[CompleteDecision] = []
    decision_counts = [0, 0, 0]
    action_count = 0
    terminal_result: StepResult | None = None
    landlord_strength = 0.0
    captured_landlord: int | None = None
    while not environment.terminal:
        if action_count >= max_actions:
            raise RuntimeError("complete episode exceeded max_actions")
        seat = environment.current_player
        observation = environment.observe(seat)
        actions = tuple(environment.legal_actions())
        landlord = observation["landlord"]
        context = PolicyDecisionContext(
            deal_index=0,
            deal_seed=seed,
            match_id=f"joint-{seed}",
            seat=seat,
            role=None if landlord is None else role_for_game_seat(seat, landlord),
            decision_index=decision_counts[seat],
        )
        if observation["phase"] == "bidding":
            selected = bidding_policy.select_action(observation, actions, context)
            bucket = bidding
        elif observation["phase"] == "card_play":
            selected = cardplay_policy.select_action(observation, actions, context)
            bucket = cardplay
        elif observation["phase"] == "doubling":
            desired = "double" if double else "decline"
            selected = next(
                (index for index, action in enumerate(actions) if action.get("double") == desired),
                -1,
            )
            bucket = None
        else:
            raise RuntimeError(f"collector reached unexpected phase {observation['phase']}")
        if isinstance(selected, bool):
            raise RuntimeError("complete-game policy returned a boolean action")
        try:
            selected_index = operator.index(selected)
        except TypeError as error:
            raise RuntimeError("complete-game policy returned a non-integer action") from error
        if not 0 <= selected_index < len(actions):
            raise RuntimeError("complete-game policy returned an invalid action index")
        if bucket is not None:
            bucket.append(CompleteDecision(observation, actions, selected_index))
        terminal_result = environment.step(actions[selected_index])
        action_count += 1
        decision_counts[seat] += 1
        next_observation = environment.observe(environment.current_player)
        next_landlord = next_observation["landlord"]
        if captured_landlord is None and next_landlord is not None:
            captured_landlord = next_landlord
            landlord_hand = environment.observe(next_landlord)["own_hand"]
            landlord_strength = normalized_hand_strength(landlord_hand)
    if terminal_result is None:
        raise RuntimeError("complete collector ended without a transition")
    terminal = environment.observe(environment.current_player)
    landlord = terminal["landlord"]
    winning_bid = _winning_bid(terminal)
    payoff = terminal_result["raw_payoff"]
    return CompleteEpisode(
        seed=seed,
        bidding=tuple(bidding),
        cardplay=tuple(cardplay),
        terminal_payoff=(payoff[0], payoff[1], payoff[2]),
        landlord=landlord,
        winning_bid=winning_bid,
        all_pass=landlord is None,
        action_count=action_count,
        landlord_strength=landlord_strength,
    )


def build_joint_bid_batch(episode: CompleteEpisode, rules: RuleConfig) -> JointBidBatch:
    """Attach the same terminal outcome to each bidding decision from one episode."""
    if not episode.bidding:
        raise ValueError("complete episode contains no bidding decisions")
    batch = encode_bid_batch(
        tuple(decision.observation for decision in episode.bidding),
        tuple(decision.legal_actions for decision in episode.bidding),
        rules,
    )
    chosen = batch.action_offsets[:-1] + torch.tensor(
        [decision.selected_index for decision in episode.bidding], dtype=torch.int64
    )
    seats = [decision.observation["observer"] for decision in episode.bidding]
    terminal_score = torch.tensor(
        [episode.terminal_payoff[seat] for seat in seats], dtype=torch.float32
    )
    terminal_win = (terminal_score > 0).to(torch.float32)
    if episode.all_pass:
        terminal_win.fill_(0.5)
    return JointBidBatch(batch, chosen, terminal_win, terminal_score)


def combine_joint_training_loss(
    bid_loss: BidLossOutput,
    cardplay_loss: Tensor,
    curriculum: CurriculumState,
    *,
    cardplay_weight: float = 1.0,
) -> JointTrainingLoss:
    """Combine losses only after the metric-gated curriculum unfreezes Cardplay."""
    if cardplay_loss.ndim != 0 or not torch.isfinite(cardplay_loss):
        raise ValueError("cardplay loss must be one finite scalar")
    if not math.isfinite(cardplay_weight) or cardplay_weight < 0.0:
        raise ValueError("cardplay loss weight must be finite and non-negative")
    if curriculum.cardplay_frozen:
        return JointTrainingLoss(bid_loss.total, bid_loss.total, cardplay_loss.detach(), False)
    total = bid_loss.total + cardplay_weight * cardplay_loss
    return JointTrainingLoss(total, bid_loss.total, cardplay_loss, True)


def set_cardplay_frozen(module: torch.nn.Module, frozen: bool) -> None:
    """Apply the curriculum freeze switch to every Cardplay parameter."""
    for parameter in module.parameters():
        parameter.requires_grad_(not frozen)


def normalized_hand_strength(hand: Sequence[int]) -> float:
    """Stable 0..1 monitoring statistic; it is never used as a reward."""
    if len(hand) != 15 or any(count < 0 for count in hand):
        raise ValueError("landlord strength hand must be 15 non-negative rank counts")
    weighted = sum((rank + 1) * count for rank, count in enumerate(hand))
    maximum = sum((rank + 1) * (1 if rank >= 13 else 4) for rank in range(15))
    return weighted / maximum


def bid_supervised_loss(
    output: BidHeadOutput,
    labels: MonteCarloBidLabels,
    action_offsets: Tensor,
    config: BidLossConfig,
) -> BidLossOutput:
    """Fit candidate outcomes and an outcome-ranked policy to Monte Carlo labels."""
    if output.policy_logits.shape != labels.win_target.shape:
        raise ValueError("Bid Head output and Monte Carlo labels differ in action count")
    win_target = labels.win_target.to(output.win_logit)
    score_target = labels.score_target.to(output.expected_score) / config.score_scale
    win = functional.binary_cross_entropy_with_logits(output.win_logit, win_target)
    score = functional.smooth_l1_loss(output.expected_score / config.score_scale, score_target)
    utility = win_target + score_target * config.score_weight
    target_policy = segment_softmax(utility / config.policy_temperature, action_offsets)
    log_probability = torch.log(
        output.policy_probability.clamp_min(torch.finfo(torch.float32).tiny)
    )
    policy = -segment_sum(target_policy * log_probability, action_offsets).mean()
    entropy = -segment_sum(output.policy_probability * log_probability, action_offsets).mean()
    total = (
        config.policy_weight * policy
        + config.win_weight * win
        + config.score_weight * score
        - config.entropy_weight * entropy
    )
    return BidLossOutput(total, policy, win, score, entropy)


def joint_bid_loss(
    output: BidHeadOutput,
    chosen_flat_index: Tensor,
    terminal_win: Tensor,
    terminal_score: Tensor,
    config: BidLossConfig,
) -> BidLossOutput:
    """Apply terminal outcomes to chosen bids while retaining ragged policy gradients."""
    batch_size = chosen_flat_index.shape[0]
    if chosen_flat_index.dtype != torch.int64 or terminal_win.shape != (batch_size,):
        raise ValueError("joint bid chosen indices/win targets must be [B]")
    if terminal_score.shape != (batch_size,):
        raise ValueError("joint bid score targets must be [B]")
    chosen_win = output.win_logit[chosen_flat_index]
    chosen_score = output.expected_score[chosen_flat_index]
    win_target = terminal_win.to(chosen_win)
    score_target = terminal_score.to(chosen_score)
    win = functional.binary_cross_entropy_with_logits(chosen_win, win_target)
    score = functional.smooth_l1_loss(
        chosen_score / config.score_scale, score_target / config.score_scale
    )
    value_baseline = (
        torch.sigmoid(chosen_win.detach())
        + config.score_weight * chosen_score.detach() / config.score_scale
    )
    return_signal = win_target + config.score_weight * score_target / config.score_scale
    log_probability = torch.log(
        output.policy_probability[chosen_flat_index].clamp_min(torch.finfo(torch.float32).tiny)
    )
    policy = -(log_probability * (return_signal - value_baseline)).mean()
    full_log_probability = torch.log(
        output.policy_probability.clamp_min(torch.finfo(torch.float32).tiny)
    )
    entropy = -(output.policy_probability * full_log_probability).mean()
    total = (
        config.policy_weight * policy
        + config.win_weight * win
        + config.score_weight * score
        - config.entropy_weight * entropy
    )
    return BidLossOutput(total, policy, win, score, entropy)


class WinScoreCurriculum:
    """Advance only after observed calibration and non-degeneration gates pass."""

    def __init__(self, thresholds: CurriculumThresholds, score_weight: float) -> None:
        if score_weight < 0.0 or not math.isfinite(score_weight):
            raise ValueError("curriculum score weight must be finite and non-negative")
        self.thresholds = thresholds
        self._score_weight = score_weight
        self._stage = BiddingStage.BID_WIN_FROZEN

    @property
    def state(self) -> CurriculumState:
        """Return the model-freeze and loss switches for the current stage."""
        return CurriculumState(
            stage=self._stage,
            cardplay_frozen=self._stage is BiddingStage.BID_WIN_FROZEN,
            win_weight=1.0,
            score_weight=self._score_weight if self._stage is BiddingStage.JOINT_SCORE else 0.0,
        )

    def maybe_advance(self, metrics: CurriculumMetrics) -> bool:
        """Advance one stage when every checked metric clears its configured gate."""
        gate = self.thresholds
        valid = (
            metrics.game_count >= gate.min_complete_games
            and metrics.calibration_error <= gate.max_calibration_error
            and gate.min_call_rate <= metrics.call_rate <= gate.max_call_rate
            and metrics.redeal_rate <= gate.max_redeal_rate
        )
        if not valid or self._stage is BiddingStage.JOINT_SCORE:
            return False
        self._stage = (
            BiddingStage.JOINT_WIN
            if self._stage is BiddingStage.BID_WIN_FROZEN
            else BiddingStage.JOINT_SCORE
        )
        return True

    def restore(self, stage: BiddingStage) -> None:
        """Restore an exact checkpointed stage without replaying metric gates."""
        if not isinstance(stage, BiddingStage):
            raise ValueError("curriculum restore requires a BiddingStage")
        self._stage = stage


class BiddingDistributionMonitor:
    """Bounded monitor for bidder-induced role and hand-strength distributions."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("bidding monitor capacity must be positive")
        self._rows: deque[BiddingEpisodeSummary] = deque(maxlen=capacity)

    def add(self, summary: BiddingEpisodeSummary) -> None:
        self._rows.append(summary)

    def report(self, min_call_rate: float, max_call_rate: float) -> BiddingDistributionReport:
        """Summarize the current window without inventing missing bid buckets."""
        if not self._rows:
            raise ValueError("cannot report an empty bidding monitor")
        if not 0.0 <= min_call_rate < max_call_rate <= 1.0:
            raise ValueError("monitor call-rate interval is invalid")
        rows = tuple(self._rows)
        strengths = [row.landlord_strength for row in rows]
        mean_strength = sum(strengths) / len(strengths)
        variance = sum((value - mean_strength) ** 2 for value in strengths) / len(strengths)
        bid_counts = [sum(row.winning_bid == bid for row in rows) for bid in (1, 2, 3)]
        bid_ratio = tuple(count / len(rows) for count in bid_counts)
        win_by_bid: list[float] = []
        score_by_bid: list[float] = []
        for bid, count in zip((1, 2, 3), bid_counts, strict=True):
            bucket = [row for row in rows if row.winning_bid == bid]
            win_by_bid.append(
                0.0 if count == 0 else sum(row.landlord_won for row in bucket) / count
            )
            score_by_bid.append(
                0.0 if count == 0 else sum(row.landlord_score for row in bucket) / count
            )
        call_rate = sum(row.positive_bid_count for row in rows) / sum(
            row.bid_action_count for row in rows
        )
        redeal_rate = sum(row.redeal_count for row in rows) / (
            len(rows) + sum(row.redeal_count for row in rows)
        )
        degenerate = not min_call_rate <= call_rate <= max_call_rate or max(bid_ratio) >= 0.99
        return BiddingDistributionReport(
            game_count=len(rows),
            landlord_strength_mean=mean_strength,
            landlord_strength_std=math.sqrt(variance),
            bid_ratio=cast(tuple[float, float, float], bid_ratio),
            call_rate=call_rate,
            redeal_rate=redeal_rate,
            win_rate_by_bid=cast(tuple[float, float, float], tuple(win_by_bid)),
            mean_score_by_bid=cast(tuple[float, float, float], tuple(score_by_bid)),
            degenerate=degenerate,
        )


def binary_calibration_error(
    probability: Sequence[float], outcome: Sequence[bool], bins: int = 10
) -> float:
    """Compute equal-width expected calibration error for final-win predictions."""
    if not probability or len(probability) != len(outcome) or bins <= 0:
        raise ValueError("calibration inputs must be non-empty/equal and bins positive")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probability):
        raise ValueError("calibration probabilities must be finite in [0, 1]")
    total = len(probability)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, value in enumerate(probability)
            if lower <= value < upper or (bin_index == bins - 1 and value == 1.0)
        ]
        if indices:
            confidence = sum(probability[index] for index in indices) / len(indices)
            accuracy = sum(outcome[index] for index in indices) / len(indices)
            error += len(indices) / total * abs(confidence - accuracy)
    return error


def evaluate_bidding_acceptance(
    candidate: BiddingDistributionReport,
    reference: BiddingDistributionReport,
    calibration_error: float,
    paired_improvement_ci_lower: float,
    thresholds: BiddingAcceptanceThresholds,
) -> BiddingAcceptanceReport:
    """Require all M9 empirical gates before permitting an improvement claim."""
    reasons: list[str] = []
    nondegenerate = (
        not candidate.degenerate
        and thresholds.min_call_rate <= candidate.call_rate <= thresholds.max_call_rate
        and candidate.redeal_rate <= thresholds.max_redeal_rate
    )
    if not nondegenerate:
        reasons.append("bidding choices are degenerate or redeal rate is excessive")
    calibrated = calibration_error <= thresholds.max_calibration_error
    if not calibrated:
        reasons.append("final-win prediction is not calibrated")
    mean_drift = abs(candidate.landlord_strength_mean - reference.landlord_strength_mean)
    reference_std = max(reference.landlord_strength_std, 1.0e-9)
    std_ratio = candidate.landlord_strength_std / reference_std
    distribution_stable = (
        mean_drift <= thresholds.max_strength_mean_drift
        and 1.0 / thresholds.max_strength_std_ratio
        <= std_ratio
        <= thresholds.max_strength_std_ratio
    )
    if not distribution_stable:
        reasons.append("landlord strength distribution drift exceeds the configured gate")
    beats_fixed = paired_improvement_ci_lower > 0.0
    if not beats_fixed:
        reasons.append("paired lower confidence bound does not beat the fixed bidder")
    accepted = nondegenerate and calibrated and distribution_stable and beats_fixed
    return BiddingAcceptanceReport(
        accepted, nondegenerate, calibrated, distribution_stable, beats_fixed, tuple(reasons)
    )


def _winning_bid(observation: Observation) -> int:
    result = 0
    for event in observation["bid_history"]:
        action = event["action"]
        if action in ("call", "rob"):
            result = 1
        elif isinstance(action, dict):
            result = max(result, action["score"])
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _rank_rows(
    value: object,
    row_count: int,
    card_count: int,
    label: str,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != row_count:
        raise RuntimeError(f"serialized {label} must contain {row_count} rows")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 15
            or any(not isinstance(item, int) or isinstance(item, bool) for item in row)
            or any(item < 0 for item in row)
            or sum(row) != card_count
        ):
            raise RuntimeError(f"serialized {label} contains an invalid rank-count row")
        rows.append(tuple(cast(list[int], row)))
    return tuple(rows)


def _expand_counts(counts: Sequence[int]) -> list[int]:
    return [rank for rank, count in enumerate(counts) for _ in range(count)]


def _counts(cards: Sequence[int]) -> tuple[int, ...]:
    counts = [0] * 15
    for card in cards:
        if not 0 <= card < 15:
            raise RuntimeError(f"sampled rank is outside 0..14: {card}")
        counts[card] += 1
    return tuple(counts)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"bidding training config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"bidding training config {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"bidding training config {key} must be finite")
    return numeric


__all__ = (
    "BIDDING_TRAINING_SCHEMA_VERSION",
    "BidLossConfig",
    "BidLossOutput",
    "BidHeadPolicy",
    "BiddingAcceptanceReport",
    "BiddingAcceptanceThresholds",
    "BiddingDistributionMonitor",
    "BiddingDistributionReport",
    "BiddingEpisodeSummary",
    "BiddingStage",
    "BiddingTrainingConfig",
    "CompleteDealSample",
    "CompleteDecision",
    "CompleteEpisode",
    "CurriculumMetrics",
    "CurriculumState",
    "CurriculumThresholds",
    "JointBidBatch",
    "JointTrainingLoss",
    "MonteCarloBidLabels",
    "MonteCarloConfig",
    "WinScoreCurriculum",
    "bid_supervised_loss",
    "binary_calibration_error",
    "build_joint_bid_batch",
    "collect_complete_episode",
    "combine_joint_training_loss",
    "evaluate_bidding_acceptance",
    "generate_initial_bid_mc_labels",
    "joint_bid_loss",
    "load_bidding_training_config",
    "normalized_hand_strength",
    "sample_initial_bid_deals",
    "set_cardplay_frozen",
)
