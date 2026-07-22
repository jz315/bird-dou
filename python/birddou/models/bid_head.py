"""Information-set-safe candidate-conditioned bidding model."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from birddou.belief.three_container_crf import (
    ThreeContainerMarginals,
    three_container_marginals,
)
from birddou.env_types import Action, BidAction, BidGameAction, Observation, RuleConfig
from birddou.models.rank_mixer import RmsNorm
from birddou.models.segment_ops import segment_softmax, segment_state_index

BID_HEAD_SCHEMA_VERSION = 2
BID_HEAD_ARCHITECTURE = "bird_dou_bid_head_v2"
BID_ACTION_COUNT = 6
BID_HISTORY_PAD = BID_ACTION_COUNT
BID_ACTOR_PAD = 3


@dataclass(frozen=True, slots=True)
class BidHeadConfig:
    """Versioned dimensions for the bidding-only actor and outcome heads."""

    schema_version: int
    architecture: str
    d_model: int
    rank_layers: int
    history_layers: int
    attention_heads: int
    hidden_multiplier: int
    history_max_length: int
    dropout: float

    def __post_init__(self) -> None:
        if self.schema_version != BID_HEAD_SCHEMA_VERSION:
            raise ValueError("unsupported Bid Head schema")
        if self.architecture != BID_HEAD_ARCHITECTURE:
            raise ValueError("unsupported Bid Head architecture")
        positive = (
            self.d_model,
            self.rank_layers,
            self.history_layers,
            self.attention_heads,
            self.hidden_multiplier,
            self.history_max_length,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Bid Head dimensions and layer counts must be positive")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("Bid Head d_model must be divisible by attention_heads")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("Bid Head dropout must be in [0, 1)")

    def fingerprint(self) -> str:
        """Return a stable architecture/configuration identity."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class BidBatch:
    """Batched legal observations and ragged candidate bids, with no hidden cards."""

    own_hand: Tensor
    unknown_counts: Tensor
    history_action: Tensor
    history_actor: Tensor
    history_mask: Tensor
    seat: Tensor
    rule_mode: Tensor
    rule_features: Tensor
    capacity_a: Tensor
    capacity_b: Tensor
    legal_action_code: Tensor
    action_state_index: Tensor
    action_offsets: Tensor

    def __post_init__(self) -> None:
        batch_size = self.own_hand.shape[0]
        if self.own_hand.dtype != torch.int64 or self.own_hand.shape != (batch_size, 15):
            raise ValueError("BidBatch own_hand must be int64 [B, 15]")
        if self.unknown_counts.dtype != torch.int64 or self.unknown_counts.shape != (
            batch_size,
            15,
        ):
            raise ValueError("BidBatch unknown_counts must be int64 [B, 15]")
        history_length = self.history_action.shape[1]
        if self.history_action.dtype != torch.int64 or self.history_action.shape != (
            batch_size,
            history_length,
        ):
            raise ValueError("BidBatch history_action must be int64 [B, H]")
        if self.history_actor.dtype != torch.int64 or self.history_actor.shape != (
            batch_size,
            history_length,
        ):
            raise ValueError("BidBatch history_actor must be int64 [B, H]")
        if self.history_mask.dtype != torch.bool or self.history_mask.shape != (
            batch_size,
            history_length,
        ):
            raise ValueError("BidBatch history_mask must be bool [B, H]")
        for value, label in (
            (self.seat, "seat"),
            (self.rule_mode, "rule_mode"),
            (self.capacity_a, "capacity_a"),
            (self.capacity_b, "capacity_b"),
        ):
            if value.dtype != torch.int64 or value.shape != (batch_size,):
                raise ValueError(f"BidBatch {label} must be int64 [B]")
        if self.rule_features.dtype != torch.float32 or self.rule_features.shape != (
            batch_size,
            4,
        ):
            raise ValueError("BidBatch rule_features must be float32 [B, 4]")
        action_count = self.legal_action_code.shape[0]
        if self.legal_action_code.dtype != torch.int64 or self.legal_action_code.shape != (
            action_count,
        ):
            raise ValueError("BidBatch legal_action_code must be int64 [M]")
        if self.action_state_index.dtype != torch.int64 or self.action_state_index.shape != (
            action_count,
        ):
            raise ValueError("BidBatch action_state_index must be int64 [M]")
        expected_state_index = segment_state_index(self.action_offsets)
        if not torch.equal(self.action_state_index, expected_state_index):
            raise ValueError("BidBatch state index differs from action offsets")
        tensors = (
            self.unknown_counts,
            self.history_action,
            self.history_actor,
            self.history_mask,
            self.seat,
            self.rule_mode,
            self.rule_features,
            self.capacity_a,
            self.capacity_b,
            self.legal_action_code,
            self.action_state_index,
            self.action_offsets,
        )
        if any(value.device != self.own_hand.device for value in tensors):
            raise ValueError("every BidBatch tensor must share one device")

    @property
    def batch_size(self) -> int:
        return self.own_hand.shape[0]

    @property
    def action_count(self) -> int:
        return self.legal_action_code.shape[0]

    def to(self, device: str | torch.device) -> BidBatch:
        """Move the complete batch without changing its ragged segmentation."""
        return BidBatch(
            own_hand=self.own_hand.to(device),
            unknown_counts=self.unknown_counts.to(device),
            history_action=self.history_action.to(device),
            history_actor=self.history_actor.to(device),
            history_mask=self.history_mask.to(device),
            seat=self.seat.to(device),
            rule_mode=self.rule_mode.to(device),
            rule_features=self.rule_features.to(device),
            capacity_a=self.capacity_a.to(device),
            capacity_b=self.capacity_b.to(device),
            legal_action_code=self.legal_action_code.to(device),
            action_state_index=self.action_state_index.to(device),
            action_offsets=self.action_offsets.to(device),
        )


@dataclass(frozen=True, slots=True)
class BidHeadOutput:
    """Candidate DMC value/outcome predictions and constrained belief diagnostics."""

    mc_q: Tensor
    policy_logits: Tensor
    policy_probability: Tensor
    win_logit: Tensor
    win_probability: Tensor
    expected_score: Tensor
    state: Tensor
    belief_scores: Tensor
    belief: ThreeContainerMarginals


def load_bid_head_config(path: Path) -> BidHeadConfig:
    """Load the JSON-subset YAML Bid Head configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "Bid Head config")
    return BidHeadConfig(
        schema_version=_integer(root, "schema_version"),
        architecture=_string(root, "architecture"),
        d_model=_integer(root, "d_model"),
        rank_layers=_integer(root, "rank_layers"),
        history_layers=_integer(root, "history_layers"),
        attention_heads=_integer(root, "attention_heads"),
        hidden_multiplier=_integer(root, "hidden_multiplier"),
        history_max_length=_integer(root, "history_max_length"),
        dropout=_number(root, "dropout"),
    )


def encode_bid_batch(
    observations: Sequence[Observation],
    legal_actions: Sequence[Sequence[Action]],
    rules: RuleConfig,
    *,
    history_max_length: int = 3,
) -> BidBatch:
    """Encode public bidding observations and their complete legal candidate sets."""
    if not observations or len(observations) != len(legal_actions):
        raise ValueError("bid observations/actions must be non-empty and have equal length")
    if history_max_length <= 0:
        raise ValueError("history_max_length must be positive")
    own_hands: list[list[int]] = []
    unknown_counts: list[list[int]] = []
    history_actions: list[list[int]] = []
    history_actors: list[list[int]] = []
    history_masks: list[list[bool]] = []
    seats: list[int] = []
    modes: list[int] = []
    rule_features: list[list[float]] = []
    capacities_a: list[int] = []
    capacities_b: list[int] = []
    action_codes: list[int] = []
    offsets = [0]
    for state_index, (observation, actions) in enumerate(
        zip(observations, legal_actions, strict=True)
    ):
        _validate_bid_observation(observation, actions, rules, state_index)
        observer = observation["observer"]
        own_hands.append(list(observation["own_hand"]))
        unknown_counts.append(list(observation["unknown_pool"]))
        seats.append(observer)
        mode = rules["bidding"]["mode"]
        modes.append(0 if mode == "score" else 1)
        maximum_bid = rules["bidding"]["max_bid"]
        rule_features.append(
            [
                0.0 if maximum_bid is None else maximum_bid / 3.0,
                float(rules["bottom_cards_public"]),
                float(rules["doubling_enabled"]),
                0.0 if rules["score_cap"] is None else min(rules["score_cap"], 256) / 256.0,
            ]
        )
        capacities_a.append(observation["cards_left"][(observer + 1) % 3])
        capacities_b.append(observation["cards_left"][(observer + 2) % 3])
        events = observation["bid_history"]
        if len(events) > history_max_length:
            raise ValueError("bidding history exceeds configured maximum length")
        encoded_actions = [_bid_action_code(event["action"]) for event in events]
        encoded_actors = [(event["actor"] - observer) % 3 for event in events]
        padding = history_max_length - len(events)
        history_actions.append(encoded_actions + [BID_HISTORY_PAD] * padding)
        history_actors.append(encoded_actors + [BID_ACTOR_PAD] * padding)
        history_masks.append([True] * len(events) + [False] * padding)
        action_codes.extend(
            _bid_action_code(cast(BidGameAction, action)["bid"]) for action in actions
        )
        offsets.append(len(action_codes))
    action_offsets = torch.tensor(offsets, dtype=torch.int64)
    return BidBatch(
        own_hand=torch.tensor(own_hands, dtype=torch.int64),
        unknown_counts=torch.tensor(unknown_counts, dtype=torch.int64),
        history_action=torch.tensor(history_actions, dtype=torch.int64),
        history_actor=torch.tensor(history_actors, dtype=torch.int64),
        history_mask=torch.tensor(history_masks, dtype=torch.bool),
        seat=torch.tensor(seats, dtype=torch.int64),
        rule_mode=torch.tensor(modes, dtype=torch.int64),
        rule_features=torch.tensor(rule_features, dtype=torch.float32),
        capacity_a=torch.tensor(capacities_a, dtype=torch.int64),
        capacity_b=torch.tensor(capacities_b, dtype=torch.int64),
        legal_action_code=torch.tensor(action_codes, dtype=torch.int64),
        action_state_index=segment_state_index(action_offsets),
        action_offsets=action_offsets,
    )


class BidHead(nn.Module):
    """Score every legal bid by final win probability and score under Cardplay."""

    def __init__(self, config: BidHeadConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        hidden = width * config.hidden_multiplier
        self.rank_embedding = nn.Embedding(15, width)
        self.count_embedding = nn.Embedding(5, width)
        rank_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.attention_heads,
            dim_feedforward=hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rank_encoder = nn.TransformerEncoder(
            rank_layer, config.rank_layers, enable_nested_tensor=False
        )
        self.history_action_embedding = nn.Embedding(BID_ACTION_COUNT + 1, width)
        self.history_actor_embedding = nn.Embedding(4, width)
        self.history_encoder = nn.GRU(
            width,
            width,
            num_layers=config.history_layers,
            batch_first=True,
            dropout=config.dropout if config.history_layers > 1 else 0.0,
        )
        self.empty_history = nn.Parameter(torch.zeros(width))
        self.seat_embedding = nn.Embedding(3, width)
        self.rule_mode_embedding = nn.Embedding(2, width)
        self.rule_projection = nn.Linear(4, width)
        self.public_fusion = nn.Sequential(
            nn.Linear(width * 5, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, width),
            RmsNorm(width),
        )
        self.belief_score = nn.Sequential(
            nn.Linear(width * 2, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 25),
        )
        self.belief_rank_projection = nn.Sequential(nn.Linear(9, width), nn.SiLU(), RmsNorm(width))
        self.belief_fusion = nn.Sequential(
            nn.Linear(width * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
            RmsNorm(width),
        )
        self.action_embedding = nn.Embedding(BID_ACTION_COUNT, width)
        self.outcome_head = nn.Sequential(
            nn.Linear(width * 2, hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, batch: BidBatch) -> BidHeadOutput:
        """Run public encoding, exact three-container belief, and ragged action scoring."""
        rank_ids = torch.arange(15, device=batch.own_hand.device)[None, :]
        rank_tokens = self.rank_encoder(
            self.rank_embedding(rank_ids) + self.count_embedding(batch.own_hand)
        )
        hand_state = rank_tokens.mean(dim=1)
        history_tokens = self.history_action_embedding(
            batch.history_action
        ) + self.history_actor_embedding(batch.history_actor)
        history_tokens = history_tokens * batch.history_mask.unsqueeze(-1)
        history_length = batch.history_mask.sum(dim=1)
        expected_mask = (
            torch.arange(batch.history_mask.shape[1], device=batch.history_mask.device)[None, :]
            < history_length[:, None]
        )
        if not torch.equal(batch.history_mask, expected_mask):
            raise ValueError("Bid Head history_mask must be a contiguous valid prefix")
        history_state = self.empty_history[None].expand(batch.batch_size, -1)
        for length in torch.unique(history_length).tolist():
            if length == 0:
                continue
            indices = torch.nonzero(history_length == length, as_tuple=False).squeeze(-1)
            encoded, _ = self.history_encoder(
                history_tokens.index_select(0, indices)[:, :length]
            )
            history_state = history_state.index_copy(0, indices, encoded[:, -1])
        public_state = self.public_fusion(
            torch.cat(
                (
                    hand_state,
                    history_state,
                    self.seat_embedding(batch.seat),
                    self.rule_mode_embedding(batch.rule_mode),
                    self.rule_projection(batch.rule_features),
                ),
                dim=-1,
            )
        )
        belief_input = torch.cat((rank_tokens, public_state[:, None].expand(-1, 15, -1)), dim=-1)
        belief_scores = self.belief_score(belief_input).view(-1, 15, 5, 5)
        belief = three_container_marginals(
            belief_scores.float(),
            batch.unknown_counts,
            batch.capacity_a,
            batch.capacity_b,
        )
        belief_features = torch.cat(
            (
                belief.expected / 4.0,
                belief.variance / 4.0,
                belief.entropy / math.log(5.0),
            ),
            dim=-1,
        )
        belief_rank = self.belief_rank_projection(belief_features)
        state = public_state + self.belief_fusion(
            torch.cat((public_state, belief_rank.mean(dim=1), belief_rank.amax(dim=1)), dim=-1)
        )
        action_state = state[batch.action_state_index]
        action = self.action_embedding(batch.legal_action_code)
        predictions = self.outcome_head(torch.cat((action_state, action), dim=-1))
        mc_q, win_logit, expected_score = predictions.unbind(dim=-1)
        policy_logits = mc_q
        policy_probability = segment_softmax(policy_logits.float(), batch.action_offsets)
        return BidHeadOutput(
            mc_q=mc_q,
            policy_logits=policy_logits,
            policy_probability=policy_probability,
            win_logit=win_logit,
            win_probability=torch.sigmoid(win_logit),
            expected_score=expected_score,
            state=state,
            belief_scores=belief_scores,
            belief=belief,
        )


def _validate_bid_observation(
    observation: Observation,
    actions: Sequence[Action],
    rules: RuleConfig,
    state_index: int,
) -> None:
    if observation["phase"] != "bidding" or observation["landlord"] is not None:
        raise ValueError(f"state {state_index} is not an unresolved bidding observation")
    if observation["role"] != "unassigned":
        raise ValueError("bidding observation role must be unassigned")
    if observation["observer"] != observation["current_player"]:
        raise ValueError("BidBatch requires current-player observations")
    if sum(observation["own_hand"]) != 17 or sum(observation["unknown_pool"]) != 37:
        raise ValueError("bidding observation must expose 17 own and 37 unknown cards")
    if sum(observation["public_bottom_cards"]) != 0:
        raise ValueError("unresolved bidding observation cannot expose bottom cards")
    if rules["profile"] != "canonical_full" or rules["bidding"]["mode"] == "disabled":
        raise ValueError("BidBatch requires a complete rule profile with bidding enabled")
    if not actions or any("bid" not in action for action in actions):
        raise ValueError(f"state {state_index} must contain only legal bid actions")


def _bid_action_code(action: BidAction) -> int:
    if action == "pass":
        return 0
    if action == "call":
        return 4
    if action == "rob":
        return 5
    if isinstance(action, dict) and set(action) == {"score"}:
        score = action["score"]
        if isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 3:
            return score
    raise ValueError(f"unknown bid action: {action!r}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Bid Head config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Bid Head config {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Bid Head config {key} must be finite")
    return numeric


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Bid Head config {key} must be a non-empty string")
    return value


__all__ = (
    "BID_ACTION_COUNT",
    "BID_HEAD_ARCHITECTURE",
    "BID_HEAD_SCHEMA_VERSION",
    "BidBatch",
    "BidHead",
    "BidHeadConfig",
    "BidHeadOutput",
    "encode_bid_batch",
    "load_bid_head_config",
)
