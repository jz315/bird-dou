"""COMA-style farmer credit assignment, safe specialization, and rollout data."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional

from birddou import PyDdzEnv
from birddou.env_types import Action, Observation, RuleConfig, StepResult
from birddou.eval.metrics import ArenaReport
from birddou.eval.paired_deals import PairedDealSet, ScheduledMatch, SeatAssignment
from birddou.models.bird_dou import BirdDouModel
from birddou.models.segment_ops import segment_sum

FARMER_COORDINATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FarmerCoordinationConfig:
    """Switchable farmer-team loss, rollout, and specialist-update controls."""

    schema_version: int = FARMER_COORDINATION_SCHEMA_VERSION
    critic_coef: float = 1.0
    counterfactual_actor_coef: float = 0.1
    rollout_coef: float = 0.25
    huber_delta: float = 1.0
    rollout_top_n: int = 4
    max_rollout_states_per_batch: int = 32
    max_rollout_actions: int = 1_000
    specialist_learning_rate: float = 1e-4
    specialist_weight_decay: float = 1e-5
    protect_landlord: bool = True
    handcrafted_cooperation_rewards: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != FARMER_COORDINATION_SCHEMA_VERSION:
            raise ValueError("unsupported farmer coordination schema")
        coefficients = (
            self.critic_coef,
            self.counterfactual_actor_coef,
            self.rollout_coef,
            self.specialist_learning_rate,
            self.specialist_weight_decay,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in coefficients):
            raise ValueError("farmer coordination coefficients must be finite/non-negative")
        if self.critic_coef == 0.0 or self.specialist_learning_rate == 0.0:
            raise ValueError("farmer critic and specialist learning-rate must be positive")
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0.0:
            raise ValueError("farmer coordination Huber delta must be positive")
        if (
            min(
                self.rollout_top_n,
                self.max_rollout_states_per_batch,
                self.max_rollout_actions,
            )
            <= 0
        ):
            raise ValueError("farmer rollout limits must be positive")
        if self.handcrafted_cooperation_rewards:
            raise ValueError("handcrafted farmer cooperation rewards are forbidden")


def load_farmer_coordination_config(path: Path) -> FarmerCoordinationConfig:
    """Load the versioned JSON-subset YAML M8 training configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = _mapping(raw, "farmer coordination config")
    return FarmerCoordinationConfig(
        schema_version=_integer(values, "schema_version"),
        critic_coef=_number(values, "critic_coef"),
        counterfactual_actor_coef=_number(values, "counterfactual_actor_coef"),
        rollout_coef=_number(values, "rollout_coef"),
        huber_delta=_number(values, "huber_delta"),
        rollout_top_n=_integer(values, "rollout_top_n"),
        max_rollout_states_per_batch=_integer(values, "max_rollout_states_per_batch"),
        max_rollout_actions=_integer(values, "max_rollout_actions"),
        specialist_learning_rate=_number(values, "specialist_learning_rate"),
        specialist_weight_decay=_number(values, "specialist_weight_decay"),
        protect_landlord=_boolean(values, "protect_landlord"),
        handcrafted_cooperation_rewards=_boolean(values, "handcrafted_cooperation_rewards"),
    )


@dataclass(frozen=True, slots=True)
class CounterfactualAdvantage:
    """Exact per-state policy baseline and selected-action advantage."""

    baseline: Tensor
    chosen_q: Tensor
    advantage: Tensor


def counterfactual_advantage(
    team_q: Tensor,
    actor_probability: Tensor,
    action_offsets: Tensor,
    chosen_action_flat_index: Tensor,
) -> CounterfactualAdvantage:
    """Compute `Q(a_taken) - sum_a pi(a|I)Q(a)` within each legal segment."""
    if team_q.ndim != 1 or actor_probability.shape != team_q.shape:
        raise ValueError("team Q and actor probability must share flat action shape")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all()
        for value in (team_q, actor_probability)
    ):
        raise ValueError("counterfactual action values must be finite and floating")
    state_count = action_offsets.numel() - 1
    if action_offsets.dtype != torch.int64 or action_offsets.ndim != 1 or state_count <= 0:
        raise ValueError("counterfactual offsets must be int64 [B+1]")
    if chosen_action_flat_index.dtype != torch.int64 or chosen_action_flat_index.shape != (
        state_count,
    ):
        raise ValueError("counterfactual chosen actions must be int64 [B]")
    devices = {
        value.device
        for value in (
            team_q,
            actor_probability,
            action_offsets,
            chosen_action_flat_index,
        )
    }
    if len(devices) != 1:
        raise ValueError("counterfactual tensors must share one device")
    if torch.any((actor_probability < 0.0) | (actor_probability > 1.0)):
        raise ValueError("counterfactual actor probabilities must be in 0..1")
    offsets = action_offsets.detach().cpu().tolist()
    for state, chosen in enumerate(chosen_action_flat_index.detach().cpu().tolist()):
        if not offsets[state] <= chosen < offsets[state + 1]:
            raise ValueError("counterfactual chosen action lies outside its legal segment")
    probability_sum = segment_sum(actor_probability, action_offsets)
    if not torch.allclose(
        probability_sum,
        torch.ones_like(probability_sum),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("actor probabilities must normalize inside every segment")
    baseline = segment_sum(actor_probability * team_q, action_offsets)
    chosen_q = team_q[chosen_action_flat_index]
    return CounterfactualAdvantage(baseline, chosen_q, chosen_q - baseline)


@dataclass(frozen=True, slots=True)
class FarmerCoordinationLoss:
    """Central Critic, counterfactual Actor, and sparse rollout supervision."""

    total: Tensor
    critic: Tensor
    counterfactual_actor: Tensor
    rollout: Tensor
    counterfactual: CounterfactualAdvantage


def farmer_coordination_loss(
    config: FarmerCoordinationConfig,
    *,
    team_q: Tensor,
    actor_probability: Tensor,
    actor_log_probability: Tensor,
    action_offsets: Tensor,
    chosen_action_flat_index: Tensor,
    terminal_team_target: Tensor,
    rollout_action_flat_index: Tensor | None = None,
    rollout_team_target: Tensor | None = None,
) -> FarmerCoordinationLoss:
    """Train only from true team returns and full-state counterfactual values."""
    counterfactual = counterfactual_advantage(
        team_q,
        actor_probability,
        action_offsets,
        chosen_action_flat_index,
    )
    state_count = chosen_action_flat_index.numel()
    if (
        actor_log_probability.shape != team_q.shape
        or not actor_log_probability.is_floating_point()
        or not torch.isfinite(actor_log_probability).all()
    ):
        raise ValueError("farmer actor log probabilities must match team Q")
    if (
        terminal_team_target.shape != (state_count,)
        or not terminal_team_target.is_floating_point()
        or not torch.isfinite(terminal_team_target).all()
    ):
        raise ValueError("terminal farmer-team targets must be finite [B]")
    critic = functional.huber_loss(
        counterfactual.chosen_q,
        terminal_team_target,
        delta=config.huber_delta,
    )
    actor = -(
        actor_log_probability[chosen_action_flat_index] * counterfactual.advantage.detach()
    ).mean()
    zero = team_q.new_zeros(())
    if (rollout_action_flat_index is None) != (rollout_team_target is None):
        raise ValueError("rollout action indices and targets must be supplied together")
    rollout = zero
    if rollout_action_flat_index is not None and rollout_team_target is not None:
        if (
            rollout_action_flat_index.dtype != torch.int64
            or rollout_action_flat_index.ndim != 1
            or rollout_team_target.shape != rollout_action_flat_index.shape
            or torch.any(
                (rollout_action_flat_index < 0) | (rollout_action_flat_index >= team_q.numel())
            )
            or not rollout_team_target.is_floating_point()
            or not torch.isfinite(rollout_team_target).all()
        ):
            raise ValueError("sparse farmer rollout targets are invalid")
        rollout = functional.huber_loss(
            team_q[rollout_action_flat_index],
            rollout_team_target,
            delta=config.huber_delta,
        )
    total = (
        config.critic_coef * critic
        + config.counterfactual_actor_coef * actor
        + config.rollout_coef * rollout
    )
    output = FarmerCoordinationLoss(total, critic, actor, rollout, counterfactual)
    if any(
        not getattr(output, field).isfinite()
        for field in ("total", "critic", "counterfactual_actor", "rollout")
    ):
        raise RuntimeError("farmer coordination loss produced a non-finite value")
    return output


class FarmerSpecialistOptimizer:
    """Update the two farmer adapters/heads while protecting landlord behavior."""

    def __init__(self, model: BirdDouModel, config: FarmerCoordinationConfig) -> None:
        self.model = model
        self.config = config
        if config.protect_landlord:
            for model_parameter in model.parameters():
                model_parameter.requires_grad_(False)
            packed = [
                model.role_adapter.role_embedding.weight,
                model.role_adapter.seat_embedding.weight,
                model.role_adapter.norm.weight.weight,
                model.role_adapter.norm.bias.weight,
            ]
            specialist = [
                *model.role_adapter.adapters[1].parameters(),
                *model.role_adapter.adapters[2].parameters(),
                *model.output_heads.heads[1].parameters(),
                *model.output_heads.heads[2].parameters(),
            ]
            for selected_parameter in (*packed, *specialist):
                selected_parameter.requires_grad_(True)
            parameter_groups: list[dict[str, Any]] = [
                {"params": packed, "weight_decay": 0.0},
                {"params": specialist, "weight_decay": config.specialist_weight_decay},
            ]
        else:
            for model_parameter in model.parameters():
                model_parameter.requires_grad_(True)
            parameter_groups = [
                {
                    "params": list(model.parameters()),
                    "weight_decay": config.specialist_weight_decay,
                }
            ]
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=config.specialist_learning_rate,
        )
        self._landlord_snapshot = _landlord_dependency_snapshot(model)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Apply one update and fail immediately if protected landlord tensors drift."""
        if self.config.protect_landlord and not _landlord_packed_gradients_are_zero(self.model):
            raise RuntimeError("farmer-only update received a landlord gradient")
        self.optimizer.step()
        if self.config.protect_landlord and not _snapshot_matches(
            self.model, self._landlord_snapshot
        ):
            _restore_snapshot(self.model, self._landlord_snapshot)
            raise RuntimeError("protected landlord parameters changed in farmer-only update")

    def state_dict(self) -> dict[str, object]:
        return cast(dict[str, object], self.optimizer.state_dict())

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.optimizer.load_state_dict(dict(state))
        self._landlord_snapshot = _landlord_dependency_snapshot(self.model)


def _landlord_dependency_snapshot(model: BirdDouModel) -> dict[str, Tensor]:
    snapshot: dict[str, Tensor] = {}
    packed_names = {
        "role_adapter.role_embedding.weight",
        "role_adapter.seat_embedding.weight",
        "role_adapter.norm.weight.weight",
        "role_adapter.norm.bias.weight",
    }
    ignored_prefixes = (
        "role_adapter.adapters.1.",
        "role_adapter.adapters.2.",
        "output_heads.heads.1.",
        "output_heads.heads.2.",
    )
    for name, parameter in model.named_parameters():
        if name in packed_names:
            snapshot[name] = parameter[0].detach().clone()
        elif not name.startswith(ignored_prefixes):
            snapshot[name] = parameter.detach().clone()
    return snapshot


def _snapshot_matches(model: BirdDouModel, snapshot: Mapping[str, Tensor]) -> bool:
    current = dict(model.named_parameters())
    for name, expected in snapshot.items():
        value = current[name][0] if expected.shape != current[name].shape else current[name]
        if not torch.equal(value.detach(), expected):
            return False
    return True


def _restore_snapshot(model: BirdDouModel, snapshot: Mapping[str, Tensor]) -> None:
    current = dict(model.named_parameters())
    with torch.no_grad():
        for name, expected in snapshot.items():
            value = current[name]
            if expected.shape != value.shape:
                value[0].copy_(expected)
            else:
                value.copy_(expected)


def _landlord_packed_gradients_are_zero(model: BirdDouModel) -> bool:
    packed = (
        model.role_adapter.role_embedding.weight,
        model.role_adapter.seat_embedding.weight,
        model.role_adapter.norm.weight.weight,
        model.role_adapter.norm.bias.weight,
    )
    return all(
        parameter.grad is None or torch.count_nonzero(parameter.grad[0]).item() == 0
        for parameter in packed
    )


class RolloutPolicy(Protocol):
    """Public-observation continuation policy used only inside training rollouts."""

    def select_action(
        self,
        observation: Observation,
        legal_actions: tuple[Action, ...],
        seed: int,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class CounterfactualRolloutTarget:
    action_index: int
    canonical_action: bytes
    actor_score: float
    team_return: float
    rollout_actions: int


@dataclass(frozen=True, slots=True)
class CounterfactualRolloutBatch:
    serialized_state: bytes
    state_sha256: str
    observer: int
    targets: tuple[CounterfactualRolloutTarget, ...]


def generate_counterfactual_rollouts(
    serialized_state: bytes,
    rules: RuleConfig,
    actor_scores: Sequence[float],
    continuation_policy: RolloutPolicy,
    config: FarmerCoordinationConfig,
    *,
    seed: int,
) -> CounterfactualRolloutBatch:
    """Roll out only Top-N alternatives from one exact farmer decision state."""
    if seed < 0:
        raise ValueError("counterfactual rollout seed must be non-negative")
    root = PyDdzEnv()
    observation = root.restore(serialized_state, rules)
    if observation["role"] != "farmer":
        raise ValueError("counterfactual farmer rollout requires a farmer decision")
    legal_actions = tuple(root.legal_actions())
    if len(actor_scores) != len(legal_actions) or any(
        not math.isfinite(float(score)) for score in actor_scores
    ):
        raise ValueError("counterfactual actor scores must match legal actions and be finite")
    count = min(config.rollout_top_n, len(legal_actions))
    selected = sorted(
        range(len(legal_actions)),
        key=lambda index: (-float(actor_scores[index]), index),
    )[:count]
    targets: list[CounterfactualRolloutTarget] = []
    for branch_index, action_index in enumerate(selected):
        environment = PyDdzEnv()
        environment.restore(serialized_state, rules)
        terminal: StepResult = environment.step(legal_actions[action_index])
        action_count = 1
        while not terminal["terminal"]:
            if action_count >= config.max_rollout_actions:
                raise RuntimeError("counterfactual rollout exceeded its action limit")
            public = environment.observe(environment.current_player)
            actions = tuple(environment.legal_actions())
            branch_seed = (seed + branch_index * 1_000_003 + action_count) & ((1 << 64) - 1)
            choice = continuation_policy.select_action(public, actions, branch_seed)
            if isinstance(choice, bool):
                raise ValueError("rollout policy returned bool instead of an action index")
            try:
                local_index = operator.index(choice)
            except TypeError as error:
                raise ValueError("rollout policy returned a non-integer action index") from error
            if not 0 <= local_index < len(actions):
                raise ValueError("rollout policy action index is outside the legal range")
            terminal = environment.step(actions[local_index])
            action_count += 1
        farmer_down = terminal["objective_payoff"][1]
        farmer_up = terminal["objective_payoff"][2]
        if farmer_down != farmer_up:
            raise RuntimeError("native farmer terminal returns must share one team value")
        canonical = json.dumps(
            legal_actions[action_index], sort_keys=True, separators=(",", ":")
        ).encode()
        targets.append(
            CounterfactualRolloutTarget(
                action_index=action_index,
                canonical_action=canonical,
                actor_score=float(actor_scores[action_index]),
                team_return=float(farmer_down),
                rollout_actions=action_count,
            )
        )
    return CounterfactualRolloutBatch(
        serialized_state=serialized_state,
        state_sha256=hashlib.sha256(serialized_state).hexdigest(),
        observer=observation["observer"],
        targets=tuple(targets),
    )


def select_high_value_farmer_states(
    priority: Tensor,
    state_seat: Tensor,
    config: FarmerCoordinationConfig,
) -> Tensor:
    """Select a deterministic bounded subset for expensive alternative rollouts."""
    if priority.ndim != 1 or not priority.is_floating_point() or not torch.isfinite(priority).all():
        raise ValueError("farmer rollout priority must be finite floating [B]")
    if state_seat.dtype != torch.int64 or state_seat.shape != priority.shape:
        raise ValueError("farmer rollout seats must be int64 [B]")
    if state_seat.device != priority.device or torch.any((state_seat != 1) & (state_seat != 2)):
        raise ValueError("rollout state selection accepts only farmer seats")
    count = min(config.max_rollout_states_per_batch, priority.numel())
    order = sorted(
        range(priority.numel()),
        key=lambda index: (-float(priority[index].item()), index),
    )[:count]
    return torch.tensor(order, dtype=torch.int64, device=priority.device)


@dataclass(frozen=True, slots=True)
class FarmerExploiterSpec:
    target_landlord_policy_id: str
    champion_farmer_policy_id: str
    exploiter_farmer_policy_id: str

    def __post_init__(self) -> None:
        identifiers = (
            self.target_landlord_policy_id,
            self.champion_farmer_policy_id,
            self.exploiter_farmer_policy_id,
        )
        if any(not value for value in identifiers) or len(set(identifiers)) != 3:
            raise ValueError("farmer exploiter policy IDs must be non-empty and distinct")


def generate_farmer_exploiter_schedule(
    deal_set: PairedDealSet,
    spec: FarmerExploiterSpec,
) -> tuple[ScheduledMatch, ...]:
    """Pair champion/exploiter farmer teams against one frozen landlord."""
    matches: list[ScheduledMatch] = []
    for deal in deal_set.deals:
        for label, farmer in (
            ("champion", spec.champion_farmer_policy_id),
            ("exploiter", spec.exploiter_farmer_policy_id),
        ):
            matches.append(
                ScheduledMatch(
                    match_id=f"{deal.deal_id}-farmer-{label}",
                    deal=deal,
                    assignment=SeatAssignment((spec.target_landlord_policy_id, farmer, farmer)),
                )
            )
    return tuple(matches)


@dataclass(frozen=True, slots=True)
class FarmerAcceptanceThresholds:
    minimum_team_win_delta: float = 0.0
    maximum_seat_win_regression: float = 0.02
    require_team_ci_above_threshold: bool = True

    def __post_init__(self) -> None:
        values = (self.minimum_team_win_delta, self.maximum_seat_win_regression)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("farmer acceptance thresholds must be finite")
        if self.maximum_seat_win_regression < 0.0:
            raise ValueError("maximum farmer seat regression must be non-negative")


@dataclass(frozen=True, slots=True)
class FarmerAcceptanceReport:
    passed: bool
    team_win_delta: float
    downstream_win_delta: float
    upstream_win_delta: float
    landlord_parameters_unchanged: bool
    failures: tuple[str, ...]


def evaluate_farmer_acceptance(
    arena_report: ArenaReport,
    *,
    landlord_parameters_unchanged: bool,
    thresholds: FarmerAcceptanceThresholds | None = None,
) -> FarmerAcceptanceReport:
    """Apply the predeclared team, both-seat, and landlord-isolation gate."""
    settings = thresholds if thresholds is not None else FarmerAcceptanceThresholds()
    team = arena_report.farmer_team.win_rate
    downstream = arena_report.landlord_down.win_rate.mean_delta
    upstream = arena_report.landlord_up.win_rate.mean_delta
    failures: list[str] = []
    team_value = (
        team.delta_ci.lower if settings.require_team_ci_above_threshold else team.mean_delta
    )
    if team_value <= settings.minimum_team_win_delta:
        failures.append("farmer team improvement threshold was not reached")
    if downstream < -settings.maximum_seat_win_regression:
        failures.append("downstream farmer regressed beyond threshold")
    if upstream < -settings.maximum_seat_win_regression:
        failures.append("upstream farmer regressed beyond threshold")
    if not landlord_parameters_unchanged:
        failures.append("landlord execution parameters changed")
    return FarmerAcceptanceReport(
        passed=not failures,
        team_win_delta=team.mean_delta,
        downstream_win_delta=downstream,
        upstream_win_delta=upstream,
        landlord_parameters_unchanged=landlord_parameters_unchanged,
        failures=tuple(failures),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"farmer coordination config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"farmer coordination config {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"farmer coordination config {key} must be finite")
    return numeric


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"farmer coordination config {key} must be bool")
    return value


__all__ = (
    "FARMER_COORDINATION_SCHEMA_VERSION",
    "CounterfactualAdvantage",
    "CounterfactualRolloutBatch",
    "CounterfactualRolloutTarget",
    "FarmerAcceptanceReport",
    "FarmerAcceptanceThresholds",
    "FarmerCoordinationConfig",
    "FarmerCoordinationLoss",
    "FarmerExploiterSpec",
    "FarmerSpecialistOptimizer",
    "RolloutPolicy",
    "counterfactual_advantage",
    "evaluate_farmer_acceptance",
    "farmer_coordination_loss",
    "generate_counterfactual_rollouts",
    "generate_farmer_exploiter_schedule",
    "load_farmer_coordination_config",
    "select_high_value_farmer_states",
)
