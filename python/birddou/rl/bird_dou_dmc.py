"""Shared-model Deep Monte Carlo training for BIRD-Dou no-Belief v1."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from birddou import PyDdzEnv, load_rule_config
from birddou.env_types import Action, Observation, RuleConfig
from birddou.eval.arena import Arena
from birddou.eval.baselines import PolicyDecisionContext, SeededRandomPolicy
from birddou.eval.bootstrap import BootstrapConfig
from birddou.eval.metrics import ArenaReport
from birddou.eval.paired_deals import SeatRole, generate_paired_deals, role_for_seat, splitmix64
from birddou.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    RaggedBatch,
    encode_ragged_batch,
    load_feature_config,
)
from birddou.league import LeagueSnapshot, create_self_play_snapshot
from birddou.models.bird_dou import (
    BIRD_DOU_MODEL_SCHEMA_VERSION,
    BirdDouConfig,
    BirdDouModel,
    BirdDouOutput,
    DecisionMode,
    decision_values,
    load_bird_dou_config,
)
from birddou.models.segment_ops import segment_state_index
from birddou.rl.losses import DmcLossName, dmc_value_loss

BIRD_DOU_DMC_CONFIG_SCHEMA_VERSION = 1
BIRD_DOU_DMC_CHECKPOINT_SCHEMA_VERSION = 2
BirdDouTrainerMode = Literal["bird_dou_dmc"]


class BirdDouTrainingError(RuntimeError):
    """Invalid numerical state, transition, or resumable artifact."""


@dataclass(frozen=True, slots=True)
class BirdDouDmcConfig:
    """Versioned single-actor E020 DMC smoke configuration."""

    schema_version: int
    trainer_mode: BirdDouTrainerMode
    rules_path: Path
    model_path: Path
    feature_path: Path
    output_directory: Path
    episodes: int
    master_seed: int
    epsilon: float
    decision_mode: DecisionMode
    learning_rate: float
    weight_decay: float
    loss: DmcLossName
    huber_delta: float
    max_grad_norm: float
    device: str
    amp: bool
    checkpoint_every: int
    decomposition_features: bool
    mc_q_weight: float
    policy_weight: float
    win_weight: float
    score_weight: float
    turns_weight: float
    quantile_weight: float
    evaluation_deals: int
    evaluation_seed: int
    bootstrap_resamples: int

    def __post_init__(self) -> None:
        if self.schema_version != BIRD_DOU_DMC_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported BIRD-Dou DMC config schema")
        if self.trainer_mode != "bird_dou_dmc":
            raise ValueError("E020 requires trainer_mode=bird_dou_dmc")
        if self.episodes <= 0 or self.checkpoint_every <= 0:
            raise ValueError("episodes and checkpoint_every must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in 0..1")
        if self.decision_mode not in ("policy", "wp", "score", "mc_q", "risk"):
            raise ValueError("unknown BIRD-Dou decision mode")
        if self.loss not in ("mse", "huber"):
            raise ValueError("BIRD-Dou DMC loss must be mse or huber")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if self.huber_delta <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("Huber delta and gradient norm must be positive")
        weights = (
            self.mc_q_weight,
            self.policy_weight,
            self.win_weight,
            self.score_weight,
            self.turns_weight,
            self.quantile_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("BIRD-Dou loss weights must be finite and non-negative")
        if self.mc_q_weight <= 0.0:
            raise ValueError("DMC requires a positive mc_q_weight")
        if self.evaluation_deals <= 0 or self.bootstrap_resamples <= 0:
            raise ValueError("evaluation budgets must be positive")
        if not 0 <= self.evaluation_seed < 1 << 64:
            raise ValueError("evaluation_seed must fit uint64")
        if not 0 <= self.master_seed < 1 << 64:
            raise ValueError("master_seed must fit uint64")
        if self.amp and not self.device.startswith("cuda"):
            raise ValueError("AMP is supported only on CUDA devices")

    def fingerprint(self) -> str:
        """Hash training semantics while allowing output and total budget changes."""
        payload = self.to_dict()
        for key in (
            "output_directory",
            "episodes",
            "evaluation_deals",
            "evaluation_seed",
            "bootstrap_resamples",
        ):
            del payload[key]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, object]:
        result = cast(dict[str, object], asdict(self))
        for key in ("rules_path", "model_path", "feature_path", "output_directory"):
            result[key] = str(result[key])
        return result


@dataclass(frozen=True, slots=True)
class BirdDouTransition:
    """One cached information-set feature batch and its terminal supervision."""

    serialized_state: bytes
    seat: int
    role: SeatRole
    batch: RaggedBatch
    chosen_action_index: int
    behavior_logprob: float
    policy_version: int
    target: float
    raw_score: float
    win_target: float
    turns_to_finish: float


@dataclass(frozen=True, slots=True)
class BirdDouEpisode:
    """Complete shared-model self-play trajectory."""

    seed: int
    transitions: tuple[BirdDouTransition, ...]
    action_count: int
    winner_seat: int
    raw_payoff: tuple[int, int, int]
    objective_payoff: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _PendingTransition:
    serialized_state: bytes
    seat: int
    role: SeatRole
    batch: RaggedBatch
    chosen_action_index: int
    behavior_logprob: float
    policy_version: int


@dataclass(frozen=True, slots=True)
class BirdDouLosses:
    """Total loss and individually auditable multi-head components."""

    total: Tensor
    mc_q: Tensor
    policy: Tensor
    win: Tensor
    score: Tensor
    turns: Tensor
    quantile: Tensor

    def detached(self) -> dict[str, float]:
        return {
            key: float(getattr(self, key).detach().cpu().item())
            for key in ("total", "mc_q", "policy", "win", "score", "turns", "quantile")
        }


@dataclass(slots=True)
class BirdDouTrainingState:
    episodes: int = 0
    frames: int = 0
    learner_updates: int = 0
    policy_version: int = 0
    landlord_frames: int = 0
    landlord_down_frames: int = 0
    landlord_up_frames: int = 0


@dataclass(frozen=True, slots=True)
class BirdDouTrainResult:
    state: BirdDouTrainingState
    losses: Mapping[str, float]
    metrics_history: tuple[Mapping[str, object], ...]
    checkpoint_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class BirdDouEvaluation:
    report: ArenaReport
    beats_random: bool


def load_bird_dou_dmc_config(path: Path) -> BirdDouDmcConfig:
    """Load JSON-subset YAML and resolve all repository-relative paths."""
    resolved = path.resolve()
    raw = _object_mapping(json.loads(resolved.read_text(encoding="utf-8")), "config")
    root = resolved.parents[2]
    return BirdDouDmcConfig(
        schema_version=_integer(raw, "schema_version"),
        trainer_mode=cast(BirdDouTrainerMode, _string(raw, "trainer_mode")),
        rules_path=_project_path(root, _string(raw, "rules_path")),
        model_path=_project_path(root, _string(raw, "model_path")),
        feature_path=_project_path(root, _string(raw, "feature_path")),
        output_directory=_project_path(root, _string(raw, "output_directory")),
        episodes=_integer(raw, "episodes"),
        master_seed=_integer(raw, "master_seed"),
        epsilon=_number(raw, "epsilon"),
        decision_mode=cast(DecisionMode, _string(raw, "decision_mode")),
        learning_rate=_number(raw, "learning_rate"),
        weight_decay=_number(raw, "weight_decay"),
        loss=cast(DmcLossName, _string(raw, "loss")),
        huber_delta=_number(raw, "huber_delta"),
        max_grad_norm=_number(raw, "max_grad_norm"),
        device=_string(raw, "device"),
        amp=_boolean(raw, "amp"),
        checkpoint_every=_integer(raw, "checkpoint_every"),
        decomposition_features=_boolean(raw, "decomposition_features"),
        mc_q_weight=_number(raw, "mc_q_weight"),
        policy_weight=_number(raw, "policy_weight"),
        win_weight=_number(raw, "win_weight"),
        score_weight=_number(raw, "score_weight"),
        turns_weight=_number(raw, "turns_weight"),
        quantile_weight=_number(raw, "quantile_weight"),
        evaluation_deals=_integer(raw, "evaluation_deals"),
        evaluation_seed=_integer(raw, "evaluation_seed"),
        bootstrap_resamples=_integer(raw, "bootstrap_resamples"),
    )


def collect_bird_dou_episode(
    seed: int,
    rules: RuleConfig,
    model: BirdDouModel,
    feature_config: FeatureConfig,
    rng: np.random.Generator,
    *,
    epsilon: float,
    policy_version: int,
    decision_mode: DecisionMode = "mc_q",
    device: str = "cpu",
    max_actions: int = 1_000,
) -> BirdDouEpisode:
    """Play one epsilon-greedy game and cache every full legal-action segment."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in 0..1")
    environment = PyDdzEnv()
    environment.reset(seed, rules)
    pending: list[_PendingTransition] = []
    terminal_result: Mapping[str, object] | None = None

    while not environment.terminal:
        if len(pending) >= max_actions:
            raise BirdDouTrainingError(f"actor exceeded {max_actions} actions for seed {seed}")
        seat = environment.current_player
        observation = environment.observe(seat)
        legal_actions = tuple(environment.legal_actions())
        batch = encode_ragged_batch(
            (observation,),
            (legal_actions,),
            rules,
            config=feature_config,
        )
        model.eval()
        with torch.inference_mode():
            output = model(batch.to(device))
            scores = decision_values(output, decision_mode).detach().cpu().numpy()
        if scores.shape != (len(legal_actions),) or not np.isfinite(scores).all():
            raise BirdDouTrainingError("BIRD-Dou actor produced invalid decision scores")
        greedy = int(np.argmax(scores))
        explore = len(legal_actions) > 1 and float(rng.random()) < epsilon
        selected = int(rng.integers(len(legal_actions))) if explore else greedy
        random_probability = epsilon / len(legal_actions)
        probability = random_probability + (1.0 - epsilon if selected == greedy else 0.0)
        pending.append(
            _PendingTransition(
                serialized_state=environment.serialize(),
                seat=seat,
                role=role_for_seat(seat),
                batch=batch,
                chosen_action_index=selected,
                behavior_logprob=math.log(probability),
                policy_version=policy_version,
            )
        )
        terminal_result = cast(Mapping[str, object], environment.step(legal_actions[selected]))

    if terminal_result is None:
        raise BirdDouTrainingError("BIRD-Dou actor produced no terminal transition")
    raw = _payoff_tuple(terminal_result.get("raw_payoff"), "raw_payoff")
    objective = _payoff_tuple(terminal_result.get("objective_payoff"), "objective_payoff")
    event = terminal_result.get("event")
    if not isinstance(event, Mapping) or not isinstance(event.get("actor"), int):
        raise BirdDouTrainingError("terminal event has no winner actor")
    action_count = len(pending)
    transitions = tuple(
        BirdDouTransition(
            serialized_state=item.serialized_state,
            seat=item.seat,
            role=item.role,
            batch=item.batch,
            chosen_action_index=item.chosen_action_index,
            behavior_logprob=item.behavior_logprob,
            policy_version=item.policy_version,
            target=float(objective[item.seat]),
            raw_score=float(raw[item.seat]),
            win_target=float(raw[item.seat] > 0),
            turns_to_finish=float(action_count - index),
        )
        for index, item in enumerate(pending)
    )
    return BirdDouEpisode(
        seed=seed,
        transitions=transitions,
        action_count=action_count,
        winner_seat=cast(int, event["actor"]),
        raw_payoff=raw,
        objective_payoff=objective,
    )


def collate_bird_dou_transitions(
    transitions: Sequence[BirdDouTransition],
) -> RaggedBatch:
    """Pack cached one-state decisions and their chosen rows into one ragged batch."""
    if not transitions:
        raise ValueError("at least one BIRD-Dou transition is required")
    batches = tuple(item.batch for item in transitions)
    if any(batch.batch_size != 1 for batch in batches):
        raise ValueError("cached BIRD-Dou transitions must each contain one state")
    history_length = batches[0].history_rank_counts.shape[1]
    if any(batch.history_rank_counts.shape[1] != history_length for batch in batches):
        raise ValueError("cached BIRD-Dou history lengths must match")
    lengths = [batch.action_count for batch in batches]
    offsets = [0]
    chosen: list[int] = []
    for index, (transition, length) in enumerate(zip(transitions, lengths, strict=True)):
        if not 0 <= transition.chosen_action_index < length:
            raise ValueError(f"transition {index} chosen action lies outside its segment")
        chosen.append(offsets[-1] + transition.chosen_action_index)
        offsets.append(offsets[-1] + length)
    action_offsets = torch.tensor(offsets, dtype=torch.int64)
    return RaggedBatch(
        schema_version=FEATURE_SCHEMA_VERSION,
        rank_categorical=torch.cat([batch.rank_categorical for batch in batches]),
        rank_numeric=torch.cat([batch.rank_numeric for batch in batches]),
        history_rank_counts=torch.cat([batch.history_rank_counts for batch in batches]),
        history_meta=torch.cat([batch.history_meta for batch in batches]),
        history_mask=torch.cat([batch.history_mask for batch in batches]),
        scalars=torch.cat([batch.scalars for batch in batches]),
        action_rank_counts=torch.cat([batch.action_rank_counts for batch in batches]),
        post_hand_counts=torch.cat([batch.post_hand_counts for batch in batches]),
        action_meta=torch.cat([batch.action_meta for batch in batches]),
        action_state_index=segment_state_index(action_offsets),
        action_offsets=action_offsets,
        chosen_action_flat_index=torch.tensor(chosen, dtype=torch.int64),
    )


def bird_dou_dmc_loss(
    output: BirdDouOutput,
    batch: RaggedBatch,
    terminal_target: Tensor,
    raw_score: Tensor,
    win_target: Tensor,
    turns_target: Tensor,
    config: BirdDouDmcConfig,
) -> BirdDouLosses:
    """Train DMC Q plus policy, outcome, conditional score, turn, and quantile heads."""
    chosen = batch.chosen_action_flat_index
    if torch.any(chosen < 0):
        raise ValueError("DMC loss requires one chosen action per state")
    expected_shape = (batch.batch_size,)
    targets = (terminal_target, raw_score, win_target, turns_target)
    if any(target.shape != expected_shape for target in targets):
        raise ValueError("BIRD-Dou DMC targets must use shape [B]")
    if any(not torch.isfinite(target).all() for target in targets):
        raise ValueError("BIRD-Dou DMC targets contain NaN or infinity")
    if chosen.device != output.mc_q.device or any(
        target.device != output.mc_q.device for target in targets
    ):
        raise ValueError("BIRD-Dou DMC output, batch, and targets must share one device")
    if torch.any((win_target < 0.0) | (win_target > 1.0)):
        raise ValueError("BIRD-Dou win targets must be in 0..1")
    if torch.any(turns_target <= 0.0):
        raise ValueError("BIRD-Dou turn targets must be positive")
    terminal_target = terminal_target.float()
    raw_score = raw_score.float()
    win_target = win_target.float()
    turns_target = turns_target.float()
    mc_q = dmc_value_loss(
        output.mc_q[chosen].float(), terminal_target, config.loss, config.huber_delta
    )
    policy = -output.policy_log_probability[chosen].mean()
    win = functional.binary_cross_entropy_with_logits(output.win_logit[chosen].float(), win_target)
    conditional_score = torch.where(
        win_target > 0.5,
        output.score_if_win[chosen].float(),
        output.score_if_loss[chosen].float(),
    )
    score = functional.huber_loss(conditional_score, raw_score, delta=config.huber_delta)
    turns = functional.huber_loss(
        output.turns_to_finish[chosen].float(), turns_target, delta=config.huber_delta
    )
    conditional_quantiles = torch.where(
        (win_target > 0.5).unsqueeze(-1),
        output.score_win_quantiles[chosen].float(),
        output.score_loss_quantiles[chosen].float(),
    )
    quantile = _quantile_huber_loss(conditional_quantiles, raw_score, config.huber_delta)
    total = (
        config.mc_q_weight * mc_q
        + config.policy_weight * policy
        + config.win_weight * win
        + config.score_weight * score
        + config.turns_weight * turns
        + config.quantile_weight * quantile
    )
    losses = BirdDouLosses(total, mc_q, policy, win, score, turns, quantile)
    if any(not torch.isfinite(getattr(losses, key)) for key in losses.__dataclass_fields__):
        raise BirdDouTrainingError("BIRD-Dou multi-head loss is non-finite")
    return losses


class BirdDouPolicy:
    """Arena policy for a shared BIRD-Dou checkpoint."""

    def __init__(
        self,
        policy_id: str,
        model: BirdDouModel,
        rules: RuleConfig,
        feature_config: FeatureConfig,
        *,
        decision_mode: DecisionMode = "mc_q",
        device: str = "cpu",
    ) -> None:
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        self._policy_id = policy_id
        self._model = model
        self._rules = rules
        self._feature_config = feature_config
        self._decision_mode = decision_mode
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
        del context
        batch = encode_ragged_batch(
            (observation,),
            (legal_actions,),
            self._rules,
            config=self._feature_config,
        )
        self._model.eval()
        with torch.inference_mode():
            values = decision_values(
                self._model(batch.to(self._device)), self._decision_mode
            ).detach()
        if values.shape != (len(legal_actions),) or not torch.isfinite(values).all():
            raise BirdDouTrainingError("BIRD-Dou policy produced invalid values")
        return int(torch.argmax(values).cpu().item())


class BirdDouDmcTrainer:
    """Deterministic one-actor shared-model DMC loop with exact resume."""

    def __init__(
        self,
        config: BirdDouDmcConfig,
        *,
        model_config: BirdDouConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise BirdDouTrainingError(f"requested unavailable CUDA device: {config.device}")
        self.config = config
        self.rules = load_rule_config(config.rules_path)
        self.rules_hash = _stable_hash(self.rules)
        self.model_config = (
            load_bird_dou_config(config.model_path) if model_config is None else model_config
        )
        loaded_features = (
            load_feature_config(config.feature_path) if feature_config is None else feature_config
        )
        self.feature_config = replace(
            loaded_features,
            decomposition_features=config.decomposition_features,
        )
        if self.model_config.feature_schema_version != self.feature_config.schema_version:
            raise BirdDouTrainingError("model and feature schema versions differ")
        if self.model_config.history.max_length != self.feature_config.history_max_length:
            raise BirdDouTrainingError("model and feature history lengths differ")
        if (
            self.model_config.action.decomposition_count_cap
            != self.feature_config.min_decompositions_cap
        ):
            raise BirdDouTrainingError("model and feature decomposition caps differ")
        torch.manual_seed(config.master_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.master_seed)
        self.rng = np.random.default_rng(config.master_seed)
        self.model = BirdDouModel(self.model_config).to(config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _step: 1.0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
        self.state = BirdDouTrainingState()
        self.losses: dict[str, float] = {
            key: math.nan
            for key in ("total", "mc_q", "policy", "win", "score", "turns", "quantile")
        }
        self.metrics_history: list[dict[str, object]] = []
        self.league = create_self_play_snapshot(
            "bird-dou:current",
            str(self.checkpoint_path),
            seed=config.master_seed,
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.config.output_directory / "checkpoint.pt"

    @property
    def manifest_path(self) -> Path:
        return self.config.output_directory / "manifest.json"

    def train(self, episodes: int | None = None) -> BirdDouTrainResult:
        remaining = self.config.episodes - self.state.episodes
        count = remaining if episodes is None else episodes
        if count < 0 or count > remaining:
            raise ValueError(f"requested {count} episodes with only {remaining} remaining")
        for _ in range(count):
            episode_seed = splitmix64(self.config.master_seed + self.state.episodes)
            episode = collect_bird_dou_episode(
                episode_seed,
                self.rules,
                self.model,
                self.feature_config,
                self.rng,
                epsilon=self.config.epsilon,
                policy_version=self.state.policy_version,
                decision_mode=self.config.decision_mode,
                device=self.config.device,
            )
            self.losses = self._learn_episode(episode)
            self.state.episodes += 1
            self.state.frames += episode.action_count
            self.state.policy_version += 1
            self.metrics_history.append(
                {
                    "episode": self.state.episodes,
                    "seed": episode.seed,
                    "frames": self.state.frames,
                    "policy_version": self.state.policy_version,
                    "action_count": episode.action_count,
                    "raw_payoff": list(episode.raw_payoff),
                    "objective_payoff": list(episode.objective_payoff),
                    "losses": dict(self.losses),
                }
            )
            if self.state.episodes % self.config.checkpoint_every == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        return BirdDouTrainResult(
            state=BirdDouTrainingState(**asdict(self.state)),
            losses=dict(self.losses),
            metrics_history=tuple(dict(item) for item in self.metrics_history),
            checkpoint_path=self.checkpoint_path,
            manifest_path=self.manifest_path,
        )

    def evaluate_against_random(self) -> BirdDouEvaluation:
        candidate = BirdDouPolicy(
            "bird-dou:current",
            self.model,
            self.rules,
            self.feature_config,
            decision_mode=self.config.decision_mode,
            device=self.config.device,
        )
        baseline = SeededRandomPolicy("baseline:seeded_random", self.config.evaluation_seed)
        run = Arena(self.rules, (candidate, baseline)).evaluate_paired(
            generate_paired_deals(self.config.evaluation_seed, self.config.evaluation_deals),
            candidate.policy_id,
            baseline.policy_id,
            BootstrapConfig(
                resamples=self.config.bootstrap_resamples,
                seed=self.config.evaluation_seed,
            ),
        )
        beats_random = (
            run.report.overall.win_rate.candidate_mean > run.report.overall.win_rate.baseline_mean
        )
        return BirdDouEvaluation(run.report, beats_random)

    def _learn_episode(self, episode: BirdDouEpisode) -> dict[str, float]:
        batch = collate_bird_dou_transitions(episode.transitions).to(self.config.device)
        target = torch.tensor(
            [item.target for item in episode.transitions],
            dtype=torch.float32,
            device=self.config.device,
        )
        raw_score = torch.tensor(
            [item.raw_score for item in episode.transitions],
            dtype=torch.float32,
            device=self.config.device,
        )
        win = torch.tensor(
            [item.win_target for item in episode.transitions],
            dtype=torch.float32,
            device=self.config.device,
        )
        turns = torch.tensor(
            [item.turns_to_finish for item in episode.transitions],
            dtype=torch.float32,
            device=self.config.device,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if self.config.device.startswith("cuda") else "cpu",
            enabled=self.config.amp,
        ):
            output = self.model(batch)
            losses = bird_dou_dmc_loss(
                output,
                batch,
                target,
                raw_score,
                win,
                turns,
                self.config,
            )
        if self.config.amp:
            torch.autograd.backward((self.scaler.scale(losses.total),))
            self.scaler.unscale_(self.optimizer)
        else:
            torch.autograd.backward((losses.total,))
        gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            raise BirdDouTrainingError("BIRD-Dou gradient norm is non-finite")
        if self.config.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.state.learner_updates += 1
        for transition in episode.transitions:
            if transition.role is SeatRole.LANDLORD:
                self.state.landlord_frames += 1
            elif transition.role is SeatRole.LANDLORD_DOWN:
                self.state.landlord_down_frames += 1
            else:
                self.state.landlord_up_frames += 1
        return losses.detached()

    def save_checkpoint(self) -> None:
        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        self.league = self.league.with_runtime_progress(
            checkpoint=str(self.checkpoint_path),
            policy_version=self.state.policy_version,
            schedule_cursor=self.state.episodes,
        )
        league_path = output / "league.json"
        league_file_sha256 = self.league.save(league_path)
        checkpoint = {
            "checkpoint_schema_version": BIRD_DOU_DMC_CHECKPOINT_SCHEMA_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "rules_hash": self.rules_hash,
            "model_fingerprint": self.model_config.fingerprint(),
            "feature_fingerprint": _stable_hash(asdict(self.feature_config)),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_schema_version": BIRD_DOU_MODEL_SCHEMA_VERSION,
            "trainer_mode": "bird_dou_dmc",
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "amp_scaler": self.scaler.state_dict(),
            "state": asdict(self.state),
            "losses": dict(self.losses),
            "metrics_history": list(self.metrics_history),
            "numpy_rng_state": json.dumps(self.rng.bit_generator.state, sort_keys=True),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "training_phase": "bird_dou_no_belief_dmc",
            "league_snapshot": self.league.to_dict(),
        }
        temporary = self.checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(self.checkpoint_path)
        digest = _sha256_file(self.checkpoint_path)
        manifest = {
            "checkpoint_schema_version": BIRD_DOU_DMC_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_file": self.checkpoint_path.name,
            "checkpoint_sha256": digest,
            "git_commit": _git_commit(self.config.rules_path.parents[2]),
            "rules_hash": self.rules_hash,
            "model_fingerprint": self.model_config.fingerprint(),
            "feature_fingerprint": _stable_hash(asdict(self.feature_config)),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_schema_version": BIRD_DOU_MODEL_SCHEMA_VERSION,
            "trainer_mode": "bird_dou_dmc",
            "frames": self.state.frames,
            "episodes": self.state.episodes,
            "learner_updates": self.state.learner_updates,
            "policy_version": self.state.policy_version,
            "optimizer_state": True,
            "scheduler_state": True,
            "amp_scaler_state": True,
            "rng_state": True,
            "training_phase": "bird_dou_no_belief_dmc",
            "league_snapshot": self.league.fingerprint(),
            "league_snapshot_file": league_path.name,
            "league_snapshot_sha256": league_file_sha256,
            "losses": dict(self.losses),
            "metrics_file": "metrics.jsonl",
        }
        temporary_manifest = self.manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(self.manifest_path)
        metrics_path = output / "metrics.jsonl"
        temporary_metrics = metrics_path.with_suffix(".jsonl.tmp")
        temporary_metrics.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in self.metrics_history),
            encoding="utf-8",
        )
        temporary_metrics.replace(metrics_path)
        temporary_weights = (output / "bird_dou.ckpt").with_suffix(".ckpt.tmp")
        torch.save(self.model.state_dict(), temporary_weights)
        temporary_weights.replace(output / "bird_dou.ckpt")

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        path = self.checkpoint_path if checkpoint_path is None else checkpoint_path.resolve()
        checkpoint = _object_mapping(
            torch.load(path, map_location=self.config.device, weights_only=True),
            "checkpoint",
        )
        expected = {
            "checkpoint_schema_version": BIRD_DOU_DMC_CHECKPOINT_SCHEMA_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "rules_hash": self.rules_hash,
            "model_fingerprint": self.model_config.fingerprint(),
            "feature_fingerprint": _stable_hash(asdict(self.feature_config)),
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise BirdDouTrainingError(f"checkpoint {key} mismatch")
        self.model.load_state_dict(_object_mapping(checkpoint.get("model"), "model"), strict=True)
        self.optimizer.load_state_dict(
            dict(_object_mapping(checkpoint.get("optimizer"), "optimizer"))
        )
        self.scheduler.load_state_dict(
            dict(_object_mapping(checkpoint.get("scheduler"), "scheduler"))
        )
        self.scaler.load_state_dict(
            dict(_object_mapping(checkpoint.get("amp_scaler"), "amp_scaler"))
        )
        state = _object_mapping(checkpoint.get("state"), "training state")
        self.state = BirdDouTrainingState(
            episodes=_mapping_integer(state, "episodes"),
            frames=_mapping_integer(state, "frames"),
            learner_updates=_mapping_integer(state, "learner_updates"),
            policy_version=_mapping_integer(state, "policy_version"),
            landlord_frames=_mapping_integer(state, "landlord_frames"),
            landlord_down_frames=_mapping_integer(state, "landlord_down_frames"),
            landlord_up_frames=_mapping_integer(state, "landlord_up_frames"),
        )
        losses = _object_mapping(checkpoint.get("losses"), "losses")
        self.losses = {key: _mapping_number(losses, key) for key in self.losses}
        history = checkpoint.get("metrics_history")
        if not isinstance(history, list):
            raise BirdDouTrainingError("checkpoint metrics_history must be a list")
        self.metrics_history = [
            dict(_object_mapping(item, f"metrics_history[{index}]"))
            for index, item in enumerate(history)
        ]
        if len(self.metrics_history) != self.state.episodes:
            raise BirdDouTrainingError("checkpoint metric count differs from episode count")
        try:
            self.league = LeagueSnapshot.from_dict(checkpoint.get("league_snapshot"))
        except ValueError as error:
            raise BirdDouTrainingError(f"checkpoint league snapshot is invalid: {error}") from error
        if self.league.schedule_cursor != self.state.episodes:
            raise BirdDouTrainingError("checkpoint league cursor differs from episode count")
        if self.league.population.champion.policy_version != self.state.policy_version:
            raise BirdDouTrainingError("checkpoint league champion version differs from learner")
        numpy_state = checkpoint.get("numpy_rng_state")
        if not isinstance(numpy_state, str):
            raise BirdDouTrainingError("checkpoint NumPy RNG state is absent")
        self.rng.bit_generator.state = dict(
            _object_mapping(json.loads(numpy_state), "NumPy RNG state")
        )
        torch_state = checkpoint.get("torch_rng_state")
        if not isinstance(torch_state, Tensor):
            raise BirdDouTrainingError("checkpoint Torch RNG state is absent")
        torch.set_rng_state(torch_state.cpu())
        cuda_states = checkpoint.get("cuda_rng_states")
        if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


def _quantile_huber_loss(prediction: Tensor, target: Tensor, delta: float) -> Tensor:
    count = prediction.shape[1]
    quantiles = (
        torch.arange(count, dtype=prediction.dtype, device=prediction.device) + 0.5
    ) / count
    error = target.unsqueeze(-1) - prediction
    absolute = error.abs()
    huber = torch.where(
        absolute <= delta,
        0.5 * error.square(),
        delta * (absolute - 0.5 * delta),
    )
    weight = torch.abs(quantiles - (error.detach() < 0.0).to(prediction.dtype))
    return (weight * huber / delta).mean()


def _payoff_tuple(value: object, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise BirdDouTrainingError(f"terminal {label} must contain three integers")
    return cast(tuple[int, int, int], tuple(value))


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "uncommitted"


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _object_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise BirdDouTrainingError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config {key} must be a non-empty string")
    return value


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"config {key} must be a finite number")
    return float(value)


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"config {key} must be a boolean")
    return value


def _mapping_integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BirdDouTrainingError(f"checkpoint {key} must be an integer")
    return value


def _mapping_number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BirdDouTrainingError(f"checkpoint {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BirdDouTrainingError(f"checkpoint {key} must be finite")
    return result


__all__ = (
    "BIRD_DOU_DMC_CHECKPOINT_SCHEMA_VERSION",
    "BIRD_DOU_DMC_CONFIG_SCHEMA_VERSION",
    "BirdDouDmcConfig",
    "BirdDouDmcTrainer",
    "BirdDouEpisode",
    "BirdDouEvaluation",
    "BirdDouLosses",
    "BirdDouPolicy",
    "BirdDouTrainResult",
    "BirdDouTrainingError",
    "BirdDouTrainingState",
    "BirdDouTransition",
    "bird_dou_dmc_loss",
    "collate_bird_dou_transitions",
    "collect_bird_dou_episode",
    "load_bird_dou_dmc_config",
)
