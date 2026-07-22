"""Resumable metric-gated joint Bid Head and Cardplay training on complete games."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from birddou import PyDdzEnv, RuleConfig, load_rule_config
from birddou.belief.data import concatenate_ragged_batches
from birddou.eval.baselines import FirstLegalPolicy, FixedBidPolicy, LongestMovePolicy, Policy
from birddou.eval.paired_deals import splitmix64
from birddou.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    encode_ragged_batch,
    load_feature_config,
)
from birddou.league import LeagueSnapshot, create_self_play_snapshot
from birddou.models.bid_head import (
    BID_HEAD_ARCHITECTURE,
    BID_HEAD_SCHEMA_VERSION,
    BidHead,
    encode_bid_batch,
    load_bid_head_config,
)
from birddou.models.bird_dou import (
    BIRD_DOU_ARCHITECTURE,
    BIRD_DOU_MODEL_SCHEMA_VERSION,
    BirdDouModel,
    load_bird_dou_config,
)
from birddou.rl.bidding import (
    BiddingDistributionMonitor,
    BiddingEpisodeSummary,
    BiddingStage,
    BidHeadPolicy,
    CompleteEpisode,
    CurriculumMetrics,
    WinScoreCurriculum,
    bid_supervised_loss,
    binary_calibration_error,
    build_joint_bid_batch,
    collect_complete_episode,
    combine_joint_training_loss,
    generate_initial_bid_mc_labels,
    joint_bid_loss,
    load_bidding_training_config,
    sample_initial_bid_deals,
    set_cardplay_frozen,
)
from birddou.rl.bird_dou_dmc import (
    BirdDouDmcConfig,
    BirdDouPolicy,
    bird_dou_dmc_loss,
    load_bird_dou_dmc_config,
)

FULL_GAME_CONFIG_SCHEMA_VERSION = 2
FULL_GAME_CHECKPOINT_SCHEMA_VERSION = 3


class FullGameTrainingError(RuntimeError):
    """Complete-game training or restoration violated a hard invariant."""


@dataclass(frozen=True, slots=True)
class FullGameConfig:
    """All semantic inputs for deterministic staged complete-game training."""

    schema_version: int
    trainer_mode: str
    rules_path: Path
    bid_model_path: Path
    cardplay_model_path: Path
    feature_path: Path
    bidding_training_path: Path
    cardplay_training_path: Path
    output_directory: Path
    episodes: int
    master_seed: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    device: str
    amp: bool
    checkpoint_every: int
    decomposition_features: bool
    bid_pretraining_batches: int
    bid_pretraining_hidden_samples: int
    initial_fixed_bid_score: int
    fixed_bid_warmup_episodes: int
    maximum_redeals: int

    def __post_init__(self) -> None:
        if self.schema_version != FULL_GAME_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported full-game config schema")
        if self.trainer_mode != "full_game_joint":
            raise ValueError("full-game trainer_mode must be full_game_joint")
        if self.episodes <= 0 or self.checkpoint_every <= 0:
            raise ValueError("full-game episodes and checkpoint interval must be positive")
        if not 0 <= self.master_seed < 1 << 64:
            raise ValueError("full-game master_seed must fit uint64")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("full-game optimizer settings are invalid")
        if self.max_grad_norm <= 0.0 or not math.isfinite(self.max_grad_norm):
            raise ValueError("full-game max_grad_norm must be finite and positive")
        if not 1 <= self.initial_fixed_bid_score <= 3:
            raise ValueError("initial_fixed_bid_score must be in 1..3")
        if self.fixed_bid_warmup_episodes < 0:
            raise ValueError("fixed_bid_warmup_episodes must be non-negative")
        if self.bid_pretraining_batches < 0 or self.bid_pretraining_hidden_samples <= 0:
            raise ValueError("full-game bid pretraining budget is invalid")
        if self.maximum_redeals < 0:
            raise ValueError("maximum_redeals must be non-negative")
        if self.amp and not self.device.startswith("cuda"):
            raise ValueError("full-game AMP requires a CUDA device")

    def to_dict(self) -> dict[str, object]:
        result = cast(dict[str, object], asdict(self))
        for key in (
            "rules_path",
            "bid_model_path",
            "cardplay_model_path",
            "feature_path",
            "bidding_training_path",
            "cardplay_training_path",
            "output_directory",
        ):
            result[key] = str(result[key])
        return result

    def fingerprint(self) -> str:
        payload = self.to_dict()
        del payload["episodes"]
        del payload["output_directory"]
        return _stable_hash(payload)


@dataclass(slots=True)
class FullGameTrainingState:
    """Checkpointed counters and current metric-gated curriculum stage."""

    episodes: int = 0
    frames: int = 0
    learner_updates: int = 0
    bid_pretraining_updates: int = 0
    policy_version: int = 0
    redeals: int = 0
    stage: str = BiddingStage.BID_WIN_FROZEN.value


@dataclass(frozen=True, slots=True)
class FullGameTrainResult:
    """Final state and atomic artifact paths."""

    state: FullGameTrainingState
    losses: Mapping[str, float]
    metrics_history: tuple[Mapping[str, object], ...]
    checkpoint_path: Path
    manifest_path: Path


def load_full_game_config(path: Path) -> FullGameConfig:
    """Load JSON-subset YAML and resolve every project-relative path."""
    resolved = path.resolve()
    values = _mapping(json.loads(resolved.read_text(encoding="utf-8")), "full-game config")
    root = resolved.parents[2]
    return FullGameConfig(
        schema_version=_integer(values, "schema_version"),
        trainer_mode=_string(values, "trainer_mode"),
        rules_path=_project_path(root, _string(values, "rules_path")),
        bid_model_path=_project_path(root, _string(values, "bid_model_path")),
        cardplay_model_path=_project_path(root, _string(values, "cardplay_model_path")),
        feature_path=_project_path(root, _string(values, "feature_path")),
        bidding_training_path=_project_path(root, _string(values, "bidding_training_path")),
        cardplay_training_path=_project_path(root, _string(values, "cardplay_training_path")),
        output_directory=_project_path(root, _string(values, "output_directory")),
        episodes=_integer(values, "episodes"),
        master_seed=_integer(values, "master_seed"),
        learning_rate=_number(values, "learning_rate"),
        weight_decay=_number(values, "weight_decay"),
        max_grad_norm=_number(values, "max_grad_norm"),
        device=_string(values, "device"),
        amp=_boolean(values, "amp"),
        checkpoint_every=_integer(values, "checkpoint_every"),
        decomposition_features=_boolean(values, "decomposition_features"),
        bid_pretraining_batches=_integer(values, "bid_pretraining_batches"),
        bid_pretraining_hidden_samples=_integer(values, "bid_pretraining_hidden_samples"),
        initial_fixed_bid_score=_integer(values, "initial_fixed_bid_score"),
        fixed_bid_warmup_episodes=_integer(values, "fixed_bid_warmup_episodes"),
        maximum_redeals=_integer(values, "maximum_redeals"),
    )


class FullGameTrainer:
    """Train bidding first, then jointly unfreeze Cardplay only through metric gates."""

    def __init__(self, config: FullGameConfig) -> None:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise FullGameTrainingError(f"requested unavailable CUDA device: {config.device}")
        self.config = config
        self.rules: RuleConfig = load_rule_config(config.rules_path)
        if self.rules["profile"] != "canonical_full":
            raise FullGameTrainingError("full-game trainer requires canonical_full rules")
        self.rules_hash = _stable_hash(self.rules)
        self.bidding_config = load_bidding_training_config(config.bidding_training_path)
        self.cardplay_training: BirdDouDmcConfig = load_bird_dou_dmc_config(
            config.cardplay_training_path
        )
        self.feature_config: FeatureConfig = replace(
            load_feature_config(config.feature_path),
            decomposition_features=config.decomposition_features,
        )
        torch.manual_seed(config.master_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.master_seed)
        self.rng = np.random.default_rng(config.master_seed)
        self.bid_model_config = load_bid_head_config(config.bid_model_path)
        self.cardplay_model_config = load_bird_dou_config(config.cardplay_model_path)
        if self.cardplay_model_config.feature_schema_version != self.feature_config.schema_version:
            raise FullGameTrainingError("cardplay model and feature schema versions differ")
        if self.cardplay_model_config.history.max_length != self.feature_config.history_max_length:
            raise FullGameTrainingError("cardplay model and feature history lengths differ")
        if (
            self.cardplay_model_config.action.decomposition_count_cap
            != self.feature_config.min_decompositions_cap
        ):
            raise FullGameTrainingError("cardplay model and feature decomposition caps differ")
        self.bid_model = BidHead(self.bid_model_config).to(config.device)
        self.cardplay_model = BirdDouModel(self.cardplay_model_config).to(config.device)
        parameters = (*self.bid_model.parameters(), *self.cardplay_model.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _step: 1.0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
        self.curriculum = WinScoreCurriculum(
            self.bidding_config.curriculum,
            self.bidding_config.loss.score_weight,
        )
        self.monitor = BiddingDistributionMonitor(
            max(100_000, self.bidding_config.curriculum.min_complete_games)
        )
        self._summaries: deque[BiddingEpisodeSummary] = deque(
            maxlen=max(100_000, self.bidding_config.curriculum.min_complete_games)
        )
        self._calibration_probabilities: deque[float] = deque(maxlen=100_000)
        self._calibration_targets: deque[bool] = deque(maxlen=100_000)
        self.state = FullGameTrainingState()
        self.losses = {"total": math.nan, "bid": math.nan, "cardplay": math.nan}
        self.metrics_history: list[dict[str, object]] = []
        self.pretraining_history: list[dict[str, object]] = []
        self.league = create_self_play_snapshot(
            "bird-dou:full-game-current",
            str(self.checkpoint_path),
            seed=config.master_seed,
        )
        set_cardplay_frozen(self.cardplay_model, True)

    @property
    def checkpoint_path(self) -> Path:
        return self.config.output_directory / "checkpoint.pt"

    @property
    def manifest_path(self) -> Path:
        return self.config.output_directory / "manifest.json"

    def train(self, episodes: int | None = None) -> FullGameTrainResult:
        """Run the remaining deterministic complete-game budget and checkpoint it."""
        self._complete_bid_pretraining()
        remaining = self.config.episodes - self.state.episodes
        count = remaining if episodes is None else episodes
        if count < 0 or count > remaining:
            raise ValueError(f"requested {count} episodes with only {remaining} remaining")
        for _ in range(count):
            episode, redeals = self._collect_resolved_episode()
            self.losses, calibration_probability, calibration_targets = self._learn(episode)
            self._record_calibration(calibration_probability, calibration_targets)
            summary = self._summary(episode, redeals)
            self.monitor.add(summary)
            self._summaries.append(summary)
            self.state.episodes += 1
            self.state.frames += episode.action_count
            self.state.learner_updates += 1
            self.state.policy_version += 1
            self.state.redeals += redeals
            distribution = self.monitor.report(
                self.bidding_config.curriculum.min_call_rate,
                self.bidding_config.curriculum.max_call_rate,
            )
            calibration = binary_calibration_error(
                tuple(self._calibration_probabilities),
                tuple(self._calibration_targets),
            )
            advanced = self.curriculum.maybe_advance(
                CurriculumMetrics(
                    game_count=distribution.game_count,
                    calibration_error=calibration,
                    call_rate=distribution.call_rate,
                    redeal_rate=distribution.redeal_rate,
                )
            )
            set_cardplay_frozen(self.cardplay_model, self.curriculum.state.cardplay_frozen)
            self.state.stage = self.curriculum.state.stage.value
            self.metrics_history.append(
                {
                    "episode": self.state.episodes,
                    "seed": episode.seed,
                    "frames": self.state.frames,
                    "policy_version": self.state.policy_version,
                    "stage": self.state.stage,
                    "stage_advanced": advanced,
                    "redeals": redeals,
                    "action_count": episode.action_count,
                    "winning_bid": episode.winning_bid,
                    "terminal_payoff": list(episode.terminal_payoff),
                    "calibration_error": calibration,
                    "call_rate": distribution.call_rate,
                    "redeal_rate": distribution.redeal_rate,
                    "losses": dict(self.losses),
                }
            )
            if self.state.episodes % self.config.checkpoint_every == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        return FullGameTrainResult(
            state=FullGameTrainingState(**asdict(self.state)),
            losses=dict(self.losses),
            metrics_history=tuple(dict(item) for item in self.metrics_history),
            checkpoint_path=self.checkpoint_path,
            manifest_path=self.manifest_path,
        )

    def _complete_bid_pretraining(self) -> None:
        """Finish resumable privileged MC initialization before joint episodes."""
        while self.state.bid_pretraining_updates < self.config.bid_pretraining_batches:
            update = self.state.bid_pretraining_updates
            seed = splitmix64((self.config.master_seed + 0xB1D00000 + update) & ((1 << 64) - 1))
            samples = sample_initial_bid_deals(
                seed,
                self.rules,
                self.config.bid_pretraining_hidden_samples,
            )
            labels = generate_initial_bid_mc_labels(
                samples,
                self.rules,
                LongestMovePolicy("bid-mc-frozen-cardplay"),
                self.bidding_config.monte_carlo,
            )
            reference = samples[0]
            environment = PyDdzEnv()
            observation = environment.reset_complete_deal(
                [list(hand) for hand in reference.hands],
                list(reference.bottom_cards),
                reference.first_bidder,
                self.rules,
            )
            legal_actions = tuple(environment.legal_actions())
            batch = encode_bid_batch(
                (observation,),
                (legal_actions,),
                self.rules,
                history_max_length=self.bid_model_config.history_max_length,
            ).to(self.config.device)
            self.bid_model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda" if self.config.device.startswith("cuda") else "cpu",
                enabled=self.config.amp,
            ):
                output = self.bid_model(batch)
                loss = bid_supervised_loss(
                    output,
                    labels,
                    batch.action_offsets,
                    self.bidding_config.loss,
                )
            if not torch.isfinite(loss.total):
                raise FullGameTrainingError("Bid Head MC pretraining loss is non-finite")
            if self.config.amp:
                torch.autograd.backward((self.scaler.scale(loss.total),))
                self.scaler.unscale_(self.optimizer)
            else:
                torch.autograd.backward((loss.total,))
            gradient_norm = nn.utils.clip_grad_norm_(
                self.bid_model.parameters(),
                self.config.max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FullGameTrainingError("Bid Head MC pretraining gradient is non-finite")
            if self.config.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.state.bid_pretraining_updates += 1
            self.state.learner_updates += 1
            self.state.policy_version += 1
            self.losses = {
                "total": float(loss.total.detach().cpu().item()),
                "bid": float(loss.total.detach().cpu().item()),
                "cardplay": 0.0,
            }
            self.pretraining_history.append(
                {
                    "update": self.state.bid_pretraining_updates,
                    "seed": seed,
                    "hidden_samples": self.config.bid_pretraining_hidden_samples,
                    "policy_version": self.state.policy_version,
                    "loss": self.losses["bid"],
                }
            )
            self.save_checkpoint()

    def _collect_resolved_episode(self) -> tuple[CompleteEpisode, int]:
        base_seed = splitmix64(self.config.master_seed + self.state.episodes)
        cardplay_policy: Policy
        if self.curriculum.state.cardplay_frozen:
            cardplay_policy = LongestMovePolicy("frozen-cardplay-baseline")
        else:
            cardplay_policy = BirdDouPolicy(
                "full-game-cardplay",
                self.cardplay_model,
                self.rules,
                self.feature_config,
                device=self.config.device,
            )
        if self.state.episodes < self.config.fixed_bid_warmup_episodes:
            bidding_policy: Policy = FixedBidPolicy(
                "fixed-bid-initializer",
                FirstLegalPolicy("unused-cardplay"),
                score_bid=self.config.initial_fixed_bid_score,
            )
        else:
            bidding_policy = BidHeadPolicy(
                "learned-bid",
                self.bid_model,
                self.rules,
                self.config.device,
            )
        for redeals in range(self.config.maximum_redeals + 1):
            active_seed = splitmix64((base_seed + redeals) & ((1 << 64) - 1))
            episode = collect_complete_episode(
                active_seed,
                self.rules,
                bidding_policy,
                cardplay_policy,
                max_actions=self.bidding_config.monte_carlo.max_actions,
            )
            if not episode.all_pass:
                return episode, redeals
        raise FullGameTrainingError("full-game collector exceeded maximum_redeals")

    def _learn(
        self,
        episode: CompleteEpisode,
    ) -> tuple[dict[str, float], tuple[float, ...], tuple[bool, ...]]:
        joint = build_joint_bid_batch(episode, self.rules)
        bid_batch = joint.batch.to(self.config.device)
        chosen = joint.chosen_flat_index.to(self.config.device)
        terminal_win = joint.terminal_win.to(self.config.device)
        terminal_score = joint.terminal_score.to(self.config.device)
        self.bid_model.train()
        self.cardplay_model.train(not self.curriculum.state.cardplay_frozen)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if self.config.device.startswith("cuda") else "cpu",
            enabled=self.config.amp,
        ):
            bid_output = self.bid_model(bid_batch)
            stage_loss = replace(
                self.bidding_config.loss,
                score_weight=self.curriculum.state.score_weight,
            )
            bid_loss = joint_bid_loss(
                bid_output,
                chosen,
                terminal_win,
                terminal_score,
                stage_loss,
            )
            cardplay_loss = self._cardplay_loss(episode)
            combined = combine_joint_training_loss(
                bid_loss,
                cardplay_loss,
                self.curriculum.state,
            )
        if not torch.isfinite(combined.total):
            raise FullGameTrainingError("full-game loss is non-finite")
        if self.config.amp:
            torch.autograd.backward((self.scaler.scale(combined.total),))
            self.scaler.unscale_(self.optimizer)
        else:
            torch.autograd.backward((combined.total,))
        trainable = tuple(
            parameter
            for parameter in (*self.bid_model.parameters(), *self.cardplay_model.parameters())
            if parameter.requires_grad
        )
        gradient_norm = nn.utils.clip_grad_norm_(trainable, self.config.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            raise FullGameTrainingError("full-game gradient norm is non-finite")
        if self.config.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        probability = torch.sigmoid(bid_output.win_logit[chosen]).detach().cpu().tolist()
        target = terminal_win.detach().cpu().tolist()
        losses = {
            "total": float(combined.total.detach().cpu().item()),
            "bid": float(combined.bid.detach().cpu().item()),
            "cardplay": float(combined.cardplay.detach().cpu().item()),
        }
        return (
            losses,
            tuple(float(value) for value in probability),
            tuple(value > 0.5 for value in target),
        )

    def _cardplay_loss(self, episode: CompleteEpisode) -> Tensor:
        if self.curriculum.state.cardplay_frozen or not episode.cardplay:
            return torch.zeros((), dtype=torch.float32, device=self.config.device)
        batches = []
        for decision in episode.cardplay:
            one = encode_ragged_batch(
                (decision.observation,),
                (decision.legal_actions,),
                self.rules,
                config=self.feature_config,
            )
            batches.append(
                replace(
                    one,
                    chosen_action_flat_index=torch.tensor(
                        [decision.selected_index], dtype=torch.int64
                    ),
                )
            )
        batch = concatenate_ragged_batches(batches).to(self.config.device)
        payoff = episode.terminal_payoff
        raw = torch.tensor(
            [float(payoff[decision.observation["observer"]]) for decision in episode.cardplay],
            dtype=torch.float32,
            device=self.config.device,
        )
        win = (raw > 0.0).to(torch.float32)
        turns = torch.arange(
            len(episode.cardplay),
            0,
            -1,
            dtype=torch.float32,
            device=self.config.device,
        )
        output = self.cardplay_model(batch)
        return bird_dou_dmc_loss(
            output,
            batch,
            raw,
            raw,
            win,
            turns,
            self.cardplay_training,
        ).total

    def _record_calibration(
        self,
        probabilities: tuple[float, ...],
        targets: tuple[bool, ...],
    ) -> None:
        self._calibration_probabilities.extend(probabilities)
        self._calibration_targets.extend(targets)

    def _summary(self, episode: CompleteEpisode, redeals: int) -> BiddingEpisodeSummary:
        if episode.landlord is None:
            raise FullGameTrainingError("resolved full-game episode has no landlord")
        positive = 0
        for decision in episode.bidding:
            action = decision.legal_actions[decision.selected_index]
            bid = action.get("bid")
            positive += int(bid != "pass")
        payoff = episode.terminal_payoff[episode.landlord]
        return BiddingEpisodeSummary(
            landlord_strength=episode.landlord_strength,
            winning_bid=episode.winning_bid,
            redeal_count=redeals,
            bid_action_count=len(episode.bidding),
            positive_bid_count=positive,
            landlord_won=payoff > 0,
            landlord_score=float(payoff),
        )

    def save_checkpoint(self) -> None:
        """Atomically save full training/RNG/curriculum/League state and manifest."""
        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        self.league = self.league.with_runtime_progress(
            checkpoint=str(self.checkpoint_path),
            policy_version=self.state.policy_version,
            schedule_cursor=self.state.episodes,
        )
        league_path = output / "league.json"
        league_sha = self.league.save(league_path)
        checkpoint: dict[str, object] = {
            "checkpoint_schema_version": FULL_GAME_CHECKPOINT_SCHEMA_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "rules_hash": self.rules_hash,
            "feature_fingerprint": _stable_hash(asdict(self.feature_config)),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "bid_model_fingerprint": self.bid_model_config.fingerprint(),
            "bid_model_schema_version": BID_HEAD_SCHEMA_VERSION,
            "bid_model_architecture": BID_HEAD_ARCHITECTURE,
            "cardplay_model_fingerprint": self.cardplay_model_config.fingerprint(),
            "cardplay_model_schema_version": BIRD_DOU_MODEL_SCHEMA_VERSION,
            "cardplay_model_architecture": BIRD_DOU_ARCHITECTURE,
            "trainer_mode": self.config.trainer_mode,
            "bid_model": self.bid_model.state_dict(),
            "cardplay_model": self.cardplay_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "amp_scaler": self.scaler.state_dict(),
            "state": asdict(self.state),
            "losses": dict(self.losses),
            "pretraining_history": list(self.pretraining_history),
            "metrics_history": list(self.metrics_history),
            "summaries": [asdict(item) for item in self._summaries],
            "calibration_probabilities": list(self._calibration_probabilities),
            "calibration_targets": list(self._calibration_targets),
            "numpy_rng_state": json.dumps(self.rng.bit_generator.state, sort_keys=True),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "league_snapshot": self.league.to_dict(),
        }
        temporary = self.checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(self.checkpoint_path)
        checkpoint_sha = _sha256_file(self.checkpoint_path)
        manifest = {
            "checkpoint_schema_version": FULL_GAME_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_file": self.checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "git_commit": _git_commit(self.config.rules_path.parents[2]),
            "rules_hash": self.rules_hash,
            "feature_fingerprint": _stable_hash(asdict(self.feature_config)),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_arch_version": f"{BID_HEAD_ARCHITECTURE}+{BIRD_DOU_ARCHITECTURE}",
            "bid_model_fingerprint": self.bid_model_config.fingerprint(),
            "bid_model_schema_version": BID_HEAD_SCHEMA_VERSION,
            "bid_model_architecture": BID_HEAD_ARCHITECTURE,
            "cardplay_model_fingerprint": self.cardplay_model_config.fingerprint(),
            "cardplay_model_schema_version": BIRD_DOU_MODEL_SCHEMA_VERSION,
            "cardplay_model_architecture": BIRD_DOU_ARCHITECTURE,
            "trainer_mode": self.config.trainer_mode,
            "frames": self.state.frames,
            "episodes": self.state.episodes,
            "learner_updates": self.state.learner_updates,
            "bid_pretraining_updates": self.state.bid_pretraining_updates,
            "policy_version": self.state.policy_version,
            "training_phase": self.state.stage,
            "optimizer_state": True,
            "scheduler_state": True,
            "amp_scaler_state": True,
            "rng_state": True,
            "league_snapshot": self.league.fingerprint(),
            "league_snapshot_file": league_path.name,
            "league_snapshot_sha256": league_sha,
            "metrics_file": "metrics.jsonl",
            "pretraining_metrics_file": "bid_pretraining_metrics.jsonl",
            "losses": dict(self.losses),
        }
        _atomic_text(self.manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _atomic_text(
            output / "metrics.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.metrics_history),
        )
        _atomic_text(
            output / "bid_pretraining_metrics.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.pretraining_history),
        )
        for name, model in (
            ("bid_head.ckpt", self.bid_model),
            ("cardplay.ckpt", self.cardplay_model),
        ):
            weight_path = output / name
            temporary_weight = weight_path.with_suffix(weight_path.suffix + ".tmp")
            torch.save(model.state_dict(), temporary_weight)
            temporary_weight.replace(weight_path)

    def load_checkpoint(self, path: Path | None = None) -> None:
        """Restore exact joint-training state and reject semantic drift."""
        checkpoint_path = self.checkpoint_path if path is None else path.resolve()
        checkpoint = _mapping(
            torch.load(checkpoint_path, map_location=self.config.device, weights_only=True),
            "full-game checkpoint",
        )
        expected = {
            "checkpoint_schema_version": FULL_GAME_CHECKPOINT_SCHEMA_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "rules_hash": self.rules_hash,
            "feature_fingerprint": _stable_hash(asdict(self.feature_config)),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "bid_model_fingerprint": self.bid_model_config.fingerprint(),
            "bid_model_schema_version": BID_HEAD_SCHEMA_VERSION,
            "bid_model_architecture": BID_HEAD_ARCHITECTURE,
            "cardplay_model_fingerprint": self.cardplay_model_config.fingerprint(),
            "cardplay_model_schema_version": BIRD_DOU_MODEL_SCHEMA_VERSION,
            "cardplay_model_architecture": BIRD_DOU_ARCHITECTURE,
            "trainer_mode": self.config.trainer_mode,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise FullGameTrainingError(f"checkpoint {key} mismatch")
        self.bid_model.load_state_dict(_mapping(checkpoint.get("bid_model"), "bid_model"))
        self.cardplay_model.load_state_dict(
            _mapping(checkpoint.get("cardplay_model"), "cardplay_model")
        )
        self.optimizer.load_state_dict(dict(_mapping(checkpoint.get("optimizer"), "optimizer")))
        self.scheduler.load_state_dict(dict(_mapping(checkpoint.get("scheduler"), "scheduler")))
        self.scaler.load_state_dict(dict(_mapping(checkpoint.get("amp_scaler"), "amp_scaler")))
        state = _mapping(checkpoint.get("state"), "training state")
        self.state = FullGameTrainingState(
            episodes=_integer(state, "episodes"),
            frames=_integer(state, "frames"),
            learner_updates=_integer(state, "learner_updates"),
            bid_pretraining_updates=_integer(state, "bid_pretraining_updates"),
            policy_version=_integer(state, "policy_version"),
            redeals=_integer(state, "redeals"),
            stage=_string(state, "stage"),
        )
        try:
            stage = BiddingStage(self.state.stage)
        except ValueError as error:
            raise FullGameTrainingError("checkpoint curriculum stage is invalid") from error
        self.curriculum.restore(stage)
        set_cardplay_frozen(self.cardplay_model, self.curriculum.state.cardplay_frozen)
        losses = _mapping(checkpoint.get("losses"), "losses")
        self.losses = {key: _number(losses, key) for key in self.losses}
        history = checkpoint.get("metrics_history")
        if not isinstance(history, list):
            raise FullGameTrainingError("checkpoint metrics_history must be a list")
        self.metrics_history = [dict(_mapping(row, "metric row")) for row in history]
        if len(self.metrics_history) != self.state.episodes:
            raise FullGameTrainingError("checkpoint metric count differs from episode count")
        pretraining = checkpoint.get("pretraining_history")
        if not isinstance(pretraining, list):
            raise FullGameTrainingError("checkpoint pretraining_history must be a list")
        self.pretraining_history = [
            dict(_mapping(row, "pretraining metric row")) for row in pretraining
        ]
        if len(self.pretraining_history) != self.state.bid_pretraining_updates:
            raise FullGameTrainingError(
                "checkpoint pretraining metric count differs from update count"
            )
        self._restore_monitor(checkpoint)
        numpy_state = checkpoint.get("numpy_rng_state")
        if not isinstance(numpy_state, str):
            raise FullGameTrainingError("checkpoint NumPy RNG state is missing")
        self.rng.bit_generator.state = dict(_mapping(json.loads(numpy_state), "NumPy RNG state"))
        torch_state = checkpoint.get("torch_rng_state")
        if not isinstance(torch_state, Tensor):
            raise FullGameTrainingError("checkpoint Torch RNG state is missing")
        torch.set_rng_state(torch_state.cpu())
        cuda_states = checkpoint.get("cuda_rng_states")
        if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        try:
            self.league = LeagueSnapshot.from_dict(checkpoint.get("league_snapshot"))
        except ValueError as error:
            raise FullGameTrainingError(
                f"checkpoint League snapshot is invalid: {error}"
            ) from error
        if self.league.schedule_cursor != self.state.episodes:
            raise FullGameTrainingError("checkpoint League cursor differs from episode count")
        if self.league.population.champion.policy_version != self.state.policy_version:
            raise FullGameTrainingError("checkpoint League policy version differs from learner")

    def _restore_monitor(self, checkpoint: Mapping[str, object]) -> None:
        summaries = checkpoint.get("summaries")
        probabilities = checkpoint.get("calibration_probabilities")
        targets = checkpoint.get("calibration_targets")
        if (
            not isinstance(summaries, list)
            or not isinstance(probabilities, list)
            or not isinstance(targets, list)
        ):
            raise FullGameTrainingError("checkpoint bidding monitor state is missing")
        self.monitor = BiddingDistributionMonitor(
            max(100_000, self.bidding_config.curriculum.min_complete_games)
        )
        self._summaries.clear()
        for raw in summaries:
            row = _mapping(raw, "bidding summary")
            summary = BiddingEpisodeSummary(
                landlord_strength=_number(row, "landlord_strength"),
                winning_bid=_integer(row, "winning_bid"),
                redeal_count=_integer(row, "redeal_count"),
                bid_action_count=_integer(row, "bid_action_count"),
                positive_bid_count=_integer(row, "positive_bid_count"),
                landlord_won=_boolean(row, "landlord_won"),
                landlord_score=_number(row, "landlord_score"),
            )
            self.monitor.add(summary)
            self._summaries.append(summary)
        self._calibration_probabilities = deque(
            (_number_value(value, "calibration probability") for value in probabilities),
            maxlen=100_000,
        )
        self._calibration_targets = deque(
            (_bool_value(value, "calibration target") for value in targets),
            maxlen=100_000,
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    return _number_value(values.get(key), key)


def _number_value(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _boolean(values: Mapping[str, object], key: str) -> bool:
    return _bool_value(values.get(key), key)


def _bool_value(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


__all__ = (
    "FULL_GAME_CHECKPOINT_SCHEMA_VERSION",
    "FULL_GAME_CONFIG_SCHEMA_VERSION",
    "FullGameConfig",
    "FullGameTrainResult",
    "FullGameTrainer",
    "FullGameTrainingError",
    "FullGameTrainingState",
    "load_full_game_config",
)
