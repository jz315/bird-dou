"""Minimal deterministic Deep Monte Carlo actor-learner and checkpoint loop."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn

from birddou import load_rule_config
from birddou.actors.actor_worker import ActorModel, DmcEpisode, DmcTransition, collect_dmc_episode
from birddou.env_types import Action, Observation, RuleConfig
from birddou.eval.arena import Arena
from birddou.eval.baselines import PolicyDecisionContext, SeededRandomPolicy
from birddou.eval.bootstrap import BootstrapConfig
from birddou.eval.metrics import ArenaReport
from birddou.eval.paired_deals import SEAT_ROLES, SeatRole, generate_paired_deals, splitmix64
from birddou.features import DOUZERO_FEATURE_SCHEMA_VERSION, encode_douzero_features
from birddou.league import LeagueSnapshot, create_self_play_snapshot
from birddou.models.baseline_douzero import load_official_checkpoint_set
from birddou.models.douzero_model import (
    DOUZERO_MODEL_SCHEMA_VERSION,
    create_douzero_model,
)
from birddou.rl.losses import DmcLossName, dmc_value_loss

DMC_CONFIG_SCHEMA_VERSION = 1
DMC_CHECKPOINT_SCHEMA_VERSION = 2
TrainerMode = Literal["dmc"]
Initialization = Literal["random", "douzero_ADP", "douzero_WP"]


class DmcTrainingError(RuntimeError):
    """Invalid numerical state, artifact, or training transition."""


class TrainableModel(ActorModel, Protocol):
    """Actor model plus the state and gradient surfaces used by the learner."""

    def train(self, mode: bool = True) -> TrainableModel: ...

    def to(self, device: str) -> TrainableModel: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def state_dict(self) -> Mapping[str, Tensor]: ...

    def load_state_dict(self, state_dict: Mapping[str, object], strict: bool = True) -> object: ...


@dataclass(frozen=True, slots=True)
class DmcConfig:
    """Versioned E015 smoke-training configuration."""

    schema_version: int
    trainer_mode: TrainerMode
    rules_path: Path
    output_directory: Path
    initialization: Initialization
    baseline_manifest: Path
    episodes: int
    master_seed: int
    epsilon: float
    learning_rate: float
    weight_decay: float
    loss: DmcLossName
    huber_delta: float
    max_grad_norm: float
    device: str
    amp: bool
    checkpoint_every: int
    evaluation_deals: int
    evaluation_seed: int
    bootstrap_resamples: int
    require_beats_random: bool

    def __post_init__(self) -> None:
        if self.schema_version != DMC_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported DMC config schema")
        if self.trainer_mode != "dmc":
            raise ValueError("E015 supports only trainer_mode=dmc")
        if self.episodes <= 0 or self.checkpoint_every <= 0:
            raise ValueError("episodes and checkpoint_every must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in 0..1")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.huber_delta <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("huber_delta and max_grad_norm must be positive")
        if self.evaluation_deals <= 0 or self.bootstrap_resamples <= 0:
            raise ValueError("evaluation_deals and bootstrap_resamples must be positive")
        if not 0 <= self.master_seed < 1 << 64:
            raise ValueError("master_seed must fit uint64")
        if self.amp and not self.device.startswith("cuda"):
            raise ValueError("AMP is supported only on CUDA devices")

    def fingerprint(self) -> str:
        """Hash training semantics while allowing budgets and outputs to change."""
        payload = self.to_dict()
        for key in (
            "output_directory",
            "episodes",
            "evaluation_deals",
            "evaluation_seed",
            "bootstrap_resamples",
            "require_beats_random",
        ):
            del payload[key]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible configuration record."""
        result = cast(dict[str, object], asdict(self))
        result["rules_path"] = str(self.rules_path)
        result["output_directory"] = str(self.output_directory)
        result["baseline_manifest"] = str(self.baseline_manifest)
        return result


@dataclass(slots=True)
class DmcTrainingState:
    """Mutable counters that are checkpointed exactly."""

    episodes: int = 0
    frames: int = 0
    learner_updates: int = 0
    policy_version: int = 0
    landlord_frames: int = 0
    landlord_down_frames: int = 0
    landlord_up_frames: int = 0


@dataclass(frozen=True, slots=True)
class DmcTrainResult:
    """Training counters, final losses, and artifact paths."""

    state: DmcTrainingState
    role_losses: Mapping[str, float]
    metrics_history: tuple[Mapping[str, object], ...]
    checkpoint_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class DmcEvaluation:
    """Fixed-deal random-baseline evaluation and acceptance outcome."""

    report: ArenaReport
    beats_random: bool


def load_dmc_config(path: Path) -> DmcConfig:
    """Load the JSON-subset YAML smoke configuration without another dependency."""
    resolved = path.resolve()
    raw = _object_dict(json.loads(resolved.read_text(encoding="utf-8")), "config")
    project_root = resolved.parents[2]
    return DmcConfig(
        schema_version=_integer(raw, "schema_version"),
        trainer_mode=cast(TrainerMode, _string(raw, "trainer_mode")),
        rules_path=_project_path(project_root, _string(raw, "rules_path")),
        output_directory=_project_path(project_root, _string(raw, "output_directory")),
        initialization=cast(Initialization, _string(raw, "initialization")),
        baseline_manifest=_project_path(project_root, _string(raw, "baseline_manifest")),
        episodes=_integer(raw, "episodes"),
        master_seed=_integer(raw, "master_seed"),
        epsilon=_number(raw, "epsilon"),
        learning_rate=_number(raw, "learning_rate"),
        weight_decay=_number(raw, "weight_decay"),
        loss=cast(DmcLossName, _string(raw, "loss")),
        huber_delta=_number(raw, "huber_delta"),
        max_grad_norm=_number(raw, "max_grad_norm"),
        device=_string(raw, "device"),
        amp=_boolean(raw, "amp"),
        checkpoint_every=_integer(raw, "checkpoint_every"),
        evaluation_deals=_integer(raw, "evaluation_deals"),
        evaluation_seed=_integer(raw, "evaluation_seed"),
        bootstrap_resamples=_integer(raw, "bootstrap_resamples"),
        require_beats_random=_boolean(raw, "require_beats_random"),
    )


class DmcGreedyPolicy:
    """Arena policy over an in-memory set of three learner models."""

    def __init__(
        self,
        policy_id: str,
        models: Mapping[SeatRole, TrainableModel],
        device: str,
    ) -> None:
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        self._policy_id = policy_id
        self._models = models
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
        if not legal_actions:
            raise DmcTrainingError("DMC policy received no legal actions")
        if context.role is None:
            raise DmcTrainingError("post-bid DMC policy cannot act before landlord resolution")
        features = encode_douzero_features(observation, legal_actions)
        model = self._models[context.role].eval()
        with torch.inference_mode():
            output = model(
                torch.from_numpy(features.z_batch).to(self._device),
                torch.from_numpy(features.x_batch).to(self._device),
                return_value=True,
            )
        scores = output["values"].detach().cpu().numpy()[:, 0]
        if scores.shape != (len(legal_actions),) or not np.isfinite(scores).all():
            raise DmcTrainingError("DMC policy produced invalid scores")
        return int(np.argmax(scores))


class DmcTrainer:
    """Single-actor deterministic DMC trainer suitable for end-to-end gates."""

    def __init__(self, config: DmcConfig) -> None:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise DmcTrainingError(f"requested unavailable CUDA device: {config.device}")
        self.config = config
        self.rules = load_rule_config(config.rules_path)
        self.rules_hash = _rules_hash(self.rules)
        torch.manual_seed(config.master_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.master_seed)
        self.rng = np.random.default_rng(config.master_seed)
        self.models = self._create_models()
        self.optimizers = {
            role: torch.optim.Adam(
                self.models[role].parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            for role in SEAT_ROLES
        }
        self.schedulers = {
            role: torch.optim.lr_scheduler.LambdaLR(self.optimizers[role], lambda _step: 1.0)
            for role in SEAT_ROLES
        }
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
        self.state = DmcTrainingState()
        self.role_losses: dict[str, float] = {role.value: math.nan for role in SEAT_ROLES}
        self.metrics_history: list[dict[str, object]] = []
        self.league = create_self_play_snapshot(
            "dmc:current",
            str(self.checkpoint_path),
            seed=config.master_seed,
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.config.output_directory / "checkpoint.pt"

    @property
    def manifest_path(self) -> Path:
        return self.config.output_directory / "manifest.json"

    def train(self, episodes: int | None = None) -> DmcTrainResult:
        """Collect complete games and regress every chosen value to its terminal return."""
        remaining = self.config.episodes - self.state.episodes
        count = remaining if episodes is None else episodes
        if count < 0 or count > remaining:
            raise ValueError(f"requested {count} episodes with only {remaining} remaining")
        for _ in range(count):
            episode_seed = splitmix64(self.config.master_seed + self.state.episodes)
            episode = collect_dmc_episode(
                episode_seed,
                self.rules,
                cast(Mapping[SeatRole, ActorModel], self.models),
                self.rng,
                epsilon=self.config.epsilon,
                policy_version=self.state.policy_version,
                device=self.config.device,
            )
            self._learn_episode(episode)
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
                    "role_losses": dict(self.role_losses),
                }
            )
            if self.state.episodes % self.config.checkpoint_every == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        return DmcTrainResult(
            state=DmcTrainingState(**asdict(self.state)),
            role_losses=dict(self.role_losses),
            metrics_history=tuple(dict(item) for item in self.metrics_history),
            checkpoint_path=self.checkpoint_path,
            manifest_path=self.manifest_path,
        )

    def evaluate_against_random(self) -> DmcEvaluation:
        """Run the role-balanced fixed-deal E015 acceptance comparison."""
        candidate = DmcGreedyPolicy("dmc:current", self.models, self.config.device)
        baseline = SeededRandomPolicy("baseline:seeded_random", self.config.evaluation_seed)
        arena = Arena(self.rules, (candidate, baseline))
        run = arena.evaluate_paired(
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
        return DmcEvaluation(run.report, beats_random)

    def save_checkpoint(self) -> None:
        """Atomically save learner, optimizer, scheduler, scaler, and RNG state."""
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
            "checkpoint_schema_version": DMC_CHECKPOINT_SCHEMA_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "rules_hash": self.rules_hash,
            "feature_schema_version": DOUZERO_FEATURE_SCHEMA_VERSION,
            "model_schema_version": DOUZERO_MODEL_SCHEMA_VERSION,
            "trainer_mode": "dmc",
            "models": {role.value: self.models[role].state_dict() for role in SEAT_ROLES},
            "optimizers": {role.value: self.optimizers[role].state_dict() for role in SEAT_ROLES},
            "schedulers": {role.value: self.schedulers[role].state_dict() for role in SEAT_ROLES},
            "amp_scaler": self.scaler.state_dict(),
            "state": asdict(self.state),
            "role_losses": dict(self.role_losses),
            "metrics_history": list(self.metrics_history),
            "numpy_rng_state": json.dumps(self.rng.bit_generator.state, sort_keys=True),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "league_snapshot": self.league.to_dict(),
            "training_phase": "dmc_smoke",
            "reward_curriculum": {"reward_mode": self.rules["reward_mode"]},
        }
        temporary = self.checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(self.checkpoint_path)
        digest = _sha256_file(self.checkpoint_path)
        manifest = {
            "checkpoint_schema_version": DMC_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_file": self.checkpoint_path.name,
            "checkpoint_sha256": digest,
            "git_commit": _git_commit(self.config.rules_path.parents[2]),
            "rules_hash": self.rules_hash,
            "feature_schema_version": DOUZERO_FEATURE_SCHEMA_VERSION,
            "model_schema_version": DOUZERO_MODEL_SCHEMA_VERSION,
            "trainer_mode": "dmc",
            "frames": self.state.frames,
            "episodes": self.state.episodes,
            "learner_updates": self.state.learner_updates,
            "policy_version": self.state.policy_version,
            "optimizer_state": True,
            "scheduler_state": True,
            "amp_scaler_state": True,
            "rng_state": True,
            "training_phase": "dmc_smoke",
            "reward_curriculum": {"reward_mode": self.rules["reward_mode"]},
            "league_snapshot": self.league.fingerprint(),
            "league_snapshot_file": league_path.name,
            "league_snapshot_sha256": league_file_sha256,
            "metrics": {f"loss_{key}": value for key, value in self.role_losses.items()},
            "metrics_file": "metrics.jsonl",
        }
        temporary_manifest = self.manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(self.manifest_path)
        metrics_path = output / "metrics.jsonl"
        temporary_metrics = metrics_path.with_suffix(".jsonl.tmp")
        temporary_metrics.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in self.metrics_history),
            encoding="utf-8",
        )
        temporary_metrics.replace(metrics_path)
        for role in SEAT_ROLES:
            role_path = output / f"{role.value}.ckpt"
            temporary_role = role_path.with_suffix(".ckpt.tmp")
            torch.save(self.models[role].state_dict(), temporary_role)
            temporary_role.replace(role_path)

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        """Restore every state required for an exact deterministic continuation."""
        path = self.checkpoint_path if checkpoint_path is None else checkpoint_path.resolve()
        raw = torch.load(path, map_location=self.config.device, weights_only=True)
        checkpoint = _object_mapping(raw, "checkpoint")
        if checkpoint.get("checkpoint_schema_version") != DMC_CHECKPOINT_SCHEMA_VERSION:
            raise DmcTrainingError("unsupported DMC checkpoint schema")
        if checkpoint.get("config_fingerprint") != self.config.fingerprint():
            raise DmcTrainingError("checkpoint configuration fingerprint mismatch")
        if checkpoint.get("rules_hash") != self.rules_hash:
            raise DmcTrainingError("checkpoint rule hash mismatch")
        models = _object_mapping(checkpoint.get("models"), "models")
        optimizers = _object_mapping(checkpoint.get("optimizers"), "optimizers")
        schedulers = _object_mapping(checkpoint.get("schedulers"), "schedulers")
        for role in SEAT_ROLES:
            self.models[role].load_state_dict(
                _object_mapping(models.get(role.value), f"{role.value} model"),
                strict=True,
            )
            self.optimizers[role].load_state_dict(
                _object_dict(optimizers.get(role.value), f"{role.value} optimizer")
            )
            self.schedulers[role].load_state_dict(
                _object_dict(schedulers.get(role.value), f"{role.value} scheduler")
            )
        self.scaler.load_state_dict(_object_dict(checkpoint.get("amp_scaler"), "amp_scaler"))
        state = _object_mapping(checkpoint.get("state"), "training state")
        self.state = DmcTrainingState(
            episodes=_mapping_integer(state, "episodes"),
            frames=_mapping_integer(state, "frames"),
            learner_updates=_mapping_integer(state, "learner_updates"),
            policy_version=_mapping_integer(state, "policy_version"),
            landlord_frames=_mapping_integer(state, "landlord_frames"),
            landlord_down_frames=_mapping_integer(state, "landlord_down_frames"),
            landlord_up_frames=_mapping_integer(state, "landlord_up_frames"),
        )
        losses = _object_mapping(checkpoint.get("role_losses"), "role_losses")
        self.role_losses = {role.value: _mapping_number(losses, role.value) for role in SEAT_ROLES}
        history = checkpoint.get("metrics_history")
        if not isinstance(history, list):
            raise DmcTrainingError("checkpoint metrics_history must be a list")
        self.metrics_history = [
            _object_dict(item, f"metrics_history[{index}]") for index, item in enumerate(history)
        ]
        if len(self.metrics_history) != self.state.episodes:
            raise DmcTrainingError("checkpoint metric count differs from episode count")
        try:
            self.league = LeagueSnapshot.from_dict(checkpoint.get("league_snapshot"))
        except ValueError as error:
            raise DmcTrainingError(f"checkpoint league snapshot is invalid: {error}") from error
        if self.league.schedule_cursor != self.state.episodes:
            raise DmcTrainingError("checkpoint league cursor differs from episode count")
        if self.league.population.champion.policy_version != self.state.policy_version:
            raise DmcTrainingError("checkpoint league champion version differs from learner")
        numpy_state = checkpoint.get("numpy_rng_state")
        if not isinstance(numpy_state, str):
            raise DmcTrainingError("checkpoint NumPy RNG state is absent")
        self.rng.bit_generator.state = _object_dict(json.loads(numpy_state), "NumPy RNG state")
        torch_state = checkpoint.get("torch_rng_state")
        if not isinstance(torch_state, Tensor):
            raise DmcTrainingError("checkpoint Torch RNG state is absent")
        torch.set_rng_state(torch_state.cpu())
        cuda_states = checkpoint.get("cuda_rng_states")
        if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)

    def _create_models(self) -> dict[SeatRole, TrainableModel]:
        models = {
            role: cast(
                TrainableModel,
                create_douzero_model(role.value),
            ).to(self.config.device)
            for role in SEAT_ROLES
        }
        if self.config.initialization == "random":
            return models
        checkpoints = load_official_checkpoint_set(
            self.config.baseline_manifest,
            self.config.initialization,
        )
        for role in SEAT_ROLES:
            raw = torch.load(
                checkpoints.file_for_role(role).path,
                map_location=self.config.device,
                weights_only=True,
            )
            models[role].load_state_dict(
                _object_mapping(raw, f"{role.value} checkpoint"), strict=True
            )
        return models

    def _learn_episode(self, episode: DmcEpisode) -> None:
        for role in SEAT_ROLES:
            transitions = episode.transitions_for(role)
            if not transitions:
                raise DmcTrainingError(f"episode {episode.seed} has no {role.value} decisions")
            self.role_losses[role.value] = self._learn_role(role, transitions)
            count = len(transitions)
            if role is SeatRole.LANDLORD:
                self.state.landlord_frames += count
            elif role is SeatRole.LANDLORD_DOWN:
                self.state.landlord_down_frames += count
            else:
                self.state.landlord_up_frames += count

    def _learn_role(self, role: SeatRole, transitions: Sequence[DmcTransition]) -> float:
        model = self.models[role].train()
        optimizer = self.optimizers[role]
        z = torch.from_numpy(np.stack([item.z for item in transitions])).to(self.config.device)
        x = torch.from_numpy(np.stack([item.x for item in transitions])).to(self.config.device)
        target = torch.tensor(
            [item.target for item in transitions],
            dtype=torch.float32,
            device=self.config.device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if self.config.device.startswith("cuda") else "cpu",
            enabled=self.config.amp,
        ):
            prediction = model(z, x, return_value=True)["values"]
            loss = dmc_value_loss(
                prediction,
                target,
                self.config.loss,
                self.config.huber_delta,
            )
        if not torch.isfinite(loss):
            raise DmcTrainingError(f"non-finite {role.value} loss")
        if self.config.amp:
            torch.autograd.backward((self.scaler.scale(loss),))
            self.scaler.unscale_(optimizer)
        else:
            torch.autograd.backward((loss,))
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            raise DmcTrainingError(f"non-finite {role.value} gradient norm")
        if self.config.amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        self.schedulers[role].step()
        self.state.learner_updates += 1
        return float(loss.detach().cpu().item())


def _rules_hash(rules: RuleConfig) -> str:
    canonical = json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


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
        raise DmcTrainingError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _object_dict(value: object, label: str) -> dict[str, object]:
    return dict(_object_mapping(value, label))


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
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DmcTrainingError(f"checkpoint {key} must be a non-negative integer")
    return value


def _mapping_number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DmcTrainingError(f"checkpoint {key} must be numeric")
    return float(value)


__all__ = (
    "DMC_CHECKPOINT_SCHEMA_VERSION",
    "DMC_CONFIG_SCHEMA_VERSION",
    "DmcConfig",
    "DmcEvaluation",
    "DmcGreedyPolicy",
    "DmcTrainResult",
    "DmcTrainer",
    "DmcTrainingError",
    "DmcTrainingState",
    "Initialization",
    "TrainerMode",
    "load_dmc_config",
)
