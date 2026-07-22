"""Adapter for the source-pinned official DouZero features and checkpoints."""

from __future__ import annotations

import hashlib
import importlib
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from birddou.env_types import Action, Observation, PlayGameAction, RankCounts
from birddou.eval.baselines import PolicyDecisionContext
from birddou.eval.douzero_differential import rank_counts_to_douzero_cards
from birddou.eval.paired_deals import (
    SEAT_ROLES,
    SeatRole,
    role_for_game_seat,
)
from birddou.features.douzero import encode_douzero_features

DOUZERO_ADAPTER_SCHEMA_VERSION = 1
DOUZERO_SOURCE_COMMIT = "718a5c920bf3361e34178a38f3b80458e176b351"
_OFFICIAL_POSITIONS = tuple(role.value for role in SEAT_ROLES)
DOUZERO_FEATURE_ENCODERS = ("native", "official_reference")


class DouZeroAdapterError(RuntimeError):
    """Source, checkpoint, feature, or inference contract failure."""


class _OfficialInfoSet(Protocol):
    player_position: str
    player_hand_cards: list[int]
    num_cards_left_dict: dict[str, int]
    three_landlord_cards: list[int]
    card_play_action_seq: list[list[int]]
    other_hand_cards: list[int]
    legal_actions: list[list[int]]
    last_move: list[int]
    last_two_moves: list[list[int]]
    last_move_dict: dict[str, list[int]]
    played_cards: dict[str, list[int]]
    all_handcards: object
    last_pid: object
    bomb_num: int


class _Tensor(Protocol):
    @property
    def shape(self) -> object: ...

    def to(self, device: str) -> _Tensor: ...

    def detach(self) -> _Tensor: ...

    def cpu(self) -> _Tensor: ...

    def numpy(self) -> NDArray[np.float32]: ...


class _Model(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object], strict: bool = True) -> object: ...

    def to(self, device: str) -> _Model: ...

    def eval(self) -> _Model: ...

    def __call__(self, z: _Tensor, x: _Tensor, *, return_value: bool) -> Mapping[str, _Tensor]: ...


class _CudaModule(Protocol):
    def is_available(self) -> bool: ...


class _TorchModule(Protocol):
    cuda: _CudaModule

    def from_numpy(self, value: NDArray[np.float32]) -> _Tensor: ...

    def load(
        self,
        path: str,
        *,
        map_location: str,
        weights_only: bool,
    ) -> object: ...

    def inference_mode(self) -> AbstractContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class CheckpointFile:
    """One immutable role checkpoint from the tracked provenance manifest."""

    role: SeatRole
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class OfficialCheckpointSet:
    """Complete three-role DouZero checkpoint set and pinned source checkout."""

    schema_version: int
    name: str
    source: Path
    source_commit: str
    files: tuple[CheckpointFile, CheckpointFile, CheckpointFile]

    def __post_init__(self) -> None:
        if self.schema_version != DOUZERO_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported DouZero adapter schema")
        if tuple(item.role for item in self.files) != SEAT_ROLES:
            raise ValueError("checkpoint files must be ordered landlord/down/up")

    def file_for_role(self, role: SeatRole) -> CheckpointFile:
        """Return the declared checkpoint for a named seat role."""
        return self.files[SEAT_ROLES.index(role)]


@dataclass(frozen=True, slots=True)
class OfficialFeatures:
    """Official Table 4/5 features aligned to BIRD-Dou legal-action order."""

    schema_version: int
    position: SeatRole
    x_batch: NDArray[np.float32]
    z_batch: NDArray[np.float32]
    legal_action_cards: tuple[tuple[int, ...], ...]


def load_official_checkpoint_set(
    manifest_path: Path,
    weight_set: str,
) -> OfficialCheckpointSet:
    """Load and checksum a complete ADP or WP set from the tracked manifest."""
    resolved_manifest = manifest_path.resolve()
    raw = cast(
        dict[str, object],
        tomllib.loads(resolved_manifest.read_text(encoding="utf-8")),
    )
    source_commit = _required_str(raw, "commit")
    if source_commit != DOUZERO_SOURCE_COMMIT:
        raise DouZeroAdapterError(
            f"manifest source commit {source_commit} differs from adapter "
            f"commit {DOUZERO_SOURCE_COMMIT}"
        )
    source = resolved_manifest.parent / _required_str(raw, "source_directory")
    weights_root = (resolved_manifest.parent / _required_str(raw, "weights_directory")).resolve()
    declared = raw.get("weight_files")
    if not isinstance(declared, list):
        raise DouZeroAdapterError("manifest weight_files must be an array of tables")
    by_role: dict[SeatRole, CheckpointFile] = {}
    for value in declared:
        table = _string_table(value, "weight_files item")
        if _required_str(table, "set") != weight_set:
            continue
        try:
            role = SeatRole(_required_str(table, "role"))
        except ValueError as error:
            raise DouZeroAdapterError("manifest contains an unknown checkpoint role") from error
        size = table.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise DouZeroAdapterError("checkpoint size must be a positive integer")
        path = (weights_root / _required_str(table, "relative_path")).resolve()
        try:
            path.relative_to(weights_root)
        except ValueError as error:
            raise DouZeroAdapterError(
                f"checkpoint path escapes weights directory: {path}"
            ) from error
        checkpoint = CheckpointFile(
            role=role,
            path=path,
            size=size,
            sha256=_required_str(table, "sha256"),
        )
        if role in by_role:
            raise DouZeroAdapterError(f"duplicate {weight_set} checkpoint role: {role.value}")
        _verify_checkpoint(checkpoint)
        by_role[role] = checkpoint
    missing = set(SEAT_ROLES) - by_role.keys()
    if missing:
        missing_names = sorted(role.value for role in missing)
        raise DouZeroAdapterError(
            f"checkpoint set {weight_set} is incomplete; missing {missing_names}. "
            f"Run scripts/fetch_douzero_baseline.py --weight-set {weight_set}."
        )
    return OfficialCheckpointSet(
        schema_version=DOUZERO_ADAPTER_SCHEMA_VERSION,
        name=weight_set,
        source=source,
        source_commit=source_commit,
        files=(
            by_role[SeatRole.LANDLORD],
            by_role[SeatRole.LANDLORD_DOWN],
            by_role[SeatRole.LANDLORD_UP],
        ),
    )


def encode_official_features(
    observation: Observation,
    legal_actions: Sequence[Action],
    source: Path,
) -> OfficialFeatures:
    """Run the pinned official feature function over a safe BIRD-Dou observation."""
    module = _import_official_module(source, "douzero.env.env")
    game_module = _import_official_module(source, "douzero.env.game")
    factory = cast(Callable[[str], _OfficialInfoSet], game_module.InfoSet)
    get_obs = cast(Callable[[_OfficialInfoSet], Mapping[str, object]], module.get_obs)
    info_set = _build_official_info_set(observation, legal_actions, factory)
    encoded = get_obs(info_set)
    landlord = observation["landlord"]
    if landlord is None:
        raise DouZeroAdapterError("official features require a resolved landlord")
    role = role_for_game_seat(observation["observer"], landlord)
    x_batch = _float32_array(encoded.get("x_batch"), "x_batch")
    z_batch = _float32_array(encoded.get("z_batch"), "z_batch")
    expected_width = 373 if role is SeatRole.LANDLORD else 484
    expected_actions = len(legal_actions)
    if x_batch.shape != (expected_actions, expected_width):
        raise DouZeroAdapterError(
            f"official {role.value} x_batch has shape {x_batch.shape}; "
            f"expected {(expected_actions, expected_width)}"
        )
    if z_batch.shape != (expected_actions, 5, 162):
        raise DouZeroAdapterError(
            f"official z_batch has shape {z_batch.shape}; expected {(expected_actions, 5, 162)}"
        )
    return OfficialFeatures(
        schema_version=DOUZERO_ADAPTER_SCHEMA_VERSION,
        position=role,
        x_batch=x_batch,
        z_batch=z_batch,
        legal_action_cards=tuple(
            tuple(_action_to_official_cards(action)) for action in legal_actions
        ),
    )


class OfficialDouZeroPolicy:
    """Arena policy backed by all three official DouZero role networks."""

    def __init__(
        self,
        policy_id: str,
        checkpoints: OfficialCheckpointSet,
        device: str = "cpu",
        feature_encoder: str = "native",
    ) -> None:
        if not policy_id or policy_id.strip() != policy_id:
            raise ValueError("policy_id must be non-empty without surrounding whitespace")
        torch = _load_torch()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise DouZeroAdapterError(f"requested unavailable CUDA device: {device}")
        if feature_encoder not in DOUZERO_FEATURE_ENCODERS:
            raise ValueError(
                f"feature_encoder must be one of {DOUZERO_FEATURE_ENCODERS}, "
                f"got {feature_encoder!r}"
            )
        self._policy_id = policy_id
        self._checkpoints = checkpoints
        self._device = device
        self._feature_encoder = feature_encoder
        model_module = importlib.import_module("birddou.models.douzero_model")
        model_dict = cast(
            Mapping[str, Callable[[], _Model]],
            model_module.DOUZERO_MODEL_FACTORIES,
        )
        self._models: dict[SeatRole, _Model] = {}
        for role in SEAT_ROLES:
            try:
                model = model_dict[role.value]()
            except KeyError as error:
                raise DouZeroAdapterError(
                    f"official model_dict has no {role.value} factory"
                ) from error
            checkpoint = checkpoints.file_for_role(role)
            loaded = torch.load(
                str(checkpoint.path),
                map_location=device,
                weights_only=True,
            )
            state = _string_mapping(loaded, f"{role.value} checkpoint")
            expected_keys = set(model.state_dict())
            actual_keys = set(state)
            if actual_keys != expected_keys:
                raise DouZeroAdapterError(
                    f"{role.value} checkpoint keys differ: "
                    f"missing={sorted(expected_keys - actual_keys)}, "
                    f"extra={sorted(actual_keys - expected_keys)}"
                )
            model.load_state_dict(state, strict=True)
            self._models[role] = model.to(device).eval()

    @classmethod
    def from_manifest(
        cls,
        policy_id: str,
        manifest_path: Path,
        weight_set: str,
        device: str = "cpu",
        feature_encoder: str = "native",
    ) -> OfficialDouZeroPolicy:
        """Checksum and load a declared official checkpoint set."""
        return cls(
            policy_id,
            load_official_checkpoint_set(manifest_path, weight_set),
            device,
            feature_encoder,
        )

    @property
    def policy_id(self) -> str:
        """Stable Arena policy identifier."""
        return self._policy_id

    @property
    def checkpoint_set_name(self) -> str:
        """ADP/WP manifest set name."""
        return self._checkpoints.name

    @property
    def feature_encoder(self) -> str:
        """Selected native or upstream-reference feature path."""
        return self._feature_encoder

    def score_actions(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> NDArray[np.float32]:
        """Return one official value score per canonical BIRD-Dou legal action."""
        _validate_policy_input(observation, legal_actions, context)
        if context.role is None:
            raise DouZeroAdapterError("official post-bid policy cannot act during bidding")
        if self._feature_encoder == "native":
            native_features = encode_douzero_features(observation, legal_actions)
            x_batch = native_features.x_batch
            z_batch = native_features.z_batch
        else:
            reference_features = encode_official_features(
                observation,
                legal_actions,
                self._checkpoints.source,
            )
            x_batch = reference_features.x_batch
            z_batch = reference_features.z_batch
        torch = _load_torch()
        model = self._models[context.role]
        z_tensor = torch.from_numpy(z_batch).to(self._device)
        x_tensor = torch.from_numpy(x_batch).to(self._device)
        with torch.inference_mode():
            output = model(z_tensor, x_tensor, return_value=True)
        try:
            values = output["values"]
        except KeyError as error:
            raise DouZeroAdapterError("official model returned no values tensor") from error
        scores = np.asarray(values.detach().cpu().numpy(), dtype=np.float32)
        if scores.shape != (len(legal_actions), 1):
            raise DouZeroAdapterError(
                f"official model returned shape {scores.shape}; expected {(len(legal_actions), 1)}"
            )
        if not np.isfinite(scores).all():
            raise DouZeroAdapterError("official model returned NaN or infinity")
        return np.ascontiguousarray(scores[:, 0])

    def select_action(
        self,
        observation: Observation,
        legal_actions: Sequence[Action],
        context: PolicyDecisionContext,
    ) -> int:
        """Select the first maximum-valued action, matching official inference."""
        _validate_policy_input(observation, legal_actions, context)
        if len(legal_actions) == 1:
            return 0
        return int(np.argmax(self.score_actions(observation, legal_actions, context)))


def _build_official_info_set(
    observation: Observation,
    legal_actions: Sequence[Action],
    factory: Callable[[str], _OfficialInfoSet],
) -> _OfficialInfoSet:
    observer = observation["observer"]
    if observation["current_player"] != observer:
        raise DouZeroAdapterError("official features require the current player's observation")
    if observation["phase"] != "card_play":
        raise DouZeroAdapterError(
            f"official DouZero adapter only supports card_play, got {observation['phase']}"
        )
    landlord = observation["landlord"]
    if landlord is None:
        raise DouZeroAdapterError("official features require a resolved landlord")
    role = role_for_game_seat(observer, landlord)
    info_set = factory(role.value)
    info_set.player_hand_cards = _counts_to_official(observation["own_hand"])
    info_set.other_hand_cards = _counts_to_official(observation["unknown_pool"])
    info_set.legal_actions = [_action_to_official_cards(action) for action in legal_actions]
    info_set.last_move = (
        []
        if observation["last_non_pass"] is None
        else _counts_to_official(observation["last_non_pass"]["cards"])
    )
    cardplay_events = [event for event in observation["history"] if "play" in event["action"]]
    history = [_action_to_official_cards(event["action"]) for event in cardplay_events]
    info_set.card_play_action_seq = history
    info_set.last_two_moves = [list(action) for action in history[-2:]]
    seat_roles = tuple(role_for_game_seat(seat, landlord) for seat in range(3))
    info_set.num_cards_left_dict = {
        seat_roles[seat].value: observation["cards_left"][seat] for seat in range(3)
    }
    info_set.played_cards = {
        seat_roles[seat].value: _counts_to_official(observation["public_played"][seat])
        for seat in range(3)
    }
    last_move_dict: dict[str, list[int]] = {position.value: [] for position in SEAT_ROLES}
    for event in cardplay_events:
        event_role = role_for_game_seat(event["actor"], landlord)
        last_move_dict[event_role.value] = _action_to_official_cards(event["action"])
    info_set.last_move_dict = last_move_dict
    info_set.three_landlord_cards = _counts_to_official(observation["public_bottom_cards"])
    info_set.bomb_num = observation["bomb_count"]
    info_set.all_handcards = None
    info_set.last_pid = None
    return info_set


def _validate_policy_input(
    observation: Observation,
    legal_actions: Sequence[Action],
    context: PolicyDecisionContext,
) -> None:
    if not legal_actions:
        raise DouZeroAdapterError("official policy received no legal actions")
    if context.role is None:
        raise DouZeroAdapterError("official post-bid policy cannot act during bidding")
    if observation["observer"] != context.seat:
        raise DouZeroAdapterError("policy context seat differs from observation observer")
    landlord = observation["landlord"]
    if landlord is None:
        raise DouZeroAdapterError("official post-bid policy requires a resolved landlord")
    if role_for_game_seat(context.seat, landlord) is not context.role:
        raise DouZeroAdapterError("policy context role differs from resolved landlord")


def _counts_to_official(counts: RankCounts) -> list[int]:
    return rank_counts_to_douzero_cards(tuple(counts))


def _action_to_official_cards(action: Action) -> list[int]:
    if "play" not in action:
        raise DouZeroAdapterError("official post-bid adapter received a non-play action")
    play = cast(PlayGameAction, action)["play"]
    return _counts_to_official(play["cards"])


def _import_official_module(source: Path, module_name: str) -> ModuleType:
    resolved_source = source.resolve()
    package_path = (resolved_source / "douzero").resolve()
    if not package_path.is_dir():
        raise DouZeroAdapterError(
            f"DouZero source is absent at {resolved_source}; run scripts/fetch_douzero_baseline.py"
        )
    existing = sys.modules.get("douzero")
    if existing is None:
        package = ModuleType("douzero")
        package.__dict__["__path__"] = [str(package_path)]
        sys.modules["douzero"] = package
    else:
        raw_paths = getattr(existing, "__path__", ())
        loaded_paths = {Path(item).resolve() for item in raw_paths}
        if package_path not in loaded_paths:
            raise DouZeroAdapterError(
                "a different douzero package is already imported; restart Python "
                f"before loading pinned source {resolved_source}"
            )
    parts = module_name.split(".")
    for length in range(2, len(parts)):
        parent_name = ".".join(parts[:length])
        parent_path = resolved_source.joinpath(*parts[:length]).resolve()
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = ModuleType(parent_name)
            parent.__dict__["__path__"] = [str(parent_path)]
            sys.modules[parent_name] = parent
        elif parent_path not in {Path(item).resolve() for item in getattr(parent, "__path__", ())}:
            raise DouZeroAdapterError(f"a different {parent_name} package is already imported")
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    module_file = module.__file__
    if module_file is None:
        raise DouZeroAdapterError(f"official module {module_name} has no source path")
    try:
        Path(module_file).resolve().relative_to(resolved_source)
    except ValueError as error:
        raise DouZeroAdapterError(
            f"imported {module_name} from {module_file}, outside {resolved_source}"
        ) from error
    return module


def _load_torch() -> _TorchModule:
    try:
        module = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise DouZeroAdapterError(
            "PyTorch is required for official checkpoint inference; install bird-dou[model]"
        ) from error
    return cast(_TorchModule, module)


def _verify_checkpoint(checkpoint: CheckpointFile) -> None:
    if not checkpoint.path.is_file():
        raise DouZeroAdapterError(
            f"checkpoint is absent at {checkpoint.path}; run "
            "scripts/fetch_douzero_baseline.py --weight-set douzero_ADP "
            "or --weight-set douzero_WP"
        )
    if checkpoint.path.stat().st_size != checkpoint.size:
        raise DouZeroAdapterError(
            f"checkpoint size mismatch for {checkpoint.path}: "
            f"{checkpoint.path.stat().st_size} != {checkpoint.size}"
        )
    digest = hashlib.sha256(checkpoint.path.read_bytes()).hexdigest()
    if digest != checkpoint.sha256:
        raise DouZeroAdapterError(
            f"checkpoint SHA-256 mismatch for {checkpoint.path}: {digest} != {checkpoint.sha256}"
        )


def _float32_array(value: object, label: str) -> NDArray[np.float32]:
    if not isinstance(value, np.ndarray):
        raise DouZeroAdapterError(f"official {label} is not a NumPy array")
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise DouZeroAdapterError(f"official {label} contains NaN or infinity")
    return np.ascontiguousarray(array)


def _string_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DouZeroAdapterError(f"{label} must be a string-keyed state dict")
    return cast(Mapping[str, object], value)


def _string_table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DouZeroAdapterError(f"manifest {label} must be a table")
    return cast(dict[str, object], value)


def _required_str(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise DouZeroAdapterError(f"manifest {key} must be a non-empty string")
    return value


__all__ = (
    "DOUZERO_ADAPTER_SCHEMA_VERSION",
    "DOUZERO_FEATURE_ENCODERS",
    "DOUZERO_SOURCE_COMMIT",
    "CheckpointFile",
    "DouZeroAdapterError",
    "OfficialCheckpointSet",
    "OfficialDouZeroPolicy",
    "OfficialFeatures",
    "encode_official_features",
    "load_official_checkpoint_set",
)
