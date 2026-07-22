"""Differential testing against the pinned official DouZero card-play engine."""

from __future__ import annotations

import importlib
import json
import random
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import IO, Protocol, TypeAlias, cast

RankCounts: TypeAlias = tuple[int, ...]
Hands: TypeAlias = tuple[RankCounts, RankCounts, RankCounts]

_POSITION_TO_SEAT: dict[str, int] = {
    "landlord": 0,
    "landlord_down": 1,
    "landlord_up": 2,
}
_SEAT_TO_POSITION: tuple[str, str, str] = (
    "landlord",
    "landlord_down",
    "landlord_up",
)
_DOUZERO_RANKS: tuple[int, ...] = (*range(3, 15), 17, 20, 30)
_DOUZERO_TO_RANK_ID: dict[int, int] = {rank: rank_id for rank_id, rank in enumerate(_DOUZERO_RANKS)}


class DifferentialError(RuntimeError):
    """Base error for reproducibility, protocol, or comparison failures."""


class DifferentialMismatch(DifferentialError):
    """Raised on the first state difference between the two engines."""


@dataclass(frozen=True)
class BaselineManifest:
    """Auditable source pin for the external reference engine."""

    schema_version: int
    name: str
    repository: str
    commit: str
    license: str
    license_file: str
    source_directory: str
    weights_required: bool
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class Deal:
    """One rank-level deal shared by both engines."""

    hands: Hands
    bottom_cards: RankCounts
    douzero_hands: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    douzero_bottom_cards: tuple[int, ...]


@dataclass(frozen=True)
class EngineSnapshot:
    """Normalized state surface compared after every transition."""

    current_player: int
    hands: Hands
    played_cards: Hands
    cards_left: tuple[int, int, int]
    last_non_pass: RankCounts | None
    last_non_pass_player: int | None
    consecutive_passes: int
    bomb_count: int
    multiplier_exp: int
    terminal: bool
    winner: int | None
    raw_payoff: tuple[int, int, int]
    objective_payoff: tuple[int, int, int]
    legal_actions: frozenset[RankCounts]


@dataclass(frozen=True)
class DifferentialReport:
    """Successful differential run summary."""

    schema_version: int
    baseline_commit: str
    games: int
    compared_states: int
    applied_actions: int
    maximum_legal_actions: int
    mismatches: int
    seed: int

    def to_json(self) -> str:
        """Return a stable machine-readable report."""
        return json.dumps(asdict(self), sort_keys=True, indent=2)


class _OfficialInfoSet(Protocol):
    player_hand_cards: list[int]
    legal_actions: list[list[int]]


class _OfficialGame(Protocol):
    acting_player_position: str
    bomb_num: int
    card_play_action_seq: list[list[int]]
    game_infoset: _OfficialInfoSet
    game_over: bool
    info_sets: dict[str, _OfficialInfoSet]
    last_pid: str
    played_cards: dict[str, list[int]]
    winner: str

    def card_play_init(self, card_play_data: dict[str, list[int]]) -> None: ...

    def reset(self) -> None: ...

    def step(self) -> None: ...


class _ControlledPlayer:
    def __init__(self, position: str) -> None:
        self.position = position
        self._action: list[int] = []

    def set_action(self, action: Sequence[int]) -> None:
        self._action = list(action)

    def act(self, _infoset: object) -> list[int]:
        return self._action.copy()


def load_manifest(path: Path) -> BaselineManifest:
    """Load the tracked baseline manifest with explicit type validation."""
    raw = cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))
    required_files_value = raw.get("required_files")
    if not isinstance(required_files_value, list) or not all(
        isinstance(item, str) for item in required_files_value
    ):
        raise DifferentialError("manifest required_files must be a string array")

    return BaselineManifest(
        schema_version=_required_int(raw, "schema_version"),
        name=_required_str(raw, "name"),
        repository=_required_str(raw, "repository"),
        commit=_required_str(raw, "commit"),
        license=_required_str(raw, "license"),
        license_file=_required_str(raw, "license_file"),
        source_directory=_required_str(raw, "source_directory"),
        weights_required=_required_bool(raw, "weights_required"),
        required_files=tuple(cast(list[str], required_files_value)),
    )


def validate_baseline_source(source: Path, manifest: BaselineManifest) -> None:
    """Prove that a local checkout is exactly the manifest's source commit."""
    if manifest.schema_version != 1:
        raise DifferentialError(
            f"unsupported baseline manifest schema {manifest.schema_version}; expected 1"
        )
    if not source.is_dir():
        raise DifferentialError(
            f"DouZero source is absent at {source}; run scripts/fetch_douzero_baseline.py"
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise DifferentialError(f"cannot verify DouZero checkout at {source}: {error}") from error
    if revision != manifest.commit:
        raise DifferentialError(
            f"DouZero checkout is {revision}, but manifest requires {manifest.commit}"
        )

    required = (*manifest.required_files, manifest.license_file)
    missing = [relative for relative in required if not (source / relative).is_file()]
    if missing:
        raise DifferentialError(f"DouZero checkout is missing required files: {missing}")


def douzero_cards_to_rank_counts(cards: Sequence[int]) -> RankCounts:
    """Convert official integer ranks to the project's 15-count representation."""
    counts = [0] * 15
    for card in cards:
        try:
            rank_id = _DOUZERO_TO_RANK_ID[card]
        except KeyError as error:
            raise DifferentialError(f"unknown DouZero card rank {card}") from error
        counts[rank_id] += 1
    return tuple(counts)


def rank_counts_to_douzero_cards(cards: RankCounts) -> list[int]:
    """Expand project rank counts to the official sorted integer-rank action."""
    if len(cards) != 15:
        raise DifferentialError(f"rank counts have length {len(cards)}; expected 15")
    expanded: list[int] = []
    for rank, count in zip(_DOUZERO_RANKS, cards, strict=True):
        if count < 0 or count > (1 if rank >= 20 else 4):
            raise DifferentialError(f"invalid count {count} for DouZero rank {rank}")
        expanded.extend([rank] * count)
    return expanded


def deal_from_seed(seed: int) -> Deal:
    """Create the official 20/17/17 post-bid partition from a stable Python seed."""
    deck = [rank for rank in range(3, 15) for _ in range(4)]
    deck.extend([17] * 4)
    deck.extend([20, 30])
    random.Random(seed).shuffle(deck)
    official_hands = (
        tuple(sorted(deck[:20])),
        tuple(sorted(deck[20:37])),
        tuple(sorted(deck[37:])),
    )
    bottom = tuple(sorted(deck[17:20]))
    return Deal(
        hands=cast(Hands, tuple(douzero_cards_to_rank_counts(hand) for hand in official_hands)),
        bottom_cards=douzero_cards_to_rank_counts(bottom),
        douzero_hands=official_hands,
        douzero_bottom_cards=bottom,
    )


class OfficialDouZeroEngine:
    """Thin typed adapter over the pinned official ``GameEnv`` implementation."""

    def __init__(self, source: Path) -> None:
        package_name = "douzero"
        environment_package_name = f"{package_name}.env"
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__dict__["__path__"] = [str((source / "douzero").resolve())]
            environment_package = ModuleType(environment_package_name)
            environment_package.__dict__["__path__"] = [str((source / "douzero/env").resolve())]
            sys.modules[package_name] = package
            sys.modules[environment_package_name] = environment_package
        importlib.invalidate_caches()
        module = importlib.import_module(f"{environment_package_name}.game")
        _assert_module_from_source(module, source)
        factory = cast(
            Callable[[dict[str, _ControlledPlayer]], _OfficialGame],
            module.GameEnv,
        )
        self._players = {position: _ControlledPlayer(position) for position in _SEAT_TO_POSITION}
        self._game = factory(self._players)

    def reset(self, deal: Deal) -> EngineSnapshot:
        """Initialize the reference engine with an externally generated deal."""
        self._game.reset()
        self._game.card_play_init(
            {
                "landlord": list(deal.douzero_hands[0]),
                "landlord_down": list(deal.douzero_hands[1]),
                "landlord_up": list(deal.douzero_hands[2]),
                "three_landlord_cards": list(deal.douzero_bottom_cards),
            }
        )
        return self.snapshot()

    def step(self, action: RankCounts) -> EngineSnapshot:
        """Apply the shared normalized action to the reference engine."""
        position = self._game.acting_player_position
        self._players[position].set_action(rank_counts_to_douzero_cards(action))
        self._game.step()
        return self.snapshot()

    def snapshot(self) -> EngineSnapshot:
        """Normalize official state semantics to the Rust probe schema."""
        hands = cast(
            Hands,
            tuple(
                douzero_cards_to_rank_counts(self._game.info_sets[position].player_hand_cards)
                for position in _SEAT_TO_POSITION
            ),
        )
        played = cast(
            Hands,
            tuple(
                douzero_cards_to_rank_counts(self._game.played_cards[position])
                for position in _SEAT_TO_POSITION
            ),
        )
        target_cards = _official_active_target(self._game.card_play_action_seq)
        target = douzero_cards_to_rank_counts(target_cards) if target_cards is not None else None
        target_player = _POSITION_TO_SEAT[self._game.last_pid] if target is not None else None
        legal_actions = frozenset(
            douzero_cards_to_rank_counts(action)
            for action in ([] if self._game.game_over else self._game.game_infoset.legal_actions)
        )
        winner = _official_winner(self._game, hands)
        raw_payoff, objective_payoff = _official_payoffs(
            terminal=self._game.game_over,
            winner=winner,
            bomb_count=self._game.bomb_num,
        )
        return EngineSnapshot(
            current_player=_POSITION_TO_SEAT[self._game.acting_player_position],
            hands=hands,
            played_cards=played,
            cards_left=cast(tuple[int, int, int], tuple(sum(hand) for hand in hands)),
            last_non_pass=target,
            last_non_pass_player=target_player,
            consecutive_passes=(
                1
                if target is not None
                and self._game.card_play_action_seq
                and not self._game.card_play_action_seq[-1]
                else 0
            ),
            bomb_count=self._game.bomb_num,
            multiplier_exp=self._game.bomb_num,
            terminal=self._game.game_over,
            winner=winner,
            raw_payoff=raw_payoff,
            objective_payoff=objective_payoff,
            legal_actions=legal_actions,
        )


class RustDifferentialProbe:
    """Persistent JSON-lines client for the Rust differential example."""

    def __init__(self, command: Sequence[str], repository_root: Path) -> None:
        self._process = subprocess.Popen(
            list(command),
            cwd=repository_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def __enter__(self) -> RustDifferentialProbe:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def reset(self, deal: Deal) -> EngineSnapshot:
        """Reset the probe to the shared deal and return its first decision state."""
        return self._request(
            {
                "command": "reset",
                "hands": [list(hand) for hand in deal.hands],
                "bottom_cards": list(deal.bottom_cards),
                "landlord": 0,
            }
        )

    def step(self, action: RankCounts) -> EngineSnapshot:
        """Apply one normalized action and return the resulting state."""
        return self._request({"command": "step", "cards": list(action)})

    def close(self) -> None:
        """Stop the child process without leaking it on comparison failure."""
        if self._process.poll() is not None:
            return
        stdin = _required_stream(self._process.stdin, "stdin")
        stdout = _required_stream(self._process.stdout, "stdout")
        try:
            stdin.write('{"command":"shutdown"}\n')
            stdin.flush()
            stdout.readline()
            self._process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self._process.kill()
            self._process.wait(timeout=10)

    def _request(self, payload: Mapping[str, object]) -> EngineSnapshot:
        stdin = _required_stream(self._process.stdin, "stdin")
        stdout = _required_stream(self._process.stdout, "stdout")
        stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        stdin.flush()
        line = stdout.readline()
        if not line:
            stderr = _required_stream(self._process.stderr, "stderr").read()
            raise DifferentialError(
                f"Rust differential probe exited unexpectedly with stderr:\n{stderr}"
            )
        decoded_value: object = json.loads(line)
        decoded = _required_mapping(decoded_value, "probe response")
        status = decoded.get("status")
        if status == "error":
            raise DifferentialError(f"Rust probe rejected request: {decoded.get('message')}")
        if status != "ok":
            raise DifferentialError(f"unexpected Rust probe status {status!r}")
        return _snapshot_from_json(decoded.get("snapshot"))


def run_differential(
    *,
    repository_root: Path,
    source: Path,
    manifest: BaselineManifest,
    games: int,
    seed: int,
    rust_command: Sequence[str] | None = None,
) -> DifferentialReport:
    """Compare legal sets and synchronized random trajectories for ``games`` deals."""
    if games <= 0:
        raise DifferentialError("games must be positive")
    validate_baseline_source(source, manifest)
    reference = OfficialDouZeroEngine(source)
    command = (
        list(rust_command)
        if rust_command is not None
        else [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "ddz-rules",
            "--example",
            "differential_probe",
        ]
    )
    decision_rng = random.Random(seed ^ 0xB1D_D0)
    compared_states = 0
    applied_actions = 0
    maximum_legal_actions = 0

    with RustDifferentialProbe(command, repository_root) as candidate:
        for game_index in range(games):
            deal_seed = seed + game_index
            deal = deal_from_seed(deal_seed)
            reference_state = reference.reset(deal)
            candidate_state = candidate.reset(deal)
            turn = 0

            while True:
                _assert_snapshots_equal(
                    reference_state,
                    candidate_state,
                    game_index=game_index,
                    deal_seed=deal_seed,
                    turn=turn,
                )
                compared_states += 1
                maximum_legal_actions = max(
                    maximum_legal_actions, len(reference_state.legal_actions)
                )
                if reference_state.terminal:
                    break
                if turn >= 200:
                    raise DifferentialError(
                        f"game {game_index} seed {deal_seed} exceeded 200 actions"
                    )

                actions = sorted(reference_state.legal_actions)
                if not actions:
                    raise DifferentialMismatch(
                        f"non-terminal game {game_index} seed {deal_seed} has no action"
                    )
                action = actions[decision_rng.randrange(len(actions))]
                reference_state = reference.step(action)
                candidate_state = candidate.step(action)
                applied_actions += 1
                turn += 1

    return DifferentialReport(
        schema_version=1,
        baseline_commit=manifest.commit,
        games=games,
        compared_states=compared_states,
        applied_actions=applied_actions,
        maximum_legal_actions=maximum_legal_actions,
        mismatches=0,
        seed=seed,
    )


def _assert_snapshots_equal(
    reference: EngineSnapshot,
    candidate: EngineSnapshot,
    *,
    game_index: int,
    deal_seed: int,
    turn: int,
) -> None:
    fields = (
        "current_player",
        "hands",
        "played_cards",
        "cards_left",
        "last_non_pass",
        "last_non_pass_player",
        "consecutive_passes",
        "bomb_count",
        "multiplier_exp",
        "terminal",
        "winner",
        "raw_payoff",
        "objective_payoff",
    )
    differences = {
        field: (getattr(reference, field), getattr(candidate, field))
        for field in fields
        if getattr(reference, field) != getattr(candidate, field)
    }
    missing = sorted(reference.legal_actions - candidate.legal_actions)
    extra = sorted(candidate.legal_actions - reference.legal_actions)
    if not differences and not missing and not extra:
        return

    detail = {
        "game_index": game_index,
        "deal_seed": deal_seed,
        "turn": turn,
        "field_differences": differences,
        "missing_actions": missing,
        "extra_actions": extra,
    }
    raise DifferentialMismatch(json.dumps(detail, sort_keys=True, indent=2))


def _snapshot_from_json(value: object) -> EngineSnapshot:
    snapshot = _required_mapping(value, "snapshot")
    return EngineSnapshot(
        current_player=_json_int(snapshot, "current_player"),
        hands=_json_hands(snapshot, "hands"),
        played_cards=_json_hands(snapshot, "played_cards"),
        cards_left=_json_int_triple(snapshot, "cards_left"),
        last_non_pass=_json_optional_rank_counts(snapshot, "last_non_pass"),
        last_non_pass_player=_json_optional_int(snapshot, "last_non_pass_player"),
        consecutive_passes=_json_int(snapshot, "consecutive_passes"),
        bomb_count=_json_int(snapshot, "bomb_count"),
        multiplier_exp=_json_int(snapshot, "multiplier_exp"),
        terminal=_json_bool(snapshot, "terminal"),
        winner=_json_optional_int(snapshot, "winner"),
        raw_payoff=_json_int_triple(snapshot, "raw_payoff"),
        objective_payoff=_json_int_triple(snapshot, "objective_payoff"),
        legal_actions=frozenset(
            _rank_counts(item)
            for item in _required_list(snapshot.get("legal_actions"), "legal_actions")
        ),
    )


def _official_active_target(sequence: Sequence[Sequence[int]]) -> list[int] | None:
    if not sequence:
        return None
    if sequence[-1]:
        return list(sequence[-1])
    if len(sequence) >= 2 and sequence[-2]:
        return list(sequence[-2])
    return None


def _official_winner(game: _OfficialGame, hands: Hands) -> int | None:
    if not game.game_over:
        return None
    if game.winner == "landlord":
        return 0
    for seat in (1, 2):
        if sum(hands[seat]) == 0:
            return seat
    raise DifferentialError("official farmer win has no empty farmer hand")


def _official_payoffs(
    *, terminal: bool, winner: int | None, bomb_count: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if not terminal:
        return (0, 0, 0), (0, 0, 0)
    if winner is None:
        raise DifferentialError("terminal official state has no winner")
    magnitude = 2**bomb_count
    landlord_won = winner == 0
    raw = (
        2 * magnitude if landlord_won else -2 * magnitude,
        -magnitude if landlord_won else magnitude,
        -magnitude if landlord_won else magnitude,
    )
    objective = (
        magnitude if landlord_won else -magnitude,
        -magnitude if landlord_won else magnitude,
        -magnitude if landlord_won else magnitude,
    )
    return raw, objective


def _assert_module_from_source(module: ModuleType, source: Path) -> None:
    module_file = module.__file__
    if module_file is None:
        raise DifferentialError("imported DouZero game module has no source path")
    try:
        Path(module_file).resolve().relative_to(source.resolve())
    except ValueError as error:
        raise DifferentialError(
            f"imported DouZero from {module_file}, outside pinned checkout {source}"
        ) from error


def _required_str(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        raise DifferentialError(f"manifest {key} must be a string")
    return value


def _required_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DifferentialError(f"manifest {key} must be an integer")
    return value


def _required_bool(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise DifferentialError(f"manifest {key} must be a boolean")
    return value


def _required_stream(stream: IO[str] | None, name: str) -> IO[str]:
    if stream is None:
        raise DifferentialError(f"Rust probe has no {name} stream")
    return stream


def _required_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DifferentialError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _required_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise DifferentialError(f"{name} must be an array")
    return cast(list[object], value)


def _rank_counts(value: object) -> RankCounts:
    items = _required_list(value, "rank counts")
    if len(items) != 15 or not all(isinstance(item, int) for item in items):
        raise DifferentialError("rank counts must contain 15 integers")
    return tuple(cast(list[int], items))


def _json_hands(values: Mapping[str, object], key: str) -> Hands:
    items = _required_list(values.get(key), key)
    if len(items) != 3:
        raise DifferentialError(f"{key} must contain three hands")
    return (_rank_counts(items[0]), _rank_counts(items[1]), _rank_counts(items[2]))


def _json_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DifferentialError(f"snapshot {key} must be an integer")
    return value


def _json_bool(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise DifferentialError(f"snapshot {key} must be a boolean")
    return value


def _json_optional_int(values: Mapping[str, object], key: str) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    return _json_int(values, key)


def _json_optional_rank_counts(values: Mapping[str, object], key: str) -> RankCounts | None:
    value = values.get(key)
    return None if value is None else _rank_counts(value)


def _json_int_triple(values: Mapping[str, object], key: str) -> tuple[int, int, int]:
    items = _required_list(values.get(key), key)
    if len(items) != 3 or not all(isinstance(item, int) for item in items):
        raise DifferentialError(f"snapshot {key} must contain three integers")
    integers = cast(list[int], items)
    return integers[0], integers[1], integers[2]
