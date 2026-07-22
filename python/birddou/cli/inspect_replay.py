"""Validate and inspect a serialized native state without exposing hidden hands by default."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from birddou import PyDdzEnv, load_rule_config
from birddou.env_types import RuleConfig

REPLAY_INSPECTION_SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    """Build the strict local state-inspection command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--observer", type=int, choices=(0, 1, 2))
    parser.add_argument("--show-legal-actions", action="store_true")
    parser.add_argument(
        "--privileged",
        action="store_true",
        help="include the replay-validated full state for offline debugging",
    )
    parser.add_argument("--output", type=Path)
    return parser


def inspect_serialized_state(
    serialized_state: bytes,
    rules: RuleConfig,
    *,
    observer: int | None = None,
    show_legal_actions: bool = False,
    privileged: bool = False,
) -> dict[str, object]:
    """Restore through Rust, then return a public audit view and optional debug state."""
    if not serialized_state:
        raise ValueError("serialized replay state cannot be empty")
    environment = PyDdzEnv()
    current = environment.restore(serialized_state, rules)
    selected_observer = current["current_player"] if observer is None else observer
    observation = environment.observe(selected_observer)
    payload: dict[str, object] = {
        "schema_version": REPLAY_INSPECTION_SCHEMA_VERSION,
        "serialized_sha256": hashlib.sha256(serialized_state).hexdigest(),
        "rule_config_id": rules["rule_config_id"],
        "phase": observation["phase"],
        "observer": selected_observer,
        "current_player": observation["current_player"],
        "landlord": observation["landlord"],
        "role": observation["role"],
        "terminal": environment.terminal,
        "cards_left": list(observation["cards_left"]),
        "bomb_count": observation["bomb_count"],
        "multiplier_exp": observation["multiplier_exp"],
        "history_count": len(observation["history"]),
        "bid_history_count": len(observation["bid_history"]),
        "observation": observation,
    }
    if show_legal_actions:
        payload["legal_actions"] = environment.legal_actions()
    if privileged:
        decoded = json.loads(serialized_state)
        if not isinstance(decoded, Mapping):
            raise ValueError("serialized native state envelope must be a mapping")
        payload["privileged_state_envelope"] = cast(dict[str, object], dict(decoded))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one state and emit deterministic JSON to stdout or a file."""
    arguments = build_parser().parse_args(argv)
    report = inspect_serialized_state(
        arguments.state.read_bytes(),
        load_rule_config(arguments.rules),
        observer=arguments.observer,
        show_legal_actions=arguments.show_legal_actions,
        privileged=arguments.privileged,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = cast(Path | None, arguments.output)
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "REPLAY_INSPECTION_SCHEMA_VERSION",
    "build_parser",
    "inspect_serialized_state",
    "main",
)
