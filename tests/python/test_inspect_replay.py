"""Validated public replay inspection and explicit privileged-mode tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from birddou import PyDdzEnv, load_rule_config
from birddou.cli.inspect_replay import inspect_serialized_state, main

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "configs" / "rules" / "douzero_post_bid.yaml"


def test_replay_inspection_is_public_by_default_and_cli_writes_json(tmp_path: Path) -> None:
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    environment.reset(7009, rules)
    serialized = environment.serialize()
    report = inspect_serialized_state(serialized, rules, observer=1, show_legal_actions=True)

    assert report["observer"] == 1
    assert report["serialized_sha256"]
    assert "legal_actions" in report
    assert "privileged_state_envelope" not in report
    observation = cast(dict[str, object], report["observation"])
    assert "hands" not in observation

    state_path = tmp_path / "state.json"
    output_path = tmp_path / "inspection.json"
    state_path.write_bytes(serialized)
    assert (
        main(
            (
                "--state",
                str(state_path),
                "--rules",
                str(RULES_PATH),
                "--output",
                str(output_path),
            )
        )
        == 0
    )
    written = cast(dict[str, object], json.loads(output_path.read_text(encoding="utf-8")))
    assert written["serialized_sha256"] == report["serialized_sha256"]


def test_privileged_replay_inspection_requires_explicit_switch() -> None:
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    environment.reset(7010, rules)
    report = inspect_serialized_state(environment.serialize(), rules, privileged=True)
    envelope = cast(dict[str, object], report["privileged_state_envelope"])
    state = cast(dict[str, object], envelope["state"])
    assert len(cast(list[object], state["hands"])) == 3
