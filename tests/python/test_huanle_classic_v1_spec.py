"""R001 machine-checkable specification corpus for the future Huanle profile."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "rules" / "huanle_classic_v1"
SPEC = ROOT / "docs" / "HUANLE_CLASSIC_V1.md"

FROZEN_IDS = {
    "HCV1-DEAL-001",
    "HCV1-BOTTOM-001",
    "HCV1-REVEAL-001",
    "HCV1-REVEAL-002",
    "HCV1-REVEAL-003",
    "HCV1-CALL-001",
    "HCV1-CALL-002",
    "HCV1-CALL-003",
    "HCV1-ROB-001",
    "HCV1-ROB-002",
    "HCV1-ROB-003",
    "HCV1-POST-001",
    "HCV1-DOUBLE-001",
    "HCV1-SETTLE-001",
    "HCV1-PLAY-001",
    "HCV1-PLAY-002",
}
MOVE_IDS = {f"HCV1-MOVE-{index:03d}" for index in range(1, 9)}
UNRESOLVED_IDS = {
    "HCV1-CONFIG-DEAL-REVEAL-SCHEDULE",
    "HCV1-CONFIG-AIRPLANE-SINGLE-WINGS",
    "HCV1-CONFIG-FOUR-TWO-SINGLE-WINGS",
    "HCV1-CONFIG-SPRING",
    "HCV1-CONFIG-SCORE-CAP",
    "HCV1-CONFIG-CALLER-RECLAIM",
}


def test_huanle_spec_corpus_is_complete_and_explicit() -> None:
    metadata = _mapping(_read("metadata.json"), "metadata")
    assert metadata == {
        "schema_version": 1,
        "profile": "huanle_classic_v1",
        "status": "specification_only",
        "source": "BIRD-Dou 欢乐斗地主规则与全系统修复计划 v2.0, sections 1 and 8",
        "compatibility": {
            "douzero_post_bid": "preserved",
            "canonical_full": "legacy_experimental_preserved",
            "new_profile_engine_implemented": False,
        },
        "files": ["frozen_rules.json", "move_goldens.json", "unresolved_config.json"],
    }

    frozen = _cases("frozen_rules.json")
    moves = _cases("move_goldens.json")
    unresolved = _cases("unresolved_config.json")
    assert _ids(frozen) == FROZEN_IDS
    assert _ids(moves) == MOVE_IDS
    assert _ids(unresolved) == UNRESOLVED_IDS
    for case in (*frozen, *moves, *unresolved):
        _assert_positive_negative_pair(case)

    for case in unresolved:
        assert case["fallback"] == "forbidden"
        assert isinstance(case["required_config_key"], str) and case["required_config_key"]


def test_huanle_spec_document_maps_every_machine_checked_rule() -> None:
    document = SPEC.read_text(encoding="utf-8")
    for case_id in FROZEN_IDS | MOVE_IDS:
        assert f"<!-- rule:{case_id} -->" in document
    for case_id in UNRESOLVED_IDS:
        assert f"<!-- config:{case_id} -->" in document
    assert "MUST NOT" in document
    assert "不得" in document
    assert "specification_only" not in document


def _read(name: str) -> object:
    return json.loads((CORPUS / name).read_text(encoding="utf-8"))


def _cases(name: str) -> tuple[Mapping[str, object], ...]:
    root = _mapping(_read(name), name)
    assert root["schema_version"] == 1
    assert root["profile"] == "huanle_classic_v1"
    raw_cases = root.get("cases")
    assert isinstance(raw_cases, list) and raw_cases
    return tuple(_mapping(case, f"{name} case") for case in raw_cases)


def _ids(cases: Iterable[Mapping[str, object]]) -> set[str]:
    values = [case.get("id") for case in cases]
    assert all(isinstance(value, str) and value for value in values)
    identifiers = cast(list[str], values)
    assert len(identifiers) == len(set(identifiers))
    return set(identifiers)


def _assert_positive_negative_pair(case: Mapping[str, object]) -> None:
    assert case.get("requirement") in ("MUST", "MUST NOT")
    positive = _mapping(case.get("positive"), "positive")
    negative = _mapping(case.get("negative"), "negative")
    assert isinstance(positive.get("scenario"), str) and positive["scenario"]
    assert isinstance(negative.get("scenario"), str) and negative["scenario"]
    assert isinstance(positive.get("expected"), Mapping) and positive["expected"]
    assert isinstance(negative.get("expected_rejection"), str) and negative["expected_rejection"]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{label} must be an object"
    assert all(isinstance(key, str) for key in value), f"{label} keys must be strings"
    return cast(Mapping[str, object], value)
