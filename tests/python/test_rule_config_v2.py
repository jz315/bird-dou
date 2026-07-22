"""R002 integration tests for rule-schema isolation at the Python ABI boundary."""

from pathlib import Path
from typing import cast

import pytest

from birddou import (
    PyDdzEnv,
    RuleConfig,
    load_rule_config,
    load_versioned_rule_config,
    parse_rule_config,
    parse_versioned_rule_config,
    rule_config_hash,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY_RULES = ROOT / "configs" / "rules" / "canonical_full.yaml"
HUANLE_FIXTURE = ROOT / "tests" / "rules" / "huanle_classic_v1" / "parser_fixture_v2.yaml"


def test_python_can_parse_and_hash_v2_without_reclassifying_legacy_v1() -> None:
    legacy = load_rule_config(LEGACY_RULES)
    huanle_yaml = HUANLE_FIXTURE.read_text(encoding="utf-8")
    huanle = parse_versioned_rule_config(huanle_yaml)

    assert legacy["schema_version"] == 1
    assert legacy["profile"] == "canonical_full"
    assert huanle["schema_version"] == 2
    assert huanle["profile"] == "huanle_classic_v1"
    assert load_versioned_rule_config(HUANLE_FIXTURE) == huanle
    assert len(rule_config_hash(huanle_yaml)) == 64


def test_python_legacy_entrypoints_reject_v2_before_the_legacy_engine_can_interpret_it() -> None:
    huanle_yaml = HUANLE_FIXTURE.read_text(encoding="utf-8")
    huanle = parse_versioned_rule_config(huanle_yaml)

    with pytest.raises(ValueError, match="expected 1"):
        parse_rule_config(huanle_yaml)
    with pytest.raises(ValueError, match="legacy v1 engine cannot interpret rule schema 2"):
        PyDdzEnv().reset(17, cast(RuleConfig, huanle))
