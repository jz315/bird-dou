"""Cross-play CLI and required reproducibility entrypoint tests."""

import json
from pathlib import Path

from birddou.cli.crossplay import main
from birddou.cli.evaluate import main as evaluate_main
from birddou.cli.policy_artifacts import parse_named_checkpoints

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_crossplay_cli_writes_every_requested_builtin_cell(tmp_path: Path) -> None:
    output = tmp_path / "crossplay.json"
    assert (
        main(
            (
                "--landlord-policies",
                "first_legal,longest_move",
                "--farmer-policies",
                "first_legal,longest_move",
                "--deals",
                "1",
                "--bootstrap-resamples",
                "100",
                "--output",
                str(output),
            )
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert len(payload["cells"]) == 4
    assert len(payload["matches"]) == 4
    assert set(payload["game_performance"]) == {
        "crossplay:first_legal",
        "crossplay:longest_move",
    }


def test_specified_shell_entrypoints_exist_and_fail_fast() -> None:
    for name in (
        "reproduce_douzero.sh",
        "train_belief.sh",
        "train_cardplay.sh",
        "train_full_game.sh",
        "run_crossplay.sh",
    ):
        content = (REPOSITORY_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")


def test_evaluation_cli_composes_fixed_bidder_with_cardplay_baselines(
    tmp_path: Path,
) -> None:
    output = tmp_path / "complete.json"

    assert (
        evaluate_main(
            (
                "--rules",
                str(REPOSITORY_ROOT / "configs" / "rules" / "canonical_full.yaml"),
                "--candidate",
                "longest_move",
                "--baseline",
                "first_legal",
                "--deals",
                "1",
                "--bootstrap-resamples",
                "100",
                "--output",
                str(output),
            )
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rules_profile"] == "canonical_full"
    assert payload["match_count"] == 6
    assert len(payload["matches"]) == 6
    assert all(match["bidding_record_json"] != "[]" for match in payload["matches"])
    assert payload["game_performance"]["candidate:longest_move"]["raw_score"]["sample_count"] > 0


def test_named_checkpoint_options_reject_malformed_and_duplicate_names() -> None:
    assert parse_named_checkpoints(("champion=model.pt",), "policy")["champion"].is_absolute()
    try:
        parse_named_checkpoints(("missing-separator",), "policy")
    except ValueError as error:
        assert "NAME=PATH" in str(error)
    else:
        raise AssertionError("malformed policy definition was accepted")
    try:
        parse_named_checkpoints(("old=a.pt", "old=b.pt"), "policy")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate policy name was accepted")
