"""Optional local integration check against the fetched official source."""

from pathlib import Path

import pytest

from birddou.eval.douzero_differential import load_manifest, run_differential


@pytest.mark.differential
def test_three_synchronized_games_match_the_pinned_douzero_engine() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest_path = repository_root / "artifacts/baselines/douzero/manifest.toml"
    manifest = load_manifest(manifest_path)
    source = manifest_path.parent / manifest.source_directory
    if not source.is_dir():
        pytest.skip("DouZero source not fetched; run scripts/fetch_douzero_baseline.py")

    report = run_differential(
        repository_root=repository_root,
        source=source,
        manifest=manifest,
        games=3,
        seed=7008,
    )
    assert report.games == 3
    assert report.compared_states > report.games
    assert report.applied_actions > 0
