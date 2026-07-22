"""Run legal-action and synchronized-game differential checks against DouZero."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from birddou.eval.douzero_differential import (
    DifferentialError,
    load_manifest,
    run_differential,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/baselines/douzero/manifest.toml"),
    )
    parser.add_argument(
        "--douzero-source",
        type=Path,
        help="override the manifest-relative ignored source checkout",
    )
    parser.add_argument(
        "--rust-probe",
        type=Path,
        help="prebuilt differential_probe executable; cargo run is used by default",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = cast(Path, args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    source_argument = cast(Path | None, args.douzero_source)
    source = (
        source_argument.resolve()
        if source_argument is not None
        else manifest_path.parent / manifest.source_directory
    )
    probe_argument = cast(Path | None, args.rust_probe)
    rust_command = [str(probe_argument.resolve())] if probe_argument is not None else None

    try:
        report = run_differential(
            repository_root=repository_root,
            source=source,
            manifest=manifest,
            games=cast(int, args.games),
            seed=cast(int, args.seed),
            rust_command=rust_command,
        )
    except DifferentialError as error:
        print(f"differential failed: {error}", file=sys.stderr)
        return 1

    rendered = report.to_json() + "\n"
    output = cast(Path | None, args.json_output)
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
