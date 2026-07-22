"""Fetch and verify the pinned DouZero source and optional official weights."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class WeightFile:
    """One checksummed checkpoint declared by the tracked manifest."""

    weight_set: str
    role: str
    relative_path: str
    mirror_path: str
    size: int
    sha256: str


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/baselines/douzero/manifest.toml"),
    )
    parser.add_argument(
        "--weight-set",
        action="append",
        choices=("douzero_ADP", "douzero_WP"),
        default=[],
        help="also fetch one three-role checkpoint set (repeatable)",
    )
    parser.add_argument(
        "--skip-source",
        action="store_true",
        help="verify/download requested weights without touching the source checkout",
    )
    args = parser.parse_args()
    manifest_path = cast(Path, args.manifest).resolve()
    manifest = cast(dict[str, object], tomllib.loads(manifest_path.read_text(encoding="utf-8")))
    if not cast(bool, args.skip_source):
        _fetch_source(manifest_path, manifest)
    for weight_set in cast(list[str], args.weight_set):
        _fetch_weight_set(manifest_path, manifest, weight_set)
    return 0


def _fetch_source(manifest_path: Path, manifest: dict[str, object]) -> None:
    repository = _required_str(manifest, "repository")
    commit = _required_str(manifest, "commit")
    source = manifest_path.parent / _required_str(manifest, "source_directory")

    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(source)])
    elif not (source / ".git").is_dir():
        raise RuntimeError(f"refusing to replace non-git path {source}")
    elif _run(["git", "-C", str(source), "status", "--porcelain"]):
        raise RuntimeError(f"refusing to change dirty baseline checkout {source}")

    _run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", commit])
    _run(["git", "-C", str(source), "checkout", "--detach", commit])
    actual = _run(["git", "-C", str(source), "rev-parse", "HEAD"])
    if actual != commit:
        raise RuntimeError(f"checkout resolved to {actual}; expected {commit}")

    print(f"verified DouZero source at {source}")
    print(f"commit={actual}")


def _fetch_weight_set(
    manifest_path: Path,
    manifest: dict[str, object],
    weight_set: str,
) -> None:
    mirror = _required_mapping(manifest, "weight_mirror")
    base_url = _required_str(mirror, "base_url").rstrip("/")
    weights_root = (manifest_path.parent / _required_str(manifest, "weights_directory")).resolve()
    entries = [entry for entry in _weight_files(manifest) if entry.weight_set == weight_set]
    if {entry.role for entry in entries} != {"landlord", "landlord_down", "landlord_up"}:
        raise RuntimeError(f"manifest does not define a complete three-role set: {weight_set}")

    for entry in entries:
        destination = (weights_root / entry.relative_path).resolve()
        try:
            destination.relative_to(weights_root)
        except ValueError as error:
            raise RuntimeError(
                f"checkpoint escapes weights directory: {entry.relative_path}"
            ) from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _verify_weight(destination, entry)
        else:
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists():
                temporary.unlink()
            request = urllib.request.Request(
                f"{base_url}/{entry.mirror_path}",
                headers={"User-Agent": "BIRD-Dou-baseline-fetch/1"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
                _verify_weight(temporary, entry)
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        print(f"verified {weight_set}/{entry.role} at {destination} sha256={entry.sha256}")


def _verify_weight(path: Path, entry: WeightFile) -> None:
    if path.stat().st_size != entry.size:
        raise RuntimeError(
            f"checkpoint size mismatch for {path}: {path.stat().st_size} != {entry.size}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.sha256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch for {path}: {digest} != {entry.sha256}")


def _weight_files(manifest: dict[str, object]) -> tuple[WeightFile, ...]:
    raw_entries = manifest.get("weight_files")
    if not isinstance(raw_entries, list):
        raise RuntimeError("manifest weight_files must be an array of tables")
    entries: list[WeightFile] = []
    for raw in raw_entries:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise RuntimeError("each manifest weight_files item must be a table")
        values = cast(dict[str, object], raw)
        size = values.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError("checkpoint size must be a positive integer")
        entries.append(
            WeightFile(
                weight_set=_required_str(values, "set"),
                role=_required_str(values, "role"),
                relative_path=_required_str(values, "relative_path"),
                mirror_path=_required_str(values, "mirror_path"),
                size=size,
                sha256=_required_str(values, "sha256"),
            )
        )
    return tuple(entries)


def _required_mapping(values: dict[str, object], key: str) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, dict) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"manifest {key} must be a table")
    return cast(dict[str, object], value)


def _required_str(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"manifest {key} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
