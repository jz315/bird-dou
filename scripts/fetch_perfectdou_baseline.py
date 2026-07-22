"""Fetch and checksum the pinned official PerfectDou evaluation release."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class RequiredFile:
    relative_path: str
    size: int
    sha256: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/baselines/perfectdou/manifest.toml"),
    )
    arguments = parser.parse_args()
    manifest_path = cast(Path, arguments.manifest).resolve()
    manifest = _mapping(tomllib.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    source = manifest_path.parent / _string(manifest, "source_directory")
    repository = _string(manifest, "repository")
    commit = _string(manifest, "commit")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        _run(("git", "clone", "--filter=blob:none", "--no-checkout", repository, str(source)))
    elif not (source / ".git").is_dir():
        raise RuntimeError(f"refusing to replace non-git path {source}")
    elif _run(("git", "-C", str(source), "status", "--porcelain")):
        raise RuntimeError(f"refusing to change dirty baseline checkout {source}")
    _run(("git", "-C", str(source), "fetch", "--depth", "1", "origin", commit))
    _run(("git", "-C", str(source), "checkout", "--detach", commit))
    actual = _run(("git", "-C", str(source), "rev-parse", "HEAD"))
    if actual != commit:
        raise RuntimeError(f"PerfectDou checkout resolved to {actual}; expected {commit}")
    for entry in _required_files(manifest):
        path = (source / entry.relative_path).resolve()
        try:
            path.relative_to(source.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"PerfectDou file escapes checkout: {entry.relative_path}"
            ) from error
        if not path.is_file() or path.stat().st_size != entry.size:
            raise RuntimeError(f"PerfectDou artifact size mismatch: {entry.relative_path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.sha256:
            raise RuntimeError(f"PerfectDou artifact checksum mismatch: {entry.relative_path}")
    print(f"verified PerfectDou source and weights at {source}")
    print(f"commit={actual}")
    return 0


def _required_files(manifest: Mapping[str, object]) -> tuple[RequiredFile, ...]:
    raw = manifest.get("required_files")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("PerfectDou manifest required_files must be a non-empty array")
    result: list[RequiredFile] = []
    for item in raw:
        values = _mapping(item, "required_files item")
        size = values.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError("PerfectDou artifact size must be a positive integer")
        result.append(
            RequiredFile(
                relative_path=_string(values, "relative_path"),
                size=size,
                sha256=_string(values, "sha256"),
            )
        )
    return tuple(result)


def _run(command: tuple[str, ...]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"PerfectDou manifest {key} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
