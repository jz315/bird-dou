"""Smoke tests for the installable Python package."""

import tomllib
from pathlib import Path
from typing import cast

import birddou


def test_version_matches_project_metadata() -> None:
    """Keep the public package version synchronized with build metadata."""
    repository_root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((repository_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert cast(str, metadata["project"]["version"]) == birddou.__version__
