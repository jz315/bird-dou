"""Acceptance coverage for the E001 repository skeleton."""

from pathlib import Path

REQUIRED_PATHS: tuple[str, ...] = (
    "Cargo.toml",
    "pyproject.toml",
    "README.md",
    ".github/workflows/ci.yml",
    "docs/IMPLEMENTATION_PLAN.md",
    "docs/RULES.md",
    "docs/FEATURE_SCHEMA.md",
    "docs/MODEL_ARCHITECTURE.md",
    "docs/TRAINING.md",
    "docs/EVALUATION.md",
    "docs/LICENSE_AUDIT.md",
    "crates/ddz-core/src/lib.rs",
    "crates/ddz-rules/src/lib.rs",
    "crates/ddz-batch/src/lib.rs",
    "crates/ddz-search/src/lib.rs",
    "crates/ddz-pyo3/src/lib.rs",
    "python/birddou/__init__.py",
    "python/birddou/features",
    "python/birddou/belief",
    "python/birddou/models",
    "python/birddou/rl",
    "python/birddou/actors",
    "python/birddou/league",
    "python/birddou/eval",
    "python/birddou/cli",
    "configs/rules",
    "configs/model",
    "configs/train",
    "configs/eval",
    "tests/rust",
    "tests/differential",
    "tests/golden_replays",
    "tests/performance",
    "scripts",
    "artifacts",
)


def test_required_repository_paths_exist() -> None:
    """Ensure the architecture's top-level boundaries remain present."""
    repository_root = Path(__file__).resolve().parents[2]
    missing = [path for path in REQUIRED_PATHS if not (repository_root / path).exists()]

    assert not missing, f"missing E001 repository paths: {missing}"
