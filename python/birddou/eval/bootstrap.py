"""Deterministic bounded-memory bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import sqrt
from typing import cast

import numpy as np
from numpy.typing import NDArray

BOOTSTRAP_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Versioned percentile-bootstrap controls."""

    confidence_level: float = 0.95
    resamples: int = 10_000
    seed: int = 20260722
    max_chunk_elements: int = 1_000_000

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1")
        if self.resamples < 100:
            raise ValueError("resamples must be at least 100")
        if not 0 <= self.seed < 1 << 64:
            raise ValueError("bootstrap seed must fit uint64")
        if self.max_chunk_elements <= 0:
            raise ValueError("max_chunk_elements must be positive")


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Percentile confidence interval for one scalar mean."""

    schema_version: int
    sample_count: int
    point_estimate: float
    standard_error: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    seed: int

    @property
    def width(self) -> float:
        """Full confidence interval width."""
        return self.upper - self.lower

    @property
    def half_width(self) -> float:
        """Half confidence interval width used by precision stopping rules."""
        return self.width / 2.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation including derived widths."""
        result = cast(dict[str, object], asdict(self))
        result["width"] = self.width
        result["half_width"] = self.half_width
        return result


def bootstrap_mean_ci(
    values: Iterable[float],
    config: BootstrapConfig | None = None,
) -> BootstrapCI:
    """Estimate a mean CI by resampling independent clusters with replacement."""
    settings = config if config is not None else BootstrapConfig()
    samples = _finite_samples(values, "values")
    sample_count = len(samples)
    point_estimate = float(samples.mean())
    standard_error = float(samples.std(ddof=1) / sqrt(sample_count)) if sample_count > 1 else 0.0

    random = np.random.Generator(np.random.PCG64(settings.seed))
    bootstrap_means = np.empty(settings.resamples, dtype=np.float64)
    completed = 0
    chunk_size = max(1, settings.max_chunk_elements // sample_count)
    while completed < settings.resamples:
        current = min(chunk_size, settings.resamples - completed)
        indices = random.integers(
            0,
            sample_count,
            size=(current, sample_count),
            dtype=np.int64,
        )
        bootstrap_means[completed : completed + current] = samples[indices].mean(axis=1)
        completed += current

    tail = (1.0 - settings.confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail], method="linear")
    return BootstrapCI(
        schema_version=BOOTSTRAP_SCHEMA_VERSION,
        sample_count=sample_count,
        point_estimate=point_estimate,
        standard_error=standard_error,
        lower=float(lower),
        upper=float(upper),
        confidence_level=settings.confidence_level,
        resamples=settings.resamples,
        seed=settings.seed,
    )


def bootstrap_paired_difference_ci(
    candidate: Iterable[float],
    baseline: Iterable[float],
    config: BootstrapConfig | None = None,
) -> BootstrapCI:
    """Bootstrap the paired mean of `candidate - baseline`."""
    candidate_samples = _finite_samples(candidate, "candidate")
    baseline_samples = _finite_samples(baseline, "baseline")
    if candidate_samples.shape != baseline_samples.shape:
        raise ValueError("candidate and baseline must contain the same number of paired samples")
    return bootstrap_mean_ci(candidate_samples - baseline_samples, config)


def _finite_samples(values: Iterable[float], label: str) -> NDArray[np.float64]:
    samples = np.asarray(tuple(values), dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if samples.size == 0:
        raise ValueError(f"{label} must contain at least one sample")
    if not np.isfinite(samples).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return samples


__all__ = (
    "BOOTSTRAP_SCHEMA_VERSION",
    "BootstrapCI",
    "BootstrapConfig",
    "bootstrap_mean_ci",
    "bootstrap_paired_difference_ci",
)
