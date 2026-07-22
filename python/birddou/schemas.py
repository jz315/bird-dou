"""Machine-readable identities for public tensor contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, cast

from birddou.features.ragged import (
    ACTION_META_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    HISTORY_META_COLUMNS,
    RANK_CATEGORICAL_COLUMNS,
    RANK_NUMERIC_COLUMNS,
    SCALAR_COLUMNS,
    FeatureConfig,
    RaggedBatch,
)

Dimension = int | Literal["B", "B+1", "H", "M"]


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    """Dtype, symbolic shape, and visibility of one tensor field."""

    name: str
    dtype: str
    shape: tuple[Dimension, ...]
    visibility: Literal["public", "training_target"] = "public"

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.shape:
            raise ValueError("tensor field name, dtype, and shape must be non-empty")
        if any(isinstance(value, int) and value <= 0 for value in self.shape):
            raise ValueError("fixed tensor dimensions must be positive")


@dataclass(frozen=True, slots=True)
class TensorSchema:
    """Versioned ordered field registry with a stable fingerprint."""

    name: str
    schema_version: int
    fields: tuple[TensorFieldSpec, ...]

    def __post_init__(self) -> None:
        if not self.name or self.schema_version <= 0 or not self.fields:
            raise ValueError("tensor schema identity and fields must be non-empty")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("tensor schema field names must be unique")

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


RAGGED_BATCH_SCHEMA = TensorSchema(
    name="birddou_ragged_batch",
    schema_version=FEATURE_SCHEMA_VERSION,
    fields=(
        TensorFieldSpec("rank_categorical", "int64", ("B", 15, len(RANK_CATEGORICAL_COLUMNS))),
        TensorFieldSpec("rank_numeric", "float32", ("B", 15, len(RANK_NUMERIC_COLUMNS))),
        TensorFieldSpec("history_rank_counts", "int64", ("B", "H", 15)),
        TensorFieldSpec("history_meta", "int64", ("B", "H", len(HISTORY_META_COLUMNS))),
        TensorFieldSpec("history_mask", "bool", ("B", "H")),
        TensorFieldSpec("scalars", "float32", ("B", len(SCALAR_COLUMNS))),
        TensorFieldSpec("action_rank_counts", "int64", ("M", 15)),
        TensorFieldSpec("post_hand_counts", "int64", ("M", 15)),
        TensorFieldSpec("action_meta", "int64", ("M", len(ACTION_META_COLUMNS))),
        TensorFieldSpec("action_state_index", "int64", ("M",)),
        TensorFieldSpec("action_offsets", "int64", ("B+1",)),
        TensorFieldSpec(
            "chosen_action_flat_index",
            "int64",
            ("B",),
            visibility="training_target",
        ),
    ),
)

__all__ = (
    "Dimension",
    "FeatureConfig",
    "RAGGED_BATCH_SCHEMA",
    "RaggedBatch",
    "TensorFieldSpec",
    "TensorSchema",
)
