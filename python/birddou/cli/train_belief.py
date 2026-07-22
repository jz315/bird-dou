"""Offline pretrain the constrained Belief CRF and write calibration artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch

from birddou.belief import belief_nll, calibration_report, uniform_belief_nll
from birddou.belief.data import load_belief_dataset
from birddou.belief.training import (
    BeliefOfflineTrainer,
    BeliefPretrainConfig,
    save_calibration_json,
)
from birddou.models.belief_bird_dou import (
    BeliefBirdDouModel,
    belief_constraints_from_batch,
    load_belief_bird_dou_config,
)

DEFAULT_CONFIG = Path("configs/train/belief_pretrain.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain hidden-hand belief with exact cardinality NLL."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config_path = arguments.config.resolve()
    raw = _mapping(json.loads(config_path.read_text(encoding="utf-8")), "config")
    root = config_path.parents[2]
    dataset_path = _project_path(root, _string(raw, "dataset_path"))
    model_path = _project_path(root, _string(raw, "model_path"))
    output_directory = _project_path(root, _string(raw, "output_directory"))
    training = _mapping(raw.get("training"), "training")
    pretrain = BeliefPretrainConfig(
        schema_version=_integer(raw, "schema_version"),
        epochs=_integer(training, "epochs"),
        batch_size=_integer(training, "batch_size"),
        learning_rate=_number(training, "learning_rate"),
        weight_decay=_number(training, "weight_decay"),
        max_grad_norm=_number(training, "max_grad_norm"),
        device=_string(training, "device"),
        freeze_public_encoder=_boolean(training, "freeze_public_encoder"),
        seed=_integer(training, "seed"),
    )
    dataset = load_belief_dataset(dataset_path)
    torch.manual_seed(pretrain.seed)
    model = BeliefBirdDouModel(load_belief_bird_dou_config(model_path))
    trainer = BeliefOfflineTrainer(model, pretrain)
    output_directory.mkdir(parents=True, exist_ok=True)
    result = trainer.train(dataset, output_directory / "belief_pretrain.pt")

    model.eval()
    batch = dataset.batch.to(pretrain.device)
    labels = dataset.true_assignment_a.to(pretrain.device)
    with torch.inference_mode():
        encoding = model.encode_belief(batch)
        unknown, capacity_a, _ = belief_constraints_from_batch(batch)
        trained_nll = float(
            belief_nll(encoding.scores.float(), unknown, capacity_a, labels).cpu().item()
        )
        uniform_nll = float(uniform_belief_nll(unknown, capacity_a).cpu().item())
        key_targets = torch.stack(
            (
                (labels[:, 12] > 0).to(torch.float32),
                (labels[:, 13] > 0).to(torch.float32),
                (labels[:, 14] > 0).to(torch.float32),
                (labels[:, :13] == 4).any(dim=1).to(torch.float32),
            ),
            dim=-1,
        )
    names = ("two", "small_joker", "big_joker", "any_bomb")
    reports = {
        name: calibration_report(
            encoding.marginals.key_probability_a[:, index].cpu(),
            key_targets[:, index].cpu(),
            _integer(raw, "calibration_bins"),
        )
        for index, name in enumerate(names)
    }
    for name, report in reports.items():
        save_calibration_json(report, output_directory / f"calibration_{name}.json")
    payload = {
        "schema_version": 1,
        "checkpoint": str(result.checkpoint_path),
        "updates": result.update_count,
        "states": dataset.state_count,
        "trained_nll": trained_nll,
        "uniform_nll": uniform_nll,
        "calibration": {
            name: {
                "brier_score": report.brier_score,
                "expected_calibration_error": report.expected_calibration_error,
            }
            for name, report in reports.items()
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    report_path = arguments.report
    if report_path is None:
        print(rendered, end="")
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    return 0


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config {key} must be a non-empty string")
    return value


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"config {key} must be numeric")
    return float(value)


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"config {key} must be boolean")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
