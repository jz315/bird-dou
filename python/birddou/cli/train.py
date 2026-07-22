"""Train, resume, and optionally evaluate BIRD-Dou no-Belief v1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from birddou.rl.bird_dou_dmc import BirdDouDmcTrainer, load_bird_dou_dmc_config

DEFAULT_CONFIG = Path("configs/train/bird_dou_dmc_smoke.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the shared, structured BIRD-Dou model in DMC mode."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_bird_dou_dmc_config(arguments.config)
    if arguments.episodes is not None:
        config = replace(config, episodes=arguments.episodes)
    if arguments.output_directory is not None:
        config = replace(config, output_directory=arguments.output_directory.resolve())
    trainer = BirdDouDmcTrainer(config)
    if arguments.resume:
        trainer.load_checkpoint()
    result = trainer.train()
    payload: dict[str, object] = {
        "schema_version": 1,
        "trainer_mode": "bird_dou_dmc",
        "checkpoint": str(result.checkpoint_path),
        "manifest": str(result.manifest_path),
        "training_state": asdict(result.state),
        "losses": dict(result.losses),
        "metric_count": len(result.metrics_history),
    }
    if arguments.evaluate:
        evaluation = trainer.evaluate_against_random()
        payload["evaluation"] = evaluation.report.to_dict()
        payload["beats_random"] = evaluation.beats_random
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.report is None:
        print(rendered, end="")
    else:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
