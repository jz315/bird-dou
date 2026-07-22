"""Train, resume, and report metric-gated complete-game BIRD-Dou."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from birddou.rl.full_game import FullGameTrainer, load_full_game_config

DEFAULT_CONFIG = Path("configs/train/full_game_smoke.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Bid Head and Cardplay through the metric-gated full-game curriculum."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_full_game_config(arguments.config)
    if arguments.episodes is not None:
        config = replace(config, episodes=arguments.episodes)
    if arguments.output_directory is not None:
        config = replace(config, output_directory=arguments.output_directory.resolve())
    trainer = FullGameTrainer(config)
    if arguments.resume:
        trainer.load_checkpoint()
    result = trainer.train()
    payload = {
        "schema_version": 1,
        "trainer_mode": config.trainer_mode,
        "checkpoint": str(result.checkpoint_path),
        "manifest": str(result.manifest_path),
        "training_state": asdict(result.state),
        "losses": dict(result.losses),
        "metric_count": len(result.metrics_history),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.report is None:
        print(rendered, end="")
    else:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
