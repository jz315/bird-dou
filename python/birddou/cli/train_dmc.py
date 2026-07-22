"""Run the versioned E015 single-actor DMC smoke-training gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from birddou.rl.dmc import DmcTrainer, load_dmc_config

DEFAULT_CONFIG = Path("configs/train/dmc_smoke.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, checkpoint, resume, and evaluate the exact DouZero baseline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_dmc_config(arguments.config)
    if arguments.episodes is not None:
        config = replace(config, episodes=arguments.episodes)
    if arguments.output_directory is not None:
        config = replace(config, output_directory=arguments.output_directory.resolve())
    trainer = DmcTrainer(config)
    if arguments.resume:
        trainer.load_checkpoint()
    result = trainer.train()
    payload: dict[str, object] = {
        "schema_version": 1,
        "trainer_mode": "dmc",
        "checkpoint": str(result.checkpoint_path),
        "manifest": str(result.manifest_path),
        "training_state": {
            "episodes": result.state.episodes,
            "frames": result.state.frames,
            "learner_updates": result.state.learner_updates,
            "policy_version": result.state.policy_version,
        },
        "role_losses": dict(result.role_losses),
        "metric_count": len(result.metrics_history),
    }
    exit_code = 0
    if not arguments.skip_evaluation:
        evaluation = trainer.evaluate_against_random()
        payload["evaluation"] = evaluation.report.to_dict()
        payload["beats_random"] = evaluation.beats_random
        if config.require_beats_random and not evaluation.beats_random:
            exit_code = 2
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.report is None:
        print(rendered, end="")
    else:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
