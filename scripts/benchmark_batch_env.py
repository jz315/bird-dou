"""Compare per-environment Python objects with the packed E011 NumPy API."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

from birddou import Action, PackedActions, PyBatchDdzEnv, PyDdzEnv, RuleConfig, load_rule_config
from birddou.env_types import PlayGameAction


def parse_args() -> argparse.Namespace:
    """Parse reproducible benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("configs/rules/douzero_post_bid.yaml"),
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def action_size(action: Action) -> int:
    """Return the canonical card count used for deterministic selection."""
    return cast(PlayGameAction, action)["play"]["total_cards"]


def reset_seeds(seed: int, batch_size: int, round_index: int) -> np.ndarray:
    """Produce one non-overlapping deterministic seed range."""
    start = seed + round_index * batch_size
    return np.arange(start, start + batch_size, dtype=np.uint64)


def run_single_object_api(
    rules: RuleConfig,
    seed: int,
    batch_size: int,
    ticks: int,
) -> tuple[float, int]:
    """Run the E010 object/JSON boundary once per environment and tick."""
    environments = [PyDdzEnv() for _ in range(batch_size)]
    round_index = 0
    seeds = reset_seeds(seed, batch_size, round_index)
    for environment, deal_seed in zip(environments, seeds, strict=True):
        environment.reset(int(deal_seed), rules)

    transitions = 0
    started = perf_counter()
    for _ in range(ticks):
        if all(environment.terminal for environment in environments):
            round_index += 1
            seeds = reset_seeds(seed, batch_size, round_index)
            for environment, deal_seed in zip(environments, seeds, strict=True):
                environment.reset(int(deal_seed), rules)

        for environment in environments:
            if not environment.terminal:
                actions = environment.legal_actions()
                environment.step(max(actions, key=action_size))
                transitions += 1
            environment.observe(environment.current_player)
    return perf_counter() - started, transitions


def choose_packed_indices(actions: PackedActions) -> np.ndarray:
    """Choose the first maximum-card action in every non-empty ragged range."""
    batch_size = actions["batch_size"]
    indices = np.full(batch_size, -1, dtype=np.int64)
    for env_index in range(batch_size):
        start = int(actions["offsets"][env_index])
        end = int(actions["offsets"][env_index + 1])
        if start < end:
            indices[env_index] = int(np.argmax(actions["total_cards"][start:end]))
    return indices


def run_packed_batch_api(
    rules: RuleConfig,
    seed: int,
    batch_size: int,
    ticks: int,
) -> tuple[float, int]:
    """Run the E011 Rust batch and fixed-count NumPy boundary."""
    environment = PyBatchDdzEnv(rules)
    round_index = 0
    environment.reset(reset_seeds(seed, batch_size, round_index))

    transitions = 0
    started = perf_counter()
    for _ in range(ticks):
        if environment.all_terminal:
            round_index += 1
            environment.reset(reset_seeds(seed, batch_size, round_index))
        actions = environment.legal_actions_packed()
        result = environment.step_packed(choose_packed_indices(actions))
        transitions += int(result["acted"].sum())
    return perf_counter() - started, transitions


def median_throughput(samples: list[tuple[float, int]]) -> tuple[float, float]:
    """Return median elapsed seconds and transitions per second."""
    seconds = statistics.median(sample[0] for sample in samples)
    transitions = statistics.median(sample[1] for sample in samples)
    return seconds, transitions / seconds


def main() -> int:
    """Run warmups, paired repetitions, and print stable JSON metrics."""
    args = parse_args()
    if args.batch_size <= 0 or args.ticks <= 0 or args.repeats <= 0:
        raise ValueError("batch-size, ticks, and repeats must all be positive")
    rules = load_rule_config(args.rules)

    warmup_size = min(args.batch_size, 16)
    warmup_ticks = min(args.ticks, 8)
    run_single_object_api(rules, args.seed, warmup_size, warmup_ticks)
    run_packed_batch_api(rules, args.seed, warmup_size, warmup_ticks)

    single_samples: list[tuple[float, int]] = []
    batch_samples: list[tuple[float, int]] = []
    for _ in range(args.repeats):
        single = run_single_object_api(rules, args.seed, args.batch_size, args.ticks)
        packed = run_packed_batch_api(rules, args.seed, args.batch_size, args.ticks)
        if single[1] != packed[1]:
            raise RuntimeError(
                f"benchmark paths diverged: single={single[1]}, packed={packed[1]} transitions"
            )
        single_samples.append(single)
        batch_samples.append(packed)

    single_seconds, single_throughput = median_throughput(single_samples)
    batch_seconds, batch_throughput = median_throughput(batch_samples)
    report = {
        "schema_version": 1,
        "batch_size": args.batch_size,
        "ticks": args.ticks,
        "repeats": args.repeats,
        "seed": args.seed,
        "transitions_per_repeat": single_samples[0][1],
        "single_object_api_median_seconds": single_seconds,
        "single_object_api_transitions_per_second": single_throughput,
        "packed_batch_api_median_seconds": batch_seconds,
        "packed_batch_api_transitions_per_second": batch_throughput,
        "packed_vs_single_speedup": batch_throughput / single_throughput,
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(output, end="")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
