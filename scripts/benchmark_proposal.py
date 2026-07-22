"""Compare full BIRD-Dou scoring with Proposal + dynamic Top-K on identical states."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch

from birddou import PyDdzEnv, load_rule_config
from birddou.env_types import Action, Observation
from birddou.features.ragged import FeatureConfig, encode_ragged_batch
from birddou.models.bird_dou import BirdDouModel, load_bird_dou_config
from birddou.models.proposal import (
    ProposalNetwork,
    load_proposal_config,
    select_proposals,
    subset_ragged_batch,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--assert-min-speedup", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.states <= 0 or args.iterations <= 0 or args.warmup < 0 or args.threads <= 0:
        parser.error("states/iterations/threads must be positive and warmup non-negative")
    torch.set_num_threads(args.threads)
    rules = load_rule_config(ROOT / "configs" / "rules" / "douzero_post_bid.yaml")
    observations: list[Observation] = []
    actions: list[tuple[Action, ...]] = []
    for index in range(args.states):
        environment = PyDdzEnv()
        observation = environment.reset(args.seed + index, rules)
        observations.append(observation)
        actions.append(tuple(environment.legal_actions()))
    batch = encode_ragged_batch(
        observations,
        actions,
        rules,
        config=FeatureConfig(decomposition_features=True),
    ).to(args.device)
    full_model = (
        BirdDouModel(load_bird_dou_config(ROOT / "configs" / "model" / "bird_dou_v1.yaml"))
        .to(args.device)
        .eval()
    )
    proposal = (
        ProposalNetwork(load_proposal_config(ROOT / "configs" / "model" / "proposal_v1.yaml"))
        .to(args.device)
        .eval()
    )
    uncertainty = torch.full((batch.batch_size,), 0.5, device=args.device)

    def full_forward() -> None:
        full_model(batch)

    selected_count = 0

    def pruned_forward() -> None:
        nonlocal selected_count
        proposal_scores = proposal(batch).score
        selection = select_proposals(batch, proposal_scores, proposal.config, uncertainty)
        selected_count = int(selection.selected_flat_index.numel())
        full_model(subset_ragged_batch(batch, selection))

    with torch.inference_mode():
        for _ in range(args.warmup):
            full_forward()
            pruned_forward()
        full_seconds = _measure(full_forward, args.iterations, args.device)
        pruned_seconds = _measure(pruned_forward, args.iterations, args.device)
    speedup = full_seconds / pruned_seconds
    report = {
        "schema_version": 1,
        "device": args.device,
        "threads": args.threads,
        "states": batch.batch_size,
        "full_action_count": batch.action_count,
        "selected_action_count": selected_count,
        "retained_fraction": selected_count / batch.action_count,
        "iterations": args.iterations,
        "full_seconds": full_seconds,
        "proposal_pruned_seconds": pruned_seconds,
        "wall_clock_speedup": speedup,
        "full_effective_actions_per_second": batch.action_count * args.iterations / full_seconds,
        "pruned_effective_actions_per_second": batch.action_count
        * args.iterations
        / pruned_seconds,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    output = cast(Path | None, args.output)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    if args.assert_min_speedup is not None and speedup < args.assert_min_speedup:
        raise SystemExit(
            f"Proposal speedup {speedup:.3f} is below required {args.assert_min_speedup:.3f}"
        )
    return 0


def _measure(callback: Callable[[], None], iterations: int, device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        callback()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - started


if __name__ == "__main__":
    raise SystemExit(main())
