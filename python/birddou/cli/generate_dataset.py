"""Generate leakage-isolated mixed-policy Belief supervision data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from birddou import load_rule_config
from birddou.belief.data import generate_belief_dataset, save_belief_dataset
from birddou.eval.baselines import make_builtin_policy
from birddou.features import load_feature_config

DEFAULT_RULES = Path("configs/rules/douzero_post_bid.yaml")
DEFAULT_FEATURES = Path("configs/model/bird_dou_features_v1.yaml")
DEFAULT_OUTPUT = Path("artifacts/datasets/belief_smoke.npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate public information sets with training-only hidden-hand labels."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5005)
    parser.add_argument(
        "--strategies",
        default="seeded_random,longest_move",
        help="Comma-separated built-ins: seeded_random, longest_move, first_legal.",
    )
    parser.add_argument("--exact-decomposition", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    names = tuple(value.strip() for value in arguments.strategies.split(",") if value.strip())
    if not names:
        raise ValueError("at least one dataset strategy is required")
    policies = tuple(
        make_builtin_policy(name, f"dataset:{name}:{index}", arguments.seed + index)
        for index, name in enumerate(names)
    )
    rules = load_rule_config(arguments.rules)
    features = replace(
        load_feature_config(arguments.features),
        decomposition_features=arguments.exact_decomposition,
    )
    dataset = generate_belief_dataset(
        arguments.games,
        arguments.seed,
        rules,
        policies,
        features,
    )
    artifact = save_belief_dataset(
        dataset,
        arguments.output,
        game_count=arguments.games,
        master_seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": str(artifact.dataset_path),
                "manifest": str(artifact.manifest_path),
                "sha256": artifact.sha256,
                "games": artifact.game_count,
                "states": artifact.state_count,
                "policy_counts": dict(artifact.policy_counts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
