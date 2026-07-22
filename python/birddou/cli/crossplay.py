"""Run a complete fixed-deal landlord-policy by farmer-policy cross-play matrix."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from birddou import load_rule_config
from birddou.cli.policy_artifacts import (
    load_cardplay_checkpoint_policy,
    load_dmc_checkpoint_policy,
    load_full_game_checkpoint_policy,
    parse_named_checkpoints,
)
from birddou.env_types import RuleConfig
from birddou.eval import Arena, BootstrapConfig, Policy, generate_paired_deals
from birddou.eval.baselines import FixedBidPolicy, make_builtin_policy
from birddou.eval.metrics import summarize_game_performance
from birddou.eval.perfectdou_baseline import (
    PERFECTDOU_BASELINE_ID,
    PerfectDouPolicy,
    PerfectDouProcessBackend,
)
from birddou.eval.rlcard_baseline import RLCARD_BASELINE_ID, RlcardRulePolicy

DEFAULT_RULES = Path("configs/rules/douzero_post_bid.yaml")
DEFAULT_DOUZERO_MANIFEST = Path("artifacts/baselines/douzero/manifest.toml")
DEFAULT_BIRD_MODEL = Path("configs/model/bird_dou_v1.yaml")
DEFAULT_BIRD_FEATURES = Path("configs/model/bird_dou_features_v1.yaml")
DEFAULT_BID_MODEL = Path("configs/model/bid_head_v1.yaml")
BUILTIN_POLICIES = ("first_legal", "longest_move", "seeded_random")
OFFICIAL_POLICIES = ("douzero_ADP", "douzero_WP")
POLICY_CHOICES = (
    BUILTIN_POLICIES
    + OFFICIAL_POLICIES
    + (
        RLCARD_BASELINE_ID,
        PERFECTDOU_BASELINE_ID,
    )
)


def build_parser() -> argparse.ArgumentParser:
    """Build the reproducible cross-play command parser."""
    parser = argparse.ArgumentParser(
        description="Run an ordered landlord-policy by farmer-team-policy matrix."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--landlord-policies", default="douzero_ADP,douzero_WP")
    parser.add_argument("--farmer-policies", default="douzero_ADP,douzero_WP")
    parser.add_argument("--deals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--douzero-manifest", type=Path, default=DEFAULT_DOUZERO_MANIFEST)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--bird-dou-policy",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Register a current, historical, or exploiter card-play checkpoint.",
    )
    parser.add_argument(
        "--dmc-policy",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Register a three-role exact-DouZero DMC checkpoint.",
    )
    parser.add_argument(
        "--full-game-policy",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Register a joint Bid Head/Cardplay checkpoint.",
    )
    parser.add_argument("--bird-dou-model-config", type=Path, default=DEFAULT_BIRD_MODEL)
    parser.add_argument("--bird-dou-feature-config", type=Path, default=DEFAULT_BIRD_FEATURES)
    parser.add_argument("--bid-model-config", type=Path, default=DEFAULT_BID_MODEL)
    parser.add_argument("--fixed-bid-score", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--perfectdou-command",
        help="Quoted Python-3.7/Linux JSONL worker command required by perfectdou.",
    )
    parser.add_argument(
        "--douzero-feature-encoder",
        choices=("native", "official_reference"),
        default="native",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute and serialize one complete cross-play matrix."""
    arguments = build_parser().parse_args(argv)
    rules = load_rule_config(arguments.rules)
    bird_checkpoints = parse_named_checkpoints(arguments.bird_dou_policy, "bird-dou policy")
    dmc_checkpoints = parse_named_checkpoints(arguments.dmc_policy, "dmc policy")
    full_checkpoints = parse_named_checkpoints(arguments.full_game_policy, "full-game policy")
    _validate_registry(bird_checkpoints, dmc_checkpoints, full_checkpoints)
    available = (
        set(POLICY_CHOICES)
        | bird_checkpoints.keys()
        | dmc_checkpoints.keys()
        | full_checkpoints.keys()
    )
    landlord_names = _policy_names(arguments.landlord_policies, "landlord", available)
    farmer_names = _policy_names(arguments.farmer_policies, "farmer", available)
    names = tuple(dict.fromkeys((*landlord_names, *farmer_names)))
    policies = tuple(
        _make_policy(
            name,
            f"crossplay:{name}",
            arguments.seed + index,
            arguments.douzero_manifest,
            arguments.device,
            arguments.douzero_feature_encoder,
            arguments.perfectdou_command,
            bird_checkpoints,
            dmc_checkpoints,
            full_checkpoints,
            arguments.bird_dou_model_config,
            arguments.bird_dou_feature_config,
            arguments.bid_model_config,
            rules,
            arguments.fixed_bid_score,
        )
        for index, name in enumerate(names)
    )
    landlord_ids = tuple(f"crossplay:{name}" for name in landlord_names)
    farmer_ids = tuple(f"crossplay:{name}" for name in farmer_names)
    arena = Arena(rules, policies)
    run = arena.evaluate_cross_play(
        generate_paired_deals(arguments.seed, arguments.deals),
        landlord_ids,
        farmer_ids,
        BootstrapConfig(
            confidence_level=arguments.confidence_level,
            resamples=arguments.bootstrap_resamples,
            seed=arguments.bootstrap_seed,
        ),
    )
    report_payload = run.report.to_dict()
    report_payload["rules_profile"] = rules["profile"]
    report_payload["game_performance"] = {
        policy_id: summarize_game_performance(run.results, policy_id).to_dict()
        for policy_id in dict.fromkeys((*landlord_ids, *farmer_ids))
    }
    report_payload["matches"] = [asdict(match) for match in run.results]
    rendered = json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0


def _policy_names(raw: str, label: str, available: set[str] | None = None) -> tuple[str, ...]:
    names = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not names:
        raise ValueError(f"{label} policy list must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} policy list contains duplicates")
    known = set(POLICY_CHOICES) if available is None else available
    unknown = set(names) - known
    if unknown:
        raise ValueError(f"unknown {label} policies: {sorted(unknown)}")
    return names


def _make_policy(
    name: str,
    policy_id: str,
    seed: int,
    douzero_manifest: Path,
    device: str,
    feature_encoder: str,
    perfectdou_command: str | None,
    bird_checkpoints: dict[str, Path],
    dmc_checkpoints: dict[str, Path],
    full_checkpoints: dict[str, Path],
    bird_model_config: Path,
    feature_config: Path,
    bid_model_config: Path,
    rules: RuleConfig,
    fixed_bid_score: int,
) -> Policy:
    if name in full_checkpoints:
        return load_full_game_checkpoint_policy(
            policy_id,
            full_checkpoints[name],
            bid_model_config,
            bird_model_config,
            feature_config,
            rules,
            device,
        )
    if name in dmc_checkpoints:
        policy = load_dmc_checkpoint_policy(policy_id, dmc_checkpoints[name], device)
    elif name in bird_checkpoints:
        policy = load_cardplay_checkpoint_policy(
            policy_id,
            bird_checkpoints[name],
            bird_model_config,
            feature_config,
            rules,
            device,
        )
    elif name == PERFECTDOU_BASELINE_ID:
        if not perfectdou_command:
            raise ValueError("perfectdou requires --perfectdou-command")
        policy = PerfectDouPolicy(
            policy_id,
            PerfectDouProcessBackend(shlex.split(perfectdou_command)),
        )
    elif name == RLCARD_BASELINE_ID:
        policy = RlcardRulePolicy(policy_id, seed)
    elif name in OFFICIAL_POLICIES:
        from birddou.models.baseline_douzero import OfficialDouZeroPolicy

        policy = OfficialDouZeroPolicy.from_manifest(
            policy_id,
            douzero_manifest,
            name,
            device,
            feature_encoder,
        )
    else:
        policy = make_builtin_policy(name, policy_id, seed)
    if rules["profile"] == "canonical_full":
        return FixedBidPolicy(policy_id, policy, score_bid=fixed_bid_score)
    return policy


def _validate_registry(
    bird_checkpoints: dict[str, Path],
    dmc_checkpoints: dict[str, Path],
    full_checkpoints: dict[str, Path],
) -> None:
    registries = (set(bird_checkpoints), set(dmc_checkpoints), set(full_checkpoints))
    names = set().union(*registries)
    conflicts = names & set(POLICY_CHOICES)
    duplicates = set().union(
        registries[0] & registries[1],
        registries[0] & registries[2],
        registries[1] & registries[2],
    )
    if conflicts:
        raise ValueError(f"checkpoint policy names conflict with built-ins: {sorted(conflicts)}")
    if duplicates:
        raise ValueError(f"checkpoint policy names are duplicated: {sorted(duplicates)}")


if __name__ == "__main__":
    raise SystemExit(main())
