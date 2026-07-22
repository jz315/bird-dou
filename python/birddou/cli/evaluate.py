"""Run a deterministic paired Arena comparison with built-in or DouZero policies."""

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
from birddou.eval.arena import Arena
from birddou.eval.baselines import FixedBidPolicy, Policy, make_builtin_policy
from birddou.eval.bootstrap import BootstrapConfig
from birddou.eval.metrics import summarize_game_performance
from birddou.eval.paired_deals import generate_paired_deals
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
    """Build the public E012 evaluation command parser."""
    parser = argparse.ArgumentParser(
        description="Run a fixed-deal, role-balanced paired BIRD-Dou evaluation."
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--candidate", default="longest_move")
    parser.add_argument("--baseline", default="first_legal")
    parser.add_argument("--deals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--candidate-random-seed", type=int, default=1)
    parser.add_argument("--baseline-random-seed", type=int, default=2)
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
    """Execute a paired evaluation and emit a stable JSON artifact."""
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
    for label, name in (("candidate", arguments.candidate), ("baseline", arguments.baseline)):
        if name not in available:
            raise ValueError(f"unknown {label} policy: {name}")
    candidate_id = f"candidate:{arguments.candidate}"
    baseline_id = f"baseline:{arguments.baseline}"
    candidate = _make_policy(
        arguments.candidate,
        candidate_id,
        arguments.candidate_random_seed,
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
    baseline = _make_policy(
        arguments.baseline,
        baseline_id,
        arguments.baseline_random_seed,
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
    arena = Arena(rules, (candidate, baseline))
    deal_set = generate_paired_deals(arguments.seed, arguments.deals)
    run = arena.evaluate_paired(
        deal_set,
        candidate_id,
        baseline_id,
        BootstrapConfig(
            confidence_level=arguments.confidence_level,
            resamples=arguments.bootstrap_resamples,
            seed=arguments.bootstrap_seed,
        ),
    )
    matches = tuple(
        match for result in run.results for match in (result.candidate_match, result.baseline_match)
    )
    report_payload = run.report.to_dict()
    report_payload["rules_profile"] = rules["profile"]
    report_payload["game_performance"] = {
        candidate_id: summarize_game_performance(matches, candidate_id).to_dict(),
        baseline_id: summarize_game_performance(matches, baseline_id).to_dict(),
    }
    report_payload["matches"] = [asdict(match) for match in matches]
    payload = json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    return 0


def _make_policy(
    name: str,
    policy_id: str,
    seed: int,
    douzero_manifest: Path,
    device: str,
    douzero_feature_encoder: str,
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
            douzero_feature_encoder,
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
