"""Export strict compact inference weights and their observation-only manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from birddou import load_rule_config
from birddou.models.bid_head import BidHead, load_bid_head_config
from birddou.models.deployment import (
    CompactPolicyModel,
    export_deployment_bundle,
    load_compact_policy_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the deployment-export command line contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-config", type=Path, required=True)
    parser.add_argument("--compact-weights", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bid-config", type=Path)
    parser.add_argument("--bid-weights", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load strict state dictionaries and export a checksum-bound service bundle."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if (arguments.bid_config is None) != (arguments.bid_weights is None):
        parser.error("--bid-config and --bid-weights must be supplied together")

    rules = load_rule_config(arguments.rules)
    compact = CompactPolicyModel(load_compact_policy_config(arguments.compact_config))
    compact.load_state_dict(_load_state_dict(arguments.compact_weights), strict=True)
    bid_head: BidHead | None = None
    if arguments.bid_config is not None and arguments.bid_weights is not None:
        bid_head = BidHead(load_bid_head_config(arguments.bid_config))
        bid_head.load_state_dict(_load_state_dict(arguments.bid_weights), strict=True)
    manifest = export_deployment_bundle(arguments.output, compact, rules, bid_head=bid_head)
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


def _load_state_dict(path: Path) -> Mapping[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"{path} must contain a string-keyed state dictionary")
    values = cast(Mapping[str, object], payload)
    for key in ("state_dict", "model_state_dict", "compact_state_dict", "bid_state_dict"):
        nested = values.get(key)
        if isinstance(nested, Mapping):
            values = cast(Mapping[str, object], nested)
            break
    if not values or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in values.items()
    ):
        raise ValueError(f"{path} does not contain a tensor state dictionary")
    return cast(Mapping[str, Tensor], values)


if __name__ == "__main__":
    raise SystemExit(main())
