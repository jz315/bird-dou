"""Combine differential, random, RLCard, and resume evidence for the M3 gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from birddou.eval import evaluate_m3_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--differential", type=Path, required=True)
    parser.add_argument("--random-evaluation", type=Path, required=True)
    parser.add_argument("--rlcard-evaluation", type=Path, required=True)
    parser.add_argument("--checkpoint-resume-exact", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    differential = _load(cast(Path, arguments.differential))
    report = evaluate_m3_gate(
        differential_mismatches=_integer(differential, "mismatches"),
        random_paired_ci_lower=_overall_win_lower(_load(cast(Path, arguments.random_evaluation))),
        rlcard_paired_ci_lower=_overall_win_lower(_load(cast(Path, arguments.rlcard_evaluation))),
        checkpoint_resume_exact=cast(bool, arguments.checkpoint_resume_exact),
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    output = cast(Path | None, arguments.output)
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if report.accepted else 2


def _load(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(raw, str(path))


def _overall_win_lower(root: Mapping[str, object]) -> float:
    evaluation = root.get("evaluation", root)
    report = _mapping(evaluation, "evaluation")
    roles = _mapping(report.get("roles"), "roles")
    overall = _mapping(roles.get("overall"), "overall")
    win_rate = _mapping(overall.get("win_rate"), "win_rate")
    interval = _mapping(win_rate.get("delta_ci"), "delta_ci")
    value = interval.get("lower")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("overall win-rate lower bound must be numeric")
    return float(value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
