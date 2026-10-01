"""Seed-level aggregation for reviewer experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summary", payload)


def aggregate_condition(condition_dir: Path | str, required_seeds: list[int]) -> dict[str, float | int]:
    root = Path(condition_dir)
    summaries: list[dict] = []
    missing: list[int] = []
    for seed in required_seeds:
        candidates = list((root / f"seed_{seed}" / "metrics").glob("*summary.json"))
        if not candidates:
            missing.append(seed)
            continue
        summaries.append(_read_summary(candidates[0]))
    if missing:
        raise ValueError(f"missing seeds: {missing}")
    macro_f1 = np.asarray([float(summary["macro_f1"]) for summary in summaries])
    return {
        "seed_count": len(required_seeds),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--required_seeds", nargs="+", type=int, default=[42, 123, 2026])
    args = parser.parse_args()
    reports = {}
    for condition in sorted(args.input_root.glob("**/seed_*")):
        root = condition.parent
        if str(root) in reports:
            continue
        try:
            reports[str(root.relative_to(args.input_root))] = aggregate_condition(root, args.required_seeds)
        except ValueError:
            continue
    destination = args.input_root / "aggregate_summary.json"
    destination.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
