import json

import pytest

from SCNODE.experiments.summarize import aggregate_condition


def test_aggregate_rejects_missing_seed(tmp_path) -> None:
    metrics_dir = tmp_path / "condition_a" / "seed_42" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "summary.json").write_text(
        json.dumps({"macro_f1": 0.8}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing seeds"):
        aggregate_condition(tmp_path / "condition_a", required_seeds=[42, 123])


def test_aggregate_reports_mean_for_complete_seed_group(tmp_path) -> None:
    condition_dir = tmp_path / "condition_a"
    for seed, value in [(42, 0.8), (123, 0.9)]:
        metrics_dir = condition_dir / f"seed_{seed}" / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "summary.json").write_text(
            json.dumps({"macro_f1": value}), encoding="utf-8"
        )

    result = aggregate_condition(condition_dir, required_seeds=[42, 123])

    assert result["macro_f1_mean"] == pytest.approx(0.85)
