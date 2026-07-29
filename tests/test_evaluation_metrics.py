import numpy as np
from pathlib import Path

from blood_experiment.evaluation import build_evaluation_bundle, save_evaluation_artifacts


def test_build_evaluation_bundle_contains_per_class_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.10, 0.90],
        ]
    )

    bundle = build_evaluation_bundle(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=["ABE", "BLA"],
    )

    assert abs(bundle["summary"]["accuracy"] - 0.75) < 1e-9
    assert "balanced_accuracy" in bundle["summary"]
    assert len(bundle["per_class"]) == 2
    assert bundle["per_class"][0]["class_name"] == "ABE"
    assert "specificity" in bundle["per_class"][0]
    assert bundle["roc"]["micro_auc"] >= 0.0
    assert bundle["roc"]["macro_auc"] >= 0.0
    assert bundle["confusion_matrix"].tolist() == [[1, 1], [0, 2]]


def test_save_evaluation_artifacts_writes_json_and_csv(tmp_path: Path) -> None:
    bundle = build_evaluation_bundle(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        y_prob=np.array(
            [
                [0.90, 0.10],
                [0.20, 0.80],
                [0.30, 0.70],
                [0.10, 0.90],
            ]
        ),
        class_names=["ABE", "BLA"],
    )

    artifact_paths = save_evaluation_artifacts(
        output_dir=tmp_path,
        bundle=bundle,
        class_names=["ABE", "BLA"],
        prefix="demo",
    )

    assert artifact_paths["summary_json"].exists()
    assert artifact_paths["per_class_csv"].exists()
    assert artifact_paths["report_txt"].exists()
