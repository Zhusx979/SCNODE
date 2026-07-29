from pathlib import Path

import numpy as np

from blood_experiment.visualization import (
    save_confusion_matrix_plot,
    save_precision_recall_plot,
    save_roc_curve_plot,
)
from blood_experiment.cam import save_overlay_preview


def test_metric_plots_are_written(tmp_path: Path) -> None:
    confusion_path = save_confusion_matrix_plot(
        output_path=tmp_path / "cm.png",
        confusion=np.array([[3, 1], [0, 4]]),
        class_names=["ABE", "BLA"],
        normalize=True,
        title="Demo Confusion Matrix",
    )
    roc_path = save_roc_curve_plot(
        output_path=tmp_path / "roc.png",
        roc_payload={
            "micro_auc": 0.90,
            "macro_auc": 0.92,
            "curves": [
                {"class_name": "ABE", "fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0], "auc": 0.9},
                {"class_name": "BLA", "fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.9, 1.0], "auc": 0.95},
            ],
            "micro_curve": {"fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.9, 1.0]},
            "macro_curve": {"fpr": [0.0, 0.15, 1.0], "tpr": [0.0, 0.85, 1.0]},
        },
        class_names=["ABE", "BLA"],
        title="Demo ROC",
    )
    pr_path = save_precision_recall_plot(
        output_path=tmp_path / "pr.png",
        pr_payload={
            "micro_ap": 0.88,
            "curves": [
                {
                    "class_name": "ABE",
                    "precision": [1.0, 0.7, 0.5],
                    "recall": [0.0, 0.6, 1.0],
                    "ap": 0.75,
                },
                {
                    "class_name": "BLA",
                    "precision": [1.0, 0.8, 0.6],
                    "recall": [0.0, 0.7, 1.0],
                    "ap": 0.82,
                },
            ],
            "micro_curve": {"precision": [1.0, 0.85, 0.6], "recall": [0.0, 0.7, 1.0]},
        },
        class_names=["ABE", "BLA"],
        title="Demo PR",
    )

    assert confusion_path.exists()
    assert roc_path.exists()
    assert pr_path.exists()


def test_saliency_overlay_preview_is_written(tmp_path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 120
    heatmap = np.linspace(0, 1, 32 * 32, dtype=float).reshape(32, 32)

    output_path = save_overlay_preview(
        output_path=tmp_path / "overlay.png",
        image=image,
        heatmap=heatmap,
        title="GradCAM Preview",
    )

    assert output_path.exists()
