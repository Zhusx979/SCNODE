from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from blood_experiment.cam import generate_cam_overlays, save_overlay_preview
from blood_experiment.visualization import (
    prepare_confusion_matrix_for_display,
    save_confusion_matrix_plot,
    save_precision_recall_plot,
    save_roc_curve_plot,
)


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


def test_confusion_matrix_plot_has_no_numeric_annotations(tmp_path: Path, monkeypatch) -> None:
    text_calls = []

    def record_text(self, *args, **kwargs):
        text_calls.append((args, kwargs))
        return None

    monkeypatch.setattr("matplotlib.axes._axes.Axes.text", record_text)

    save_confusion_matrix_plot(
        output_path=tmp_path / "cm.png",
        confusion=np.array([[3, 1], [0, 4]]),
        class_names=["ABE", "BLA"],
        normalize=True,
        title="Demo Confusion Matrix",
    )

    assert text_calls == []


def test_confusion_matrix_is_row_normalized() -> None:
    normalized = prepare_confusion_matrix_for_display(
        np.array([[3, 1], [0, 4]], dtype=float),
        normalize=True,
    )
    np.testing.assert_allclose(normalized, np.array([[0.75, 0.25], [0.0, 1.0]]))


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


def test_generate_cam_overlays_moves_tensor_to_model_device(tmp_path: Path, monkeypatch) -> None:
    class TinyConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

        def forward(self, inputs):
            return self.conv(inputs).mean(dim=(2, 3))

    model = TinyConv()
    seen_devices = []

    def fake_gradcam(model_arg, input_tensor, class_index, target_layer_name=None):
        seen_devices.append(input_tensor.device.type)
        return np.ones((32, 32), dtype=float)

    def fake_smoothgrad(model_arg, input_tensor, class_index, target_layer_name=None, noise_samples=8, noise_sigma=0.1):
        seen_devices.append(input_tensor.device.type)
        return np.ones((32, 32), dtype=float)

    monkeypatch.setattr("blood_experiment.cam.generate_gradcam_heatmap", fake_gradcam)
    monkeypatch.setattr("blood_experiment.cam.generate_smoothgradcam_heatmap", fake_smoothgrad)

    artifacts = generate_cam_overlays(
        model=model,
        samples=[
            {
                "input_tensor": torch.rand(3, 32, 32),
                "image_path": "sample.jpg",
                "true_index": 0,
                "predicted_index": 0,
            }
        ],
        class_names=["ABE"],
        output_dir=tmp_path / "cam",
    )

    assert seen_devices == [next(model.parameters()).device.type, next(model.parameters()).device.type]
    assert artifacts[0].gradcam_path.exists()
