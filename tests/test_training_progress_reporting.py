import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.argv = [sys.argv[0]]

from SCNODE.training.progress_reporting import (
    append_epoch_metrics_row,
    build_epoch_summary_lines,
    format_seconds,
)
from SCNODE.training.experiment_config import ExperimentRuntimeConfig
from SCNODE.training.classification_trainer import train_val_test_model


class _AlternatingTrainLoader:
    """Yields labels that make the final epoch worse than the first one."""

    class _Dataset:
        skipped_image_count = 0

    dataset = _Dataset()

    def __init__(self) -> None:
        self.epoch = 0

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        label = 0 if self.epoch == 0 else 1
        self.epoch += 1
        yield torch.tensor([[1.0]]), torch.tensor([label])


class _StaticLoader:
    class _Dataset:
        skipped_image_count = 0

    dataset = _Dataset()

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield torch.tensor([[1.0], [1.0]]), torch.tensor([0, 1])


class _CountingStaticLoader(_StaticLoader):
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield from super().__iter__()


class _SignClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("train_calls", torch.tensor(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.train_calls.add_(1)
        sign = 1.0 if int(self.train_calls.item()) == 1 else -1.0
        logits = torch.tensor([sign, -sign], device=inputs.device).expand(inputs.size(0), -1)
        return logits + self.weight * 0.0


def test_format_seconds_handles_minutes_and_seconds() -> None:
    assert format_seconds(12.4) == "12.4s"
    assert format_seconds(75.2) == "1m 15.2s"
    assert format_seconds(3675.0) == "1h 1m 15.0s"


def test_build_epoch_summary_lines_includes_best_and_eta() -> None:
    lines = build_epoch_summary_lines(
        epoch=2,
        num_epochs=20,
        train_loss=0.4567,
        train_accuracy=81.2,
        val_loss=0.5123,
        val_accuracy=79.4,
        test_accuracy=78.8,
        macro_f1=0.7765,
        balanced_accuracy=0.7812,
        mcc=0.7421,
        learning_rate=1e-3,
        epoch_duration=74.6,
        eta_seconds=18 * 74.6,
        best_val_accuracy=79.4,
        best_test_accuracy=78.8,
        skipped_images=3,
    )

    rendered = "\n".join(lines)
    assert "Epoch 2/20" in rendered
    assert "Train  loss=0.4567  acc=81.20%" in rendered
    assert "Val    loss=0.5123  acc=79.40%" in rendered
    assert "Best   val=79.40%  test=78.80%" in rendered
    assert "Data   skipped_images=3" in rendered
    assert "ETA" in rendered


def test_append_epoch_metrics_row_creates_csv_with_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "epoch_metrics.csv"
    append_epoch_metrics_row(
        csv_path=csv_path,
        row={
            "epoch": 1,
            "train_loss": 0.5,
            "train_accuracy": 80.0,
            "val_loss": 0.4,
            "val_accuracy": 82.0,
            "test_accuracy": 81.0,
            "macro_f1": 0.8,
            "balanced_accuracy": 0.79,
            "mcc": 0.75,
            "learning_rate": 0.001,
            "epoch_duration_seconds": 73.1,
            "eta_seconds": 730.0,
            "best_val_accuracy": 82.0,
            "best_test_accuracy": 81.0,
            "skipped_images": 2,
        },
    )

    contents = csv_path.read_text(encoding="utf-8")
    assert "epoch,train_loss,train_accuracy" in contents
    assert "1,0.5,80.0,0.4,82.0,81.0" in contents


def test_epoch_metrics_include_ode_diagnostics(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    append_epoch_metrics_row(
        csv_path,
        {
            "epoch": 1,
            "mean_nfe_forward": 3.0,
            "mean_nfe_backward": 4.0,
            "peak_cuda_memory_bytes": 0,
            "max_state_l2": 0.0,
            "max_gradient_l2": 0.0,
            "nonfinite_batch_count": 0,
        },
    )

    contents = csv_path.read_text(encoding="utf-8")
    assert "mean_nfe_forward" in contents
    assert "nonfinite_batch_count" in contents


def test_training_writes_runtime_artifacts_and_uses_validation_best_checkpoint(tmp_path: Path) -> None:
    runtime_config = ExperimentRuntimeConfig(
        output_root=tmp_path,
        learning_rate=1.0,
        use_tqdm=False,
        train_log_interval=1,
        full_report=False,
        generate_visualizations=False,
        generate_cam=False,
    )
    model = _SignClassifier()

    train_val_test_model(
        model=model,
        trainloader=_AlternatingTrainLoader(),
        valloader=_StaticLoader(),
        testloader=_StaticLoader(),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        name="tiny",
        class_names=["negative", "positive"],
        num_epochs=2,
        runtime_config=runtime_config,
    )

    model_dir = tmp_path / "tiny"
    assert (model_dir / "resolved_config.json").is_file()
    assert (model_dir / "best_checkpoint.pt").is_file()
    assert (model_dir / "test_predictions.npz").is_file()
    assert json.loads((model_dir / "resolved_config.json").read_text(encoding="utf-8"))["learning_rate"] == 1.0
    assert np.load(model_dir / "test_predictions.npz")["predictions"].tolist() == [0, 0]


def test_training_evaluates_test_set_once_after_validation_selection(tmp_path: Path) -> None:
    runtime_config = ExperimentRuntimeConfig(
        output_root=tmp_path, learning_rate=1.0, use_tqdm=False, full_report=False,
        generate_visualizations=False, generate_cam=False,
    )
    testloader = _CountingStaticLoader()

    train_val_test_model(
        model=_SignClassifier(), trainloader=_AlternatingTrainLoader(),
        valloader=_StaticLoader(), testloader=testloader, criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"), name="no_test_leakage",
        class_names=["negative", "positive"], num_epochs=2, runtime_config=runtime_config,
    )

    assert testloader.iterations == 1
