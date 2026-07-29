from pathlib import Path

from SCNODE.training.progress_reporting import (
    append_epoch_metrics_row,
    build_epoch_summary_lines,
    format_seconds,
)


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
