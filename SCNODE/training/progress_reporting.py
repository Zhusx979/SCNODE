from __future__ import annotations

import csv
from pathlib import Path


def format_seconds(total_seconds: float) -> str:
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def build_epoch_summary_lines(
    *,
    epoch: int,
    num_epochs: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float | None,
    val_accuracy: float | None,
    test_accuracy: float,
    macro_f1: float,
    balanced_accuracy: float,
    mcc: float,
    learning_rate: float,
    epoch_duration: float,
    eta_seconds: float,
    best_val_accuracy: float | None,
    best_test_accuracy: float,
    skipped_images: int,
) -> list[str]:
    lines = [
        f"Epoch {epoch}/{num_epochs} | lr={learning_rate:.6g} | time={format_seconds(epoch_duration)} | ETA={format_seconds(eta_seconds)}",
        f"Train  loss={train_loss:.4f}  acc={train_accuracy:.2f}%",
    ]
    if val_loss is not None and val_accuracy is not None:
        lines.append(f"Val    loss={val_loss:.4f}  acc={val_accuracy:.2f}%")
    else:
        lines.append("Val    skipped")
    lines.extend(
        [
            f"Test   acc={test_accuracy:.2f}%  macro_f1={macro_f1:.4f}  bacc={balanced_accuracy:.4f}  mcc={mcc:.4f}",
            f"Best   val={0.0 if best_val_accuracy is None else best_val_accuracy:.2f}%  test={best_test_accuracy:.2f}%",
            f"Data   skipped_images={skipped_images}",
        ]
    )
    return lines


def append_epoch_metrics_row(csv_path: Path | str, row: dict) -> Path:
    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not destination.exists()

    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return destination
