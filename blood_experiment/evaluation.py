from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=float)[labels]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _compute_summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    matrix: np.ndarray,
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    y_true_one_hot = _one_hot(y_true, len(class_names))
    total = matrix.sum()

    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        tp = float(matrix[class_index, class_index])
        fn = float(matrix[class_index, :].sum() - tp)
        fp = float(matrix[:, class_index].sum() - tp)
        tn = float(total - tp - fn - fp)
        auc = roc_auc_score(y_true_one_hot[:, class_index], y_prob[:, class_index])
        ap = average_precision_score(y_true_one_hot[:, class_index], y_prob[:, class_index])
        rows.append(
            {
                "class_name": class_name,
                "class_index": class_index,
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
                "specificity": _safe_ratio(tn, tn + fp),
                "one_vs_rest_accuracy": _safe_ratio(tp + tn, total),
                "auc": float(auc),
                "average_precision": float(ap),
            }
        )
    return rows


def _compute_roc_payload(
    y_true_one_hot: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    curves = []
    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, class_index], y_prob[:, class_index])
        auc_value = roc_auc_score(y_true_one_hot[:, class_index], y_prob[:, class_index])
        curves.append(
            {
                "class_name": class_name,
                "class_index": class_index,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc_value),
            }
        )

    micro_fpr, micro_tpr, _ = roc_curve(y_true_one_hot.ravel(), y_prob.ravel())
    micro_auc = roc_auc_score(y_true_one_hot, y_prob, average="micro", multi_class="ovr")
    macro_auc = roc_auc_score(y_true_one_hot, y_prob, average="macro", multi_class="ovr")

    all_fpr = np.unique(np.concatenate([np.asarray(curve["fpr"]) for curve in curves]))
    mean_tpr = np.zeros_like(all_fpr)
    for curve in curves:
        mean_tpr += np.interp(all_fpr, np.asarray(curve["fpr"]), np.asarray(curve["tpr"]))
    mean_tpr /= len(curves)

    return {
        "curves": curves,
        "micro_curve": {"fpr": micro_fpr.tolist(), "tpr": micro_tpr.tolist()},
        "macro_curve": {"fpr": all_fpr.tolist(), "tpr": mean_tpr.tolist()},
        "micro_auc": float(micro_auc),
        "macro_auc": float(macro_auc),
    }


def _compute_pr_payload(
    y_true_one_hot: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    curves = []
    for class_index, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_true_one_hot[:, class_index],
            y_prob[:, class_index],
        )
        ap = average_precision_score(y_true_one_hot[:, class_index], y_prob[:, class_index])
        curves.append(
            {
                "class_name": class_name,
                "class_index": class_index,
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "ap": float(ap),
            }
        )

    micro_precision, micro_recall, _ = precision_recall_curve(
        y_true_one_hot.ravel(),
        y_prob.ravel(),
    )
    micro_ap = average_precision_score(y_true_one_hot, y_prob, average="micro")
    return {
        "curves": curves,
        "micro_curve": {
            "precision": micro_precision.tolist(),
            "recall": micro_recall.tolist(),
        },
        "micro_ap": float(micro_ap),
    }


def build_evaluation_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)
    class_count = len(class_names)

    if y_prob_array.ndim != 2 or y_prob_array.shape[1] != class_count:
        raise ValueError("y_prob must have shape [n_samples, n_classes].")

    matrix = confusion_matrix(y_true_array, y_pred_array, labels=np.arange(class_count))
    y_true_one_hot = _one_hot(y_true_array, class_count)

    return {
        "summary": _compute_summary_metrics(y_true_array, y_pred_array),
        "per_class": _compute_per_class_metrics(
            y_true_array,
            y_pred_array,
            y_prob_array,
            class_names,
            matrix,
        ),
        "roc": _compute_roc_payload(y_true_one_hot, y_prob_array, class_names),
        "precision_recall": _compute_pr_payload(y_true_one_hot, y_prob_array, class_names),
        "confusion_matrix": matrix,
    }


def _format_report_text(bundle: dict[str, Any], class_names: list[str]) -> str:
    lines = ["Summary Metrics"]
    for metric_name, metric_value in bundle["summary"].items():
        lines.append(f"- {metric_name}: {metric_value:.6f}")

    lines.append("")
    lines.append("Per-Class Metrics")
    header = "class_name,precision,recall,f1,specificity,support,auc,average_precision"
    lines.append(header)
    for row in bundle["per_class"]:
        lines.append(
            ",".join(
                [
                    row["class_name"],
                    f'{row["precision"]:.6f}',
                    f'{row["recall"]:.6f}',
                    f'{row["f1"]:.6f}',
                    f'{row["specificity"]:.6f}',
                    str(row["support"]),
                    f'{row["auc"]:.6f}',
                    f'{row["average_precision"]:.6f}',
                ]
            )
        )

    lines.append("")
    lines.append("Classes")
    lines.append(", ".join(class_names))
    return "\n".join(lines)


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    return value


def save_evaluation_artifacts(
    output_dir: Path | str,
    bundle: dict[str, Any],
    class_names: list[str],
    prefix: str,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_json = destination / f"{prefix}_summary.json"
    per_class_csv = destination / f"{prefix}_per_class_metrics.csv"
    report_txt = destination / f"{prefix}_report.txt"

    summary_payload = {
        "summary": bundle["summary"],
        "roc": bundle["roc"],
        "precision_recall": bundle["precision_recall"],
        "confusion_matrix": bundle["confusion_matrix"],
        "class_names": class_names,
    }
    summary_json.write_text(
        json.dumps(_to_json_ready(summary_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with per_class_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_name",
                "class_index",
                "precision",
                "recall",
                "f1",
                "support",
                "specificity",
                "one_vs_rest_accuracy",
                "auc",
                "average_precision",
            ],
        )
        writer.writeheader()
        writer.writerows(bundle["per_class"])

    report_txt.write_text(_format_report_text(bundle, class_names), encoding="utf-8")
    return {
        "summary_json": summary_json,
        "per_class_csv": per_class_csv,
        "report_txt": report_txt,
    }
