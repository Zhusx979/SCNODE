from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _ensure_parent(output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_confusion_matrix_plot(
    output_path: Path | str,
    confusion: np.ndarray,
    class_names: list[str],
    normalize: bool,
    title: str,
) -> Path:
    destination = _ensure_parent(output_path)
    matrix = np.asarray(confusion, dtype=float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        plot_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    else:
        plot_matrix = matrix

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.55), max(6, len(class_names) * 0.55)), dpi=180)
    heatmap = ax.imshow(plot_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = float(plot_matrix.max()) / 2 if plot_matrix.size else 0.0
    for row_index in range(plot_matrix.shape[0]):
        for col_index in range(plot_matrix.shape[1]):
            if normalize:
                label = f"{plot_matrix[row_index, col_index]:.2f}\n({int(matrix[row_index, col_index])})"
            else:
                label = str(int(matrix[row_index, col_index]))
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if plot_matrix[row_index, col_index] > threshold else "#202124",
                fontsize=8,
            )

    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_roc_curve_plot(
    output_path: Path | str,
    roc_payload: dict[str, Any],
    class_names: list[str],
    title: str,
) -> Path:
    destination = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(class_names), 1)))

    for color, curve in zip(colors, roc_payload.get("curves", [])):
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=1.8,
            color=color,
            label=f'{curve["class_name"]} (AUC={curve["auc"]:.3f})',
        )

    micro_curve = roc_payload.get("micro_curve", {})
    macro_curve = roc_payload.get("macro_curve", {})
    if micro_curve:
        ax.plot(
            micro_curve["fpr"],
            micro_curve["tpr"],
            color="#111827",
            linewidth=2.5,
            linestyle="-",
            label=f'Micro Avg (AUC={roc_payload.get("micro_auc", 0.0):.3f})',
        )
    if macro_curve:
        ax.plot(
            macro_curve["fpr"],
            macro_curve["tpr"],
            color="#B91C1C",
            linewidth=2.5,
            linestyle="--",
            label=f'Macro Avg (AUC={roc_payload.get("macro_auc", 0.0):.3f})',
        )

    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=":", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_precision_recall_plot(
    output_path: Path | str,
    pr_payload: dict[str, Any],
    class_names: list[str],
    title: str,
) -> Path:
    destination = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(class_names), 1)))

    for color, curve in zip(colors, pr_payload.get("curves", [])):
        ax.plot(
            curve["recall"],
            curve["precision"],
            linewidth=1.8,
            color=color,
            label=f'{curve["class_name"]} (AP={curve["ap"]:.3f})',
        )

    micro_curve = pr_payload.get("micro_curve", {})
    if micro_curve:
        ax.plot(
            micro_curve["recall"],
            micro_curve["precision"],
            color="#0F766E",
            linewidth=2.6,
            label=f'Micro Avg (AP={pr_payload.get("micro_ap", 0.0):.3f})',
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=14, pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination
