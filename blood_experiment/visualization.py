from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from blood_experiment.plot_style import (
    CONFUSION_CMAP,
    MACRO_CURVE_COLOR,
    MICRO_CURVE_COLOR,
    compute_training_smoother,
    finalize_figure,
    get_line_colors,
    style_axes,
)


LINE_STYLES = ["-", "--", "-.", (0, (5, 1.2)), (0, (3, 1.1, 1.1, 1.1))]


def _ensure_parent(output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _dynamic_square_size(class_count: int) -> tuple[float, float]:
    edge = min(max(6.8, class_count * 0.52), 12.8)
    return edge, edge


def _curve_figure_size(curve_count: int) -> tuple[float, float]:
    if curve_count <= 6:
        return 7.0, 5.6
    if curve_count <= 12:
        return 8.4, 6.2
    return 11.5, 7.4


def _legend_columns(item_count: int) -> int:
    if item_count <= 8:
        return 1
    if item_count <= 18:
        return 2
    return 3


def prepare_confusion_matrix_for_display(
    confusion: np.ndarray,
    normalize: bool,
) -> np.ndarray:
    matrix = np.asarray(confusion, dtype=float)
    if not normalize:
        return matrix
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def save_confusion_matrix_plot(
    output_path: Path | str,
    confusion: np.ndarray,
    class_names: list[str],
    normalize: bool,
    title: str,
) -> Path:
    destination = _ensure_parent(output_path)
    plot_matrix = prepare_confusion_matrix_for_display(confusion, normalize=normalize)
    figure_size = _dynamic_square_size(len(class_names))

    fig, ax = plt.subplots(figsize=figure_size, dpi=220)
    heatmap = ax.imshow(
        plot_matrix,
        cmap=CONFUSION_CMAP,
        aspect="equal",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0 if normalize else None,
    )
    ax.set_title(title, fontsize=14, pad=8, loc="left")
    ax.set_xlabel("Predicted Class", labelpad=12)
    ax.set_ylabel("True Class", labelpad=12)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.tick_params(axis="x", top=False, bottom=True, labeltop=False, labelbottom=True, length=4, width=1.1, pad=6)
    ax.tick_params(axis="y", left=True, right=False, length=4, width=1.1, pad=6)
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="#160B39", linewidth=0.35, alpha=0.28)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine_name in ("left", "bottom", "top", "right"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#111111")
        ax.spines[spine_name].set_linewidth(1.5)

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        orientation="horizontal",
        location="top",
        fraction=0.055,
        pad=0.025,
        shrink=1.0,
        aspect=22,
    )
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(labelsize=9, length=4, width=1.0, pad=2, colors="#111111")
    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor("#111111")
    colorbar.outline.set_linewidth(1.5)
    if normalize:
        colorbar.set_ticks(np.linspace(0.0, 1.0, 6))

    fig.tight_layout(pad=0.55)
    return finalize_figure(fig, destination)


def _plot_iso_f1_guides(ax) -> None:
    for f1_score in (0.2, 0.4, 0.6, 0.8):
        recall = np.linspace(max(f1_score / 2.0 + 1e-4, 1e-4), 1.0, 250)
        precision = (f1_score * recall) / np.maximum(2 * recall - f1_score, 1e-6)
        valid = (precision >= 0.0) & (precision <= 1.0)
        ax.plot(
            recall[valid],
            precision[valid],
            color="#CBD5E1",
            linewidth=0.8,
            linestyle=":",
            alpha=0.75,
            zorder=0,
        )


def save_roc_curve_plot(
    output_path: Path | str,
    roc_payload: dict[str, Any],
    class_names: list[str],
    title: str,
) -> Path:
    destination = _ensure_parent(output_path)
    curves = list(roc_payload.get("curves", []))
    figure_size = _curve_figure_size(len(curves))
    fig, ax = plt.subplots(figsize=figure_size, dpi=220)

    colors = get_line_colors(max(len(curves), 1))
    for index, curve in enumerate(sorted(curves, key=lambda item: item.get("auc", 0.0), reverse=True)):
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=1.6,
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            color=colors[index % len(colors)],
            alpha=0.82,
            label=f'{curve["class_name"]} (AUC={curve["auc"]:.3f})',
            zorder=2,
        )

    micro_curve = roc_payload.get("micro_curve", {})
    macro_curve = roc_payload.get("macro_curve", {})
    if micro_curve:
        ax.plot(
            micro_curve["fpr"],
            micro_curve["tpr"],
            color=MICRO_CURVE_COLOR,
            linewidth=3.1,
            linestyle="-",
            label=f'Micro Avg (AUC={roc_payload.get("micro_auc", 0.0):.3f})',
            zorder=4,
        )
    if macro_curve:
        ax.plot(
            macro_curve["fpr"],
            macro_curve["tpr"],
            color=MACRO_CURVE_COLOR,
            linewidth=3.1,
            linestyle="--",
            label=f'Macro Avg (AUC={roc_payload.get("macro_auc", 0.0):.3f})',
            zorder=4,
        )

    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", linewidth=1.4, alpha=0.9, zorder=1)
    style_axes(ax, grid_axis="both", grid_alpha=0.22)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, pad=16, loc="left")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=_legend_columns(len(curves) + 2),
        fontsize=8.4,
        handlelength=2.8,
        columnspacing=1.4,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    return finalize_figure(fig, destination)


def save_precision_recall_plot(
    output_path: Path | str,
    pr_payload: dict[str, Any],
    class_names: list[str],
    title: str,
) -> Path:
    destination = _ensure_parent(output_path)
    curves = list(pr_payload.get("curves", []))
    figure_size = _curve_figure_size(len(curves))
    fig, ax = plt.subplots(figsize=figure_size, dpi=220)

    _plot_iso_f1_guides(ax)
    colors = get_line_colors(max(len(curves), 1))
    for index, curve in enumerate(sorted(curves, key=lambda item: item.get("ap", 0.0), reverse=True)):
        ax.plot(
            curve["recall"],
            curve["precision"],
            linewidth=1.6,
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            color=colors[index % len(colors)],
            alpha=0.84,
            label=f'{curve["class_name"]} (AP={curve["ap"]:.3f})',
            zorder=2,
        )

    micro_curve = pr_payload.get("micro_curve", {})
    if micro_curve:
        ax.plot(
            micro_curve["recall"],
            micro_curve["precision"],
            color=MICRO_CURVE_COLOR,
            linewidth=3.1,
            label=f'Micro Avg (AP={pr_payload.get("micro_ap", 0.0):.3f})',
            zorder=4,
        )

    style_axes(ax, grid_axis="both", grid_alpha=0.22)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=14, pad=16, loc="left")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=_legend_columns(len(curves) + 1),
        fontsize=8.4,
        handlelength=2.8,
        columnspacing=1.4,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    return finalize_figure(fig, destination)


def _plot_metric_series(
    ax,
    epochs: np.ndarray,
    values: list[float],
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str = "o",
) -> None:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return
    ax.plot(
        epochs,
        array,
        color=color,
        linewidth=1.25,
        alpha=0.28,
        linestyle=linestyle,
        zorder=1,
    )
    ax.plot(
        epochs,
        compute_training_smoother(values).astype(float),
        color=color,
        linewidth=2.5,
        linestyle=linestyle,
        marker=marker,
        markevery=max(len(values) // 8, 1),
        markersize=4.2,
        label=label,
        zorder=3,
    )


def save_training_history_plots(
    output_dir: Path | str,
    name: str,
    train_accuracies: list[float],
    val_accuracies: list[float],
    test_accuracies: list[float],
    train_losses: list[float],
    val_losses: list[float],
) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_accuracies) + 1)
    saved_paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220)
    accuracy_ax, loss_ax = axes

    _plot_metric_series(
        accuracy_ax,
        epochs,
        train_accuracies,
        label="Train Accuracy",
        color="#2A6F97",
        marker="o",
    )
    if val_accuracies:
        _plot_metric_series(
            accuracy_ax,
            epochs[: len(val_accuracies)],
            val_accuracies,
            label="Validation Accuracy",
            color="#2A9D8F",
            linestyle="--",
            marker="s",
        )
    _plot_metric_series(
        accuracy_ax,
        epochs[: len(test_accuracies)],
        test_accuracies,
        label="Test Accuracy",
        color="#E76F51",
        linestyle="-.",
        marker="D",
    )
    style_axes(accuracy_ax, grid_axis="both", grid_alpha=0.24)
    accuracy_ax.set_title("Accuracy Trajectory", loc="left", pad=14)
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Accuracy (%)")
    accuracy_ax.legend(loc="lower right", fontsize=9)

    if test_accuracies:
        best_epoch = int(np.argmax(test_accuracies))
        best_value = float(test_accuracies[best_epoch])
        accuracy_ax.scatter(best_epoch + 1, best_value, color="#C8553D", s=42, zorder=5)
        accuracy_ax.annotate(
            f"Best test: {best_value:.2f}%",
            xy=(best_epoch + 1, best_value),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=9,
            color="#7C2D12",
            bbox={"boxstyle": "round,pad=0.22", "fc": "#FFF7ED", "ec": "#FDBA74", "lw": 0.8},
        )

    _plot_metric_series(
        loss_ax,
        epochs,
        train_losses,
        label="Train Loss",
        color="#355070",
        marker="o",
    )
    finite_val_losses = [value for value in val_losses if np.isfinite(value)]
    if finite_val_losses:
        _plot_metric_series(
            loss_ax,
            epochs[: len(finite_val_losses)],
            finite_val_losses,
            label="Validation Loss",
            color="#F4A261",
            linestyle="--",
            marker="^",
        )
    style_axes(loss_ax, grid_axis="both", grid_alpha=0.24)
    loss_ax.set_title("Optimization Dynamics", loc="left", pad=14)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross-Entropy Loss")
    loss_ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"{name} Training Overview", x=0.07, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    saved_paths.append(finalize_figure(fig, destination / f"{name}_training_overview.png"))
    plt.close(fig)

    for metric_name, values, ylabel, color, linestyle in (
        ("accuracy_curve", train_accuracies, "Accuracy (%)", "#2A6F97", "-"),
        ("loss_curve", train_losses, "Cross-Entropy Loss", "#355070", "-"),
    ):
        fig, ax = plt.subplots(figsize=(7.1, 4.8), dpi=220)
        _plot_metric_series(
            ax,
            epochs,
            values,
            label=f"Train {ylabel}",
            color=color,
            linestyle=linestyle,
            marker="o",
        )
        if metric_name == "accuracy_curve":
            if val_accuracies:
                _plot_metric_series(
                    ax,
                    epochs[: len(val_accuracies)],
                    val_accuracies,
                    label="Validation Accuracy",
                    color="#2A9D8F",
                    linestyle="--",
                    marker="s",
                )
            if test_accuracies:
                _plot_metric_series(
                    ax,
                    epochs[: len(test_accuracies)],
                    test_accuracies,
                    label="Test Accuracy",
                    color="#E76F51",
                    linestyle="-.",
                    marker="D",
                )
            ax.set_title(f"{name} Accuracy Curves", loc="left", pad=14)
        else:
            if finite_val_losses:
                _plot_metric_series(
                    ax,
                    epochs[: len(finite_val_losses)],
                    finite_val_losses,
                    label="Validation Loss",
                    color="#F4A261",
                    linestyle="--",
                    marker="^",
                )
            ax.set_title(f"{name} Loss Curves", loc="left", pad=14)
        style_axes(ax, grid_axis="both", grid_alpha=0.24)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        saved_paths.append(finalize_figure(fig, destination / f"{name}_{metric_name}.png"))
        plt.close(fig)

    return saved_paths
