from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


SERIF_FONTS = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
SANS_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]
NEUTRAL_TEXT = "#243447"
NEUTRAL_GRID = "#D7DEE7"
NEUTRAL_SPINE = "#C7D1DD"
FIGURE_FACE = "#FBFCFE"
AXES_FACE = "#F8FAFC"
OCEAN_DUSK = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#5E81AC",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#CC79A7",
]
MICRO_CURVE_COLOR = "#0F4C5C"
MACRO_CURVE_COLOR = "#C8553D"
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "bm_confusion",
    ["#F8FBFF", "#D6E9F8", "#8FBFE0", "#2D6A8A", "#153B50"],
)
CAM_CMAP = LinearSegmentedColormap.from_list(
    "bm_cam",
    ["#120D31", "#432371", "#FAAE7B", "#FDE74C"],
)


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_FONTS,
            "font.sans-serif": SANS_FONTS,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": NEUTRAL_TEXT,
            "axes.edgecolor": NEUTRAL_SPINE,
            "axes.linewidth": 0.9,
            "axes.facecolor": AXES_FACE,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": NEUTRAL_TEXT,
            "ytick.color": NEUTRAL_TEXT,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#D9E1EA",
            "legend.framealpha": 0.96,
            "legend.fancybox": True,
            "legend.borderpad": 0.5,
            "figure.facecolor": FIGURE_FACE,
            "savefig.facecolor": FIGURE_FACE,
            "savefig.edgecolor": FIGURE_FACE,
            "figure.dpi": 220,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "lines.linewidth": 2.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(
    ax,
    *,
    grid_axis: str = "both",
    grid_alpha: float = 0.28,
    spine_left: bool = True,
    spine_bottom: bool = True,
) -> None:
    ax.set_facecolor(AXES_FACE)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.8, alpha=grid_alpha, color=NEUTRAL_GRID)
    ax.spines["left"].set_visible(spine_left)
    ax.spines["bottom"].set_visible(spine_bottom)
    if spine_left:
        ax.spines["left"].set_color(NEUTRAL_SPINE)
    if spine_bottom:
        ax.spines["bottom"].set_color(NEUTRAL_SPINE)
    ax.tick_params(axis="both", which="major", labelsize=9.5)


def get_line_colors(count: int) -> list[str]:
    if count <= len(OCEAN_DUSK):
        return OCEAN_DUSK[:count]
    cmap = plt.get_cmap("tab20")
    return [mpl.colors.to_hex(cmap(index / max(count - 1, 1))) for index in range(count)]


def finalize_figure(fig, output_path: Path | str, *, transparent: bool = False) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, transparent=transparent)
    if destination.suffix.lower() != ".pdf":
        fig.savefig(destination.with_suffix(".pdf"), transparent=transparent)
    return destination


def compute_training_smoother(values: list[float], alpha: float = 0.3) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    smoothed = np.empty_like(array)
    smoothed[0] = array[0]
    for index in range(1, len(array)):
        smoothed[index] = alpha * array[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


apply_publication_style()
