"""Create static-image computational-trajectory figures from saved BM checkpoints.

The figures describe continuous-depth states of fixed classifiers.  The input
dataset contains independent static images, so numerical integration time is
never interpreted as elapsed biological time or as a longitudinal cell trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import make_interp_spline
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

try:
    import umap
except ImportError:  # pragma: no cover - environment dependency
    umap = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SCNODE.blood_experiment.data import build_default_transforms
from SCNODE.diagnostics.trajectory_experiment import forward_with_solver_trajectories
from SCNODE.training.experiment_config import AVAILABLE_MODELS


MODEL_SPECS = (
    ("SCNODE_ResNet18", "SCNODE"),
    ("AnodeV2_ResNet18", "ANODEV2"),
    ("ANODE", "ANODE"),
    ("NODE", "NODE"),
)
STAGES = ("PMO", "MYB", "MMZ", "NGB", "NGS")
STAGE_COLORS = {
    "PMO": "#1B9E77",
    "MYB": "#4AA8D8",
    "MMZ": "#7666B5",
    "NGB": "#D35F8D",
    "NGS": "#D55E00",
}


@dataclass(frozen=True)
class Sample:
    sample_index: int
    class_name: str
    class_index: int
    image_path: Path


@dataclass(frozen=True)
class CheckpointChoice:
    seed: int
    checkpoint: Path
    validation_macro_f1: float
    all_scores: tuple[tuple[int, float], ...]


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _load_stage_samples(records_path: Path) -> list[Sample]:
    by_stage: dict[str, list[Sample]] = {stage: [] for stage in STAGES}
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stage = row["class_name"]
            if stage in by_stage:
                by_stage[stage].append(
                    Sample(
                        sample_index=int(row["sample_index"]),
                        class_name=stage,
                        class_index=int(row["class_index"]),
                        image_path=Path(row["image_path"]),
                    )
                )
    return [sample for stage in STAGES for sample in by_stage[stage]]


def _select_trajectory_samples(samples: list[Sample], samples_per_stage: int, seed: int) -> list[Sample]:
    generator = np.random.default_rng(seed)
    selected: list[Sample] = []
    for stage in STAGES:
        candidates = [sample for sample in samples if sample.class_name == stage]
        if len(candidates) < samples_per_stage:
            raise ValueError(f"{stage} has only {len(candidates)} test records; need {samples_per_stage}")
        selected.extend(candidates[index] for index in generator.choice(len(candidates), samples_per_stage, replace=False))
    return selected


def _select_trajectory_samples_global(samples: list[Sample], samples_per_stage: int, seed: int) -> list[Sample]:
    """Select a fresh deterministic pair per stage using one global RNG."""
    generator = np.random.default_rng(seed)
    selected: list[Sample] = []
    for stage in STAGES:
        candidates = [sample for sample in samples if sample.class_name == stage]
        if len(candidates) < samples_per_stage:
            raise ValueError(f"{stage} has only {len(candidates)} test records; need {samples_per_stage}")
        selected.extend(candidates[index] for index in generator.choice(len(candidates), samples_per_stage, replace=False))
    return selected


def _list_checkpoints(runs_root: Path, model_name: str) -> list[tuple[int, float, Path]]:
    scores: list[tuple[int, float, Path]] = []
    for checkpoint in sorted((runs_root / model_name).glob(f"seed_*/{model_name}/best_checkpoint.pt")):
        seed = int(checkpoint.parents[1].name.removeprefix("seed_"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        score = float(payload["validation_macro_f1"])
        scores.append((seed, score, checkpoint))
    return scores


def _load_model(model_name: str, checkpoint: Path, device: torch.device):
    model = AVAILABLE_MODELS[model_name].load_factory()(num_classes=21).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def _load_batch(samples: list[Sample], transform, device: torch.device) -> torch.Tensor:
    tensors = []
    for sample in samples:
        with Image.open(sample.image_path) as image:
            tensors.append(transform(image.convert("RGB")))
    return torch.stack(tensors).to(device)


def _extract_final_block_states(
    model,
    model_name: str,
    samples: list[Sample],
    *,
    batch_size: int,
    time_points: torch.Tensor,
    solver_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return pooled states [sample, time, feature] from the final ODE block."""
    _, transform = build_default_transforms(224)
    states_by_batch: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    final_block_name: str | None = None
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            inputs = _load_batch(batch_samples, transform, device)
            logits, trajectories = forward_with_solver_trajectories(
                model, model_name, inputs, time_points, solver="rk4", rtol=1e-3,
                atol=1e-3, solver_steps=solver_steps,
            )
            block_name = next(reversed(trajectories))
            if final_block_name is None:
                final_block_name = block_name
            elif final_block_name != block_name:
                raise RuntimeError("Final ODE block changed across batches")
            states = trajectories[block_name].mean(dim=(-1, -2)).permute(1, 0, 2)
            states_by_batch.append(states.detach().cpu().numpy())
            probabilities.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(states_by_batch), np.concatenate(probabilities), str(final_block_name)


def _metrics(states: np.ndarray, stage_names: np.ndarray) -> dict[str, object]:
    eps = 1e-12
    deltas = np.diff(states, axis=1)
    baseline_norm = np.linalg.norm(states[:, :-1], axis=-1).clip(min=eps)
    relative_steps = np.linalg.norm(deltas, axis=-1) / baseline_norm
    second_difference = states[:, 2:] - 2 * states[:, 1:-1] + states[:, :-2]
    relative_curvature = np.linalg.norm(second_difference, axis=-1) / relative_steps[:, 1:].clip(min=eps)
    endpoints = StandardScaler().fit_transform(states[:, -1])
    centroids = np.stack([endpoints[stage_names == stage].mean(axis=0) for stage in STAGES])
    adjacent_distances = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    pc1 = PCA(n_components=1, random_state=42).fit_transform(endpoints).ravel()
    stage_positions = np.asarray([pc1[stage_names == stage].mean() for stage in STAGES])
    if stage_positions[-1] < stage_positions[0]:
        stage_positions *= -1
    rho, rho_p = spearmanr(np.arange(len(STAGES)), stage_positions)
    return {
        "relative_step_variance_mean": float(np.var(relative_steps, axis=1, ddof=1).mean()),
        "relative_step_variance_median": float(np.median(np.var(relative_steps, axis=1, ddof=1))),
        "relative_curvature_mean": float(relative_curvature.mean()),
        "adjacent_centroid_distance_mean": float(adjacent_distances.mean()),
        "adjacent_centroid_distance_cv": float(adjacent_distances.std(ddof=1) / adjacent_distances.mean()),
        "endpoint_pc1_stage_order_spearman_rho": float(rho),
        "endpoint_pc1_stage_order_p_value": float(rho_p),
        "endpoint_pc1_stage_positions": stage_positions.tolist(),
        "adjacent_centroid_distances": adjacent_distances.tolist(),
        "relative_steps": relative_steps,
    }


def _fit_endpoint_projection(
    endpoint_states: np.ndarray, trajectory_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Fit on all t=1 endpoints, then map only selected full trajectories."""
    scaler = StandardScaler().fit(endpoint_states)
    standardized_endpoints = scaler.transform(endpoint_states)
    standardized_trajectories = scaler.transform(trajectory_states.reshape(-1, trajectory_states.shape[-1]))
    if umap is not None:
        reducer = umap.UMAP(
            n_neighbors=25, min_dist=0.24, metric="cosine", random_state=42, n_jobs=1,
        )
        endpoint_embedding = reducer.fit_transform(standardized_endpoints)
        trajectory_embedding = reducer.transform(standardized_trajectories).reshape(trajectory_states.shape[0], trajectory_states.shape[1], 2)
        return endpoint_embedding, trajectory_embedding, "UMAP"
    reducer = PCA(n_components=2, random_state=42).fit(standardized_endpoints)
    return (
        reducer.transform(standardized_endpoints),
        reducer.transform(standardized_trajectories).reshape(trajectory_states.shape[0], trajectory_states.shape[1], 2),
        "PCA (UMAP unavailable)",
    )


@dataclass
class _Projection:
    """Reusable fitted projection: standardise then reduce endpoint states."""
    scaler: StandardScaler
    reducer: object
    projection_name: str

    def transform_points(self, points_2d: np.ndarray) -> np.ndarray:
        return self.reducer.transform(self.scaler.transform(points_2d))

    def map_trajectories(self, trajectory_states: np.ndarray) -> np.ndarray:
        n_samples, n_time, n_feature = trajectory_states.shape
        projected = self.transform_points(trajectory_states.reshape(-1, n_feature))
        return projected.reshape(n_samples, n_time, 2)


def _fit_projection(endpoint_states: np.ndarray) -> _Projection:
    """Fit a single projection on all t=1 endpoints for reuse across selections."""
    scaler = StandardScaler().fit(endpoint_states)
    standardized_endpoints = scaler.transform(endpoint_states)
    if umap is not None:
        reducer = umap.UMAP(
            n_neighbors=25, min_dist=0.24, metric="cosine", random_state=42, n_jobs=1,
        )
        reducer.fit(standardized_endpoints)
        projection_name = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=42).fit(standardized_endpoints)
        projection_name = "PCA (UMAP unavailable)"
    return _Projection(scaler=scaler, reducer=reducer, projection_name=projection_name)


def _trajectory_axis(
    ax,
    endpoint_embedding: np.ndarray,
    endpoint_stage_names: np.ndarray,
    trajectory_embedding: np.ndarray,
    trajectory_stage_names: np.ndarray,
    title: str,
    projection_name: str,
) -> None:
    for stage in STAGES:
        mask = endpoint_stage_names == stage
        ax.scatter(
            endpoint_embedding[mask, 0], endpoint_embedding[mask, 1],
            s=4, color=STAGE_COLORS[stage], alpha=0.15, linewidths=0, rasterized=True,
        )
        for index in np.flatnonzero(trajectory_stage_names == stage):
            time = np.linspace(0.0, 1.0, trajectory_embedding.shape[1])
            smooth_time = np.linspace(0.0, 1.0, 121)
            smooth_curve = make_interp_spline(time, trajectory_embedding[index], k=3)(smooth_time)
            ax.plot(smooth_curve[:, 0], smooth_curve[:, 1], color=STAGE_COLORS[stage],
                    linewidth=1.3, alpha=0.78, zorder=3)
            ax.scatter(trajectory_embedding[index, 0, 0], trajectory_embedding[index, 0, 1], s=11, facecolors="white",
                       edgecolors=STAGE_COLORS[stage], linewidths=0.8, zorder=4)
            ax.scatter(trajectory_embedding[index, -1, 0], trajectory_embedding[index, -1, 1], s=13,
                       color=STAGE_COLORS[stage], linewidths=0, zorder=5)
    ax.set_title(title, loc="center")
    axis_prefix = "UMAP" if projection_name == "UMAP" else "PC"
    ax.set_xlabel(f"{axis_prefix} 1")
    ax.set_ylabel(f"{axis_prefix} 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    arrow_style = {"arrowstyle": "-|>", "color": "black", "linewidth": 0.9, "mutation_scale": 10}
    ax.annotate("", xy=(1.02, 0.0), xytext=(0.0, 0.0), xycoords="axes fraction",
                arrowprops=arrow_style, annotation_clip=False)
    ax.annotate("", xy=(0.0, 1.02), xytext=(0.0, 0.0), xycoords="axes fraction",
                arrowprops=arrow_style, annotation_clip=False)


def _individual_figure(
    output_path: Path,
    display_name: str,
    seed: int,
    validation_macro_f1: float,
    endpoint_embedding: np.ndarray,
    endpoint_stage_names: np.ndarray,
    trajectory_embedding: np.ndarray,
    trajectory_stage_names: np.ndarray,
    projection_name: str,
) -> None:
    figure, trajectory_axis = plt.subplots(figsize=(6.1, 4.8), constrained_layout=True)
    _trajectory_axis(
        trajectory_axis, endpoint_embedding, endpoint_stage_names, trajectory_embedding,
        trajectory_stage_names, f"{display_name} | seed {seed} | validation Macro-F1 {validation_macro_f1:.3f}", projection_name,
    )
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _comparison_figure(output_path: Path, analyses: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), constrained_layout=True)
    for panel_index, (axis, analysis) in enumerate(zip(axes, analyses)):
        _trajectory_axis(
            axis, analysis["endpoint_embedding"], analysis["endpoint_stage_names"], analysis["trajectory_embedding"],
            analysis["trajectory_stage_names"],
            analysis["display_name"], analysis["projection_name"],
        )
        if panel_index:
            axis.set_ylabel("")
    handles = [mpl.lines.Line2D([0], [0], color=STAGE_COLORS[stage], linewidth=1.6, label=stage) for stage in STAGES]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _checkpoint_for_seed(runs_root: Path, model_name: str, seed: int) -> CheckpointChoice:
    for candidate_seed, score, checkpoint in _list_checkpoints(runs_root, model_name):
        if candidate_seed == seed:
            return CheckpointChoice(
                seed=candidate_seed,
                checkpoint=checkpoint,
                validation_macro_f1=score,
                all_scores=(),
            )
    raise FileNotFoundError(f"No {model_name} checkpoint found for seed {seed}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_scalars(metrics: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key not in {"relative_steps", "endpoint_pc1_stage_positions", "adjacent_centroid_distances"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate computational trajectory figures from saved five-seed checkpoints.")
    parser.add_argument("--experiment-root", type=Path, default=ROOT_DIR / "artifacts/experiments/bm_imagelevel_5seed_e20_b256")
    parser.add_argument("--trajectory-samples-per-stage", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=1)
    parser.add_argument(
        "--figure-seeds", type=int, nargs="+", default=None,
        help="Optional subset of common training seeds to render.",
    )
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--solver-steps", type=int, default=10)
    parser.add_argument(
        "--alternate-samples", action="store_true",
        help="For seed 42, render additional figures with fresh two-image-per-stage selections.",
    )
    parser.add_argument(
        "--alternate-selection-seeds", type=int, nargs="+", default=list(range(1, 11)),
        help="Selection seeds used with --alternate-samples (default: 1..10).",
    )
    parser.add_argument(
        "--all-seeds-quantitative", action="store_true",
        help="Compute the scalar trajectory diagnostics for all saved seeds without creating extra figures.",
    )
    args = parser.parse_args()

    _configure_matplotlib()
    experiment_root = args.experiment_root.resolve()
    output_root = experiment_root / "trajectory_interpretability"
    output_root.mkdir(exist_ok=True)
    background_samples = _load_stage_samples(experiment_root / "data/test_records.csv")
    trajectory_samples = _select_trajectory_samples(
        background_samples, args.trajectory_samples_per_stage, args.selection_seed,
    )
    background_stage_names = np.asarray([sample.class_name for sample in background_samples])
    trajectory_stage_names = np.asarray([sample.class_name for sample in trajectory_samples])
    _write_csv(
        output_root / "selected_trajectory_images.csv",
        [{"sample_index": sample.sample_index, "class_name": sample.class_name, "class_index": sample.class_index, "image_path": str(sample.image_path)} for sample in trajectory_samples],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selection_rows: list[dict[str, object]] = []
    endpoint_time_points = torch.tensor([0.0, 1.0], device=device)
    trajectory_time_points = torch.linspace(0.0, 1.0, 11, device=device)
    seed_sets = {
        model_name: {seed for seed, _score, _checkpoint in _list_checkpoints(experiment_root / "runs", model_name)}
        for model_name, _display_name in MODEL_SPECS
    }
    seeds = sorted(set.intersection(*seed_sets.values()))
    if len(seeds) != 5:
        raise ValueError(f"Expected exactly five common seeds, found {seeds}")
    if args.figure_seeds is not None:
        requested_seeds = sorted(set(args.figure_seeds))
        unknown_seeds = sorted(set(requested_seeds) - set(seeds))
        if unknown_seeds:
            raise ValueError(f"Requested seeds are unavailable for one or more models: {unknown_seeds}")
        seeds = requested_seeds
    elif args.alternate_samples:
        seeds = [42]
    for seed in seeds:
        analyses: list[dict[str, object]] = []
        for model_name, display_name in MODEL_SPECS:
            choice = _checkpoint_for_seed(experiment_root / "runs", model_name, seed)
            print(f"Rendering seed {seed}: {display_name}", flush=True)
            model = _load_model(model_name, choice.checkpoint, device)
            endpoint_states, _endpoint_probabilities, final_block = _extract_final_block_states(
                model, model_name, background_samples, batch_size=args.inference_batch_size,
                time_points=endpoint_time_points, solver_steps=args.solver_steps, device=device,
            )
            trajectory_states, _trajectory_probabilities, trajectory_block = _extract_final_block_states(
                model, model_name, trajectory_samples, batch_size=args.inference_batch_size,
                time_points=trajectory_time_points, solver_steps=args.solver_steps, device=device,
            )
            if trajectory_block != final_block:
                raise RuntimeError("Final ODE block changed between endpoint and trajectory extraction")
            endpoint_embedding, trajectory_embedding, projection_name = _fit_endpoint_projection(
                endpoint_states[:, -1], trajectory_states,
            )
            analyses.append({
                "display_name": display_name, "seed": seed, "endpoint_embedding": endpoint_embedding,
                "endpoint_stage_names": background_stage_names, "trajectory_embedding": trajectory_embedding,
                "trajectory_stage_names": trajectory_stage_names, "projection_name": projection_name,
            })
            selection_rows.append({
                "figure_seed": seed, "selection_seed": args.selection_seed, "model": display_name,
                "checkpoint": str(choice.checkpoint), "validation_macro_f1": choice.validation_macro_f1,
                "final_ode_block": final_block,
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _comparison_figure(output_root / f"granulocyte_endpoint_background_trajectory_seed_{seed}.pdf", analyses)

    if args.alternate_samples:
        if seeds != [42]:
            raise ValueError("--alternate-samples renders only seed 42; pass --figure-seeds 42")
        selections = {
            selection_seed: _select_trajectory_samples_global(
                background_samples, args.trajectory_samples_per_stage, selection_seed,
            )
            for selection_seed in sorted(set(args.alternate_selection_seeds))
        }
        alternate_analyses: dict[int, list[dict[str, object]]] = {
            selection_seed: [] for selection_seed in selections
        }
        for model_name, display_name in MODEL_SPECS:
            choice = _checkpoint_for_seed(experiment_root / "runs", model_name, 42)
            print(f"[seed 42 / alternate] loading {display_name}", flush=True)
            model = _load_model(model_name, choice.checkpoint, device)
            endpoint_states, _endpoint_probabilities, final_block = _extract_final_block_states(
                model, model_name, background_samples, batch_size=args.inference_batch_size,
                time_points=endpoint_time_points, solver_steps=args.solver_steps, device=device,
            )
            projection = _fit_projection(endpoint_states[:, -1])
            endpoint_embedding = projection.transform_points(endpoint_states[:, -1])
            for selection_seed, alternate_samples in selections.items():
                alternate_stage_names = np.asarray([sample.class_name for sample in alternate_samples])
                trajectory_states, _trajectory_probabilities, trajectory_block = _extract_final_block_states(
                    model, model_name, alternate_samples, batch_size=args.inference_batch_size,
                    time_points=trajectory_time_points, solver_steps=args.solver_steps, device=device,
                )
                if trajectory_block != final_block:
                    raise RuntimeError("Final ODE block changed between endpoint and trajectory extraction")
                alternate_analyses[selection_seed].append({
                    "display_name": display_name, "seed": 42,
                    "endpoint_embedding": endpoint_embedding,
                    "endpoint_stage_names": background_stage_names,
                    "trajectory_embedding": projection.map_trajectories(trajectory_states),
                    "trajectory_stage_names": alternate_stage_names,
                    "projection_name": projection.projection_name,
                })
                selection_rows.append({
                    "figure_seed": 42, "selection_seed": selection_seed, "model": display_name,
                    "checkpoint": str(choice.checkpoint), "validation_macro_f1": choice.validation_macro_f1,
                    "final_ode_block": final_block,
                })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for selection_seed, analyses in alternate_analyses.items():
            _write_csv(
                output_root / f"selected_trajectory_images_seed_42_sel{selection_seed:02d}.csv",
                [
                    {
                        "sample_index": sample.sample_index, "class_name": sample.class_name,
                        "class_index": sample.class_index, "image_path": str(sample.image_path),
                    }
                    for sample in selections[selection_seed]
                ],
            )
            _comparison_figure(
                output_root / f"granulocyte_endpoint_background_trajectory_seed_42_sel{selection_seed:02d}.pdf",
                analyses,
            )

    _write_csv(output_root / "checkpoint_selection.csv", selection_rows)
    if args.all_seeds_quantitative:
        all_seed_rows: list[dict[str, object]] = []
        for model_name, display_name in MODEL_SPECS:
            for seed, validation_macro_f1, checkpoint in _list_checkpoints(experiment_root / "runs", model_name):
                print(f"Quantifying {display_name}, seed {seed}", flush=True)
                model = _load_model(model_name, checkpoint, device)
                states, _probabilities, final_block = _extract_final_block_states(
                    model, model_name, trajectory_samples, batch_size=args.inference_batch_size,
                    time_points=trajectory_time_points,
                    solver_steps=args.solver_steps, device=device,
                )
                metrics = _metrics(states, trajectory_stage_names)
                all_seed_rows.append(
                    {
                        "model": display_name,
                        "model_name": model_name,
                        "seed": seed,
                        "validation_macro_f1": validation_macro_f1,
                        "final_ode_block": final_block,
                        **_metric_scalars(metrics),
                    }
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        _write_csv(output_root / "quantitative_trajectory_metrics_by_seed.csv", all_seed_rows)
        scalar_columns = list(_metric_scalars(_metrics(states, trajectory_stage_names)))
        summary_rows: list[dict[str, object]] = []
        for _model_name, display_name in MODEL_SPECS:
            model_rows = [row for row in all_seed_rows if row["model"] == display_name]
            summary: dict[str, object] = {"model": display_name, "seed_count": len(model_rows)}
            for column in scalar_columns:
                values = np.asarray([row[column] for row in model_rows], dtype=float)
                summary[f"{column}_mean"] = float(values.mean())
                summary[f"{column}_std"] = float(values.std(ddof=1))
            summary_rows.append(summary)
        _write_csv(output_root / "quantitative_trajectory_metrics_5seed_summary.csv", summary_rows)
    (output_root / "README.txt").write_text(
        "Each model panel is fitted independently using the t=1 pooled final-ODE-block states of all held-out test images in PMO, MYB, MMZ, NGB, and NGS. "
        "Each PDF compares SCNODE, ANODEv2, ANODE, and NODE checkpoints trained with the same one of five random seeds. "
        "Only the same four fixed images per stage contribute the t=0-to-1 path overlays. "
        "The figures describe model-internal computational trajectories only; they do not establish observed cellular differentiation trajectories.\n",
        encoding="utf-8",
    )
    print(f"Wrote trajectory figures and metrics to {output_root}")


if __name__ == "__main__":
    main()

