"""Execution lifecycle for declarative reviewer-response conditions."""

from __future__ import annotations

import hashlib
import json
import random
import csv
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from SCNODE.blood_experiment.data import ManifestImageDataset, build_default_transforms, prepare_experiment_splits
from SCNODE.blood_experiment.robustness import build_corruption
from SCNODE.experiments.configs import ReviewCondition
from SCNODE.diagnostics.trajectory import build_time_grid, pool_trajectory, trajectory_geometry
from SCNODE.diagnostics.ode_metrics import mean_endpoint_relative_error
from SCNODE.training.classification_trainer import conv_init, train_val_test_model
from SCNODE.training.experiment_config import AVAILABLE_MODELS, ExperimentRuntimeConfig


def _condition_key(condition: ReviewCondition) -> str:
    payload = json.dumps(condition.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def artifact_dir(output_root: Path | str, condition: ReviewCondition) -> Path:
    return Path(output_root) / condition.experiment / condition.model_name / _condition_key(condition) / f"seed_{condition.seed}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_eval_transform(image_size: int, condition: ReviewCondition):
    if condition.corruption is None:
        return build_default_transforms(image_size)[1]
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("torchvision is required for review experiments") from exc
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        build_corruption(condition.corruption, condition.corruption_severity, condition.seed),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _build_model(condition: ReviewCondition, num_classes: int, device: torch.device) -> nn.Module:
    spec = AVAILABLE_MODELS[condition.model_name]
    factory = spec.load_factory()
    kwargs = {"num_classes": num_classes}
    if condition.model_name.startswith("SCNODE_"):
        kwargs["config"] = condition.scnode_config
    model = factory(**kwargs)
    model.apply(conv_init)
    return model.to(device)


def _collect_ode_endpoints(model: nn.Module, inputs: torch.Tensor) -> list[torch.Tensor]:
    endpoints: list[torch.Tensor] = []
    handles = []
    for module in model.modules():
        if hasattr(module, "odefunc"):
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, output: endpoints.append(output.detach())
            ))
    try:
        model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return endpoints


def _endpoint_error_against_dopri5(model: nn.Module, condition: ReviewCondition, inputs: torch.Tensor, num_classes: int, device: torch.device) -> float:
    """Compare this integration to a tight-tolerance reference with equal weights."""
    if not condition.model_name.startswith("SCNODE_"):
        return float("nan")
    reference_config = replace(condition.scnode_config, solver="dopri5", rtol=1e-5, atol=1e-5)
    reference_condition = replace(condition, scnode_config=reference_config)
    reference_model = _build_model(reference_condition, num_classes, device)
    reference_model.load_state_dict(model.state_dict())
    model.eval()
    reference_model.eval()
    with torch.no_grad():
        candidate_endpoints = _collect_ode_endpoints(model, inputs)
        reference_endpoints = _collect_ode_endpoints(reference_model, inputs)
    return mean_endpoint_relative_error(candidate_endpoints, reference_endpoints)


def _latest_epoch_diagnostics(model_dir: Path, model_name: str) -> dict[str, float | int]:
    path = model_dir / f"{model_name}_epoch_metrics.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[-1]
    keys = ("mean_nfe_forward", "mean_nfe_backward", "peak_cuda_memory_bytes",
            "max_state_l2", "max_gradient_l2", "nonfinite_batch_count",
            "epoch_duration_seconds")
    return {key: (int(float(row[key])) if key in {"peak_cuda_memory_bytes", "nonfinite_batch_count"}
                  else float(row[key])) for key in keys if row.get(key) not in {None, ""}}


def _export_block_trajectories(model, dataset, destination: Path, points: int, samples_per_class: int, device: torch.device) -> None:
    """Export a fixed stratified subset of computational paths for comments 4/10."""
    if not hasattr(model, "forward_with_trajectory"):
        return
    selected, seen = [], set()
    for index, record in enumerate(dataset.records):
        if record.class_index not in seen:
            selected.append(index)
            seen.add(record.class_index)
        if len(selected) >= samples_per_class:
            break
    time_points = build_time_grid(points).to(device)
    collected: dict[str, dict[str, list]] = {}
    labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for index in selected:
            image, label, _ = dataset[index]
            logits, trajectories = model.forward_with_trajectory(image.unsqueeze(0).to(device), time_points)
            labels.append(label)
            predictions.append(int(logits.argmax(dim=1).item()))
            for block_name, states in trajectories.items():
                pooled = pool_trajectory(states, model.config.augment_dim)
                entry = collected.setdefault(block_name, {"main": [], "auxiliary": [], "geometry": []})
                entry["main"].append(pooled["main"].squeeze(1).cpu().numpy())
                entry["auxiliary"].append(pooled["auxiliary"].squeeze(1).cpu().numpy())
                entry["geometry"].append(trajectory_geometry(states))
    trajectory_dir = destination / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    for block_name, values in collected.items():
        geometry_columns = ("mean_path_length", "mean_adjacent_displacement", "mean_curvature")
        geometry = np.asarray(
            [[item[column] for column in geometry_columns] for item in values["geometry"]], dtype=float
        )
        np.savez_compressed(
            trajectory_dir / f"{block_name}.npz", time=np.linspace(0.0, 1.0, points),
            main=np.asarray(values["main"]), auxiliary=np.asarray(values["auxiliary"]),
            labels=np.asarray(labels), predictions=np.asarray(predictions),
            geometry=geometry, geometry_columns=np.asarray(geometry_columns),
        )


def run_condition(
    condition: ReviewCondition,
    *,
    output_root: Path | str,
    raw_data_root: Path | str,
    prepared_data_root: Path | str,
    image_size: int = 224,
    batch_size: int = 64,
    test_batch_size: int = 64,
    num_workers: int = 0,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    use_gpu: bool = True,
    trajectory_steps: int = 11,
    trajectory_samples: int = 21,
    split_seed: int = 42,
) -> Path:
    """Train one condition and persist a seed-level, self-describing artifact."""
    destination = artifact_dir(output_root, condition)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "resolved_config.json").write_text(
        json.dumps({"condition": condition.to_dict(), "num_epochs": num_epochs}, indent=2), encoding="utf-8"
    )
    set_seed(condition.seed)
    manifest = prepare_experiment_splits(
        raw_root=raw_data_root, output_dir=prepared_data_root, seed=split_seed
    )
    train_transform, _ = build_default_transforms(image_size)
    eval_transform = _build_eval_transform(image_size, condition)
    datasets = {
        "train": ManifestImageDataset(manifest, "train", train_transform),
        "val": ManifestImageDataset(manifest, "val", build_default_transforms(image_size)[1]),
        "test": ManifestImageDataset(manifest, "test", eval_transform),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(datasets["val"], batch_size=test_batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=test_batch_size, shuffle=False, num_workers=num_workers),
    }
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = _build_model(condition, len(datasets["train"].classes), device)
    runtime = ExperimentRuntimeConfig(
        output_root=destination,
        learning_rate=learning_rate,
        use_tqdm=True,
        full_report=False,
        generate_visualizations=False,
        generate_cam=False,
        collect_ode_diagnostics=condition.model_name.startswith("SCNODE_") or "ODE" in condition.model_name,
        evaluate_test_each_epoch=False,
    )
    try:
        bundle = train_val_test_model(
            model=model, trainloader=loaders["train"], valloader=loaders["val"], testloader=loaders["test"],
            criterion=nn.CrossEntropyLoss(), device=device, name=condition.model_name,
            class_names=datasets["train"].classes, num_epochs=num_epochs, runtime_config=runtime,
        )
        summary = dict(bundle["summary"])
        summary["parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
        summary.update(_latest_epoch_diagnostics(destination / condition.model_name, condition.model_name))
        if condition.experiment == "resolution":
            diagnostic_inputs, _, _ = next(iter(loaders["test"]))
            summary["endpoint_relative_error"] = _endpoint_error_against_dopri5(
                model, condition, diagnostic_inputs.to(device), len(datasets["train"].classes), device
            )
        if condition.experiment == "augmentation" and condition.model_name.startswith("SCNODE_"):
            _export_block_trajectories(model, datasets["test"], destination, trajectory_steps, trajectory_samples, device)
        metrics_dir = destination / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        (metrics_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (destination / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    except Exception as exc:
        (destination / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}), encoding="utf-8")
        raise
    return destination


def run_conditions(conditions: Iterable[ReviewCondition], *, output_root: Path | str, dry_run: bool = False, **kwargs) -> list[ReviewCondition]:
    resolved = list(conditions)
    if dry_run:
        for condition in resolved:
            print(json.dumps(condition.to_dict(), sort_keys=True))
        return resolved
    for condition in resolved:
        run_condition(condition, output_root=output_root, **kwargs)
    return resolved
