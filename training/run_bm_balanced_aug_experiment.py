from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SCNODE.blood_experiment.bm_balanced import BalancedTrainDataset, EvaluationDataset
from SCNODE.blood_experiment.data import load_manifest_records, prepare_experiment_splits
from SCNODE.models.ode.scnode.scnode_resnet import Get_time_AnodeV2_ResNet18
from SCNODE.training.classification_trainer import conv_init
from SCNODE.training.ode_runtime import get_and_reset_ode_nfe


CSV_FIELDS = [
    "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "test_accuracy",
    "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall",
    "weighted_f1", "balanced_accuracy", "learning_rate", "epoch_duration_seconds",
    "best_val_accuracy", "skipped_images", "mean_nfe_forward", "mean_nfe_backward",
    "nonfinite_batch_count",
]


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu().clone())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([value.detach().cpu().clone() for value in state["torch_cuda"]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SCNODE BM 6000-per-class training")
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--prepared-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-per-class", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--translate-fraction", type=float, default=0.10)
    parser.add_argument("--shear-degrees", type=float, default=5.0)
    parser.add_argument("--stain-probability", type=float, default=0.25)
    parser.add_argument("--stain-strength", type=float, default=0.5)
    parser.add_argument("--background-fill", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser


def _metric_row(labels, predictions, class_count: int) -> dict[str, float]:
    if not labels:
        return {
            "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
            "weighted_precision": 0.0, "weighted_recall": 0.0, "weighted_f1": 0.0,
            "balanced_accuracy": 0.0, "per_class": [
                {"precision": 0.0, "recall": 0.0, "f1": 0.0} for _ in range(class_count)
            ],
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(class_count)), average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "macro_precision": float(macro_precision), "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1), "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall), "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "per_class": [
            {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i in range(class_count)
        ],
    }


def _write_epoch_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.resume_checkpoint is not None:
        args.resume = True
    if args.target_per_class <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("target-per-class, epochs, and batch-size must be positive")
    if args.resume and args.resume_checkpoint is None:
        args.resume_checkpoint = args.output_root / "SCNODE_ResNet18" / "last_checkpoint.pt"
    if args.resume and not args.resume_checkpoint.is_file():
        raise FileNotFoundError("Resume checkpoint does not exist: {}".format(args.resume_checkpoint))
    set_seed(args.seed)
    manifest = prepare_experiment_splits(
        args.raw_data_root, args.prepared_data_root, 0.8, 0.1, 0.1,
        seed=args.seed, max_samples_per_class=None,
    )
    train_records = load_manifest_records(manifest, "train")
    val_records = load_manifest_records(manifest, "val")
    test_records = load_manifest_records(manifest, "test")
    train_dataset = BalancedTrainDataset(
        train_records, args.target_per_class, 224, args.seed, args.translate_fraction,
        args.shear_degrees, args.stain_probability, args.stain_strength, args.background_fill,
    )
    val_dataset = EvaluationDataset(val_records, 224)
    test_dataset = EvaluationDataset(test_records, 224)
    classes = train_dataset.classes
    loaders = {
        "train": DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True),
        "val": DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Get_time_AnodeV2_ResNet18(num_classes=len(classes))
    model.apply(conv_init)
    gpu_count = torch.cuda.device_count() if use_cuda else 0
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model_dir = args.output_root / "SCNODE_ResNet18"
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "target_per_class": args.target_per_class, "epochs": args.epochs, "batch_size": args.batch_size,
        "seed": args.seed, "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
        "translate_fraction": args.translate_fraction, "shear_degrees": args.shear_degrees,
        "stain_probability": args.stain_probability, "stain_strength": args.stain_strength,
        "background_fill": "four_corner_median" if args.background_fill else "black",
        "manifest": str(manifest), "train_counts": {name: sum(r.class_name == name for r in train_dataset.records) for name in classes},
        "val_counts": {name: sum(r.class_name == name for r in val_records) for name in classes},
        "test_counts": {name: sum(r.class_name == name for r in test_records) for name in classes},
        "model": "SCNODE_ResNet18", "optimizer": "Adam", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "device": str(device), "gpu_count": gpu_count,
        "data_parallel": gpu_count > 1, "twbn_grids": 11, "twbn_window": 5,
        "twbn_statistics_risk": "DataParallel replicas do not synchronize custom TWBN buffers; checkpoint stores the primary replica.",
    }
    (args.output_root / "augmentation_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_root / "augmentation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    csv_path = model_dir / "SCNODE_ResNet18_epoch_metrics.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
    start_epoch = 0
    best_accuracy = float("-inf")
    best_epoch = 0
    best_model_state = None
    if args.resume:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_accuracy = float(checkpoint["best_validation_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        best_model_state = checkpoint["best_model_state_dict"]
        restore_rng_state(checkpoint["rng_state"])
    print("Data: train={} val={} test={} | classes={} | DataParallel={} GPUs={}".format(
        len(train_dataset), len(val_dataset), len(test_dataset), len(classes), gpu_count > 1, gpu_count
    ))
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        train_result = _run_epoch_with_optimizer(model, loaders["train"], criterion, optimizer, device, True, len(classes))
        val_result = _run_epoch_with_optimizer(model, loaders["val"], criterion, device, False, len(classes))
        val_accuracy = val_result["accuracy"]
        improved = val_accuracy > best_accuracy
        if improved:
            best_accuracy, best_epoch = val_accuracy, epoch + 1
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        payload = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1, "validation_accuracy": val_accuracy,
            "best_validation_accuracy": best_accuracy, "best_epoch": best_epoch,
            "best_model_state_dict": best_model_state, "rng_state": capture_rng_state(),
        }
        atomic_torch_save(payload, model_dir / "last_checkpoint.pt")
        if improved:
            atomic_torch_save(payload, model_dir / "best_checkpoint.pt")
        _write_epoch_json(args.output_root / "SCNODE_ResNet18" / "epoch_logs" / "epoch_{:03d}.json".format(epoch + 1), {
            "epoch": epoch + 1, "train": {k: v for k, v in train_result.items() if k not in {"labels", "predictions"}},
            "val": {k: v for k, v in val_result.items() if k not in {"labels", "predictions"}}, "best_validation_accuracy": best_accuracy,
        })
        row = {field: "" for field in CSV_FIELDS}
        row.update({"epoch": epoch + 1, "train_loss": train_result["loss"], "train_accuracy": train_result["accuracy"],
                    "val_loss": val_result["loss"], "val_accuracy": val_accuracy,
                    **{key: val_result["metrics"][key] for key in ("macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "balanced_accuracy")},
                    "learning_rate": optimizer.param_groups[0]["lr"], "epoch_duration_seconds": train_result["duration"],
                    "best_val_accuracy": best_accuracy, "skipped_images": train_dataset.skipped_image_count + val_dataset.skipped_image_count,
                    "mean_nfe_forward": train_result["nfe_forward"] / max(len(loaders["train"]), 1),
                    "mean_nfe_backward": train_result["nfe_backward"] / max(len(loaders["train"]), 1), "nonfinite_batch_count": train_result["nonfinite"]})
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_FIELDS).writerow(row)
        with (model_dir / "training_history.txt").open("a", encoding="utf-8") as handle:
            handle.write(
                "Epoch {}/{} | train_loss={:.6f} train_acc={:.4f} val_loss={:.6f} val_acc={:.4f} best_val={:.4f}\n".format(
                    epoch + 1, args.epochs, train_result["loss"], train_result["accuracy"],
                    val_result["loss"], val_accuracy, best_accuracy
                )
            )
        print("\nEpoch {}/{} [train]\nTrain loss={:.4f} acc={:.2f}%\n\nEpoch {}/{} [val]\nVal   loss={:.4f} acc={:.2f}%\nBest  val={:.2f}%\nData  skipped_images={}\nODE   nfe_f={:.2f} nfe_b={:.2f}".format(
            epoch + 1, args.epochs, train_result["loss"], train_result["accuracy"], epoch + 1, args.epochs,
            val_result["loss"], val_accuracy, best_accuracy, train_dataset.skipped_image_count + val_dataset.skipped_image_count,
            train_result["nfe_forward"] / max(len(loaders["train"]), 1), train_result["nfe_backward"] / max(len(loaders["train"]), 1)))
    if not (model_dir / "best_checkpoint.pt").is_file():
        best_model_state = best_model_state or model.state_dict()
        fallback = {
            "model_state_dict": best_model_state, "optimizer_state_dict": optimizer.state_dict(),
            "epoch": start_epoch, "validation_accuracy": best_accuracy,
            "best_validation_accuracy": best_accuracy, "best_epoch": best_epoch,
            "best_model_state_dict": best_model_state, "rng_state": capture_rng_state(),
        }
        atomic_torch_save(fallback, model_dir / "best_checkpoint.pt")
    best_checkpoint = torch.load(model_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final = _run_epoch_with_optimizer(model, loaders["test"], criterion, device, False, len(classes))
    metrics = _metric_row(final["labels"], final["predictions"], len(classes))
    report = classification_report(final["labels"], final["predictions"], labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    summary = {"accuracy": final["accuracy"] / 100.0, **{key: metrics[key] for key in metrics if key != "per_class"}, "per_class": report, "confusion_matrix": confusion_matrix(final["labels"], final["predictions"], labels=list(range(len(classes)))).tolist(), "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy, "mcc": float(matthews_corrcoef(final["labels"], final["predictions"]))}
    (model_dir / "final_test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "balanced_accuracy", "mcc")}, indent=2))


def _run_epoch_with_optimizer(model, loader, criterion, optimizer_or_device, device_or_train, train_or_classes, classes=None):
    if isinstance(optimizer_or_device, torch.optim.Optimizer):
        optimizer, device, train, class_count = optimizer_or_device, device_or_train, train_or_classes, classes
    else:
        optimizer, device, train, class_count = None, optimizer_or_device, device_or_train, train_or_classes
    model.train(train)
    total_loss, total, correct, nonfinite = 0.0, 0, 0, 0
    labels_all, predictions_all = [], []
    nfe_forward, nfe_backward = 0.0, 0.0
    start = time.time()
    for inputs, labels, _ in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            if train:
                nfe_forward += get_and_reset_ode_nfe(model)
            loss = criterion(outputs, labels)
            if not torch.isfinite(outputs).all() or not torch.isfinite(loss):
                nonfinite += 1
                continue
            if train:
                loss.backward()
                nfe_backward += get_and_reset_ode_nfe(model)
                optimizer.step()
        total_loss += float(loss.detach())
        predicted = outputs.detach().argmax(dim=1)
        total += labels.numel(); correct += int((predicted == labels).sum())
        labels_all.extend(labels.detach().cpu().tolist()); predictions_all.extend(predicted.cpu().tolist())
    return {"loss": total_loss / max(len(loader), 1), "accuracy": 100.0 * correct / max(total, 1),
            "labels": labels_all, "predictions": predictions_all, "metrics": _metric_row(labels_all, predictions_all, class_count),
            "nfe_forward": nfe_forward, "nfe_backward": nfe_backward, "nonfinite": nonfinite, "duration": time.time() - start}


if __name__ == "__main__":
    main()
