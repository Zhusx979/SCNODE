from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SCNODE.blood_experiment.data import (
    ManifestImageDataset,
    build_default_transforms,
    get_class_names_from_manifest,
    load_manifest_records,
    prepare_experiment_splits,
    save_dataset_summary,
)
from SCNODE.blood_experiment.selectable_data import EvalDataset, NoAugmentationDataset
from SCNODE.training.classification_trainer import conv_init, train_val_test_model
from SCNODE.training.experiment_config import AVAILABLE_MODELS, ExperimentRuntimeConfig
from SCNODE.training.run_bm_experiment import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selectable SCNODE BM training protocol")
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--target-per-class", type=int, default=6000)
    parser.add_argument("--augmentation-mode", choices=("none", "stain_geometry"), default="stain_geometry")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--gpus", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _copy_subset_records(data_root: Path, prepared_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "raw_subset").mkdir(exist_ok=True)
    for filename in ("subset_selection.csv", "subset_metadata.json"):
        source = prepared_root / filename
        if source.exists():
            shutil.copy2(source, data_root / filename)


def build_selectable_dataloaders(args):
    data_root = args.experiment_root / "data"
    prepared_root = data_root / "prepared_split"
    manifest_path = prepare_experiment_splits(
        raw_root=args.raw_data_root,
        output_dir=prepared_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_samples_per_class=args.target_per_class,
    )
    _copy_subset_records(data_root, prepared_root)
    summary_path = data_root / "dataset_summary.csv"
    save_dataset_summary(manifest_path, summary_path)
    records = load_manifest_records(manifest_path)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    test_records = [record for record in records if record.split == "test"]

    if args.augmentation_mode == "none":
        train_dataset = NoAugmentationDataset(train_records, args.image_size)
        val_dataset = EvalDataset(val_records, args.image_size)
        test_dataset = EvalDataset(test_records, args.image_size)
        upsampling = False
    else:
        train_transform, eval_transform = build_default_transforms(args.image_size, paper_augmentation=True)
        train_dataset = ManifestImageDataset(
            manifest_path, "train", transform=train_transform,
            target_samples_per_class=args.target_per_class, seed=args.seed,
        )
        val_dataset = ManifestImageDataset(manifest_path, "val", transform=eval_transform)
        test_dataset = ManifestImageDataset(manifest_path, "test", transform=eval_transform)
        upsampling = True
    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }
    return loaders, get_class_names_from_manifest(manifest_path), manifest_path, upsampling


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.target_per_class <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("target-per-class, epochs, and batch-size must be positive")
    set_seed(args.seed)
    use_cuda = args.gpus and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_count = torch.cuda.device_count() if use_cuda else 0
    loaders, class_names, manifest_path, upsampling = build_selectable_dataloaders(args)
    results_root = args.experiment_root / "results"
    runtime_config = ExperimentRuntimeConfig(
        output_root=results_root,
        learning_rate=1e-3,
        optimizer="adam",
        scheduler="none",
        use_tqdm=True,
        generate_visualizations=False,
        generate_cam=False,
        full_report=True,
    )
    run_config = {
        "target_per_class": args.target_per_class,
        "augmentation_mode": args.augmentation_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "upsampling": upsampling,
        "geometry_augmentation": args.augmentation_mode == "stain_geometry",
        "stain_augmentation": args.augmentation_mode == "stain_geometry",
        "gpu_count": gpu_count,
        "model_name": "SCNODE_ResNet18",
        "train_samples": len(loaders["train"].dataset),
        "val_samples": len(loaders["val"].dataset),
        "test_samples": len(loaders["test"].dataset),
        "manifest_path": str(manifest_path.relative_to(args.experiment_root)),
    }
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    (args.experiment_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print("Training mode: {}".format(args.augmentation_mode))
    print("Target per class: {}".format(args.target_per_class))
    print("Upsampling: {}".format("enabled" if upsampling else "disabled"))
    print("Geometry augmentation: {}".format("enabled" if args.augmentation_mode == "stain_geometry" else "disabled"))
    print("Stain augmentation: {}".format("enabled" if args.augmentation_mode == "stain_geometry" else "disabled"))
    print("Batch size: {}".format(args.batch_size))
    print("Epochs: {}".format(args.epochs))
    print("GPUs: {}".format(gpu_count))
    spec = AVAILABLE_MODELS["SCNODE_ResNet18"]
    model = build_model(spec, len(class_names), device, argparse.Namespace(gpus=args.gpus))
    model.apply(conv_init)
    train_val_test_model(
        model=model, trainloader=loaders["train"], valloader=loaders["val"], testloader=loaders["test"],
        criterion=torch.nn.CrossEntropyLoss(), device=device, name="SCNODE_ResNet18",
        class_names=class_names, runtime_config=runtime_config, num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
