import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from blood_experiment.data import (
    ManifestImageDataset,
    build_default_transforms,
    prepare_experiment_splits,
    save_dataset_summary,
)
from SCNODE.training.experiment_config import args, models
from SCNODE.training.classification_trainer import conv_init, train_val_test_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders() -> tuple[dict[str, DataLoader], list[str], Path]:
    manifest_path = prepare_experiment_splits(
        raw_root=args.raw_data_root,
        output_dir=args.prepared_data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_dataset_summary(
        manifest_path=manifest_path,
        output_path=Path(args.prepared_data_root) / "dataset_summary.csv",
    )

    train_transform, eval_transform = build_default_transforms(args.image_size)
    train_dataset = ManifestImageDataset(manifest_path, split="train", transform=train_transform)
    val_dataset = ManifestImageDataset(manifest_path, split="val", transform=eval_transform)
    test_dataset = ManifestImageDataset(manifest_path, split="test", transform=eval_transform)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }
    return dataloaders, train_dataset.classes, manifest_path


def build_model(model_spec, num_classes: int, device: torch.device) -> nn.Module:
    model_func = model_spec.load_factory()
    model = model_func(num_classes=num_classes)
    model.apply(conv_init)

    use_cuda = args.gpus and torch.cuda.is_available()
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    return model.to(device)


def main() -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if args.gpus and torch.cuda.is_available() else "cpu")

    dataloaders, class_names, manifest_path = build_dataloaders()
    print(f"Manifest prepared at: {manifest_path}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(
        f"Dataset sizes -> train: {len(dataloaders['train'].dataset)}, "
        f"val: {len(dataloaders['val'].dataset)}, "
        f"test: {len(dataloaders['test'].dataset)}"
    )

    for model_spec, name in models:
        print(f"Training model: {name}")
        model = build_model(model_spec, num_classes=len(class_names), device=device)
        criterion = torch.nn.CrossEntropyLoss()
        train_val_test_model(
            model=model,
            trainloader=dataloaders["train"],
            valloader=dataloaders["val"],
            testloader=dataloaders["test"],
            criterion=criterion,
            device=device,
            name=name,
            class_names=class_names,
            num_epochs=args.num_epochs,
        )
        print(f"Completed training for model: {name}")


if __name__ == "__main__":
    main()
