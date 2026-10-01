from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SCNODE.training.classification_trainer import conv_init, train_val_test_model
from SCNODE.training.experiment_config import (
    ExperimentRuntimeConfig,
    get_selected_models,
)


CIFAR10_MODELS = (
    "ResNet18",
    "FCANet18",
    "SwinT",
    "ConvNeXtT",
    "DeiTTiny",
    "AnodeV2_ResNet18",
    "SCNODE_ResNet18",
    "NODE",
    "ANODE",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train selected models on CIFAR-10")
    parser.add_argument("--model_names", nargs="+", choices=CIFAR10_MODELS, default=list(CIFAR10_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data_root", type=Path, default=Path("artifacts/datasets/cifar10"))
    parser.add_argument("--output_root", type=Path, default=Path("artifacts/cifar10_experiments"))
    parser.add_argument("--no_download", action="store_false", dest="download", default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser


def _import_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "CIFAR-10 execution requires a compatible torchvision installation."
        ) from exc
    return datasets, transforms


def build_cifar10_dataloaders(args: argparse.Namespace):
    datasets, transforms = _import_torchvision()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=str(args.data_root), train=True, download=args.download, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(args.data_root), train=False, download=args.download, transform=eval_transform
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_dataset, test_dataset, trainloader, testloader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    args = build_parser().parse_args()
    selected_models = get_selected_models(args.model_names)
    train_dataset, test_dataset, trainloader, testloader = build_cifar10_dataloaders(args)
    class_names = list(train_dataset.classes)
    device = torch.device("cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
    runtime_config = ExperimentRuntimeConfig(
        output_root=args.output_root,
        learning_rate=args.learning_rate,
        collect_ode_diagnostics=True,
        evaluate_test_each_epoch=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")

    for seed in args.seeds:
        for model_spec, model_name in selected_models:
            set_seed(seed)
            model = model_spec.load_factory()(num_classes=len(class_names)).to(device)
            model.apply(conv_init)
            train_val_test_model(
                model=model,
                trainloader=trainloader,
                valloader=None,
                testloader=testloader,
                criterion=torch.nn.CrossEntropyLoss(),
                device=device,
                name=f"{model_name}_seed{seed}",
                class_names=class_names,
                runtime_config=runtime_config,
                num_epochs=args.num_epochs,
            )


if __name__ == "__main__":
    main()
