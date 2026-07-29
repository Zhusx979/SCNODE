from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_default_raw_dataset_root() -> Path:
    return Path(r"E:\School Work\Deep Learning\Paper\blood\code\BM_cytomorphology_data")


def discover_class_names(raw_root: Path | str) -> list[str]:
    root = Path(raw_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def discover_image_paths(class_dir: Path | str) -> list[Path]:
    root = Path(class_dir)
    image_paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(image_paths)


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must add up to a positive number.")
    return {
        "train": train_ratio / total,
        "val": val_ratio / total,
        "test": test_ratio / total,
    }


def allocate_split_counts(
    sample_count: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")

    normalized = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    exact = {split: sample_count * ratio for split, ratio in normalized.items()}
    counts = {split: math.floor(value) for split, value in exact.items()}
    remainder = sample_count - sum(counts.values())

    ranked_splits = sorted(
        exact.items(),
        key=lambda item: (item[1] - math.floor(item[1]), item[0] == "train"),
        reverse=True,
    )
    for index in range(remainder):
        split_name = ranked_splits[index % len(ranked_splits)][0]
        counts[split_name] += 1

    positive_splits = [name for name, ratio in normalized.items() if ratio > 0]
    if sample_count >= len(positive_splits):
        for split_name in positive_splits:
            if counts[split_name] == 0:
                donor = max(
                    (name for name in positive_splits if counts[name] > 1),
                    key=lambda name: counts[name],
                )
                counts[donor] -= 1
                counts[split_name] += 1

    return counts


def _rows_for_class(
    class_name: str,
    class_index: int,
    image_paths: Iterable[Path],
    counts: dict[str, int],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    start = 0
    for split_name in ("train", "val", "test"):
        end = start + counts[split_name]
        for image_path in list(image_paths)[start:end]:
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "class_index": class_index,
                    "image_path": str(image_path),
                }
            )
        start = end
    return rows


def create_split_manifest(
    raw_root: Path | str,
    output_dir: Path | str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Path:
    raw_root_path = Path(raw_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir_path / "split_manifest.csv"
    rows: list[dict[str, str | int]] = []
    generator = random.Random(seed)

    for class_index, class_name in enumerate(discover_class_names(raw_root_path)):
        class_paths = discover_image_paths(raw_root_path / class_name)
        if not class_paths:
            continue
        generator.shuffle(class_paths)
        counts = allocate_split_counts(
            len(class_paths),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        rows.extend(_rows_for_class(class_name, class_index, class_paths, counts))

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_name", "class_index", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


@dataclass(frozen=True)
class ManifestRecord:
    split: str
    class_name: str
    class_index: int
    image_path: Path


def load_manifest_records(
    manifest_path: Path | str,
    split: str | None = None,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            records.append(
                ManifestRecord(
                    split=row["split"],
                    class_name=row["class_name"],
                    class_index=int(row["class_index"]),
                    image_path=Path(row["image_path"]),
                )
            )
    return records


def prepare_experiment_splits(
    raw_root: Path | str,
    output_dir: Path | str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    destination = Path(output_dir)
    manifest_path = destination / "split_manifest.csv"
    if manifest_path.exists():
        return manifest_path
    return create_split_manifest(
        raw_root=raw_root,
        output_dir=destination,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def get_class_names_from_manifest(manifest_path: Path | str) -> list[str]:
    records = load_manifest_records(manifest_path)
    unique_pairs = sorted({(record.class_index, record.class_name) for record in records})
    return [class_name for _, class_name in unique_pairs]


def save_dataset_summary(
    manifest_path: Path | str,
    output_path: Path | str,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = load_manifest_records(manifest_path)
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        summary.setdefault(record.class_name, {"train": 0, "val": 0, "test": 0, "total": 0})
        summary[record.class_name][record.split] += 1
        summary[record.class_name]["total"] += 1

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "train", "val", "test", "total"],
        )
        writer.writeheader()
        for class_name in sorted(summary):
            writer.writerow({"class_name": class_name, **summary[class_name]})
    return destination


class ManifestImageDataset:
    def __init__(
        self,
        manifest_path: Path | str,
        split: str,
        transform=None,
    ) -> None:
        self.records = load_manifest_records(manifest_path, split=split)
        self.transform = transform
        self.classes = get_class_names_from_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, record.class_index, str(record.image_path)


def build_default_transforms(image_size: int = 224):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to build training transforms. "
            "Install torchvision in the training environment before running experiments."
        ) from exc

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform
