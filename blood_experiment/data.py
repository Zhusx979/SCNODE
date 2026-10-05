from __future__ import annotations

import csv
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError


ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_TARGET_SAMPLES_PER_CLASS = 6000


def get_default_raw_dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "BM_cytomorphology_data"


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
    max_samples_per_class: int | None = DEFAULT_TARGET_SAMPLES_PER_CLASS,
) -> Path:
    raw_root_path = Path(raw_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir_path / "split_manifest.csv"
    selection_rows: list[dict[str, str | int]] = []
    rows: list[dict[str, str | int]] = []
    generator = random.Random(seed)

    for class_index, class_name in enumerate(discover_class_names(raw_root_path)):
        class_paths = discover_image_paths(raw_root_path / class_name)
        if not class_paths:
            continue
        generator.shuffle(class_paths)
        original_count = len(class_paths)
        if max_samples_per_class is not None:
            if max_samples_per_class <= 0:
                raise ValueError("max_samples_per_class must be positive or None.")
            class_paths = class_paths[:max_samples_per_class]
        for image_path in class_paths:
            selection_rows.append(
                {
                    "class_name": class_name,
                    "class_index": class_index,
                    "image_path": str(image_path),
                    "original_count": original_count,
                    "selection": "downsampled" if original_count > len(class_paths) else "retained",
                }
            )
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

    with (output_dir_path / "subset_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "class_index", "image_path", "original_count", "selection"],
        )
        writer.writeheader()
        writer.writerows(selection_rows)

    metadata_path = output_dir_path / "split_manifest_config.json"
    metadata_path.write_text(
        json.dumps(
            {
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "seed": seed,
                "max_samples_per_class": max_samples_per_class,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir_path / "subset_metadata.json").write_text(
        json.dumps(
            {
                "selection_seed": seed,
                "max_samples_per_class": max_samples_per_class,
                "class_count": len(discover_class_names(raw_root_path)),
                "selection_policy": "downsample classes above the cap; retain all classes at or below the cap",
                "image_storage": "manifest references original files; no image copies are created",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

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
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_samples_per_class: int | None = DEFAULT_TARGET_SAMPLES_PER_CLASS,
) -> Path:
    destination = Path(output_dir)
    manifest_path = destination / "split_manifest.csv"
    metadata_path = destination / "split_manifest_config.json"
    expected = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "max_samples_per_class": max_samples_per_class,
    }
    required_metadata = destination / "subset_metadata.json"
    required_selection = destination / "subset_selection.csv"
    if manifest_path.exists() and metadata_path.exists() and required_metadata.exists() and required_selection.exists():
        try:
            if json.loads(metadata_path.read_text(encoding="utf-8")) == expected:
                return manifest_path
        except (OSError, ValueError):
            pass
    return create_split_manifest(
        raw_root=raw_root,
        output_dir=destination,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        max_samples_per_class=max_samples_per_class,
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
        target_samples_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        self.records = load_manifest_records(manifest_path, split=split)
        if target_samples_per_class is not None:
            if target_samples_per_class <= 0:
                raise ValueError("target_samples_per_class must be positive or None.")
            self.records = _upsample_records(
                self.records, target_samples_per_class=target_samples_per_class, seed=seed
            )
        self.transform = transform
        self.classes = get_class_names_from_manifest(manifest_path)
        self._bad_image_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def skipped_image_count(self) -> int:
        return len(self._bad_image_paths)

    def _load_image(self, record: ManifestRecord):
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image

    def __getitem__(self, index: int):
        if not self.records:
            raise IndexError("ManifestImageDataset is empty.")

        attempted_paths: list[str] = []
        for offset in range(len(self.records)):
            record = self.records[(index + offset) % len(self.records)]
            try:
                image = self._load_image(record)
                return image, record.class_index, str(record.image_path)
            except (OSError, UnidentifiedImageError) as exc:
                bad_path = str(record.image_path)
                attempted_paths.append(bad_path)
                if bad_path not in self._bad_image_paths:
                    warnings.warn(
                        f"Skipping unreadable image '{bad_path}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._bad_image_paths.add(bad_path)

        attempted = ", ".join(attempted_paths)
        raise RuntimeError(
            "No readable images were found in the dataset records attempted: "
            f"{attempted}"
        )


def _upsample_records(
    records: list[ManifestRecord], target_samples_per_class: int, seed: int
) -> list[ManifestRecord]:
    grouped: dict[int, list[ManifestRecord]] = {}
    for record in records:
        grouped.setdefault(record.class_index, []).append(record)
    generator = random.Random(seed)
    balanced: list[ManifestRecord] = []
    for class_index in sorted(grouped):
        class_records = grouped[class_index]
        if len(class_records) >= target_samples_per_class:
            balanced.extend(class_records[:target_samples_per_class])
            continue
        balanced.extend(class_records)
        current_count = len(class_records)
        while current_count < target_samples_per_class:
            balanced.append(generator.choice(class_records))
            current_count += 1
    return balanced


class MacenkoTellezStainAugmentation:
    def __init__(self, sigma: float = 0.1, bias: float = 0.1, probability: float = 0.25) -> None:
        self.sigma = sigma
        self.bias = bias
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        try:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            optical_density = -np.log((rgb + 1.0) / 256.0)
            mask = optical_density.reshape(-1, 3).mean(axis=1) > 0.15
            if mask.sum() < 10:
                return image
            od = optical_density.reshape(-1, 3)[mask]
            covariance = np.cov(od, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            basis = eigenvectors[:, np.argsort(eigenvalues)[-2:]]
            projections = od @ basis
            angles = np.arctan2(projections[:, 1], projections[:, 0])
            vectors = []
            for percentile in (1, 99):
                theta = np.percentile(angles, percentile)
                vector = basis @ np.asarray([np.cos(theta), np.sin(theta)])
                vector /= np.linalg.norm(vector) + 1e-8
                vectors.append(vector)
            stain_matrix = np.stack(vectors, axis=0)
            if stain_matrix[0, 0] < stain_matrix[1, 0]:
                stain_matrix = stain_matrix[::-1]
            concentrations = optical_density.reshape(-1, 3) @ np.linalg.pinv(stain_matrix)
            scales = np.exp(np.random.normal(0.0, self.sigma, size=(1, 2)))
            offsets = np.random.normal(0.0, self.bias, size=(1, 2))
            concentrations = np.maximum(concentrations * scales + offsets, 0.0)
            reconstructed = concentrations @ stain_matrix
            augmented = np.clip(255.0 * np.exp(-reconstructed), 0, 255).reshape(rgb.shape)
            if not np.isfinite(augmented).all() or augmented.std() < 1e-3:
                return image
            return Image.fromarray(augmented.astype(np.uint8), mode="RGB")
        except (ImportError, FloatingPointError, ValueError, np.linalg.LinAlgError):
            return image


MAX_TRANSLATION_FRACTION = 0.10
MAX_SHEAR_DEGREES = 5.0
STAIN_AUGMENTATION_PROBABILITY = 0.25
STAIN_INTENSITY_SIGMA = 0.1
STAIN_INTENSITY_BIAS = 0.1


class RandomAffineMedianFill:
    def __init__(self, degrees, translate, shear):
        self.degrees = degrees
        self.translate = translate
        self.shear = shear

    @staticmethod
    def _corner_median(image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = array.shape[:2]
        patches = (
            array[0, 0], array[0, width - 1], array[height - 1, 0], array[height - 1, width - 1],
        )
        pixels = np.asarray(patches, dtype=np.uint8)
        return tuple(np.median(pixels, axis=0).round().astype(np.uint8).tolist())

    def __call__(self, image):
        from torchvision.transforms import InterpolationMode
        from torchvision import transforms
        from torchvision.transforms import functional as TF

        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            self.degrees, self.translate, None, self.shear, [image.height, image.width]
        )
        return TF.affine(
            image,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=self._corner_median(image),
        )


def macenko_tellez_perturb(image: Image.Image, rng: np.random.Generator, strength: float = 0.5) -> Image.Image:
    try:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width, _ = arr.shape
        flat = arr.reshape(-1, 3)
        od = -np.log(np.clip(flat, 1e-6, 1.0))
        keep = np.all(od > 0.15, axis=1)
        sample = od[keep]
        if sample.shape[0] < 20:
            return image
        _, _, vh = np.linalg.svd(sample - sample.mean(axis=0), full_matrices=False)
        basis = vh[:2]
        projection = sample @ basis.T
        theta = np.arctan2(projection[:, 1], projection[:, 0])
        low, high = np.percentile(theta, [1, 99])
        vectors = np.stack((np.cos([low, high]), np.sin([low, high])), axis=1) @ basis
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
        if vectors[0, 0] < vectors[1, 0]:
            vectors = vectors[::-1]
        stain_matrix = vectors.T
        concentrations = np.linalg.lstsq(stain_matrix, od.T, rcond=None)[0]
        concentration_scale = rng.lognormal(mean=0.0, sigma=0.08 * strength, size=(2, 1))
        perturbed_matrix = stain_matrix * (1.0 + rng.normal(0.0, 0.03 * strength, stain_matrix.shape))
        perturbed_matrix /= np.linalg.norm(perturbed_matrix, axis=0, keepdims=True).clip(min=1e-8)
        reconstructed_od = perturbed_matrix @ (concentrations * concentration_scale)
        output = np.exp(-np.clip(reconstructed_od.T, -8.0, 8.0)).reshape(height, width, 3)
        if not np.isfinite(output).all():
            return image
        return Image.fromarray(np.uint8(np.clip(output * 255.0, 0, 255)), mode="RGB")
    except (FloatingPointError, ValueError, np.linalg.LinAlgError, OverflowError):
        return image


def build_default_transforms(image_size: int = 224, paper_augmentation: bool = True):
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
            RandomAffineMedianFill(
                degrees=(-180, 0),
                translate=(MAX_TRANSLATION_FRACTION, MAX_TRANSLATION_FRACTION),
                shear=(-MAX_SHEAR_DEGREES, MAX_SHEAR_DEGREES),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            MacenkoTellezStainAugmentation(
                sigma=STAIN_INTENSITY_SIGMA,
                bias=STAIN_INTENSITY_BIAS,
                probability=STAIN_AUGMENTATION_PROBABILITY,
            ) if paper_augmentation else transforms.Lambda(lambda image: image),
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
