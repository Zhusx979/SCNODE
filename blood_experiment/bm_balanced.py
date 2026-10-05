from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from SCNODE.blood_experiment.data import ManifestRecord, macenko_tellez_perturb


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def corner_median_fill(image: Image.Image) -> tuple[int, int, int]:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    patches = (
        array[0, 0], array[0, width - 1], array[height - 1, 0], array[height - 1, width - 1],
    )
    pixels = np.asarray(patches, dtype=np.uint8)
    if pixels.size == 0:
        return (0, 0, 0)
    return tuple(np.median(pixels, axis=0).round().astype(np.uint8).tolist())


def deterministic_records(
    records: Sequence[ManifestRecord], target_per_class: int, seed: int
) -> list[ManifestRecord]:
    if target_per_class <= 0:
        raise ValueError("target_per_class must be positive")
    grouped: dict[int, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.class_index].append(record)
    rng = random.Random(seed)
    balanced: list[ManifestRecord] = []
    for class_index in sorted(grouped):
        candidates = list(grouped[class_index])
        rng.shuffle(candidates)
        if len(candidates) >= target_per_class:
            balanced.extend(candidates[:target_per_class])
            continue
        balanced.extend(candidates)
        for index in range(target_per_class - len(candidates)):
            balanced.append(candidates[index % len(candidates)])
    return balanced


class EvaluationDataset(Dataset):
    def __init__(self, records: Sequence[ManifestRecord], image_size: int = 224):
        self.records = list(records)
        self.image_size = image_size
        self.classes = [name for _, name in sorted({(r.class_index, r.class_name) for r in records})]
        self._bad_image_paths: set[str] = set()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def skipped_image_count(self) -> int:
        return len(self._bad_image_paths)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record.image_path) as image:
                image = image.convert("RGB")
            image = TF.resize(image, [self.image_size, self.image_size])
            image = TF.to_tensor(image)
            image = TF.normalize(image, MEAN, STD)
            return image, record.class_index, str(record.image_path)
        except (OSError, UnidentifiedImageError) as exc:
            self._bad_image_paths.add(str(record.image_path))
            raise RuntimeError("Unable to read image '{}': {}".format(record.image_path, exc)) from exc


class BalancedTrainDataset(EvaluationDataset):
    def __init__(
        self,
        records: Sequence[ManifestRecord],
        target_per_class: int = 6000,
        image_size: int = 224,
        seed: int = 42,
        translate_fraction: float = 0.10,
        shear_degrees: float = 5.0,
        stain_probability: float = 0.25,
        stain_strength: float = 0.5,
        background_fill: bool = True,
    ):
        super().__init__(records, image_size)
        self.records = deterministic_records(records, target_per_class, seed)
        self.seed = seed
        self.epoch = 0
        self.translate_fraction = translate_fraction
        self.shear_degrees = shear_degrees
        self.stain_probability = stain_probability
        self.stain_strength = stain_strength
        self.background_fill = background_fill

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.epoch * max(len(self.records), 1) + index)

    def _augment(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        fill = corner_median_fill(image) if self.background_fill else (0, 0, 0)
        angle = float(rng.uniform(0.0, 180.0))
        image = TF.rotate(image, angle=-angle, fill=fill)
        if rng.random() < 0.5:
            image = TF.hflip(image)
        if rng.random() < 0.5:
            image = TF.vflip(image)
        tx = int(rng.uniform(-self.translate_fraction, self.translate_fraction) * image.width)
        ty = int(rng.uniform(-self.translate_fraction, self.translate_fraction) * image.height)
        shear = float(rng.uniform(-self.shear_degrees, self.shear_degrees))
        image = TF.affine(image, angle=0.0, translate=(tx, ty), scale=1.0, shear=(shear, 0.0), fill=fill)
        if rng.random() < self.stain_probability:
            image = macenko_tellez_perturb(image, rng=rng, strength=self.stain_strength)
        return image

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record.image_path) as image:
                image = image.convert("RGB")
            image = self._augment(image, self._rng(index))
            image = TF.resize(image, [self.image_size, self.image_size])
            image = TF.to_tensor(image)
            image = TF.normalize(image, MEAN, STD)
            return image, record.class_index, str(record.image_path)
        except (OSError, UnidentifiedImageError) as exc:
            self._bad_image_paths.add(str(record.image_path))
            raise RuntimeError("Unable to read image '{}': {}".format(record.image_path, exc)) from exc
