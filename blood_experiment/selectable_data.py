from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from SCNODE.blood_experiment.data import ManifestRecord


class NoAugmentationDataset(Dataset):
    def __init__(self, records: Sequence[ManifestRecord], image_size: int = 224) -> None:
        self.records = list(records)
        self.image_size = image_size
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
            image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            return image, record.class_index, str(record.image_path)
        except (OSError, UnidentifiedImageError) as exc:
            self._bad_image_paths.add(str(record.image_path))
            raise RuntimeError("Unable to read image '{}': {}".format(record.image_path, exc)) from exc


class EvalDataset(NoAugmentationDataset):
    pass
