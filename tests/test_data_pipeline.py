from pathlib import Path

import csv
from PIL import Image

from blood_experiment.data import (
    ManifestImageDataset,
    allocate_split_counts,
    create_split_manifest,
    discover_class_names,
    get_default_raw_dataset_root,
)


def _make_fake_class(root: Path, class_name: str, count: int) -> None:
    image_dir = root / class_name / "0001-1000"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (image_dir / f"{class_name}_{index:04d}.jpg").write_bytes(b"fake")


def test_default_raw_dataset_root_points_to_workspace_dataset() -> None:
    dataset_root = get_default_raw_dataset_root()
    assert dataset_root.name == "BM_cytomorphology_data"


def test_allocate_split_counts_preserves_total_and_each_split() -> None:
    counts = allocate_split_counts(8, train_ratio=0.75, val_ratio=0.125, test_ratio=0.125)
    assert counts == {"train": 6, "val": 1, "test": 1}


def test_create_split_manifest_generates_three_splits(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _make_fake_class(raw_root, "ABE", 8)
    _make_fake_class(raw_root, "BLA", 8)

    manifest_path = create_split_manifest(
        raw_root=raw_root,
        output_dir=tmp_path / "prepared",
        train_ratio=0.75,
        val_ratio=0.125,
        test_ratio=0.125,
        seed=7,
    )

    assert manifest_path.exists()
    assert discover_class_names(raw_root) == ["ABE", "BLA"]

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 16
    assert {row["split"] for row in rows} == {"train", "val", "test"}

    per_class = {}
    for row in rows:
        per_class.setdefault(row["class_name"], {}).setdefault(row["split"], 0)
        per_class[row["class_name"]][row["split"]] += 1

    assert per_class["ABE"] == {"train": 6, "val": 1, "test": 1}
    assert per_class["BLA"] == {"train": 6, "val": 1, "test": 1}


def test_manifest_dataset_skips_broken_images(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    image_dir = raw_root / "ABE" / "0001-1000"
    image_dir.mkdir(parents=True, exist_ok=True)

    broken_path = image_dir / "ABE_0000.jpg"
    broken_path.write_bytes(b"not-a-valid-image")

    valid_path = image_dir / "ABE_0001.jpg"
    Image.new("RGB", (12, 12), color=(120, 30, 220)).save(valid_path)

    manifest_path = tmp_path / "prepared" / "split_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_name", "class_index", "image_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "class_name": "ABE",
                "class_index": 0,
                "image_path": str(broken_path),
            }
        )
        writer.writerow(
            {
                "split": "train",
                "class_name": "ABE",
                "class_index": 0,
                "image_path": str(valid_path),
            }
        )

    dataset = ManifestImageDataset(manifest_path, split="train", transform=None)
    image, class_index, image_path = dataset[0]

    assert image.size == (12, 12)
    assert class_index == 0
    assert image_path == str(valid_path)
