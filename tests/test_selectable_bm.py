from pathlib import Path

import pytest
from PIL import Image

from SCNODE.training.run_bm_experiment_selectable import (
    build_parser,
    build_selectable_dataloaders,
)


def _make_dataset(root: Path) -> None:
    for class_name in ("A", "B"):
        directory = root / class_name
        directory.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (12, 12), color=(index * 20, 30, 40)).save(directory / "{}.png".format(index))


def _args(tmp_path: Path, mode: str):
    return build_parser().parse_args(
        [
            "--raw-data-root", str(tmp_path / "raw"),
            "--experiment-root", str(tmp_path / "experiment"),
            "--target-per-class", "5",
            "--augmentation-mode", mode,
            "--num-workers", "0",
        ]
    )


def test_none_mode_does_not_upsample_or_use_stain(tmp_path, monkeypatch):
    _make_dataset(tmp_path / "raw")
    args = _args(tmp_path, "none")
    monkeypatch.setattr(
        "SCNODE.blood_experiment.data.MacenkoTellezStainAugmentation.__call__",
        lambda self, image: (_ for _ in ()).throw(AssertionError("stain augmentation called")),
    )
    loaders, _, _, upsampling = build_selectable_dataloaders(args)
    assert upsampling is False
    assert len(loaders["train"].dataset) == 4
    assert len(loaders["val"].dataset) == 2
    assert len(loaders["test"].dataset) == 2


def test_stain_geometry_mode_upsamples(tmp_path):
    _make_dataset(tmp_path / "raw")
    args = _args(tmp_path, "stain_geometry")
    loaders, _, _, upsampling = build_selectable_dataloaders(args)
    assert upsampling is True
    assert len(loaders["train"].dataset) == 10


def test_invalid_augmentation_mode_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, "invalid")


def test_selectable_defaults_use_six_thousand_training_augmentation():
    args = build_parser().parse_args(["--raw-data-root", "raw", "--experiment-root", "experiment"])
    assert args.target_per_class == 6000
    assert args.augmentation_mode == "stain_geometry"
