from pathlib import Path

import numpy as np
import torch
from PIL import Image

from SCNODE.blood_experiment.bm_balanced import BalancedTrainDataset, EvaluationDataset
from SCNODE.blood_experiment.data import ManifestRecord
from SCNODE.models.ode.scnode.scnode_resnet import BasicBlock, BasicBlock2, TWBN
from SCNODE.training.run_bm_balanced_aug_experiment import atomic_torch_save, build_parser


def _records(tmp_path: Path):
    records = []
    for class_index, name in enumerate(("A", "B")):
        path = tmp_path / name / "image.png"
        path.parent.mkdir()
        Image.new("RGB", (32, 32), (50 + class_index * 100, 80, 120)).save(path)
        records.append(ManifestRecord("train", name, class_index, path))
    return records


def test_balanced_dataset_reaches_target_and_is_finite(tmp_path: Path):
    dataset = BalancedTrainDataset(_records(tmp_path), target_per_class=6, image_size=16)
    assert len(dataset) == 12
    assert all(sum(record.class_index == index for record in dataset.records) == 6 for index in (0, 1))
    image, label, _ = dataset[0]
    assert image.shape == (3, 16, 16)
    assert torch.isfinite(image).all()
    assert label in (0, 1)


def test_evaluation_dataset_is_deterministic(tmp_path: Path):
    dataset = EvaluationDataset(_records(tmp_path), image_size=16)
    first = dataset[0][0]
    second = dataset[0][0]
    assert torch.equal(first, second)


def test_training_entry_defaults_match_bm_protocol():
    args = build_parser().parse_args(["--raw-data-root", "raw", "--prepared-data-root", "split", "--output-root", "out"])
    assert (args.target_per_class, args.epochs, args.batch_size, args.num_workers, args.seed) == (6000, 20, 64, 8, 42)
    assert (args.translate_fraction, args.shear_degrees, args.stain_probability, args.stain_strength) == (0.1, 5.0, 0.25, 0.5)


def test_atomic_checkpoint_round_trip(tmp_path: Path):
    path = tmp_path / "last_checkpoint.pt"
    atomic_torch_save({"epoch": 1, "value": torch.tensor([1.0])}, path)
    payload = torch.load(path, map_location="cpu")
    assert payload["epoch"] == 1
    assert torch.equal(payload["value"], torch.tensor([1.0]))


def test_normalization_roles_are_separate():
    ordinary = BasicBlock(64, 64)
    ode = BasicBlock2(64, 65, augment_dim=1, use_bn=True, use_twbn=True)
    assert isinstance(ordinary.bn1, torch.nn.BatchNorm2d)
    assert isinstance(ordinary.bn2, torch.nn.BatchNorm2d)
    assert isinstance(ode.bn1, TWBN)
    assert isinstance(ode.bn2, TWBN)
    assert ode.bn1.num_grids == ode.bn2.num_grids == 11
    assert ode.bn1.window_size == ode.bn2.window_size == 5
    assert np.isfinite(ode.bn1.running_mean.detach().numpy()).all()
    assert np.isfinite(ode.bn1.running_var.detach().numpy()).all()
