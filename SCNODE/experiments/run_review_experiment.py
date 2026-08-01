"""Command-line entry point for reviewer-response experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from blood_experiment.data import get_default_raw_dataset_root
from blood_experiment.robustness import CORRUPTION_NAMES
from SCNODE.experiments.configs import build_conditions
from SCNODE.experiments.runner import run_conditions
from SCNODE.training.experiment_config import AVAILABLE_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible SCNODE reviewer experiments")
    parser.add_argument("--experiment", choices=["resolution", "augmentation", "time", "robustness", "all"], required=True)
    parser.add_argument("--models", nargs="+", default=["SCNODE_ResNet18"], choices=sorted(AVAILABLE_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--solver", choices=["euler", "rk4", "dopri5"], default="rk4")
    parser.add_argument("--ode_steps", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--time_modes", nargs="+", choices=["none", "concat", "fourier_film"], default=["concat"])
    parser.add_argument("--augment_dims", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--ode_entry_sizes", nargs="+", type=int, default=[56, 28])
    parser.add_argument("--downsampling", nargs="+", choices=["maxpool", "avgpool", "stride_conv"], default=["maxpool", "avgpool", "stride_conv"])
    parser.add_argument("--corruptions", nargs="+", choices=CORRUPTION_NAMES, default=list(CORRUPTION_NAMES))
    parser.add_argument("--corruption_severities", nargs="+", type=int, choices=[0, 1, 2, 3], default=[0, 1, 2, 3])
    parser.add_argument("--trajectory_steps", type=int, default=11)
    parser.add_argument("--trajectory_samples", type=int, default=21)
    parser.add_argument("--output_root", type=Path, default=Path("artifacts/review_experiments"))
    parser.add_argument("--raw_data_root", type=Path, default=get_default_raw_dataset_root())
    parser.add_argument("--prepared_data_root", type=Path, default=Path("artifacts/datasets/bm_21class_split"))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    conditions = build_conditions(
        args.experiment, models=args.models, seeds=args.seeds, ode_entry_sizes=args.ode_entry_sizes,
        downsampling=args.downsampling, augment_dims=args.augment_dims, time_modes=args.time_modes,
        corruptions=args.corruptions, corruption_severities=args.corruption_severities,
        solver=args.solver, ode_steps=args.ode_steps, rtol=args.rtol, atol=args.atol,
    )
    run_conditions(
        conditions, output_root=args.output_root, dry_run=args.dry_run, raw_data_root=args.raw_data_root,
        prepared_data_root=args.prepared_data_root, image_size=args.image_size, batch_size=args.batch_size,
        test_batch_size=args.test_batch_size, num_workers=args.num_workers, num_epochs=args.num_epochs,
        learning_rate=args.learning_rate, use_gpu=not args.cpu,
        trajectory_steps=args.trajectory_steps, trajectory_samples=args.trajectory_samples,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
