from pathlib import Path


def test_scnode_uses_descriptive_training_and_model_paths() -> None:
    scnode_root = Path("SCNODE")
    expected_paths = [
        scnode_root / "training" / "run_bm_experiment.py",
        scnode_root / "training" / "classification_trainer.py",
        scnode_root / "training" / "experiment_config.py",
        scnode_root / "training" / "run_cifar10_experiment.py",
        scnode_root / "diagnostics" / "inspect_ode_attention.py",
        scnode_root / "models" / "cnn" / "resnet18_family.py",
        scnode_root / "models" / "cnn" / "resnet32_family.py",
        scnode_root / "models" / "cnn" / "resnet50_family.py",
        scnode_root / "models" / "baselines" / "efficientnet_baseline.py",
        scnode_root / "models" / "baselines" / "vision_transformer_baseline.py",
        scnode_root / "models" / "ode" / "odenet_variants.py",
        scnode_root / "models" / "ode" / "odenet_reference.py",
        scnode_root / "models" / "ode" / "scnode" / "scnode_resnet.py",
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, missing


def test_scnode_archives_legacy_results_and_datasets() -> None:
    scnode_root = Path("SCNODE")
    expected_paths = [
        scnode_root / "archive" / "datasets" / "all_dataset",
        scnode_root / "archive" / "datasets" / "cifar10_data",
        scnode_root / "archive" / "results" / "bm_experiment_result",
        scnode_root / "archive" / "results" / "cifar10_experiment_result",
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, missing


def test_old_ambiguous_top_level_filenames_are_removed() -> None:
    scnode_root = Path("SCNODE")
    old_paths = [
        scnode_root / "Train_Test_Val.py",
        scnode_root / "Function_Train_Test_Val.py",
        scnode_root / "config.py",
        scnode_root / "cifar10_train.py",
        scnode_root / "check.py",
        scnode_root / "model",
        scnode_root / "compare_experience",
        scnode_root / "ODEmodel",
    ]

    remaining = [str(path) for path in old_paths if path.exists()]
    assert not remaining, remaining
