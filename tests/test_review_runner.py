from pathlib import Path

from SCNODE.experiments.runner import artifact_dir, run_conditions
from SCNODE.experiments.configs import build_conditions


def test_dry_run_returns_conditions_without_creating_artifacts(tmp_path: Path) -> None:
    conditions = build_conditions(
        "time", models=["SCNODE_ResNet18"], seeds=[42], time_modes=["concat"]
    )

    output_root = tmp_path / "review_artifacts"
    returned = run_conditions(conditions, output_root=output_root, dry_run=True)

    assert returned == conditions
    assert not output_root.exists()


def test_artifact_dir_is_stable_and_includes_seed(tmp_path: Path) -> None:
    condition = build_conditions(
        "resolution", models=["SCNODE_ResNet18"], seeds=[42],
        ode_entry_sizes=[56], downsampling=["maxpool"],
    )[0]

    path = artifact_dir(tmp_path, condition)

    assert path.name == "seed_42"
    assert "resolution" in path.as_posix()
