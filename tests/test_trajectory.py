import pytest
import torch

from SCNODE.diagnostics.trajectory import build_time_grid, pool_trajectory, trajectory_geometry


def test_pool_trajectory_separates_main_and_auxiliary_channels() -> None:
    states = torch.ones(3, 2, 5, 2, 2)

    pooled = pool_trajectory(states, augment_dim=2)

    assert pooled["main"].shape == (3, 2, 3)
    assert pooled["auxiliary"].shape == (3, 2, 2)


def test_time_grid_has_evenly_spaced_endpoints() -> None:
    assert build_time_grid(3).tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_trajectory_geometry_is_zero_for_constant_path() -> None:
    states = torch.ones(4, 2, 3, 2, 2)

    metrics = trajectory_geometry(states)

    assert metrics["mean_path_length"] == pytest.approx(0.0)
    assert metrics["mean_curvature"] == pytest.approx(0.0)
