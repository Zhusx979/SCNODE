"""Solver-trajectory adapter used by visualization scripts."""

from __future__ import annotations

import torch


def forward_with_solver_trajectories(
    model,
    model_name: str,
    inputs: torch.Tensor,
    time_points: torch.Tensor,
    *,
    solver: str = "rk4",
    rtol: float = 1e-3,
    atol: float = 1e-3,
    solver_steps: int = 10,
):
    if not hasattr(model, "forward_with_trajectory"):
        raise ValueError(f"{model_name} does not expose forward_with_trajectory")
    try:
        return model.forward_with_trajectory(inputs, time_points, solver=solver, rtol=rtol, atol=atol)
    except TypeError:
        return model.forward_with_trajectory(inputs, time_points)
