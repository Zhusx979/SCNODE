"""Exportable summaries of model-internal ODE trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def build_time_grid(points: int) -> torch.Tensor:
    if points < 2:
        raise ValueError("points must be at least 2")
    return torch.linspace(0.0, 1.0, points)


def pool_trajectory(states: torch.Tensor, augment_dim: int) -> dict[str, torch.Tensor]:
    if states.ndim != 5:
        raise ValueError("states must have shape [time, batch, channels, height, width]")
    if augment_dim < 0 or augment_dim > states.shape[2]:
        raise ValueError("augment_dim must be between zero and the channel count")
    pooled = states.mean(dim=(-1, -2))
    split = states.shape[2] - augment_dim
    return {"main": pooled[..., :split], "auxiliary": pooled[..., split:]}


def trajectory_geometry(states: torch.Tensor) -> dict[str, float]:
    if states.ndim != 5 or states.shape[0] < 2:
        raise ValueError("states must contain at least two [T, B, C, H, W] states")
    flattened = states.flatten(start_dim=2)
    delta = flattened[1:] - flattened[:-1]
    curvature = (
        flattened[2:] - 2 * flattened[1:-1] + flattened[:-2]
        if states.shape[0] > 2 else torch.zeros_like(delta[:1])
    )
    return {
        "mean_path_length": float(delta.norm(dim=-1).sum(dim=0).mean().detach().cpu()),
        "mean_adjacent_displacement": float(delta.norm(dim=-1).mean().detach().cpu()),
        "mean_curvature": float(curvature.norm(dim=-1).mean().detach().cpu()),
    }


def export_trajectory_npz(path: Path | str, states: torch.Tensor, augment_dim: int, **metadata) -> Path:
    """Store pooled features only, avoiding very large spatial-state artifacts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pooled = pool_trajectory(states, augment_dim)
    np.savez_compressed(
        destination,
        time=np.linspace(0.0, 1.0, states.shape[0]),
        main=pooled["main"].detach().cpu().numpy(),
        auxiliary=pooled["auxiliary"].detach().cpu().numpy(),
        **{key: np.asarray(value) for key, value in metadata.items()},
    )
    return destination
