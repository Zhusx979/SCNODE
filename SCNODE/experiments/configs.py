"""Declarative condition grids for reviewer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

from SCNODE.models.ode.scnode.config import ScnodeConfig


@dataclass(frozen=True)
class ReviewCondition:
    experiment: str
    reviewer_comments: tuple[str, ...]
    model_name: str
    seed: int
    scnode_config: ScnodeConfig
    corruption: str | None = None
    corruption_severity: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scnode_config"] = asdict(self.scnode_config)
        return payload


def build_conditions(
    experiment: str,
    *,
    models: Iterable[str],
    seeds: Iterable[int],
    ode_entry_sizes: Iterable[int] = (56,),
    downsampling: Iterable[str] = ("maxpool",),
    augment_dims: Iterable[int] = (1,),
    time_modes: Iterable[str] = ("concat",),
    corruptions: Iterable[str] = ("brightness",),
    corruption_severities: Iterable[int] = (0, 1, 2, 3),
    solver: str = "rk4",
    ode_steps: int = 4,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> list[ReviewCondition]:
    if experiment == "all":
        return sum((build_conditions(name, models=models, seeds=seeds, ode_entry_sizes=ode_entry_sizes,
                                     downsampling=downsampling, augment_dims=augment_dims,
                                     time_modes=time_modes, corruptions=corruptions,
                                     corruption_severities=corruption_severities, solver=solver,
                                     ode_steps=ode_steps, rtol=rtol, atol=atol)
                    for name in ("resolution", "augmentation", "time", "robustness")), [])
    comment_map = {"resolution": ("3",), "augmentation": ("4", "10"), "time": ("5",), "robustness": ("11",)}
    if experiment not in comment_map:
        raise ValueError(f"unknown experiment {experiment!r}")
    conditions: list[ReviewCondition] = []
    if experiment == "resolution":
        grid = product(models, seeds, ode_entry_sizes, downsampling)
        for model, seed, size, mode in grid:
            conditions.append(ReviewCondition(experiment, comment_map[experiment], model, seed,
                ScnodeConfig(solver=solver, ode_steps=ode_steps, rtol=rtol, atol=atol,
                             ode_entry_size=size, downsampling=mode)))
    elif experiment == "augmentation":
        for model, seed in product(models, seeds):
            # Legacy ANODE factories do not expose the same augmentation-width
            # interface; include one explicitly labelled baseline rather than
            # silently duplicating the identical run for every width.
            dimensions = augment_dims if model.startswith("SCNODE_") else (1,)
            for dim in dimensions:
                conditions.append(ReviewCondition(experiment, comment_map[experiment], model, seed,
                    ScnodeConfig(solver=solver, ode_steps=ode_steps, rtol=rtol, atol=atol, augment_dim=dim)))
    elif experiment == "time":
        for model, seed, mode in product(models, seeds, time_modes):
            conditions.append(ReviewCondition(experiment, comment_map[experiment], model, seed,
                ScnodeConfig(solver=solver, ode_steps=ode_steps, rtol=rtol, atol=atol, time_mode=mode)))
    else:
        for model, seed in product(models, seeds):
            config = ScnodeConfig(solver=solver, ode_steps=ode_steps, rtol=rtol, atol=atol)
            # Clean evaluation is shared across corruption families.  Repeating
            # it once per family wastes runs and would overweight clean scores.
            if 0 in corruption_severities:
                conditions.append(ReviewCondition(experiment, comment_map[experiment], model, seed, config))
            for corruption, severity in product(corruptions, corruption_severities):
                if severity:
                    conditions.append(ReviewCondition(experiment, comment_map[experiment], model, seed, config, corruption, severity))
    return conditions
