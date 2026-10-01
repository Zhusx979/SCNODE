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
    use_bn: Iterable[bool] = (True,),
    use_sam: Iterable[bool] = (True,),
    use_tconv: Iterable[bool] = (True,),
    use_twbn: Iterable[bool] = (True,),
    twbn_windows: Iterable[int] = (5,),
    twbn_grids: Iterable[int] = (11,),
    grid_policies: Iterable[str] = ("uniform",),
) -> list[ReviewCondition]:
    if experiment == "all":
        return sum((build_conditions(name, models=models, seeds=seeds, ode_entry_sizes=ode_entry_sizes,
                                     downsampling=downsampling, augment_dims=augment_dims,
                                     time_modes=time_modes, corruptions=corruptions,
                                     corruption_severities=corruption_severities, solver=solver,
                                     ode_steps=ode_steps, rtol=rtol, atol=atol,
                                     use_bn=use_bn, use_sam=use_sam, use_tconv=use_tconv, use_twbn=use_twbn,
                                     twbn_windows=twbn_windows, twbn_grids=twbn_grids, grid_policies=grid_policies)
                    for name in ("resolution", "augmentation", "time", "robustness", "ablation", "component_ablation")), [])
    comment_map = {"resolution": ("3",), "augmentation": ("4", "10"), "time": ("5",), "robustness": ("11",), "ablation": (), "component_ablation": ()}
    if experiment not in comment_map:
        raise ValueError(f"unknown experiment {experiment!r}")
    conditions: list[ReviewCondition] = []
    if experiment == "ablation":
        base = dict(solver=solver, ode_steps=ode_steps, rtol=rtol, atol=atol)
        grids = [
            ("A_entry_28", dict(ode_entry_size=28)), ("A_entry_112", dict(ode_entry_size=112)),
            ("A_avgpool", dict(downsampling="avgpool")), ("A_stride_conv", dict(downsampling="stride_conv")),
            ("B_aux_2", dict(augment_dim=2)), ("B_aux_4", dict(augment_dim=4)),
            ("B_aux_8", dict(augment_dim=8)), ("B_time_aux_2", dict(time_mode="concat", augment_dim=2)),
            ("B_time_aux_4", dict(time_mode="concat", augment_dim=4)), ("B_time_aux_8", dict(time_mode="concat", augment_dim=8)),
            ("C_autonomous", dict(time_mode="none", use_tconv=False)),
            ("C_fourier_film", dict(time_mode="fourier_film")),
            ("D_window_3", dict(twbn_window=3)), ("D_window_7", dict(twbn_window=7)),
            ("D_grid_9", dict(twbn_grids=9)), ("D_grid_17", dict(twbn_grids=17)),
            ("D_grid_33", dict(twbn_grids=33)), ("D_solver_time", dict(grid_policy="solver_time")),
            ("D_quantile", dict(grid_policy="quantile")),
        ]
        for row_name, overrides in grids:
            for model, seed in product(models, seeds):
                conditions.append(ReviewCondition(row_name, comment_map[experiment], model, seed,
                    ScnodeConfig(**{**base, **overrides})))
        return conditions
    if experiment == "component_ablation":
        component_rows = {
            "bn_only": (True, False, False, False),
            "sam_only": (False, True, False, False),
            "tconv_only": (False, False, True, False),
            "twbn_only": (False, False, False, True),
            "full_scnnode": (True, True, True, True),
        }
        for row_name, flags in component_rows.items():
            for model, seed in product(models, seeds):
                base = dict(solver=solver, ode_steps=ode_steps, rtol=rtol, atol=atol,
                            use_bn=flags[0], use_sam=flags[1], use_tconv=flags[2], use_twbn=flags[3])
                conditions.append(ReviewCondition(row_name, comment_map[experiment], model, seed, ScnodeConfig(**base)))
    elif experiment == "resolution":
        grid = product(models, seeds, ode_entry_sizes, downsampling)
        for model, seed, size, mode in grid:
            conditions.append(ReviewCondition(experiment, comment_map[experiment], model, seed,
                ScnodeConfig(solver=solver, ode_steps=ode_steps, rtol=rtol, atol=atol,
                             ode_entry_size=size, downsampling=mode)))
    elif experiment == "augmentation":
        for model, seed in product(models, seeds):



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


            if 0 in corruption_severities:
                conditions.append(ReviewCondition(experiment, comment_map[experiment], model, seed, config))
            for corruption, severity in product(corruptions, corruption_severities):
                if severity:
                    conditions.append(ReviewCondition(experiment, comment_map[experiment], model, seed, config, corruption, severity))
    return conditions
