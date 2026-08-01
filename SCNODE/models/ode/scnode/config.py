"""Configuration shared by SCNODE model factories and ODE blocks."""

from dataclasses import dataclass
import math
from typing import Literal


SolverName = Literal["euler", "rk4", "dopri5"]
TimeMode = Literal["none", "concat", "fourier_film"]
DownsamplingMode = Literal["maxpool", "avgpool", "stride_conv"]

SUPPORTED_ODE_ENTRY_SIZES = frozenset({14, 28, 56, 112})


@dataclass(frozen=True)
class ScnodeConfig:
    """Validated architecture and integration settings for SCNODE."""

    solver: SolverName = "rk4"
    ode_steps: int = 4
    rtol: float = 1e-3
    atol: float = 1e-3
    time_mode: TimeMode = "concat"
    augment_dim: int = 1
    downsampling: DownsamplingMode = "maxpool"
    ode_entry_size: int = 56

    def __post_init__(self) -> None:
        if not isinstance(self.solver, str) or self.solver not in {"euler", "rk4", "dopri5"}:
            raise ValueError("solver must be one of: euler, rk4, dopri5")
        if type(self.ode_steps) is not int or self.ode_steps < 1:
            raise ValueError("ode_steps must be an integer at least 1")
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and positive")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.time_mode, str) or self.time_mode not in {"none", "concat", "fourier_film"}:
            raise ValueError("time_mode must be one of: none, concat, fourier_film")
        if type(self.augment_dim) is not int or self.augment_dim < 0:
            raise ValueError("augment_dim must be an integer non-negative value")
        if not isinstance(self.downsampling, str) or self.downsampling not in {"maxpool", "avgpool", "stride_conv"}:
            raise ValueError("downsampling must be one of: maxpool, avgpool, stride_conv")
        if (
            type(self.ode_entry_size) is not int
            or self.ode_entry_size not in SUPPORTED_ODE_ENTRY_SIZES
        ):
            supported = ", ".join(str(size) for size in sorted(SUPPORTED_ODE_ENTRY_SIZES))
            raise ValueError(f"ode_entry_size must be one of: {supported}")
