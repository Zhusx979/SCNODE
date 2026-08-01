"""Numerical diagnostics shared by review experiments.

The values in this module diagnose a trained model's numerical behaviour; they
are not biological measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def endpoint_relative_error(candidate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> float:
    """Return a scale-normalised endpoint discrepancy."""
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have identical shapes")
    numerator = torch.linalg.vector_norm((candidate - reference).reshape(-1))
    denominator = torch.linalg.vector_norm(reference.reshape(-1)).clamp_min(eps)
    return float((numerator / denominator).detach().cpu())


@dataclass
class OdeMetricCollector:
    """Collect conservative maxima without retaining GPU activation graphs."""

    max_state_l2: float = 0.0
    max_gradient_l2: float = 0.0
    nonfinite_batch_count: int = 0

    def reset(self) -> None:
        self.max_state_l2 = 0.0
        self.max_gradient_l2 = 0.0
        self.nonfinite_batch_count = 0

    def record_state(self, state: torch.Tensor) -> None:
        detached = state.detach()
        if not torch.isfinite(detached).all():
            self.nonfinite_batch_count += 1
            return
        self.max_state_l2 = max(
            self.max_state_l2,
            float(torch.linalg.vector_norm(detached.reshape(-1)).cpu()),
        )

    def record_gradient(self, gradient: torch.Tensor) -> None:
        detached = gradient.detach()
        if not torch.isfinite(detached).all():
            self.nonfinite_batch_count += 1
            return
        self.max_gradient_l2 = max(
            self.max_gradient_l2,
            float(torch.linalg.vector_norm(detached.reshape(-1)).cpu()),
        )

    def snapshot(self) -> dict[str, float | int]:
        return {
            "max_state_l2": self.max_state_l2,
            "max_gradient_l2": self.max_gradient_l2,
            "nonfinite_batch_count": self.nonfinite_batch_count,
        }


def register_ode_state_hooks(model, collector: OdeMetricCollector):
    """Record ODE block endpoints without coupling diagnostics to one model class."""
    handles = []
    for module in model.modules():
        if hasattr(module, "odefunc"):
            def hook(_module, _inputs, output, target=collector):
                if isinstance(output, torch.Tensor):
                    target.record_state(output[-1] if output.ndim == 5 else output)
            handles.append(module.register_forward_hook(hook))
    return handles
