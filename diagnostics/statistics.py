"""Configuration and utilities for paired significance tests with BH-FDR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import ttest_rel


@dataclass(frozen=True)
class SignificanceConfig:
    alpha: float = 0.05
    alternative: str = "two-sided"
    correction: str = "bh"

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.alternative not in {"two-sided", "less", "greater"}:
            raise ValueError("alternative must be two-sided, less, or greater")
        if self.correction != "bh":
            raise ValueError("only Benjamini-Hochberg correction is supported")


def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite probabilities")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q_values = np.empty_like(adjusted)
    q_values[order] = adjusted
    return q_values, q_values <= alpha


def paired_ttest_bh(
    comparisons: Iterable[tuple[str, Iterable[float], Iterable[float]]],
    config: SignificanceConfig = SignificanceConfig(),
) -> list[dict[str, object]]:
    rows = []
    for name, reference, candidate in comparisons:
        reference = np.asarray(list(reference), dtype=float)
        candidate = np.asarray(list(candidate), dtype=float)
        if reference.shape != candidate.shape or reference.ndim != 1 or len(reference) < 2:
            raise ValueError("paired samples must be one-dimensional and have at least two values")
        statistic, p_value = ttest_rel(reference, candidate, alternative=config.alternative)
        rows.append({"comparison": name, "statistic": float(statistic), "p_value": float(p_value), "n": int(len(reference))})
    q_values, reject = benjamini_hochberg([row["p_value"] for row in rows], config.alpha)
    for row, q_value, is_rejected in zip(rows, q_values, reject):
        row["q_value_bh"] = float(q_value)
        row["reject_bh"] = bool(is_rejected)
        row["alpha"] = config.alpha
    return rows

