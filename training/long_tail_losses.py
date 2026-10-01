"""Long-tailed single-label classification losses used by the training entry points."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CountInput = Union[Iterable[int], Tensor]


def _counts(values: CountInput, num_classes: int) -> Tensor:
    counts = torch.as_tensor(values, dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() != num_classes or torch.any(counts <= 0):
        raise ValueError("class_counts must contain one positive count per class")
    return counts


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss with deferred re-weighting."""

    def __init__(self, class_counts: CountInput, max_margin: float = 0.5,
                 drw_start: Optional[int] = None, drw_beta: float = 0.9999):
        super().__init__()
        counts = _counts(class_counts, len(class_counts) if not isinstance(class_counts, Tensor) else class_counts.numel())
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        self.register_buffer("margins", margins * (max_margin / margins.max()))
        self.register_buffer("class_counts", counts)
        self.drw_start = drw_start
        self.drw_beta = drw_beta
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        adjusted = logits.clone()
        adjusted[torch.arange(target.numel(), device=target.device), target] -= self.margins[target]
        weight = None
        if self.drw_start is not None and self.epoch >= self.drw_start:
            effective = 1.0 - torch.pow(torch.tensor(self.drw_beta, device=logits.device), self.class_counts)
            weight = (1.0 - self.drw_beta) / effective
            weight = weight / weight.mean()
        return F.cross_entropy(adjusted, target, weight=weight)


class LogitAdjustmentLoss(nn.Module):
    """Cross entropy with log-prior adjustment (Menon et al.)."""

    def __init__(self, class_counts: CountInput, tau: float = 1.0):
        super().__init__()
        counts = _counts(class_counts, len(class_counts) if not isinstance(class_counts, Tensor) else class_counts.numel())
        self.register_buffer("log_prior", torch.log(counts / counts.sum()) * tau)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits + self.log_prior.to(logits), target)


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax cross entropy using training class frequencies."""

    def __init__(self, class_counts: CountInput):
        super().__init__()
        counts = _counts(class_counts, len(class_counts) if not isinstance(class_counts, Tensor) else class_counts.numel())
        self.register_buffer("log_counts", torch.log(counts))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits + self.log_counts.to(logits), target)


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss using the effective-number weighting."""

    def __init__(self, class_counts: CountInput, beta: float = 0.9999, gamma: float = 2.0):
        super().__init__()
        counts = _counts(class_counts, len(class_counts) if not isinstance(class_counts, Tensor) else class_counts.numel())
        effective = 1.0 - torch.pow(torch.tensor(beta), counts)
        weights = (1.0 - beta) / effective
        self.register_buffer("weights", weights / weights.mean())
        self.gamma = gamma

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        focal = (1.0 - prob).pow(self.gamma)
        return F.nll_loss(log_prob * focal, target, weight=self.weights.to(logits))


class SeesawLoss(nn.Module):
    """Single-label Seesaw loss adapted from the official MMDetection formulation."""

    def __init__(self, class_counts: CountInput, p: float = 0.8, q: float = 2.0):
        super().__init__()
        counts = _counts(class_counts, len(class_counts) if not isinstance(class_counts, Tensor) else class_counts.numel())
        self.register_buffer("class_counts", counts)
        self.p, self.q = p, q

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        batch = target.numel()
        adjusted = logits.clone()
        for row, cls in enumerate(target):
            compensation = (self.class_counts[cls] / self.class_counts.clamp_min(1)).pow(self.p)
            compensation = compensation.clamp_max(1.0)
            adjusted[row] = adjusted[row] + self.q * compensation.log()
            adjusted[row, cls] = logits[row, cls]
        return F.cross_entropy(adjusted, target)


class EQLv2Loss(nn.Module):
    """EQL-v2-style dynamic negative-gradient suppression for single-label data."""

    def __init__(self, num_classes: int, gamma: float = 12.0, momentum: float = 0.9):
        super().__init__()
        self.gamma, self.momentum = gamma, momentum
        self.register_buffer("positive", torch.zeros(num_classes))
        self.register_buffer("negative", torch.zeros(num_classes))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = logits.sigmoid()
        one_hot = F.one_hot(target, logits.shape[1]).to(logits)
        pos = (probs * one_hot).sum(0)
        neg = ((1 - probs) * (1 - one_hot)).sum(0)
        self.positive.mul_(self.momentum).add_(pos.detach(), alpha=1 - self.momentum)
        self.negative.mul_(self.momentum).add_(neg.detach(), alpha=1 - self.momentum)
        ratio = self.positive / (self.negative + 1e-6)
        gate = torch.sigmoid(self.gamma * (ratio - 0.5)).detach()
        weight = one_hot + (1 - one_hot) * gate
        return F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)


def build_long_tail_loss(name: str, class_counts: CountInput, *, num_classes: Optional[int] = None) -> nn.Module:
    """Build a named long-tail loss; ``cross_entropy`` keeps the baseline behavior."""
    key = name.lower().replace("-", "_")
    counts = torch.as_tensor(class_counts)
    if key in {"cross_entropy", "ce", "none"}:
        return nn.CrossEntropyLoss()
    if key == "ldam_drw":
        return LDAMLoss(counts, drw_start=10)
    if key in {"logit_adjustment", "logit_adjust"}:
        return LogitAdjustmentLoss(counts)
    if key == "balanced_softmax":
        return BalancedSoftmaxLoss(counts)
    if key in {"class_balanced_focal", "cb_focal"}:
        return ClassBalancedFocalLoss(counts)
    if key == "seesaw":
        return SeesawLoss(counts)
    if key in {"eql_v2", "eqlv2"}:
        return EQLv2Loss(num_classes or counts.numel())
    raise ValueError("unknown long-tail loss: " + name)
