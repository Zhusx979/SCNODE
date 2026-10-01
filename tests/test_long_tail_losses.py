import pytest
import torch

from SCNODE.training.long_tail_losses import (
    BalancedSoftmaxLoss,
    ClassBalancedFocalLoss,
    EQLv2Loss,
    LDAMLoss,
    LogitAdjustmentLoss,
    SeesawLoss,
    build_long_tail_loss,
)


@pytest.mark.parametrize("name", ["cross_entropy", "ldam_drw", "logit_adjustment", "balanced_softmax", "class_balanced_focal", "seesaw", "eql_v2"])
def test_long_tail_factory_returns_finite_scalar(name: str) -> None:
    loss = build_long_tail_loss(name, [100, 20, 5], num_classes=3)
    value = loss(torch.randn(4, 3, requires_grad=True), torch.tensor([0, 1, 2, 1]))
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_ldam_deferred_reweighting_changes_after_schedule() -> None:
    loss = LDAMLoss([100, 10], drw_start=1)
    logits = torch.zeros(2, 2)
    target = torch.tensor([0, 1])
    before = loss(logits, target)
    loss.set_epoch(1)
    after = loss(logits, target)
    assert before != after


def test_loss_classes_are_exposed() -> None:
    assert all(cls is not None for cls in (BalancedSoftmaxLoss, ClassBalancedFocalLoss, EQLv2Loss, LogitAdjustmentLoss, SeesawLoss))
