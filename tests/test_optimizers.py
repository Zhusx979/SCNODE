import pytest
import torch

from SCNODE.training.optimizers import build_optimizer, build_scheduler


@pytest.mark.parametrize(
    "name,expected",
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("sgd", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
        ("adagrad", torch.optim.Adagrad),
    ],
)
def test_build_optimizer_supports_requested_methods(name, expected):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = build_optimizer([parameter], name=name, learning_rate=1e-2)
    assert isinstance(optimizer, expected)


def test_sgd_uses_momentum_and_cosine_scheduler():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = build_optimizer([parameter], name="sgd", learning_rate=0.1, momentum=0.9)
    assert optimizer.param_groups[0]["momentum"] == 0.9
    scheduler = build_scheduler(optimizer, name="cosine", num_epochs=4)
    initial_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < initial_lr


def test_invalid_optimizer_and_scheduler_are_rejected():
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError):
        build_optimizer([parameter], name="unknown", learning_rate=1e-3)
    optimizer = build_optimizer([parameter], name="adam", learning_rate=1e-3)
    with pytest.raises(ValueError):
        build_scheduler(optimizer, name="unknown", num_epochs=2)
