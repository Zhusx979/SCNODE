import torch.nn as nn

from SCNODE.training.ode_runtime import get_and_reset_ode_nfe


class _FakeOdeFunc:
    def __init__(self, nfe: float) -> None:
        self.nfe = nfe


class _FakeOdeBlock:
    def __init__(self, nfe: float) -> None:
        self.odefunc = _FakeOdeFunc(nfe)


class _WrappedWithOdeBlock(nn.Module):
    def __init__(self, nfe: float) -> None:
        super().__init__()
        self.odeblock = _FakeOdeBlock(nfe)


class _WrappedWithModelNfe(nn.Module):
    def __init__(self, nfe: float) -> None:
        super().__init__()
        self.nfe = nfe


class _DataParallelLike:
    def __init__(self, module) -> None:
        self.module = module


def test_get_and_reset_ode_nfe_supports_dataparallel_wrapped_model_attribute() -> None:
    wrapped = _DataParallelLike(_WrappedWithModelNfe(17))

    value = get_and_reset_ode_nfe(wrapped)

    assert value == 17
    assert wrapped.module.nfe == 0


def test_get_and_reset_ode_nfe_supports_dataparallel_wrapped_odeblock() -> None:
    wrapped = _DataParallelLike(_WrappedWithOdeBlock(23))

    value = get_and_reset_ode_nfe(wrapped)

    assert value == 23
    assert wrapped.module.odeblock.odefunc.nfe == 0
