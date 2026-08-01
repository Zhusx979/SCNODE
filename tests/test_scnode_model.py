import pytest
import torch
import torch.nn as nn

pytest.importorskip("torchdiffeq")

import SCNODE.models.ode.scnode.scnode_resnet as scnode_resnet
from SCNODE.models.ode.scnode.config import ScnodeConfig
from SCNODE.models.ode.scnode.scnode_resnet import (
    BasicBlock,
    BasicBlock2,
    FourierFiLMConv2d,
    Get_time_AnodeV2_ResNet18,
    ODEBlock,
    ResNet,
)


@pytest.mark.parametrize("time_mode", ["none", "concat", "fourier_film"])
def test_all_time_modes_use_group_norm(time_mode: str) -> None:
    block = BasicBlock2(dim=64, num_filters=65, augment_dim=1, time_mode=time_mode)

    assert type(block.bn1) is nn.GroupNorm
    assert type(block.bn2) is nn.GroupNorm


@pytest.mark.parametrize(
    ("ode_entry_size", "expected_shape", "downsampling_applied"),
    [
        (112, (112, 112), False),
        (56, (56, 56), True),
        (28, (28, 28), True),
        (14, (14, 14), True),
    ],
)
def test_ode_entry_size_controls_first_ode_input_shape(
    ode_entry_size: int,
    expected_shape: tuple[int, int],
    downsampling_applied: bool,
) -> None:
    model = Get_time_AnodeV2_ResNet18(
        21,
        config=ScnodeConfig(
            solver="euler",
            ode_steps=1,
            ode_entry_size=ode_entry_size,
        ),
    ).eval()

    with torch.no_grad():
        first_ode_input = model.forward_to_first_ode_input(torch.randn(1, 3, 224, 224))

    assert first_ode_input.shape[-2:] == expected_shape
    assert model.first_ode_input_shape == expected_shape
    assert model.ode_entry_metadata["downsampling_applied"] is downsampling_applied


def test_ode_entry_size_does_not_upscale_smaller_stem_features() -> None:
    model = Get_time_AnodeV2_ResNet18(
        21,
        config=ScnodeConfig(solver="euler", ode_steps=1, ode_entry_size=56),
    ).eval()

    with torch.no_grad():
        first_ode_input = model.forward_to_first_ode_input(torch.randn(1, 3, 64, 64))

    assert first_ode_input.shape[-2:] == (32, 32)
    assert model.ode_entry_metadata["downsampling_applied"] is False
    assert model.ode_entry_metadata["resize_applied"] is False


@pytest.mark.parametrize("downsampling", ["maxpool", "avgpool", "stride_conv"])
def test_downsamplers_reach_default_ode_entry_size(downsampling: str) -> None:
    model = Get_time_AnodeV2_ResNet18(
        21,
        config=ScnodeConfig(solver="euler", ode_steps=1, downsampling=downsampling),
    ).eval()

    with torch.no_grad():
        first_ode_input = model.forward_to_first_ode_input(torch.randn(1, 3, 224, 224))

    assert first_ode_input.shape == (1, 64, 56, 56)
    assert model.ode_entry_metadata["downsampling_applied"] is True


@pytest.mark.parametrize("time_mode", ["none", "concat", "fourier_film"])
def test_scnode_full_forward_supports_time_modes_with_default_augmentation(
    time_mode: str,
) -> None:
    model = Get_time_AnodeV2_ResNet18(
        21,
        config=ScnodeConfig(solver="euler", ode_steps=1, time_mode=time_mode),
    ).eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))

    assert logits.shape == (1, 21)
    assert model.augment_dim == 1
    assert model.ode_entry_metadata["downsampling_applied"] is True


class ConstantODEFunc(nn.Module):
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


def test_ode_block_can_zero_only_auxiliary_channels_after_integration() -> None:
    block = ODEBlock(
        ConstantODEFunc(),
        config=ScnodeConfig(solver="euler", ode_steps=1, augment_dim=2),
    )
    inputs = torch.randn(1, 3, 4, 4)

    with torch.no_grad():
        regular = block(inputs)
        zeroed = block(inputs, zero_auxiliary=True)

    assert regular.shape == zeroed.shape == (1, 5, 4, 4)
    assert torch.allclose(regular[:, :3], zeroed[:, :3])
    assert torch.count_nonzero(regular[:, 3:]) > 0
    assert torch.count_nonzero(zeroed[:, 3:]) == 0


def test_ode_block_returns_requested_internal_solver_states() -> None:
    block = ODEBlock(
        ConstantODEFunc(),
        config=ScnodeConfig(solver="euler", ode_steps=2, augment_dim=1),
    )

    with torch.no_grad():
        states = block(torch.zeros(1, 3, 2, 2), return_states=True)

    assert states.shape == (3, 1, 4, 2, 2)
    assert torch.allclose(states[0, :, :3], torch.zeros(1, 3, 2, 2))


def test_resnet_forward_can_zero_auxiliary_channels_without_changing_logits_shape() -> None:
    model = Get_time_AnodeV2_ResNet18(
        21,
        config=ScnodeConfig(solver="euler", ode_steps=1, augment_dim=2),
    ).eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64), zero_auxiliary=True)

    assert logits.shape == (1, 21)


def test_fourier_film_uses_fourier_features_and_two_layer_mlp() -> None:
    layer = FourierFiLMConv2d(3, 5, kernel_size=3, padding=1)
    observed_features: list[torch.Tensor] = []

    handle = layer.time_affine[0].register_forward_pre_hook(
        lambda _, inputs: observed_features.append(inputs[0].detach().clone())
    )

    assert isinstance(layer.time_affine[0], nn.Linear)
    assert isinstance(layer.time_affine[1], nn.SiLU)
    assert isinstance(layer.time_affine[2], nn.Linear)
    assert layer.time_affine[0].in_features == 2
    assert layer.time_affine[-1].out_features == 10

    with torch.no_grad():
        output = layer(torch.tensor(0.25), torch.randn(2, 3, 8, 8))

    handle.remove()

    assert output.shape == (2, 5, 8, 8)
    assert torch.allclose(observed_features[0], torch.tensor([[1.0, 0.0]]), atol=1e-6)


class LegacyODEBlock(nn.Module):
    def __init__(self, block: nn.Module, augment_dim: int = 0) -> None:
        super().__init__()
        self.block = block
        self.augment_dim = augment_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.augment_dim:
            augment = x.new_zeros(x.shape[0], self.augment_dim, *x.shape[-2:])
            x = torch.cat((x, augment), dim=1)
        return self.block(torch.zeros((), device=x.device, dtype=x.dtype), x)


def test_custom_ode_block_keeps_legacy_constructor_signature() -> None:
    model = ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=21,
        ODEBlock_=LegacyODEBlock,
        config=ScnodeConfig(augment_dim=1),
    )

    assert isinstance(model.layer1[1], LegacyODEBlock)
    assert model.layer1[1].augment_dim == 1

    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))

    assert logits.shape == (1, 21)


def test_resnet_routes_auxiliary_zeroing_to_only_the_terminal_builtin_ode_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Get_time_AnodeV2_ResNet18(
        21,
        config=ScnodeConfig(solver="euler", ode_steps=1, augment_dim=1),
    ).eval()
    calls: list[bool] = []
    original_forward = ODEBlock.forward

    def recording_forward(
        self: ODEBlock,
        x: torch.Tensor,
        eval_times: torch.Tensor | None = None,
        zero_auxiliary: bool = False,
    ) -> torch.Tensor:
        calls.append(zero_auxiliary)
        return original_forward(self, x, eval_times, zero_auxiliary)

    monkeypatch.setattr(ODEBlock, "forward", recording_forward)

    with torch.no_grad():
        model(torch.randn(1, 3, 64, 64), zero_auxiliary=True)

    assert calls == [False, False, False, True]


class LegacyODEBlock(ODEBlock):
    """Legacy custom block whose forward method has no auxiliary intervention API."""

    def __init__(self, odefunc: nn.Module, augment_dim: int) -> None:
        nn.Module.__init__(self)
        del odefunc
        self.augment_dim = augment_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        auxiliary = x.new_zeros(x.shape[0], self.augment_dim, *x.shape[-2:])
        return torch.cat((x, auxiliary), dim=1)


def test_resnet_does_not_pass_auxiliary_intervention_to_legacy_ode_blocks() -> None:
    model = ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=21,
        ODEBlock_=LegacyODEBlock,
        config=ScnodeConfig(solver="euler", ode_steps=1, augment_dim=1),
    ).eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64), zero_auxiliary=True)

    assert logits.shape == (1, 21)


@pytest.mark.parametrize("solver", ["euler", "rk4"])
def test_fixed_solver_passes_time_grid_tolerances_and_step_size_to_odeint(
    monkeypatch: pytest.MonkeyPatch, solver: str
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_odeint(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        state = args[1]
        assert isinstance(state, torch.Tensor)
        return torch.stack((state, state))

    monkeypatch.setattr(scnode_resnet, "odeint", fake_odeint)
    ode_block = ODEBlock(
        ConstantODEFunc(),
        config=ScnodeConfig(
            solver=solver, ode_steps=4, rtol=1e-5, atol=1e-6, augment_dim=0
        ),
    )

    with torch.no_grad():
        ode_block(torch.randn(1, 3, 4, 4))

    args, kwargs = calls.pop()
    assert torch.equal(args[2], torch.linspace(0.0, 1.0, 5))
    assert kwargs == {
        "method": solver,
        "rtol": 1e-5,
        "atol": 1e-6,
        "options": {"step_size": 0.25},
    }


def test_dopri5_passes_endpoint_grid_without_fixed_step_options_to_odeint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_odeint(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append((args, kwargs))
        state = args[1]
        assert isinstance(state, torch.Tensor)
        return torch.stack((state, state))

    monkeypatch.setattr(scnode_resnet, "odeint", fake_odeint)
    ode_block = ODEBlock(
        ConstantODEFunc(),
        config=ScnodeConfig(
            solver="dopri5", ode_steps=4, rtol=1e-5, atol=1e-6, augment_dim=0
        ),
    )

    with torch.no_grad():
        ode_block(torch.randn(1, 3, 4, 4))

    args, kwargs = calls.pop()
    assert torch.equal(args[2], torch.tensor([0.0, 1.0]))
    assert kwargs == {"method": "dopri5", "rtol": 1e-5, "atol": 1e-6}
