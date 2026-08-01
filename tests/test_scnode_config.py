from dataclasses import FrozenInstanceError

import pytest

from SCNODE.models.ode.scnode.config import ScnodeConfig


def test_scnode_config_defaults_are_immutable() -> None:
    config = ScnodeConfig()

    assert config.solver == "rk4"
    assert config.ode_steps == 4
    assert config.rtol == pytest.approx(1e-3)
    assert config.atol == pytest.approx(1e-3)
    assert config.time_mode == "concat"
    assert config.augment_dim == 1
    assert config.downsampling == "maxpool"
    assert config.ode_entry_size == 56

    with pytest.raises(FrozenInstanceError):
        config.ode_steps = 8


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"solver": "adams"}, "solver"),
        ({"solver": ["rk4"]}, "solver"),
        ({"solver": {"name": "rk4"}}, "solver"),
        ({"ode_steps": 0}, "ode_steps"),
        ({"ode_steps": True}, "ode_steps"),
        ({"ode_steps": 1.5}, "ode_steps"),
        ({"rtol": 0}, "rtol"),
        ({"rtol": -1e-3}, "rtol"),
        ({"rtol": float("inf")}, "rtol"),
        ({"rtol": float("nan")}, "rtol"),
        ({"atol": 0}, "atol"),
        ({"atol": -1e-3}, "atol"),
        ({"atol": float("inf")}, "atol"),
        ({"atol": float("nan")}, "atol"),
        ({"time_mode": "learned"}, "time_mode"),
        ({"time_mode": ["concat"]}, "time_mode"),
        ({"time_mode": {"name": "concat"}}, "time_mode"),
        ({"downsampling": "bilinear"}, "downsampling"),
        ({"downsampling": ["maxpool"]}, "downsampling"),
        ({"downsampling": {"name": "maxpool"}}, "downsampling"),
        ({"augment_dim": -1}, "augment_dim"),
        ({"augment_dim": True}, "augment_dim"),
        ({"augment_dim": 0.5}, "augment_dim"),
        ({"ode_entry_size": 48}, "ode_entry_size"),
        ({"ode_entry_size": True}, "ode_entry_size"),
        ({"ode_entry_size": 56.0}, "ode_entry_size"),
    ],
)
def test_scnode_config_rejects_invalid_values(kwargs: dict[str, object], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        ScnodeConfig(**kwargs)
