from __future__ import annotations


def unwrap_parallel_model(model):
    return model.module if hasattr(model, "module") else model


def get_and_reset_ode_nfe(model) -> float:
    base_model = unwrap_parallel_model(model)
    if hasattr(base_model, "odeblock"):
        value = float(base_model.odeblock.odefunc.nfe)
        base_model.odeblock.odefunc.nfe = 0
        return value
    if hasattr(base_model, "nfe"):
        value = float(base_model.nfe)
        base_model.nfe = 0
        return value
    raise AttributeError("Model does not expose 'odeblock.odefunc.nfe' or 'nfe'.")
