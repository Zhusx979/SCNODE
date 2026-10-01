from __future__ import annotations


def unwrap_parallel_model(model):
    return model.module if hasattr(model, "module") else model


def get_and_reset_ode_nfe(model) -> float:
    base_model = unwrap_parallel_model(model)
    ode_functions = []
    seen = set()

    direct_odefunc = getattr(getattr(base_model, "odeblock", None), "odefunc", None)
    if direct_odefunc is not None and hasattr(direct_odefunc, "nfe"):
        seen.add(id(direct_odefunc))
        ode_functions.append(direct_odefunc)

    for module in base_model.modules():
        odefunc = getattr(module, "odefunc", None)
        if odefunc is not None and hasattr(odefunc, "nfe") and id(odefunc) not in seen:
            seen.add(id(odefunc))
            ode_functions.append(odefunc)

    if ode_functions:
        value = sum(float(odefunc.nfe) for odefunc in ode_functions)
        for odefunc in ode_functions:
            odefunc.nfe = 0
        return value
    if hasattr(base_model, "nfe"):
        value = float(base_model.nfe)
        base_model.nfe = 0
        return value
    raise AttributeError("Model does not expose an ODE function 'nfe' counter or model-level 'nfe'.")
