import pytest
import torch

from SCNODE.diagnostics.ode_metrics import OdeMetricCollector, endpoint_relative_error, register_ode_state_hooks


def test_endpoint_relative_error_is_zero_for_identical_tensors() -> None:
    state = torch.tensor([[1.0, 2.0]])

    assert endpoint_relative_error(state, state) == pytest.approx(0.0)


def test_collector_records_norms_and_nonfinite_states() -> None:
    collector = OdeMetricCollector()
    collector.record_state(torch.tensor([3.0, 4.0]))
    collector.record_state(torch.tensor([float("nan")]))
    collector.record_gradient(torch.tensor([5.0, 12.0]))

    metrics = collector.snapshot()

    assert metrics["max_state_l2"] == pytest.approx(5.0)
    assert metrics["max_gradient_l2"] == pytest.approx(13.0)
    assert metrics["nonfinite_batch_count"] == 1


def test_registered_hooks_record_ode_block_endpoint() -> None:
    class OdeLike(torch.nn.Module):
        odefunc = object()

        def forward(self, x):
            return x * 2

    collector = OdeMetricCollector()
    model = OdeLike()
    handles = register_ode_state_hooks(model, collector)
    model(torch.tensor([3.0, 4.0]))
    for handle in handles:
        handle.remove()

    assert collector.snapshot()["max_state_l2"] == pytest.approx(10.0)
