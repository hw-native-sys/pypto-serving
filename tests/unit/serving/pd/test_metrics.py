# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.

from pypto_serving.serving.pd.metrics import PDMetrics


def test_peak_gauge_is_monotonic_and_current_gauge_can_return_to_zero() -> None:
    metrics = PDMetrics("test", "node")
    metrics.set_peak_gauge("active_peak", 2)
    metrics.set_peak_gauge("active_peak", 1)
    metrics.set_peak_gauge("active_peak", 4)
    metrics.set_gauge("active", 4)
    metrics.set_gauge("active", 0)

    gauges = metrics.snapshot()["gauges"]
    assert gauges == {"active": 0, "active_peak": 4}
