# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from pypto_serving.serving.pd.observability import PDMetrics


def test_unknown_router_token_digest_is_not_an_empty_sequence_claim():
    metrics = PDMetrics("router", "node")
    metrics.record_terminal(request_id="unknown", state="COMPLETED")
    metrics.record_terminal(request_id="known", state="COMPLETED", token_ids=(1, 2))
    recent = metrics.snapshot()["recent_terminal"]
    assert recent[0]["token_ids_sha256"] == ""
    assert len(recent[1]["token_ids_sha256"]) == 64


def test_peak_gauge_is_monotonic_and_current_gauge_can_return_to_zero() -> None:
    metrics = PDMetrics("test", "node")
    metrics.set_peak_gauge("active_peak", 2)
    metrics.set_peak_gauge("active_peak", 1)
    metrics.set_peak_gauge("active_peak", 4)
    metrics.set_gauge("active", 4)
    metrics.set_gauge("active", 0)

    gauges = metrics.snapshot()["gauges"]
    assert gauges == {"active": 0, "active_peak": 4}
