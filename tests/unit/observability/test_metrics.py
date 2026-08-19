# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pypto_serving.observability import InMemoryStatLogger, SchedulerStats


def test_request_metrics_are_cumulative_and_terminal_is_idempotent():
    metrics = InMemoryStatLogger("test-model", [0])
    metrics.start_request(0, "request-1", arrival_monotonic=10.0, num_prompt_tokens=5)
    metrics.record_output(0, "request-1", completion_tokens=1, timestamp=11.0)
    metrics.record_output(0, "request-1", completion_tokens=3, timestamp=13.0)
    metrics.finish_request(0, "request-1", "FINISHED_LENGTH", timestamp=14.0)
    metrics.finish_request(0, "request-1", "FINISHED_LENGTH", timestamp=15.0)

    replica = metrics.snapshot()["replicas"][0]
    assert replica["counters"]["prompt_tokens"] == 5
    assert replica["counters"]["generation_tokens"] == 3
    assert replica["counters"]["requests_finished"] == 1
    assert replica["histograms"]["ttft"]["count"] == 1
    assert replica["histograms"]["tpot"]["count"] == 1
    assert replica["histograms"]["e2e"]["sum"] == 4.0


def test_scheduler_metrics_and_prometheus_exposition():
    metrics = InMemoryStatLogger("model-with-quotes", [0])
    metrics.record_scheduler(
        0,
        SchedulerStats(
            num_running_reqs=3,
            num_waiting_reqs=2,
            kv_cache_usage=0.75,
            prefix_cache_queries=128,
            prefix_cache_hits=96,
            num_preemptions=1,
        ),
    )

    text = metrics.render_prometheus()
    assert '# TYPE pypto:num_requests_running gauge' in text
    assert 'pypto:num_requests_running{model_name="model-with-quotes",engine="0"} 3' in text
    assert 'pypto:prefix_cache_queries_total{model_name="model-with-quotes",engine="0"} 128' in text
    assert 'pypto:time_to_first_token_seconds_bucket' in text
