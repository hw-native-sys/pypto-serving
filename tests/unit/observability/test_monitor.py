# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import time

from tools.monitor.collector import aggregate_snapshot, snapshot_delta
from tools.monitor.store import MonitorStore


def _snapshot(timestamp, *, generation_tokens, requests, ttft_count):
    return {
        "server_id": "server-1",
        "timestamp": timestamp,
        "model_name": "test-model",
        "replicas": [{
            "engine": 0,
            "gauges": {"running": 2, "waiting": 1, "kv_cache_usage": 0.5},
            "counters": {
                "prompt_tokens": generation_tokens * 2,
                "prefill_tokens": generation_tokens * 2,
                "generation_tokens": generation_tokens,
                "prefix_cache_queries": generation_tokens,
                "prefix_cache_hits": generation_tokens // 2,
                "preemptions": 0,
                "requests_finished": requests,
                "requests_error": 0,
                "requests_aborted": 0,
            },
            "histograms": {
                "ttft": {
                    "buckets": [{"le": 0.1, "count": ttft_count}, {"le": 1.0, "count": ttft_count}],
                    "count": ttft_count,
                    "sum": ttft_count * 0.1,
                },
                "tpot": {
                    "buckets": [{"le": 0.05, "count": ttft_count}],
                    "count": ttft_count,
                    "sum": ttft_count * 0.05,
                },
            },
        }],
    }


def test_collector_delta_and_sqlite_rollup(tmp_path):
    now = time.time()
    first = aggregate_snapshot(_snapshot(now - 2, generation_tokens=10, requests=1, ttft_count=1))
    second = aggregate_snapshot(_snapshot(now - 1, generation_tokens=14, requests=3, ttft_count=3))
    baseline = snapshot_delta(first, None)
    interval = snapshot_delta(second, first)

    assert baseline["counter_deltas"]["generation_tokens"] == 0
    assert interval["counter_deltas"]["generation_tokens"] == 4
    assert interval["counter_deltas"]["requests_finished"] == 2
    assert interval["histogram_deltas"]["ttft"]["count"] == 2

    store = MonitorStore(tmp_path / "monitor.sqlite3", timezone_name="UTC")
    store.record(baseline)
    store.record(interval)
    summary = store.summary(300)
    history = store.history(3600)
    daily = store.daily(1)
    store.close()

    assert summary["model_name"] == "test-model"
    assert summary["current"]["running"] == 2
    assert summary["current"]["ttft_p50_seconds"] == 0.1
    assert summary["today"]["generation_tokens"] == 4
    assert history
    assert daily[0]["request_count"] == 2
