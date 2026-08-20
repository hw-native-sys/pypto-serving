# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .store import MonitorStore


def aggregate_snapshot(snapshot: dict) -> dict:
    """Aggregate per-replica metrics without losing cumulative histograms."""
    counters: dict[str, int] = {}
    histograms: dict[str, dict] = {}
    running = 0
    waiting = 0
    kv_cache_usage = 0.0

    for replica in snapshot.get("replicas", []):
        gauges = replica.get("gauges", {})
        running += int(gauges.get("running", 0))
        waiting += int(gauges.get("waiting", 0))
        kv_cache_usage = max(kv_cache_usage, float(gauges.get("kv_cache_usage", 0.0)))
        for key, value in replica.get("counters", {}).items():
            counters[key] = counters.get(key, 0) + int(value)
        for key, histogram in replica.get("histograms", {}).items():
            target = histograms.setdefault(
                key,
                {
                    "buckets": [
                        {"le": float(bucket["le"]), "count": 0}
                        for bucket in histogram.get("buckets", [])
                    ],
                    "count": 0,
                    "sum": 0.0,
                },
            )
            for index, bucket in enumerate(histogram.get("buckets", [])):
                target["buckets"][index]["count"] += int(bucket["count"])
            target["count"] += int(histogram.get("count", 0))
            target["sum"] += float(histogram.get("sum", 0.0))

    return {
        "server_id": str(snapshot.get("server_id", "")),
        "timestamp": float(snapshot.get("timestamp", time.time())),
        "model_name": str(snapshot.get("model_name", "unknown")),
        "gauges": {
            "running": running,
            "waiting": waiting,
            "kv_cache_usage": kv_cache_usage,
        },
        "counters": counters,
        "histograms": histograms,
    }


def snapshot_delta(current: dict, previous: dict | None) -> dict:
    """Convert cumulative serving metrics into one collector interval."""
    same_server = previous is not None and current["server_id"] == previous["server_id"]
    counters = {}
    for key, value in current["counters"].items():
        old = previous["counters"].get(key, 0) if same_server else value
        counters[key] = max(0, int(value) - int(old))

    histograms = {}
    for key, histogram in current["histograms"].items():
        old_histogram = previous["histograms"].get(key, {}) if same_server else {}
        old_buckets = old_histogram.get("buckets", [])
        buckets = []
        for index, bucket in enumerate(histogram.get("buckets", [])):
            old_count = old_buckets[index]["count"] if index < len(old_buckets) else bucket["count"]
            buckets.append({
                "le": bucket["le"],
                "count": max(0, int(bucket["count"]) - int(old_count)),
            })
        old_count = old_histogram.get("count", histogram.get("count", 0))
        old_sum = old_histogram.get("sum", histogram.get("sum", 0.0))
        histograms[key] = {
            "buckets": buckets,
            "count": max(0, int(histogram.get("count", 0)) - int(old_count)),
            "sum": max(0.0, float(histogram.get("sum", 0.0)) - float(old_sum)),
        }

    elapsed = (
        max(0.001, current["timestamp"] - previous["timestamp"])
        if same_server
        else 1.0
    )
    return {
        **current,
        "counter_deltas": counters,
        "histogram_deltas": histograms,
        "elapsed": elapsed,
    }


@dataclass
class CollectorStatus:
    connected: bool = False
    last_collected_at: float | None = None
    last_error: str = "Waiting for the first metrics sample"
    model_name: str = "unknown"


class MetricsCollector:
    def __init__(
        self,
        target: str,
        store: MonitorStore,
        *,
        interval: float = 1.0,
        timeout: float = 2.0,
    ) -> None:
        self.target = target.rstrip("/")
        self.metrics_url = self.target + "/metrics/json"
        self.store = store
        self.interval = max(0.25, float(interval))
        self.timeout = max(0.25, float(timeout))
        self.status = CollectorStatus()
        self._previous: dict | None = None
        self._stop_event = asyncio.Event()

    def fetch(self) -> dict:
        request = urllib.request.Request(
            self.metrics_url,
            headers={"Accept": "application/json", "User-Agent": "pypto-monitor/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    async def collect_once(self) -> None:
        try:
            snapshot = await asyncio.to_thread(self.fetch)
            current = aggregate_snapshot(snapshot)
            current["timestamp"] = time.time()
            interval = snapshot_delta(current, self._previous)
            await asyncio.to_thread(self.store.record, interval)
            self._previous = current
            self.status.connected = True
            self.status.last_collected_at = current["timestamp"]
            self.status.last_error = ""
            self.status.model_name = current["model_name"]
        except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
            self.status.connected = False
            self.status.last_error = str(exc)

    async def run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            await self.collect_once()
            delay = max(0.0, self.interval - (time.monotonic() - started))
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop_event.set()
