# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def _percentile(histogram: dict, quantile: float) -> float | None:
    count = int(histogram.get("count", 0))
    if count <= 0:
        return None
    rank = max(1, math.ceil(count * quantile))
    for bucket in histogram.get("buckets", []):
        if int(bucket["count"]) >= rank:
            return float(bucket["le"])
    return None


def _merge_histograms(values: list[dict]) -> dict:
    merged: dict = {"buckets": [], "count": 0, "sum": 0.0}
    for histogram in values:
        if not merged["buckets"]:
            merged["buckets"] = [
                {"le": float(bucket["le"]), "count": 0}
                for bucket in histogram.get("buckets", [])
            ]
        for index, bucket in enumerate(histogram.get("buckets", [])):
            merged["buckets"][index]["count"] += int(bucket["count"])
        merged["count"] += int(histogram.get("count", 0))
        merged["sum"] += float(histogram.get("sum", 0.0))
    return merged


class MonitorStore:
    def __init__(
        self,
        path: Path | str,
        *,
        timezone_name: str = "local",
        retention_seconds: int = 86400,
    ) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_seconds = max(3600, int(retention_seconds))
        self.timezone = self._resolve_timezone(timezone_name)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    @staticmethod
    def _resolve_timezone(name: str):
        if name == "local":
            return None
        return ZoneInfo(name)

    def _datetime(self, timestamp: float | None = None) -> datetime:
        if timestamp is None:
            current = datetime.now(timezone.utc)
        else:
            current = datetime.fromtimestamp(timestamp, timezone.utc)
        return current.astimezone(self.timezone)

    def _initialize(self) -> None:
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    timestamp REAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    running INTEGER NOT NULL,
                    waiting INTEGER NOT NULL,
                    kv_cache_usage REAL NOT NULL,
                    request_delta INTEGER NOT NULL,
                    prompt_token_delta INTEGER NOT NULL,
                    prefill_token_delta INTEGER NOT NULL,
                    generation_token_delta INTEGER NOT NULL,
                    cache_query_delta INTEGER NOT NULL,
                    cache_hit_delta INTEGER NOT NULL,
                    preemption_delta INTEGER NOT NULL,
                    error_delta INTEGER NOT NULL,
                    aborted_delta INTEGER NOT NULL,
                    elapsed REAL NOT NULL,
                    ttft_histogram TEXT NOT NULL,
                    tpot_histogram TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_totals (
                    day TEXT PRIMARY KEY,
                    request_count INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    prefill_tokens INTEGER NOT NULL,
                    generation_tokens INTEGER NOT NULL,
                    errors INTEGER NOT NULL,
                    aborted INTEGER NOT NULL,
                    cache_queries INTEGER NOT NULL,
                    cache_hits INTEGER NOT NULL
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def record(self, sample: dict) -> None:
        timestamp = float(sample["timestamp"])
        gauges = sample["gauges"]
        counters = sample["counter_deltas"]
        histograms = sample["histogram_deltas"]
        day = self._datetime(timestamp).date().isoformat()
        values = {
            "request_delta": int(counters.get("requests_finished", 0)),
            "prompt_token_delta": int(counters.get("prompt_tokens", 0)),
            "prefill_token_delta": int(counters.get("prefill_tokens", 0)),
            "generation_token_delta": int(counters.get("generation_tokens", 0)),
            "cache_query_delta": int(counters.get("prefix_cache_queries", 0)),
            "cache_hit_delta": int(counters.get("prefix_cache_hits", 0)),
            "preemption_delta": int(counters.get("preemptions", 0)),
            "error_delta": int(counters.get("requests_error", 0)),
            "aborted_delta": int(counters.get("requests_aborted", 0)),
        }
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO samples VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    timestamp,
                    sample["model_name"],
                    int(gauges["running"]),
                    int(gauges["waiting"]),
                    float(gauges["kv_cache_usage"]),
                    values["request_delta"],
                    values["prompt_token_delta"],
                    values["prefill_token_delta"],
                    values["generation_token_delta"],
                    values["cache_query_delta"],
                    values["cache_hit_delta"],
                    values["preemption_delta"],
                    values["error_delta"],
                    values["aborted_delta"],
                    float(sample["elapsed"]),
                    json.dumps(histograms.get("ttft", {}), separators=(",", ":")),
                    json.dumps(histograms.get("tpot", {}), separators=(",", ":")),
                ),
            )
            if cursor.rowcount == 0:
                return
            self._connection.execute(
                """
                INSERT INTO daily_totals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    request_count = request_count + excluded.request_count,
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    prefill_tokens = prefill_tokens + excluded.prefill_tokens,
                    generation_tokens = generation_tokens + excluded.generation_tokens,
                    errors = errors + excluded.errors,
                    aborted = aborted + excluded.aborted,
                    cache_queries = cache_queries + excluded.cache_queries,
                    cache_hits = cache_hits + excluded.cache_hits
                """,
                (
                    day,
                    values["request_delta"],
                    values["prompt_token_delta"],
                    values["prefill_token_delta"],
                    values["generation_token_delta"],
                    values["error_delta"],
                    values["aborted_delta"],
                    values["cache_query_delta"],
                    values["cache_hit_delta"],
                ),
            )
            self._connection.execute(
                "DELETE FROM samples WHERE timestamp < ?",
                (timestamp - self.retention_seconds,),
            )

    def summary(self, window_seconds: int = 300) -> dict:
        since = time.time() - max(10, int(window_seconds))
        with self._lock:
            latest = self._connection.execute(
                "SELECT * FROM samples ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            rows = self._connection.execute(
                "SELECT * FROM samples WHERE timestamp >= ? ORDER BY timestamp",
                (since,),
            ).fetchall()
            today = self._datetime().date().isoformat()
            daily = self._connection.execute(
                "SELECT * FROM daily_totals WHERE day = ?",
                (today,),
            ).fetchone()

        if latest is None:
            return {"model_name": "unknown", "current": {}, "today": {}}
        elapsed = sum(float(row["elapsed"]) for row in rows) or 1.0
        generation_tokens = sum(int(row["generation_token_delta"]) for row in rows)
        prefill_tokens = sum(int(row["prefill_token_delta"]) for row in rows)
        queries = sum(int(row["cache_query_delta"]) for row in rows)
        hits = sum(int(row["cache_hit_delta"]) for row in rows)
        ttft = _merge_histograms([json.loads(row["ttft_histogram"]) for row in rows])
        tpot = _merge_histograms([json.loads(row["tpot_histogram"]) for row in rows])
        today_values = dict(daily) if daily is not None else {
            "day": today,
            "request_count": 0,
            "prompt_tokens": 0,
            "prefill_tokens": 0,
            "generation_tokens": 0,
            "errors": 0,
            "aborted": 0,
            "cache_queries": 0,
            "cache_hits": 0,
        }
        return {
            "model_name": latest["model_name"],
            "timestamp": latest["timestamp"],
            "window_seconds": window_seconds,
            "current": {
                "running": latest["running"],
                "waiting": latest["waiting"],
                "kv_cache_usage": latest["kv_cache_usage"],
                "generation_tokens_per_second": generation_tokens / elapsed,
                "prefill_tokens_per_second": prefill_tokens / elapsed,
                "cache_hit_rate": hits / queries if queries else None,
                "ttft_p50_seconds": _percentile(ttft, 0.50),
                "ttft_p90_seconds": _percentile(ttft, 0.90),
                "ttft_p99_seconds": _percentile(ttft, 0.99),
                "tpot_p50_seconds": _percentile(tpot, 0.50),
                "tpot_p90_seconds": _percentile(tpot, 0.90),
                "tpot_p99_seconds": _percentile(tpot, 0.99),
            },
            "today": today_values,
        }

    def history(self, range_seconds: int) -> list[dict]:
        range_seconds = min(self.retention_seconds, max(60, int(range_seconds)))
        step = max(1, math.ceil(range_seconds / 600))
        since = time.time() - range_seconds
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT
                    CAST(timestamp / ? AS INTEGER) * ? AS bucket,
                    SUM(generation_token_delta) /
                        CASE WHEN SUM(elapsed) > 0.001 THEN SUM(elapsed) ELSE 0.001 END
                        AS generation_tps,
                    SUM(prefill_token_delta) /
                        CASE WHEN SUM(elapsed) > 0.001 THEN SUM(elapsed) ELSE 0.001 END
                        AS prefill_tps,
                    AVG(running) AS running,
                    AVG(waiting) AS waiting,
                    MAX(kv_cache_usage) AS kv_cache_usage
                FROM samples
                WHERE timestamp >= ?
                GROUP BY bucket
                ORDER BY bucket
                """,
                (step, step, since),
            ).fetchall()
        return [dict(row) for row in rows]

    def daily(self, days: int) -> list[dict]:
        days = min(365, max(1, int(days)))
        first_day = (self._datetime().date() - timedelta(days=days - 1)).isoformat()
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM daily_totals WHERE day >= ? ORDER BY day",
                (first_day,),
            ).fetchall()
        return [dict(row) for row in rows]
