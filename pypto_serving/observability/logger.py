# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .stats import FinishedRequestStats, IterationStats, SchedulerStats

_LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    640.0,
    2560.0,
)

_TPOT_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    40.0,
    80.0,
)


class StatLoggerBase(ABC):
    """Publisher interface modeled after vLLM's stat logger boundary."""

    @abstractmethod
    def record_scheduler(self, engine_index: int, stats: SchedulerStats) -> None:
        pass

    @abstractmethod
    def record_iteration(self, engine_index: int, stats: IterationStats) -> None:
        pass


@dataclass
class _Histogram:
    buckets: tuple[float, ...]
    bucket_counts: list[int] = field(init=False)
    count: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        self.bucket_counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        self.count += 1
        self.total += value
        for index, boundary in enumerate(self.buckets):
            if value <= boundary:
                self.bucket_counts[index] += 1

    def snapshot(self) -> dict:
        return {
            "buckets": [
                {"le": boundary, "count": count}
                for boundary, count in zip(self.buckets, self.bucket_counts)
            ],
            "count": self.count,
            "sum": self.total,
        }


@dataclass
class _RequestState:
    arrival_monotonic: float
    num_prompt_tokens: int
    num_generation_tokens: int = 0
    first_token_monotonic: float | None = None
    last_token_monotonic: float | None = None
    finished: bool = False


@dataclass
class _EngineMetrics:
    running: int = 0
    waiting: int = 0
    kv_cache_usage: float = 0.0
    prompt_tokens: int = 0
    prefill_tokens: int = 0
    generation_tokens: int = 0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    preemptions: int = 0
    finish_reasons: dict[str, int] = field(default_factory=dict)
    ttft: _Histogram = field(default_factory=lambda: _Histogram(_LATENCY_BUCKETS))
    itl: _Histogram = field(default_factory=lambda: _Histogram(_TPOT_BUCKETS))
    tpot: _Histogram = field(default_factory=lambda: _Histogram(_TPOT_BUCKETS))
    e2e: _Histogram = field(default_factory=lambda: _Histogram(_LATENCY_BUCKETS))


class InMemoryStatLogger(StatLoggerBase):
    """Thread-safe cumulative metrics registry for one API server process."""

    def __init__(self, model_name: str, engine_indexes: list[int]) -> None:
        self.model_name = model_name
        self.server_id = uuid.uuid4().hex
        self.started_at = time.time()
        self._lock = threading.RLock()
        self._engines = {index: _EngineMetrics() for index in engine_indexes}
        self._requests: dict[tuple[int, str], _RequestState] = {}

    def start_request(
        self,
        engine_index: int,
        request_id: str,
        *,
        arrival_monotonic: float,
        num_prompt_tokens: int,
    ) -> None:
        with self._lock:
            key = (engine_index, request_id)
            if key in self._requests:
                return
            self._requests[key] = _RequestState(
                arrival_monotonic=arrival_monotonic,
                num_prompt_tokens=num_prompt_tokens,
            )
            self._engine(engine_index).prompt_tokens += num_prompt_tokens

    def record_output(
        self,
        engine_index: int,
        request_id: str,
        *,
        completion_tokens: int,
        timestamp: float | None = None,
    ) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            state = self._requests.get((engine_index, request_id))
            if state is None or state.finished:
                return
            new_tokens = max(0, int(completion_tokens) - state.num_generation_tokens)
            if new_tokens == 0:
                return

            iteration = IterationStats(num_generation_tokens=new_tokens)
            if state.first_token_monotonic is None:
                state.first_token_monotonic = now
                iteration.time_to_first_tokens.append(now - state.arrival_monotonic)
                remaining = new_tokens - 1
            else:
                remaining = new_tokens

            if state.last_token_monotonic is not None and remaining > 0:
                elapsed = max(0.0, now - state.last_token_monotonic)
                iteration.inter_token_latencies.append(elapsed)
                iteration.inter_token_latencies.extend([0.0] * (remaining - 1))
            elif remaining > 0:
                iteration.inter_token_latencies.extend([0.0] * remaining)

            state.num_generation_tokens += new_tokens
            state.last_token_monotonic = now
            self._record_iteration_locked(engine_index, iteration)

    def finish_request(
        self,
        engine_index: int,
        request_id: str,
        finish_reason: str,
        *,
        timestamp: float | None = None,
    ) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            state = self._requests.get((engine_index, request_id))
            if state is None or state.finished:
                return
            state.finished = True
            ttft = (
                state.first_token_monotonic - state.arrival_monotonic
                if state.first_token_monotonic is not None
                else None
            )
            tpot = None
            if state.num_generation_tokens > 1 and state.first_token_monotonic is not None:
                last = state.last_token_monotonic or now
                tpot = (last - state.first_token_monotonic) / (
                    state.num_generation_tokens - 1
                )
            finished = FinishedRequestStats(
                finish_reason=finish_reason,
                e2e_latency=max(0.0, now - state.arrival_monotonic),
                num_prompt_tokens=state.num_prompt_tokens,
                num_generation_tokens=state.num_generation_tokens,
                time_to_first_token=ttft,
                mean_time_per_output_token=tpot,
            )
            self._record_iteration_locked(
                engine_index,
                IterationStats(finished_requests=[finished]),
            )
            del self._requests[(engine_index, request_id)]

    def record_scheduler(self, engine_index: int, stats: SchedulerStats) -> None:
        with self._lock:
            engine = self._engine(engine_index)
            engine.running = stats.num_running_reqs
            engine.waiting = stats.num_waiting_reqs
            engine.kv_cache_usage = min(1.0, max(0.0, stats.kv_cache_usage))
            engine.prefix_cache_queries += stats.prefix_cache_queries
            engine.prefix_cache_hits += stats.prefix_cache_hits
            engine.preemptions += stats.num_preemptions

    def record_iteration(self, engine_index: int, stats: IterationStats) -> None:
        with self._lock:
            self._record_iteration_locked(engine_index, stats)

    def _record_iteration_locked(self, engine_index: int, stats: IterationStats) -> None:
        engine = self._engine(engine_index)
        engine.prefill_tokens += stats.num_prefill_tokens
        engine.generation_tokens += stats.num_generation_tokens
        for value in stats.time_to_first_tokens:
            engine.ttft.observe(value)
        for value in stats.inter_token_latencies:
            engine.itl.observe(value)
        for finished in stats.finished_requests:
            reason = finished.finish_reason.lower()
            engine.finish_reasons[reason] = engine.finish_reasons.get(reason, 0) + 1
            engine.e2e.observe(finished.e2e_latency)
            if finished.mean_time_per_output_token is not None:
                engine.tpot.observe(finished.mean_time_per_output_token)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema_version": 1,
                "server_id": self.server_id,
                "started_at": self.started_at,
                "timestamp": time.time(),
                "model_name": self.model_name,
                "replicas": [
                    self._engine_snapshot(index, metrics)
                    for index, metrics in sorted(self._engines.items())
                ],
            }

    def render_prometheus(self) -> str:
        snapshot = self.snapshot()
        lines: list[str] = []
        self._render_gauge(lines, snapshot, "num_requests_running", "Requests in execution batches", "running")
        self._render_gauge(lines, snapshot, "num_requests_waiting", "Requests waiting to be processed", "waiting")
        self._render_gauge(lines, snapshot, "kv_cache_usage_perc", "KV cache usage from zero to one", "kv_cache_usage")
        self._render_counter(lines, snapshot, "prompt_tokens", "Prompt tokens received", "prompt_tokens")
        self._render_counter(lines, snapshot, "prefill_tokens", "Prefill tokens computed", "prefill_tokens")
        self._render_counter(lines, snapshot, "generation_tokens", "Generation tokens produced", "generation_tokens")
        self._render_counter(lines, snapshot, "prefix_cache_queries", "Prefix cache queried tokens", "prefix_cache_queries")
        self._render_counter(lines, snapshot, "prefix_cache_hits", "Prefix cache hit tokens", "prefix_cache_hits")
        self._render_counter(lines, snapshot, "num_preemptions", "Cumulative scheduler preemptions", "preemptions")
        self._render_finish_reasons(lines, snapshot)
        self._render_histogram(lines, snapshot, "time_to_first_token_seconds", "Time to first token", "ttft")
        self._render_histogram(lines, snapshot, "inter_token_latency_seconds", "Inter-token latency", "itl")
        self._render_histogram(lines, snapshot, "request_time_per_output_token_seconds", "Mean time per output token per request", "tpot")
        self._render_histogram(lines, snapshot, "e2e_request_latency_seconds", "End-to-end request latency", "e2e")
        return "\n".join(lines) + "\n"

    def _engine(self, engine_index: int) -> _EngineMetrics:
        if engine_index not in self._engines:
            self._engines[engine_index] = _EngineMetrics()
        return self._engines[engine_index]

    @staticmethod
    def _engine_snapshot(index: int, metrics: _EngineMetrics) -> dict:
        return {
            "engine": index,
            "gauges": {
                "running": metrics.running,
                "waiting": metrics.waiting,
                "kv_cache_usage": metrics.kv_cache_usage,
            },
            "counters": {
                "prompt_tokens": metrics.prompt_tokens,
                "prefill_tokens": metrics.prefill_tokens,
                "generation_tokens": metrics.generation_tokens,
                "prefix_cache_queries": metrics.prefix_cache_queries,
                "prefix_cache_hits": metrics.prefix_cache_hits,
                "preemptions": metrics.preemptions,
                "requests_finished": sum(metrics.finish_reasons.values()),
                "requests_error": metrics.finish_reasons.get("error", 0),
                "requests_aborted": metrics.finish_reasons.get("finished_aborted", 0),
            },
            "finish_reasons": dict(sorted(metrics.finish_reasons.items())),
            "histograms": {
                "ttft": metrics.ttft.snapshot(),
                "itl": metrics.itl.snapshot(),
                "tpot": metrics.tpot.snapshot(),
                "e2e": metrics.e2e.snapshot(),
            },
        }

    def _labels(self, engine_index: int, extra: dict[str, str] | None = None) -> str:
        values = {
            "model_name": self.model_name,
            "engine": str(engine_index),
        }
        if extra:
            values.update(extra)
        encoded = ",".join(
            f'{key}="{self._escape_label(value)}"' for key, value in values.items()
        )
        return "{" + encoded + "}"

    @staticmethod
    def _escape_label(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    def _render_gauge(self, lines: list[str], snapshot: dict, name: str, help_text: str, key: str) -> None:
        metric = f"pypto:{name}"
        lines.extend((f"# HELP {metric} {help_text}.", f"# TYPE {metric} gauge"))
        for replica in snapshot["replicas"]:
            lines.append(f"{metric}{self._labels(replica['engine'])} {replica['gauges'][key]}")

    def _render_counter(self, lines: list[str], snapshot: dict, name: str, help_text: str, key: str) -> None:
        metric = f"pypto:{name}_total"
        lines.extend((f"# HELP {metric} {help_text}.", f"# TYPE {metric} counter"))
        for replica in snapshot["replicas"]:
            lines.append(f"{metric}{self._labels(replica['engine'])} {replica['counters'][key]}")

    def _render_finish_reasons(self, lines: list[str], snapshot: dict) -> None:
        metric = "pypto:request_success_total"
        lines.extend((f"# HELP {metric} Count of terminated requests.", f"# TYPE {metric} counter"))
        for replica in snapshot["replicas"]:
            for reason, count in replica["finish_reasons"].items():
                labels = self._labels(replica["engine"], {"finished_reason": reason})
                lines.append(f"{metric}{labels} {count}")

    def _render_histogram(
        self,
        lines: list[str],
        snapshot: dict,
        name: str,
        help_text: str,
        key: str,
    ) -> None:
        metric = f"pypto:{name}"
        lines.extend((f"# HELP {metric} {help_text} in seconds.", f"# TYPE {metric} histogram"))
        for replica in snapshot["replicas"]:
            histogram = replica["histograms"][key]
            for bucket in histogram["buckets"]:
                labels = self._labels(replica["engine"], {"le": self._format_float(bucket["le"])})
                lines.append(f"{metric}_bucket{labels} {bucket['count']}")
            labels = self._labels(replica["engine"], {"le": "+Inf"})
            lines.append(f"{metric}_bucket{labels} {histogram['count']}")
            base_labels = self._labels(replica["engine"])
            lines.append(f"{metric}_sum{base_labels} {histogram['sum']}")
            lines.append(f"{metric}_count{base_labels} {histogram['count']}")

    @staticmethod
    def _format_float(value: float) -> str:
        if math.isinf(value):
            return "+Inf"
        return f"{value:g}"
