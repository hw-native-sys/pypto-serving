# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Translate measured model work into inputs for adaptive cache retention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable

from pypto_serving.serving.agent_ttl import AgentTTLStatistics


@dataclass(frozen=True)
class AgentModelCosts:
    prefill_fixed: float
    prefill_per_1000: float
    prefill_per_1000_squared: float
    prefill_interference: float
    decode_fixed: float
    decode_per_session: float
    decode_per_token: float

    def reload_cost(self, context: int, reusable: int) -> float:
        missing = (context - reusable) / 1000
        if missing <= 0:
            return 0.0
        prefill = (
            self.prefill_fixed + missing * self.prefill_per_1000 + self.prefill_per_1000_squared * missing**2
        )
        return prefill + self.prefill_interference * prefill

    def decode_rate(self, batch: int, context: int) -> float:
        return batch / (self.decode_fixed + self.decode_per_session * batch + self.decode_per_token * context)


@dataclass(frozen=True)
class AgentRound:
    prompt: int
    generated: int
    total: int
    prior_context: int
    cached: int
    shared: int
    local: int
    reclaimable: int
    duration: float
    decode: float | None
    active: int
    paused: int
    gap: float
    queue_time: float
    finished: float

    @classmethod
    def from_metrics(cls, metric: Callable[[str], object], finished: float) -> AgentRound:
        def number(name: str) -> float | None:
            value = metric(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                return max(0, value)
            return None

        def tokens(name: str) -> int:
            return int(number(name) or 0)

        prompt, generated = tokens("prompt_tokens"), tokens("completion_tokens")
        cached = min(prompt, tokens("cached_prefix_tokens"))
        local = number("hbm_cached_prefix_tokens")
        local = min(prompt, cached if local is None else int(local))
        return cls(
            prompt=prompt,
            generated=generated,
            total=tokens("total_tokens") or prompt + generated,
            prior_context=tokens("previous_context_tokens"),
            cached=cached,
            shared=min(cached, tokens("shared_prefix_tokens")),
            local=local,
            reclaimable=min(local, tokens("hbm_hit_ref_zero_tokens")),
            duration=number("request_latency_seconds") or 0.0,
            decode=number("decode_seconds"),
            active=tokens("active_programs"),
            paused=tokens("waiting_programs"),
            gap=number("inter_request_gap_seconds") or 0.0,
            queue_time=number("scheduler_queue_seconds") or 0.0,
            finished=finished,
        )

    @property
    def maintains_context(self) -> bool:
        return self.prior_context == 0 or self.local > 0.9 * self.prior_context

    def collect(self, stats: AgentTTLStatistics, *, first: bool, uninterrupted: int) -> None:
        stats.observe_request_interval(
            started_at_monotonic_s=max(0.0, self.finished - self.duration),
            finished_at_monotonic_s=self.finished,
        )
        newly_used = self.prompt - self.local + self.generated
        stats.update_request(
            prompt_tokens=self.prompt,
            cached_prefix_tokens=self.cached,
            completion_tokens=self.generated,
            total_tokens=self.total,
            request_latency_seconds=self.duration,
            decode_seconds=self.decode,
            active_programs=self.active,
            waiting_programs=self.paused,
            input_token_growth=max(0, self.total - self.prior_context),
            cache_churn_tokens=newly_used + (self.reclaimable if first else 0),
            inter_request_gap_seconds=self.gap,
            scheduler_queue_seconds=self.queue_time,
            rounds_since_ttl_pause=uninterrupted,
        )


def prefix_refresh_interval(
    stats: AgentTTLStatistics,
    *,
    running: int,
    pool_tokens: int | None,
    warmup: float,
    turnovers: float,
) -> float:
    churn = stats.avg_cache_churn_tokens_per_round
    if not stats.request_window_complete or not running or pool_tokens is None or churn <= 0:
        return warmup
    return turnovers * pool_tokens * stats.avg_request_latency_seconds / running / churn
