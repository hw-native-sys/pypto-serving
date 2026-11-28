# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Workload measurements and adaptive cache reservation for agent sessions."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class TTLEstimate:
    ttl_seconds: float
    candidate_ttl_seconds: float
    candidate_utility_seconds: float | None
    uses_fitted_distribution: bool


@dataclass(frozen=True)
class ReturnInterval:
    seconds: float
    miss_cost: float
    reservation: float


@dataclass(frozen=True)
class _ReturnDistribution:
    log_center: float
    log_spread: float

    @classmethod
    def fit(cls, intervals: deque[ReturnInterval]) -> _ReturnDistribution | None:
        values = [math.log(interval.seconds) for interval in intervals if interval.seconds > 0]
        if len(values) < 2:
            return None
        center = fmean(values)
        return cls(center, math.sqrt(fmean((value - center) ** 2 for value in values)))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def expected_utility(self, seconds: float, miss_cost: float) -> float:
        position = (math.log(max(seconds, 1e-6)) - self.log_center) / self.log_spread
        hit_probability = self._normal_cdf(position)
        mean_delay = math.exp(self.log_center + self.log_spread**2 / 2.0)
        idle_time_on_hit = mean_delay * self._normal_cdf(position - self.log_spread)
        return miss_cost * hit_probability - idle_time_on_hit - seconds * (1.0 - hit_probability)

    def best_reservation(self, miss_cost: float, minimum: float, maximum: float) -> float:
        if minimum == maximum:
            return minimum
        floor_log = math.log(max(minimum, 1e-6))
        ceiling_log = math.log(max(maximum, 1e-6))
        candidates = {minimum, maximum, min(maximum, max(minimum, miss_cost))}
        for step in range(32):
            candidates.add(math.exp(floor_log + (ceiling_log - floor_log) * step / 31))
        # Prefer the smaller reservation if two candidates have equal utility.
        return max(candidates, key=lambda ttl: (self.expected_utility(ttl, miss_cost), -ttl))


# When a column has no measurements, retain its previous estimate. These are
# the initial values used before the first refresh.
_INITIAL_AVERAGES = {
    "avg_prompt_tokens": 1024.0,
    "avg_cached_prefix_tokens": 0.0,
    "avg_uncached_prompt_tokens": 1024.0,
    "avg_completion_tokens": 256.0,
    "avg_total_tokens": 1280.0,
    "avg_request_latency_seconds": 1.0,
    "avg_decode_seconds": 1.0,
    "avg_active_programs": 1.0,
    "avg_waiting_programs": 0.0,
    "avg_input_token_growth_per_round": 1024.0,
    "avg_cache_churn_tokens_per_round": 1280.0,
    "avg_inter_request_gap_seconds": 0.0,
    "avg_scheduler_queue_seconds": 0.0,
    "avg_rounds_since_ttl_pause": 0.0,
}


class AgentTTLStatistics:
    def __init__(self, request_window_size: int = 100, min_update_window_fraction: float = 0.5) -> None:
        if request_window_size <= 0:
            raise ValueError("request_window_size must be positive")
        if not 0 < min_update_window_fraction <= 1:
            raise ValueError("min_update_window_fraction must be in (0, 1]")
        self.request_window_size = request_window_size
        self._refresh_at = max(1, int(request_window_size * min_update_window_fraction))
        self._requests: deque[tuple[int | float | None, ...]] = deque()
        self._sums = [0] * len(_INITIAL_AVERAGES)
        self._counts = [0] * len(_INITIAL_AVERAGES)
        self._request_times: deque[tuple[float, float]] = deque(maxlen=request_window_size)
        self._rounds_before_pause: deque[int] = deque(maxlen=request_window_size)
        self._return_intervals: deque[ReturnInterval] = deque(maxlen=request_window_size)
        self._return_distribution: _ReturnDistribution | None = None
        self.avg_ttl_pause_segment_rounds = 1.0
        self.request_count = 0

        self.avg_prompt_tokens = _INITIAL_AVERAGES["avg_prompt_tokens"]
        self.avg_cached_prefix_tokens = _INITIAL_AVERAGES["avg_cached_prefix_tokens"]
        self.avg_uncached_prompt_tokens = _INITIAL_AVERAGES["avg_uncached_prompt_tokens"]
        self.avg_completion_tokens = _INITIAL_AVERAGES["avg_completion_tokens"]
        self.avg_total_tokens = _INITIAL_AVERAGES["avg_total_tokens"]
        self.avg_request_latency_seconds = _INITIAL_AVERAGES["avg_request_latency_seconds"]
        self.avg_decode_seconds = _INITIAL_AVERAGES["avg_decode_seconds"]
        self.avg_active_programs = _INITIAL_AVERAGES["avg_active_programs"]
        self.avg_waiting_programs = _INITIAL_AVERAGES["avg_waiting_programs"]
        self.avg_input_token_growth_per_round = _INITIAL_AVERAGES["avg_input_token_growth_per_round"]
        self.avg_cache_churn_tokens_per_round = _INITIAL_AVERAGES["avg_cache_churn_tokens_per_round"]
        self.avg_inter_request_gap_seconds = _INITIAL_AVERAGES["avg_inter_request_gap_seconds"]
        self.avg_scheduler_queue_seconds = _INITIAL_AVERAGES["avg_scheduler_queue_seconds"]
        self.avg_rounds_since_ttl_pause = _INITIAL_AVERAGES["avg_rounds_since_ttl_pause"]

    def update_request(
        self,
        *,
        prompt_tokens: int,
        cached_prefix_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        request_latency_seconds: float,
        active_programs: int,
        waiting_programs: int,
        input_token_growth: int,
        inter_request_gap_seconds: float,
        cache_churn_tokens: int | None = None,
        rounds_since_ttl_pause: int = 0,
        decode_seconds: float | None = None,
        scheduler_queue_seconds: float = 0.0,
    ) -> None:
        durations = (request_latency_seconds, inter_request_gap_seconds, scheduler_queue_seconds)
        if any(not math.isfinite(duration) for duration in durations):
            raise ValueError(
                "request latency, inter-request gap, and scheduler queue duration must be finite"
            )
        if decode_seconds is not None and (not math.isfinite(decode_seconds) or decode_seconds < 0):
            raise ValueError("decode duration must be finite and non-negative")
        if cache_churn_tokens is None:
            cache_churn_tokens = max(0, prompt_tokens - cached_prefix_tokens) + max(0, completion_tokens)
        prompt = max(0, prompt_tokens)
        cached = max(0, min(prompt, cached_prefix_tokens))
        total = max(0, total_tokens)
        completion = max(0, completion_tokens)
        valid_completion = total > 0 and completion <= total and (prompt > 0 or completion < total)
        valid_growth = 0 < input_token_growth < total - input_token_growth
        sample = (
            prompt,
            cached,
            prompt - cached,
            completion if valid_completion else None,
            total,
            max(request_latency_seconds, 0.0),
            decode_seconds,
            max(0, active_programs),
            max(0, waiting_programs),
            input_token_growth if valid_growth else None,
            max(0, cache_churn_tokens),
            max(inter_request_gap_seconds, 0.0),
            max(scheduler_queue_seconds, 0.0),
            max(0, rounds_since_ttl_pause),
        )
        self.request_count += 1
        self._requests.append(sample)
        self._add_columns(sample, 1)
        if len(self._requests) > self.request_window_size:
            self._add_columns(self._requests.popleft(), -1)
        if self.request_sample_count >= self._refresh_at:
            for name, total, count in zip(_INITIAL_AVERAGES, self._sums, self._counts):
                if count:
                    setattr(self, name, total / count)

    def _add_columns(self, sample: tuple[int | float | None, ...], sign: int) -> None:
        for column, value in enumerate(sample):
            if value is not None:
                self._sums[column] += sign * value
                self._counts[column] += sign

    @property
    def request_sample_count(self) -> int:
        return len(self._requests)

    @property
    def request_window_complete(self) -> bool:
        return self.request_sample_count >= self.request_window_size

    def estimated_continuity_rounds(self) -> int:
        return max(1, math.ceil(2 * self.avg_rounds_since_ttl_pause)) if self.request_window_complete else 0

    def update_ttl_pause(self, *, served_rounds: int) -> None:
        self._rounds_before_pause.append(max(0, served_rounds))
        if self.ttl_pause_window_ready:
            self.avg_ttl_pause_segment_rounds = fmean(self._rounds_before_pause)

    @property
    def ttl_pause_sample_count(self) -> int:
        return len(self._rounds_before_pause)

    @property
    def ttl_pause_window_ready(self) -> bool:
        return self.ttl_pause_sample_count >= self._refresh_at

    def observe_request_interval(
        self, *, started_at_monotonic_s: float, finished_at_monotonic_s: float
    ) -> None:
        start, end = started_at_monotonic_s, finished_at_monotonic_s
        if not (0 <= start <= end < math.inf):
            raise ValueError("request interval must be finite, non-negative, and ordered")
        self._request_times.append((start, end))

    @property
    def request_throughput_per_second(self) -> float:
        spans = iter(sorted(self._request_times))
        current = next(spans, None)
        if current is None:
            return 0.0
        left, right = current
        busy = 0.0
        for start, end in spans:
            if start > right:
                busy += right - left
                left = start
            right = max(right, end)
        busy += right - left
        return len(self._request_times) / busy if busy > 0 else 0.0

    @property
    def continuity_sample_count(self) -> int:
        return len(self._return_intervals)

    @property
    def continuity_window_complete(self) -> bool:
        return self.continuity_sample_count >= self.request_window_size

    def observe_continuity(
        self,
        *,
        interval_seconds: float,
        cache_miss_impact_seconds: float,
        assigned_ttl_seconds: float,
    ) -> ReturnInterval:
        sample = ReturnInterval(
            max(0.0, interval_seconds),
            max(0.0, cache_miss_impact_seconds),
            max(0.0, assigned_ttl_seconds),
        )
        self._return_intervals.append(sample)
        if self.continuity_window_complete:
            self._return_distribution = _ReturnDistribution.fit(self._return_intervals)
        return sample

    def recalibrate_continuity_window(
        self,
        *,
        minimum_seconds: float,
        maximum_seconds: float,
        impact_ratio: float,
    ) -> None:
        if not self.continuity_window_complete:
            return
        recalibrated: deque[ReturnInterval] = deque(maxlen=self.request_window_size)
        for before in self._return_intervals:
            estimate = self.estimate_ttl(
                impact_seconds=before.miss_cost,
                minimum_seconds=minimum_seconds,
                maximum_seconds=maximum_seconds,
                impact_ratio=impact_ratio,
            )
            recalibrated.append(ReturnInterval(before.seconds, before.miss_cost, estimate.ttl_seconds))
        self._return_intervals = recalibrated

    def estimate_ttl(
        self,
        *,
        impact_seconds: float,
        minimum_seconds: float,
        maximum_seconds: float,
        impact_ratio: float,
    ) -> TTLEstimate:
        if not 0 <= impact_ratio <= 1:
            raise ValueError("impact_ratio must be in [0, 1]")
        if minimum_seconds > maximum_seconds:
            raise ValueError("maximum_seconds must be >= minimum_seconds")
        if not self.continuity_window_complete:
            return TTLEstimate(0.0, 0.0, None, False)
        impact = max(0.0, impact_seconds)
        distribution = self._return_distribution
        cap = impact * impact_ratio
        if distribution is None or distribution.log_spread <= 1e-9:
            reservation = min(cap, maximum_seconds, max(minimum_seconds, impact))
            return TTLEstimate(reservation, reservation, None, False)
        candidate = min(cap, distribution.best_reservation(impact, minimum_seconds, maximum_seconds))
        utility = distribution.expected_utility(candidate, impact)
        return TTLEstimate(max(0.0, candidate) if utility >= 0 else 0.0, candidate, utility, True)
