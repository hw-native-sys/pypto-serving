# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Reusable bounded rolling statistics for the Progress-TTL policy.

This is the single shared implementation used by Router-hosted and engine-embedded Progress-TTL runtimes. It preserves
the original PT request-window sizing, cold-start threshold, gradual eviction, sample sanitization, and pause window.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestStatSample:
    """Sanitized per-request observation used by windowed PT statistics."""

    prompt_tokens: int
    cached_prefix_tokens: int
    uncached_prompt_tokens: int
    completion_tokens: int | None
    total_tokens: int
    request_latency_seconds: float
    decode_seconds: float | None
    active_programs: int
    waiting_programs: int
    input_token_growth: int | None
    cache_churn_tokens: int
    inter_request_gap_seconds: float
    scheduler_queue_seconds: float
    rounds_since_ttl_pause: int


@dataclass(frozen=True)
class ContinuitySample:
    """One completed tool-call interval used to evaluate continuity protection."""

    interval_seconds: float
    cache_miss_impact_seconds: float
    assigned_ttl_seconds: float
    utility_seconds: float


@dataclass(frozen=True)
class TTLEstimate:
    """One TTL decision and its fitted expected utility, when a complete interval window exists."""

    ttl_seconds: float
    candidate_ttl_seconds: float
    candidate_utility_seconds: float | None
    uses_fitted_distribution: bool


@dataclass(frozen=True)
class RequestIntervalSample:
    """One completed request interval retained for busy-time throughput estimation."""

    started_at_monotonic_s: float
    finished_at_monotonic_s: float


@dataclass
class ProgressTTLGlobalFactors:
    """Policy-wide decision factors derived from bounded recent request and pause windows."""

    request_count: int = 0
    request_window_size: int = 100
    min_update_window_fraction: float = 0.5
    avg_prompt_tokens: float = 1024.0
    avg_cached_prefix_tokens: float = 0.0
    avg_uncached_prompt_tokens: float = 1024.0
    avg_completion_tokens: float = 256.0
    avg_total_tokens: float = 1280.0
    avg_scheduler_queue_seconds: float = 0.0
    avg_request_latency_seconds: float = 1.0
    avg_decode_seconds: float = 1.0
    avg_active_programs: float = 1.0
    avg_waiting_programs: float = 0.0
    avg_ttl_pause_segment_rounds: float = 1.0
    avg_input_token_growth_per_round: float = 1024.0
    avg_cache_churn_tokens_per_round: float = 1280.0
    avg_rounds_since_ttl_pause: float = 0.0
    avg_inter_request_gap_seconds: float = 0.0
    avg_reasoning_seconds: float = 1.0
    avg_acting_seconds: float = 0.0
    cold_prefill_cost_seconds: float = 1.0
    continuity_enabled: bool = False
    continuity_log_mu: float | None = None
    continuity_log_sigma: float | None = None
    continuity_utility_seconds: float = 0.0
    _request_samples: deque[RequestStatSample] = field(default_factory=deque, init=False, repr=False)
    _prompt_token_sum: int = field(default=0, init=False, repr=False)
    _cached_prefix_token_sum: int = field(default=0, init=False, repr=False)
    _uncached_prompt_token_sum: int = field(default=0, init=False, repr=False)
    _completion_token_sum: int = field(default=0, init=False, repr=False)
    _completion_token_count: int = field(default=0, init=False, repr=False)
    _total_token_sum: int = field(default=0, init=False, repr=False)
    _request_latency_sum: float = field(default=0.0, init=False, repr=False)
    _decode_seconds_sum: float = field(default=0.0, init=False, repr=False)
    _decode_seconds_count: int = field(default=0, init=False, repr=False)
    _active_program_sum: int = field(default=0, init=False, repr=False)
    _waiting_program_sum: int = field(default=0, init=False, repr=False)
    _inter_request_gap_sum: float = field(default=0.0, init=False, repr=False)
    _scheduler_queue_seconds_sum: float = field(default=0.0, init=False, repr=False)
    _rounds_since_ttl_pause_sum: int = field(default=0, init=False, repr=False)
    _cache_churn_token_sum: int = field(default=0, init=False, repr=False)
    _ttl_pause_segment_rounds: deque[int] = field(default_factory=deque, init=False, repr=False)
    _ttl_pause_segment_rounds_sum: int = field(default=0, init=False, repr=False)
    _continuity_samples: deque[ContinuitySample] = field(default_factory=deque, init=False, repr=False)
    _request_intervals: deque[RequestIntervalSample] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate bounded-window controls and finite initial decision values."""
        if self.request_count < 0:
            raise ValueError("request_count must be non-negative")
        if self.request_window_size <= 0:
            raise ValueError("request_window_size must be positive")
        if not math.isfinite(self.min_update_window_fraction) or not 0 < self.min_update_window_fraction <= 1:
            raise ValueError("min_update_window_fraction must be in (0, 1]")
        decision_values = (
            self.avg_prompt_tokens,
            self.avg_cached_prefix_tokens,
            self.avg_uncached_prompt_tokens,
            self.avg_completion_tokens,
            self.avg_total_tokens,
            self.avg_scheduler_queue_seconds,
            self.avg_request_latency_seconds,
            self.avg_decode_seconds,
            self.avg_active_programs,
            self.avg_waiting_programs,
            self.avg_ttl_pause_segment_rounds,
            self.avg_input_token_growth_per_round,
            self.avg_cache_churn_tokens_per_round,
            self.avg_rounds_since_ttl_pause,
            self.avg_inter_request_gap_seconds,
            self.avg_reasoning_seconds,
            self.avg_acting_seconds,
            self.cold_prefill_cost_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in decision_values):
            raise ValueError("initial Progress-TTL decision values must be finite and non-negative")
        if not math.isfinite(self.continuity_utility_seconds):
            raise ValueError("continuity utility must be finite")

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
        """Update request-level rolling averages using one sanitized sample."""
        sample = self._sanitize_request_sample(
            prompt_tokens=prompt_tokens,
            cached_prefix_tokens=cached_prefix_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_latency_seconds=request_latency_seconds,
            active_programs=active_programs,
            waiting_programs=waiting_programs,
            input_token_growth=input_token_growth,
            cache_churn_tokens=(
                max(0, prompt_tokens - cached_prefix_tokens) + max(0, completion_tokens)
                if cache_churn_tokens is None
                else cache_churn_tokens
            ),
            inter_request_gap_seconds=inter_request_gap_seconds,
            scheduler_queue_seconds=scheduler_queue_seconds,
            rounds_since_ttl_pause=rounds_since_ttl_pause,
            decode_seconds=decode_seconds,
        )
        self.request_count += 1
        self._append_request_sample(sample)
        self._evict_request_samples_to_target()
        self._refresh_request_averages_if_ready()

    def update_ttl_pause(self, *, served_rounds: int) -> None:
        """Update the bounded workload-continuity window at an actual TTL pause."""
        sanitized_rounds = max(served_rounds, 0)
        self._ttl_pause_segment_rounds.append(sanitized_rounds)
        self._ttl_pause_segment_rounds_sum += sanitized_rounds
        while len(self._ttl_pause_segment_rounds) > self.request_window_size:
            self._ttl_pause_segment_rounds_sum -= self._ttl_pause_segment_rounds.popleft()
        if self.ttl_pause_window_ready:
            self.avg_ttl_pause_segment_rounds = self._ttl_pause_segment_rounds_sum / len(self._ttl_pause_segment_rounds)

    def observe_request_interval(
        self,
        *,
        started_at_monotonic_s: float,
        finished_at_monotonic_s: float,
    ) -> None:
        """Record one completed request interval for aggregate throughput estimation.

        Args:
            started_at_monotonic_s: Monotonic time at which the request entered scheduling.
            finished_at_monotonic_s: Monotonic time at which the inference request completed.
        """
        if (
            not math.isfinite(started_at_monotonic_s)
            or started_at_monotonic_s < 0
            or not math.isfinite(finished_at_monotonic_s)
            or finished_at_monotonic_s < started_at_monotonic_s
        ):
            raise ValueError("request interval must be finite, non-negative, and ordered")
        self._request_intervals.append(
            RequestIntervalSample(
                started_at_monotonic_s=started_at_monotonic_s,
                finished_at_monotonic_s=finished_at_monotonic_s,
            )
        )
        while len(self._request_intervals) > self.request_window_size:
            self._request_intervals.popleft()

    @property
    def request_throughput_per_second(self) -> float:
        """Return completions per second after excluding uncovered idle gaps."""
        if not self._request_intervals:
            return 0.0
        covered_seconds = 0.0
        intervals = sorted(
            self._request_intervals,
            key=lambda sample: (sample.started_at_monotonic_s, sample.finished_at_monotonic_s),
        )
        covered_start = intervals[0].started_at_monotonic_s
        covered_end = intervals[0].finished_at_monotonic_s
        for sample in intervals[1:]:
            if sample.started_at_monotonic_s > covered_end:
                covered_seconds += covered_end - covered_start
                covered_start = sample.started_at_monotonic_s
                covered_end = sample.finished_at_monotonic_s
            else:
                covered_end = max(covered_end, sample.finished_at_monotonic_s)
        covered_seconds += covered_end - covered_start
        if covered_seconds <= 0:
            return 0.0
        return len(intervals) / covered_seconds

    @property
    def ttl_pause_sample_count(self) -> int:
        """Return actual TTL pauses retained for workload-continuity estimation."""
        return len(self._ttl_pause_segment_rounds)

    @property
    def ttl_pause_window_ready(self) -> bool:
        """Return whether TTL-pause rounds are mature enough to refresh their workload average."""
        return len(self._ttl_pause_segment_rounds) >= self._min_samples_for_update(self.request_window_size)

    @property
    def request_window_complete(self) -> bool:
        """Return whether rolling request statistics are sufficient for capacity growth projection."""
        return len(self._request_samples) >= self.request_window_size

    @property
    def request_sample_count(self) -> int:
        """Return the bounded number of request samples retained for policy estimates."""
        return len(self._request_samples)

    def estimated_continuity_rounds(self) -> int:
        """Return the unbounded workload estimate used before policy-specific segment caps."""
        if not self.request_window_complete:
            return 0
        return max(1, math.ceil(2 * self.avg_rounds_since_ttl_pause))

    def observe_continuity(
        self,
        *,
        interval_seconds: float,
        cache_miss_impact_seconds: float,
        assigned_ttl_seconds: float,
    ) -> ContinuitySample:
        """Record one complete arrival interval and refresh fitted interval factors when possible."""
        interval = max(0.0, interval_seconds)
        impact = max(0.0, cache_miss_impact_seconds)
        ttl = max(0.0, assigned_ttl_seconds)
        # Retaining an acting Program until ``ttl`` preserves its KV only when
        # the next request returns before that deadline. A hit avoids the
        # estimated cold-prefill impact and pays only the actual idle interval;
        # an expiry pays the full TTL reservation cost without a cache benefit.
        utility = impact - interval if interval <= ttl else -ttl
        self._continuity_samples.append(ContinuitySample(interval, impact, ttl, utility))
        while len(self._continuity_samples) > self.request_window_size:
            self._continuity_samples.popleft()
        if self.continuity_window_complete:
            self._refresh_continuity_fit()
            self._refresh_continuity_utility()
        return self._continuity_samples[-1]

    def recalibrate_continuity_window(
        self,
        *,
        minimum_seconds: float,
        maximum_seconds: float,
        impact_ratio: float,
    ) -> ContinuitySample | None:
        """Re-estimate every retained sample TTL after the first complete fitted interval window.

        Warm-up samples were assigned before an interval distribution existed. Replacing only their accounting TTLs
        prevents those provisional values from biasing the first posterior utility used by AUTO mode; it never mutates
        a live Program deadline.
        """
        if not self.continuity_window_complete:
            return None
        if not 0 <= impact_ratio <= 1:
            raise ValueError("impact_ratio must be in [0, 1]")
        self._continuity_samples = deque(
            self._reestimated_sample(
                sample,
                minimum_seconds=minimum_seconds,
                maximum_seconds=maximum_seconds,
                impact_ratio=impact_ratio,
            )
            for sample in self._continuity_samples
        )
        self._refresh_continuity_utility()
        return self._continuity_samples[-1]

    def update_continuity_mode(
        self,
        *,
        enable_threshold_seconds: float,
        disable_threshold_seconds: float,
    ) -> bool:
        """Apply AUTO hysteresis to posterior utility and return whether the enabled state changed."""
        if not self.continuity_window_complete:
            return False
        if not 0 <= disable_threshold_seconds <= enable_threshold_seconds:
            raise ValueError("AUTO disable threshold must be non-negative and no greater than the enable threshold")
        previous = self.continuity_enabled
        if self.continuity_enabled:
            if self.continuity_utility_seconds < disable_threshold_seconds:
                self.continuity_enabled = False
        elif self.continuity_utility_seconds >= enable_threshold_seconds:
            self.continuity_enabled = True
        return previous != self.continuity_enabled

    @property
    def continuity_window_complete(self) -> bool:
        """Return whether the interval window has reached its fixed sample target."""
        return len(self._continuity_samples) >= self.request_window_size

    @property
    def continuity_sample_count(self) -> int:
        """Return the number of complete inter-request intervals in the continuity window."""
        return len(self._continuity_samples)

    def recommended_ttl_seconds(
        self, *, impact_seconds: float, minimum_seconds: float, maximum_seconds: float
    ) -> float:
        """Return zero during warm-up or the complete-window lognormal expected-utility optimum."""
        impact = max(0.0, impact_seconds)
        if maximum_seconds < minimum_seconds:
            raise ValueError("maximum_seconds must be >= minimum_seconds")
        if not self.continuity_window_complete:
            return 0.0
        if self.continuity_log_mu is None or self.continuity_log_sigma is None:
            return min(maximum_seconds, max(minimum_seconds, impact))
        sigma = self.continuity_log_sigma
        if sigma <= 1e-9:
            return min(maximum_seconds, max(minimum_seconds, impact))
        candidates = self._log_spaced_candidates(minimum_seconds, maximum_seconds, impact)
        return max(candidates, key=lambda ttl: self._lognormal_utility(impact, ttl, self.continuity_log_mu, sigma))

    def estimate_ttl(
        self,
        *,
        impact_seconds: float,
        minimum_seconds: float,
        maximum_seconds: float,
        impact_ratio: float,
    ) -> TTLEstimate:
        """Return an impact-capped fitted TTL, disabling TTL when its fitted expected utility is negative."""
        if not 0 <= impact_ratio <= 1:
            raise ValueError("impact_ratio must be in [0, 1]")
        impact = max(0.0, impact_seconds)
        candidate = min(
            self.recommended_ttl_seconds(
                impact_seconds=impact,
                minimum_seconds=minimum_seconds,
                maximum_seconds=maximum_seconds,
            ),
            impact * impact_ratio,
        )
        if not self._has_fitted_continuity_distribution():
            return TTLEstimate(candidate, candidate, None, False)
        utility = self._lognormal_utility(
            impact,
            candidate,
            self.continuity_log_mu,
            self.continuity_log_sigma,
        )
        return TTLEstimate(0.0 if utility < 0 else candidate, candidate, utility, True)

    def _reestimated_sample(
        self,
        sample: ContinuitySample,
        *,
        minimum_seconds: float,
        maximum_seconds: float,
        impact_ratio: float,
    ) -> ContinuitySample:
        estimate = self.estimate_ttl(
            impact_seconds=sample.cache_miss_impact_seconds,
            minimum_seconds=minimum_seconds,
            maximum_seconds=maximum_seconds,
            impact_ratio=impact_ratio,
        )
        ttl = estimate.ttl_seconds
        return ContinuitySample(
            interval_seconds=sample.interval_seconds,
            cache_miss_impact_seconds=sample.cache_miss_impact_seconds,
            assigned_ttl_seconds=ttl,
            utility_seconds=self._sample_utility(sample.interval_seconds, sample.cache_miss_impact_seconds, ttl),
        )

    def _refresh_continuity_fit(self) -> None:
        logs = [math.log(sample.interval_seconds) for sample in self._continuity_samples if sample.interval_seconds > 0]
        if len(logs) < 2:
            self.continuity_log_mu = None
            self.continuity_log_sigma = None
            return
        self.continuity_log_mu = statistics.fmean(logs)
        self.continuity_log_sigma = math.sqrt(statistics.fmean((value - self.continuity_log_mu) ** 2 for value in logs))

    def _refresh_continuity_utility(self) -> None:
        self.continuity_utility_seconds = sum(sample.utility_seconds for sample in self._continuity_samples)

    def _has_fitted_continuity_distribution(self) -> bool:
        return (
            self.continuity_window_complete
            and self.continuity_log_mu is not None
            and self.continuity_log_sigma is not None
            and self.continuity_log_sigma > 1e-9
        )

    @staticmethod
    def _sample_utility(interval_seconds: float, impact_seconds: float, ttl_seconds: float) -> float:
        return impact_seconds - interval_seconds if interval_seconds <= ttl_seconds else -ttl_seconds

    @staticmethod
    def _log_spaced_candidates(minimum: float, maximum: float, impact: float) -> tuple[float, ...]:
        if minimum == maximum:
            return (minimum,)
        log_min = math.log(max(minimum, 1e-6))
        log_max = math.log(max(maximum, 1e-6))
        grid = tuple(math.exp(log_min + (log_max - log_min) * index / 31) for index in range(32))
        return tuple(sorted({minimum, maximum, min(maximum, max(minimum, impact)), *grid}))

    @staticmethod
    def _lognormal_utility(impact: float, ttl: float, mu: float, sigma: float) -> float:
        z = (math.log(max(ttl, 1e-6)) - mu) / sigma
        normal_cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        shifted_cdf = 0.5 * (1.0 + math.erf((z - sigma) / math.sqrt(2.0)))
        truncated_interval = math.exp(mu + sigma * sigma / 2.0) * shifted_cdf
        # U(T) = E[(C - interval) * 1(interval <= T)] - T * P(interval > T)
        #      = C * F(T) - E[interval * 1(interval <= T)] - T * (1 - F(T)).
        return impact * normal_cdf - truncated_interval - ttl * (1.0 - normal_cdf)

    def _sanitize_request_sample(
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
        cache_churn_tokens: int,
        inter_request_gap_seconds: float,
        scheduler_queue_seconds: float,
        rounds_since_ttl_pause: int,
        decode_seconds: float | None,
    ) -> RequestStatSample:
        if (
            not math.isfinite(request_latency_seconds)
            or not math.isfinite(inter_request_gap_seconds)
            or not math.isfinite(scheduler_queue_seconds)
        ):
            raise ValueError("request latency, inter-request gap, and scheduler queue duration must be finite")
        if decode_seconds is not None and (not math.isfinite(decode_seconds) or decode_seconds < 0):
            raise ValueError("decode duration must be finite and non-negative")
        prompt_value = max(prompt_tokens, 0)
        cached_value = min(prompt_value, max(cached_prefix_tokens, 0))
        total_value = max(total_tokens, 0)
        completion_value = self._sanitize_completion_tokens(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_value,
            total_tokens=total_value,
        )
        input_growth = self._sanitize_input_token_growth(
            input_token_growth=input_token_growth,
            previous_total_tokens=max(0, total_value - max(input_token_growth, 0)),
        )
        return RequestStatSample(
            prompt_tokens=prompt_value,
            cached_prefix_tokens=cached_value,
            uncached_prompt_tokens=prompt_value - cached_value,
            completion_tokens=completion_value,
            total_tokens=total_value,
            request_latency_seconds=max(request_latency_seconds, 0.0),
            decode_seconds=decode_seconds,
            active_programs=max(active_programs, 0),
            waiting_programs=max(waiting_programs, 0),
            input_token_growth=input_growth,
            cache_churn_tokens=max(0, cache_churn_tokens),
            inter_request_gap_seconds=max(inter_request_gap_seconds, 0.0),
            scheduler_queue_seconds=max(scheduler_queue_seconds, 0.0),
            rounds_since_ttl_pause=max(rounds_since_ttl_pause, 0),
        )

    @staticmethod
    def _sanitize_completion_tokens(*, completion_tokens: int, prompt_tokens: int, total_tokens: int) -> int | None:
        completion_value = max(completion_tokens, 0)
        if total_tokens <= 0:
            return None
        if prompt_tokens <= 0 and completion_value >= total_tokens:
            return None
        if completion_value > total_tokens:
            return None
        return completion_value

    @staticmethod
    def _sanitize_input_token_growth(*, input_token_growth: int, previous_total_tokens: int) -> int | None:
        """Return a credible per-round context increase, excluding reset-like outliers.

        A normal agent round appends less context than it already carries. A growth at least as large as the whole
        prior context indicates a context replacement, a new prompt lineage, or an inconsistent token report; it is
        not representative of the continuous-growth capacity reserve and is excluded rather than clipped.
        """
        if previous_total_tokens <= 0 or input_token_growth <= 0:
            return None
        if input_token_growth < previous_total_tokens:
            return input_token_growth
        return None

    def _append_request_sample(self, sample: RequestStatSample) -> None:
        self._request_samples.append(sample)
        self._prompt_token_sum += sample.prompt_tokens
        self._cached_prefix_token_sum += sample.cached_prefix_tokens
        self._uncached_prompt_token_sum += sample.uncached_prompt_tokens
        if sample.completion_tokens is not None:
            self._completion_token_sum += sample.completion_tokens
            self._completion_token_count += 1
        self._total_token_sum += sample.total_tokens
        self._request_latency_sum += sample.request_latency_seconds
        if sample.decode_seconds is not None:
            self._decode_seconds_sum += sample.decode_seconds
            self._decode_seconds_count += 1
        self._active_program_sum += sample.active_programs
        self._waiting_program_sum += sample.waiting_programs
        self._inter_request_gap_sum += sample.inter_request_gap_seconds
        self._scheduler_queue_seconds_sum += sample.scheduler_queue_seconds
        self._rounds_since_ttl_pause_sum += sample.rounds_since_ttl_pause
        self._cache_churn_token_sum += sample.cache_churn_tokens

    def _remove_request_sample(self) -> None:
        sample = self._request_samples.popleft()
        self._prompt_token_sum -= sample.prompt_tokens
        self._cached_prefix_token_sum -= sample.cached_prefix_tokens
        self._uncached_prompt_token_sum -= sample.uncached_prompt_tokens
        if sample.completion_tokens is not None:
            self._completion_token_sum -= sample.completion_tokens
            self._completion_token_count -= 1
        self._total_token_sum -= sample.total_tokens
        self._request_latency_sum -= sample.request_latency_seconds
        if sample.decode_seconds is not None:
            self._decode_seconds_sum -= sample.decode_seconds
            self._decode_seconds_count -= 1
        self._active_program_sum -= sample.active_programs
        self._waiting_program_sum -= sample.waiting_programs
        self._inter_request_gap_sum -= sample.inter_request_gap_seconds
        self._scheduler_queue_seconds_sum -= sample.scheduler_queue_seconds
        self._rounds_since_ttl_pause_sum -= sample.rounds_since_ttl_pause
        self._cache_churn_token_sum -= sample.cache_churn_tokens

    def _evict_request_samples_to_target(self) -> None:
        while len(self._request_samples) > self.request_window_size:
            self._remove_request_sample()

    def _refresh_request_averages_if_ready(self) -> None:
        sample_count = len(self._request_samples)
        if sample_count < self._min_samples_for_update(self.request_window_size):
            return
        self.avg_prompt_tokens = self._prompt_token_sum / sample_count
        self.avg_cached_prefix_tokens = self._cached_prefix_token_sum / sample_count
        self.avg_uncached_prompt_tokens = self._uncached_prompt_token_sum / sample_count
        if self._completion_token_count > 0:
            self.avg_completion_tokens = self._completion_token_sum / self._completion_token_count
        self.avg_total_tokens = self._total_token_sum / sample_count
        self.avg_request_latency_seconds = self._request_latency_sum / sample_count
        if self._decode_seconds_count > 0:
            self.avg_decode_seconds = self._decode_seconds_sum / self._decode_seconds_count
        self.avg_active_programs = self._active_program_sum / sample_count
        self.avg_waiting_programs = self._waiting_program_sum / sample_count
        growth_samples = [
            sample.input_token_growth for sample in self._request_samples if sample.input_token_growth is not None
        ]
        if growth_samples:
            self.avg_input_token_growth_per_round = sum(growth_samples) / len(growth_samples)
        self.avg_cache_churn_tokens_per_round = self._cache_churn_token_sum / sample_count
        self.avg_inter_request_gap_seconds = self._inter_request_gap_sum / sample_count
        self.avg_scheduler_queue_seconds = self._scheduler_queue_seconds_sum / sample_count
        self.avg_rounds_since_ttl_pause = self._rounds_since_ttl_pause_sum / sample_count

    def _min_samples_for_update(self, window_size: int) -> int:
        return max(1, int(window_size * self.min_update_window_fraction))
