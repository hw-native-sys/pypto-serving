# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Per-Program scheduling decision factors maintained by Progress-TTL.

Global rolling workload values live in ``rolling_stats``. This module retains segment progress, cache-recency and wait
deadlines, TTL deadlines, and bounded privilege used by admission, resume, pause, handoff, and release decisions without
replacing the Program lifecycle facts owned by the Scheduler runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressTTLProgramFactors:
    """Decision factors for one exact Program generation's progress, TTL, privilege, recency, and bounded waiting."""

    segment_served_rounds: int = 0
    segment_started_at_monotonic_s: float | None = None
    segment_prompt_tokens: int = 0
    segment_completion_tokens: int = 0
    segment_first_request_pending: bool = True
    last_request_prompt_tokens: int = 0
    last_request_context_tokens: int = 0
    lifetime_generated_tokens: int = 0
    segment_share_tokens: int = 0
    rounds_since_ttl_pause: int = 0
    previous_ttl_seconds: float | None = None
    last_segment_served_rounds: int = 0
    last_segment_prompt_tokens: int = 0
    last_segment_completion_tokens: int = 0
    is_evictable_after_min_rounds: bool = False
    is_privileged: bool = False
    privilege_reason: str | None = None
    privilege_deadline_monotonic_s: float | None = None
    privilege_ttl_expired: bool = False
    wait_started_at_monotonic_s: float | None = None
    request_wait_started_at_monotonic_s: float | None = None
    force_resume_timeout_seconds: float | None = None
    force_resume_deadline_monotonic_s: float | None = None
    force_resume_active_remaining_rounds: float = 0.0
    force_resume_pool_remaining_rounds: float = 0.0
    force_resume_request_throughput_per_second: float = 0.0
    last_request_finished_at_monotonic_s: float | None = None
    acting_since_monotonic_s: float | None = None
    ttl_deadline_monotonic_s: float | None = None
    ttl_expiry_observed: bool = False
    release_deadline_monotonic_s: float | None = None
    last_pause_at_monotonic_s: float | None = None
    last_resume_at_monotonic_s: float | None = None
    pause_reason: str | None = None
    resume_reason: str | None = None
    last_inter_request_gap_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate counters, request gap, and optional monotonic timestamps."""
        counters = (
            self.segment_served_rounds,
            self.segment_prompt_tokens,
            self.segment_completion_tokens,
            self.last_request_prompt_tokens,
            self.last_request_context_tokens,
            self.lifetime_generated_tokens,
            self.segment_share_tokens,
            self.rounds_since_ttl_pause,
            self.last_segment_served_rounds,
            self.last_segment_prompt_tokens,
            self.last_segment_completion_tokens,
        )
        if min(counters) < 0:
            raise ValueError("Progress-TTL Program counters must be non-negative")
        timestamps = (
            self.segment_started_at_monotonic_s,
            self.wait_started_at_monotonic_s,
            self.request_wait_started_at_monotonic_s,
            self.force_resume_deadline_monotonic_s,
            self.last_request_finished_at_monotonic_s,
            self.acting_since_monotonic_s,
            self.ttl_deadline_monotonic_s,
            self.release_deadline_monotonic_s,
            self.privilege_deadline_monotonic_s,
            self.last_pause_at_monotonic_s,
            self.last_resume_at_monotonic_s,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in timestamps):
            raise ValueError("Progress-TTL timestamps must be finite and non-negative")
        if self.previous_ttl_seconds is not None and (
            not math.isfinite(self.previous_ttl_seconds) or self.previous_ttl_seconds < 0
        ):
            raise ValueError("previous_ttl_seconds must be finite and non-negative when provided")
        if self.force_resume_timeout_seconds is not None and (
            not math.isfinite(self.force_resume_timeout_seconds) or self.force_resume_timeout_seconds < 0
        ):
            raise ValueError("force_resume_timeout_seconds must be finite and non-negative when provided")
        force_resume_values = (
            self.force_resume_active_remaining_rounds,
            self.force_resume_pool_remaining_rounds,
            self.force_resume_request_throughput_per_second,
        )
        if any(not math.isfinite(value) or value < 0 for value in force_resume_values):
            raise ValueError("force-resume workload estimates must be finite and non-negative")
        boolean_factors = (
            self.segment_first_request_pending,
            self.ttl_expiry_observed,
            self.privilege_ttl_expired,
        )
        if not all(isinstance(value, bool) for value in boolean_factors):
            raise ValueError("Progress-TTL boolean factors must be booleans")
        if not math.isfinite(self.last_inter_request_gap_seconds) or self.last_inter_request_gap_seconds < 0:
            raise ValueError("last_inter_request_gap_seconds must be finite and non-negative")
