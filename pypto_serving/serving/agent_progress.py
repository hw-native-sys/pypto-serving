# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Track the scheduling progress and timers of each live agent session."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pypto_serving.serving.agent_policy_state import ProgramRef
from pypto_serving.serving.agent_ttl import AgentTTLStatistics


@dataclass(frozen=True)
class ProgressTTLProgramFactors:
    # Active service and previous segment history.
    segment_started_at_monotonic_s: float | None = None
    segment_served_rounds: int = 0
    segment_prompt_tokens: int = 0
    segment_completion_tokens: int = 0
    segment_first_request_pending: bool = True
    is_evictable_after_min_rounds: bool = False
    last_segment_served_rounds: int = 0
    last_segment_prompt_tokens: int = 0
    last_segment_completion_tokens: int = 0
    lifetime_generated_tokens: int = 0

    # Observed continuity and the latest idle reservation.
    last_request_finished_at_monotonic_s: float | None = None
    last_inter_request_gap_seconds: float = 0.0
    last_request_context_tokens: int = 0
    last_request_prompt_tokens: int = 0
    segment_share_tokens: int = 0
    rounds_since_ttl_pause: int = 0
    previous_ttl_seconds: float | None = None
    acting_since_monotonic_s: float | None = None
    ttl_deadline_monotonic_s: float | None = None
    ttl_expiry_observed: bool = False
    release_deadline_monotonic_s: float | None = None

    # Queue deadlines and recent placement changes.
    wait_started_at_monotonic_s: float | None = None
    request_wait_started_at_monotonic_s: float | None = None
    force_resume_deadline_monotonic_s: float | None = None
    force_resume_timeout_seconds: float | None = None
    force_resume_request_throughput_per_second: float = 0.0
    force_resume_active_remaining_rounds: float = 0.0
    force_resume_pool_remaining_rounds: float = 0.0
    last_resume_at_monotonic_s: float | None = None
    resume_reason: str | None = None
    last_pause_at_monotonic_s: float | None = None
    pause_reason: str | None = None

    # One expiring priority slot shared by related sessions.
    is_privileged: bool = False
    privilege_deadline_monotonic_s: float | None = None
    privilege_ttl_expired: bool = False
    privilege_reason: str | None = None


class ProgressTTLFactors:
    def __init__(self) -> None:
        self.global_factors = AgentTTLStatistics()
        self._active: dict[ProgramRef, ProgressTTLProgramFactors] = {}

    @property
    def program_factors(self) -> tuple[tuple[ProgramRef, ProgressTTLProgramFactors], ...]:
        return tuple(self._active.items())

    def get_or_create(self, ref: ProgramRef) -> ProgressTTLProgramFactors:
        profile = self._active.get(ref)
        if profile is None:
            profile = self._active[ref] = ProgressTTLProgramFactors()
        return profile

    def for_program(self, ref: ProgramRef) -> ProgressTTLProgramFactors | None:
        return self._active.get(ref)

    def update(self, ref: ProgramRef, **fields) -> None:
        self._active[ref] = replace(self._active[ref], **fields)

    def set_program_factors(self, ref: ProgramRef, profile: ProgressTTLProgramFactors) -> None:
        self._active[ref] = profile

    def remove_program_factors(self, ref: ProgramRef) -> None:
        self._active.pop(ref, None)
