# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import pytest

from pypto_serving.serving.agentinfer_core.scheduling import (
    ProgramRef, ProgramState, ProgramStatus, ProgramTokenObservation, ProgramView,
    SchedulingEvent, SchedulingEventKind, SchedulingSnapshot, TransitionController,
    TransitionKind, TransitionRequest, TransitionResult,
)
from pypto_serving.serving.agentinfer_core.scheduling.progress_ttl import (
    ProgressTTLConfig, ProgressTTLProgramFactors, build_progress_ttl_strategy,
)


@pytest.mark.parametrize(
    ("headroom", "capacity", "preserve_cache", "uncached_head", "expected_resume"),
    [
        (0, 1171, False, None, True),
        (512, 1171, False, None, False),
        (512, 1172, False, None, True),
        (0, 1171, True, None, False),
        (0, 1171, True, 199, False),
        (0, 1171, True, 200, True),
    ],
)
def test_wait_relax_requires_extra_physical_headroom(
    monkeypatch, headroom: int, capacity: int, preserve_cache: bool,
    uncached_head: int | None, expected_resume: bool,
) -> None:
    components = build_progress_ttl_strategy(ProgressTTLConfig(
        decode_buffer_tokens=0,
        resume_capacity_ratio=0.7,
        resume_wait_relax_seconds=8,
        resume_wait_growth_reserve_fraction=0.1,
        resume_wait_extra_headroom_tokens=headroom,
        resume_wait_preserve_cached_blocks=preserve_cache,
        enable_batch_gain_admission=False,
    ))
    strategy, factors = components.strategy, components.initial_factors
    monkeypatch.setattr(strategy, "_continuous_growth_reserve_tokens", lambda *args, **kwargs: 600)
    active = ProgramView(
        ref=ProgramRef("active", 0), state=ProgramState.ACTIVE, status=ProgramStatus.REASONING,
        tokens=ProgramTokenObservation(estimated_context_tokens=400), backend_id="backend",
    )
    waiter = ProgramView(
        ref=ProgramRef("waiter", 0), state=ProgramState.PAUSED, status=ProgramStatus.REASONING,
        tokens=ProgramTokenObservation(estimated_context_tokens=200),
    )
    factors.set_program_factors(
        waiter.ref, ProgressTTLProgramFactors(request_wait_started_at_monotonic_s=90.0),
    )
    snapshot = SchedulingSnapshot(
        observed_at_monotonic_s=100.0, backend_id="backend", total_kv_tokens=capacity,
        programs=(active, waiter), waiting_programs=(waiter.ref,),
        uncached_free_head_kv_tokens=uncached_head,
    )
    calls: list[TransitionRequest] = []

    def apply(request: TransitionRequest) -> TransitionResult:
        calls.append(request)
        assert request.kind is TransitionKind.RESUME
        return TransitionResult(
            request.kind, request.program, True, request.reason,
            SchedulingEvent(
                event_id="resume", sequence=1, kind=SchedulingEventKind.PROGRAM_RESUMED,
                occurred_at_monotonic_s=100.0, reason=request.reason, program=request.program,
            ),
        )

    strategy.schedule_resume(snapshot, factors, TransitionController(apply))

    assert [request.reason for request in calls] == (
        ["progress_ttl_wait_capacity_relax"] if expected_resume else []
    )


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_wait_headroom_rejects_non_integer_or_negative_values(value) -> None:
    with pytest.raises(ValueError, match="resume_wait_extra_headroom_tokens"):
        ProgressTTLConfig(resume_wait_relax_seconds=8, resume_wait_extra_headroom_tokens=value)


def test_wait_headroom_requires_wait_relax() -> None:
    with pytest.raises(ValueError, match="requires resume_wait_relax_seconds"):
        ProgressTTLConfig(resume_wait_extra_headroom_tokens=512)


def test_wait_cache_protection_requires_wait_relax() -> None:
    with pytest.raises(ValueError, match="requires resume_wait_relax_seconds"):
        ProgressTTLConfig(resume_wait_preserve_cached_blocks=True)
