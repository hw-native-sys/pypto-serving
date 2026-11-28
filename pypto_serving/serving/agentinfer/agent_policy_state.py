# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""State exchanged between replica admission and the adaptive policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class ProgramState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


class ProgramStatus(str, Enum):
    ACTING = "acting"
    REASONING = "reasoning"


class ProgramLifecycle(str, Enum):
    UNKNOWN = "unknown"
    CONTINUE = "continue"
    TERMINAL = "terminal"


class SharedPrefixAttribution(str, Enum):
    UNKNOWN = "unknown"
    PARTIAL = "partial"
    EXACT = "exact"


class TokenObservationSource(str, Enum):
    ROUTER_ESTIMATE = "router_estimate"
    MIXED = "mixed"


@dataclass(frozen=True, order=True)
class ProgramRef:
    program_id: str
    generation: int


@dataclass(frozen=True)
class ProgramTokenObservation:
    estimated_context_tokens: int
    shared_prefix_tokens: int = 0
    shared_prefix_fresh_until_monotonic_s: float | None = None
    shared_prefix_attribution: SharedPrefixAttribution = SharedPrefixAttribution.UNKNOWN
    source: TokenObservationSource = TokenObservationSource.ROUTER_ESTIMATE
    observed_at_monotonic_s: float | None = None


@dataclass(frozen=True)
class ProgramView:
    ref: ProgramRef
    tokens: ProgramTokenObservation
    state: ProgramState
    status: ProgramStatus
    backend_id: str | None = None
    task_id: str | None = None
    parent_program_id: str | None = None
    blocked_on_child: bool = False
    expected_resume: bool = True
    step_count: int = 0
    wait_started_at_monotonic_s: float | None = None
    marked_for_pause: bool = False


@dataclass(frozen=True)
class SchedulingSnapshot:
    observed_at_monotonic_s: float
    backend_id: str
    total_kv_tokens: int | None
    programs: tuple[ProgramView, ...]
    waiting_programs: tuple[ProgramRef, ...]
    native_used_kv_tokens: int | None = None
    native_waiting_kv_tokens: int | None = None
    uncached_free_head_kv_tokens: int | None = None
    request_pool_waiting_requests: int = 0
    _sessions: dict[ProgramRef, ProgramView] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sessions = {session.ref: session for session in self.programs}
        if len(sessions) != len(self.programs) or not sessions.keys() >= set(self.waiting_programs):
            raise ValueError("inconsistent live and waiting session references")
        object.__setattr__(self, "_sessions", sessions)

    def program(self, ref: ProgramRef) -> ProgramView | None:
        return self._sessions.get(ref)


class AdmissionDisposition(str, Enum):
    QUEUED = "queued"
    ADMITTED = "admitted"


@dataclass(frozen=True)
class AdmissionOutcome:
    disposition: AdmissionDisposition
    reason: str
    required_tokens: int = 0
    reserve_tokens: int = 0
    deficit_tokens: int = 0
    transitioned_programs: tuple[ProgramRef, ...] = ()


class SchedulingEventKind(str, Enum):
    REQUEST_PENDING = "request_pending"
    REQUEST_ADMITTED = "request_admitted"
    REQUEST_FINISHED = "request_finished"
    PROGRAM_STATUS_CHANGED = "program_status_changed"
    PROGRAM_PAUSED = "program_paused"
    PROGRAM_MARKED_FOR_PAUSE = "program_marked_for_pause"
    PROGRAM_RESUMED = "program_resumed"
    PROGRAM_RELEASED = "program_released"


DiagnosticValue = str | int | float | bool | None


@dataclass(frozen=True)
class SchedulingEvent:
    kind: SchedulingEventKind
    occurred_at_monotonic_s: float
    reason: str
    program: ProgramRef
    previous_state: ProgramState | None = None
    current_state: ProgramState | None = None
    previous_status: ProgramStatus | None = None
    current_status: ProgramStatus | None = None
    fields: dict[str, DiagnosticValue] = field(default_factory=dict)

    def field(self, name: str) -> DiagnosticValue:
        return self.fields.get(name)


class TransitionKind(str, Enum):
    ADMIT = "admit"
    RESUME = "resume"
    QUEUE = "queue"
    PAUSE = "pause"
    MARK_FOR_PAUSE = "mark_for_pause"
    RELEASE = "release"

    @property
    def event(self) -> SchedulingEventKind:
        return {
            TransitionKind.ADMIT: SchedulingEventKind.REQUEST_ADMITTED,
            TransitionKind.RESUME: SchedulingEventKind.PROGRAM_RESUMED,
            TransitionKind.QUEUE: SchedulingEventKind.REQUEST_PENDING,
            TransitionKind.PAUSE: SchedulingEventKind.PROGRAM_PAUSED,
            TransitionKind.MARK_FOR_PAUSE: SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE,
            TransitionKind.RELEASE: SchedulingEventKind.PROGRAM_RELEASED,
        }[self]


@dataclass(frozen=True)
class TransitionRequest:
    kind: TransitionKind
    program: ProgramRef
    reason: str
    backend_id: str | None = None


@dataclass(frozen=True)
class TransitionResult:
    kind: TransitionKind
    program: ProgramRef
    applied: bool
    reason: str


class TransitionController:
    """Report which live placements changed during a single policy pass."""

    def __init__(self, apply_transition: Callable[[TransitionRequest], TransitionResult]) -> None:
        self._apply = apply_transition
        self._changed: set[ProgramRef] = set()

    def take_changed_sessions(self) -> set[ProgramRef]:
        changed, self._changed = self._changed, set()
        return changed

    def _change(
        self,
        kind: TransitionKind,
        ref: ProgramRef,
        reason: str,
        target: str | None = None,
    ) -> TransitionResult:
        result = self._apply(TransitionRequest(kind, ref, reason, target))
        if result.applied and kind is not TransitionKind.QUEUE:
            self._changed.add(ref)
        return result

    def admit(self, program: ProgramRef, *, reason: str, backend_id: str) -> TransitionResult:
        return self._change(TransitionKind.ADMIT, program, reason, backend_id)

    def resume(self, program: ProgramRef, *, reason: str, backend_id: str) -> TransitionResult:
        return self._change(TransitionKind.RESUME, program, reason, backend_id)

    def queue(self, program: ProgramRef, *, reason: str) -> TransitionResult:
        return self._change(TransitionKind.QUEUE, program, reason)

    def pause(self, program: ProgramRef, *, reason: str) -> TransitionResult:
        return self._change(TransitionKind.PAUSE, program, reason)

    def mark_for_pause(self, program: ProgramRef, *, reason: str) -> TransitionResult:
        return self._change(TransitionKind.MARK_FOR_PAUSE, program, reason)

    def release(self, program: ProgramRef, *, reason: str) -> TransitionResult:
        return self._change(TransitionKind.RELEASE, program, reason)
