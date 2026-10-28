# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Typed boundary for strategy-requested, runtime-owned Program transitions.

``TransitionRequest`` describes an admit, queue, pause, delayed pause, resume, or release against an exact
generation-safe ``ProgramRef``. ``TransitionResult`` reports whether host runtime accepted it and links the matching
event. ``TransitionController`` is constructed for one synchronous Strategy Hook while the caller holds the scheduling
lock. This module does not select candidates, calculate capacity, acquire locks, or implement the Program state
machine: the host runtime callback validates current state and performs the mutation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramRef
from pypto_serving.serving.agentinfer_core.scheduling.events import SchedulingEvent, SchedulingEventKind


class TransitionKind(str, Enum):
    """Core transitions validated and executed by the host runtime."""

    ADMIT = "admit"
    QUEUE = "queue"
    PAUSE = "pause"
    MARK_FOR_PAUSE = "mark_for_pause"
    RESUME = "resume"
    RELEASE = "release"


TRANSITION_EVENT_KINDS = {
    TransitionKind.ADMIT: SchedulingEventKind.REQUEST_ADMITTED,
    TransitionKind.QUEUE: SchedulingEventKind.REQUEST_PENDING,
    TransitionKind.PAUSE: SchedulingEventKind.PROGRAM_PAUSED,
    TransitionKind.MARK_FOR_PAUSE: SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE,
    TransitionKind.RESUME: SchedulingEventKind.PROGRAM_RESUMED,
    TransitionKind.RELEASE: SchedulingEventKind.PROGRAM_RELEASED,
}


@dataclass(frozen=True)
class TransitionRequest:
    """One core transition requested by a strategy.

    Args:
        kind: Program transition to apply.
        program: Exact Program generation observed by the strategy.
        reason: Stable machine-readable decision reason.
        backend_id: Target backend for admission or resume.
    """

    kind: TransitionKind
    program: ProgramRef
    reason: str
    backend_id: str | None = None

    def __post_init__(self) -> None:
        """Validate reason and backend requirements."""
        if not self.reason:
            raise ValueError("transition reason must not be empty")
        needs_backend = self.kind in (TransitionKind.ADMIT, TransitionKind.RESUME)
        if needs_backend and not self.backend_id:
            raise ValueError("admit and resume transitions require backend_id")
        if not needs_backend and self.backend_id is not None:
            raise ValueError("backend_id is valid only for admit and resume transitions")


@dataclass(frozen=True)
class TransitionResult:
    """Result returned after the host runtime validates one requested transition.

    Args:
        kind: Requested transition kind.
        program: Exact Program generation targeted by the request.
        applied: Whether the runtime committed the transition.
        reason: Stable machine-readable result reason.
        event: Committed scheduling event when a transition was applied.
    """

    kind: TransitionKind
    program: ProgramRef
    applied: bool
    reason: str
    event: SchedulingEvent | None = None

    def __post_init__(self) -> None:
        """Validate result identity and any attached lifecycle event."""
        if not self.reason:
            raise ValueError("transition reason must not be empty")
        if self.applied != (self.event is not None):
            raise ValueError("only an applied transition may carry its committed event")
        if self.event is not None and self.event.program != self.program:
            raise ValueError("transition event must reference the transitioned program generation")
        if self.event is not None and self.event.kind is not TRANSITION_EVENT_KINDS[self.kind]:
            raise ValueError("transition event kind must match the applied transition")


class TransitionController:
    """Lock-scoped strategy capability bound to the host runtime transition function.

    Args:
        apply_transition: host runtime callback that validates and commits one typed request.
    """

    def __init__(self, apply_transition: Callable[[TransitionRequest], TransitionResult]) -> None:
        self._apply_transition = apply_transition
        self._applied_requests: list[TransitionRequest] = []

    def apply(self, request: TransitionRequest) -> TransitionResult:
        """Apply a typed transition and reject a mismatched runtime result.

        Args:
            request: Exact Program transition requested by a strategy Hook.
        """
        result = self._apply_transition(request)
        if result.kind is not request.kind or result.program != request.program:
            raise ValueError("transition result does not match its request")
        if result.applied:
            self._applied_requests.append(request)
        return result

    def take_applied_requests(self) -> tuple[TransitionRequest, ...]:
        """Return and clear the accepted transition ledger since the previous call."""
        requests = tuple(self._applied_requests)
        self._applied_requests.clear()
        return requests

    def admit(self, program: ProgramRef, *, reason: str, backend_id: str) -> TransitionResult:
        """Request admission of a Program to the configured backend."""
        return self._request(TransitionKind.ADMIT, program, reason, backend_id=backend_id)

    def queue(self, program: ProgramRef, *, reason: str) -> TransitionResult:
        """Request placement of a Program in the waiting pool."""
        return self._request(TransitionKind.QUEUE, program, reason)

    def pause(self, program: ProgramRef, *, reason: str) -> TransitionResult:
        """Request immediate pause of an eligible acting Program."""
        return self._request(TransitionKind.PAUSE, program, reason)

    def mark_for_pause(self, program: ProgramRef, *, reason: str) -> TransitionResult:
        """Request delayed pause of an in-flight reasoning Program."""
        return self._request(TransitionKind.MARK_FOR_PAUSE, program, reason)

    def resume(self, program: ProgramRef, *, reason: str, backend_id: str) -> TransitionResult:
        """Request resume of a paused Program on the configured backend."""
        return self._request(TransitionKind.RESUME, program, reason, backend_id=backend_id)

    def release(self, program: ProgramRef, *, reason: str) -> TransitionResult:
        """Request terminal release of a Program."""
        return self._request(TransitionKind.RELEASE, program, reason)

    def _request(
        self,
        kind: TransitionKind,
        program: ProgramRef,
        reason: str,
        *,
        backend_id: str | None = None,
    ) -> TransitionResult:
        """Construct and apply one request shared by the public typed helpers."""
        return self.apply(TransitionRequest(kind=kind, program=program, reason=reason, backend_id=backend_id))
