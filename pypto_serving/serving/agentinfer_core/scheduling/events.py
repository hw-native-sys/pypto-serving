# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Normalized lifecycle events and diagnostics emitted by scheduling runtimes.

``SchedulingEvent`` records a host runtime fact after a request or Program transition has been committed and supplies the
single input dispatched to strategy lifecycle callbacks. ``StrategyDiagnostic`` records how a policy calculated a
decision without pretending that calculation is a committed state change. Both are immutable, sink-neutral records;
this module performs no callback dispatch, serialization, I/O, policy decision, or runtime mutation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramRef, ProgramState, ProgramStatus

DiagnosticValue = str | int | float | bool | None
DiagnosticFields = tuple[tuple[str, DiagnosticValue], ...]


class SchedulingEventKind(str, Enum):
    """Logical boundaries emitted by the unified Scheduler runtime."""

    REQUEST_PENDING = "request_pending"
    REQUEST_ADMITTED = "request_admitted"
    REQUEST_FINISHED = "request_finished"
    PROGRAM_STATUS_CHANGED = "program_status_changed"
    PROGRAM_PAUSED = "program_paused"
    PROGRAM_MARKED_FOR_PAUSE = "program_marked_for_pause"
    PROGRAM_RESUMED = "program_resumed"
    PROGRAM_RELEASED = "program_released"
    CAPACITY_OBSERVED = "capacity_observed"
    STRATEGY_FACTORS_CHANGED = "strategy_factors_changed"


PROGRAM_SCOPED_EVENT_KINDS = frozenset(
    {
        SchedulingEventKind.REQUEST_PENDING,
        SchedulingEventKind.REQUEST_ADMITTED,
        SchedulingEventKind.REQUEST_FINISHED,
        SchedulingEventKind.PROGRAM_STATUS_CHANGED,
        SchedulingEventKind.PROGRAM_PAUSED,
        SchedulingEventKind.PROGRAM_MARKED_FOR_PAUSE,
        SchedulingEventKind.PROGRAM_RESUMED,
        SchedulingEventKind.PROGRAM_RELEASED,
    }
)


@dataclass(frozen=True)
class SchedulingEvent:
    """One immutable scheduling fact emitted after runtime state is committed.

    Lifecycle events may be dispatched to a strategy callback, while observation-only events are consumed only by observability sinks.

    Args:
        event_id: Integration-generated idempotency key.
        sequence: Monotonic event sequence within one runtime process.
        kind: Logical event boundary.
        occurred_at_monotonic_s: Monotonic event timestamp.
        reason: Stable machine-readable reason.
        schema_version: Version of the shared scheduling-event contract.
        program: Affected program generation, when applicable.
        previous_state: Core forwarding state before the event.
        current_state: Core forwarding state after the event.
        previous_status: Model-service status before the event.
        current_status: Model-service status after the event.
        fields: Typed scalar facts such as usage, score, or capacity values.
    """

    event_id: str
    sequence: int
    kind: SchedulingEventKind
    occurred_at_monotonic_s: float
    reason: str
    schema_version: int = 1
    program: ProgramRef | None = None
    previous_state: ProgramState | None = None
    current_state: ProgramState | None = None
    previous_status: ProgramStatus | None = None
    current_status: ProgramStatus | None = None
    fields: DiagnosticFields = ()
    _fields_by_name: Mapping[str, DiagnosticValue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate event identity, ordering fields, and diagnostic keys."""
        normalized_fields = tuple(self.fields)
        object.__setattr__(self, "fields", normalized_fields)
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.schema_version <= 0:
            raise ValueError("event schema_version must be positive")
        if self.sequence < 0 or not math.isfinite(self.occurred_at_monotonic_s) or self.occurred_at_monotonic_s < 0:
            raise ValueError("event sequence and timestamp must be finite and non-negative")
        if not self.reason:
            raise ValueError("event reason must not be empty")
        if self.kind in PROGRAM_SCOPED_EVENT_KINDS and self.program is None:
            raise ValueError("program-scoped event must reference an exact program generation")
        keys = [key for key, _ in normalized_fields]
        if len(keys) != len(set(keys)):
            raise ValueError("event fields contain duplicate keys")
        if any(isinstance(value, float) and not math.isfinite(value) for _, value in normalized_fields):
            raise ValueError("event numeric fields must be finite")
        object.__setattr__(self, "_fields_by_name", MappingProxyType(dict(normalized_fields)))

    def field(self, name: str) -> DiagnosticValue:
        """Return one scalar event field without rebuilding a lookup mapping."""
        return self._fields_by_name.get(name)


@dataclass(frozen=True)
class StrategyDiagnostic:
    """Inspectable strategy calculation associated with one Hook invocation."""

    name: str
    reason: str
    fields: DiagnosticFields = ()

    def __post_init__(self) -> None:
        """Validate stable names, reasons, and diagnostic keys."""
        normalized_fields = tuple(self.fields)
        object.__setattr__(self, "fields", normalized_fields)
        if not self.name or not self.reason:
            raise ValueError("diagnostic name and reason must not be empty")
        keys = [key for key, _ in normalized_fields]
        if len(keys) != len(set(keys)):
            raise ValueError("diagnostic fields contain duplicate keys")
        if any(isinstance(value, float) and not math.isfinite(value) for _, value in normalized_fields):
            raise ValueError("diagnostic numeric fields must be finite")
