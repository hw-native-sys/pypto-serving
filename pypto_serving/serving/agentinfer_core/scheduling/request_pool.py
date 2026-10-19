# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
# Modified by PyPTO Serving: internal imports use the bundled scheduling package.
"""Scheduler-owned request retention and admission-result storage.

``RequestPool`` contains no asyncio primitive and no engine type. host adapters may attach a Future as the retained
object, while embedded engine adapters retain the native request. Program lifecycle remains represented by
``ProgramRef`` rather than by a second Program waiting state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from pypto_serving.serving.agentinfer_core.scheduling.backend import DispatchTarget
from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramRef

RetainedRequestT = TypeVar("RetainedRequestT")


class RequestPoolStatus(str, Enum):
    """Execution status of one retained request attempt."""

    WAITING = "waiting"
    ADMITTED = "admitted"


@dataclass
class RequestPoolEntry(Generic[RetainedRequestT]):
    """One request attempt retained before downstream admission."""

    request_id: str
    program: ProgramRef | None
    arrived_at_monotonic_s: float
    retained_request: RetainedRequestT
    status: RequestPoolStatus = RequestPoolStatus.WAITING
    target: DispatchTarget | None = None

    def __post_init__(self) -> None:
        """Validate request identity and arrival time."""
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not math.isfinite(self.arrived_at_monotonic_s) or self.arrived_at_monotonic_s < 0:
            raise ValueError("request arrival time must be finite and non-negative")


class RequestPool(Generic[RetainedRequestT]):
    """Insertion-ordered request attempts and one-shot admission decisions."""

    def __init__(self) -> None:
        self._entries: dict[str, RequestPoolEntry[RetainedRequestT]] = {}
        self._recent_admit: dict[str, DispatchTarget] = {}

    @property
    def waiting_request_ids(self) -> tuple[str, ...]:
        """Return waiting attempts in arrival order."""
        return tuple(
            request_id for request_id, entry in self._entries.items() if entry.status is RequestPoolStatus.WAITING
        )

    @property
    def unfinished_count(self) -> int:
        """Return retained attempts not yet transferred to the host."""
        return len(self._entries)

    def add(self, entry: RequestPoolEntry[RetainedRequestT]) -> None:
        """Retain a new unique request attempt."""
        if entry.request_id in self._entries:
            raise ValueError(f"duplicate request_id: {entry.request_id}")
        self._entries[entry.request_id] = entry

    def get(self, request_id: str) -> RequestPoolEntry[RetainedRequestT] | None:
        """Return one retained attempt without mutation."""
        return self._entries.get(request_id)

    @property
    def entries(self) -> tuple[RequestPoolEntry[RetainedRequestT], ...]:
        """Return retained attempts in arrival order for snapshot construction."""
        return tuple(self._entries.values())

    def admit(self, request_id: str, target: DispatchTarget) -> bool:
        """Commit one admission and expose its one-shot request-to-target decision."""
        entry = self._entries.get(request_id)
        if entry is None or entry.status is RequestPoolStatus.ADMITTED:
            return False
        entry.status = RequestPoolStatus.ADMITTED
        entry.target = target
        self._recent_admit[request_id] = target
        return True

    def consume_admitted(self) -> tuple[tuple[RequestPoolEntry[RetainedRequestT], DispatchTarget], ...]:
        """Remove and return all newly admitted attempts in decision order."""
        admitted: list[tuple[RequestPoolEntry[RetainedRequestT], DispatchTarget]] = []
        for request_id, target in tuple(self._recent_admit.items()):
            entry = self._entries.pop(request_id, None)
            self._recent_admit.pop(request_id, None)
            if entry is not None:
                admitted.append((entry, target))
        return tuple(admitted)

    def consume_admitted_request(
        self,
        request_id: str,
    ) -> tuple[RequestPoolEntry[RetainedRequestT], DispatchTarget] | None:
        """Remove one immediately admitted attempt before the host inserts it natively."""
        target = self._recent_admit.pop(request_id, None)
        if target is None:
            return None
        entry = self._entries.pop(request_id, None)
        return (entry, target) if entry is not None else None

    def cancel(self, request_id: str) -> RequestPoolEntry[RetainedRequestT] | None:
        """Remove a cancelled or timed-out retained request attempt."""
        self._recent_admit.pop(request_id, None)
        return self._entries.pop(request_id, None)
