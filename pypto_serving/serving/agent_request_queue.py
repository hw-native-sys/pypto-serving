# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retain requests until the admission policy hands them to the native scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from pypto_serving.serving.agent_policy_state import ProgramRef

RetainedRequestT = TypeVar("RetainedRequestT")


@dataclass(frozen=True)
class QueuedRequest(Generic[RetainedRequestT]):
    request_id: str
    program: ProgramRef
    arrived_at_monotonic_s: float
    retained_request: RetainedRequestT

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        arrival = self.arrived_at_monotonic_s
        if arrival < 0 or not math.isfinite(arrival):
            raise ValueError("request arrival time must be finite and non-negative")


class AgentRequestQueue(Generic[RetainedRequestT]):
    """Pair retained requests with dispatch decisions that can be consumed once."""

    def __init__(self) -> None:
        self._requests: dict[str, QueuedRequest[RetainedRequestT]] = {}
        self._ready: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._requests)

    @property
    def waiting_count(self) -> int:
        return len(self._requests) - len(self._ready)

    @property
    def entries(self) -> tuple[QueuedRequest[RetainedRequestT], ...]:
        return tuple(self._requests.values())

    def append(self, request: QueuedRequest[RetainedRequestT]) -> None:
        request_id = request.request_id
        if request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request_id}")
        self._requests[request_id] = request

    def waiting(self, request_id: str) -> QueuedRequest[RetainedRequestT] | None:
        return None if request_id in self._ready else self._requests.get(request_id)

    def grant(self, request_id: str, target: str) -> None:
        if request_id in self._requests:
            self._ready.setdefault(request_id, target)

    def take(self, request_id: str) -> tuple[QueuedRequest[RetainedRequestT], str] | None:
        target = self._ready.pop(request_id, None)
        if target is None:
            return None
        return self._requests.pop(request_id), target

    def drain(self) -> tuple[tuple[QueuedRequest[RetainedRequestT], str], ...]:
        ready = self._ready
        self._ready = {}
        return tuple((self._requests.pop(request_id), target) for request_id, target in ready.items())

    def cancel(self, request_id: str) -> QueuedRequest[RetainedRequestT] | None:
        self._ready.pop(request_id, None)
        return self._requests.pop(request_id, None)
