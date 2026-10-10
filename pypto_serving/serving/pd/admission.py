# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fair request and byte admission for the fixed 1P1D data path."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from collections import deque
from contextlib import asynccontextmanager


class PDBackpressureError(RuntimeError):
    """The fixed P/D pair already has its configured number of waiters."""


class BoundedSerialAdmission:
    """Bound the active plus queued handoffs while preserving wire ordering.

    One control session has one ordered reply stream.  Concurrent transfers on
    that stream would need request-aware demultiplexing, so the first version
    deliberately serializes admitted handoffs.  The separate counter prevents
    an unbounded number of HTTP coroutines from accumulating behind the lock.
    """

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("PD pending handoff limit must be a positive integer")
        self.limit = limit
        self._count = 0
        self._state_lock = asyncio.Lock()
        self._serial_lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return self._count

    @asynccontextmanager
    async def admit(self):
        async with self._state_lock:
            if self._count >= self.limit:
                raise PDBackpressureError(
                    f"PD handoff backlog reached its configured limit ({self.limit})"
                )
            self._count += 1

        acquired = False
        try:
            await self._serial_lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                self._serial_lock.release()
            async with self._state_lock:
                self._count -= 1


class FairHandoffAdmission:
    """FIFO admission with independent active and total-request bounds.

    ``total_limit`` includes both active requests and queued waiters.  The
    explicit waiter queue makes cancellation and fairness observable instead
    of relying on implementation details of :class:`asyncio.Semaphore`.
    """

    def __init__(
        self,
        active_limit: int,
        total_limit: int,
        *,
        state_observer: Callable[[int, int], None] | None = None,
    ) -> None:
        if type(active_limit) is not int or active_limit < 1:
            raise ValueError("PD active handoff limit must be a positive integer")
        if type(total_limit) is not int or total_limit < active_limit:
            raise ValueError("PD total handoff limit must cover the active limit")
        self.active_limit = active_limit
        self.total_limit = total_limit
        self._active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._lock = asyncio.Lock()
        self._state_observer = state_observer

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        return len(self._waiters)

    @property
    def count(self) -> int:
        return self.active + self.queued

    def _wake_locked(self) -> None:
        while self._active < self.active_limit and self._waiters:
            waiter = self._waiters.popleft()
            if waiter.cancelled():
                continue
            self._active += 1
            waiter.set_result(None)

    def _observe_locked(self) -> None:
        if self._state_observer is not None:
            self._state_observer(self._active, len(self._waiters))

    @asynccontextmanager
    async def admit(self):
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        async with self._lock:
            if self.count >= self.total_limit:
                raise PDBackpressureError(
                    f"PD handoff backlog reached its configured limit ({self.total_limit})"
                )
            self._waiters.append(waiter)
            self._wake_locked()
            self._observe_locked()
        acquired = False
        try:
            await waiter
            acquired = True
            yield
        finally:
            async with self._lock:
                if not acquired:
                    try:
                        self._waiters.remove(waiter)
                    except ValueError:
                        # A woken task can be cancelled before ``await waiter``
                        # returns.  It already owns an active slot in that case.
                        if waiter.done() and not waiter.cancelled():
                            self._active -= 1
                else:
                    self._active -= 1
                self._wake_locked()
                self._observe_locked()
