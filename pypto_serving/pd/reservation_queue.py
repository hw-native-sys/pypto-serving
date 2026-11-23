# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Order pending Router requests when Decode reservation is temporarily full."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from itertools import takewhile

from pypto_serving.router.http import RouterError


RETRY_INTERVAL = 1.0


@dataclass(eq=False)
class ReservationWaiter:
    position: int = 0
    generation: int = 0
    retry_at: float = 0.0
    queued: bool = False
    ready: asyncio.Event = field(default_factory=asyncio.Event)


class ReservationQueue:
    def __init__(self, capacity, first_position):
        self.capacity = capacity
        self.first_position = first_position
        self.pending = deque()
        self.protected = None
        self.generation = 0
        self._next = self._timer = None

    def __bool__(self):
        return bool(self.pending) or self.protected is not None

    def put(self, waiter, *, generation=None, rejected=False):
        protected = self.protected is waiter
        if not protected and len(self.pending) + (self.protected is not None) >= self.capacity:
            raise RouterError(503, "capacity_exhausted", "Decode reservation wait queue is full")
        if rejected:
            waiter.position = max(1, waiter.position - 1) if waiter.position else self.first_position
        waiter.generation = self.generation if generation is None else generation
        waiter.retry_at = 0 if generation is None else asyncio.get_running_loop().time() + RETRY_INTERVAL
        waiter.queued = True
        if protected:
            self.protected = None
            self.pending.appendleft(waiter)
        elif waiter.position:
            protected_count = sum(1 for _ in takewhile(lambda entry: entry.position == 1, self.pending))
            self.pending.insert(max(protected_count, waiter.position - 1), waiter)
        else:
            self.pending.append(waiter)
        self._wake_next()

    def released(self):
        self.generation += 1
        if self.pending and self.protected is None:
            self._wake_next()

    def take(self, waiter):
        if waiter is not self._next:
            return False
        self.pending.remove(waiter)
        waiter.queued = False
        if waiter.position == 1:
            self.protected = waiter
        self._wake_next()
        return True

    def leave(self, waiter):
        if self.protected is waiter:
            self.protected = None
        elif waiter.queued:
            self.pending.remove(waiter)
            waiter.queued = False
        else:
            return
        self._wake_next()

    def _wake_next(self):
        if self._timer is not None:
            self._timer.cancel()
        self._next = self._timer = None
        if self.protected is not None or not self.pending:
            return
        loop = asyncio.get_running_loop()
        now = loop.time()
        retry_at = float("inf")
        for waiter in self.pending:
            if waiter.generation != self.generation or waiter.retry_at <= now:
                self._next = waiter
                waiter.ready.set()
                return
            retry_at = min(retry_at, waiter.retry_at)
            if waiter.position == 1:
                break
        self._timer = loop.call_at(retry_at, self._wake_next)
