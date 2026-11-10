# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-writer DDR residency, checkpoint membership and transfer references."""

from collections import deque
from dataclasses import dataclass, field
from itertools import count
import time

from .policy import LRU


@dataclass(eq=False)
class Page:
    key: tuple
    handle: int
    size: int
    source: object
    lease: object
    charge: int
    future: object = None
    ready: bool = False
    readers: int = 0
    units: set = field(default_factory=set)


class CacheTier:
    def __init__(self, capacity, backend, release_pin, *, policy=None, clock=time.monotonic):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("DDR capacity must be positive")
        self.capacity, self.backend, self.release_pin = capacity, backend, release_pin
        self.policy, self.clock = policy if policy is not None else LRU(), clock
        self.pages, self.units = {}, {}
        self.pending_units, self.pending_by_page = {}, {}
        self.by_handle, self.inflight = {}, {}
        self.queued = deque()
        self.retiring = {}
        self.sequence = count(1)
        self.used = 0
        self.error = None
        self.closing = False
        self.counters = dict(stored_bytes=0, loaded_bytes=0, evicted_bytes=0, stores=0, loads=0)

    def capture(self, key, source, size, lease, *, charge=None):
        if self.closing or self.error:
            self.release_pin(lease)
            raise RuntimeError("DDR tier is not accepting publications")
        if key in self.pages:
            self.release_pin(lease)
            self.touch(key)
            return
        charge = size if charge is None else charge
        if (type(size) is not int or type(charge) is not int
                or not 0 < size <= charge <= self.capacity):
            self.release_pin(lease)
            raise ValueError("a cache page exceeds the DDR capacity")
        page = Page(key, next(self.sequence), size, source, lease, charge)
        self.pages[key] = page
        self.by_handle[page.handle] = page
        self.queued.append(page)

    def publish(self, identity, keys):
        """A unit is the complete logical KV/state recovery set, not a physical allocation."""
        if identity in self.units or identity in self.pending_units:
            return
        keys = tuple(dict.fromkeys(keys))
        if not keys or any(key not in self.pages for key in keys):
            return
        if not all(self.pages[key].ready for key in keys):
            self.pending_units[identity] = keys
            for key in keys:
                self.pending_by_page.setdefault(key, set()).add(identity)
            return
        self._publish_ready(identity, keys)

    def _forget_pending(self, identity):
        for key in self.pending_units.pop(identity):
            awaiting = self.pending_by_page[key]
            awaiting.remove(identity)
            if not awaiting:
                del self.pending_by_page[key]

    def _publish_ready(self, identity, keys):
        if identity in self.pending_units:
            self._forget_pending(identity)
        self.units[identity] = keys
        for key in keys:
            page = self.pages[key]
            self.policy.on_remove(("page", page.handle))
            page.units.add(identity)
        self.policy.on_publish(("unit", identity), self.clock())

    def lookup(self, key):
        page = self.pages.get(key)
        if page is None or not page.ready:
            return None
        return page

    def touch(self, key):
        page = self.lookup(key)
        if page is not None:
            for identity in page.units:
                self.policy.on_access(("unit", identity), self.clock())
            if not page.units:
                self.policy.on_access(("page", page.handle), self.clock())

    def acquire(self, keys):
        pages = tuple(self.lookup(key) for key in dict.fromkeys(keys))
        if self.closing or self.error or any(page is None for page in pages):
            raise ValueError("DDR content is unavailable")
        for page in pages:
            page.readers += 1
            self.touch(page.key)
        return pages

    def release(self, pages):
        for page in pages:
            if self.pages.get(page.key) is not page or page.readers <= 0:
                raise RuntimeError("stale DDR reader")
        for page in pages:
            page.readers -= 1

    def load(self, page, destination):
        if self.pages.get(page.key) is not page or not page.ready or page.readers <= 0:
            raise ValueError("LOAD requires a current DDR reader")
        return self.backend.load(page.handle, destination)

    @staticmethod
    def completed(future, expected):
        result = future.result()
        if (not result.get("stopped") or not result.get("native_completed")
                or result.get("bytes") != expected):
            raise RuntimeError("cache transfer has no complete native proof")
        return result

    def _retire(self, pages):
        for page in pages:
            if self.pages.get(page.key) is not page or not page.ready or page.readers or page.units:
                raise RuntimeError("cannot reclaim a referenced DDR page")
        for page in pages:
            for identity in tuple(self.pending_by_page.get(page.key, ())):
                self._forget_pending(identity)
            del self.pages[page.key]
            del self.by_handle[page.handle]
            self.retiring[page.handle] = (page, self.backend.release(page.handle))

    def _evict(self):
        for kind, identity in self.policy.candidates(self.clock()):
            if kind == "unit":
                keys = self.units.get(identity, ())
                pages = tuple(self.pages[key] for key in keys)
            else:
                page = self.by_handle.get(identity)
                pages = (page,) if page is not None else ()
            if not pages or any(not page.ready or page.readers for page in pages):
                continue
            self.policy.on_remove((kind, identity))
            if kind == "unit":
                self.units.pop(identity)
                for page in pages:
                    page.units.remove(identity)
            self._retire(tuple(page for page in pages if not page.units))
            return True
        return False

    def _reap_retired(self):
        for handle, (page, future) in tuple(self.retiring.items()):
            if future.done():
                future.result()
                self.used -= page.charge
                self.counters["evicted_bytes"] += page.charge
                del self.retiring[handle]

    def pump(self):
        if self.error:
            raise RuntimeError("DDR work is quarantined") from self.error
        try:
            for page in tuple(self.inflight.values()):
                if not page.future.done():
                    continue
                self.completed(page.future, page.size)
                page.future = None
                del self.inflight[page.handle]
                page.ready = True
                self.release_pin(page.lease)
                page.lease = page.source = None
                self.counters["stored_bytes"] += page.size
                self.counters["stores"] += 1
                if not page.units:
                    self.policy.on_publish(("page", page.handle), self.clock())
                for identity in tuple(self.pending_by_page.get(page.key, ())):
                    keys = self.pending_units[identity]
                    if all(self.pages[key].ready for key in keys):
                        self._publish_ready(identity, keys)
            self._reap_retired()
            while self.queued and not self.closing:
                page = self.queued[0]
                if (self.used + page.charge > self.capacity
                        or not self.backend.reserve(page.handle, page.source)):
                    retiring = sum(old.charge for old, _ in self.retiring.values())
                    if retiring and self.used + page.charge - retiring <= self.capacity:
                        break
                    if self._evict():
                        self._reap_retired()
                        if self.retiring:
                            break
                        continue
                    break
                self.queued.popleft()
                self.used += page.charge
                page.future = self.backend.store(page.handle, page.source)
                self.inflight[page.handle] = page
        except BaseException as error:
            self.error = error
            raise

    def begin_close(self):
        self.closing = True
        self.pending_units.clear()
        self.pending_by_page.clear()
        for identity in self.units:
            self.policy.on_remove(("unit", identity))
        self.units.clear()
        for page in self.pages.values():
            page.units.clear()
            if page.ready:
                self.policy.on_publish(("page", page.handle), self.clock())
        while self.queued:
            page = self.queued.popleft()
            self.release_pin(page.lease)
            del self.pages[page.key]
            del self.by_handle[page.handle]
        for handle, page in tuple(self.inflight.items()):
            if page.future.cancel():
                self.release_pin(page.lease)
                self.used -= page.charge
                del self.inflight[handle]
                del self.pages[page.key]
                del self.by_handle[handle]

    def close_ready(self):
        self.pump()
        return (not self.queued and not self.inflight and not self.retiring
                and all(page.ready and not page.readers for page in self.pages.values()))

    def finish_close(self):
        """Forget residency only after the backend confirms owner storage is closed."""
        if not self.closing or not self.close_ready():
            raise RuntimeError("DDR work must drain before storage retirement")
        for page in self.pages.values():
            self.policy.on_remove(("page", page.handle))
        self.pages.clear()
        self.by_handle.clear()
        self.counters["evicted_bytes"] += self.used
        self.used = 0
