# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cache-manager integration: immutable publication and shared HBM/DDR prefix lookup."""

import asyncio
from dataclasses import dataclass, field
import logging
import time

from ..kv_cache import KVCacheCapacityError
from .manager import CacheTier, Page

logger = logging.getLogger(__name__)


@dataclass
class RestorePins:
    snapshot: object
    remaining: int


@dataclass
class Load:
    page: Page
    block: object
    partition: int
    pins: RestorePins
    future: object
    waiters: set = field(default_factory=set)
    ready: bool = False
    request: str | None = None


@dataclass
class PendingPrefix:
    hashes: dict
    tokens: int
    boundary: int
    partition: int
    snapshot: object
    loads: tuple
    started_ns: int = 0
    remaining: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)


class TieredCache:
    def __init__(self, cache, backend, capacity, *, trace=None, notify=None):
        self.cache, self.backend = cache, backend
        self.trace = trace
        self.tier = CacheTier(capacity, backend, cache.release_group_block_snapshot, trace=trace, notify=notify)
        self.pending, self.loads = {}, {}
        self.quarantined = []

    def capture(self, name, partition, block):
        # Rolling pages are reusable only with a confirmed joint checkpoint.
        if self.cache._group_pools[name].spec.sliding_window is None:
            self._capture(name, partition, block)

    def _capture(self, name, partition, block):
        key = partition, name, block.block_hash
        if key in self.tier.pages:
            return
        pool = self.cache._group_pools[name]
        source = name, partition, pool.local_block_id(block)
        pin = self.cache.pin_cached_group_blocks({name: (block,)}, partition)
        self.tier.capture(key, source, self.backend.size(source), pin, charge=self.backend.footprint(source))

    async def restore(self, request, hashes, tokens, *, maximum, groups=None, cancelled=lambda: False):
        try:
            while True:
                if cancelled():
                    raise ValueError("request cancelled during cache restore")
                if self.prepare(request, hashes, tokens, maximum=maximum, groups=groups):
                    return
                pending = self.pending[request]
                pending.changed.clear()
                if pending.remaining:
                    await pending.changed.wait()
                if self.pending.get(request) is not pending:
                    raise ValueError("request cancelled during cache restore")
        except BaseException:
            self.cancel(request)
            raise

    def publish_checkpoint(self, hashes, tokens, partition, boundary):
        identity = partition, boundary, tuple(sorted(
            (name, values[boundary // self.cache._group_pools[name].spec.spec.token_capacity - 1])
            for name, values in hashes.items() if values))
        checkpoints = self.cache._group_checkpoints
        if (identity not in checkpoints.checkpoints or identity in self.tier.units
                or identity in self.tier.pending_units):
            return
        keys = []
        for name, (values, end) in checkpoints.histories[identity].items():
            pool = self.cache._group_pools[name]
            for value in values[:end]:
                key = partition, name, value
                if key not in self.tier.pages:
                    self._capture(name, partition, pool.hash_to_block[partition, value])
                keys.append(key)
        for name, block_id in checkpoints.checkpoints[identity]:
            block = self.cache._group_pools[name].blocks[block_id]
            self._capture(name, partition, block)
            keys.append((partition, name, block.block_hash))
        self.tier.publish(identity, keys)

    def progress(self):
        try:
            self._progress()
        except BaseException as error:
            self.tier.error = error
            self.wake()
            raise

    def wake(self):
        """Recheck existing restore waiters; this is not a completion or stop proof."""
        for pending in self.pending.values():
            pending.changed.set()

    def _progress(self):
        self.tier.pump()
        reaped_pages = reaped_bytes = published_pages = 0
        requests = set()
        for handle in self.backend.pop_completed("load"):
            page = self.tier.by_handle[handle]
            key = page.key
            load = self.loads.get(key)
            if load is None or load.ready:
                continue
            try:
                self.tier.completed(load.future, load.page.size)
            except BaseException as error:
                if self.trace is not None:
                    self.trace.emit("restore_failed", requests=sorted(load.waiters),
                                    error_type=type(error).__name__, native_completed=False)
                raise
            if load.waiters:
                pool = self.cache._group_pools[key[1]]
                self.cache._cache_group_block(pool, load.partition, load.block, key[2])
                published_pages += 1
            self.tier.counters["loads"] += 1
            self.tier.counters["loaded_bytes"] += load.page.size
            self.tier.release((load.page,))
            load.ready = True
            if self.trace is not None:
                reaped_pages += 1
                reaped_bytes += load.page.size
                requests.update(load.waiters)
            for request in load.waiters:
                pending = self.pending[request]
                pending.remaining -= 1
                if pending.remaining == 0:
                    pending.changed.set()
            if not load.waiters:
                self._finish_load(key)
        if self.trace is not None:
            if reaped_pages:
                self.trace.emit("load_reaped", pages=reaped_pages, bytes=reaped_bytes,
                                published_pages=published_pages, requests=sorted(requests))
            self.trace.snapshot(self)

    def _lookup(self, pool, partition, value):
        resident = pool.hash_to_block.get((partition, value))
        return resident if resident is not None else self.tier.lookup((partition, pool.spec.name, value))

    def prepare(self, request, hashes, tokens, *, maximum, groups=None, partition=None):
        started_ns = time.monotonic_ns() if self.trace is not None else 0
        if self.tier.error:
            raise RuntimeError("DDR work is quarantined") from self.tier.error
        pending = self.pending.get(request)
        if pending is not None:
            if pending.remaining:
                return False
            self.cache._protect_group_checkpoint(None, pending.hashes, pending.tokens,
                                                 pending.partition, {}, pending.boundary)
            self.cancel(request, completed=True)
            return True
        if not self.tier.pages:
            return True
        checkpoints = self.cache._group_checkpoints
        candidates = set(checkpoints.checkpoints if checkpoints is not None else ())
        candidates.update(self.tier.units)
        tables, boundary, selected = self.cache.select_group_prefix_blocks(
            hashes, max_cache_hit_tokens=maximum, shareable_group_names=groups, partition=partition,
            lookup=self._lookup, checkpoint_keys=candidates)
        if not boundary:
            return True
        for name, values in tables.items():
            for page in values:
                if page is not None:
                    self.tier.touch(page.key if isinstance(page, Page) else (selected, name, page.block_hash))
        missing = {page.key: page for values in tables.values() for page in values if isinstance(page, Page)}
        if not missing:
            return True
        resident = {name: tuple(page for page in values if page is not None and not isinstance(page, Page))
                    for name, values in tables.items()}
        source_pin = self.cache.pin_cached_group_blocks(resident, selected)
        fresh = {key: page for key, page in missing.items() if key not in self.loads}
        shared_with = ({self.loads[key].request for key in missing if key not in fresh}
                       if self.trace is not None else set())
        targets = target_pin = None
        try:
            if fresh:
                targets, target_pin = self.cache.reserve_group_restore_pages(
                    tuple((key[1], key[2]) for key in fresh), selected)
                readers = self.tier.acquire(fresh)
                pins = RestorePins(target_pin, len(fresh))
                for page in readers:
                    key = page.key
                    block = targets[key[1], key[2]]
                    destination = key[1], selected, self.cache._group_pools[key[1]].local_block_id(block)
                    self.loads[key] = Load(page, block, selected, pins,
                        self.tier.load(page, destination, request=request), request=request)
            for key in missing:
                self.loads[key].waiters.add(request)
            self.pending[request] = PendingPrefix(hashes, tokens, boundary, selected, source_pin,
                tuple(missing), started_ns=started_ns,
                remaining=sum(not self.loads[key].ready for key in missing))
            if self.trace is not None:
                counts = {}
                for key in missing:
                    counts[key[1]] = counts.get(key[1], 0) + 1
                self.trace.emit("restore_start", request_id=request, prompt_tokens=tokens,
                    prefix_tokens=boundary, partition=selected, pages=len(missing),
                    new_pages=len(fresh), groups=counts,
                    restore_bytes=sum(page.size for page in missing.values()),
                    shared_with=sorted(shared_with), prepare_ns=time.monotonic_ns()-started_ns)
            logger.info("kv_offload_restore request_id=%s prefix_tokens=%d partition=%d pages=%d",
                        request, boundary, selected, len(missing))
            return False
        except KVCacheCapacityError:
            self.cache.release_group_block_snapshot(source_pin)
            if self.trace is not None:
                self.trace.emit("restore_skipped", request_id=request, reason="hbm_capacity",
                                prefix_tokens=boundary, partition=selected)
            return True
        except BaseException as error:
            # Once submitted, native LOAD owns its targets even if later submission fails.
            self.quarantined.append((source_pin, target_pin, tuple(fresh)))
            self.tier.error = error
            if self.trace is not None:
                self.trace.emit("restore_failed", requests=[request], error_type=type(error).__name__,
                                native_completed=False)
            raise

    def _finish_load(self, key):
        load = self.loads.pop(key)
        load.pins.remaining -= 1
        if load.pins.remaining == 0:
            self.cache.release_group_block_snapshot(load.pins.snapshot)

    def cancel(self, request, *, completed=False):
        pending = self.pending.pop(request, None)
        if pending is None:
            return
        pending.changed.set()
        self.cache.release_group_block_snapshot(pending.snapshot)
        remaining = 0
        for key in pending.loads:
            load = self.loads[key]
            remaining += not load.ready
            load.waiters.discard(request)
            if load.ready and not load.waiters:
                self._finish_load(key)
        if self.trace is not None:
            self.trace.emit("restore_ready" if completed else "restore_cancelled",
                request_id=request, prefix_tokens=pending.boundary, partition=pending.partition,
                started_ns=pending.started_ns, elapsed_ns=time.monotonic_ns()-pending.started_ns,
                pages=len(pending.loads),
                unconfirmed_pages=remaining)

    def begin_close(self):
        for request in tuple(self.pending):
            self.cancel(request)
        self.tier.begin_close()

    def close_ready(self):
        self.progress()
        return not self.loads and self.tier.close_ready()
