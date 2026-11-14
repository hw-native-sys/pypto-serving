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
from itertools import chain
import logging
import time

from ..kv_cache import KVCacheCapacityError, KVCacheRestoreError
from .manager import CacheTier
from .residency import CheckpointIndex, DDRSource, Replica

logger = logging.getLogger(__name__)


@dataclass
class RestorePins:
    snapshot: object
    remaining: int


@dataclass
class Load:
    source: Replica
    reader: object
    block: object
    partition: int
    pins: RestorePins
    future: object
    waiters: set = field(default_factory=set)
    ready: bool = False
    request: str | None = None
    error: object = None


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
        self.sources = [DDRSource(self.tier)]
        self.checkpoints = CheckpointIndex()
        self.tier.on_store, self.tier.on_remove = self._stored, self._removed
        self.pending, self.loads = {}, {}
        self.quarantined = []

    def add_store(self, store):
        if any(source.name == store.name for source in self.sources):
            raise ValueError("duplicate cache store")
        store.on_remove = self._removed
        self.sources.append(store)

    def _stored(self, page):
        for store in self.sources[1:]:
            store.stored(page)

    def _removed(self, key):
        if self.resident(key) is None and not any(source.contains(key) for source in self.sources):
            self.checkpoints.remove(key)

    def capture(self, name, partition, block):
        # Rolling pages are reusable only with a confirmed joint checkpoint.
        if self.cache._group_pools[name].spec.sliding_window is None:
            self._capture(name, partition, block)

    def _capture(self, name, partition, block):
        key = partition, name, block.block_hash
        if key in self.tier.pages or key in self.loads:
            return
        if any(source.lookup(key) is not None for source in self.sources):
            return  # Reuse durable contents; DDR promotion is already opportunistic.
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
        self.checkpoints.publish(identity, keys)
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
        for store in self.sources[1:]:
            store.pump()
        self.tier.pump()
        reaped_pages = reaped_bytes = published_pages = 0
        requests = set()
        keys = chain.from_iterable(source.pop_loaded() for source in self.sources)
        for key in keys:
            load = self.loads.get(key)
            if load is None or load.ready:
                continue
            try:
                self.tier.completed(load.future, load.source.size)
            except KVCacheRestoreError as error:
                load.error = error
                if self.trace is not None:
                    self.trace.emit("restore_failed", requests=sorted(load.waiters), stage="file",
                                    error_type=type(error).__name__, native_submitted=False,
                                    source_tier=load.source.store.name)
            except BaseException as error:
                if self.trace is not None:
                    self.trace.emit("restore_failed", requests=sorted(load.waiters),
                                    error_type=type(error).__name__, native_completed=False)
                raise
            if load.waiters and load.error is None:
                pool = self.cache._group_pools[key[1]]
                self.cache._cache_group_block(pool, load.partition, load.block, key[2])
                published_pages += 1
            if load.error is None:
                self.tier.counters["loads"] += 1
                self.tier.counters["loaded_bytes"] += load.source.size
            load.source.store.release(load.reader)
            load.reader = None
            load.ready = True
            if self.trace is not None:
                reaped_pages += 1
                reaped_bytes += load.source.size
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
        if resident is not None:
            return resident
        key = partition, pool.spec.name, value
        for source in self.sources:
            page = source.lookup(key)
            if page is not None:
                return Replica(source, page)
        return None

    def resident(self, key):
        partition, name, value = key
        return self.cache._group_pools[name].hash_to_block.get((partition, value))

    def evicted(self, name, partition, value):
        self._removed((partition, name, value))

    def prepare(self, request, hashes, tokens, *, maximum, groups=None, partition=None):
        started_ns = time.monotonic_ns() if self.trace is not None else 0
        if self.tier.error:
            raise RuntimeError("DDR work is quarantined") from self.tier.error
        pending = self.pending.get(request)
        if pending is not None:
            if pending.remaining:
                return False
            error = next((self.loads[key].error for key in pending.loads if self.loads[key].error), None)
            if error is not None:
                self.cancel(request)
                raise error
            self.cache._protect_group_checkpoint(None, pending.hashes, pending.tokens,
                                                 pending.partition, {}, pending.boundary)
            self.cancel(request, completed=True)
            return True
        if not any(source.has_pages() for source in self.sources):
            return True
        checkpoints = self.cache._group_checkpoints
        candidates = set(checkpoints.checkpoints if checkpoints is not None else ())
        candidates.update(self.checkpoints.units)
        tables, boundary, selected = self.cache.select_group_prefix_blocks(
            hashes, max_cache_hit_tokens=maximum, shareable_group_names=groups, partition=partition,
            lookup=self._lookup, checkpoint_keys=candidates)
        if not boundary:
            return True
        for name, values in tables.items():
            for page in values:
                if page is not None:
                    key = page.key if isinstance(page, Replica) else (selected, name, page.block_hash)
                    for source in self.sources:
                        source.touch(key)
        missing = {page.key: page for values in tables.values() for page in values if isinstance(page, Replica)}
        if not missing:
            return True
        resident = {name: tuple(page for page in values if page is not None and not isinstance(page, Replica))
                    for name, values in tables.items()}
        source_pin = self.cache.pin_cached_group_blocks(resident, selected)
        fresh = {key: page for key, page in missing.items() if key not in self.loads}
        shared_with = ({self.loads[key].request for key in missing if key not in fresh}
                       if self.trace is not None else set())
        readers = {}
        targets = target_pin = None
        try:
            for source in self.sources:
                keys = tuple(key for key, replica in fresh.items() if replica.store is source)
                if not keys:
                    continue
                acquired = source.acquire(keys)
                if acquired is None:
                    for key, reader in readers.items():
                        fresh[key].store.release(reader)
                    self.cache.release_group_block_snapshot(source_pin)
                    return True  # A source disappeared before admission: a normal cache miss.
                readers.update(acquired)
            if fresh:
                targets, target_pin = self.cache.reserve_group_restore_pages(
                    tuple((key[1], key[2]) for key in fresh), selected)
                pins = RestorePins(target_pin, len(fresh))
                for replica in fresh.values():
                    key = replica.key
                    block = targets[key[1], key[2]]
                    destination = key[1], selected, self.cache._group_pools[key[1]].local_block_id(block)
                    future = replica.store.load(replica.page, destination, readers[key], request=request)
                    self.loads[key] = Load(replica, readers[key], block, selected, pins, future, request=request)
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
            for key, reader in readers.items():
                fresh[key].store.release(reader)
            if self.trace is not None:
                self.trace.emit("restore_skipped", request_id=request, reason="hbm_capacity",
                                prefix_tokens=boundary, partition=selected)
            return True
        except BaseException as error:
            # Once submitted, native LOAD owns its targets even if later submission fails.
            self.quarantined.append((source_pin, target_pin, tuple(fresh), readers))
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
        for store in self.sources[1:]:
            store.begin_close()
        for request in tuple(self.pending):
            self.cancel(request)
        self.tier.begin_close()

    def close_ready(self):
        self.progress()
        return (not self.loads and self.tier.close_ready()
                and all(store.idle() for store in self.sources[1:]))

    def close_storage(self):
        try:
            for store in reversed(self.sources[1:]):
                store.close()
        finally:
            self.backend.close()
