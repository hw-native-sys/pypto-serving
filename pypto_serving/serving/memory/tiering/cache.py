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
from collections import Counter
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
        key = name, block.block_hash
        if key in self.tier.pages:
            return
        pool = self.cache._group_pools[name]
        source = name, partition, pool.local_block_id(block)
        pin = self.cache.pin_cached_group_blocks({name: (block,)}, partition)
        self.tier.capture(key, source, self.backend.size(source), pin, charge=self.backend.footprint(source))

    async def restore(
        self, request, hashes, tokens, *, maximum, groups=None, cancelled=lambda: False,
    ) -> int | None:
        try:
            while True:
                if cancelled():
                    raise ValueError("request cancelled during cache restore")
                ready, destination = self.prepare(request, hashes, tokens, maximum=maximum, groups=groups)
                if ready:
                    return destination
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
        identity = boundary, tuple(sorted(
            (name, values[boundary // self.cache._group_pools[name].spec.spec.token_capacity - 1])
            for name, values in hashes.items() if values))
        checkpoints = self.cache._group_checkpoints
        local = (partition, *identity)
        if (local not in checkpoints.checkpoints or identity in self.tier.units
                or identity in self.tier.pending_units):
            return
        keys = []
        for name, (values, end) in checkpoints.histories[local].items():
            pool = self.cache._group_pools[name]
            for value in values[:end]:
                key = name, value
                if key not in self.tier.pages:
                    self._capture(name, partition, pool.hash_to_block[partition, value])
                keys.append(key)
        for name, block_id in checkpoints.checkpoints[local]:
            block = self.cache._group_pools[name].blocks[block_id]
            self._capture(name, partition, block)
            keys.append((name, block.block_hash))
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
        for handle, partition in self.backend.pop_completed("load"):
            page = self.tier.by_handle[handle]
            key = (partition, *page.key)
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
        return resident if resident is not None else self.tier.lookup((pool.spec.name, value))

    def _restore_score(self, selected, tables, pressure, shared_cost=None):
        resident_free = Counter()
        if shared_cost is None:
            total = 0
            fresh_pages = Counter()
            for name, pages in tables.items():
                for page in pages:
                    if isinstance(page, Page):
                        total += page.size
                        if (selected, *page.key) not in self.loads:
                            fresh_pages[name] += 1
                    elif page is not None and not page.ref_cnt:
                        resident_free[name] += 1
        else:
            total, fresh_pages = shared_cost
        if any(self.cache._group_pools[name].num_free_blocks_in(selected) <
               count + resident_free[name] + 1 for name, count in fresh_pages.items()):
            return None
        return (
            total, sum(fresh_pages.values()), pressure[selected],
            (selected - self.cache._next_group_partition) % self.cache.group_partition_count,
        )

    def _ddr_only_partitions(self, hashes, maximum):
        """Identify destinations with identical logical tables from DDR alone."""
        remaining = set(range(self.cache.group_partition_count))
        for name, values in hashes.items():
            pool = self.cache._group_pools[name]
            bound = min(len(values), maximum // pool.spec.spec.token_capacity)
            index = pool.hash_to_block
            if bound <= 0 or not index:
                continue
            if len(index) < bound * len(remaining):
                wanted = set(values[:bound])
                remaining.difference_update(owner for owner, value in index if value in wanted)
            else:
                relevant = values[:bound]
                remaining = {owner for owner in remaining if not any((owner, value) in index for value in relevant)}
            if not remaining:
                break
        return remaining

    def _select(self, hashes, maximum, groups, partition, boundaries):
        pressure = Counter(self.cache._group_request_partitions.values())
        pressure.update(value.partition for value in self.pending.values())
        ddr_only = self._ddr_only_partitions(hashes, maximum) if partition is None else set()
        loading = {key[0] for key in self.loads} if ddr_only else set()
        shared_cost = None

        def score(selected, tables):
            nonlocal shared_cost
            if selected not in ddr_only or selected in loading:
                return self._restore_score(selected, tables, pressure)
            if shared_cost is None:
                shared_cost = (sum(page.size for pages in tables.values() for page in pages if page is not None),
                               {name: sum(page is not None for page in pages)
                                for name, pages in tables.items() if any(page is not None for page in pages)})
            return self._restore_score(selected, tables, pressure, shared_cost)

        return self.cache.select_group_prefix_blocks(
            hashes, max_cache_hit_tokens=maximum, shareable_group_names=groups, partition=partition,
            lookup=self._lookup, shared_checkpoint_boundaries=boundaries, score=score,
            equivalent_partitions=ddr_only,
        )

    def prepare(self, request, hashes, tokens, *, maximum, groups=None, partition=None) -> tuple[bool, int | None]:
        """Report readiness and the partition of a completed restore, if any."""
        started_ns = time.monotonic_ns() if self.trace is not None else 0
        if self.tier.error:
            raise RuntimeError("DDR work is quarantined") from self.tier.error
        pending = self.pending.get(request)
        if pending is not None:
            if pending.remaining:
                return False, None
            self.cache._protect_group_checkpoint(None, pending.hashes, pending.tokens,
                                                 pending.partition, {}, pending.boundary)
            self.cancel(request, completed=True)
            return True, pending.partition
        match = self.match_prefix(hashes, maximum, groups=groups, partition=partition)
        return not self.start_restore(request, hashes, tokens, match, started_ns=started_ns), None

    def match_prefix(self, hashes, maximum, *, groups=None, partition=None, score=True):
        """Read a candidate without changing page ownership or eviction order."""
        if maximum <= 0 or not self.tier.pages:
            return {}, 0, None
        sizes = {name: self.cache._group_pools[name].spec.spec.token_capacity for name in hashes}
        boundaries = {
            boundary for boundary, identity in self.tier.units if boundary <= maximum and
            all(name in sizes and boundary // sizes[name] <= len(hashes[name]) and
                hashes[name][boundary // sizes[name] - 1] == value for name, value in identity)
        }
        if score:
            return self._select(hashes, maximum, groups, partition, boundaries)
        return self.cache.select_group_prefix_blocks(
            hashes, max_cache_hit_tokens=maximum, shareable_group_names=groups, partition=partition,
            lookup=self._lookup, shared_checkpoint_boundaries=boundaries,
        )

    def restore_demand(self, tables, partition):
        """Count new targets and bytes for this destination, reusing in-flight LOAD."""
        fresh = Counter()
        size = 0
        needs_load = False
        for name, values in tables.items():
            for page in values:
                if isinstance(page, Page):
                    needs_load = True
                    if (partition, *page.key) not in self.loads:
                        fresh[name] += 1
                        size += page.size
        return fresh, size, needs_load

    def restore_queue_rounds(self, partition):
        load_queued = any(load.partition == partition and not load.ready for load in self.loads.values())
        store_queued = any(page.source[1] == partition for page in self.tier.inflight.values())
        return int(load_queued) + int(store_queued)

    def touch_prefix(self, tables):
        for name, values in tables.items():
            for page in values:
                if page is not None:
                    self.tier.touch(page.key if isinstance(page, Page) else (name, page.block_hash))

    def start_restore(self, request, hashes, tokens, match, *, reserve_step=None, started_ns=0):
        """Take destination/step pages before making a LOAD visible to the backend."""
        tables, boundary, selected = match
        if not boundary:
            return False
        if self.tier.error:
            raise RuntimeError("DDR work is quarantined") from self.tier.error
        if self.trace is not None and not started_ns:
            started_ns = time.monotonic_ns()
        # Selection returns physical ring slots. Freeze logical page order for
        # delayed resident/restore pins before filtering holes or DDR entries.
        ordered = dict(tables)
        for name, values in ordered.items():
            spec = self.cache._group_pools[name].spec
            if spec.sliding_window is not None:
                ordered[name] = tuple(reversed(tuple(self.cache._group_tail(
                    values, boundary // spec.spec.token_capacity))))
        self.touch_prefix(ordered)
        missing = {(selected, *page.key): page for values in ordered.values()
                   for page in values if isinstance(page, Page)}
        if not missing:
            return False
        resident = {name: tuple(page for page in values if page is not None and not isinstance(page, Page))
                    for name, values in ordered.items()}
        source_pin = self.cache.pin_cached_group_blocks(resident, selected)
        fresh = {key: page for key, page in missing.items() if key not in self.loads}
        shared_with = ({self.loads[key].request for key in missing if key not in fresh}
                       if self.trace is not None else set())
        targets, target_pin = {}, None
        try:
            if fresh:
                targets, target_pin = self.cache.reserve_group_restore_pages(
                    tuple((key[1], key[2]) for key in fresh), selected, headroom=int(reserve_step is None))
            if reserve_step is not None:
                targets_for_step = {**{page.key: self.loads[key].block for key, page in missing.items()
                                       if key in self.loads}, **targets}
                reserve_step({
                    name: [targets_for_step[page.key] if isinstance(page, Page) else page for page in values]
                    for name, values in tables.items()
                })
        except BaseException as error:
            self.cache.release_group_block_snapshot(source_pin)
            self.cache.release_group_block_snapshot(target_pin)
            if isinstance(error, KVCacheCapacityError) and reserve_step is None:
                if self.trace is not None:
                    self.trace.emit("restore_skipped", request_id=request, reason="hbm_capacity",
                                    prefix_tokens=boundary, partition=selected)
                return False
            raise

        try:
            if fresh:
                readers = self.tier.acquire(page.key for page in fresh.values())
                pins = RestorePins(target_pin, len(fresh))
                placements = [(page, targets[page.key]) for page in readers]
                copies = []
                for page, block in placements:
                    name, _ = page.key
                    local_id = self.cache._group_pools[name].local_block_id(block)
                    copies.append((page, (name, selected, local_id)))
                futures = self.tier.load(copies, request=request)
                for (page, block), future in zip(placements, futures, strict=True):
                    key = (selected, *page.key)
                    self.loads[key] = Load(page, block, selected, pins, future, request=request)
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
