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


@dataclass
class PendingPrefix:
    hashes: dict
    tokens: int
    boundary: int
    partition: int
    snapshot: object
    loads: tuple


class TieredCache:
    def __init__(self, cache, backend, capacity):
        self.cache, self.backend = cache, backend
        self.tier = CacheTier(capacity, backend, cache.release_group_block_snapshot)
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
                await asyncio.sleep(.002)
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
        self.tier.pump()
        for key, load in tuple(self.loads.items()):
            if load.ready or not load.future.done():
                continue
            try:
                self.tier.completed(load.future, load.page.size)
            except BaseException as error:
                self.tier.error = error
                raise
            if load.waiters:
                pool = self.cache._group_pools[key[1]]
                self.cache._cache_group_block(pool, load.partition, load.block, key[2])
            self.tier.counters["loads"] += 1
            self.tier.counters["loaded_bytes"] += load.page.size
            self.tier.release((load.page,))
            load.ready = True
            if not load.waiters:
                self._finish_load(key)

    def _lookup(self, pool, partition, value):
        resident = pool.hash_to_block.get((partition, value))
        return resident if resident is not None else self.tier.lookup((partition, pool.spec.name, value))

    def prepare(self, request, hashes, tokens, *, maximum, groups=None, partition=None):
        if self.tier.error:
            raise RuntimeError("DDR work is quarantined") from self.tier.error
        pending = self.pending.get(request)
        if pending is not None:
            if not all(self.loads[key].ready for key in pending.loads):
                return False
            self.cache._protect_group_checkpoint(None, pending.hashes, pending.tokens,
                                                 pending.partition, {}, pending.boundary)
            self.cancel(request)
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
                    self.loads[key] = Load(page, block, selected, pins, self.tier.load(page, destination))
            for key in missing:
                self.loads[key].waiters.add(request)
            self.pending[request] = PendingPrefix(hashes, tokens, boundary, selected, source_pin, tuple(missing))
            logger.info("kv_offload_restore request_id=%s prefix_tokens=%d partition=%d pages=%d",
                        request, boundary, selected, len(missing))
            return False
        except KVCacheCapacityError:
            self.cache.release_group_block_snapshot(source_pin)
            return True
        except BaseException as error:
            # Once submitted, native LOAD owns its targets even if later submission fails.
            self.quarantined.append((source_pin, target_pin, tuple(fresh)))
            self.tier.error = error
            raise

    def _finish_load(self, key):
        load = self.loads.pop(key)
        load.pins.remaining -= 1
        if load.pins.remaining == 0:
            self.cache.release_group_block_snapshot(load.pins.snapshot)

    def cancel(self, request):
        pending = self.pending.pop(request, None)
        if pending is None:
            return
        self.cache.release_group_block_snapshot(pending.snapshot)
        for key in pending.loads:
            load = self.loads[key]
            load.waiters.discard(request)
            if load.ready and not load.waiters:
                self._finish_load(key)

    def begin_close(self):
        for request in tuple(self.pending):
            self.cancel(request)
        self.tier.begin_close()

    def close_ready(self):
        self.progress()
        return not self.loads and self.tier.close_ready()
