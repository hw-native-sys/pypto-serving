# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real grouped allocator/checkpoint integration, without SDK or device initialization."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.serving.memory.tiering.cache import TieredCache
from pypto_serving.serving.sched.scheduler import Request
from tests.unit.serving.memory.test_tiering import finished
from tests.unit.serving.test_grouped_prefix_cache import (
    _finish_prefill, _prefill_scheduler, manager, prompt,
)


class MemoryBackend:
    def __init__(self):
        self.live, self.device, self.waiting = {}, {}, []
        self.hold = False

    def size(self, source):
        return 1

    def footprint(self, source):
        return 1

    def store(self, handle, source):
        self.live[handle] = self.device.setdefault(source, ("bytes", source))
        return finished(dict(bytes=1, stopped=True, native_completed=True))

    def load(self, handle, destination):
        self.device[destination] = self.live[handle]
        future = Future()
        if self.hold:
            self.waiting.append(future)
        else:
            future.set_result(dict(bytes=1, stopped=True, native_completed=True))
        return future

    def release(self, handle):
        del self.live[handle]
        return finished(None)


def fixture(*, state=True, partitions=1):
    cache = manager(slots=5, partitions=partitions, checkpoint_state=state)
    backend = MemoryBackend()
    cache.tiering = TieredCache(cache, backend, 200_000)
    return cache, cache.tiering, backend


def warm(cache, tier, tokens, request="warm"):
    scheduler = _prefill_scheduler(cache)
    schedule = scheduler.schedule

    def progress(**kwargs):
        tier.progress()
        tier.progress()
        return schedule(**kwargs)

    scheduler.schedule = progress
    boundary = _finish_prefill(scheduler, Request(request, tokens, max_new_tokens=512))
    tier.progress()
    tier.progress()
    assert not tier.tier.queued and not tier.tier.inflight
    return boundary


def evict(cache, *, names=None):
    for name, pool in cache._group_pools.items():
        if names is not None and name not in names:
            continue
        for (partition, _), block in tuple(pool.hash_to_block.items()):
            assert block.ref_cnt == 0
            cache._invalidate_group_block(pool, partition, block)
    if cache._group_checkpoints:
        cache._group_checkpoints.prune(cache._group_pools)


@pytest.mark.parametrize("length", [129, 512, 513, 639, 640, 8205])
def test_complete_checkpoint_restores_after_hbm_eviction_with_new_physical_pages(length):
    cache, tier, backend = fixture()
    tokens = prompt(0, length)
    boundary = warm(cache, tier, tokens)
    hashes = cache.compute_group_block_hashes(tokens)
    expected = {key: backend.live[page.handle] for key, page in tier.tier.pages.items()}
    evict(cache)
    assert cache.acquire_group_prefix_blocks("miss", hashes, max_cache_hit_tokens=boundary)[1] == 0
    if not boundary:
        return
    assert not tier.prepare("restored", hashes, length, maximum=boundary)
    tier.progress()
    assert tier.prepare("restored", hashes, length, maximum=boundary)
    tables, hit, partition = cache.acquire_group_prefix_blocks("restored", hashes, max_cache_hit_tokens=boundary)
    assert hit == boundary and partition == 0
    moved = False
    for name, ids in tables.items():
        pool = cache._group_pools[name]
        for local in ids:
            block = pool.block_from_local_id(0, local)
            key = 0, name, block.block_hash
            if key in expected:
                assert backend.device[name, 0, local] == expected[key]
                moved |= expected[key][1][2] != local
    assert moved
    cache.release_all_group_requests("restored")
    assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)


def test_mixed_hbm_ddr_restore_deduplicates_loads_and_cancellation_retains_native_targets():
    cache, tier, backend = fixture()
    tokens = prompt(0, 513)
    boundary = warm(cache, tier, tokens)
    hashes = cache.compute_group_block_hashes(tokens)
    evict(cache, names=("ori", "idx", "csa_state"))
    backend.hold = True
    for request in ("one", "two"):
        assert not tier.prepare(request, hashes, len(tokens), maximum=boundary)
    assert len(tier.loads) == len(backend.waiting)
    tier.cancel("one")
    assert all(load.block.ref_cnt > 0 and load.page.readers == 1 for load in tier.loads.values())
    for future in backend.waiting:
        future.set_result(dict(bytes=1, stopped=True, native_completed=True))
    tier.progress()
    assert tier.prepare("two", hashes, len(tokens), maximum=boundary)
    assert cache.acquire_group_prefix_blocks("two", hashes, max_cache_hit_tokens=boundary)[1] == boundary
    cache.release_all_group_requests("two")
    assert not tier.pending and not tier.loads
    assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)


def test_d_history_restore_uses_its_own_groups_and_does_not_require_p_state():
    from pypto_serving.pd.adapter import DeepSeekDSpark
    cache, tier, _ = fixture(partitions=4)
    tokens = prompt(0, 513)
    warm(cache, tier, tokens)
    groups = tuple(DeepSeekDSpark().prefix_spec(tokens, cache).hashes)
    hashes = cache.compute_group_block_hashes(tokens, group_names=groups)
    evict(cache)
    assert not tier.prepare("d", hashes, len(tokens), maximum=len(tokens), groups=groups)
    tier.progress()
    assert tier.prepare("d", hashes, len(tokens), maximum=len(tokens), groups=groups)
    result = cache.reserve_group_cache("d", "d", 131_585, prompt_token_count=len(tokens),
        prefix_block_hashes=hashes, shareable_group_names=groups)
    assert result.prefix_hit_tokens == 512
    assert all(not table for name, table in result.shared_block_ids_by_group.items() if name.endswith("state"))
    cache.release_group_cache("d", confirmed_stopped=True)


@pytest.mark.parametrize("cancelled", [False, True])
def test_d_begin_waits_for_ddr_before_publishing_placement_and_reserving_state(cancelled):
    from pypto_serving.pd.adapter import DeepSeekDSpark
    from pypto_serving.pd.protocol import Binding, Continuation
    from tests.unit.pd.test_service import service

    async def run():
        cache, tier, backend = fixture()
        tokens = prompt(0, 513)
        warm(cache, tier, tokens)
        evict(cache)
        backend.hold = True
        node, source, destination = service()
        node.core.kv_cache_manager = cache
        node.cache = cache
        node.core.config = SimpleNamespace(max_num_running_reqs=8,
                                           runtime_config=SimpleNamespace(max_seq_len=655360))
        node.adapter = DeepSeekDSpark()
        calls = []

        async def worker(operation, payload):
            calls.append(operation)
            assert all(load.ready for load in tier.loads.values())
            return {"bootstrap": 0}

        node.worker = worker
        entry = node.register_http(Binding.create(source, destination, {}), {})
        continuation = Continuation(tuple(tokens), {"max_new_tokens": 32, "temperature": 0}, None, None)
        task = asyncio.create_task(node.control(source, (), "begin", dict(
            binding=msgspec.to_builtins(entry.binding), continuation=msgspec.to_builtins(continuation))))
        for _ in range(1000):
            if tier.loads:
                break
            await asyncio.sleep(.002)
        else:
            pytest.fail("D did not start its local DDR restore")
        assert entry.reservation is None and not calls
        if cancelled:
            await node.abort(entry, stopped=True)
            with pytest.raises(ValueError, match="cancelled"):
                await task
        for future in backend.waiting:
            future.set_result(dict(bytes=1, stopped=True, native_completed=True))
        tier.progress()
        if not cancelled:
            placement = await task
            assert placement.hit == 512 and calls == ["reserve_state"]
            await node.abort(entry, stopped=True)
        assert not tier.pending and not tier.loads
        assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)

    asyncio.run(run())


def test_missing_ddr_state_cannot_claim_complete_resume():
    cache, tier, _ = fixture()
    tokens = prompt(0, 513)
    boundary = warm(cache, tier, tokens)
    evict(cache)
    identity, keys = next((identity, keys) for identity, keys in tier.tier.units.items()
                          if identity[1] == boundary)
    page = next(tier.tier.pages[key] for key in keys if key[1] == "csa_state")
    # Simulate safe retirement of one joint unit under capacity pressure.
    for unit in tuple(page.units):
        tier.tier.policy.on_access(("unit", unit), -1)
    while identity in tier.tier.units:
        assert tier.tier._evict()
    tier.progress()
    hashes = cache.compute_group_block_hashes(tokens)
    assert tier.prepare("query", hashes, len(tokens), maximum=boundary)
    assert cache.acquire_group_prefix_blocks("query", hashes, max_cache_hit_tokens=boundary)[1] == 0


def test_cancelled_d_restore_does_not_proceed_to_reservation():
    cache, tier, backend = fixture()
    tokens = prompt(0, 513)
    boundary = warm(cache, tier, tokens)
    evict(cache)
    backend.hold = True
    hashes = cache.compute_group_block_hashes(tokens)

    async def scenario():
        stopped = False
        task = asyncio.create_task(tier.restore("d", hashes, len(tokens), maximum=boundary,
                                                cancelled=lambda: stopped))
        await asyncio.sleep(.01)
        stopped = True
        with pytest.raises(ValueError, match="cancelled"):
            await task

    asyncio.run(scenario())
    assert not tier.pending and not cache._group_request_partitions
    assert tier.loads and all(load.block.ref_cnt > 0 for load in tier.loads.values())
    for future in backend.waiting:
        future.set_result(dict(bytes=1, stopped=True, native_completed=True))
    tier.progress()
    assert not tier.loads
    assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)


def test_unknown_load_keeps_destination_and_ddr_reference_without_publishing():
    cache, tier, backend = fixture()
    tokens = prompt(0, 513)
    boundary = warm(cache, tier, tokens)
    evict(cache, names=("idx",))
    backend.hold = True
    hashes = cache.compute_group_block_hashes(tokens)
    assert not tier.prepare("uncertain", hashes, len(tokens), maximum=boundary)
    for future in backend.waiting:
        future.set_result(dict(bytes=1, stopped=False, native_completed=False))
    with pytest.raises(RuntimeError, match="proof"):
        tier.progress()
    with pytest.raises(RuntimeError, match="quarantined"):
        tier.prepare("uncertain", hashes, len(tokens), maximum=boundary)
    tier.cancel("uncertain")
    assert all(load.block.ref_cnt > 0 and load.page.readers for load in tier.loads.values())
    assert not cache._group_pools["idx"].hash_to_block
    assert cache._group_pools["cmp_c4"].hash_to_block


def test_restore_capacity_miss_releases_temporary_resident_pins():
    cache, tier, _ = fixture()
    tokens = prompt(0, 513)
    boundary = warm(cache, tier, tokens)
    evict(cache, names=("idx",))
    pool = cache._group_pools["idx"]
    held = [cache._take_group_block(pool, 0) for _ in range(pool.num_free_blocks_in(0))]
    hashes = cache.compute_group_block_hashes(tokens)
    assert tier.prepare("capacity", hashes, len(tokens), maximum=boundary)
    assert not tier.pending and not tier.loads
    assert all(b.ref_cnt == 0 for name, p in cache._group_pools.items()
               if name != "idx" for b in p.blocks)
    for block in held:
        block.ref_cnt -= 1
        pool.free_queues[0].append(block)


def test_eight_groups_restore_joint_boundaries_across_partitions_and_growth():
    cache, tier, _ = fixture(partitions=4)
    for turn in range(2):
        length = 513 + turn * 512
        for group in range(8):
            cache._next_group_partition = group % 4
            warm(cache, tier, [1000000 + group] + list(range(1, length)), f"warm-{turn}-{group}")
        evict(cache)
        for group in range(8):
            request = f"restored-{turn}-{group}"
            hashes = cache.compute_group_block_hashes([1000000 + group] + list(range(1, length)))
            boundary = (length - 128) // 128 * 128
            assert not tier.prepare(request, hashes, length, maximum=boundary)
            tier.progress()
            assert tier.prepare(request, hashes, length, maximum=boundary)
            _, hit, partition = cache.acquire_group_prefix_blocks(request, hashes, max_cache_hit_tokens=boundary)
            assert hit == boundary and partition == group % 4
            cache.release_all_group_requests(request)
    tier.begin_close()
    for _ in range(4):
        if tier.close_ready():
            break
    assert not tier.tier.pages and not tier.loads and tier.tier.used == 0
    assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)
