# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import asyncio
from collections import deque
from concurrent.futures import Future
import threading
from types import SimpleNamespace

import pytest
import msgspec

from pypto_serving.serving.memory.tiering.cache import TieredCache
from pypto_serving.serving.memory.tiering.manager import CacheTier
from pypto_serving.serving.memory.tiering.integration import Session
from pypto_serving.serving.sched.scheduler import Request
from tests.unit.serving.test_grouped_prefix_cache import _finish_prefill, _prefill_scheduler, manager, prompt


class Backend:
    def __init__(self):
        self.live, self.device, self.waiting = {}, {}, []
        self.completed = {op: deque() for op in ('store', 'load')}
        self.hold, self.notify = False, None

    def set_notify(self, notify):
        self.notify = notify

    def pop_completed(self, operation):
        while self.completed[operation]:
            yield self.completed[operation].popleft()

    def job(self, operation, handle, result):
        future = Future()
        def done(_):
            self.completed[operation].append(handle)
            if self.notify is not None:
                self.notify()
        future.add_done_callback(done)
        if self.hold:
            self.waiting.append(future)
        else:
            future.set_result(result)
        return future

    def size(self, source):
        return 1

    footprint = size

    def reserve(self, handle, source):
        return True

    def store(self, handle, source):
        self.live[handle] = self.device.setdefault(source, source)
        return self.job('store', handle, dict(bytes=1, stopped=True, native_completed=True))

    def load(self, handle, destination, *, request=None):
        self.device[destination] = self.live[handle]
        return self.job('load', handle, dict(bytes=1, stopped=True, native_completed=True))

    def release(self, handle):
        del self.live[handle]
        future = Future()
        future.set_result(None)
        return future

    def close(self):
        self.live.clear()


def fixture(partitions=1):
    cache, backend = manager(slots=5, partitions=partitions, checkpoint_state=True), Backend()
    cache.tiering = TieredCache(cache, backend, 200_000)
    return cache, cache.tiering, backend


def warm(cache, tokens, request='warm'):
    scheduler = _prefill_scheduler(cache)
    schedule = scheduler.schedule
    def progress(**kwargs):
        cache.tiering.progress()
        cache.tiering.progress()
        return schedule(**kwargs)
    scheduler.schedule = progress
    boundary = _finish_prefill(scheduler, Request(request, tokens, max_new_tokens=512))
    cache.tiering.progress()
    cache.tiering.progress()
    return boundary


def evict(cache, names=None):
    for name, pool in cache._group_pools.items():
        if names is None or name in names:
            for (partition, _), block in tuple(pool.hash_to_block.items()):
                assert not block.ref_cnt
                cache._invalidate_group_block(pool, partition, block)
    if cache._group_checkpoints is not None:
        cache._group_checkpoints.prune(cache._group_pools)


def attach(cache):
    session = Session.__new__(Session)
    session.__dict__.update(cache=cache, trace=None, _loop=asyncio.get_running_loop(),
        _notify_lock=threading.Lock(), _notified=False,
        core=SimpleNamespace(_running=True, _request_contexts={}, kv_cache_manager=cache.cache),
        pool=SimpleNamespace(close=lambda: None), copy_library=SimpleNamespace(cleanup=lambda: None))
    cache.backend.set_notify(session._notify)
    cache.tier.notify = session._notify
    return session


async def until(condition):
    for _ in range(50):
        if condition():
            return
        await asyncio.sleep(0)
    pytest.fail('expected Host transition did not occur')


@pytest.mark.parametrize('length', [129, 513, 8192, 8205, 8320])
def test_joint_checkpoint_restore_preserves_content_and_alignment(length):
    cache, tier, backend = fixture()
    tokens = prompt(0, length)
    boundary = warm(cache, tokens)
    hashes = cache.compute_group_block_hashes(tokens)
    expected = {key: backend.live[page.handle] for key, page in tier.tier.pages.items()}
    evict(cache)
    assert cache.acquire_group_prefix_blocks('miss', hashes, max_cache_hit_tokens=boundary)[1] == 0
    if not boundary:
        return
    assert not tier.prepare('restore', hashes, length, maximum=boundary)
    tier.progress()
    assert tier.prepare('restore', hashes, length, maximum=boundary)
    tables, hit, partition = cache.acquire_group_prefix_blocks('restore', hashes, max_cache_hit_tokens=boundary)
    assert hit == boundary and partition == 0
    for name, ids in tables.items():
        for local in ids:
            block = cache._group_pools[name].block_from_local_id(0, local)
            assert backend.device[name, 0, local] == expected[0, name, block.block_hash]
    cache.release_all_group_requests('restore')
    assert all(not b.ref_cnt for p in cache._group_pools.values() for b in p.blocks)


@pytest.mark.parametrize('outcome', ['shared', 'unknown'])
def test_async_shared_restore_cancel_and_unknown(outcome):
    async def run():
        cache, tier, backend = fixture()
        tokens = prompt(0, 513)
        boundary = warm(cache, tokens)
        evict(cache, ('ori', 'idx', 'csa_state'))
        backend.hold = True
        session = attach(tier)
        hashes = cache.compute_group_block_hashes(tokens)
        tasks = [asyncio.create_task(tier.restore(r, hashes, len(tokens), maximum=boundary)) for r in ('a', 'b')]
        await until(lambda: len(tier.pending) == 2)
        assert len(tier.loads) == len(backend.waiting)
        tier.cancel('a')
        with pytest.raises(ValueError, match='cancelled'):
            await tasks[0]
        assert all(load.block.ref_cnt and load.source.page.readers for load in tier.loads.values())
        proof = dict(bytes=1, stopped=outcome != 'unknown', native_completed=outcome != 'unknown')
        for future in backend.waiting[:-1]:
            future.set_result(proof)
        await until(lambda: tier.tier.error or all(load.ready for load in list(tier.loads.values())[:-1]))
        if outcome == 'unknown':
            with pytest.raises(RuntimeError, match='quarantined'):
                await tasks[1]
            assert not session.core._running
            assert all(load.block.ref_cnt and load.source.page.readers for load in tier.loads.values())
            backend.set_notify(None)
            return
        if outcome == 'shared':
            assert not tasks[1].done() and tier.pending['b'].remaining == 1
        backend.waiting[-1].set_result(proof)
        await until(lambda: not tier.loads)
        if outcome == 'shared':
            await tasks[1]
        assert not tier.pending
        await session.close()
        assert all(not b.ref_cnt for p in cache._group_pools.values() for b in p.blocks)
    asyncio.run(run())


@pytest.mark.parametrize('cancelled', [False, True])
def test_d_reserves_only_after_its_independent_history_restore(cancelled):
    from pypto_serving.pd.adapter import DeepSeekDSpark
    from pypto_serving.pd.protocol import Binding, Continuation
    from tests.unit.pd.test_service import service
    async def run():
        cache, tier, backend = fixture(partitions=4)
        tokens = prompt(0, 513)
        warm(cache, tokens)
        evict(cache)
        backend.hold = True
        session = attach(tier)
        node, source, destination = service()
        node.cache = node.core.kv_cache_manager = cache
        node.core.config = SimpleNamespace(max_num_running_reqs=8, runtime_config=SimpleNamespace(max_seq_len=655360))
        node.adapter, calls = DeepSeekDSpark(), []
        async def worker(operation, payload):
            calls.append(operation)
            assert not tier.pending
            return {'bootstrap': 0}
        node.worker = worker
        entry = node.register_http(Binding.create(source, destination, {}), {})
        task = asyncio.create_task(node.control(source, (), 'begin', dict(binding=msgspec.to_builtins(entry.binding),
            continuation=msgspec.to_builtins(Continuation(tuple(tokens), {'max_new_tokens': 32}, None, None)))))
        await until(lambda: bool(tier.loads))
        assert entry.reservation is None and not calls
        assert all(name in ('cmp_c128', 'cmp_c4', 'idx') for _, name, _ in tier.loads)
        if cancelled:
            await node.abort(entry, stopped=True)
            with pytest.raises(ValueError, match='cancelled'):
                await task
        for future in backend.waiting:
            future.set_result(dict(bytes=1, stopped=True, native_completed=True))
        await until(lambda: not tier.loads)
        if not cancelled:
            assert (await task).hit == 512 and calls == ['reserve_state']
            await node.abort(entry, stopped=True)
        await session.close()
    asyncio.run(run())


def test_lru_reclaims_a_whole_checkpoint_after_last_reader():
    backend = Backend()
    tier = CacheTier(2, backend, lambda pin: None)
    for key in ('kv', 'state'):
        tier.capture(key, key, 1, key)
    tier.pump()
    tier.pump()
    tier.publish('checkpoint', ('kv', 'state'))
    readers = tier.acquire(('kv',))
    tier.capture('next', 'next', 1, 'next')
    tier.pump()
    assert tier.queued and 'checkpoint' in tier.units
    tier.release(readers)
    tier.pump()
    tier.pump()
    assert not tier.units and set(tier.pages) == {'next'} and tier.used == 1
