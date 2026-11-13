# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Batch/TP completion, diagnostics and the engine-loop lifecycle seam."""
import asyncio
from mmap import PAGESIZE
import threading
from types import SimpleNamespace

import pytest

from pypto_serving.serving.memory.tiering.backend import Endpoint, SimplerBackend
from tests.unit.serving.memory.test_cache_offload import attach, fixture, prompt, until, warm


class Owner:
    def __init__(self, failed=False):
        self.failed, self.calls = failed, []
        self.parent = SimpleNamespace(close=lambda: None)

    def call(self, operation, payload=None):
        self.calls.append((operation, payload))
        if operation == 'copy_batch':
            if self.failed:
                raise RuntimeError('native result unknown')
            return dict(pages=[dict(handle=h, bytes=sum(s[3]*s[4] for s in spans),
                stopped=True, native_completed=True) for h, _, _, spans in payload[1]],
                prepare_start_ns=0, prepare_end_ns=1, started_ns=2, submitted_ns=3,
                completed_ns=4, native_commands=len(payload[1]))


@pytest.mark.parametrize('tp,failed', [(1, False), (4, False), (4, True)])
def test_tp_index_scale_completion_and_unknown_retirement(tp, failed):
    owners = [Owner(failed and i == 0) for i in range(tp)]
    backend = SimplerBackend(owners, dict(tp=tp, groups={'idx': (('k', 2, 7, 128), ('scale', 2, 7, 4))}),
                             tp*4*PAGESIZE)
    assert backend.reserve(1, ('idx', 0, 0)) and len(set(backend.records[1].extents)) == tp
    future = backend.store(1, ('idx', 0, 0))
    if failed:
        with pytest.raises(RuntimeError, match='unknown'):
            future.result(2)
        with pytest.raises(RuntimeError, match='uncertain'):
            backend.close()
        assert backend.records and not any(op == 'stop' for o in owners for op, _ in o.calls)
    else:
        assert future.result(2)['bytes'] == tp*264
        assert backend.load(1, ('idx', 0, 1), request='r').result(2)['native_completed']
        backend.release(1).result(2)
        assert backend.extents.free == [(0, tp*4*PAGESIZE)]
        backend.close()
        assert all(o.calls[-1][0] == 'stop' for o in owners)


def test_completion_releases_copy_refs_before_model_result_and_ignores_late_notifications():
    async def run():
        cache, tier, backend = fixture()
        backend.hold = True
        session = attach(tier)
        model = asyncio.get_running_loop().create_future()
        warm(cache, prompt(0, 129))
        assert tier.tier.inflight and any(b.ref_cnt for p in cache._group_pools.values() for b in p.blocks)
        await asyncio.to_thread(lambda: [f.set_result(dict(bytes=1, stopped=True, native_completed=True))
                                        for f in backend.waiting])
        await until(lambda: not tier.tier.inflight)
        assert not model.done() and all(not b.ref_cnt for p in cache._group_pools.values() for b in p.blocks)
        session._notify()
        await session.close()
        session._notify()
        session._consume()
        await asyncio.sleep(0)
        assert cache.tiering is None and tier.tier.used == 0
        model.set_result(None)
    asyncio.run(run())


def test_timeout_quarantines_endpoint_before_any_later_send():
    sent = []
    pipe = SimpleNamespace(send=sent.append, poll=lambda timeout: False)
    endpoint = Endpoint(pipe, None, SimpleNamespace(value=0), threading.Lock(),
                        SimpleNamespace(value=False), SimpleNamespace(value=False))
    with pytest.raises(RuntimeError, match='response timeout'):
        endpoint.call('copy_batch')
    with pytest.raises(RuntimeError, match='quarantined'):
        endpoint.call('stop')
    assert len(sent) == 1 and not endpoint.stopped.value
    with pytest.raises(ValueError, match='largest page footprint'):
        SimplerBackend([], dict(tp=4, groups={'idx': (('k', 2, 7, 128),)}), PAGESIZE)


@pytest.mark.parametrize('started,failed', [(False, False), (True, False), (True, True)])
def test_partial_startup_releases_pool_only_after_owner_ack(started, failed):
    async def run():
        _, tier, _ = fixture()
        session, events = attach(tier), []
        def stop(*args):
            events.append('owner')
            if failed:
                raise RuntimeError('stop unknown')
        session.cache, session._worker_started = None, started
        session.owners = [SimpleNamespace(stopped=SimpleNamespace(value=False), call=stop)]
        session.pool.close = lambda: events.append('pool')
        session.copy_library.cleanup = lambda: events.append('helper')
        if failed:
            with pytest.raises(RuntimeError, match='stop unknown'):
                await session.close()
            assert events == ['owner']
        else:
            await session.close()
            assert events == (['owner'] if started else []) + ['pool', 'helper']
    asyncio.run(run())
