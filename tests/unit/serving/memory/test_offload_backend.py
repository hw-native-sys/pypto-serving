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
import json
from mmap import PAGESIZE
import threading
from types import SimpleNamespace

import pytest

from pypto_serving.serving.memory.tiering.backend import SimplerBackend
from pypto_serving.serving.memory.tiering.trace import OffloadTrace
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
def test_tp_index_scale_completion_and_unknown_retirement(tmp_path, tp, failed):
    owners = [Owner(failed and i == 0) for i in range(tp)]
    trace = OffloadTrace(tmp_path)
    backend = SimplerBackend(owners, dict(tp=tp, groups={'idx': (('k', 2, 7, 128), ('scale', 2, 7, 4))}),
                             tp*4*PAGESIZE, trace=trace)
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
    trace.close()
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert rows[-1]['complete'] and not rows[-1]['dropped_events']
    assert any(r['event'] == 'batch_failed' for r in rows) == failed
    if not failed:
        assert sum(r['bytes'] for r in rows if r['event'] == 'batch' and r['direction'] == 'load') == tp*264


@pytest.mark.parametrize('layers,width,pages,expected', [(20, 1024, 512, [1, 512]), (1, 1 << 20, 260, [1, 256, 4])])
def test_batch_bytes_compact_spans_and_one_notification_per_batch(layers, width, pages, expected):
    owner, entered, proceed = Owner(), threading.Event(), threading.Event()
    original = owner.call
    def call(operation, payload=None):
        if operation == 'copy_batch' and not entered.is_set():
            entered.set()
            assert proceed.wait(5)
        return original(operation, payload)
    owner.call = call
    backend = SimplerBackend([owner], dict(tp=1, groups={'kv': (('k', layers, pages+1, width),)}),
                             (pages+1)*layers*width)
    notifications = []
    backend.set_notify(lambda: notifications.append(True))
    try:
        futures = []
        for i in range(pages+1):
            assert backend.reserve(i+1, ('kv', 0, i))
            futures.append(backend.store(i+1, ('kv', 0, i)))
            if i == 0:
                assert entered.wait(5)
        proceed.set()
        assert all(f.result(5)['native_completed'] for f in futures)
    finally:
        proceed.set()
        backend.close()
    assert [len(payload[1]) for op, payload in owner.calls if op == 'copy_batch'] == expected
    assert len(notifications) == len(expected)


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
