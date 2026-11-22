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
import ctypes
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

    def stop(self):
        self.call("stop")

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
    owners = [Owner(failed and i == 0) for i in range(2 * tp)]
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
        for partition in range(2):
            loaded, = backend.load(((1, ('idx', partition, partition + 1)),), request='r')
            assert loaded.result(2)['native_completed']
        for lane, extent in enumerate(backend.records[1].extents):
            load, = [
                payload for op, payload in owners[tp + lane].calls if op == 'copy_batch'
            ]
            direction, ((handle, offset, size, spans),) = load
            assert (direction, handle, (offset, size)) == ('load', 1, extent)
            assert [(span[0], span[1]) for span in spans] == [('k', 2 * 128), ('scale', 2 * 4)]
        backend.release(1).result(2)
        assert backend.extents.free == [(0, tp*4*PAGESIZE)]
        backend.close()
        assert list(backend.pop_completed('load')) == [(1, 0), (1, 1)]
        assert all(o.calls[-1][0] == 'stop' for o in owners)


def test_load_batch_is_validated_and_published_together():
    owner = Owner()
    backend = SimplerBackend([owner], dict(tp=1, groups={'idx': (('k', 1, 3, 128),)}), 3 * PAGESIZE)
    try:
        for handle in (1, 2):
            assert backend.reserve(handle, ('idx', 0, handle - 1))
            backend.store(handle, ('idx', 0, handle - 1)).result(2)
        first = (1, ('idx', 0, 1))
        with pytest.raises(ValueError, match='page outside'):
            backend.load((first, (2, ('idx', 0, 3))), request='invalid')
        assert not any(storage.active for storage in backend.records.values())
        owner.calls.clear()
        futures = backend.load((first, (2, ('idx', 0, 2))), request='valid')
        assert all(future.result(2)['native_completed'] for future in futures)
        load, = [payload for op, payload in owner.calls if op == 'copy_batch']
        assert load[0] == 'load' and [page[0] for page in load[1]] == [1, 2]
    finally:
        backend.close()


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
        endpoint.stop()
    assert len(sent) == 1 and not endpoint.stopped.value
    with pytest.raises(ValueError, match='largest page footprint'):
        SimplerBackend([], dict(tp=4, groups={'idx': (('k', 2, 7, 128),)}), PAGESIZE)


@pytest.mark.parametrize("poisoned", [False, True])
def test_registration_error_only_allows_confirmed_cleanup(poisoned):
    sent = []
    replies = iter(((1, False, ("partial import", poisoned)), (2, True, None)))
    pipe = SimpleNamespace(send=sent.append, poll=lambda _: True, recv=lambda: next(replies))
    endpoint = Endpoint(pipe, None, SimpleNamespace(value=0), threading.Lock(),
                        SimpleNamespace(value=False), SimpleNamespace(value=False))
    with pytest.raises(RuntimeError, match="partial import"):
        endpoint.call("register")
    with pytest.raises(RuntimeError, match="quarantined"):
        endpoint.call("copy_batch")
    if poisoned:
        with pytest.raises(RuntimeError, match="quarantined"):
            endpoint.call("stop")
        assert len(sent) == 1 and not endpoint.stopped.value
    else:
        endpoint.call("stop")
        assert len(sent) == 2 and endpoint.stopped.value


@pytest.mark.parametrize("poisoned", [False, True])
def test_export_is_released_only_after_cache_stop_ack(poisoned):
    from pypto_serving.model.deepseek_dspark.cache_offload import CacheOffloadExtension

    sent, released = [], []
    replies = iter(((1, False, ("partial import", False)),
                    (2, False, ("unmap failed", poisoned)), (3, True, None)))
    pipe = SimpleNamespace(send=sent.append, poll=lambda _: True, recv=lambda: next(replies))
    endpoint = Endpoint(pipe, None, SimpleNamespace(value=0), threading.Lock(),
                        SimpleNamespace(value=False), SimpleNamespace(value=False))

    def release(export):
        assert endpoint.stopped.value
        released.append(export)

    worker = SimpleNamespace(release_export=release)
    extension = CacheOffloadExtension(SimpleNamespace(_shared_l3_worker=lambda: worker), (endpoint,), None, None)
    extension.exports.append("lease")
    extension.processes.append(SimpleNamespace(join=lambda _: None, exitcode=0))
    with pytest.raises(RuntimeError, match="partial import"):
        endpoint.call("register")
    if poisoned:
        with pytest.raises(RuntimeError, match="unmap failed"):
            extension.close()
        assert len(sent) == 2 and not released and extension.exports == ["lease"]
    else:
        extension.close()
        assert len(sent) == 3 and released == ["lease"] and not extension.exports


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
        session.owners = [SimpleNamespace(stopped=SimpleNamespace(value=False), stop=stop)]
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


def test_drain_failure_stops_progress_thread_and_retains_pool():
    async def run():
        owner = Owner(failed=True)
        backend = SimplerBackend([owner], dict(tp=1, groups={'idx': (('k', 1, 2, 4),)}), PAGESIZE)
        assert backend.reserve(1, ('idx', 0, 0))
        with pytest.raises(RuntimeError, match='unknown'):
            backend.store(1, ('idx', 0, 0)).result(2)
        _, tier, _ = fixture()
        tier.backend = backend
        session, freed = attach(tier), []
        tier.begin_close = lambda: (_ for _ in ()).throw(RuntimeError('drain failed'))
        session.pool.close = lambda: freed.append('pool')
        session.copy_library.cleanup = lambda: freed.append('helper')
        with pytest.raises(RuntimeError, match='drain failed'):
            await session.close()
        assert not backend.thread.is_alive() and backend.records and not freed
        assert not any(op == 'stop' for op, _ in owner.calls)
    asyncio.run(run())


def test_copy_owner_finishing_initialization_after_timeout_stops(monkeypatch):
    from pypto_serving.serving.memory.tiering import simpler
    release, closed = threading.Event(), []
    retired = SimpleNamespace(value=False)
    monkeypatch.setattr(simpler, 'DeviceContext', lambda _: SimpleNamespace(
        bind=lambda: None, close=lambda: closed.append('context')))
    monkeypatch.setattr(simpler, 'NativeCopy', lambda _: SimpleNamespace(
        initialize_pool=lambda _: release.wait(2), close=lambda: closed.append(True)))
    monkeypatch.setenv('PYPTO_WORKER_INIT_TIMEOUT', '0')
    monkeypatch.setattr(simpler.logger, 'exception', lambda *_: release.set())
    process_main = threading.Thread(target=simpler.serve_copy,
        args=(None, SimpleNamespace(close=lambda: None), {}, retired), daemon=True)
    process_main.start()
    try:
        process_main.join(2)
        assert not process_main.is_alive() and retired.value and closed == [True, 'context']
    finally:
        release.set()


@pytest.mark.parametrize('stage,rollback_fails', [
    ('context', False), ('attach', False), ('stream', False), ('import', False),
    ('prefault', False), ('register', False),
    ('helper', False), ('helper', True)])
def test_copy_owner_initialization_rolls_back_only_acquired_stages(monkeypatch, stage, rollback_fails):
    from simpler import buffer
    from pypto_serving.serving.memory.tiering import simpler
    events, retired = [], SimpleNamespace(value=False)
    def context(device):
        events.append('context')
        if stage == 'context':
            raise RuntimeError('ACL context unavailable')
        def bind():
            events.append('attach')
            if stage == 'attach':
                raise RuntimeError('attach failed')
        return SimpleNamespace(bind=bind, close=lambda: events.append('context-close'))
    monkeypatch.setattr(simpler, 'DeviceContext', context)
    def call(name, *args):
        events.append(name)
        if name == 'aclrtCreateStream':
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(1)
        return int((stage == 'stream' and name == 'aclrtCreateStream')
                   or (stage == 'register' and name == 'aclrtHostRegisterV2')
                   or (rollback_fails and name == 'aclrtHostUnregister'))
    acl = SimpleNamespace(**{name: (lambda *args, n=name: call(n, *args)) for name in (
        'aclrtCreateStream', 'aclrtDestroyStream', 'aclrtSynchronizeStream',
        'aclrtHostRegisterV2', 'aclrtHostUnregister')})
    def acquire(name):
        events.append(name)
        if stage == name:
            raise ValueError('injected ' + name)
        return SimpleNamespace(base=1, nbytes=4096, identity='pool')
    helper = SimpleNamespace(prefault_read=lambda *args: acquire('prefault'))
    if stage != 'helper':
        helper.copy_cache_batch = lambda *args: events.append('copy')
    monkeypatch.setattr(simpler.ctypes, 'CDLL', lambda name: acl if name == 'libascendcl.so' else helper)
    monkeypatch.setattr(simpler, '_descriptor', lambda wire: wire)
    monkeypatch.setattr(simpler, 'advise', lambda *args: None)
    monkeypatch.setattr(buffer, 'ImportRegistry', lambda _: SimpleNamespace(
        materialize=lambda _: acquire('import'), unregister=lambda _: events.append('unmap')))
    service = simpler.CopyService.__new__(simpler.CopyService)
    with pytest.raises(RuntimeError, match='ACL context' if stage == 'context' else 'initialization failed'):
        service.__init__(None, SimpleNamespace(close=lambda: None), {'copy_library': 'helper'}, retired)
    if stage == 'context':
        assert retired.value and not hasattr(service, 'thread')
        return
    service.thread.join(2)
    assert not service.thread.is_alive() and 'copy' not in events
    assert ('aclrtHostUnregister' in events) == (stage == 'helper')
    assert ('unmap' in events) == (stage not in ('attach', 'stream', 'import') and not rollback_fails)
    assert ('aclrtDestroyStream' in events) == (stage not in ('attach', 'stream') and not rollback_fails)
    if rollback_fails:
        assert not retired.value and isinstance(service.error, AttributeError)
        with pytest.raises(RuntimeError, match='not been safely retired'):
            service.close()
    else:
        assert retired.value
        service.close()
