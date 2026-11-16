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
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import pytest

from pypto_serving.serving.memory.tiering.backend import Endpoint, SimplerBackend, endpoints
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


def test_rpc_trace_preserves_pipe_completion_and_causal_timestamps():
    endpoint, = endpoints(mp.get_context("spawn"), 1)
    records = []
    def serve():
        while True:
            sequence, operation, payload = endpoint.child.recv()
            received = time.monotonic_ns()
            result = Owner().call(operation, payload)
            if operation == "copy_batch":
                result.update({key: received for key in ("prepare_start_ns", "prepare_end_ns", "started_ns",
                                                         "submitted_ns", "completed_ns")})
                result["rpc_timing"] = dict(owner_received_ns=received,
                                            owner_reply_start_ns=time.monotonic_ns())
            endpoint.child.send((sequence, True, result))
            if operation == "stop":
                endpoint.child.close()
                return
    owner = threading.Thread(target=serve)
    owner.start()
    trace = SimpleNamespace(emit=lambda event, **fields: records.append((event, fields)))
    backend = SimplerBackend((endpoint,), dict(tp=1, groups={'idx': (('k', 2, 7, 128),)}),
                             4 * PAGESIZE, trace=trace)
    try:
        assert backend.reserve(1, ('idx', 0, 0))
        assert backend.store(1, ('idx', 0, 0)).result(2)['native_completed']
        backend.release(1).result(2)
    finally:
        backend.close()
        owner.join(2)
    row, = [r for event, r in records if event == "batch"]
    assert not owner.is_alive() and endpoint.stopped.value and not endpoint.failed.value
    parent = [row[k] for k in ("rpc_start_ns", "rpc_task_start_ns", "rpc_lock_acquired_ns",
                               "rpc_send_start_ns", "rpc_send_end_ns", "rpc_recv_end_ns",
                               "rpc_validated_ns", "rpc_end_ns")]
    assert parent == sorted(parent)
    assert row['rpc_send_start_ns'] <= row['owner_received_ns'] <= row['prepare_start_ns']
    assert row['completed_ns'] <= row['owner_reply_start_ns'] <= row['rpc_recv_end_ns']


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


@pytest.mark.parametrize('outcome', ['ready', 'cancel', 'cancel_failed_constructor'])
def test_ssd_startup_remains_owned_until_archive_retirement(monkeypatch, outcome):
    from pypto_serving.serving.memory.tiering import archive, integration
    async def run():
        cache, tier, backend = fixture()
        session, events = attach(tier), []
        entered, release = threading.Event(), threading.Event()
        session.owners = [SimpleNamespace(child=SimpleNamespace(close=lambda: None))]
        session.metadata = SimpleNamespace(get=lambda **kw: dict(ranks=1, tp=1, groups=cache.group_names))
        session.core.config = SimpleNamespace(runtime_config=SimpleNamespace(kv_ssd_bytes=4096, kv_ssd_path='ssd'))
        session.pool.nbytes = 200_000
        session.pool.close = lambda: events.append('pool')
        session.copy_library.cleanup = lambda: events.append('helper')
        def construct(*args):
            entered.set()
            assert release.wait(2)
            assert not events  # Construction must never outlive the backing.
            if outcome == 'cancel_failed_constructor':
                raise OSError('injected SSD initialization failure')
            return SimpleNamespace(name='ssd', work_bytes=0, window_bytes=0,
                begin_close=lambda: None, pump=lambda: None, pop_loaded=lambda: (),
                idle=lambda: True, close=lambda: events.append('archive'))
        monkeypatch.setattr(integration, 'SimplerBackend', lambda *args, **kw: backend)
        monkeypatch.setattr(integration, 'TieredCache', lambda *args, **kw: tier)
        monkeypatch.setattr(integration.OffloadTrace, 'from_env', lambda: None)
        monkeypatch.setattr(archive, 'SSDArchive', construct)
        starting = asyncio.create_task(session.start())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            if outcome != 'ready':
                starting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await starting
                closing = asyncio.create_task(session.close())
                await asyncio.sleep(.01)
                assert not closing.done() and not events
            else:
                release.set()
                await starting
                closing = asyncio.create_task(session.close())
        finally:
            release.set()
        await asyncio.wait_for(closing, 2)
        assert events == ([] if outcome == 'cancel_failed_constructor' else ['archive']) + ['pool', 'helper']
        assert cache.tiering is None and session._archive_start.done()
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
    def current_context(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(1)
        return 0
    acl = SimpleNamespace(aclrtGetCurrentContext=current_context, aclrtSetCurrentContext=lambda _: 0)
    monkeypatch.setattr(simpler.ctypes, 'CDLL', lambda _: acl)
    monkeypatch.setattr(simpler, 'NativeCopy', lambda _: SimpleNamespace(
        initialize_pool=lambda _: release.wait(2), close=lambda: closed.append(True)))
    monkeypatch.setenv('PYPTO_WORKER_INIT_TIMEOUT', '0')
    service = simpler.CopyService.__new__(simpler.CopyService)
    try:
        with pytest.raises(RuntimeError, match='initialization failed'):
            service.__init__(None, SimpleNamespace(close=lambda: None), {}, SimpleNamespace(value=False))
    finally:
        release.set()
        service.thread.join(2)
    assert not service.thread.is_alive() and closed == [True]


@pytest.mark.parametrize('stage,rollback_fails', [
    ('context', False), ('attach', False), ('stream', False), ('import', False),
    ('prefault', False), ('register', False),
    ('helper', False), ('helper', True)])
def test_copy_owner_initialization_rolls_back_only_acquired_stages(monkeypatch, stage, rollback_fails):
    from simpler import buffer
    from pypto_serving.serving.memory.tiering import simpler
    events, retired = [], SimpleNamespace(value=False)
    def call(name, *args):
        events.append(name)
        if name in ('aclrtGetCurrentContext', 'aclrtCreateStream'):
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(1)
        return int((stage == 'context' and name == 'aclrtGetCurrentContext')
                   or (stage == 'attach' and name == 'aclrtSetCurrentContext')
                   or (stage == 'stream' and name == 'aclrtCreateStream')
                   or (stage == 'register' and name == 'aclrtHostRegisterV2')
                   or (rollback_fails and name == 'aclrtHostUnregister'))
    acl = SimpleNamespace(**{name: (lambda *args, n=name: call(n, *args)) for name in (
        'aclrtGetCurrentContext', 'aclrtSetCurrentContext', 'aclrtCreateStream', 'aclrtDestroyStream',
        'aclrtSynchronizeStream', 'aclrtHostRegisterV2', 'aclrtHostUnregister')})
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
