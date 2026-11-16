# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import sys

import pytest

from pypto_serving.tools.profile import request as trace


def test_semantic_await_thread_reuse_and_retirement(tmp_path):
    capture = trace.Capture("host-one", tmp_path, "node", "replica")
    namespace = dict(asyncio=asyncio)
    exec(compile("async def work():\n    await asyncio.sleep(.04)\n    return 7\n",
                 "/app/pypto_serving/sample.py", "exec"), namespace)
    async def run(root):
        with trace.scope(capture.wire((root,)), "logical.await"):
            return await namespace['work']()
    async def both():
        return await asyncio.gather(run("first"), run("second"))
    assert asyncio.run(both()) == [7, 7]
    assert sys.getprofile() is None
    tag = capture.wire(("thread",))
    def job(selected):
        with trace.scope(tag if selected else None, "thread.job"):
            trace.event("picked")
        return sys.getprofile()
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(job, True).result() is None
        assert pool.submit(job, False).result() is None
    trace.bind("resource", tag)
    assert not capture.close()
    trace.release("resource")
    assert capture.close()
    rows = [json.loads(line) for line in capture.path.read_text().splitlines()]
    assert not any(r.get('cat') in ('python', 'native-call') for r in rows)
    assert not any(r['name'] == 'work' for r in rows)
    waits = [r for r in rows if r['name'] == 'logical.await']
    assert len(waits) == 2 and all(r['dur'] >= 35000 for r in waits)
    assert len([r for r in rows if r['name'] == 'picked']) == 1
    assert all(r['ts'] > 10**15 for r in rows if 'ts' in r)
    assert json.loads(capture.path.with_suffix('.status.json').read_text())['complete']


def test_shared_work_keeps_only_selected_members_until_completion(tmp_path):
    capture = trace.Capture("host-two", tmp_path, "node", "decode")
    trace.bind("selected", capture.wire(("root",)))
    tag = trace.context(("not-selected", "selected"))
    assert tag['roots'] == ['root']
    trace.track("copy", "job", tag)
    trace.release("selected")
    assert not capture.close()
    trace.finish("copy", "job")
    assert capture.close()
    assert trace.context(("selected",)) is None
    rows = [json.loads(line) for line in capture.path.read_text().splitlines()]
    lifetimes = [row for row in rows if row["name"] == "copy.lifetime"]
    assert len(lifetimes) == 1 and lifetimes[0]["dur"] >= 0


def test_http_selection_stop_drain_and_repeat(tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from httpx import ASGITransport, AsyncClient
    from pypto_serving.tools.profile.http import Controller
    monkeypatch.setenv('PYPTO_REQUEST_PROFILE_ROOT', str(tmp_path))
    async def scenario():
        app = FastAPI()
        controller = Controller(node='ingress')
        controller.install(app)
        entered, proceed, retired = asyncio.Event(), asyncio.Event(), asyncio.Event()
        @app.post('/v1/completions')
        async def generate(request: Request):
            body = await request.body()
            if body == b'hold':
                trace.bind('late-resource')
                entered.set()
                await proceed.wait()
                async def cleanup():
                    await retired.wait()
                    trace.release('late-resource')
                asyncio.create_task(cleanup())
            return dict(ok=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.post('/v1/completions', content='old')).status_code == 200
            assert (await client.post('/start_request_profile')).status_code == 200
            selected = asyncio.create_task(client.post('/v1/completions', content='hold'))
            await entered.wait()
            stopped = await client.post('/stop_request_profile')
            assert stopped.status_code == 202 and not stopped.json()['accepting_new']
            assert (await client.post('/v1/completions', content='not-selected')).status_code == 200
            proceed.set()
            assert (await selected).status_code == 200
            assert not controller.capture.closed
            retired.set()
            await asyncio.wait_for(controller.task, 5)
            result = (await client.post('/stop_request_profile')).json()['result']
            assert result['complete']
            events = json.loads(open(result['trace']).read())['traceEvents']
            assert len([e for e in events if e['name'] == 'http.ingress']) == 1
            assert any(e['name'] == 'request.resources_retired' for e in events)
            assert all(isinstance(e['pid'], int) and isinstance(e['tid'], int) for e in events)
            assert len(json.loads(open(result['requests']).read())) == 1
            assert (await client.post('/start_request_profile')).status_code == 200
            await client.post('/stop_request_profile')
            await asyncio.wait_for(controller.task, 5)
    asyncio.run(scenario())


def test_default_off_headers_and_loss_reporting(tmp_path, monkeypatch):
    from pypto_serving.router.http import request_headers
    monkeypatch.setattr(trace, 'FILE_BYTES', 1024)
    capture = trace.Capture('bounded', tmp_path, 'node', 'replica')
    marker = json.dumps(dict(capture=capture.id, roots=['a'*32]))
    assert trace.HEADER not in request_headers({trace.HEADER: marker})
    with trace.scope(capture.wire(('a'*32,)), 'selected'):
        assert json.loads(request_headers({})[trace.HEADER])['roots'] == ['a'*32]
        for _ in range(50):
            trace.event('padding', count=123)
    assert trace.current.get() is None
    assert capture.close()
    status = json.loads(capture.path.with_suffix('.status.json').read_text())
    assert status['dropped'] > 0 and not status['complete']


def test_unknown_exports_incomplete_without_releasing_facts(tmp_path, monkeypatch):
    from pypto_serving.tools.profile.http import Controller
    monkeypatch.setenv('PYPTO_REQUEST_PROFILE_ROOT', str(tmp_path))
    async def scenario():
        controller = Controller(node='fault')
        await controller.start()
        tag = controller.capture.wire(('b'*32,))
        trace.bind('unknown-write', tag, owner='pd')
        trace.incomplete('native_completion_unknown', tag)
        await controller.stop()
        await asyncio.wait_for(controller.task, 5)
        assert controller.result['complete'] is False
        manifest = json.loads(open(controller.result['manifest']).read())
        assert manifest['local']['pending'] == 1
        assert trace.context(('unknown-write',)) == tag
        trace.release('unknown-write', owner='pd')
    asyncio.run(scenario())


def test_mixed_batch_does_not_attach_unselected_work(tmp_path):
    capture = trace.Capture('mixed', tmp_path, 'node', 'worker')
    tag = capture.wire(('c'*32,))
    with trace.scope(tag, 'physical.batch'):
        trace.event('selected')
        trace.event('unselected', tag=None)
        with trace.scope(None):
            trace.event('unselected.inner')
        trace.event('selected.again')
    assert capture.close()
    names = {json.loads(line)['name'] for line in capture.path.read_text().splitlines()}
    assert {'physical.batch','selected','selected.again'} <= names
    assert not {'unselected','unselected.inner'} & names


def test_forked_owner_uses_own_writer(tmp_path):
    import multiprocessing
    context = multiprocessing.get_context('fork')
    capture = trace.Capture('forked', tmp_path, 'node', 'replica')
    tag = capture.wire(('d'*32,))
    def owner():
        with trace.scope(tag, 'owner.native'):
            trace.event('owner.return')
        assert trace.close(tag['capture'])
    process = context.Process(target=owner)
    process.start()
    process.join(5)
    assert process.exitcode == 0 and capture.close()
    statuses = [json.loads(p.read_text()) for p in tmp_path.glob('*.status.json')]
    assert len(statuses) == 2 and all(s['complete'] for s in statuses)
    assert len({s['process'] for s in statuses}) == 2


def test_ingress_selection_precedes_slow_body_and_ignores_old_request(tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from httpx import ASGITransport, AsyncClient
    from pypto_serving.tools.profile.http import Controller
    monkeypatch.setenv('PYPTO_REQUEST_PROFILE_ROOT', str(tmp_path))
    async def scenario():
        app, controller = FastAPI(), Controller(node='slow-body')
        controller.install(app)
        @app.post('/v1/chat/completions')
        async def generate(request: Request):
            await request.body()
            return dict(ok=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            gates = [asyncio.Event() for _ in range(4)]
            async def body(entered, proceed):
                entered.set()
                await proceed.wait()
                yield b'private-content-must-not-be-recorded'
            old = asyncio.create_task(client.post('/v1/chat/completions', content=body(*gates[:2])))
            await gates[0].wait()
            await controller.start()
            selected = asyncio.create_task(client.post('/v1/chat/completions', content=body(*gates[2:])))
            await gates[2].wait()
            await controller.stop()
            gates[1].set()
            gates[3].set()
            assert all(r.status_code == 200 for r in await asyncio.gather(old, selected))
            await asyncio.wait_for(controller.task,5)
            text = open(controller.result['trace']).read()
            rows = json.loads(text)['traceEvents']
            assert len([r for r in rows if r['name'] == 'http.ingress']) == 1
            assert 'private-content-must-not-be-recorded' not in text
    asyncio.run(scenario())


def test_stop_serializes_with_preparation_and_does_not_install_hooks(tmp_path, monkeypatch):
    from pypto_serving.tools.profile.http import Controller
    monkeypatch.setenv('PYPTO_REQUEST_PROFILE_ROOT',str(tmp_path))
    async def scenario():
        entered,proceed=asyncio.Event(),asyncio.Event()
        async def workers(active,capture):
            if active:
                entered.set()
                await proceed.wait()
        controller=Controller(node='linear',workers=workers)
        start=asyncio.create_task(controller.start())
        await entered.wait()
        stop=asyncio.create_task(controller.stop())
        await asyncio.sleep(0)
        assert not stop.done()
        proceed.set()
        await start
        await stop
        await controller.task
        assert not controller.accepting and controller.result['complete']
        hook=lambda *args: None
        sys.setprofile(hook)
        try:
            await controller.start()
            with trace.scope(controller.capture.wire(('f'*32,)), 'explicit'):
                assert sys.getprofile() is hook
            await controller.stop()
            await controller.task
            assert sys.getprofile() is hook
        finally:
            sys.setprofile(None)
    asyncio.run(scenario())


def test_real_ack_releases_finished_tombstone_but_not_preempted_request(tmp_path):
    from types import SimpleNamespace
    from pypto_serving.serving.engine.async_engine import ReplicaEngineCore
    from pypto_serving.serving.sched.scheduler import Request, RequestStatus
    capture=trace.Capture('retained-finished',tmp_path,'node','replica')
    tag=capture.wire(('e'*32,))
    trace.bind('finished',tag)
    trace.bind('preempted',tag)
    core=ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.scheduler=SimpleNamespace(requests={
        'finished':Request('finished',[1],max_new_tokens=1,status=RequestStatus.FINISHED_LENGTH),
        'preempted':Request('preempted',[1],max_new_tokens=1,status=RequestStatus.PREEMPTED)})
    core._request_contexts={}
    core._acknowledge_worker_frees(('finished','preempted'))
    assert trace.context(('finished',)) is None
    assert trace.context(('preempted',)) == tag and capture.pending()==1
    core.scheduler.requests.pop('preempted')
    core._acknowledge_worker_frees(('preempted',))
    assert capture.close()


def test_clock_bounds_exclude_remote_handler_time():
    from pypto_serving.tools.profile.http import clock_sample
    # Remote clock is +100ns, with symmetric 5ns network legs and a 1s handler.
    sample=clock_sample(1000,1_000_001_010,dict(received_ns='1105',wall_ns='1000001105'))
    assert sample['valid'] and sample['offset_estimate_ns']==100 and sample['uncertainty_ns']==5
    assert not clock_sample(100,101,dict(received_ns='100',wall_ns='200'))['valid']


def test_peer_export_and_missing_node_are_explicit(tmp_path,monkeypatch):
    import httpx
    from pypto_serving.tools.profile.http import Controller
    monkeypatch.setenv('PYPTO_REQUEST_PROFILE_ROOT',str(tmp_path))
    real=httpx.AsyncClient
    async def scenario(failed):
        def handler(request):
            import time
            if request.url.path.endswith('/prepare'):
                stamp=str(time.time_ns())
                return httpx.Response(200,json=dict(wall_ns=stamp,received_ns=stamp))
            if request.url.path.endswith('/seal'):
                if failed:return httpx.Response(503)
                stamp=str(time.time_ns())
                return httpx.Response(200,json=dict(complete=True,wall_ns=stamp,received_ns=stamp,
                    processes=[dict(process='peer',complete=True)]))
            return httpx.Response(200,content=json.dumps(dict(name='peer',ph='i',s='t',
                pid='peer',tid='worker',ts=time.time_ns()//1000,args={})).encode()+b'\n')
        monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
        controller=Controller(node='collector',peers=lambda:('http://peer',))
        await controller.start()
        await controller.stop()
        await asyncio.wait_for(controller.task,5)
        assert controller.result['complete'] is not failed
        manifest=json.loads(open(controller.result['manifest']).read())
        assert bool(manifest['missing']) is failed
    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


@pytest.mark.parametrize('selected_join', [False, True])
def test_shared_load_keeps_capture_until_native_reap_after_waiter_cancel(tmp_path, selected_join):
    from tests.unit.serving.memory.test_cache_offload import fixture, warm, evict, attach, until, prompt
    capture = trace.Capture('shared-load', tmp_path, 'node', 'replica')
    async def scenario():
        cache, tier, backend = fixture()
        tokens = prompt(0, 513)
        boundary = warm(cache, tokens)
        evict(cache)
        backend.hold = True
        session = attach(tier)
        trace.bind('a', capture.wire(('a'*32,)))
        first = asyncio.create_task(tier.restore('a', cache.compute_group_block_hashes(tokens), len(tokens), maximum=boundary))
        await until(lambda: bool(tier.loads))
        if selected_join:
            trace.bind('b', capture.wire(('b'*32,)))
        second = asyncio.create_task(tier.restore('b', cache.compute_group_block_hashes(tokens), len(tokens), maximum=boundary))
        await until(lambda: len(tier.pending) == 2)
        assert len(backend.waiting) == len(tier.loads)
        tier.cancel('a')
        with pytest.raises(ValueError, match='cancelled'):
            await first
        trace.release('a')
        assert capture.pending() and not capture.close()
        for future in backend.waiting:
            future.set_result(dict(bytes=1, stopped=True, native_completed=True))
        await second
        await until(lambda: not tier.loads)
        trace.release('b')
        await session.close()
    asyncio.run(scenario())
    assert capture.close()
    events = [json.loads(line) for line in capture.path.read_text().splitlines()]
    completed = [e for e in events if e['name'] == 'ddr.restore.complete']
    assert completed and all(set(e['args']['roots']) == {'a'*32, *(['b'*32] if selected_join else [])} for e in completed)


def test_batch_purpose_membership_and_explicit_generation_endpoint(tmp_path):
    from pypto_serving.tools.profile.export import merge
    capture = trace.Capture('batch-purposes', tmp_path, 'node', 'worker')
    for name, root in (('p', 'a'), ('d', 'b'), ('done', 'c')):
        trace.bind(name, capture.wire((root*32,)))
    tag = trace.batch(7, prefill=('p',), decode=('d', 'unselected'), register=('p',), release=('done',))
    assert trace.part(tag, 'compute')['roots'] == ['a'*32, 'b'*32]
    assert trace.part(tag, 'release')['roots'] == ['c'*32]
    with trace.scope(trace.part(tag, 'compute'), 'physical.batch'):
        with trace.scope(trace.subset(('p',)), 'bootstrap'):
            trace.event('generation.finished', request_id='p')
        with trace.scope(trace.subset(('d',)), 'decode'):
            trace.event('engine.tokens')
    with trace.scope(trace.part(tag, 'release'), 'cleanup'):
        trace.event('http.end')
    for name in ('p', 'd', 'done'):
        trace.release(name)
    assert capture.close()
    rows = [json.loads(line) for line in capture.path.read_text().splitlines()]
    assert next(row for row in rows if row['name'] == 'bootstrap')['args']['roots'] == ['a'*32]
    assert next(row for row in rows if row['name'] == 'decode')['args']['roots'] == ['b'*32]
    result = merge(tmp_path)
    requests = json.loads(open(result['requests']).read())
    assert 'generation.finished' in requests['a'*32]['milestones']
    assert 'generation.finished' not in requests['b'*32]['milestones']
    assert 'http.end' not in requests['a'*32]['milestones']


def test_semantic_hooks_do_not_trace_business_calls_or_change_timestamp(tmp_path):
    import time
    capture = trace.Capture('lightweight', tmp_path, 'node', 'worker')
    with trace.scope(capture.wire(('a'*32,)), 'sdk.submit'):
        for i in range(10000):
            assert len((i,)) == 1
        before = time.time_ns() // 1000
        trace.event('native.submitted')
        after = time.time_ns() // 1000
    assert sys.getprofile() is None and capture.close()
    rows = [json.loads(line) for line in capture.path.read_text().splitlines()]
    assert len(rows) < 10
    assert before <= next(row['ts'] for row in rows if row['name'] == 'native.submitted') <= after


def test_worker_capture_control_uses_semantic_observer_only(tmp_path):
    from queue import Queue
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    from pypto_serving.serving.server.ipc import ProfileCommand, decode_profile_result
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._services = None
    worker.profile_output_queue = Queue()
    tag = dict(capture='worker-control',directory=str(tmp_path),node='node',role='worker',roots=[])
    for active in (True, False):
        worker._handle_profile_command(ProfileCommand(active=active, request_capture=tag))
        result = decode_profile_result(worker.profile_output_queue.get_nowait())
        assert result.error is None and result.active == active
        assert bool(result.request_processes) is active
        assert sys.getprofile() is None


def test_partial_worker_preparation_is_rolled_back(tmp_path, monkeypatch):
    from pypto_serving.tools.profile.http import Controller
    monkeypatch.setenv('PYPTO_REQUEST_PROFILE_ROOT', str(tmp_path))
    async def scenario():
        calls = []
        async def worker(active, capture):
            calls.append(active)
            if active:
                raise RuntimeError('worker unavailable')
        controller = Controller(node='partial', workers=worker)
        with pytest.raises(RuntimeError, match='unavailable'):
            await controller.start()
        assert calls[0] is True and not any(calls[1:])
        assert controller.capture.closed and not controller.capture._writer.is_alive()
        assert not controller.accepting
    asyncio.run(scenario())
