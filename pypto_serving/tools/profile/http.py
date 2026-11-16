# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request selection at ASGI ingress, with HTTP-only distributed collection."""

import asyncio
import json
import os
from pathlib import Path
import socket
import time
import uuid

from . import request as trace
from .export import merge

PATHS = frozenset(('/v1/completions','/v1/chat/completions'))


class RequestProfileMiddleware:
    def __init__(self, app, controller):
        self.app, self.controller = app, controller

    async def __call__(self, scope, receive, send):
        controller = self.controller
        if scope['type'] != 'http' or scope['path'] not in PATHS:
            return await self.app(scope, receive, send)
        tag = None
        if controller.accepting:
            tag = controller.capture.wire((uuid.uuid4().hex,))
        elif controller.prepared_by_peer and controller.capture is not None and not controller.capture.closed:
            raw = dict(scope.get('headers',())).get(trace.HEADER.encode())
            if raw:
                try:
                    tag = trace.incoming(json.loads(raw))
                except (ValueError, KeyError, TypeError):
                    pass
        if tag is None:
            return await self.app(scope, receive, send)
        status, ended, first = None, False, False
        async def read():
            with trace.scope(tag, 'http.receive'):
                message = await receive()
            trace.event('http.received', type=message['type'], bytes=len(message.get('body',b'')))
            return message
        async def write(message):
            nonlocal status, ended, first
            if message['type']=='http.response.start':
                status = message['status']
            if message['type']=='http.response.body':
                if not first and message.get('body'):
                    trace.event('http.first_body')
                    first = True
                ended = not message.get('more_body',False)
            with trace.scope(tag, 'http.send', type=message['type'], bytes=len(message.get('body',b''))):
                await send(message)
        with trace.scope(tag, 'http.request', path=scope['path']):
            trace.event('http.ingress', path=scope['path'], node=controller.node, role=controller.role)
            try:
                await self.app(scope, read, write)
            finally:
                trace.event('http.end', status=status, body_completed=ended)


class Controller:
    def __init__(self, *, node=None, role='replica', peers=lambda: (), workers=None):
        self.node = node or socket.gethostname()+'-'+uuid.uuid4().hex[:8]
        self.role, self.peers, self.workers = role, peers, workers
        self.capture = None
        self.accepting = False
        self.task = None
        self.remote = {}
        self.expected = []
        self.result = None
        self.prepared_by_peer = False
        self._lock = asyncio.Lock()

    async def prepare(self, capture_id):
        if self.capture is not None and (not self.capture.closed or self.capture.pending()):
            if self.capture.id == capture_id:
                return
            raise ValueError('request capture is still active')
        from .recorder import get_profiler
        if get_profiler(initially_active=False).active:
            raise ValueError('time profiling is active')
        root = Path(os.environ.get('PYPTO_REQUEST_PROFILE_ROOT','serving_log/request_profile'))
        self.capture = trace.Capture(capture_id, root/capture_id/self.node/self.role, self.node, self.role)
        self.result = None
        if self.capture.failure:
            raise ValueError("request profile output directory is unavailable")
        if self.workers is not None:
            try:
                self.expected = await self.workers(True, self.capture.wire()) or []
            except BaseException:
                self.capture.failure = "worker_prepare_failed"
                try:
                    await self.workers(False, self.capture.wire())
                finally:
                    await asyncio.to_thread(self.capture.close, incomplete=True)
                raise

    async def start(self):
        import httpx
        async with self._lock:
            if self.capture is not None and (not self.capture.closed or self.capture.pending() or self.task is not None and not self.task.done()):
                raise ValueError('previous request capture has not finished')
            capture_id = time.strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:8]
            self.remote = {}
            try:
                await self.prepare(capture_id)
                async with httpx.AsyncClient(trust_env=False,timeout=60) as client:
                    for url in dict.fromkeys(self.peers()):
                        sent = time.time_ns()
                        self.remote[url] = {}  # Include partial peer preparation in rollback.
                        response = await client.post(url+'/internal/request_profile/prepare',json=dict(capture=capture_id))
                        response.raise_for_status()
                        received = time.time_ns()
                        value = response.json()
                        self.remote[url] = clock_sample(sent, received, value)
                self.prepared_by_peer = False
                self.accepting = True
                return self.status()
            except BaseException:
                self.accepting = False
                if self.capture is not None:
                    if self.workers is not None:
                        await self.workers(False, self.capture.wire())
                    await asyncio.to_thread(self.capture.close)
                async with httpx.AsyncClient(trust_env=False,timeout=30) as client:
                    await asyncio.gather(*(client.post(u+'/internal/request_profile/seal',
                        json=dict(capture=capture_id)) for u in self.remote),return_exceptions=True)
                raise

    def status(self):
        return dict(capture=self.capture.id if self.capture else None, accepting_new=self.accepting,
            directory=str(self.capture.directory) if self.capture else None,
            pending=self.capture.pending() if self.capture is not None else 0,
            state=('complete' if self.result.get('complete') else 'incomplete') if self.result else
                  'armed' if self.accepting else 'draining',
            result=self.result)

    async def seal(self):
        if self.capture is None:
            return dict(complete=True)
        if self.capture.pending() and not self.capture.failure:
            return dict(complete=False, pending=self.capture.pending())
        if self.workers is not None and not self.capture.closed and not self.capture.failure:
            try:
                await asyncio.wait_for(self.workers(False, self.capture.wire()), 30)
            except BlockingIOError:
                return dict(complete=False, pending="worker_scopes")
            except Exception as error:
                self.capture.failure = "worker_seal_" + type(error).__name__
        closed = await asyncio.to_thread(self.capture.close, incomplete=bool(self.capture.failure))
        statuses = [json.loads(p.read_text()) for p in self.capture.directory.glob('*.status.json')]
        missing = sorted({self.capture.key, *self.expected} - {row["process"] for row in statuses})
        return dict(complete=closed, processes=statuses, missing=missing,
                    error=self.capture.failure, pending=self.capture.pending(), wall_ns=str(time.time_ns()))

    async def stop(self):
        async with self._lock:
            self.accepting = False
            if self.capture is None:
                raise ValueError('request profiling has not started')
            if self.task is None or self.task.done() and self.result is None:
                self.task = asyncio.create_task(self._finalize())
            # Fast completion is useful for an idle capture; never wait for a request timeout.
            await asyncio.sleep(0)
            return self.status()

    async def _finalize(self):
        import httpx
        missing, nodes = {}, {}
        try:
            while self.capture.pending() and not self.capture.failure:
                await asyncio.sleep(.1)
            async with httpx.AsyncClient(trust_env=False,timeout=60) as client:
                for url in self.remote:
                    try:
                        while True:
                            sent = time.time_ns()
                            response = await client.post(url+'/internal/request_profile/seal',json=dict(capture=self.capture.id))
                            response.raise_for_status()
                            received = time.time_ns()
                            value = response.json()
                            value["clock_measurement"] = clock_sample(sent, received, value)
                            if value['complete']:
                                nodes[url] = value
                                break
                            await asyncio.sleep(.1)
                        path = self.capture.directory/('remote.'+uuid.uuid5(uuid.NAMESPACE_URL,url).hex+'.jsonl')
                        async with client.stream('GET',url+'/internal/request_profile/events',
                                params=dict(capture=self.capture.id)) as events:
                            events.raise_for_status()
                            with path.open('wb') as output:
                                async for chunk in events.aiter_bytes():
                                    await asyncio.to_thread(output.write,chunk)
                    except Exception as error:
                        missing[url] = type(error).__name__
            local = await self.seal()
            while not local['complete']:
                await asyncio.sleep(.1)
                local = await self.seal()
            for url in set(self.peers())-set(self.remote):
                missing[url] = "not_prepared_at_start"
            self.result = await asyncio.to_thread(merge,self.capture.directory)
            manifest = dict(capture=self.capture.id, clock='unix_epoch_microseconds',
                clocks=self.remote, local=local, nodes=nodes, missing=missing,
                device_timeline='not_enabled_host_envelopes_only',
                duration_semantics='Host durations include waits; nested/shared spans are not additive',
                complete=local['complete'] and not missing and not self.result['broken_records'] and all(
                    not n.get('missing') and not n.get('error') for n in [local,*nodes.values()]) and all(
                    p['complete'] for n in [local,*nodes.values()] for p in n.get('processes',())),
                **self.result)
            (self.capture.directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
            self.result['manifest'] = str(self.capture.directory/'manifest.json')
            self.result['complete'] = manifest['complete']
        except Exception as error:
            self.result = dict(complete=False,error=type(error).__name__,capture=self.capture.id)

    def install(self, app):
        from fastapi.responses import JSONResponse, StreamingResponse
        app.add_middleware(RequestProfileMiddleware,controller=self)
        async def start():
            try:
                return JSONResponse(await self.start())
            except ValueError as error:
                return JSONResponse(dict(error=str(error)),status_code=409)
        async def stop():
            try:
                value = await self.stop()
                return JSONResponse(value,status_code=200 if value['result'] else 202)
            except ValueError as error:
                return JSONResponse(dict(error=str(error)),status_code=409)
        async def prepare(payload: dict):
            received = time.time_ns()
            try:
                await self.prepare(payload['capture'])
                self.prepared_by_peer = True
                return dict(wall_ns=str(time.time_ns()),received_ns=str(received),node=self.node)
            except ValueError as error:
                return JSONResponse(dict(error=str(error)),status_code=409)
        async def seal(payload: dict):
            received = time.time_ns()
            if self.capture is None or self.capture.id!=payload['capture']:
                return JSONResponse(dict(error='unknown capture'),status_code=404)
            value = await self.seal()
            value.update(received_ns=str(received), wall_ns=str(time.time_ns()))
            return value
        async def events(capture: str):
            if self.capture is None or self.capture.id!=capture or not self.capture.closed:
                return JSONResponse(dict(error='capture is not sealed'),status_code=409)
            def contents():
                for path in sorted(self.capture.directory.glob('trace.*.jsonl')):
                    with path.open('rb') as stream:
                        while chunk := stream.read(1<<16):
                            yield chunk
            return StreamingResponse(contents(),media_type='application/x-ndjson')
        app.add_api_route('/start_request_profile',start,methods=['POST'])
        app.add_api_route('/stop_request_profile',stop,methods=['POST'])
        app.add_api_route('/internal/request_profile/prepare',prepare,methods=['POST'])
        app.add_api_route('/internal/request_profile/seal',seal,methods=['POST'])
        app.add_api_route('/internal/request_profile/events',events,methods=['GET'])


def engine_controller(engine, role="replica"):
    """Reuse the ordered worker control and existing DDR endpoints, with no resource ownership."""
    async def workers(active, capture):
        from pypto_serving.serving.server.ipc import ProfileCommand, decode_profile_result
        processes = []
        for core in engine._cores:
            result = await core._call_worker_command(
                ProfileCommand(active=active, request_capture=capture), decode_profile_result)
            if result.error:
                raise RuntimeError(result.error)
            if result.request_pending:
                raise BlockingIOError("worker request profile is draining")
            processes.extend(result.request_processes)
            tiering = getattr(core, "_cache_tiering", None)
            if tiering is not None:
                owners = await asyncio.gather(*(asyncio.to_thread(owner.call, "request_profile", (active, capture))
                                               for owner in tiering.owners))
                if active:
                    processes.extend(owners)
                elif not all(owners):
                    raise BlockingIOError("DDR request profile is draining")
        return processes
    return Controller(role=role, workers=workers)


def clock_sample(sent, received, reply):
    """Four timestamps exclude remote preparation/sealing from clock uncertainty."""
    entered, departed = int(reply["received_ns"]), int(reply["wall_ns"])
    rtt = received-sent
    transport = rtt-(departed-entered)
    valid = transport >= 0 and departed >= entered and received >= sent
    return dict(clock=str(departed), remote_received_ns=str(entered), sent_ns=str(sent),
        received_ns=str(received), rtt_ns=rtt, rtt_excluding_handler_ns=transport, valid=valid,
        offset_estimate_ns=((entered-sent)+(departed-received))//2 if valid else None,
        uncertainty_ns=(transport+1)//2 if valid else None)
