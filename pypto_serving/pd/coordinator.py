# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""P/D request coordination inside the common Serving Router."""

import asyncio
import json
import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import msgspec

from .protocol import Binding, MAX_FRAME, Node, decode
from pypto_serving.router.proxy import SESSION_HEADER, resolve_session_id
from pypto_serving.router.routing import RoundRobin

logger = logging.getLogger(__name__)


class PDCoordinator:
    """Coordinate a request; the common router owns HTTP and node liveness."""

    def __init__(self, config, registry, client, probe_client):
        self.config, self.registry, self.client = config, registry, client
        self.probe_client = probe_client
        self.nodes = {}
        self.policy = RoundRobin()
        self.requests = asyncio.Semaphore(config.capacity * len(config.decode))
        self.handoffs = asyncio.Semaphore(config.capacity * len(config.prefill))
        self.active = {}
        self.closing = False

    async def probe(self, spec):
        try:
            response = await self.probe_client.get(spec.base_url + "/internal/pd/descriptor", timeout=5)
            response.raise_for_status()
            node = decode(response.json(), Node)
            if (node.node_id != spec.name or node.role != spec.role
                    or node.provider != self.config.provider or not node.healthy):
                raise ValueError("descriptor differs from configured healthy PD node")
            self.nodes[spec.name] = node
            return True
        except (httpx.HTTPError, ValueError):
            self.nodes.pop(spec.name, None)
            return False

    def pairs(self, model=None):
        p = [self.nodes[s.name] for s in self.registry.candidates("prefill") if s.name in self.nodes]
        d = [self.nodes[s.name] for s in self.registry.candidates("decode") if s.name in self.nodes]
        return [(p[i], d[(i + offset) % len(d)])
                for offset in range(len(d)) for i in range(len(p))
                if p[i].compatible(d[(i + offset) % len(d)])
                and (model is None or p[i].contract.model == model)]

    async def close(self):
        self.closing = True
        active = tuple(self.active.items())
        pending = tuple(t for t, _ in active if t is not asyncio.current_task() and not t.done())
        for task in pending:
            task.cancel()
        retiring = tuple(retire() for _, retire in active if retire is not None)
        if pending or retiring:
            await asyncio.wait_for(asyncio.gather(*pending, *retiring, return_exceptions=True),
                                   self.config.handoff_timeout + 5)

    async def forward(self, request: Request, path: str):
        if self.closing:
            raise HTTPException(503, "router is closing")
        owner = asyncio.current_task()
        self.active[owner] = None
        try:
            return await self._forward(request, path, owner)
        except BaseException as exc:
            self.active.pop(owner, None)
            if isinstance(exc, ValueError):
                raise HTTPException(400, str(exc)) from exc
            if isinstance(exc, httpx.HTTPError):
                raise HTTPException(502, str(exc)) from exc
            raise

    async def abort(self, binding, endpoints):
        await asyncio.gather(*(self.client.post(e.base_url + "/internal/pd/abort",
            json=msgspec.to_builtins(binding), timeout=5) for e in endpoints), return_exceptions=True)

    async def _forward(self, request, path, owner):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(400, "request must be a JSON object")
        session_id = resolve_session_id(request.headers, await request.body())
        await self.requests.acquire()
        handoff_held = False
        p_task = d_task = response = binding = None
        endpoints = ()
        finished = False
        cleanup_task = None
        p_released = False
        async def cleanup():
            nonlocal handoff_held, p_released
            try:
                if binding is not None and not finished:
                    # Abort wakes D's pre-header rendezvous before HTTP tasks are joined.
                    await self.abort(binding, endpoints)
                tasks = tuple(t for t in (p_task, d_task) if t is not None)
                if tasks:
                    try:
                        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                               self.config.handoff_timeout)
                    except asyncio.TimeoutError:
                        pass  # Node-side leases remain gated until positive native proof.
                received = response
                if received is None and d_task is not None and d_task.done() and not d_task.cancelled():
                    if d_task.exception() is None:
                        received = d_task.result()
                if received is not None:
                    await received.aclose()
            finally:
                if handoff_held:
                    handoff_held = False
                    self.handoffs.release()
                if endpoints:
                    if not p_released:
                        p_released = True
                        self.registry.release(endpoints[0].name)
                    self.registry.release(endpoints[1].name)
                self.requests.release()
                self.active.pop(owner, None)

        def retire():
            nonlocal cleanup_task
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(cleanup())
            return cleanup_task

        self.active[owner] = retire

        async def ready():
            result = await p_task
            if result.status_code != 200:
                raise HTTPException(result.status_code, result.text[:1000])
            if result.json() != {"request": binding.request, "status": "ready"}:
                raise RuntimeError(f"Prefill rejected handoff: {result.text[:1000]}")

        try:
            await self.handoffs.acquire()
            handoff_held = True
            p, d = self.policy.select(self.pairs(payload.get("model") or None))
            binding = Binding.create(p, d, payload)
            logger.info("route request=%s prefill=%s decode=%s", binding.request, p.node_id, d.node_id)
            endpoints = tuple(self.registry.state(n.node_id).spec for n in (p, d))
            for endpoint in endpoints:
                self.registry.acquire(endpoint.name)
                self.registry.state(endpoint.name).routed += 1
            def headers(peer):
                return {"x-pd-context": msgspec.json.encode(dict(binding=binding, peer=peer)).decode()}
            p_task = asyncio.create_task(self.client.post(endpoints[0].base_url + path,
                json=payload, headers=headers(d), timeout=self.config.request_timeout))
            d_task = asyncio.create_task(self.client.send(self.client.build_request(
                "POST", endpoints[1].base_url + path, json=payload, headers=headers(p),
                timeout=self.config.request_timeout), stream=True))
            # JSON headers follow generation; SSE headers only acknowledge setup.
            header_timeout = (self.config.handoff_timeout if payload.get("stream", False)
                              else self.config.request_timeout)
            done, _ = await asyncio.wait((p_task, d_task), return_when=asyncio.FIRST_COMPLETED,
                                         timeout=header_timeout)
            if not done:
                raise RuntimeError("handoff header deadline exceeded")
            if p_task in done:
                await ready()
            response = await asyncio.wait_for(asyncio.shield(d_task), header_timeout)
            if response.status_code != 200:
                body = await response.aread()
                raise HTTPException(response.status_code, body.decode()[:1000])
            def release_handoff(done):
                nonlocal handoff_held, p_released
                if not p_released:
                    p_released = True
                    self.registry.release(endpoints[0].name)
                if handoff_held:
                    handoff_held = False
                    self.handoffs.release()
            p_task.add_done_callback(release_handoff)
            if not payload.get("stream", False):
                body = await response.aread()
                await ready()
                finished = True
                result = JSONResponse(json.loads(body), headers={SESSION_HEADER: session_id})
                await asyncio.shield(retire())
                return result

            async def stream():
                nonlocal finished
                frame = []
                size = 0
                try:
                    async for line in response.aiter_lines():
                        size += len(line)
                        if size > MAX_FRAME:
                            raise ValueError("SSE frame exceeds control limit")
                        if line:
                            frame.append(line)
                            continue
                        text = "\n".join(frame)
                        frame, size = [], 0
                        if text.startswith("data: "):
                            data = text[6:]
                            terminal = data == "[DONE]"
                            value = None if terminal else json.loads(data)
                            if terminal or (value and any(c.get("finish_reason") for c in value.get("choices", []))):
                                await ready()
                            if terminal:
                                finished = True
                            yield text + "\n\n"
                    if not finished:
                        raise RuntimeError("Decode stream ended without terminal event")
                except Exception as exc:
                    yield "data: " + json.dumps({"error": {"message": str(exc), "type": "pd_execution_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    task = retire()
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        await asyncio.shield(task)
                        raise
            return StreamingResponse(stream(), media_type="text/event-stream",
                                     headers={SESSION_HEADER: session_id})
        except BaseException as exc:
            task = retire()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.shield(task)
                raise
            if isinstance(exc, HTTPException):
                exc.headers = {**(exc.headers or {}), SESSION_HEADER: session_id}
            elif isinstance(exc, (RuntimeError, ValueError, TimeoutError, httpx.HTTPError)):
                status = 400 if isinstance(exc, ValueError) else 502 if isinstance(exc, httpx.HTTPError) else 503
                raise HTTPException(status, str(exc) or "PD operation deadline exceeded",
                                    headers={SESSION_HEADER: session_id}) from exc
            raise
