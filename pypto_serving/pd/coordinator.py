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

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
import msgspec
from pydantic import TypeAdapter

from .protocol import Binding, MAX_FRAME, Node, VERSION, decode
from pypto_serving.router.proxy import (
    SESSION_HEADER, _DROP_REQUEST, _DROP_RESPONSE, _filter_headers, resolve_session_id,
)
from pypto_serving.router.routing import RoundRobin

logger = logging.getLogger(__name__)
ERROR_SOURCE_HEADER = "x-pypto-error-source"
_STREAM = TypeAdapter(bool)


class _Failure(Exception):
    def __init__(self, status, code, message, upstream=None):
        super().__init__(message)
        self.status, self.code, self.upstream = status, code, upstream

    @property
    def source(self):
        return "upstream" if self.upstream is not None else "router"


def _failure(exc):
    if isinstance(exc, _Failure):
        return exc
    if isinstance(exc, httpx.PoolTimeout):
        return _Failure(503, "http_pool_exhausted", "Router HTTP connection pool is busy")
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return _Failure(503, "pd_operation_timeout", str(exc) or "PD HTTP operation timed out")
    if isinstance(exc, httpx.HTTPError):
        return _Failure(502, "upstream_connection_error", str(exc))
    if isinstance(exc, ValueError):
        return _Failure(400, "invalid_request", str(exc))
    return _Failure(503, "pd_protocol_error", str(exc))


def _response_headers(response, session_id):
    headers = _filter_headers(response.headers, _DROP_RESPONSE | {"content-encoding", ERROR_SOURCE_HEADER})
    return {**headers, SESSION_HEADER: session_id}


class PDCoordinator:
    """Coordinate a request; the common router owns HTTP and node liveness."""

    def __init__(self, config, registry, client, probe_client):
        self.config, self.registry, self.client = config, registry, client
        self.probe_client = probe_client
        self.nodes = {}
        self.models, self.discovered = set(), set()
        self.policy = RoundRobin()
        self.pair_ids = tuple((p.node_id, config.decode[(i + offset) % len(config.decode)].node_id)
                              for offset in range(len(config.decode)) for i, p in enumerate(config.prefill))
        self.requests = self.handoffs = 0
        self.active = {}
        self.closing = False

    def _timeout(self, read, *, control=False):
        values = self.client.timeout.as_dict()
        if control:
            values = {k: min(v, read) if v is not None else read for k, v in values.items()}
        return httpx.Timeout(**{**values, "read": read})

    async def probe(self, spec):
        try:
            response = await self.probe_client.get(spec.base_url + "/internal/pd/descriptor",
                                             timeout=self._timeout(5, control=True))
            response.raise_for_status()
            node = decode(response.json(), Node)
            if (node.node_id != spec.name or node.role != spec.role
                    or node.provider != self.config.provider or node.version != VERSION
                    or not node.contract.model):
                raise ValueError("descriptor differs from configured PD node")
            self.models.add(node.contract.model)
            self.discovered.add(spec.name)
            if not node.healthy:
                self.nodes.pop(spec.name, None)
                self.registry.set_ready(spec.name, False)
                return False
            self.nodes[spec.name] = node
            return True
        except httpx.PoolTimeout:
            # Local pool pressure is not a liveness observation about this node.
            logger.warning("descriptor probe deferred: HTTP pool busy node=%s", spec.name)
            return None
        except (httpx.HTTPError, ValueError):
            self.nodes.pop(spec.name, None)
            self.registry.set_ready(spec.name, False)
            return False

    def _candidates(self, model):
        result = []
        for p_id, d_id in self.pair_ids:
            p, d = self.nodes.get(p_id), self.nodes.get(d_id)
            valid = (p is not None and d is not None and p.compatible(d)
                     and all(self.registry.state(n).routable for n in (p_id, d_id))
                     and (model is None or p.contract.model == model))
            result.append((p, d) if valid else None)
        return result

    def pairs(self, model=None):
        return [pair for pair in self._candidates(model) if pair is not None]

    def _admit(self, payload):
        """Select current immutable node snapshots and reserve counts without yielding."""
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError("model must be a string")
        candidates = self._candidates(model or None)
        if not any(candidates):
            if model and model not in self.models and len(self.discovered) == len(self.registry.states):
                raise _Failure(404, "model_not_found", f"Unknown model: {model}")
            raise _Failure(503, "no_available_backend", "No compatible healthy P/D pair is available")
        available = [pair if pair and all(self.registry.state(n.node_id).outstanding < self.config.capacity
                                          for n in pair) else None for pair in candidates]
        if (self.requests >= self.config.capacity * len(self.config.decode)
                or self.handoffs >= self.config.capacity * len(self.config.prefill) or not any(available)):
            raise _Failure(503, "capacity_exhausted", "All compatible P/D pairs are at capacity. Retry later.")
        p, d = self.policy.select(available)
        binding = Binding.create(p, d, payload)
        endpoints = tuple(self.registry.state(n.node_id).spec for n in (p, d))
        self.requests += 1
        self.handoffs += 1
        for endpoint in endpoints:
            self.registry.acquire(endpoint.name)
        self.registry.record_route(*(e.name for e in endpoints))
        return p, d, binding, endpoints

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
        owner = asyncio.current_task()
        self.active[owner] = None
        session_id = resolve_session_id(request.headers, b"")
        try:
            if self.closing:
                raise _Failure(503, "router_closing", "Router is closing")
            body = await request.body()
            session_id = resolve_session_id(request.headers, body)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            payload["stream"] = _STREAM.validate_python(payload.get("stream", False))
            return await self._forward(request, path, owner, payload, session_id)
        except (_Failure, ValueError, RuntimeError, asyncio.TimeoutError, TimeoutError, httpx.HTTPError) as exc:
            error = _failure(exc)
            if error.code in {"invalid_request", "model_not_found", "no_available_backend",
                              "capacity_exhausted", "router_closing"}:
                self.registry.rejected += 1
            headers = {SESSION_HEADER: session_id, ERROR_SOURCE_HEADER: error.source}
            if error.upstream is not None:
                return Response(error.upstream.content, status_code=error.status,
                                headers={**_response_headers(error.upstream, session_id), **headers})
            if error.code == "capacity_exhausted":
                headers["Retry-After"] = "1"
            return JSONResponse({"object": "error", "code": error.code, "message": str(error)},
                                status_code=error.status, headers=headers)
        finally:
            if self.active.get(owner) is None:
                self.active.pop(owner, None)

    async def abort(self, binding, endpoints):
        results = await asyncio.gather(*(self.client.post(e.base_url + "/internal/pd/abort",
            json=msgspec.to_builtins(binding), timeout=self._timeout(5, control=True))
            for e in endpoints), return_exceptions=True)
        for endpoint, result in zip(endpoints, results):
            if isinstance(result, BaseException) or result.status_code != 200:
                logger.warning("HTTP abort not confirmed node=%s result=%s", endpoint.name, result)

    async def _forward(self, request, path, owner, payload, session_id):
        p, d, binding, endpoints = self._admit(payload)
        p_task = d_task = response = None
        finished = False
        cleanup_task = None
        start_timer = None
        p_released = False

        def release_handoff(_done=None):
            nonlocal p_released
            if not p_released:
                p_released = True
                self.registry.release(endpoints[0].name)
                self.handoffs -= 1

        async def cleanup():
            try:
                if not finished:
                    # Abort wakes D's pre-header rendezvous before HTTP tasks are joined.
                    await self.abort(binding, endpoints)
                tasks = tuple(t for t in (p_task, d_task) if t is not None)
                if tasks:
                    try:
                        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                               self.config.handoff_timeout)
                    except asyncio.TimeoutError:
                        logger.warning("PD HTTP task retirement timed out request=%s; native leases stay guarded",
                                       binding.request)
                received = response
                if received is None and d_task is not None and d_task.done() and not d_task.cancelled():
                    if d_task.exception() is None:
                        received = d_task.result()
                if received is not None:
                    await received.aclose()
            except Exception:
                logger.exception("PD HTTP cleanup failed request=%s", binding.request if binding else None)
            finally:
                release_handoff()
                self.registry.release(endpoints[1].name)
                self.requests -= 1
                self.active.pop(owner, None)

        def retire():
            nonlocal cleanup_task
            if start_timer is not None:
                start_timer.cancel()
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(cleanup())
            return cleanup_task

        self.active[owner] = retire

        async def ready():
            result = await p_task
            if result.status_code != 200:
                raise _Failure(result.status_code, "upstream_error", result.text, result)
            try:
                valid = result.json() == {"request": binding.request, "status": "ready"}
            except ValueError:
                valid = False
            if not valid:
                raise _Failure(503, "pd_protocol_error", "Prefill returned an invalid handoff confirmation")

        try:
            logger.info("route request=%s prefill=%s decode=%s", binding.request, p.node_id, d.node_id)
            forwarded = _filter_headers(request.headers, _DROP_REQUEST | {
                "content-encoding", "content-type", "x-pd-context", ERROR_SOURCE_HEADER})
            def headers(peer):
                return {**forwarded, SESSION_HEADER: session_id,
                        "x-pd-context": msgspec.json.encode(dict(binding=binding, peer=peer)).decode()}
            p_task = asyncio.create_task(self.client.post(endpoints[0].base_url + path,
                json=payload, headers=headers(d), timeout=self._timeout(self.config.request_timeout)))
            p_task.add_done_callback(release_handoff)
            d_task = asyncio.create_task(self.client.send(self.client.build_request(
                "POST", endpoints[1].base_url + path, json=payload, headers=headers(p),
                timeout=self._timeout(self.config.request_timeout)), stream=True))
            # JSON headers follow generation; SSE headers only acknowledge setup.
            header_timeout = (self.config.handoff_timeout if payload.get("stream", False)
                              else self.config.request_timeout)
            deadline = asyncio.get_running_loop().time() + header_timeout
            done, _ = await asyncio.wait((p_task, d_task), return_when=asyncio.FIRST_COMPLETED,
                                         timeout=header_timeout)
            if not done:
                raise _Failure(503, "pd_operation_timeout", "PD response header wait timed out")
            if p_task in done:
                await ready()
            try:
                response = await asyncio.wait_for(asyncio.shield(d_task),
                    max(0, deadline - asyncio.get_running_loop().time()))
            except asyncio.TimeoutError as exc:
                raise _Failure(503, "pd_operation_timeout", "Decode response header wait timed out") from exc
            if response.status_code != 200:
                await response.aread()
                raise _Failure(response.status_code, "upstream_error", response.text, response)
            if not payload.get("stream", False):
                body = await response.aread()
                await ready()
                finished = True
                result = Response(body, headers=_response_headers(response, session_id))
                await asyncio.shield(retire())
                return result

            async def stream():
                nonlocal finished
                start_timer.cancel()
                frame = []
                size = 0
                upstream_failed = False
                try:
                    if cleanup_task is not None:
                        raise _Failure(503, "pd_operation_timeout", "PD response retired before streaming began")
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
                            if not terminal and not isinstance(value, dict):
                                raise ValueError("Decode SSE data must be a JSON object")
                            if value and value.get("error") is not None:
                                upstream_failed = True
                            if not upstream_failed and (terminal or (
                                    value and any(c.get("finish_reason") for c in value.get("choices", [])))):
                                await ready()
                            if terminal:
                                finished = not upstream_failed
                            yield text + "\n\n"
                            if terminal:
                                return
                    if upstream_failed:
                        yield "data: [DONE]\n\n"
                        return
                    if not finished:
                        raise RuntimeError("Decode stream ended without terminal event")
                except Exception as exc:
                    error = (_Failure(503, "pd_protocol_error", str(exc))
                             if isinstance(exc, ValueError) else _failure(exc))
                    yield "data: " + json.dumps({"error": {"message": str(error), "type": "pd_execution_error",
                        "code": error.code, "source": error.source}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    task = retire()
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        await asyncio.shield(task)
                        raise
            # An async generator's finally cannot retire a body that never starts.
            start_timer = asyncio.get_running_loop().call_later(self.config.handoff_timeout, retire)
            return StreamingResponse(stream(), media_type="text/event-stream",
                                     headers=_response_headers(response, session_id))
        except BaseException:
            task = retire()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.shield(task)
                raise
            raise
