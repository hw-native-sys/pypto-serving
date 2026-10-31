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
from fastapi.responses import Response, StreamingResponse
import httpx
import msgspec
from pydantic import TypeAdapter

from pypto_serving.tools.observe import record
from .protocol import Binding, MAX_FRAME, Node, VERSION, decode
from pypto_serving.router.http import (
    SESSION_HEADER, Retirement, RouterError, close_requests, error_response, failure,
    request_headers, response_headers, resolve_session_id, send,
)
from pypto_serving.router.routing import RoundRobin

logger = logging.getLogger(__name__)
_STREAM = TypeAdapter(bool)


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
            raise  # The shared HealthMonitor classifies local pool pressure.
        except (httpx.HTTPError, ValueError):
            self.nodes.pop(spec.name, None)
            self.registry.set_ready(spec.name, False)
            return False

    def _candidates(self, model):
        result = []
        for p_id, d_id in self.pair_ids:
            p, d = self.nodes.get(p_id), self.nodes.get(d_id)
            valid = (p is not None and d is not None and p.compatible(d)
                     and p.role == "prefill" and d.role == "decode"
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
                raise RouterError(404, "model_not_found", f"Unknown model: {model}")
            raise RouterError(503, "no_available_backend", "No compatible healthy P/D pair is available")
        available = [pair if pair and all(self.registry.state(n.node_id).outstanding < self.config.capacity
                                          for n in pair) else None for pair in candidates]
        if not any(available):
            raise RouterError(503, "capacity_exhausted", "All compatible P/D pairs are at capacity. Retry later.")
        pair, reason = self.policy.select(available)
        if not any(pair is candidate for candidate in available):
            raise RuntimeError("policy selected an ineligible P/D pair")
        p, d = pair
        binding = Binding.create(p, d, payload)
        endpoints = tuple(self.registry.state(n.node_id).spec for n in (p, d))
        release = self.registry.acquire(*(endpoint.name for endpoint in endpoints))
        self.registry.record_route(*(e.name for e in endpoints))
        record(logger, "route", request=binding.request, prefill=p.node_id, decode=d.node_id,
               policy=self.config.policy, reason=reason, prefill_identity=p.key, decode_identity=d.key)
        return p, d, binding, endpoints, release

    async def close(self):
        self.closing = True
        await close_requests(self.active, self.config.handoff_timeout)

    async def forward(self, request: Request, path: str):
        owner = asyncio.current_task()
        self.active[owner] = None
        session_id = resolve_session_id(request.headers, b"")
        try:
            if self.closing:
                raise RouterError(503, "router_closing", "Router is closing")
            body = await request.body()
            session_id = resolve_session_id(request.headers, body)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            payload["stream"] = _STREAM.validate_python(payload.get("stream", False))
            return await self._forward(request, path, owner, payload, session_id)
        except (RouterError, ValueError, RuntimeError, asyncio.TimeoutError, OSError, httpx.HTTPError) as exc:
            error = failure(exc, timeout_code="pd_operation_timeout", protocol_code="pd_protocol_error")
            record(logger, "route.reject", session=session_id, reason=error.code, status=error.status,
                   node_snapshots=tuple((s.name, self.nodes[s.name].key if s.name in self.nodes else None,
                                         s.ready, s.draining, s.outstanding) for s in self.registry.states))
            if error.code in {"invalid_request", "model_not_found", "no_available_backend",
                              "capacity_exhausted", "router_closing"}:
                self.registry.rejected += 1
            return error_response(error, session_id)
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
        p, d, binding, endpoints, release = self._admit(payload)
        p_task = d_task = response = None
        finished = False
        async def before_close():
            if not finished:
                # Abort wakes D's rendezvous before its HTTP task is joined.
                await self.abort(binding, endpoints)
            tasks = tuple(t for t in (p_task, d_task) if t is not None)
            if tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                           self.config.handoff_timeout)
                except asyncio.TimeoutError:
                    logger.warning("PD HTTP task retirement timed out request=%s; native leases stay guarded",
                                   binding.request)
            if retirement.response is None and d_task is not None and d_task.done() and not d_task.cancelled():
                if d_task.exception() is None:
                    retirement.response = d_task.result()

        def release_all():
            release()
            record(logger, "http.retire", request=binding.request, finished=finished,
                   status=response.status_code if response is not None else "unknown")

        retirement = Retirement(self.active, owner, release_all, before_close)

        async def ready():
            result = await p_task
            if result.status_code != 200:
                raise RouterError(result.status_code, "upstream_error", result.text, result)
            try:
                valid = result.json() == {"request": binding.request, "status": "ready"}
            except ValueError:
                valid = False
            if not valid:
                raise RouterError(503, "pd_protocol_error", "Prefill returned an invalid handoff confirmation")

        try:
            def headers(peer):
                return request_headers(request.headers, drop=("content-encoding", "content-type", "x-pd-context"),
                    extra={SESSION_HEADER: session_id,
                           "x-pd-context": msgspec.json.encode(dict(binding=binding, peer=peer)).decode()})
            p_task = asyncio.create_task(send(self.client, self.registry, endpoints[0], path,
                json=payload, headers=headers(d), timeout=self._timeout(self.config.request_timeout), stream=False))
            p_task.add_done_callback(lambda _: release(endpoints[0].name))
            d_task = asyncio.create_task(send(self.client, self.registry, endpoints[1], path,
                json=payload, headers=headers(p), timeout=self._timeout(self.config.request_timeout)))
            # JSON headers follow generation; SSE headers only acknowledge setup.
            header_timeout = (self.config.handoff_timeout if payload.get("stream", False)
                              else self.config.request_timeout)
            deadline = asyncio.get_running_loop().time() + header_timeout
            done, _ = await asyncio.wait((p_task, d_task), return_when=asyncio.FIRST_COMPLETED,
                                         timeout=header_timeout)
            if not done:
                raise RouterError(503, "pd_operation_timeout", "PD response header wait timed out")
            if p_task in done:
                await ready()
            try:
                response = await asyncio.wait_for(asyncio.shield(d_task),
                    max(0, deadline - asyncio.get_running_loop().time()))
            except asyncio.TimeoutError as exc:
                raise RouterError(503, "pd_operation_timeout", "Decode response header wait timed out") from exc
            retirement.response = response
            if response.status_code != 200:
                await response.aread()
                raise RouterError(response.status_code, "upstream_error", response.text, response)
            if not payload.get("stream", False):
                body = await response.aread()
                await ready()
                finished = True
                result = Response(body, headers=response_headers(response, session_id, decoded=True))
                await retirement.wait()
                return result

            async def stream():
                nonlocal finished
                frame = []
                size = 0
                upstream_failed = False
                try:
                    retirement.begin()
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
                    error = (RouterError(503, "pd_protocol_error", str(exc))
                             if isinstance(exc, ValueError) else failure(exc,
                                 timeout_code="pd_operation_timeout", protocol_code="pd_protocol_error"))
                    yield "data: " + json.dumps({"error": {"message": str(error), "type": "pd_execution_error",
                        "code": error.code, "source": error.source}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    await retirement.wait()
            # An async generator's finally cannot retire a body that never starts.
            retirement.arm(self.config.handoff_timeout)
            return StreamingResponse(stream(), media_type="text/event-stream",
                                     headers=response_headers(response, session_id, decoded=True))
        except BaseException:
            await retirement.wait()
            raise
