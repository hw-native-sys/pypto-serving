# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host HTTP ordering and request normalization for the PD Router."""

import asyncio
import json

from contextlib import asynccontextmanager
import httpx
import msgspec
import pytest
from starlette.requests import Request

from pypto_serving.pd.protocol import Component, Contract, Node, Topology
from pypto_serving.router.app import ServingRouter
from pypto_serving.router.config import ReplicaSpec, RouterConfig


def config():
    return RouterConfig(replicas=tuple(ReplicaSpec(name, name, 8111 + i, role=role)
        for i, (name, role) in enumerate((("p", "prefill"), ("d", "decode")))),
        pd_options={"capacity": 1, "handoff_timeout": 0.05, "request_timeout": 1},
        observability={"enabled": False}, health_interval_seconds=60)


@asynccontextmanager
async def application(transport):
    router = ServingRouter(config(), client_factory=lambda _: httpx.AsyncClient(
        transport=transport, timeout=1))
    async with router.app.router.lifespan_context(router.app):
        yield router


def descriptor(name):
    return Node(name, "prefill" if name == "p" else "decode", "boot-" + name,
                Contract("test", "model", "layout", (Component("kv", "kv", "kv"),)),
                "registry-" + name, Topology.for_tp(1, 1))


def request(payload, path="/v1/completions"):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
    return Request({"type": "http", "method": "POST", "scheme": "http", "server": ("router", 8110),
                    "path": path, "query_string": b"", "headers": []}, receive)


class Body(httpx.AsyncByteStream):
    def __init__(self):
        self.closes = 0

    async def __aiter__(self):
        yield b'{"choices":[]}'

    async def aclose(self):
        self.closes += 1


@pytest.mark.parametrize("failure", ["prefill", "cancel", "timeout"])
def test_cleanup_closes_decode_response_returned_after_abort(failure):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        body, responses, aborts = Body(), [], []
        async def transport(req):
            if req.url.path.endswith("descriptor"):
                return httpx.Response(200, json=msgspec.to_builtins(descriptor(req.url.host)))
            if req.url.path.endswith("abort"):
                aborts.append(req.url.host)
                release.set()
                return httpx.Response(200, json={"accepted": True})
            if req.url.host == "p":
                await started.wait()
                if failure != "prefill":
                    await release.wait()
                return httpx.Response(500, json={"error": "prefill failed"})
            started.set()
            await release.wait()
            response = httpx.Response(200, stream=body)
            responses.append(response)
            return response
        async with application(httpx.MockTransport(transport)) as router:
            try:
                task = asyncio.create_task(router.proxy.forward(
                    request({"stream": failure == "timeout"}), "/v1/completions"))
                if failure == "cancel":
                    await asyncio.wait_for(started.wait(), 1)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 1)
                else:
                    response = await asyncio.wait_for(task, 1)
                    assert response.status_code == (500 if failure == "prefill" else 503)
                assert sorted(aborts) == ["d", "p"]
                assert len(responses) == 1 and responses[0].is_closed and body.closes == 1
                assert not any(state.outstanding for state in router.registry.states)
                assert not router.proxy.active
            finally:
                for response in responses:
                    await response.aclose()
    asyncio.run(run())
