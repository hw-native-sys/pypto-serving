# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The same HTTP guarantees apply to replicas and P/D coordination."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from pypto_serving.router.app import ServingRouter
from pypto_serving.router.config import ReplicaSpec, RouterConfig
from pypto_serving.router.http import request_headers, response_headers
from tests.unit.router.test_pd_coordination import Peers, config, request


def router(pd, *, body=None, error=None):
    cfg = config() if pd else RouterConfig(replicas=(ReplicaSpec("r", "r", 8000),),
        connect_timeout_seconds=0.04, health_interval_seconds=60)
    peers = Peers(cfg) if pd else None
    if peers is not None:
        peers.d_stream = body
    async def transport(req):
        if req.method == "GET" or req.url.path.endswith("abort"):
            return await peers.handle(req) if pd else httpx.Response(200, json={})
        if error is not None:
            raise error("transport failed", request=req)
        return await peers.handle(req) if pd else httpx.Response(200, stream=body)
    return ServingRouter(cfg, client_factory=lambda _: httpx.AsyncClient(
        transport=httpx.MockTransport(transport), timeout=1)), peers


@pytest.mark.parametrize("pd", [False, True])
@pytest.mark.parametrize("error,status,code", [(httpx.PoolTimeout, 503, "http_pool_exhausted"),
                                              (httpx.ConnectError, 502, "upstream_connection_error")])
def test_transport_errors_have_common_status_origin_and_health(pd, error, status, code):
    async def run():
        instance, _ = router(pd, error=error)
        async with instance.app.router.lifespan_context(instance.app):
            await instance.health.stop()
            response = await instance.proxy.forward(request({}), "/v1/completions")
            assert response.status_code == status and json.loads(response.body)["code"] == code
            assert response.headers["x-pypto-error-source"] == "router"
            assert response.headers["x-session-id"] == "conversation"
            assert all(s.ready == (error is httpx.PoolTimeout) for s in instance.registry.states)
            assert not instance.proxy.active and all(s.outstanding == 0 for s in instance.registry.states)
    asyncio.run(run())


@pytest.mark.parametrize("pd", [False, True])
@pytest.mark.parametrize("ending", ["unstarted", "cancel", "close"])
def test_response_retirement_closes_once_and_releases_both_modes(pd, ending):
    class Body(httpx.AsyncByteStream):
        closes = 0
        async def __aiter__(self):
            self.started.set()
            await self.release.wait()
            yield b'data: {"choices":[{"text":"answer"}]}\n\ndata: [DONE]\n\n'
        async def aclose(self):
            self.closes += 1
    async def run():
        body = Body()
        body.started, body.release = asyncio.Event(), asyncio.Event()
        instance, peers = router(pd, body=body)
        async with instance.app.router.lifespan_context(instance.app):
            response = await instance.proxy.forward(request({"stream": True}), "/v1/completions")
            if ending == "cancel":
                reader = asyncio.create_task(anext(response.body_iterator))
                await asyncio.wait_for(body.started.wait(), 1)
                await asyncio.sleep(0.08)
                assert not reader.done() and body.closes == 0
                if pd:
                    assert instance.registry.state("p0").outstanding == 0
                    assert instance.registry.state("d0").outstanding == 1
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            elif ending == "close":
                await instance.proxy.close()
            else:
                await response.body_iterator.aclose()
                async def retired():
                    while instance.proxy.active:
                        await asyncio.sleep(0.005)
                await asyncio.wait_for(retired(), 1)
                assert not body.started.is_set()
            await response.body_iterator.aclose()
            assert body.closes == 1 and not instance.proxy.active
            assert all(s.outstanding == 0 for s in instance.registry.states)
            if pd:
                assert sum(path.endswith("abort") for _, path in peers.calls) == 2
    asyncio.run(run())


def test_shared_headers_preserve_raw_encoding_and_strip_decoded_framing():
    headers = httpx.Headers({"Host": "upstream", "Connection": "private", "private": "drop",
        "Content-Length": "20", "Content-Encoding": "gzip", "Authorization": "Bearer test",
        "x-pypto-error-source": "forged"})
    forwarded = request_headers(headers)
    assert forwarded == {"content-encoding": "gzip", "authorization": "Bearer test"}
    upstream = SimpleNamespace(headers=headers)
    raw = response_headers(upstream, "session", source="upstream")
    decoded = response_headers(upstream, "session", decoded=True)
    assert raw["content-encoding"] == "gzip" and raw["x-pypto-error-source"] == "upstream"
    assert decoded == {"host": "upstream", "authorization": "Bearer test", "x-session-id": "session"}
