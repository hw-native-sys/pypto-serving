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

from pypto_serving.pd.protocol import Binding, decode, digest
from pypto_serving.router.app import ServingRouter
from tests.unit.router.test_pd_coordination import Peers, config, request


@asynccontextmanager
async def application(transport):
    router = ServingRouter(config(), client_factory=lambda _: httpx.AsyncClient(
        transport=transport, timeout=1))
    async with router.app.router.lifespan_context(router.app):
        yield router


def descriptor(name):
    return Peers(config()).nodes[name]


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
            if req.url.host == "p0":
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
                assert sorted(aborts) == ["d0", "p0"]
                assert len(responses) == 1 and responses[0].is_closed and body.closes == 1
                assert not any(state.outstanding for state in router.registry.states)
                assert not router.proxy.active
            finally:
                for response in responses:
                    await response.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("fields,streaming", [({}, False), ({"stream": True}, True),
    ({"stream": "false"}, False), ({"stream": None}, None)])
def test_stream_coercion_matches_model_node_schema(kind, fields, streaming):
    from pypto_serving.serving.server.server import ChatCompletionRequest, CompletionRequest
    async def run():
        posts = []
        schema = CompletionRequest if kind == "completion" else ChatCompletionRequest
        async def transport(req):
            if req.url.path.endswith("descriptor"):
                return httpx.Response(200, json=msgspec.to_builtins(descriptor(req.url.host)))
            if req.url.path.endswith("abort"):
                return httpx.Response(200, json={"accepted": True})
            body = json.loads(req.content)
            public = schema.model_validate(body)
            binding = decode(json.loads(req.headers["x-pd-context"])["binding"], Binding)
            assert binding.input_digest == digest(body)
            assert type(body["stream"]) is bool and public.stream == streaming
            posts.append(req.url.host)
            if req.url.host == "p0":
                return httpx.Response(200, json={"request": binding.request, "status": "ready"})
            if public.stream:
                return httpx.Response(200, text='data: {"choices":[{"text":"answer","finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                                      headers={"content-type": "text/event-stream"})
            # A coerced-false request must use the longer JSON header deadline.
            await asyncio.sleep(config().pd_options["handoff_timeout"] * 2)
            return httpx.Response(200, json={"choices": [{"text": "answer"}]})
        payload = {"prompt": "hello"} if kind == "completion" else {
            "messages": [{"role": "user", "content": "hello"}]}
        path = "/v1/completions" if kind == "completion" else "/v1/chat/completions"
        async with application(httpx.MockTransport(transport)) as router:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=router.app), base_url="http://router") as client:
                response = await asyncio.wait_for(client.post(path, json={**payload, **fields}), 2)
            if streaming is None:
                assert response.status_code == 400 and not posts
            else:
                assert response.status_code == 200 and "answer" in response.text
                assert "pd_execution_error" not in response.text and sorted(posts) == ["d0", "p0"]
                media = "text/event-stream" if streaming else "application/json"
                assert response.headers["content-type"].startswith(media)
    asyncio.run(run())


def test_pd_probes_share_health_monitor_but_not_generation_pool(monkeypatch):
    from unittest.mock import AsyncMock

    async def run():
        clients = []
        mode = "ready"
        async def transport(req):
            assert req.url.path.endswith("descriptor")
            if mode == "busy":
                raise httpx.PoolTimeout("local pool busy")
            node = msgspec.structs.replace(descriptor(req.url.host), healthy=mode == "ready")
            return httpx.Response(200, json=msgspec.to_builtins(node))
        def factory(_):
            probe = len(clients) == 1
            async def handler(req):
                assert probe, "descriptor used the generation pool"
                return await transport(req)
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=1)
            clients.append(client)
            return client
        router = ServingRouter(config(), client_factory=factory)
        for operation in ("adopt_previous", "launch_initial", "stop_all"):
            monkeypatch.setattr(router.fleet, operation, AsyncMock(side_effect=AssertionError(
                "PD nodes are externally managed")))
        assert router.proxy.probe_client is router.probe_client is clients[1]
        assert router.health._client is clients[1] and router.client is clients[0]
        async with router.app.router.lifespan_context(router.app):
            before = dict(router.proxy.nodes)
            assert len(before) == 2 and (await router._health()).status_code == 200
            mode = "busy"
            await router.health.probe_once()
            assert router.proxy.nodes == before
            assert all(state.ready and state.failures == 0 for state in router.registry.states)
            mode = "unready"
            await router.health.probe_once()
            assert not router.proxy.nodes and (await router._health()).status_code == 503
            assert router.fleet.owned_names == () and not router.fleet._watchers
        assert all(client.is_closed for client in clients)
    asyncio.run(run())


@pytest.mark.parametrize("flags", [["--no-replace-dead"], ["--replace-after", "9"]])
def test_pd_rejects_replica_replacement_cli(tmp_path, monkeypatch, flags):
    from pypto_serving.router import app

    path = tmp_path / "router.json"
    path.write_text(json.dumps(dict(replicas=[
        dict(name="p", host="p", port=8111, role="prefill"),
        dict(name="d", host="d", port=8112, role="decode")])))
    monkeypatch.delenv(app.ADMIN_TOKEN_ENV, raising=False)
    monkeypatch.setattr(app, "create_router_app", lambda *_, **__: pytest.fail("PD launch must reject flags"))
    assert app.main(["--replicas", str(path), *flags]) == 1
