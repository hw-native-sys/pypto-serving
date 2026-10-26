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

from pypto_serving.pd.protocol import Binding, Component, Contract, Node, Topology, decode, digest
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


@pytest.mark.parametrize("kind", ["completion", "chat"])
@pytest.mark.parametrize("fields,streaming", [({}, False), ({"stream": False}, False),
    ({"stream": "false"}, False), ({"stream": "0"}, False), ({"stream": "off"}, False),
    ({"stream": True}, True), ({"stream": "true"}, True), ({"stream": 0}, False),
    ({"stream": 1}, True), ({"stream": None}, None), ({"stream": "invalid"}, None)])
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
            if req.url.host == "p":
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
                assert "pd_execution_error" not in response.text and sorted(posts) == ["d", "p"]
                media = "text/event-stream" if streaming else "application/json"
                assert response.headers["content-type"].startswith(media)
    asyncio.run(run())
