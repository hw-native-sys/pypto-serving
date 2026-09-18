# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Proxy behaviour: verbatim forwarding, session headers, and stream teardown.

The fakes here speak only the small client surface the proxy uses, so these
tests need neither httpx nor a live replica.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers

from pypto_serving.router.app import ServingRouter
from pypto_serving.router.config import ReplicaSpec, RouterConfig
from pypto_serving.router.proxy import SESSION_HEADER


# --- fakes ---

class _FakeRequest:
    def __init__(self, body: bytes = b"", headers: dict | None = None) -> None:
        self.headers = Headers(
            raw=[(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        )
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _FakeUpstream:
    def __init__(self, chunks, status_code: int = 200, headers: dict | None = None) -> None:
        self.status_code = status_code
        self.headers = Headers(
            raw=[(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        )
        self._chunks = list(chunks)
        self.closed = False

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True

    def json(self):
        return json.loads(b"".join(self._chunks))


class _FakeClient:
    """Minimal stand-in for httpx.AsyncClient."""

    def __init__(self, chunks=(b"ok",), status_code: int = 200,
                 headers: dict | None = None, send_error: Exception | None = None) -> None:
        self.chunks = list(chunks)
        self.status_code = status_code
        self.headers = headers or {}
        self.send_error = send_error
        self.sent: list = []
        self.upstreams: list[_FakeUpstream] = []
        self.gets: list[str] = []

    def build_request(self, method, url, content=None, headers=None):
        return SimpleNamespace(method=method, url=url, content=content, headers=headers)

    async def send(self, request, stream: bool = False):
        self.sent.append(request)
        if self.send_error is not None:
            raise self.send_error
        upstream = _FakeUpstream(self.chunks, self.status_code, self.headers)
        self.upstreams.append(upstream)
        return upstream

    async def get(self, url, timeout=None):
        self.gets.append(url)
        return _FakeUpstream(self.chunks, self.status_code, self.headers)

    async def aclose(self) -> None:
        pass


def _router(count: int = 2, client: _FakeClient | None = None, **kwargs) -> ServingRouter:
    replicas = tuple(
        ReplicaSpec(name=f"r{i}", host="h", port=8000 + i) for i in range(count)
    )
    config = RouterConfig(replicas=replicas, **kwargs)
    fake = client if client is not None else _FakeClient()
    return ServingRouter(config, client_factory=lambda _config: fake)


async def _drain(response) -> bytes:
    return b"".join([chunk async for chunk in response.body_iterator])


def _parse_sse(payload: bytes) -> list[dict]:
    """Decode an SSE stream into its JSON events, ignoring the [DONE] sentinel."""
    events = []
    for line in payload.decode().splitlines():
        if not line.startswith("data:"):
            continue
        body = line[len("data:"):].strip()
        if body == "[DONE]":
            continue
        events.append(json.loads(body))
    return events


# --- forwarding ---

def test_request_body_is_forwarded_verbatim():
    client = _FakeClient()
    router = _router(client=client)
    body = json.dumps({"prompt": "hi", "max_tokens": 3, "unknown_field": [1, 2]}).encode()

    async def check():
        response = await router.proxy.forward(_FakeRequest(body), "/v1/completions")
        await _drain(response)

    asyncio.run(check())
    assert client.sent[0].content == body
    assert client.sent[0].url.endswith("/v1/completions")


def test_sse_chunks_pass_through_unmodified():
    chunks = [
        b'data: {"id": "c1", "choices": [{"text": "he"}]}\n\n',
        b'data: {"id": "c1", "choices": [{"text": "llo"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    router = _router(client=_FakeClient(chunks=chunks))

    async def check():
        response = await router.proxy.forward(
            _FakeRequest(b'{"stream": true}'), "/v1/completions"
        )
        return await _drain(response)

    payload = asyncio.run(check())
    assert payload == b"".join(chunks)
    assert [event["choices"][0]["text"] for event in _parse_sse(payload)] == ["he", "llo"]


def test_upstream_status_and_headers_are_relayed():
    client = _FakeClient(
        status_code=400,
        headers={"content-type": "application/json", "content-length": "9", "connection": "keep-alive"},
    )
    router = _router(client=client)

    async def check():
        return await router.proxy.forward(_FakeRequest(b"{}"), "/v1/completions")

    response = asyncio.run(check())
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    # Re-framed by the router, so the upstream framing headers must not survive.
    assert "connection" not in response.headers


def test_hop_by_hop_and_host_headers_are_not_forwarded():
    client = _FakeClient()
    router = _router(client=client)
    headers = {
        "host": "router:8000",
        "content-length": "2",
        "connection": "keep-alive",
        "authorization": "Bearer token",
        "content-type": "application/json",
    }

    async def check():
        response = await router.proxy.forward(_FakeRequest(b"{}", headers), "/v1/completions")
        await _drain(response)

    asyncio.run(check())
    forwarded = {k.lower() for k in client.sent[0].headers}
    assert "host" not in forwarded
    assert "content-length" not in forwarded
    assert "connection" not in forwarded
    # Everything else the client sent must survive the hop.
    assert "authorization" in forwarded
    assert "content-type" in forwarded


# --- session identity ---

def test_session_id_is_minted_and_returned_when_absent():
    router = _router()

    async def check():
        return await router.proxy.forward(_FakeRequest(b"{}"), "/v1/completions")

    response = asyncio.run(check())
    session_id = response.headers[SESSION_HEADER]
    assert session_id and len(session_id) == 32


def test_supplied_header_is_echoed_back():
    router = _router()

    async def check():
        return await router.proxy.forward(
            _FakeRequest(b"{}", {SESSION_HEADER: "conv-1"}), "/v1/chat/completions"
        )

    response = asyncio.run(check())
    assert response.headers[SESSION_HEADER] == "conv-1"


def test_body_session_id_is_used_when_no_header():
    router = _router()
    body = json.dumps({"prompt": "hi", "session_id": "conv-body"}).encode()

    async def check():
        return await router.proxy.forward(_FakeRequest(body), "/v1/completions")

    response = asyncio.run(check())
    assert response.headers[SESSION_HEADER] == "conv-body"


def test_header_wins_over_body_session_id():
    router = _router()
    body = json.dumps({"session_id": "from-body"}).encode()

    async def check():
        return await router.proxy.forward(
            _FakeRequest(body, {SESSION_HEADER: "from-header"}), "/v1/completions"
        )

    response = asyncio.run(check())
    assert response.headers[SESSION_HEADER] == "from-header"


def test_unparsable_body_still_routes():
    """The router does not validate payloads; the replica does."""
    router = _router()

    async def check():
        return await router.proxy.forward(_FakeRequest(b"not json at all"), "/v1/completions")

    response = asyncio.run(check())
    assert response.status_code == 200
    assert len(response.headers[SESSION_HEADER]) == 32


def test_consecutive_turns_reach_the_same_replica():
    client = _FakeClient()
    router = _router(client=client)

    async def check():
        first = await router.proxy.forward(_FakeRequest(b"{}"), "/v1/chat/completions")
        await _drain(first)
        session_id = first.headers[SESSION_HEADER]
        for _ in range(4):
            response = await router.proxy.forward(
                _FakeRequest(b"{}", {SESSION_HEADER: session_id}), "/v1/chat/completions"
            )
            await _drain(response)

    asyncio.run(check())
    urls = {request.url for request in client.sent}
    assert len(urls) == 1
    assert router.registry.affinity_hits == 4


# --- lifecycle of the outstanding counter ---

def test_outstanding_is_released_after_the_stream_completes():
    router = _router()

    async def check():
        response = await router.proxy.forward(_FakeRequest(b"{}"), "/v1/completions")
        mid_flight = [state.outstanding for state in router.registry.states]
        await _drain(response)
        return mid_flight

    mid_flight = asyncio.run(check())
    assert sum(mid_flight) == 1
    assert [state.outstanding for state in router.registry.states] == [0, 0]


def test_client_disconnect_closes_upstream_and_releases():
    """Closing the relay is serving's only cancellation path: there is no abort route."""
    client = _FakeClient(chunks=[b"a", b"b", b"c"])
    router = _router(client=client)

    async def check():
        response = await router.proxy.forward(_FakeRequest(b"{}"), "/v1/completions")
        iterator = response.body_iterator
        assert await iterator.__anext__() == b"a"
        # The client hangs up mid-stream.
        await iterator.aclose()

    asyncio.run(check())
    assert client.upstreams[0].closed is True
    assert [state.outstanding for state in router.registry.states] == [0, 0]


def test_transport_failure_returns_502_and_takes_the_replica_out():
    client = _FakeClient(send_error=OSError("connection refused"))
    router = _router(count=1, client=client)

    async def check():
        return await router.proxy.forward(_FakeRequest(b"{}"), "/v1/completions")

    response = asyncio.run(check())
    assert response.status_code == 502
    assert response.headers[SESSION_HEADER]
    assert router.registry.state("r0").ready is False
    assert router.registry.state("r0").outstanding == 0


def test_all_replicas_down_returns_503():
    router = _router()
    for state in router.registry.states:
        router.registry.set_ready(state.name, False)

    async def check():
        return await router.proxy.forward(_FakeRequest(b"{}"), "/v1/completions")

    response = asyncio.run(check())
    assert response.status_code == 503
    assert router.registry.rejected == 1


# --- router's own routes ---

def test_router_health_reports_the_replica_table():
    router = _router(count=2)
    router.registry.set_ready("r1", False)

    response = asyncio.run(router._health())
    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["replicas_ready"] == 1
    assert payload["replicas_total"] == 2
    assert [replica["ready"] for replica in payload["replicas"]] == [True, False]
    assert [replica["routed"] for replica in payload["replicas"]] == [0, 0]


def test_router_health_reports_where_requests_actually_went():
    client = _FakeClient()
    router = _router(count=2, client=client)
    router.registry.set_ready("r1", False)

    async def check():
        for _ in range(3):
            response = await router.proxy.forward(_FakeRequest(b"{}"), "/v1/completions")
            await _drain(response)
        return await router._health()

    payload = json.loads(asyncio.run(check()).body)
    by_name = {replica["name"]: replica["routed"] for replica in payload["replicas"]}
    assert by_name == {"r0": 3, "r1": 0}
    assert payload["routed"] == 3


def test_router_health_is_503_when_no_replica_is_routable():
    router = _router(count=1)
    router.registry.set_ready("r0", False)

    response = asyncio.run(router._health())
    assert response.status_code == 503
    assert json.loads(response.body)["status"] == "no_replicas"


def test_list_models_is_answered_by_a_routable_replica():
    payload = {"object": "list", "data": [{"id": "qwen"}]}
    client = _FakeClient(chunks=[json.dumps(payload).encode()])
    router = _router(count=2, client=client)
    router.registry.set_ready("r0", False)

    response = asyncio.run(router._list_models())
    assert json.loads(response.body) == payload
    assert client.gets == ["http://h:8001/v1/models"]


def test_list_models_is_503_when_nothing_is_routable():
    router = _router(count=1)
    router.registry.set_ready("r0", False)

    response = asyncio.run(router._list_models())
    assert response.status_code == 503


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_both_generation_routes_are_proxied(path):
    client = _FakeClient()
    router = _router(client=client)
    # Exercise the handler the app actually registered for this path.
    handler = next(r.endpoint for r in router.app.routes if getattr(r, "path", None) == path)

    async def check():
        response = await handler(_FakeRequest(b"{}"))
        await _drain(response)

    asyncio.run(check())
    assert client.sent[0].url.endswith(path)


def test_registered_routes_match_the_serving_api():
    router = _router()
    paths = {route.path for route in router.app.routes}
    assert {"/health", "/v1/models", "/v1/completions", "/v1/chat/completions"} <= paths
