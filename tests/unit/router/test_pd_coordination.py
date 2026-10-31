# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Basic configuration, routing and request lifecycle checks for PD."""

import asyncio
from dataclasses import replace
import json

import httpx
import msgspec
import pytest
from starlette.requests import Request

from pypto_serving.pd.config import Config
from pypto_serving.pd.protocol import Binding, Component, Contract, Node, Topology
from pypto_serving.router.app import ServingRouter
from pypto_serving.router.config import HostSpec, ReplicaSpec, RouterConfig, load_router_config


def config(p=1, d=1):
    return RouterConfig(replicas=tuple(ReplicaSpec(f"{role[0]}{i}", f"{role[0]}{i}", 8000, role=role)
        for role, count in (("prefill", p), ("decode", d)) for i in range(count)),
        pd_options={"capacity": 1, "handoff_timeout": 0.04, "request_timeout": 1},
        observability={"enabled": False}, health_interval_seconds=60)


def request(payload):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode()}
    return Request({"type": "http", "method": "POST", "path": "/v1/completions",
                    "headers": [(b"x-session-id", b"conversation")]}, receive)


class Peers:
    def __init__(self, cfg):
        contract = Contract("model", "model", "layout", (Component("kv", "kv", "kv"),))
        self.nodes = {r.name: Node(r.name, r.role, "boot-" + r.name, contract, "registry-" + r.name,
                                  Topology.for_tp(4, 1)) for r in cfg.replicas}
        self.calls, self.requests, self.errors = [], [], {}
        self.p_hold = self.d_stream = self.stream_body = None

    async def handle(self, req):
        name, path = req.url.host, req.url.path
        self.calls.append((name, path))
        if path.endswith("descriptor"):
            return httpx.Response(200, json=msgspec.to_builtins(self.nodes[name]))
        if path.endswith("abort"):
            if self.p_hold is not None:
                self.p_hold.set()
            return httpx.Response(200, json={"accepted": True})
        self.requests.append(req)
        context = json.loads(req.headers["x-pd-context"])
        binding = msgspec.convert(context["binding"], Binding)
        binding.check(self.nodes[name], self.nodes[name].role)
        if self.nodes[name].role == "prefill" and self.p_hold is not None:
            await self.p_hold.wait()
        if name in self.errors:
            return self.errors[name]
        if self.nodes[name].role == "prefill":
            return httpx.Response(200, json={"request": binding.request, "status": "ready"})
        if json.loads(req.content).get("stream"):
            if self.d_stream is not None:
                return httpx.Response(200, stream=self.d_stream)
            content = self.stream_body or 'data: {"choices":[{"text":"answer"}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, text=content, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"text": "answer"}]})

    def router(self, cfg):
        return ServingRouter(cfg, client_factory=lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle), timeout=1))


def assert_idle(router):
    assert not router.proxy.active
    assert all(s.outstanding == 0 for s in router.registry.states)


def test_shared_config_and_static_pd_nodes(tmp_path):
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps({"replicas": [
        {"name": "p", "host": "127.0.0.1", "port": 8123, "role": "prefill", "control_port": 22222},
        {"name": "d", "host": "127.0.0.2", "port": 8124, "role": "decode"}],
        "pd": {"capacity": 2}, "observability": {"enabled": False}}))
    common, pd = load_router_config(path), Config.load(path)
    assert common.is_pd and pd.capacity == 2 and not pd.log_enabled
    assert pd.prefill[0].node_id == common.replicas[0].name
    assert pd.prefill[0].control_port == 22222
    assert pd.decode[0].url == common.replicas[1].base_url
    with pytest.raises(ValueError):
        replace(common, replicas=common.replicas + (ReplicaSpec("r", "r", 8000),))
    with pytest.raises(ValueError):
        replace(common, hosts=(HostSpec("host", devices=(0,), model="weights"),))


@pytest.mark.parametrize("stream", [False, True])
def test_json_and_sse_generation(stream):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        if stream:
            peers.p_hold = asyncio.Event()
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({"stream": stream}), "/v1/completions")
            assert response.status_code == 200 and response.headers["x-session-id"] == "conversation"
            if stream:
                first = await anext(response.body_iterator)
                assert "answer" in first and not peers.p_hold.is_set()
                peers.p_hold.set()
                body = first + "".join([part async for part in response.body_iterator])
                assert body.endswith("data: [DONE]\n\n")
            else:
                assert json.loads(response.body)["choices"][0]["text"] == "answer"
            assert_idle(router)
            assert not any(path.endswith("abort") for _, path in peers.calls)
    asyncio.run(run())


def test_pool_rotation_health_and_new_boot():
    async def run():
        cfg = config(2, 2)
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            assert (await router._health()).status_code == 200
            assert json.loads((await router._list_models()).body)["data"][0]["id"] == "model"
            for _ in range(4):
                assert (await router.proxy.forward(request({}), "/v1/completions")).status_code == 200
            bindings = [json.loads(req.headers["x-pd-context"])["binding"]
                        for req in peers.requests if req.url.host.startswith("p")]
            assert {(b["prefill"][0], b["decode"][0]) for b in bindings} == {
                (p, d) for p in ("p0", "p1") for d in ("d0", "d1")}
            old = Binding.create(peers.nodes["p0"], peers.nodes["d0"], {})
            peers.nodes["p0"] = msgspec.structs.replace(peers.nodes["p0"], boot="restarted")
            await router.health.probe_once()
            with pytest.raises(ValueError):
                old.check(router.proxy.nodes["p0"], "prefill")
            peers.nodes["d0"] = msgspec.structs.replace(peers.nodes["d0"], healthy=False)
            await router.health.probe_once()
            assert all(d.node_id == "d1" for _, d in router.proxy.pairs())
            assert_idle(router)
    asyncio.run(run())


@pytest.mark.parametrize("unavailable", [False, True])
def test_unknown_model_and_unavailable_backend(unavailable):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        if unavailable:
            peers.nodes["d0"] = msgspec.structs.replace(peers.nodes["d0"], healthy=False)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({"model": "model" if unavailable else "missing"}),
                                                  "/v1/completions")
            assert response.status_code == (503 if unavailable else 404)
            assert json.loads(response.body)["code"] == (
                "no_available_backend" if unavailable else "model_not_found")
            assert not peers.requests
            assert_idle(router)
    asyncio.run(run())


@pytest.mark.parametrize("node", ["p0", "d0"])
def test_node_error_is_forwarded(node):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.errors[node] = httpx.Response(503, json={"detail": "unavailable"}, headers={"Retry-After": "1"})
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({}), "/v1/completions")
            assert response.status_code == 503 and json.loads(response.body) == {"detail": "unavailable"}
            assert response.headers["x-pypto-error-source"] == "upstream"
            assert response.headers["retry-after"] == "1"
            assert sum(path.endswith("abort") for _, path in peers.calls) == 2
            assert_idle(router)
    asyncio.run(run())


@pytest.mark.parametrize("prefill_error", [False, True])
def test_sse_error_keeps_stream_boundary(prefill_error):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.p_hold = asyncio.Event()
        if prefill_error:
            peers.errors["p0"] = httpx.Response(422, json={"detail": "prefill failed"})
        else:
            peers.stream_body = 'data: {"error":{"message":"decode failed"}}\n\ndata: [DONE]\n\n'
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({"stream": True}), "/v1/completions")
            assert response.status_code == 200
            first = await anext(response.body_iterator)
            peers.p_hold.set()
            body = first + "".join([part async for part in response.body_iterator])
            errors = [json.loads(line[6:])["error"] for line in body.splitlines()
                      if line.startswith("data: {") and "error" in json.loads(line[6:])]
            assert len(errors) == 1 and body.endswith("data: [DONE]\n\n")
            if prefill_error:
                assert errors[0]["source"] == "upstream"
            else:
                assert errors[0] == {"message": "decode failed"}
            assert_idle(router)
    asyncio.run(run())


def test_capacity_rejection_and_cancel_release():
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.p_hold = asyncio.Event()
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            active = await asyncio.create_task(router.proxy.forward(request({"stream": True}), "/v1/completions"))
            assert "answer" in await anext(active.body_iterator)
            count = len(peers.requests)
            rejected = await router.proxy.forward(request({}), "/v1/completions")
            assert rejected.status_code == 503 and json.loads(rejected.body)["code"] == "capacity_exhausted"
            assert rejected.headers["retry-after"] == "1" and len(peers.requests) == count
            await active.body_iterator.aclose()
            assert_idle(router)
            assert sum(path.endswith("abort") for _, path in peers.calls) == 2
            assert (await router.proxy.forward(request({}), "/v1/completions")).status_code == 200
            assert_idle(router)
    asyncio.run(run())
