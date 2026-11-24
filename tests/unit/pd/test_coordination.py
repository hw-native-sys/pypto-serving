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
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import httpx
import msgspec
import pytest
from starlette.requests import Request

from pypto_serving.pd.integration import Server
from pypto_serving.pd.protocol import Binding, Component, Contract, Node, Topology, digest
from pypto_serving.pd.service import Service
from pypto_serving.pd.transfer import Region, TransferError
from pypto_serving.pd.worker import WorkerRuntime
from pypto_serving.router.app import ServingRouter
from pypto_serving.router.config import ReplicaSpec, RouterConfig


def config(p=1, d=1, provider="mooncake"):
    return RouterConfig(replicas=tuple(ReplicaSpec(f"{role[0]}{i}", f"{role[0]}{i}", 8000, role=role)
        for role, count in (("prefill", p), ("decode", d)) for i in range(count)),
        pd_options={"capacity": 1, "handoff_timeout": 0.04, "request_timeout": 1, "provider": provider},
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
                                  Topology.for_tp(4, 1), cfg.pd_options["provider"]) for r in cfg.replicas}
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


@pytest.mark.parametrize(("operation", "node"), [
    ("install", "p0"), ("connect_ranks", "p0"), ("prepare_link", "d0"),
])
def test_ordered_hixl_fault_withdraws_only_its_local_node(operation, node):
    async def run():
        cfg = config(2, 2, "hixl")
        peers = Peers(cfg)
        local = peers.nodes[node]
        runtime = WorkerRuntime.__new__(WorkerRuntime)
        runtime.settings = SimpleNamespace(provider="hixl", role=local.role, peer_capacity=2)
        runtime.lock, runtime.uncertain, runtime.accepting = threading.Lock(), False, True
        runtime.model = SimpleNamespace(ranks=1, tp=1, regions=lambda: {"kv": None})
        runtime.peers, runtime.peer_identities = {}, {}

        def fail_native(op, **payload):
            assert op == operation
            raise RuntimeError("injected native link fault")

        runtime.owners = (SimpleNamespace(call=fail_native),)
        region = Region("decode-boot", 0, "kv", 64, {})
        registry = {"topology": Topology.for_tp(1, 1), "regions": [region]}
        identity = ("d0", region.boot, digest(registry))
        if operation == "connect_ranks":
            runtime.peers[region.boot] = (region,)
            runtime.peer_identities[region.boot] = identity
        payload = {"install": dict(peer=identity, registry=registry),
                   "connect_ranks": dict(peer=identity, rank_pairs=((0, 0),)), "prepare_link": [0]}[operation]

        async def ordered_call(op, wire):
            # The common ordered RPC only delivers bytes on success.
            try:
                return runtime.handle(op, wire)
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc

        core = SimpleNamespace(is_ready=lambda: True, call_worker=ordered_call,
                               kv_cache_manager=SimpleNamespace(tiering=None))
        endpoint = SimpleNamespace(node_id=node)
        service = Service(SimpleNamespace(prefill=(0, 1), decode=(0, 1), provider="hixl"),
                          endpoint, local.role, None, core, None)
        service.node = local
        server = SimpleNamespace(service=service)
        handle = peers.handle

        async def probe_real_service(req):
            if req.url.host == node and req.url.path == "/internal/pd/descriptor":
                response = await Server.descriptor(server)
                return httpx.Response(response.status_code, content=response.body)
            return await handle(req)

        peers.handle = probe_real_service
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            assert len(router.proxy.pairs()) == 4 and (await Server._health(server)).status_code == 200
            with ThreadPoolExecutor() as runtime.pool:
                with pytest.raises(TransferError, match="injected native link fault") as error:
                    await service.worker(operation, payload)
            assert error.value.uncertain and runtime.uncertain and core.is_ready()
            assert (await Server._health(server)).status_code == 503
            assert json.loads((await Server.descriptor(server)).body)["healthy"] is False
            await router.health.probe_once()
            assert not router.registry.state(node).routable
            assert len(router.proxy.pairs()) == 2
            for _ in range(2):
                assert (await router.proxy.forward(request({}), "/v1/completions")).status_code == 200
            assert peers.requests and all(req.url.host != node for req in peers.requests)
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
