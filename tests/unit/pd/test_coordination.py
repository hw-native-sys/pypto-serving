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
import json

import httpx
import msgspec
import pytest
from starlette.requests import Request

from pypto_serving.pd.protocol import DECODE_RESERVATION, Binding, Component, Contract, Node, Topology
from pypto_serving.pd.reservation_queue import ReservationQueue, ReservationWaiter
from pypto_serving.router.app import ServingRouter
from pypto_serving.router.config import ReplicaSpec, RouterConfig


def config(p=1, d=1):
    return RouterConfig(replicas=tuple(ReplicaSpec(f"{role[0]}{i}", f"{role[0]}{i}", 8000, role=role)
        for role, count in (("prefill", p), ("decode", d)) for i in range(count)),
        pd_options={"capacity": 1, "handoff_timeout": 0.04, "request_timeout": 1},
        observability={"enabled": False}, health_interval_seconds=60)


def request(payload, disconnect=None):
    body = json.dumps(payload).encode()
    async def receive():
        nonlocal body
        if body is not None:
            message, body = {"type": "http.request", "body": body}, None
            return message
        await (disconnect if disconnect is not None else asyncio.Event()).wait()
        return {"type": "http.disconnect"}
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


class ReservationPeers(Peers):
    def __init__(self, cfg, *, rejects=1, stage=DECODE_RESERVATION, source="d0"):
        super().__init__(cfg)
        self.rejects, self.stage, self.source = rejects, stage, source
        self.replies, self.aborts, self.placed, self.attempts = {}, {}, set(), []
        self.release_output = asyncio.Event()
        self.abort_status = 200

    async def handle(self, req):
        path, name = req.url.path, req.url.host
        if path.endswith("descriptor"):
            return await super().handle(req)
        if path.endswith("abort"):
            binding = json.loads(req.content)["request"]
            self.aborts.setdefault(binding, asyncio.Event()).set()
            return httpx.Response(self.abort_status)
        if "/requests/" in path:
            binding = path.rsplit("/", 1)[1]
            return httpx.Response(200, json=dict(request=binding, node=msgspec.to_builtins(self.nodes[name]),
                reservation={"state": "CONSTRUCTING"}, worker={"state_reservation_confirmed": binding in self.placed}))
        binding = json.loads(req.headers["x-pd-context"])["binding"]["request"]
        reply = self.replies.setdefault(binding, asyncio.get_running_loop().create_future())
        if name == "d0":
            self.attempts.append(binding)
            rejected = len(self.attempts) <= self.rejects
            reply.set_result(rejected)
            if not rejected:
                self.placed.add(binding)
        if await reply:
            if name != self.source:
                await self.aborts.setdefault(binding, asyncio.Event()).wait()
            return httpx.Response(503, json={"code": "capacity_exhausted", "stage": self.stage})
        if json.loads(req.content).get("prompt") == "held":
            await self.release_output.wait()
        return await super().handle(req)


async def until(predicate):
    async def check():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(check(), 1)


def test_decode_retry_retires_old_attempt_and_unblocks_at_placement(monkeypatch):
    monkeypatch.setattr("pypto_serving.pd.coordinator.RETRY_INTERVAL", 0.02)
    async def run():
        cfg = config()
        cfg.pd_options.update(capacity=3, reservation_retry_position=1)
        peers = ReservationPeers(cfg, source="p0")
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            held = asyncio.create_task(router.proxy.forward(request({"prompt": "held"}), "/v1/completions"))
            await until(lambda: router.proxy.reservations.pending)
            assert len(peers.attempts) == 1 and peers.aborts[peers.attempts[0]].is_set()
            assert all(state.outstanding == 0 for state in router.registry.states)
            router.proxy.reservations.released()
            await until(lambda: bool(peers.placed))
            assert len(set(peers.attempts)) == 2
            assert router.proxy.reservations.protected is not None
            response = await router.proxy.forward(request({}), "/v1/completions")
            assert response.status_code == 200 and not held.done()
            peers.release_output.set()
            assert (await held).status_code == 200
            assert_idle(router)
    asyncio.run(run())


@pytest.mark.parametrize("reason", ["cancel", "timeout", "unstaged", "unconfirmed"])
def test_waiting_and_failed_retirement_do_not_submit_another_request(reason):
    async def run():
        cfg = config()
        cfg.pd_options.update(handoff_timeout=0.06, request_timeout=0.25 if reason == "timeout" else 0.08)
        peers = ReservationPeers(cfg, rejects=2 if reason == "timeout" else 1,
                                 stage=None if reason == "unstaged" else DECODE_RESERVATION)
        if reason == "unconfirmed":
            peers.abort_status = 503
        router = peers.router(cfg)
        disconnect = asyncio.Event()
        async with router.app.router.lifespan_context(router.app):
            task = asyncio.create_task(router.proxy.forward(request({}, disconnect), "/v1/completions"))
            if reason == "cancel":
                await until(lambda: router.proxy.reservations.pending)
                disconnect.set()
            if reason == "timeout":
                await until(lambda: router.proxy.reservations.pending)
                router.proxy.reservations.released()
                await until(lambda: len(peers.attempts) == 2 and router.proxy.reservations.pending)
            result = await task
            assert result.status_code == (499 if reason == "cancel" else 503)
            if reason == "timeout":
                assert "Timed out waiting for Decode" in json.loads(result.body)["message"]
            if reason == "unconfirmed":
                assert json.loads(result.body)["code"] == "pd_retirement_unconfirmed"
            router.proxy.reservations.released()
            assert len(peers.attempts) == (2 if reason == "timeout" else 1)
            assert not router.proxy.reservations
            assert_idle(router)
    asyncio.run(run())


def test_retry_priority_advances_and_protects_the_first_level_one_waiter():
    async def run():
        queue = ReservationQueue(8, 5)
        primary, other = ReservationWaiter(), ReservationWaiter(position=2)
        for position in (5, 4, 3, 2, 1):
            ahead = [ReservationWaiter() for _ in range(5)]
            for waiter in ahead:
                queue.put(waiter)
            queue.put(primary, generation=queue.generation, rejected=True)
            assert queue.pending[position - 1] is primary and primary.position == position
            for waiter in ahead:
                queue.leave(waiter)
            assert not queue.take(primary)
            queue.released()
            assert queue.take(primary)
        queue.put(other, generation=queue.generation, rejected=True)
        queue.released()
        assert not queue.take(other)
        queue.put(primary, generation=queue.generation, rejected=True)
        assert list(queue.pending) == [primary, other] and not queue.take(other)
        queue.released()
        assert queue.take(primary)
        queue.leave(primary)
        assert queue.take(other)
        queue.leave(other)
        assert not queue
    asyncio.run(run())


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


@pytest.mark.parametrize("prefill_error", [False, True, "capacity"])
def test_sse_error_keeps_stream_boundary(prefill_error):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.p_hold = asyncio.Event()
        if prefill_error == "capacity":
            peers.errors["p0"] = httpx.Response(503, json={"code": "capacity_exhausted", "stage": DECODE_RESERVATION})
        elif prefill_error:
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
            assert len(peers.requests) == 2
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
