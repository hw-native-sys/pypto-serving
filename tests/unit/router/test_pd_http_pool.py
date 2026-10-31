# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Control traffic must progress while real HTTP generation connections are busy."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import socket

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import msgspec
import uvicorn

from pypto_serving.pd.protocol import Component, Contract, Node, Topology
from pypto_serving.router.app import ServingRouter
from pypto_serving.router.config import ReplicaSpec, RouterConfig
from tests.unit.router.test_pd_coordination import request


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), 5)


class Peer:
    def __init__(self, role):
        self.node = Node(role, role, "boot", Contract("model", "model", "layout",
            (Component("kv", "kv", "kv"),)), "registry", Topology.for_tp(4, 1))
        self.active = set()
        self.holds = {}
        self.probes = self.aborts = 0
        self.app = FastAPI()
        self.app.add_api_route("/internal/pd/descriptor", self.descriptor, methods=["GET"])
        self.app.add_api_route("/internal/pd/abort", self.abort, methods=["POST"])
        self.app.add_api_route("/v1/completions", self.generate, methods=["POST"])

    async def descriptor(self):
        self.probes += 1
        return JSONResponse(msgspec.to_builtins(self.node))

    async def abort(self, incoming: Request):
        self.aborts += 1
        self.holds[(await incoming.json())["request"]].set()
        return JSONResponse({"accepted": True})

    async def generate(self, incoming: Request):
        import json
        identity = json.loads(incoming.headers["x-pd-context"])["binding"]["request"]
        hold = self.holds[identity] = asyncio.Event()
        self.active.add(identity)
        if self.node.role == "prefill":
            try:
                await hold.wait()
                return JSONResponse({"request": identity, "status": "ready"})
            finally:
                self.active.discard(identity)

        async def events():
            try:
                yield 'data: {"choices":[{"text":"answer"}]}\n\n'
                await hold.wait()
                yield "data: [DONE]\n\n"
            finally:
                self.active.discard(identity)
        return StreamingResponse(events(), media_type="text/event-stream")


@asynccontextmanager
async def serve(peer):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        sock.setblocking(False)
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(peer.app, log_level="error", lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            await until(lambda: server.started or task.done())
            assert server.started
            yield port
        finally:
            for hold in peer.holds.values():
                hold.set()
            server.should_exit = True
            await asyncio.wait_for(task, 5)


def test_real_pool_keeps_descriptors_and_aborts_moving():
    async def run():
        peers = [Peer(role) for role in ("prefill", "decode")]
        async with AsyncExitStack() as stack:
            ports = [await stack.enter_async_context(serve(peer)) for peer in peers]
            config = RouterConfig(replicas=tuple(
                ReplicaSpec(peer.node.node_id, "127.0.0.1", port, role=peer.node.role,
                            control_port=19001 + i)
                for i, (peer, port) in enumerate(zip(peers, ports))),
                pd_options={"capacity": 2, "handoff_timeout": 3, "request_timeout": 5},
                observability={"enabled": False}, health_interval_seconds=60)
            router = ServingRouter(config)
            async with router.app.router.lifespan_context(router.app):
                responses = await asyncio.wait_for(asyncio.gather(*(
                    router.proxy.forward(request({"stream": True}), "/v1/completions")
                    for _ in range(2))), 3)
                await until(lambda: all(len(peer.active) == 2 for peer in peers))
                # These four connections are genuinely occupied, not MockTransport responses.
                assert sum(not c.is_idle() for c in router.client._transport._pool.connections) == 4
                await asyncio.wait_for(router.health.probe_once(), 2)
                assert all(peer.probes == 2 for peer in peers)
                assert (await router._health()).status_code == 200
                for response in responses:
                    assert "answer" in await anext(response.body_iterator)
                await asyncio.wait_for(asyncio.gather(*(
                    response.body_iterator.aclose() for response in responses)), 3)
                await until(lambda: not any(peer.active for peer in peers))
                assert all(peer.aborts == 2 for peer in peers)
                assert all(s.outstanding == 0 for s in router.registry.states)
                assert not router.proxy.active
                assert all(c.is_idle() for c in router.client._transport._pool.connections)
                await asyncio.wait_for(router.health.probe_once(), 2)
                assert all(peer.probes == 3 for peer in peers)
                for response in responses:
                    await response.body_iterator.aclose()
            assert router.client.is_closed and router.health._task is None
            assert not router.client._transport._pool.connections
    asyncio.run(run())
