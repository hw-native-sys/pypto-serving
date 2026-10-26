# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Common Router integration: role pools and P/D coordination, without devices."""

import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import httpx
import msgspec
import pytest
from starlette.requests import Request

from pypto_serving.pd.config import Config
from pypto_serving.pd.protocol import Binding, Component, Contract, Node, Topology
from pypto_serving.router.app import ServingRouter, default_client_factory
from pypto_serving.router.config import ReplicaSpec, RouterConfig, load_router_config


def config(p=1, d=1, **options):
    return RouterConfig(replicas=tuple(ReplicaSpec(f"{role[0]}{i}", f"{role[0]}{i}", 8000, role=role)
        for role, count in (("prefill", p), ("decode", d)) for i in range(count)),
        pd_options={"capacity": 1, "handoff_timeout": 0.04, "request_timeout": 1, **options},
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
        self.calls = []
        self.delay = 0
        self.p_status = self.d_status = 200
        self.p_hold = self.d_hold = None
        self.missing_done = False

    async def handle(self, req):
        name, path = req.url.host, req.url.path
        self.calls.append((name, path))
        if path.endswith("descriptor"):
            return httpx.Response(200, json=msgspec.to_builtins(self.nodes[name]))
        if path.endswith("abort"):
            for event in (self.p_hold, self.d_hold):
                if event is not None:
                    event.set()
            return httpx.Response(200, json={"accepted": True})
        payload = json.loads(req.content)
        context = json.loads(req.headers["x-pd-context"])
        binding = msgspec.convert(context["binding"], Binding)
        binding.check(self.nodes[name], self.nodes[name].role)
        if self.nodes[name].role == "prefill":
            if self.p_hold is not None:
                await self.p_hold.wait()
            return httpx.Response(self.p_status, json={"request": binding.request, "status": "ready"})
        if self.d_hold is not None:
            await self.d_hold.wait()
        await asyncio.sleep(self.delay)
        if payload.get("stream"):
            content = 'data: {"choices":[{"text":"answer"}]}\n\n'
            if not self.missing_done:
                content += 'data: [DONE]\n\n'
            return httpx.Response(self.d_status, text=content, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(self.d_status, json={"choices": [{"text": "answer"}]})

    def router(self, cfg):
        return ServingRouter(cfg, client_factory=lambda _: httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle), timeout=1))


def test_common_file_supplies_router_and_model_node_identity(tmp_path):
    body = {"replicas": [
        {"name": "p", "host": "127.0.0.1", "port": 8123, "role": "prefill", "control_port": 22222},
        {"name": "d", "host": "::1", "port": 8124, "role": "decode", "transfer_hostname": "data"}],
        "pd": {"capacity": 2}, "observability": {"enabled": False}}
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(body))
    common, pd = load_router_config(path), Config.load(path)
    assert common.is_pd and pd.capacity == 2
    assert pd.endpoint("prefill", "p").control_port == 22222
    assert pd.endpoint("decode", "d").transfer_hostname == "data"
    assert common.replicas[1].base_url == pd.decode[0].url == "http://[::1]:8124"
    assert not pd.log_enabled


@pytest.mark.parametrize("change", [
    {"runtime": {"prefill": [], "decode": []}},
    {"replicas": [{"host": "p", "port": 1, "role": "prefill"}]},
    {"replicas": [{"host": "p", "port": 1, "role": "unknown"}]},
    {"replicas": [{"host": "p", "port": 1}], "pd": {}},
    {"replicas": [{"host": "p", "port": 1, "control_port": 123}]},
])
def test_invalid_or_removed_deployment_format_is_rejected(tmp_path, change):
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(change))
    with pytest.raises(ValueError):
        load_router_config(path)


@pytest.mark.parametrize("extra", [{"provider": "missing"}, {"capacity": 0},
                                   {"handoff_timeout": 2, "request_timeout": 1}])
def test_pd_options_fail_before_client_creation(extra):
    def unexpected(_):
        pytest.fail("invalid configuration must not create a client")
    with pytest.raises(ValueError):
        ServingRouter(config(**extra), client_factory=unexpected)


def test_static_pd_cannot_mix_with_ordinary_or_managed_replicas():
    cfg = config()
    with pytest.raises(ValueError):
        replace(cfg, replicas=cfg.replicas + (ReplicaSpec("ordinary", "ordinary", 1),))
    from pypto_serving.router.config import HostSpec
    with pytest.raises(ValueError):
        replace(cfg, hosts=(HostSpec("managed", devices=(0,), model="weights"),))


def test_background_descriptors_drive_pairs_models_and_boots():
    async def run():
        cfg = config(2, 2)
        peers = Peers(cfg)
        router = peers.router(cfg)
        assert (await router._health()).status_code == 503
        async with router.app.router.lifespan_context(router.app):
            assert len(peers.calls) == 4
            assert len(router.proxy.pairs()) == 4
            router.registry.set_draining("d0", True)
            assert len(router.proxy.pairs()) == 2
            router.registry.set_draining("d0", False)
            assert {(p.node_id, d.node_id) for p, d in
                    (router.proxy.policy.select(router.proxy.pairs()) for _ in range(4))} == {
                        ("p0", "d0"), ("p1", "d0"), ("p0", "d1"), ("p1", "d1")}
            assert (await router._health()).status_code == 200
            assert json.loads((await router._list_models()).body)["data"][0]["id"] == "model"
            for _ in range(2):
                response = await router.proxy.forward(request({"model": "model"}), "/v1/completions")
                assert response.headers["x-session-id"] == "conversation"
            assert sum(path.endswith("descriptor") for _, path in peers.calls) == 4
            assert all(s.outstanding == 0 and not s.owned for s in router.registry.states)
            old = router.proxy.nodes["p0"]
            bound = Binding.create(old, router.proxy.nodes["d0"], {})
            peers.nodes["p0"] = msgspec.structs.replace(old, boot="restarted")
            await router.health.probe_once()
            with pytest.raises(ValueError):
                bound.check(router.proxy.nodes["p0"], "prefill")
            for name in ("d0", "d1"):
                peers.nodes[name] = msgspec.structs.replace(peers.nodes[name], healthy=False)
            await router.health.probe_once()
            # The common two-failure threshold cannot make an invalid PD descriptor selectable.
            assert not router.proxy.pairs() and not router.proxy.nodes.keys() & {"d0", "d1"}
            assert (await router._health()).status_code == 503
            assert json.loads((await router._list_models()).body)["data"] == []
        assert router.client.is_closed
        assert router.health._task is None
        assert not any(path.endswith("abort") for _, path in peers.calls)
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_pd_header_deadline_and_one_time_release(stream):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.delay = 0.08
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            if stream:
                from fastapi import HTTPException
                with pytest.raises(HTTPException) as error:
                    await router.proxy.forward(request({"stream": True}), "/v1/completions")
                assert error.value.status_code == 503
            else:
                response = await router.proxy.forward(request({}), "/v1/completions")
                assert response.status_code == 200
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert not router.proxy.requests.locked() and not router.proxy.handoffs.locked()
            assert not router.proxy.active
    asyncio.run(run())


def test_stream_output_overlaps_prefill_cleanup_and_checks_terminal():
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.p_hold = asyncio.Event()
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({"stream": True}), "/v1/completions")
            assert "answer" in await anext(response.body_iterator)
            terminal = asyncio.create_task(anext(response.body_iterator))
            await asyncio.sleep(0)
            assert not terminal.done()
            peers.p_hold.set()
            assert "[DONE]" in await terminal
            with pytest.raises(StopAsyncIteration):
                await anext(response.body_iterator)
            assert not router.proxy.active
            assert all(s.outstanding == 0 for s in router.registry.states)
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["p_error", "d_error", "missing_done", "cancel", "close"])
def test_pd_failures_cancel_both_ends_without_leaking_slots(failure):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        router = peers.router(cfg)
        async def managed_operation():
            pytest.fail("PD must not launch, adopt or stop external model processes")
        router.fleet.adopt_previous = router.fleet.launch_initial = router.fleet.stop_all = managed_operation
        if failure == "p_error": peers.p_status = 422
        if failure == "d_error": peers.d_status = 500
        if failure == "missing_done": peers.missing_done = True
        if failure in ("cancel", "close"): peers.d_hold = asyncio.Event()
        async with router.app.router.lifespan_context(router.app):
            pending = asyncio.create_task(router.proxy.forward(request({"stream": True}), "/v1/completions"))
            if failure in ("cancel", "close"):
                while not any(path == "/v1/completions" for _, path in peers.calls):
                    await asyncio.sleep(0)
                if failure == "close": await router.proxy.close()
                else: pending.cancel()
                with pytest.raises(asyncio.CancelledError): await pending
            elif failure == "missing_done":
                response = await pending
                body = "".join([piece async for piece in response.body_iterator])
                assert "pd_execution_error" in body and "[DONE]" in body
            else:
                from fastapi import HTTPException
                with pytest.raises(HTTPException): await pending
            assert sum(path.endswith("abort") for _, path in peers.calls) == 2
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert not router.proxy.requests.locked() and not router.proxy.handoffs.locked()
            assert not router.proxy.active and not router.client.is_closed
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["identity", "role", "provider", "layout"])
def test_unusable_descriptor_cannot_receive_a_generation(fault):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        node = peers.nodes["d0"]
        changes = {"identity": {"node_id": "wrong"}, "role": {"role": "prefill"},
                   "provider": {"provider": "other"},
                   "layout": {"contract": msgspec.structs.replace(node.contract, layout="other")}}
        peers.nodes["d0"] = msgspec.structs.replace(node, **changes[fault])
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            assert (await router._health()).status_code == 503
            assert json.loads((await router._list_models()).body)["data"] == []
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as error:
                await router.proxy.forward(request({"stream": True}), "/v1/completions")
            assert error.value.status_code == 503
            assert error.value.headers["x-session-id"] == "conversation"
            assert all(path.endswith("descriptor") for _, path in peers.calls)
            assert not router.proxy.active and not router.proxy.requests.locked()
    asyncio.run(run())


def test_cancel_while_waiting_for_handoff_releases_only_owned_slots():
    async def run():
        cfg = config()
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            await router.proxy.handoffs.acquire()
            pending = asyncio.create_task(router.proxy.forward(request({}), "/v1/completions"))
            while not router.proxy.requests.locked():
                await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert not router.proxy.requests.locked()
            assert router.proxy.handoffs.locked()
            assert not router.proxy.active
            assert all(s.outstanding == 0 for s in router.registry.states)
            router.proxy.handoffs.release()
    asyncio.run(run())


def test_router_close_retires_a_response_before_its_stream_starts():
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.p_hold = asyncio.Event()
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            pending = asyncio.create_task(router.proxy.forward(request({"stream": True}), "/v1/completions"))
            response = await pending
            assert pending.done() and router.proxy.active
            # A completed HTTP handler is not proof the unconsumed stream retired.
            await router.proxy.close()
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert not router.proxy.active
            assert sum(path.endswith("abort") for _, path in peers.calls) == 2
            assert not router.client.is_closed
            await response.body_iterator.aclose()
    asyncio.run(run())


def test_connection_budget_and_import_boundaries():
    async def run():
        cfg = config(2, 3, capacity=8)
        client = default_client_factory(cfg)
        try:
            assert client._transport._pool._max_connections >= 8 * (3 * (5 + 3) + 2) + 5
        finally:
            await client.aclose()
    asyncio.run(run())
    for pd in (False, True):
        blocked = ('torch', 'pypto', 'simpler', 'mooncake', 'pypto_serving.pd.adapter')
        if not pd: blocked += ('pypto_serving.pd', 'msgspec')
        script = """import sys, importlib.abc
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if any(fullname == p or fullname.startswith(p + '.') for p in BLOCKED):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
from pypto_serving.router.app import ServingRouter
from pypto_serving.router.config import ReplicaSpec, RouterConfig
replicas = (ReplicaSpec('p','p',8000,role='prefill'), ReplicaSpec('d','d',8000,role='decode')) if PD else (ReplicaSpec('r','r',8000),)
r = ServingRouter(RouterConfig(replicas=replicas), client_factory=lambda _: None)
assert len(r.app.routes) > 4
"""
        subprocess.run([sys.executable, "-c", f"BLOCKED={blocked!r}; PD={pd!r}\n" + script],
                       check=True, timeout=20)
    assert importlib.util.find_spec("pypto_serving.pd.router") is None
    root = Path(__file__).resolve().parents[3]
    assert 'pypto-pd-router' not in (root / "pyproject.toml").read_text()
