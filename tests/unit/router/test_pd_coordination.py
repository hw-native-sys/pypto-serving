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
import gzip
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


def request(payload, *, raw=None):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode() if raw is None else raw}
    return Request({"type": "http", "method": "POST", "path": "/v1/completions",
                    "headers": [(b"x-session-id", b"conversation")]}, receive)


class Peers:
    def __init__(self, cfg):
        contract = Contract("model", "model", "layout", (Component("kv", "kv", "kv"),))
        self.nodes = {r.name: Node(r.name, r.role, "boot-" + r.name, contract, "registry-" + r.name,
                                  Topology.for_tp(4, 1)) for r in cfg.replicas}
        self.calls = []
        self.requests = []
        self.errors = {}
        self.delay = 0
        self.p_status = self.d_status = 200
        self.p_hold = self.d_hold = None
        self.missing_done = False
        self.stream_body = None

    async def handle(self, req):
        name, path = req.url.host, req.url.path
        self.calls.append((name, path))
        self.requests.append(req)
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
        if name in self.errors:
            return self.errors[name]
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
            if self.stream_body is not None:
                content = self.stream_body
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
                response = await router.proxy.forward(request({"stream": True}), "/v1/completions")
                assert response.status_code == 503
                assert json.loads(response.body)["code"] == "pd_operation_timeout"
            else:
                response = await router.proxy.forward(request({}), "/v1/completions")
                assert response.status_code == 200
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert router.proxy.requests == router.proxy.handoffs == 0
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
        releases = []
        release = router.registry.release
        def record_release(name):
            releases.append(name)
            release(name)
        router.registry.release = record_release
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
                response = await pending
                assert response.status_code == (422 if failure == "p_error" else 500)
                assert response.headers["x-pypto-error-source"] == "upstream"
            assert sum(path.endswith("abort") for _, path in peers.calls) == 2
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert router.proxy.requests == router.proxy.handoffs == 0
            assert not router.proxy.active and not router.client.is_closed
            assert releases.count("p0") == releases.count("d0") == 1
    asyncio.run(run())


@pytest.mark.parametrize("fault", ["identity", "role", "provider", "layout", "version"])
def test_unusable_descriptor_cannot_receive_a_generation(fault):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        node = peers.nodes["d0"]
        changes = {"identity": {"node_id": "wrong"}, "role": {"role": "prefill"},
                   "provider": {"provider": "other"},
                   "version": {"version": 99},
                   "layout": {"contract": msgspec.structs.replace(node.contract, layout="other")}}
        peers.nodes["d0"] = msgspec.structs.replace(node, **changes[fault])
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            assert (await router._health()).status_code == 503
            assert json.loads((await router._list_models()).body)["data"] == []
            error = await router.proxy.forward(request({"stream": True}), "/v1/completions")
            assert error.status_code == 503
            assert error.headers["x-session-id"] == "conversation"
            assert all(path.endswith("descriptor") for _, path in peers.calls)
            assert not router.proxy.active and router.proxy.requests == router.proxy.handoffs == 0
    asyncio.run(run())


@pytest.mark.parametrize("gate", ["requests", "handoffs"])
def test_a_full_global_gate_rejects_without_taking_or_releasing_a_slot(gate):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            setattr(router.proxy, gate, 1)
            response = await asyncio.wait_for(router.proxy.forward(request({}), "/v1/completions"), 0.1)
            assert response.status_code == 503 and json.loads(response.body)["code"] == "capacity_exhausted"
            assert response.headers["retry-after"] == "1" and response.headers["x-pypto-error-source"] == "router"
            assert getattr(router.proxy, gate) == 1
            assert not router.proxy.active
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert all(path.endswith("descriptor") for _, path in peers.calls)
            assert router.registry.rejected == 1 and router.registry.total_routed() == 0
            setattr(router.proxy, gate, 0)
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


@pytest.mark.parametrize("node", ["p0", "d0"])
@pytest.mark.parametrize("encoding", ["json", "text", "gzip"])
def test_node_error_keeps_body_status_and_meaningful_headers(node, encoding):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        body = (json.dumps({"detail": "reason " * 400}).encode() if encoding == "json"
                else ("unavailable " * 300).encode())
        headers = {"Content-Type": "application/json" if encoding == "json" else "text/plain",
                   "Retry-After": "7", "X-Request-Id": "backend-request",
                   "Connection": "close, x-private-hop", "X-Private-Hop": "drop", "Trailer": "Digest",
                   "X-PyPTO-Error-Source": "forged"}
        if encoding == "gzip":
            headers["Content-Encoding"] = "gzip"
        peers.errors[node] = httpx.Response(503, content=gzip.compress(body) if encoding == "gzip" else body,
                                           headers=headers)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({}), "/v1/completions")
            assert response.status_code == 503 and response.body == body
            assert response.headers["content-type"] == headers["Content-Type"]
            assert response.headers["retry-after"] == "7"
            assert response.headers["x-request-id"] == "backend-request"
            assert response.headers["x-pypto-error-source"] == "upstream"
            assert "content-encoding" not in response.headers and "x-private-hop" not in response.headers
            assert "trailer" not in response.headers
            assert int(response.headers["content-length"]) == len(body)
            assert sum(path.endswith("abort") for _, path in peers.calls) == 2
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert router.registry.rejected == 0
    asyncio.run(run())


def test_forwarded_headers_and_http_phase_timeouts():
    async def run():
        cfg = replace(config(request_timeout=0.2), connect_timeout_seconds=0.01, request_timeout_seconds=0.03)
        peers = Peers(cfg)
        router = peers.router(cfg)
        router.client.timeout = httpx.Timeout(0.03, connect=0.01, read=None)
        incoming = request({})
        incoming.scope["headers"] += [(k.encode(), v.encode()) for k, v in {
            "authorization": "Bearer test", "traceparent": "trace", "x-request-id": "request",
            "connection": "x-private-hop", "x-private-hop": "drop", "content-encoding": "identity",
            "content-type": "text/plain", "content-length": "12345", "x-pd-context": "forged",
            "x-pypto-error-source": "forged"}.items()]
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(incoming, "/v1/completions")
            assert response.status_code == 200
            posts = [r for r in peers.requests if r.url.path == "/v1/completions"]
            assert len(posts) == 2
            for post in posts:
                assert post.headers["authorization"] == "Bearer test"
                assert post.headers["traceparent"] == "trace"
                assert post.headers["x-request-id"] == "request"
                assert "binding" in json.loads(post.headers["x-pd-context"])
                assert not {"x-private-hop", "content-encoding", "x-pypto-error-source"} & post.headers.keys()
                assert post.headers["content-type"] == "application/json"
                assert int(post.headers["content-length"]) == len(post.content)
                assert post.extensions["timeout"] == {"connect": 0.01, "read": 0.2, "write": 0.03, "pool": 0.03}
            health = json.loads((await router._health()).body)
            assert health["routed"] == 1 and health["rejected"] == 0
            assert {role: values["routed"] for role, values in health["roles"].items()} == {
                "prefill": 1, "decode": 1}
    asyncio.run(run())


def test_model_discovery_separates_identity_from_health_and_unknown_models():
    async def run():
        cfg = config()
        peers = Peers(cfg)
        router = peers.router(cfg)
        original = peers.handle
        missing = {"p0", "d0"}
        async def handle(req):
            if req.url.path.endswith("descriptor") and req.url.host in missing:
                raise httpx.ConnectError("not reachable", request=req)
            return await original(req)
        router.client._transport = httpx.MockTransport(handle)
        async with router.app.router.lifespan_context(router.app):
            for remaining in ({"p0", "d0"}, {"p0"}):
                missing.clear()
                missing.update(remaining)
                await router.health.probe_once()
                error = await router.proxy.forward(request({"model": "unknown"}), "/v1/completions")
                assert error.status_code == 503 and json.loads(error.body)["code"] == "no_available_backend"
            missing.clear()
            peers.nodes["p0"] = msgspec.structs.replace(peers.nodes["p0"], healthy=False)
            await router.health.probe_once()
            assert router.proxy.models == {"model"} and len(router.proxy.discovered) == 2
            assert json.loads((await router._list_models()).body)["data"] == []
            known = await router.proxy.forward(request({"model": "model"}), "/v1/completions")
            unknown = await router.proxy.forward(request({"model": "unknown"}), "/v1/completions")
            assert known.status_code == 503 and unknown.status_code == 404
            assert json.loads(unknown.body)["code"] == "model_not_found"
            peers.nodes["p0"] = msgspec.structs.replace(peers.nodes["p0"], healthy=True, boot="new")
            await router.health.probe_once()
            assert (await router.proxy.forward(request({"model": "model"}), "/v1/completions")).status_code == 200
    asyncio.run(run())


def test_local_pool_pressure_does_not_invalidate_a_healthy_descriptor():
    async def run():
        cfg = config()
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            before = dict(router.proxy.nodes)
            async def busy(req):
                raise httpx.PoolTimeout("local pool full", request=req)
            router.client._transport = httpx.MockTransport(busy)
            await router.health.probe_once()
            await router.health.probe_once()
            assert router.proxy.nodes == before
            assert all(s.ready and s.failures == 0 for s in router.registry.states)
            error = await router.proxy.forward(request({}), "/v1/completions")
            assert error.status_code == 503 and json.loads(error.body)["code"] == "http_pool_exhausted"
            assert "retry-after" not in error.headers  # It may already have dispatched its peer POST.
            assert router.registry.rejected == 0 and all(s.outstanding == 0 for s in router.registry.states)
    asyncio.run(run())


@pytest.mark.parametrize("node_error", [False, True])
def test_errors_after_stream_headers_keep_the_stream_error_boundary(node_error):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.p_hold = asyncio.Event()
        peers.p_status = 422
        original = {"error": {"message": "node failed", "code": "node_specific"}}
        if node_error:
            peers.stream_body = "data: " + json.dumps(original) + "\n\ndata: [DONE]\n\n"
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({"stream": True}), "/v1/completions")
            assert response.status_code == 200 and "x-pypto-error-source" not in response.headers
            first = await anext(response.body_iterator)
            peers.p_hold.set()
            body = first + "".join([part async for part in response.body_iterator])
            data = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
            errors = [json.loads(line) for line in data if line != "[DONE]" and "error" in json.loads(line)]
            assert len(errors) == 1 and data[-1] == "[DONE]"
            if node_error:
                assert errors == [original]
            else:
                assert errors[0]["error"]["code"] == "upstream_error"
                assert errors[0]["error"]["source"] == "upstream"
                assert errors[0]["error"]["type"] == "pd_execution_error"
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert sum(path.endswith("abort") for _, path in peers.calls) == 2
    asyncio.run(run())


def test_a_cleanup_failure_does_not_replace_the_original_node_error(caplog):
    async def run():
        peers = Peers(config())
        peers.errors["p0"] = httpx.Response(422, content=b"original error", headers={"Content-Type": "text/plain"})
        router = peers.router(config())
        async def broken_abort(*args):
            raise RuntimeError("secondary cleanup failure")
        router.proxy.abort = broken_abort
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({}), "/v1/completions")
            assert response.status_code == 422 and response.body == b"original error"
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert not router.proxy.active
        assert "secondary cleanup failure" in caplog.text
    asyncio.run(run())


def test_prefill_completion_does_not_restart_the_header_wait_budget():
    async def run():
        cfg = config(handoff_timeout=0.1)
        peers = Peers(cfg)
        peers.p_hold = asyncio.Event()
        peers.delay = 0.15
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            async def release():
                await asyncio.sleep(0.07)
                peers.p_hold.set()
            released = asyncio.create_task(release())
            response = await router.proxy.forward(request({"stream": True}), "/v1/completions")
            await released
            assert response.status_code == 503 and json.loads(response.body)["code"] == "pd_operation_timeout"
            assert all(s.outstanding == 0 for s in router.registry.states)
    asyncio.run(run())


def test_capacity_skips_full_nodes_and_releases_once_on_cancellation():
    async def run():
        cfg = config(2, 2, handoff_timeout=1, request_timeout=2)
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            assert (await router.proxy.forward(request({}), "/v1/completions")).status_code == 200
            # The next rotation starts at p1/d1. Its occupied slots must then be
            # skipped at p0/d1 and p1/d0 before p0/d0 can accept the second request.
            peers.p_hold, peers.d_hold = asyncio.Event(), asyncio.Event()
            previous_posts = sum(path == "/v1/completions" for _, path in peers.calls)
            pending = []
            for expected in (1, 2):
                pending.append(asyncio.create_task(router.proxy.forward(request({}), "/v1/completions")))
                while sum(path == "/v1/completions" for _, path in peers.calls) < previous_posts + 2 * expected:
                    await asyncio.sleep(0)
                assert all(s.outstanding <= 1 for s in router.registry.states)
            assert router.proxy.requests == router.proxy.handoffs == 2
            assert {req.url.host for req in peers.requests if req.url.path == "/v1/completions"} == {
                "p0", "p1", "d0", "d1"}
            count = len(peers.calls)
            rejected = await asyncio.wait_for(asyncio.gather(*(
                router.proxy.forward(request({}), "/v1/completions") for _ in range(16))), 0.5)
            assert all(response.status_code == 503 for response in rejected) and len(peers.calls) == count
            assert all(json.loads(response.body)["code"] == "capacity_exhausted" for response in rejected)
            assert router.registry.rejected == 16 and router.registry.total_routed() == 3
            for task in pending:
                task.cancel()
            results = await asyncio.gather(*pending, return_exceptions=True)
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert all(s.outstanding == 0 for s in router.registry.states)
            assert router.proxy.requests == router.proxy.handoffs == 0
            assert not router.proxy.active
            assert sum(path.endswith("abort") for _, path in peers.calls) == 4
            assert (await router.proxy.forward(request({}), "/v1/completions")).status_code == 200
    asyncio.run(run())


def test_prefill_capacity_returns_before_decode_json_headers():
    async def run():
        cfg = config(1, 2, handoff_timeout=1, request_timeout=2)
        peers = Peers(cfg)
        peers.d_hold = asyncio.Event()
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            first = asyncio.create_task(router.proxy.forward(request({}), "/v1/completions"))
            while router.registry.state("d0").outstanding == 0 or router.registry.state("p0").outstanding:
                await asyncio.sleep(0)
            assert not first.done()
            assert router.proxy.requests == 1 and router.proxy.handoffs == 0
            second = asyncio.create_task(router.proxy.forward(request({}), "/v1/completions"))
            while router.registry.state("d1").outstanding == 0 or router.registry.state("p0").outstanding:
                await asyncio.sleep(0)
            assert not second.done() and router.proxy.requests == 2
            assert router.proxy.handoffs == 0
            assert (await router.proxy.forward(request({}), "/v1/completions")).status_code == 503
            peers.d_hold.set()
            assert all(response.status_code == 200 for response in await asyncio.gather(first, second))
            assert router.proxy.requests == router.proxy.handoffs == 0
            assert all(s.outstanding == 0 for s in router.registry.states)
    asyncio.run(run())


def test_round_robin_positions_survive_health_changes_and_use_current_boots():
    async def run():
        cfg = config(2, 2)
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            async def route():
                assert (await router.proxy.forward(request({}), "/v1/completions")).status_code == 200
                post = next(r for r in reversed(peers.requests) if r.url.path == "/v1/completions")
                context = json.loads(post.headers["x-pd-context"])
                return context["binding"]
            assert (await route())["decode"][0] == "d0"
            router.registry.set_draining("d1", True)
            selected = await route()
            assert (selected["prefill"][0], selected["decode"][0]) == ("p1", "d0")
            router.registry.set_draining("d1", False)
            selected = await route()
            assert (selected["prefill"][0], selected["decode"][0]) == ("p0", "d0")
            peers.nodes["p1"] = msgspec.structs.replace(peers.nodes["p1"], boot="restarted", registry="new")
            await router.health.probe_once()
            selected = await route()
            assert selected["prefill"] == ["p1", "restarted", "new"]
            assert selected["decode"][0] == "d1"
            assert router.registry.total_routed() == 4
    asyncio.run(run())


@pytest.mark.parametrize("raw", [b"{", b"null", b"[]", b'{"model":false}', b'{"model":[]}'])
def test_invalid_requests_have_stable_errors_without_dispatch(raw):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request(None, raw=raw), "/v1/completions")
            value = json.loads(response.body)
            assert response.status_code == 400 and value["code"] == "invalid_request"
            assert value["object"] == "error" and value["message"]
            assert response.headers["x-pypto-error-source"] == "router"
            assert response.headers["x-session-id"] == "conversation"
            assert router.registry.rejected == 1 and router.registry.total_routed() == 0
            assert router.proxy.requests == router.proxy.handoffs == 0 and not router.proxy.active
            assert all(path.endswith("descriptor") for _, path in peers.calls)
    asyncio.run(run())


@pytest.mark.parametrize("fault", [{"node_id": "other"}, {"role": "prefill"},
                                   {"provider": "other"}, {"version": 99}])
def test_invalid_descriptors_cannot_poison_model_discovery(fault):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        original = peers.nodes["d0"]
        peers.nodes["d0"] = msgspec.structs.replace(original,
            contract=msgspec.structs.replace(original.contract, model="unverified-model"), **fault)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            assert router.proxy.models == {"model"} and router.proxy.discovered == {"p0"}
            error = await router.proxy.forward(request({"model": "unverified-model"}), "/v1/completions")
            assert error.status_code == 503
            peers.nodes["d0"] = original
            await router.health.probe_once()
            error = await router.proxy.forward(request({"model": "unverified-model"}), "/v1/completions")
            assert error.status_code == 404 and router.proxy.models == {"model"}
            assert all(path.endswith("descriptor") for _, path in peers.calls)
    asyncio.run(run())


def test_first_invalid_health_sample_is_visible_in_pd_role_metrics():
    async def run():
        cfg = config()
        peers = Peers(cfg)
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            peers.nodes["d0"] = msgspec.structs.replace(peers.nodes["d0"], healthy=False)
            await router.health.probe_once()
            response = await router._health()
            status = json.loads(response.body)
            assert response.status_code == 503 and status["replicas_ready"] == 1
            assert status["roles"]["decode"]["ready"] == 0
            assert not router.registry.state("d0").ready
            peers.nodes["d0"] = msgspec.structs.replace(peers.nodes["d0"], healthy=True, boot="new")
            await router.health.probe_once()
            assert (await router._health()).status_code == 200
    asyncio.run(run())


@pytest.mark.parametrize("body", ['data: {broken}\n\n', 'data: []\n\n',
                                  'data: {"choices":[{"text":"partial"}]}\n\n'])
def test_malformed_or_incomplete_sse_reports_a_router_protocol_error(body):
    async def run():
        cfg = config()
        peers = Peers(cfg)
        peers.stream_body = body
        router = peers.router(cfg)
        async with router.app.router.lifespan_context(router.app):
            response = await router.proxy.forward(request({"stream": True}), "/v1/completions")
            assert response.status_code == 200
            stream = "".join([part async for part in response.body_iterator])
            errors = [json.loads(line[6:])["error"] for line in stream.splitlines()
                      if line.startswith("data: {") and '"error"' in line]
            assert len(errors) == 1
            assert errors[0]["code"] == "pd_protocol_error" and errors[0]["source"] == "router"
            assert stream.count("data: [DONE]") == 1
            assert router.proxy.requests == router.proxy.handoffs == 0
    asyncio.run(run())


@pytest.mark.parametrize("flag,value", [("--session-ttl", "600"), ("--affinity-slack", "8"),
    ("--launch-timeout", "600"), ("--drain-timeout", "300"), ("--initial-replicas", "0"),
    ("--max-replicas", "1"), ("--state-file", "state.json")])
def test_explicit_ordinary_only_cli_options_fail_for_pd(tmp_path, monkeypatch, capsys, flag, value):
    from pypto_serving.router.app import main
    path = tmp_path / "replicas.json"
    path.write_text(json.dumps({"replicas": [
        {"host": "p", "port": 8000, "role": "prefill"},
        {"host": "d", "port": 8000, "role": "decode"}]}))
    monkeypatch.delenv("PYPTO_ROUTER_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: pytest.fail("must reject before serving"))
    assert main(["--replicas", str(path), flag, value]) == 1
    assert flag in capsys.readouterr().out
