# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Launching a replica for real, over a real socket, with no NPU.

The other launcher tests fake the transport. This one does not: the process is
actually spawned, the router actually probes it over HTTP, and the request is
actually proxied. What stands in for the serving process is the stub replica --
a real one needs a device and two minutes -- but everything between the router
and that process is the production path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("httpx", reason="the router needs httpx to reach a replica")
pytest.importorskip("uvicorn", reason="the stub replica needs uvicorn")

from pypto_serving.router.app import ServingRouter  # noqa: E402
from pypto_serving.router.config import HostSpec, RouterConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
STUB = Path(__file__).with_name("stub_replica.py")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHAT = {"messages": [{"role": "user", "content": "hello"}]}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _AdminRequest:
    """Only the attribute the admin handlers touch."""

    def __init__(self) -> None:
        from starlette.datastructures import Headers
        self.headers = Headers(raw=[])


class _ProxyRequest:
    """Only what ReplicaProxy.forward touches."""

    def __init__(self, body: bytes) -> None:
        from starlette.datastructures import Headers
        self.headers = Headers(raw=[(b"content-type", b"application/json")])
        self._body = body

    async def body(self) -> bytes:
        return self._body


def _stub_host(port_base: int, log_dir: Path, devices=(0,)) -> HostSpec:
    """A host whose "serving process" is the stub replica.

    The launch wrapper is what makes this possible without a test hook in the
    launcher: the same mechanism that wraps a real launch in a broker command
    puts the stub in front of it here.
    """
    return HostSpec(
        name="local",
        devices=tuple(devices),
        port_base=port_base,
        model="/unused",
        launch_wrapper=(
            sys.executable, str(STUB),
            "--port", "{port}", "--name", "stub{device}", "--host", "127.0.0.1", "--",
        ),
        # Same self-match trap as the production default: see stop_argv.
        stop_command=("pkill", "-f", "stub_replica[.]py --port {port}( |$)"),
        workdir=str(REPO_ROOT),
        log_dir=str(log_dir),
    )


@contextlib.asynccontextmanager
async def _running_router(tmp_path, devices=(0, 1)):
    """A router with a real pool, started and torn down on one event loop.

    One loop matters: the upstream httpx client binds to the loop it was used
    on, so closing it from a second ``asyncio.run`` raises.
    """
    config = RouterConfig(
        hosts=(_stub_host(_free_port(), tmp_path / "logs", devices),),
        initial_replicas=0,
        health_interval_seconds=0.2,
        launch_timeout_seconds=20.0,
        drain_timeout_seconds=3.0,
    )
    router = ServingRouter(config, state_path=tmp_path / "fleet.json")
    await router.health.start()
    try:
        yield router
    finally:
        await router.fleet.stop_all()
        await router.health.stop()
        close = getattr(router.client, "aclose", None)
        if close is not None:
            await close()


async def _until(predicate, timeout: float = 20.0) -> bool:
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        await asyncio.sleep(0.2)
        waited += 0.2
    return False


def _post(url: str, payload: dict):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with OPENER.open(request, timeout=30) as response:
        return response.status, response.read()


def _reachable(base_url: str) -> bool:
    try:
        with OPENER.open(f"{base_url}/health", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def test_a_launched_replica_becomes_routable_and_then_serves(tmp_path):
    """The whole loop: launch, become ready, serve, stop."""

    async def check():
        async with _running_router(tmp_path) as router:
            response = await router._add_replica(_AdminRequest())
            assert response.status_code == 202
            name = json.loads(response.body)["name"]

            # Registered, but it must not take traffic before its model is up.
            assert router.registry.state(name).ready is False
            assert router.registry.ready_count() == 0

            assert await _until(lambda: router.registry.state(name).ready), \
                "the health poller never promoted the launched replica"
            assert router.registry.ready_count() == 1

            url = router.registry.state(name).spec.base_url
            status, body = await asyncio.to_thread(_post, f"{url}/v1/chat/completions", CHAT)
            assert status == 200
            assert json.loads(body)["served_by"] == "stub0"

            removed = await router._remove_replica(name, _AdminRequest())
            assert removed.status_code == 200
            assert router.registry.states == ()
            assert await _until(lambda: not _reachable(url), timeout=10.0), \
                "the replica was reported stopped but is still answering"

    asyncio.run(check())


def test_the_pool_bounds_growth_and_a_stop_gives_the_slot_back(tmp_path):
    async def check():
        async with _running_router(tmp_path, devices=(0, 1)) as router:
            first = json.loads((await router._add_replica(_AdminRequest())).body)["name"]
            await router._add_replica(_AdminRequest())

            full = await router._add_replica(_AdminRequest())
            assert full.status_code == 409, "two devices declared, so a third is too many"
            assert json.loads(full.body)["capacity"]["free_slots"] == []

            await router._remove_replica(first, _AdminRequest())
            again = await router._add_replica(_AdminRequest())
            assert again.status_code == 202, "stopping one must return its slot"

    asyncio.run(check())


def test_a_launched_replica_takes_a_request_through_the_router(tmp_path):
    """Through the router's own proxy, not straight at the replica."""

    async def check():
        async with _running_router(tmp_path) as router:
            name = json.loads((await router._add_replica(_AdminRequest())).body)["name"]
            assert await _until(lambda: router.registry.state(name).ready)

            response = await router.proxy.forward(
                _ProxyRequest(json.dumps(CHAT).encode()), "/v1/chat/completions"
            )
            chunks = [chunk async for chunk in response.body_iterator]

            assert response.status_code == 200
            assert json.loads(b"".join(chunks))["served_by"] == "stub0"
            assert router.registry.state(name).routed == 1
            # Released once the relay finished.
            assert router.registry.state(name).outstanding == 0

    asyncio.run(check())
