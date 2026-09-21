# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Launching, draining and stopping replicas, with the transport faked out.

Nothing here spawns a process or opens a socket: the transport seam stands in
for whatever would ssh out and start a server, so the slot bookkeeping and the
lifecycle can be tested at speed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from pypto_serving.router.config import HostSpec, ReplicaSpec, RouterConfig
from pypto_serving.router.fleet import FleetManager
from pypto_serving.router.launcher import (
    HostPool,
    LaunchError,
    PoolExhausted,
    SshTransport,
)
from pypto_serving.router.routing import ReplicaRegistry, SessionDirectory


class _FakeTransport:
    """Records what would have been run, and can be told to fail."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, list[str], dict]] = []
        self.detached: list[bool] = []
        self.fail = fail

    async def run(self, host, argv, env, *, detach=False, log_path=None):
        self.calls.append((host.name, list(argv), dict(env)))
        self.detached.append(detach)
        if self.fail is not None:
            raise self.fail

    def commands(self) -> list[str]:
        return [" ".join(argv) for _, argv, _ in self.calls]


def _hosts(local_devices=(0,), remote_devices=(0, 1)) -> tuple[HostSpec, ...]:
    return (
        HostSpec(name="local", devices=tuple(local_devices), port_base=8001,
                 model="/models/Qwen3-14B", served_model_name="qwen"),
        HostSpec(name="node02", ssh="terra@10.0.0.2", identity_file="/keys/id_ed25519",
                 devices=tuple(remote_devices), port_base=8001,
                 model="/weights/Qwen3-14B", served_model_name="qwen",
                 env={"PYPTO_PROG_BUILD_DIR": "/tmp/build_d{device}"},
                 launch_wrapper=("task-submit", "--device", "{device}", "--run")),
    )


def _fleet(*, initial=0, max_replicas=None, transport=None, state_path=None,
           hosts=None, replicas=(), drain_timeout=5.0, launch_timeout=5.0):
    config = RouterConfig(
        replicas=replicas,
        hosts=_hosts() if hosts is None else hosts,
        initial_replicas=initial,
        max_replicas=max_replicas,
        drain_timeout_seconds=drain_timeout,
        launch_timeout_seconds=launch_timeout,
        health_interval_seconds=0.01,
    )
    sessions = SessionDirectory(config.session_ttl_seconds)
    registry = ReplicaRegistry(config, sessions)
    fake = transport if transport is not None else _FakeTransport()
    manager = FleetManager(
        config, registry, transport_factory=lambda _host: fake, state_path=state_path,
    )
    return manager, registry, fake


# --- slot bookkeeping ---

def test_slots_are_ordered_local_first_then_config_order():
    pool = HostPool(RouterConfig(hosts=_hosts(local_devices=(3,), remote_devices=(0, 1))))
    assert [slot.name for slot in pool.free_slots()] == [
        "local-d3", "node02-d0", "node02-d1",
    ]


def test_a_slot_is_never_allocated_twice():
    """The whole point: two callers must not be handed the same card."""
    pool = HostPool(RouterConfig(hosts=_hosts()))
    taken = [pool.allocate().name for _ in range(pool.ceiling)]
    assert len(set(taken)) == len(taken) == 3
    with pytest.raises(PoolExhausted):
        pool.allocate()


def test_releasing_a_slot_makes_it_reusable():
    pool = HostPool(RouterConfig(hosts=_hosts()))
    first = pool.allocate()
    pool.release(first.name)
    assert pool.allocate().name == first.name


def test_max_replicas_lowers_the_ceiling_below_the_declared_pool():
    pool = HostPool(RouterConfig(hosts=_hosts(), max_replicas=2))
    assert pool.ceiling == 2
    pool.allocate()
    pool.allocate()
    with pytest.raises(PoolExhausted):
        pool.allocate()


def test_ports_are_derived_from_the_device_so_they_cannot_collide():
    pool = HostPool(RouterConfig(hosts=_hosts(remote_devices=(0, 1, 2))))
    ports = {slot.name: slot.port for slot in pool.free_slots()}
    assert ports == {"local-d0": 8001, "node02-d0": 8001, "node02-d1": 8002, "node02-d2": 8003}


# --- the command that gets run ---

def test_the_serve_command_carries_what_a_remote_launch_needs():
    pool = HostPool(RouterConfig(hosts=_hosts()))
    slot = next(s for s in pool.free_slots() if s.name == "node02-d1")
    argv = pool.serve_argv(slot)

    assert argv[:3] == ["python3", "-m", "pypto_serving.cli"]
    assert "--model" in argv and "/weights/Qwen3-14B" in argv
    assert argv[argv.index("--device") + 1] == "1"
    assert argv[argv.index("--port") + 1] == "8002"
    # Binds on all interfaces or the router on another host cannot reach it.
    assert argv[argv.index("--host") + 1] == "0.0.0.0"
    # Without this the startup logs go to /dev/null and a failed load is mute.
    assert "--show-startup-logs" in argv


def test_the_wrapper_receives_the_serve_command_as_one_argument():
    pool = HostPool(RouterConfig(hosts=_hosts()))
    slot = next(s for s in pool.free_slots() if s.name == "node02-d0")
    argv = pool.launch_argv(slot)
    assert argv[:4] == ["task-submit", "--device", "0", "--run"]
    assert len(argv) == 5, "the wrapper takes the command as a single argument"
    assert "pypto_serving.cli" in argv[4]


def test_a_host_without_a_wrapper_runs_the_command_bare():
    """Deployments with no broker are why the wrapper is configuration."""
    hosts = (HostSpec(name="plain", ssh="u@h", devices=(0,), model="/m"),)
    pool = HostPool(RouterConfig(hosts=hosts))
    assert pool.launch_argv(pool.free_slots()[0])[:3] == ["python3", "-m", "pypto_serving.cli"]


def test_env_placeholders_expand_per_device():
    pool = HostPool(RouterConfig(hosts=_hosts(remote_devices=(0, 1))))
    slot = next(s for s in pool.free_slots() if s.name == "node02-d1")
    assert pool.launch_env(slot) == {"PYPTO_PROG_BUILD_DIR": "/tmp/build_d1"}


def test_ssh_argv_is_batch_mode_and_uses_the_identity_file():
    host = _hosts()[1]
    argv = SshTransport().ssh_argv(host)
    # BatchMode: a missing key must fail, not sit on a password prompt.
    assert "BatchMode=yes" in argv
    assert argv[argv.index("-i") + 1] == "/keys/id_ed25519"
    assert argv[-1] == "terra@10.0.0.2"


# --- launching ---

def test_a_launched_replica_is_registered_but_not_yet_routable():
    manager, registry, transport = _fleet()

    spec = asyncio.run(manager.launch_one())
    state = registry.state(spec.name)

    assert state is not None
    assert state.ready is False, "it cannot serve until its model has loaded"
    assert state.owned is True
    assert registry.ready_count() == 0
    assert len(transport.calls) == 1
    # Detached: waiting on a launch would block the router for a model load.
    assert transport.detached == [True]


def test_initial_replicas_fill_local_first_then_spill_to_the_remote_host():
    manager, registry, _ = _fleet(initial=3)
    asyncio.run(manager.launch_initial())
    assert [state.name for state in registry.states] == [
        "local-d0", "node02-d0", "node02-d1",
    ]


def test_growing_past_the_ceiling_is_refused_and_leaves_the_pool_intact():
    manager, registry, _ = _fleet(initial=3)
    asyncio.run(manager.launch_initial())

    with pytest.raises(PoolExhausted):
        asyncio.run(manager.launch_one())
    assert len(registry.states) == 3


def test_a_failed_launch_returns_its_slot():
    """A transport error must not consume a device."""
    manager, registry, _ = _fleet(transport=_FakeTransport(fail=LaunchError("ssh: refused")))

    with pytest.raises(LaunchError):
        asyncio.run(manager.launch_one())
    assert manager.pool.in_use == 0
    assert registry.states == ()


def test_a_replica_that_never_becomes_ready_is_stopped_and_its_slot_freed():
    manager, registry, transport = _fleet(launch_timeout=0.05)

    async def check():
        await manager.launch_one()
        # The watcher runs on the event loop; give it past the timeout.
        await asyncio.sleep(0.4)

    asyncio.run(check())
    assert registry.states == (), "a launch that never serves must not linger"
    assert manager.pool.in_use == 0
    assert any("pkill" in command for command in transport.commands())


def test_a_replica_that_reports_ready_is_kept():
    manager, registry, _ = _fleet(launch_timeout=0.4)

    async def check():
        spec = await manager.launch_one()
        registry.set_ready(spec.name, True)   # what the health poller does
        await asyncio.sleep(0.3)
        return spec

    spec = asyncio.run(check())
    assert registry.state(spec.name) is not None
    assert manager.pool.in_use == 1


# --- draining and stopping ---

def test_stopping_drains_first_then_stops():
    manager, registry, transport = _fleet()

    async def check():
        spec = await manager.launch_one()
        registry.set_ready(spec.name, True)
        registry.acquire(spec.name)          # one request in flight

        stopping = asyncio.create_task(manager.stop(spec.name))
        await asyncio.sleep(0.1)
        state = registry.state(spec.name)
        # Out of rotation, but not gone: it is still finishing its work.
        assert state is not None and state.draining is True
        assert registry.ready_count() == 0
        assert not stopping.done()

        registry.release(spec.name)
        await stopping
        return spec

    spec = asyncio.run(check())
    assert registry.state(spec.name) is None
    assert manager.pool.in_use == 0
    assert any("pkill" in command for command in transport.commands())


def test_draining_gives_up_after_the_timeout_rather_than_holding_a_device():
    manager, registry, _ = _fleet(drain_timeout=0.1)

    async def check():
        spec = await manager.launch_one()
        registry.set_ready(spec.name, True)
        registry.acquire(spec.name)          # never released
        await manager.stop(spec.name)
        return spec

    spec = asyncio.run(check())
    assert registry.state(spec.name) is None
    assert manager.pool.in_use == 0


def test_draining_releases_the_pinned_sessions():
    manager, registry, _ = _fleet()

    async def check():
        spec = await manager.launch_one()
        registry.set_ready(spec.name, True)
        registry.select("s1")                 # pins s1 to the only replica
        assert manager._registry._sessions.lookup("s1") == spec.name
        await manager.stop(spec.name)
        return spec

    spec = asyncio.run(check())
    # The pin is gone, so the next turn routes somewhere live instead of
    # waiting out the TTL pointing at something that no longer exists.
    assert manager._registry._sessions.lookup("s1") is None


def test_stop_all_stops_every_owned_replica():
    manager, registry, _ = _fleet(initial=2)

    async def check():
        await manager.launch_initial()
        await manager.stop_all()

    asyncio.run(check())
    assert registry.states == ()
    assert manager.pool.in_use == 0


def test_stopping_an_unknown_replica_is_a_no_op():
    manager, _, _ = _fleet()
    assert asyncio.run(manager.stop("nope")) is False


# --- crash recovery ---

def test_state_is_recorded_so_a_restart_can_find_the_replicas(tmp_path):
    state_path = tmp_path / "fleet.json"
    manager, _, _ = _fleet(initial=2, state_path=state_path)
    asyncio.run(manager.launch_initial())

    recorded = json.loads(state_path.read_text())
    assert {entry["slot"] for entry in recorded["replicas"]} == {"local-d0", "node02-d0"}
    assert recorded["replicas"][0]["port"] == 8001


def test_a_live_replica_from_a_previous_run_is_adopted(tmp_path):
    state_path = tmp_path / "fleet.json"
    first, _, _ = _fleet(initial=1, state_path=state_path)
    asyncio.run(first.launch_initial())

    # A crash: no shutdown hook ran, so the replica is still serving.
    restarted, registry, _ = _fleet(state_path=state_path)

    async def always_up(_spec: ReplicaSpec) -> bool:
        return True

    adopted = asyncio.run(restarted.adopt_previous(always_up))
    assert adopted == ["local-d0"]
    state = registry.state("local-d0")
    assert state is not None and state.ready and state.owned
    # Its device counts against the ceiling again.
    assert restarted.pool.in_use == 1


def test_a_dead_replica_from_a_previous_run_is_not_adopted(tmp_path):
    state_path = tmp_path / "fleet.json"
    first, _, _ = _fleet(initial=1, state_path=state_path)
    asyncio.run(first.launch_initial())

    restarted, registry, _ = _fleet(state_path=state_path)

    async def never_up(_spec: ReplicaSpec) -> bool:
        return False

    assert asyncio.run(restarted.adopt_previous(never_up)) == []
    assert registry.states == ()
    assert restarted.pool.in_use == 0


def test_adoption_without_a_state_file_is_a_no_op(tmp_path):
    manager, _, _ = _fleet(state_path=tmp_path / "missing.json")

    async def probe(_spec):
        return True

    assert asyncio.run(manager.adopt_previous(probe)) == []


def test_capacity_reports_what_is_left():
    manager, _, _ = _fleet(initial=1)
    asyncio.run(manager.launch_initial())
    capacity = manager.capacity()
    assert capacity["owned"] == 1
    assert capacity["ceiling"] == 3
    assert capacity["free_slots"] == ["node02-d0", "node02-d1"]


# --- the admin API ---

class _AdminRequest:
    """Only the two attributes the admin handlers touch."""

    def __init__(self, headers: dict | None = None) -> None:
        from starlette.datastructures import Headers
        self.headers = Headers(
            raw=[(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        )


def _router_with_pool(**kwargs):
    from pypto_serving.router.app import ServingRouter

    config = RouterConfig(
        hosts=_hosts(), initial_replicas=0, health_interval_seconds=0.01, **kwargs
    )
    transport = _FakeTransport()

    class _Client:
        async def get(self, url, timeout=None):
            raise AssertionError("no upstream in this test")

        async def aclose(self):
            pass

    return ServingRouter(
        config,
        client_factory=lambda _c: _Client(),
        transport_factory=lambda _host: transport,
    ), transport


def test_post_replicas_launches_and_answers_202():
    router, transport = _router_with_pool()

    response = asyncio.run(router._add_replica(_AdminRequest()))
    payload = json.loads(response.body)

    assert response.status_code == 202, "the replica is starting, not ready"
    assert payload["name"] == "local-d0"
    assert payload["ready"] is False
    assert len(transport.calls) == 1


def test_post_replicas_answers_409_when_the_pool_is_full():
    router, _ = _router_with_pool()

    async def check():
        for _ in range(router.config.pool_ceiling):
            await router._add_replica(_AdminRequest())
        return await router._add_replica(_AdminRequest())

    response = asyncio.run(check())
    assert response.status_code == 409
    assert json.loads(response.body)["capacity"]["free_slots"] == []


def test_delete_replicas_refuses_a_replica_it_did_not_launch():
    """Only kill what we launched."""
    from pypto_serving.router.app import ServingRouter

    class _Client:
        async def aclose(self):
            pass

    config = RouterConfig(
        replicas=(ReplicaSpec(name="external", host="h", port=9000),),
        hosts=_hosts(), initial_replicas=0,
    )
    router = ServingRouter(
        config, client_factory=lambda _c: _Client(),
        transport_factory=lambda _host: _FakeTransport(),
    )

    response = asyncio.run(router._remove_replica("external", _AdminRequest()))
    assert response.status_code == 409
    assert "not launched by this router" in json.loads(response.body)["message"]
    assert router.registry.state("external") is not None


def test_delete_replicas_is_404_for_an_unknown_name():
    router, _ = _router_with_pool()
    response = asyncio.run(router._remove_replica("ghost", _AdminRequest()))
    assert response.status_code == 404


def test_delete_replicas_stops_one_it_launched():
    router, transport = _router_with_pool()

    async def check():
        created = json.loads((await router._add_replica(_AdminRequest())).body)
        router.registry.set_ready(created["name"], True)
        return await router._remove_replica(created["name"], _AdminRequest())

    response = asyncio.run(check())
    assert response.status_code == 200
    assert json.loads(response.body)["stopped"] is True
    assert router.registry.states == ()
    assert any("pkill" in command for command in transport.commands())


def test_admin_routes_require_the_token_when_one_is_configured():
    from pypto_serving.router.app import ServingRouter

    class _Client:
        async def aclose(self):
            pass

    router = ServingRouter(
        RouterConfig(hosts=_hosts(), initial_replicas=0),
        client_factory=lambda _c: _Client(),
        transport_factory=lambda _host: _FakeTransport(),
        admin_token="s3cret",
    )

    for request, expected in (
        (_AdminRequest(), 401),
        (_AdminRequest({"authorization": "Bearer wrong"}), 401),
        (_AdminRequest({"authorization": "s3cret"}), 401),
        (_AdminRequest({"authorization": "Bearer s3cret"}), 202),
    ):
        response = asyncio.run(router._add_replica(request))
        assert response.status_code == expected, request.headers.get("authorization")


def test_get_replicas_reports_the_fleet_and_what_is_left():
    router, _ = _router_with_pool()

    async def check():
        await router._add_replica(_AdminRequest())
        return await router._list_replicas(_AdminRequest())

    payload = json.loads(asyncio.run(check()).body)
    assert [r["name"] for r in payload["replicas"]] == ["local-d0"]
    assert payload["replicas"][0]["owned"] is True
    assert payload["capacity"] == {
        "owned": 1, "ceiling": 3, "free_slots": ["node02-d0", "node02-d1"],
    }
