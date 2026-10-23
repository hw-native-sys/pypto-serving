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
from pypto_serving.router.fleet import (
    STOP_GRACE_SECONDS,
    STOP_UNCONFIRMED,
    UNREACHABLE_AT_STARTUP,
    FleetManager,
)
from pypto_serving.router.launcher import (
    HostPool,
    LaunchError,
    PoolExhausted,
    SshTransport,
    _run_and_wait,
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
        HostSpec(name="node02", ssh="user@node02", identity_file="/keys/id_ed25519",
                 devices=tuple(remote_devices), port_base=8001,
                 model="/weights/Qwen3-14B", served_model_name="qwen",
                 env={"PYPTO_PROG_BUILD_DIR": "/tmp/build_d{device}"},
                 launch_wrapper=("task-submit", "--device", "{device}", "--run")),
    )


def _fleet(*, initial=0, max_replicas=None, transport=None, state_path=None,
           hosts=None, replicas=(), drain_timeout=5.0, launch_timeout=5.0, probe=None,
           alive_probe=None, stop_grace=0.3, replace_after=None, restart_backoff=0.01,
           replace_dead=True):
    config = RouterConfig(
        replicas=replicas,
        hosts=_hosts() if hosts is None else hosts,
        initial_replicas=initial,
        max_replicas=max_replicas,
        drain_timeout_seconds=drain_timeout,
        launch_timeout_seconds=launch_timeout,
        health_interval_seconds=0.01,
        replace_dead=replace_dead,
        **({} if replace_after is None else {"replace_after_seconds": replace_after}),
    )
    sessions = SessionDirectory(config.session_ttl_seconds)
    registry = ReplicaRegistry(config, sessions)
    fake = transport if transport is not None else _FakeTransport()
    manager = FleetManager(
        config, registry, transport_factory=lambda _host: fake, state_path=state_path,
        probe=probe, alive_probe=alive_probe,
        # Shrunk from 30s: these tests assert what happens at both ends of the
        # grace period, and waiting it out would dominate the suite.
        stop_grace=stop_grace,
        # Likewise the 10s a replacement waits for the dead process to let go.
        restart_backoff=restart_backoff,
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
    assert argv[argv.index("--devices") + 1] == "1"
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
    assert argv[-1] == "user@node02"


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


def test_a_replica_from_a_previous_run_that_does_not_answer_is_not_adopted(tmp_path):
    """Not adopted, but not forgotten either: its slot stays reserved.

    Silence does not prove the process exited -- it may still be loading, or the
    probe may have hit a blip -- and launching onto those cards is the one
    outcome that cannot be undone.
    """
    state_path = tmp_path / "fleet.json"
    first, _, _ = _fleet(initial=1, state_path=state_path)
    asyncio.run(first.launch_initial())

    restarted, registry, _ = _fleet(state_path=state_path)

    async def never_up(_spec: ReplicaSpec) -> bool:
        return False

    assert asyncio.run(restarted.adopt_previous(never_up)) == []
    assert registry.states == ()
    assert restarted.pool.in_use == 1
    assert restarted.capacity()["unconfirmed"] == {"local-d0": UNREACHABLE_AT_STARTUP}


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
        "unconfirmed": {}, "replacing": [], "abandoned": [],
    }


# --- a stopped slot is not free until the replica is really gone ---

def test_a_slot_is_held_until_the_stopped_replica_stops_answering():
    """Freeing on the signal alone would let the next launch race the port."""
    answers = {"alive": True}

    async def probe(_spec):
        return answers["alive"]

    config = RouterConfig(hosts=_hosts(), initial_replicas=0, health_interval_seconds=0.01)
    sessions = SessionDirectory(config.session_ttl_seconds)
    registry = ReplicaRegistry(config, sessions)
    manager = FleetManager(
        config, registry, transport_factory=lambda _h: _FakeTransport(), probe=probe,
    )

    async def check():
        spec = await manager.launch_one()
        registry.set_ready(spec.name, True)

        stopping = asyncio.create_task(manager.stop(spec.name))
        await asyncio.sleep(0.2)
        # Signalled, but still answering: the slot must not be handed out yet.
        assert not stopping.done()
        assert manager.pool.in_use == 1

        answers["alive"] = False
        await stopping
        assert manager.pool.in_use == 0

    asyncio.run(check())


def test_the_stop_grace_period_is_bounded_and_modest():
    assert 0 < STOP_GRACE_SECONDS <= 60


def test_a_replica_that_will_not_die_does_not_wedge_the_slot_forever():
    async def always_alive(_spec):
        return True

    config = RouterConfig(hosts=_hosts(), initial_replicas=0, health_interval_seconds=0.01)
    sessions = SessionDirectory(config.session_ttl_seconds)
    registry = ReplicaRegistry(config, sessions)
    manager = FleetManager(
        config, registry, transport_factory=lambda _h: _FakeTransport(), probe=always_alive,
    )

    async def check():
        spec = await manager.launch_one()
        registry.set_ready(spec.name, True)
        # Bounded: a replica that ignores its stop must not hold a device for
        # the life of the router, so the wait gives up and reuses it anyway.
        started = asyncio.get_running_loop().time()
        await manager._wait_gone(spec, timeout=0.3)
        assert asyncio.get_running_loop().time() - started < 3.0, "the wait is not bounded"

    asyncio.run(check())


# --- a replica may need more than one card ---

def _wide_host(devices=(0, 1, 2, 3, 4, 5, 6, 7), per_replica=8) -> HostSpec:
    """A host sized for a model whose replica spans several cards."""
    return HostSpec(
        name="dsv4", ssh="u@h", devices=tuple(devices),
        devices_per_replica=per_replica, port_base=8001,
        model="/weights/dsv4-flash-w8a8", served_model_name="dsv4",
        serve_args=("--dp", "8", "--ep", "8", "--tp", "1", "--block-size", "128"),
    )


def test_eight_cards_at_eight_per_replica_is_one_slot():
    """The DeepSeek V4 shape: a whole node is one replica, not eight."""
    host = _wide_host()
    assert host.replica_capacity == 1
    assert host.device_groups() == ((0, 1, 2, 3, 4, 5, 6, 7),)

    pool = HostPool(RouterConfig(hosts=(host,)))
    assert pool.ceiling == 1
    slot = pool.allocate()
    assert slot.devices == (0, 1, 2, 3, 4, 5, 6, 7)
    with pytest.raises(PoolExhausted):
        pool.allocate()


def test_sixteen_cards_at_eight_per_replica_is_two_slots():
    host = _wide_host(devices=tuple(range(16)))
    assert host.replica_capacity == 2
    assert host.device_groups() == (tuple(range(8)), tuple(range(8, 16)))

    pool = HostPool(RouterConfig(hosts=(host,)))
    names = [pool.allocate().name for _ in range(2)]
    # Named and ported by the group's first device, so the two never collide.
    assert names == ["dsv4-d0", "dsv4-d8"]
    assert [host.port_for(group) for group in host.device_groups()] == [8001, 8009]


def test_the_whole_group_is_passed_to_serving():
    pool = HostPool(RouterConfig(hosts=(_wide_host(),)))
    argv = pool.serve_argv(pool.free_slots()[0])
    # One --devices with the full group; serving validates it against what the
    # model's kernels require and refuses a wrong count at startup.
    assert argv[argv.index("--devices") + 1] == "0,1,2,3,4,5,6,7"
    assert "--device" not in argv
    assert argv[argv.index("--port") + 1] == "8001"
    for flag in ("--dp", "--ep", "--tp"):
        assert flag in argv, f"{flag} from serve_args must survive"


def test_templates_can_expand_the_group_or_its_first_device():
    host = HostSpec(
        name="dsv4", ssh="u@h", devices=tuple(range(8)), devices_per_replica=8,
        model="/m",
        launch_wrapper=("task-submit", "--device", "{devices}", "--run"),
        env={"VIS": "{devices}", "BUILD": "/b/{device}"},
    )
    pool = HostPool(RouterConfig(hosts=(host,)))
    slot = pool.free_slots()[0]
    assert pool.launch_argv(slot)[:4] == ["task-submit", "--device", "0,1,2,3,4,5,6,7", "--run"]
    assert pool.launch_env(slot) == {"VIS": "0,1,2,3,4,5,6,7", "BUILD": "/b/0"}


def test_a_device_count_that_is_not_a_whole_number_of_replicas_is_refused():
    """A leftover card is a typo or a misread model, not a partial replica."""
    with pytest.raises(ValueError, match="not a multiple of devices_per_replica=8"):
        HostSpec(name="h", devices=tuple(range(6)), devices_per_replica=8, model="/m")


def test_devices_per_replica_must_be_positive():
    with pytest.raises(ValueError, match="devices_per_replica must be positive"):
        HostSpec(name="h", devices=(0,), devices_per_replica=0, model="/m")


def test_the_pool_ceiling_mixes_hosts_of_different_widths():
    """One Qwen card next to an eight-card DeepSeek node is 1 + 1 slots."""
    qwen = HostSpec(name="qwen", devices=(4,), model="/q")
    config = RouterConfig(hosts=(qwen, _wide_host()))
    assert config.pool_ceiling == 2


def test_a_wide_replica_is_launched_and_stopped_as_one_unit():
    config = RouterConfig(hosts=(_wide_host(),), initial_replicas=0)
    sessions = SessionDirectory(config.session_ttl_seconds)
    registry = ReplicaRegistry(config, sessions)
    transport = _FakeTransport()
    manager = FleetManager(config, registry, transport_factory=lambda _h: transport)

    async def check():
        spec = await manager.launch_one()
        registry.set_ready(spec.name, True)
        assert manager.pool.in_use == 1
        assert "0,1,2,3,4,5,6,7" in transport.commands()[0]
        await manager.stop(spec.name)
        assert manager.pool.in_use == 0, "the whole group comes back, not one card"

    asyncio.run(check())


# --- configuration validation ------------------------------------------------
#
# These cover the review findings on RouterConfig and the JSON parsers: each one
# is a configuration that used to be accepted and then failed later, at a point
# where the failure was either silent or unrecognisable.


@pytest.mark.parametrize("field", ["request_timeout_seconds", "connect_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1.0])
def test_non_positive_transport_timeouts_are_refused(field, value):
    """httpx builds a client happily with these, so nothing else rejects them."""
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        RouterConfig(replicas=(ReplicaSpec(name="r", host="h", port=8001),), **{field: value})


def test_a_static_replica_may_not_take_a_generated_slot_name():
    """Otherwise the launch succeeds, the registry rejects it, and the slot leaks."""
    host = HostSpec(name="node", devices=(0, 1), model="/m")
    with pytest.raises(ValueError, match="collide with the slot names"):
        RouterConfig(
            replicas=(ReplicaSpec(name="node-d0", host="127.0.0.1", port=9000),),
            hosts=(host,),
        )


def test_a_static_replica_named_after_an_unused_device_is_fine():
    """The check is about real collisions, not about the naming scheme."""
    host = HostSpec(name="node", devices=(0, 1), model="/m")
    config = RouterConfig(
        replicas=(ReplicaSpec(name="node-d7", host="127.0.0.1", port=9000),),
        hosts=(host,),
    )
    assert config.pool_ceiling == 2


def test_grouped_devices_only_generate_slot_names_for_group_leaders():
    """With devices_per_replica=2 only d0 and d2 name slots, so d1 is free to use."""
    host = HostSpec(name="node", devices=(0, 1, 2, 3), devices_per_replica=2, model="/m")
    config = RouterConfig(
        replicas=(ReplicaSpec(name="node-d1", host="127.0.0.1", port=9000),), hosts=(host,),
    )
    assert config.pool_ceiling == 2
    with pytest.raises(ValueError, match="collide with the slot names"):
        RouterConfig(
            replicas=(ReplicaSpec(name="node-d2", host="127.0.0.1", port=9000),), hosts=(host,),
        )


@pytest.mark.parametrize("name", [["a"], 3, "", "   ", {}])
def test_a_non_string_name_is_a_configuration_error(tmp_path, name):
    """It used to reach the uniqueness check and raise TypeError, which the CLI
    does not catch -- an invalid file crashed instead of reporting itself."""
    from pypto_serving.router.config import load_fleet_file

    path = tmp_path / "fleet.json"
    path.write_text(json.dumps({"replicas": [{"name": name, "host": "h", "port": 8001}]}))
    with pytest.raises(ValueError, match="invalid name"):
        load_fleet_file(path)


def test_an_absent_name_still_gets_its_generated_default(tmp_path):
    from pypto_serving.router.config import load_fleet_file

    path = tmp_path / "fleet.json"
    path.write_text(json.dumps({"replicas": [{"host": "h", "port": 8001}]}))
    replicas, _hosts = load_fleet_file(path)
    assert replicas[0].name == "h:8001"


def test_a_host_name_is_validated_the_same_way(tmp_path):
    from pypto_serving.router.config import load_fleet_file

    path = tmp_path / "fleet.json"
    path.write_text(json.dumps({"hosts": [{"name": [], "devices": [0], "model": "/m"}]}))
    with pytest.raises(ValueError, match="invalid name"):
        load_fleet_file(path)


def test_the_remote_log_directory_is_quoted(tmp_path):
    """A log_dir with a space would otherwise split into two mkdir arguments."""
    host = HostSpec(
        name="node", ssh="user@host", devices=(0,), model="/m",
        log_dir="/var/log/pypto router",
    )
    pool = HostPool(RouterConfig(hosts=(host,)))
    slot = pool.allocate()
    transport = SshTransport()
    payload = None

    async def capture(argv):
        nonlocal payload
        payload = argv[-1]

    asyncio.run(_payload_of(transport, host, pool, slot, capture))
    assert '"$(dirname ' in payload


async def _payload_of(transport, host, pool, slot, capture):
    import pypto_serving.router.launcher as launcher_module

    original = launcher_module._run_and_wait
    launcher_module._run_and_wait = capture
    try:
        await transport.run(
            host, pool.launch_argv(slot), {}, detach=True, log_path=slot.log_path,
        )
    finally:
        launcher_module._run_and_wait = original


# --- an unconfirmed stop keeps its slot --------------------------------------
#
# Review on #254: a slot used to go back to the pool whether or not the replica
# had actually stopped. A failed stop command, or an endpoint still answering
# after the grace period, would hand those devices to the next launch and
# overlap two instances on the same cards.


def test_a_failed_stop_command_keeps_the_slot_reserved():
    async def scenario():
        transport = _FakeTransport()
        fleet, registry, _fake = _fleet(transport=transport)
        await fleet.launch_one()
        registry.set_ready("local-d0", True)

        transport.fail = LaunchError("pkill: operation not permitted")
        stopped = await fleet.stop("local-d0", drain=False)
        return stopped, fleet.capacity()

    stopped, capacity = asyncio.run(scenario())
    assert stopped is False
    assert "local-d0" not in capacity["free_slots"]
    assert capacity["unconfirmed"] == {"local-d0": STOP_UNCONFIRMED}


def test_an_endpoint_still_answering_keeps_the_slot_reserved():
    """A silent endpoint is weak evidence of exit; a noisy one is conclusive
    evidence of the opposite."""
    async def scenario():
        async def always_answers(_spec):
            return True

        fleet, registry, _fake = _fleet(alive_probe=always_answers)
        await fleet.launch_one()
        registry.set_ready("local-d0", True)
        stopped = await fleet.stop("local-d0", drain=False)
        return stopped, fleet.capacity()

    stopped, capacity = asyncio.run(scenario())
    assert stopped is False
    assert capacity["unconfirmed"] == {"local-d0": STOP_UNCONFIRMED}


def test_a_confirmed_stop_still_frees_the_slot():
    async def scenario():
        async def never_answers(_spec):
            return False

        fleet, registry, _fake = _fleet(alive_probe=never_answers)
        await fleet.launch_one()
        registry.set_ready("local-d0", True)
        stopped = await fleet.stop("local-d0", drain=False)
        return stopped, fleet.capacity()

    stopped, capacity = asyncio.run(scenario())
    assert stopped is True
    assert capacity["unconfirmed"] == {}
    assert "local-d0" in capacity["free_slots"]


# --- the reservation must survive concurrency and restarts -------------------
#
# Second review on #254. One invariant, four ways it was lost: a slot stays
# reserved until the replica that held it is confirmed retired.


def test_concurrent_stops_share_one_retirement():
    """Two DELETEs for the same replica must not retire it twice.

    One retirement each is what lost a slot: the second caller found the
    registry entry already removed, took that for proof of exit, skipped the
    wait and released the devices while the first was still watching the
    endpoint.
    """
    async def scenario():
        answering = {"yes": True}

        async def still_there(_spec):
            return answering["yes"]

        fleet, registry, fake = _fleet(alive_probe=still_there, stop_grace=5.0)
        spec = await fleet.launch_one()
        registry.set_ready(spec.name, True)

        first = asyncio.create_task(fleet.stop(spec.name, drain=False))
        second = asyncio.create_task(fleet.stop(spec.name, drain=False))
        await asyncio.sleep(0.2)
        # Still answering, so neither caller may have freed anything yet.
        held = fleet.capacity()
        answering["yes"] = False
        results = await asyncio.gather(first, second)
        stops = [argv for _host, argv, _env in fake.calls if "pkill" in " ".join(argv)]
        return results, held, fleet.capacity(), fleet.owned_names, len(stops)

    results, held, capacity, owned, stops = asyncio.run(scenario())
    assert held["free_slots"] == ["node02-d0", "node02-d1"], "freed before it was gone"
    assert results == [True, True]
    assert stops == 1, "one retirement, not one per caller"
    assert owned == ()
    assert capacity["unconfirmed"] == {}
    assert "local-d0" in capacity["free_slots"]


def test_a_late_retirement_leaves_the_replacement_alone():
    """Cleanup is keyed on the instance, not on the slot name.

    The name is reused by whatever replica comes next, so a retirement that
    finishes late must not drop the registry entry, or release the devices, of
    the instance that took its place.
    """
    async def scenario():
        fleet, registry, _fake = _fleet()
        first = await fleet.launch_one()
        stale = fleet._owned[first.name]
        assert await fleet.stop(first.name, drain=False)

        replacement = await fleet.launch_one()
        assert replacement.name == first.name, "the pool reuses the slot name"
        current = fleet._owned[replacement.name]

        # The old instance's cleanup fires now, one generation too late.
        late = fleet._finish_retirement(stale, signalled=True, gone=True)
        return late, fleet.owned_names, fleet.pool.in_use, registry.state(first.name), (
            stale.generation, current.generation
        )

    late, owned, in_use, state, generations = asyncio.run(scenario())
    assert late is True, "the old instance really did stop"
    assert owned == ("local-d0",), "but the replacement still owns the slot"
    assert in_use == 1
    assert state is not None and state.owned
    assert generations[0] != generations[1]


def test_a_replica_that_answers_503_is_not_confirmed_stopped():
    """503 is not ready, and not gone either.

    serving answers /health with 503 while its engine is not ready, so a
    replica whose worker died with its HTTP server up says 503 and still holds
    its cards. The readiness probe called that an exit and freed the devices.
    """
    async def scenario():
        async def never_ready(_spec):
            return False

        async def answers_503(_spec):
            return True

        fleet, registry, _fake = _fleet(probe=never_ready, alive_probe=answers_503)
        await fleet.launch_one()
        registry.set_ready("local-d0", True)
        stopped = await fleet.stop("local-d0", drain=False)
        return stopped, fleet.capacity()

    stopped, capacity = asyncio.run(scenario())
    assert stopped is False
    assert capacity["unconfirmed"] == {"local-d0": STOP_UNCONFIRMED}
    assert "local-d0" not in capacity["free_slots"]


def test_the_router_tells_a_503_apart_from_an_exit():
    """The two probes the fleet needs, over the same endpoint."""
    from pypto_serving.router.app import ServingRouter

    class _Response:
        def __init__(self, status_code):
            self.status_code = status_code

    class _Client:
        status: int | None = 503

        async def get(self, _url, timeout=None):
            if self.status is None:
                raise OSError("connection refused")
            return _Response(self.status)

        async def aclose(self):
            pass

    client = _Client()
    router = ServingRouter(
        RouterConfig(hosts=_hosts(), initial_replicas=0),
        client_factory=lambda _c: client,
        transport_factory=lambda _host: _FakeTransport(),
    )
    spec = ReplicaSpec(name="local-d0", host="127.0.0.1", port=8001)

    async def both():
        return await router._probe_replica(spec), await router._endpoint_answers(spec)

    assert asyncio.run(both()) == (False, True), "503: not serving, but still there"
    client.status = 200
    assert asyncio.run(both()) == (True, True)
    client.status = None
    assert asyncio.run(both()) == (False, False), "nothing answered at all"


def test_an_unconfirmed_stop_survives_a_restart(tmp_path):
    """The state file recorded only what was owned.

    So the reservation vanished at exactly the moment it mattered -- a stop that
    could not be confirmed -- and the next run saw free devices that an old
    process may still have held.
    """
    state_path = tmp_path / "fleet.json"

    async def first_run():
        transport = _FakeTransport()
        fleet, registry, _ = _fleet(transport=transport, state_path=state_path)
        await fleet.launch_one()
        registry.set_ready("local-d0", True)
        transport.fail = LaunchError("pkill: operation not permitted")
        return await fleet.stop("local-d0", drain=False)

    assert asyncio.run(first_run()) is False
    recorded = json.loads(state_path.read_text())["replicas"]
    assert [(e["slot"], e["state"]) for e in recorded] == [("local-d0", "unconfirmed")]

    restarted, registry, _ = _fleet(initial=1, state_path=state_path)

    async def second_run():
        async def up(_spec):
            return True

        adopted = await restarted.adopt_previous(up)
        launched = await restarted.launch_initial()
        return adopted, [spec.name for spec in launched]

    adopted, launched = asyncio.run(second_run())
    assert adopted == [], "nothing to adopt: the replica was told to stop"
    assert restarted.capacity()["unconfirmed"] == {"local-d0": STOP_UNCONFIRMED}
    assert "local-d0" not in restarted.capacity()["free_slots"]
    assert launched == ["node02-d0"], "the next launch went somewhere else"
    assert registry.state("local-d0") is None


def test_a_replica_still_loading_keeps_its_slot_across_a_restart(tmp_path):
    """A crash during a model load leaves a replica that answers nothing yet.

    uvicorn binds only once the engine is up, so the probe cannot tell it from a
    replica that is gone. Reserving the slot only when the probe succeeded let
    launch_initial() put a second replica on those same cards.
    """
    state_path = tmp_path / "fleet.json"
    first, _, _ = _fleet(initial=1, state_path=state_path)
    asyncio.run(first.launch_initial())

    restarted, registry, _ = _fleet(initial=1, state_path=state_path)

    async def scenario():
        async def not_yet(_spec):
            return False

        adopted = await restarted.adopt_previous(not_yet)
        launched = await restarted.launch_initial()
        return adopted, [spec.name for spec in launched]

    adopted, launched = asyncio.run(scenario())
    assert adopted == []
    assert registry.state("local-d0") is None
    assert restarted.capacity()["unconfirmed"] == {"local-d0": UNREACHABLE_AT_STARTUP}
    assert launched == ["node02-d0"], "a second replica landed on the loading one's cards"


def test_a_replica_that_answers_but_is_not_ready_is_adopted_as_starting(tmp_path):
    """Still loading, but far enough along to answer: that one is ours.

    Adopting it unroutable costs nothing -- the health poller promotes it when
    its model is up -- and leaving it unadopted would hold its slot for a
    replica that is on its way to serving.
    """
    state_path = tmp_path / "fleet.json"
    first, _, _ = _fleet(initial=1, state_path=state_path)
    asyncio.run(first.launch_initial())

    async def scenario():
        async def not_ready(_spec):
            return False

        async def answering(_spec):
            return True

        restarted, registry, _ = _fleet(state_path=state_path, alive_probe=answering)
        adopted = await restarted.adopt_previous(not_ready)
        return adopted, registry.state("local-d0"), restarted.capacity()

    adopted, state, capacity = asyncio.run(scenario())
    assert adopted == ["local-d0"]
    assert state is not None and state.owned and not state.ready
    assert capacity["owned"] == 1
    assert capacity["unconfirmed"] == {}


# --- regressions: the failures the fake transport could not express ----------


def test_the_default_stop_tolerates_a_replica_that_is_already_gone():
    """pkill exits 1 when nothing matched, which for a stop means success.

    A replica that crashed during its model load leaves nothing to kill, so
    reading that exit as a failed stop is what reserved its devices for the life
    of the router -- the commonest failed launch, permanently losing a card.
    """
    host = HostSpec(name="node", devices=(0,), port_base=8001, model="/m")
    pool = HostPool(RouterConfig(hosts=(host,)))
    slot = pool.allocate()
    argv = pool.stop_argv(slot)
    # The tolerance has to live in the command: _run_and_wait turns any non-zero
    # exit into a LaunchError and never exposes the code to its caller.
    assert argv[:2] == ["sh", "-c"]
    assert "pkill -f" in argv[2]
    assert "[ $? -eq 1 ]" in argv[2]

    async def nothing_matches():
        # The idiom, against a pattern guaranteed to match nothing. Exercising it
        # with the real pattern would risk killing a developer's own replica.
        # The character class is the same self-avoidance the production pattern
        # uses: `pkill -f` reads every command line including this shell's own,
        # which carries the pattern verbatim, so a literal one makes pkill SIGTERM
        # its own parent and the command comes back as exit -15.
        await _run_and_wait(
            ["sh", "-c", "pkill -f no-such-process-xyzz[y] || [ $? -eq 1 ]"]
        )

    asyncio.run(nothing_matches())


def test_a_detached_ssh_launch_changes_directory_inside_a_shell():
    """``nohup cd DIR && serve`` never starts anything.

    nohup execs its first argument and ``cd`` is a shell builtin with no binary,
    so the cd fails, the ``&&`` short-circuits, and the trailing ``&`` still
    makes ssh exit 0 -- a launch reported as successful that leaves no process
    and no log, because the redirect applied to the dead right-hand side.
    """
    host = HostSpec(
        name="node", ssh="user@host", devices=(0,), model="/m",
        workdir="/opt/pypto-serving",
    )
    pool = HostPool(RouterConfig(hosts=(host,)))
    slot = pool.allocate()
    payload = None

    async def capture(argv):
        nonlocal payload
        payload = argv[-1]

    asyncio.run(_payload_of(SshTransport(), host, pool, slot, capture))
    assert "nohup cd " not in payload, "nohup cannot exec a shell builtin"
    assert "nohup sh -c " in payload
    assert "cd /opt/pypto-serving" in payload


def test_a_template_typo_in_env_returns_the_slot_to_the_pool():
    """A stray brace raises KeyError from str.format, not LaunchError.

    Releasing the slot only on LaunchError left the devices taken with nothing
    running on them and nothing recording them as owned, so they vanished from
    free_slots, owned and unconfirmed at once -- invisible capacity loss.
    """
    host = HostSpec(
        name="node", devices=(0, 1), port_base=8001, model="/m",
        # Ordinary Ascend setup. ${...} is not a {device} placeholder.
        env={"LD_LIBRARY_PATH": "/usr/local/Ascend/lib64:${LD_LIBRARY_PATH}"},
    )
    manager, _registry, _fake = _fleet(hosts=(host,))

    with pytest.raises(KeyError):
        asyncio.run(manager.launch_one())

    assert manager.pool.in_use == 0
    assert [slot.name for slot in manager.pool.free_slots()] == ["node-d0", "node-d1"]


def test_launch_initial_counts_the_replicas_already_owned():
    """Adoption satisfies --initial-replicas rather than adding to it.

    _lifespan adopts a previous run's replicas and only then launches, so
    counting just the configured number grew the fleet by one card on every
    restart until it reached the ceiling, each one a fresh model load.
    """
    manager, _registry, _fake = _fleet(initial=1)

    async def scenario():
        await manager.launch_one()  # stands in for a replica adopted on restart
        return await manager.launch_initial()

    assert asyncio.run(scenario()) == []
    assert len(manager.owned_names) == 1


def test_a_non_ascii_bearer_token_is_rejected_rather_than_crashing():
    """secrets.compare_digest raises TypeError on non-ASCII *str* arguments.

    The supplied half is attacker-controlled, so comparing str turned
    `Authorization: Bearer <any non-ASCII>` into an unhandled 500 with a
    traceback on every route that launches processes on other machines.
    """
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
    response = asyncio.run(router._add_replica(_AdminRequest({"authorization": "Bearer tökén"})))
    assert response.status_code == 401


def test_an_empty_admin_token_env_var_is_refused_rather_than_ignored(tmp_path, monkeypatch):
    """`os.environ.get(...) or args.admin_token` treats "" as absent.

    A deploy script expanding a variable that was never defined would otherwise
    leave /replicas -- which starts processes on other hosts -- open to anyone.
    """
    import uvicorn

    from pypto_serving.router.app import ADMIN_TOKEN_ENV, main

    fleet = tmp_path / "fleet.json"
    fleet.write_text(json.dumps({"hosts": [{"name": "n", "devices": [0], "model": "/m"}]}))
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "")

    # main() must refuse *before* it serves. Without this guard a regression does
    # not fail the test, it binds 0.0.0.0:8000 and hangs the whole suite until the
    # runner's timeout -- so the serve call is stubbed and asserted unreached.
    served = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: served.append(True))

    assert main(["--replicas", str(fleet)]) == 1
    assert served == [], "the router started serving with an empty admin token"


@pytest.mark.parametrize("contents", [
    "null",
    '{"replicas": "node-d0"}',
    '{"replicas": ["node-d0"]}',
    '{"replicas": [{"slot": "node-d0", "address": "h", "port": "8001x"}]}',
    '{"replicas": [{"slot": "node-d0", "address": "h", "port": null}]}',
])
def test_a_malformed_state_file_is_ignored_rather_than_fatal(tmp_path, contents):
    """adopt_previous is awaited from the lifespan with no handler.

    Anything it raises stops the router from starting, so a state file that is
    valid JSON but structurally wrong -- hand-edited, or written by another
    version -- must degrade to "nothing adopted", not to a router that will not
    boot without the file being deleted by hand.
    """
    state = tmp_path / "state.json"
    state.write_text(contents)

    async def answers(_spec):
        return True

    manager, _registry, _fake = _fleet(state_path=state, probe=answers)
    assert asyncio.run(manager.adopt_previous()) == []


def test_a_slot_may_not_launch_onto_an_address_a_static_replica_holds():
    """Names were checked, addresses were not.

    Two registry entries for one process double-count its load and split its
    sessions, and stopping the slot runs the default pkill against that port --
    killing a replica this router promised never to stop.
    """
    with pytest.raises(ValueError, match="already declared as replica"):
        RouterConfig(
            replicas=(ReplicaSpec(name="existing", host="127.0.0.1", port=8001),),
            hosts=(HostSpec(name="node01", devices=(0,), port_base=8001, model="/m"),),
        )


def test_two_hosts_may_not_describe_the_same_machine_and_devices():
    """Only host *names* were required to be unique.

    Two entries for one machine inflate the reported ceiling and hand the same
    card to two replicas.
    """
    with pytest.raises(ValueError, match="collide on the same address"):
        RouterConfig(hosts=(
            HostSpec(name="a", ssh="user@node02", devices=(0, 1), model="/m"),
            HostSpec(name="b", ssh="user@node02", devices=(0, 1), model="/m"),
        ))
