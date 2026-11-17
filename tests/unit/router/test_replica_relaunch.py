# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Refilling a slot whose replica died.

A replica that was serving and then stopped used to sit in the table forever,
unroutable, holding its devices, reported exactly like one still loading its
model. Capacity only came back when a person noticed.

The transport is faked, so none of this spawns anything: what is under test is
when the router decides a replica is dead, and what it does about it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from pypto_serving.router.config import RouterConfig
from pypto_serving.router.fleet import MAX_RESTARTS
from pypto_serving.router.launcher import LaunchError
from tests.unit.router.test_launcher import (
    _AdminRequest,
    _fleet,
    _FakeTransport,
    _hosts,
    _router_with_pool,
)


def _launches(transport: _FakeTransport) -> int:
    return sum(1 for argv in transport.commands() if "pypto_serving.cli" in argv)


def _stops(transport: _FakeTransport) -> int:
    return sum(1 for argv in transport.commands() if "pkill" in argv)


async def _promote(registry, fleet, name: str = "local-d0") -> None:
    """Make a launched replica report ready, as the health poller would."""
    registry.set_ready(name, True)
    await asyncio.sleep(0.05)


async def _die(registry, name: str = "local-d0") -> None:
    registry.set_ready(name, False)


async def _until(predicate, timeout: float = 5.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


def test_a_replica_that_dies_after_being_promoted_is_replaced_on_its_own_slot():
    """The case nothing watched: the launch watcher stopped at the promotion.

    The replacement has to land on the same device group. Taking the next free
    slot instead would walk the fleet off a bad card one replica at a time,
    and the port a replica is stopped by is derived from its devices.
    """
    async def scenario():
        fleet, registry, transport = _fleet(probe=None, replace_after=0.05)
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)
        assert _launches(transport) == 1

        await _die(registry, spec.name)
        assert await _until(lambda: _launches(transport) == 2), transport.commands()
        assert await _until(lambda: fleet.owned_names == (spec.name,))
        return fleet.capacity(), fleet.owned_names, _stops(transport)

    capacity, owned, stops = asyncio.run(scenario())
    assert owned == ("local-d0",), "the replacement owns the same slot"
    assert stops == 1, "the dead one was stopped before the replacement started"
    assert capacity["owned"] == 1
    assert "local-d0" not in capacity["free_slots"]
    assert capacity["abandoned"] == []


def test_a_brief_blip_does_not_cost_a_model_load():
    """Unroutable is cheap to recover from; replacing is not.

    One failed probe already takes a replica out of rotation, so the dwell is
    what separates "it will be back" from "it is gone". A dwell that fires on
    the first blip pays minutes of model load for a replica whose next probe
    would have answered.
    """
    async def scenario():
        fleet, registry, transport = _fleet(probe=None, replace_after=2.0)
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)

        await _die(registry, spec.name)
        await asyncio.sleep(0.2)
        registry.set_ready(spec.name, True)
        await asyncio.sleep(0.5)
        return _launches(transport), _stops(transport)

    launches, stops = asyncio.run(scenario())
    assert launches == 1, "a replica that came back was replaced anyway"
    assert stops == 0


def test_a_replica_that_never_came_up_is_stopped_but_not_replaced():
    """Relaunching a launch that never worked is a crash loop by construction.

    A replica that has never reported ready is a bad checkpoint or a bad
    configuration, not a crash: its devices come back, and nothing else.
    """
    async def scenario():
        fleet, registry, transport = _fleet(
            probe=None, launch_timeout=0.2, replace_after=0.05,
        )
        await fleet.launch_one()  # never promoted
        await asyncio.sleep(1.0)
        return _launches(transport), _stops(transport), fleet.capacity()

    launches, stops, capacity = asyncio.run(scenario())
    assert launches == 1, "a replica that never came up was relaunched"
    assert stops == 1
    assert "local-d0" in capacity["free_slots"]


def test_an_unconfirmed_stop_is_never_relaunched_onto():
    """The slot is reserved precisely because something may still hold it.

    Launching there would put a second process on cards the first has not let
    go of -- the one outcome the whole module exists to avoid.
    """
    async def scenario():
        async def still_answering(_spec):
            return True

        fleet, registry, transport = _fleet(
            alive_probe=still_answering, replace_after=0.05, stop_grace=0.2,
        )
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)

        await _die(registry, spec.name)
        await asyncio.sleep(1.5)
        return _launches(transport), fleet.capacity()

    launches, capacity = asyncio.run(scenario())
    assert launches == 1, "the router launched onto a slot it could not confirm free"
    assert capacity["unconfirmed"] == {"local-d0": "stop-unconfirmed"}
    assert "local-d0" not in capacity["free_slots"]


def test_a_crash_loop_is_capped_and_the_slot_is_reported():
    """Three model loads of device time is enough to spend on one bad card.

    The slot is left genuinely free afterwards -- the router will not use it
    again by itself, but an operator who has fixed it can.
    """
    async def scenario():
        fleet, registry, transport = _fleet(probe=None, replace_after=0.05)
        spec = await fleet.launch_one()
        for _ in range(MAX_RESTARTS + 2):
            await _promote(registry, fleet, spec.name)
            await _die(registry, spec.name)
            await asyncio.sleep(0.4)
        return _launches(transport), fleet.capacity(), fleet.owned_names

    launches, capacity, owned = asyncio.run(scenario())
    assert launches == 1 + MAX_RESTARTS, "the cap did not stop the loop"
    assert capacity["abandoned"] == ["local-d0"]
    assert "local-d0" in capacity["free_slots"], "an abandoned slot is free, just not automatic"
    assert owned == ()


def test_replacement_can_be_turned_off():
    """--no-replace-dead has to reproduce the old behaviour exactly.

    Someone investigating a dead replica wants it left where it is, with its
    devices still held.
    """
    async def scenario():
        fleet, registry, transport = _fleet(
            probe=None, replace_after=0.05, replace_dead=False,
        )
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)
        await _die(registry, spec.name)
        await asyncio.sleep(0.5)
        return _launches(transport), _stops(transport), fleet.owned_names

    launches, stops, owned = asyncio.run(scenario())
    assert (launches, stops) == (1, 0)
    assert owned == ("local-d0",), "the dead replica kept its slot, as asked"


def test_a_shutdown_does_not_leave_a_replacement_behind():
    """stop_all cannot see a replacement between its stop and its launch.

    It has already left _owned, so the snapshot stop_all takes misses it, and
    the router would exit leaving a process it just started holding a card.
    """
    async def scenario():
        fleet, registry, transport = _fleet(probe=None, replace_after=0.05, restart_backoff=0.5)
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)

        await _die(registry, spec.name)
        # Inside the replacement's backoff, where the slot is reserved and
        # nothing is registered.
        assert await _until(lambda: fleet.capacity()["replacing"] == ["local-d0"])
        await fleet.stop_all()
        await asyncio.sleep(1.0)
        return _launches(transport), fleet.owned_names

    launches, owned = asyncio.run(scenario())
    assert launches == 1, "a replacement was launched into a shutdown"
    assert owned == ()


def test_a_delete_during_a_replacement_is_not_resurrected():
    """Between the stop and the launch there is no registry entry to DELETE.

    Answering 404 there and carrying on would bring back the replica the
    operator had just retired.
    """
    async def scenario():
        fleet, registry, transport = _fleet(probe=None, replace_after=0.05, restart_backoff=0.5)
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)

        await _die(registry, spec.name)
        assert await _until(lambda: fleet.capacity()["replacing"] == ["local-d0"])
        cancelled = fleet.cancel_replacement(spec.name)
        await asyncio.sleep(1.0)
        return cancelled, _launches(transport), fleet.capacity()

    cancelled, launches, capacity = asyncio.run(scenario())
    assert cancelled is True
    assert launches == 1, "the cancelled replacement launched anyway"
    assert "local-d0" in capacity["free_slots"], "a cancelled replacement left its slot reserved"
    assert capacity["replacing"] == []


def test_a_delete_during_the_replacements_own_drain_is_not_resurrected():
    """The window the first guard missed.

    A replacement arms itself and *then* drains, and a drain can last the whole
    drain timeout. A DELETE arriving in there finds the registry entry present
    and draining, joins the retirement already running, and is answered
    "stopped" -- while the replacement, still armed, puts the replica back.
    """
    async def scenario():
        # The real handler, because the fix is the order it does things in.
        router, transport = _router_with_pool(
            replace_after_seconds=0.05, drain_timeout_seconds=3.0,
        )
        created = json.loads((await router._add_replica(_AdminRequest())).body)
        name = created["name"]
        router.registry.set_ready(name, True)
        await asyncio.sleep(0.05)
        router.registry.acquire(name)  # one request in flight, so the drain waits
        router.registry.set_ready(name, False)
        assert await _until(lambda: router.fleet.capacity()["replacing"] == [name])

        response = await router._remove_replica(name, _AdminRequest())
        await asyncio.sleep(0.5)
        return response.status_code, _launches(transport), router.fleet.capacity()

    status, launches, capacity = asyncio.run(scenario())
    assert status == 200, "the operator was not told the replica had stopped"
    assert launches == 1, "the retired replica came back anyway"
    assert "local-d0" in capacity["free_slots"]


def test_a_shutdown_during_the_launch_does_not_leave_a_replica_behind():
    """stop_all cannot see a replacement that has not been registered yet.

    Its snapshot of what is owned is taken before _adopt_instance runs, so a
    launch still in flight is invisible to it and the router would exit with a
    process it had just started holding a card.
    """
    class _SlowTransport(_FakeTransport):
        """A launch that takes long enough for a shutdown to overtake it."""

        async def run(self, host, argv, env, *, detach=False, log_path=None):
            await super().run(host, argv, env, detach=detach, log_path=log_path)
            if detach:
                await asyncio.sleep(1.0)

    async def scenario():
        transport = _SlowTransport()
        fleet, registry, _ = _fleet(probe=None, replace_after=0.05, transport=transport)
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)

        await _die(registry, spec.name)
        # The replacement's launch command has gone out and has not returned.
        assert await _until(lambda: _launches(transport) == 2)
        await fleet.stop_all()
        await asyncio.sleep(2.0)
        return fleet.owned_names, _stops(transport), fleet.capacity()

    owned, stops, capacity = asyncio.run(scenario())
    assert owned == (), "the router exited owning a replica it had just launched"
    assert stops == 2, "the overtaken replacement was never stopped"
    assert capacity["owned"] == 0


def test_the_dwell_is_never_shorter_than_two_probe_intervals():
    """A dwell below the poll rate cannot observe a recovery at all."""
    fleet, _registry, _transport = _fleet(replace_after=0.0001)
    assert fleet._death_dwell() >= 2 * fleet._config.health_interval_seconds


def test_the_backoff_grows_and_is_capped():
    """A slot that keeps losing replicas must not be retried at full speed."""
    fleet, _registry, _transport = _fleet(restart_backoff=10.0)
    delays = [fleet._backoff(attempt) for attempt in range(1, 8)]
    assert delays[:3] == [10.0, 20.0, 40.0]
    assert delays == sorted(delays), "the backoff has to grow"
    assert max(delays) <= 300.0, "and stop growing"


# --- a replacement that is still launching ---

class _HoldsALaunch(_FakeTransport):
    """Records a launch, then stays inside run() until it is released.

    The window a real launch has and a fake one normally does not: the process
    is already spawned, and the call that spawned it has not returned yet.
    """

    def __init__(self, hold: int) -> None:
        super().__init__()
        self._hold = hold
        self._launched = 0
        self.inside = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, host, argv, env, *, detach=False, log_path=None):
        await super().run(host, argv, env, detach=detach, log_path=log_path)
        if any("pypto_serving.cli" in arg for arg in argv):
            self._launched += 1
            if self._launched == self._hold:
                self.inside.set()
                await self.release.wait()


def test_shutdown_waits_for_a_replacement_that_is_still_launching():
    """A replacement mid-launch is in neither _owned nor the registry.

    stop_all() snapshots _owned, so it cannot see one, and letting the loop
    teardown cancel it is worse than waiting: HostPool.launch hands the slot
    back on CancelledError, so the process it had just spawned would be left
    behind with its slot reported free and nothing owning it.
    """
    async def scenario():
        transport = _HoldsALaunch(hold=2)
        fleet, registry, _ = _fleet(transport=transport, replace_after=0.05)
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)
        await _die(registry, spec.name)
        assert await _until(transport.inside.is_set), "the replacement never launched"

        stopping = asyncio.create_task(fleet.stop_all())
        await asyncio.sleep(0.1)
        still_running = not stopping.done()
        transport.release.set()
        await asyncio.wait_for(stopping, timeout=5.0)
        return still_running, fleet.owned_names, _launches(transport), _stops(transport)

    waited, owned, launches, stops = asyncio.run(scenario())
    assert waited, "stop_all() returned while a replacement was still launching"
    assert owned == ()
    assert launches == 2
    assert stops == launches, "the router exited leaving a process it had started"


# --- what must not be mistaken for a dead replica ---


def test_a_full_connection_pool_is_not_an_exited_process():
    """The liveness probe decides whether a slot's devices are free.

    httpx raises PoolTimeout when the *router* has no free connection, which a
    coding agent's long streams can cause on a perfectly healthy fleet. Reading
    it as silence relaunches onto cards the old process still holds.
    """
    httpx = pytest.importorskip("httpx")
    router, _transport = _router_with_pool()
    slot = router.fleet.pool.free_slots()[0]

    class _Client:
        def __init__(self, failure) -> None:
            self.failure = failure

        async def get(self, url, timeout=None):
            raise self.failure

    router.probe_client = _Client(httpx.PoolTimeout("no free connection"))
    assert asyncio.run(router._endpoint_answers(slot.replica_spec())) is True

    router.probe_client = _Client(httpx.ConnectError("connection refused"))
    assert asyncio.run(router._endpoint_answers(slot.replica_spec())) is False, (
        "a refused connection is still evidence the endpoint is gone"
    )


def test_probes_do_not_share_the_proxy_connection_pool():
    """Two clients, so proxied traffic cannot starve the probes.

    With one pool the two are the same resource, and saturating it looks
    exactly like every replica dying at the same moment.
    """
    from pypto_serving.router.app import ServingRouter

    built = []

    class _Client:
        async def get(self, url, timeout=None):
            raise AssertionError("no upstream in this test")

        async def aclose(self):
            pass

    def factory(_config):
        built.append(_Client())
        return built[-1]

    router = ServingRouter(
        RouterConfig(hosts=_hosts(), initial_replicas=0, health_interval_seconds=0.01),
        client_factory=factory,
        transport_factory=lambda _host: _FakeTransport(),
    )
    assert len(built) == 2, "the probe pool has to be a second client, not the proxy's"
    assert router.probe_client is not router.client
    assert router.health._client is router.probe_client


# --- a stop that finds nothing to kill ---

class _StopFinds(_FakeTransport):
    """A stop command that reports an outcome instead of killing anything.

    ``exit_code=1`` is pkill's "no process matched"; ``None`` is a command that
    could not be run at all, which is a different answer.
    """

    def __init__(self, exit_code: int | None) -> None:
        super().__init__()
        self._exit_code = exit_code

    async def run(self, host, argv, env, *, detach=False, log_path=None):
        await super().run(host, argv, env, detach=detach, log_path=log_path)
        if any("pkill" in arg for arg in argv):
            raise LaunchError("stop reported no match", exit_code=self._exit_code)


def test_a_stop_that_matches_no_process_still_frees_the_slot():
    """A replica that crashed outright is not there for pkill to kill.

    pkill matches on the command line, so it finds a replica from the instant
    it execs rather than from the moment it binds a port: nothing matched
    therefore means nothing is running. Reading it as an unknown state strands
    the devices of every replica that died completely -- which is precisely the
    case the replacement path exists to handle.
    """
    async def scenario(exit_code):
        fleet, _registry, _ = _fleet(transport=_StopFinds(exit_code))
        spec = await fleet.launch_one()
        return await fleet.stop(spec.name), fleet.capacity()

    stopped, capacity = asyncio.run(scenario(1))
    assert stopped is True
    assert capacity["unconfirmed"] == {}
    assert "local-d0" in capacity["free_slots"], "the slot was stranded"

    stopped, capacity = asyncio.run(scenario(None))
    assert stopped is False, "a stop command that could not run says nothing"
    assert "local-d0" in capacity["unconfirmed"]


def test_a_launch_cancelled_in_flight_is_stopped_at_shutdown():
    """The launch records the slot before starting anything.

    Cancellation inside the launch is the one path that reaches neither the
    post-launch check nor _owned, so without the record the shutdown has
    nothing left to name the slot by -- and HostPool.launch has already
    reported it free.
    """
    async def scenario():
        transport = _HoldsALaunch(hold=1)
        fleet, _registry, _ = _fleet(transport=transport)
        launching = asyncio.create_task(fleet.launch_one())
        assert await _until(transport.inside.is_set), "the launch never started"
        launching.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await launching
        orphaned = fleet.owned_names == () and fleet.capacity()["launching"] == ["local-d0"]
        free_meanwhile = [s for s in fleet.capacity()["free_slots"] if s == "local-d0"]

        await fleet.stop_all()
        return orphaned, free_meanwhile, _launches(transport), _stops(transport), fleet.capacity()

    orphaned, free_meanwhile, launches, stops, capacity = asyncio.run(scenario())
    assert orphaned, "the cancelled launch left no record of its slot"
    assert free_meanwhile == [], "the slot was advertised free with a process possibly on it"
    assert launches == 1
    assert stops == 1, "shutdown left the launched process alone"
    assert "local-d0" in capacity["free_slots"], "the slot never came back"


def test_the_upstream_client_states_its_pool_limits_and_its_pool_timeout():
    """Both were inherited rather than chosen, and both are load-bearing.

    The connection ceiling is the ceiling on concurrent upstream requests, and
    httpx's default of 100 is reached with far fewer clients than that once
    they hold streams open. The pool timeout otherwise inherits
    request_timeout, so a request with no free connection waits an hour for
    one instead of failing.
    """
    httpx = pytest.importorskip("httpx")
    from pypto_serving.router.app import default_client_factory

    config = RouterConfig(
        hosts=_hosts(), request_timeout_seconds=3600.0, connect_timeout_seconds=5.0
    )
    client = default_client_factory(config)
    try:
        assert client.timeout.pool == config.connect_timeout_seconds
        assert client.timeout.read is None, "decode is slow by design"
        # Private, but httpx offers no way to read the limits back, and an
        # unverifiable claim about them is the thing worth catching.
        pool = client._transport._pool
        assert pool._max_connections == config.max_connections
        assert pool._max_keepalive_connections == config.max_keepalive_connections
        assert config.max_connections != 100, "100 is the default we are replacing"
    finally:
        asyncio.run(client.aclose())


# --- a replacement whose launch command fails ---

class _LaunchRefused(_FakeTransport):
    """Lets the first launch through, then refuses the next ``refuse`` of them."""

    def __init__(self, refuse: int) -> None:
        super().__init__()
        self._refuse = refuse
        self._launches = 0

    async def run(self, host, argv, env, *, detach=False, log_path=None):
        if detach:
            self._launches += 1
            if 1 < self._launches <= 1 + self._refuse:
                raise LaunchError("launch refused", exit_code=1)
        await super().run(host, argv, env, detach=detach, log_path=log_path)


def test_a_replacement_whose_launch_command_fails_is_retried():
    """A launch that never ran is the transient the backoff exists for.

    Stopping at the first failure leaves the slot free, the fleet one replica
    short and nothing watching it, well before the restart limit it documents.
    A launch that succeeds and never reports ready is deliberately not retried
    -- that is the launch timeout's job, and it means a bad configuration.
    """
    async def scenario(refuse, done):
        fleet, registry, transport = _fleet(transport=_LaunchRefused(refuse), replace_after=0.05)
        spec = await fleet.launch_one()
        await _promote(registry, fleet, spec.name)
        await _die(registry, spec.name)
        # Waiting on "replacing is empty" would pass before the replacement has
        # even begun; each case waits for its own end state instead.
        reached = await _until(lambda: done(fleet, transport), timeout=5.0)
        return reached, fleet.owned_names, fleet.capacity(), _launches(transport)

    reached, owned, capacity, _ = asyncio.run(
        scenario(MAX_RESTARTS - 1, lambda fleet, transport: _launches(transport) == 2)
    )
    assert reached, "the replacement gave up on a transient launch failure"
    assert owned == ("local-d0",)
    assert capacity["abandoned"] == []

    reached, owned, capacity, launches = asyncio.run(
        scenario(MAX_RESTARTS, lambda fleet, _t: fleet.capacity()["abandoned"] == ["local-d0"])
    )
    assert reached, "every attempt failed and the slot was never reported"
    assert owned == (), "nothing should be running after every attempt failed"
    assert launches == 1, "no replacement ever started, so none should be recorded"
    assert "local-d0" in capacity["free_slots"], "the slot stays usable once it is fixed"
