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

from pypto_serving.router.fleet import MAX_RESTARTS
from tests.unit.router.test_launcher import _fleet, _FakeTransport


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
