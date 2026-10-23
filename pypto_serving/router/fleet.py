# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Growing and shrinking the set of replicas the router owns.

The pool decides *where* a replica can go; this decides *when* one appears and
disappears, and keeps the registry in step. It also owns the two asymmetries a
model load forces:

* growing is slow and must not be waited on -- a replica is registered
  unroutable and promoted by the health poller minutes later;
* shrinking must not be abrupt -- a replica stops taking new sessions, finishes
  what it holds, and only then is stopped.

Underneath both runs one invariant: **a device group stays reserved until the
replica that held it is confirmed retired**. Concurrency and restarts are the
two ways to lose it, so retirement is shared per *instance* rather than per slot
name, and every unresolved reservation is written to the state file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pypto_serving.router.config import ReplicaSpec, RouterConfig
from pypto_serving.router.launcher import (
    HostPool,
    PoolExhausted,
    Slot,
    transport_for,
)
from pypto_serving.router.routing import ReplicaRegistry

logger = logging.getLogger(__name__)

# How long to wait for a signalled replica to actually stop answering before
# reusing its device anyway. A replica that ignores its stop must not hold a
# card for the life of the router.
STOP_GRACE_SECONDS = 30.0

# Why a slot is reserved with nothing of ours serving on it. Reported through
# capacity() so an operator can see why the fleet is smaller than the config.
STOP_UNCONFIRMED = "stop-unconfirmed"
UNREACHABLE_AT_STARTUP = "unreachable-at-startup"

# A replica that was serving and then stopped answering is replaced on its own
# slot. These three numbers are the whole policy.
#
# The wait before a replacement starts, doubling per attempt up to the cap. The
# first one is also the grace the dead process gets to let go of its devices:
# its socket closes before it exits, so a stop can be confirmed while the
# process is still on its way out.
RESTART_BACKOFF_SECONDS = 10.0
RESTART_BACKOFF_CAP_SECONDS = 300.0
# Consecutive replacements of one device group before the router stops spending
# model loads on it. Three is already minutes of device time on a shared host,
# and a slot that has eaten three replicas has something wrong with it that
# another launch will not fix.
MAX_RESTARTS = 3


@dataclass
class _Instance:
    """One launch onto one slot.

    The slot is reused by whatever replica comes next; the instance never is.
    Anything that can outlive a replica -- a retirement still waiting for the old
    process to go quiet -- is keyed on the instance, so a late completion cannot
    act on the replica that took its place.
    """

    slot: Slot
    generation: int
    spec: ReplicaSpec
    # One retirement per instance, shared by every caller that asks to stop it.
    retirement: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return self.slot.name


@dataclass(frozen=True)
class _Reserved:
    """A slot held with nothing of ours serving on it, and why."""

    slot: Slot
    reason: str


@dataclass
class _Restarts:
    """How many replicas one slot has lost in a row, and whether we gave up.

    Kept per slot rather than per instance: a crash loop is a property of the
    device group, not of any one replica that died on it.
    """

    count: int = 0
    given_up: bool = False


class FleetManager:
    """Launches, drains and stops the replicas this router owns."""

    def __init__(
        self,
        config: RouterConfig,
        registry: ReplicaRegistry,
        *,
        transport_factory=transport_for,
        state_path: Path | None = None,
        probe=None,
        alive_probe=None,
        stop_grace: float = STOP_GRACE_SECONDS,
        restart_backoff: float = RESTART_BACKOFF_SECONDS,
    ) -> None:
        self._config = config
        self._registry = registry
        self._pool = HostPool(config, transport_factory=transport_factory)
        self._state_path = state_path
        # Two questions, deliberately not one probe:
        #   probe        -- "is a replica *serving* here?" (readiness)
        #   alive_probe  -- "is *anything* answering here?" (any HTTP status)
        # Adoption asks the first, confirming an exit asks the second. Answering
        # the second with the first frees the devices of a process that is still
        # holding them; see _wait_gone.
        self._probe = probe
        self._alive_probe = alive_probe
        self._stop_grace = stop_grace
        self._restart_backoff = restart_backoff
        # name -> instance, for every replica this router started.
        self._owned: dict[str, _Instance] = {}
        self._watchers: dict[str, asyncio.Task] = {}
        # Slots reserved with nothing of ours on them: a stop that could not be
        # confirmed, or a replica a previous run recorded that did not answer at
        # startup. They stay taken in the pool, so nothing is launched onto
        # devices that may still be in use, and they are reported so an operator
        # can see why the capacity is lower than the configuration says.
        self._unconfirmed: dict[str, _Reserved] = {}
        self._generation = 0
        # Replacement bookkeeping: how often each slot has had to be refilled,
        # which slots are being refilled right now, and whether the router is
        # shutting down -- a replacement must not outlive stop_all().
        self._restarts: dict[str, _Restarts] = {}
        self._replacing: dict[str, bool] = {}
        self._closing = False

    @property
    def pool(self) -> HostPool:
        return self._pool

    @property
    def owned_names(self) -> tuple[str, ...]:
        return tuple(self._owned)

    def capacity(self) -> dict:
        return {
            "owned": len(self._owned),
            "ceiling": self._pool.ceiling,
            "free_slots": [slot.name for slot in self._pool.free_slots()],
            # Reserved but not serving, and why.
            "unconfirmed": {
                name: held.reason for name, held in sorted(self._unconfirmed.items())
            },
            # Being refilled right now, and given up on after too many tries.
            "replacing": sorted(self._replacing),
            "abandoned": sorted(
                name for name, record in self._restarts.items() if record.given_up
            ),
        }

    # --- growing ----------------------------------------------------------

    async def launch_one(self) -> ReplicaSpec:
        """Reserve a slot, start a replica on it, and register it unroutable.

        Returns as soon as the launch command has been accepted. The replica is
        not usable yet; the health poller promotes it when its model is up.
        """
        slot = self._pool.allocate()  # synchronous: no await may split this
        # An operator launch starts that slot's history over. The restart cap
        # exists to stop the router spending model loads on its own, not to stop
        # a human who has decided the slot is worth another try.
        self._restarts.pop(slot.name, None)
        return await self._launch_on(slot)

    async def _launch_on(self, slot: Slot) -> ReplicaSpec:
        """Start a replica on a slot that is already reserved."""
        launched = await self._pool.launch(slot)  # releases the slot if it fails
        self._adopt_instance(slot, launched.spec, ready=False)
        self._write_state()
        return launched.spec

    def _adopt_instance(self, slot: Slot, spec: ReplicaSpec, *, ready: bool) -> _Instance:
        """Record a replica on an already-reserved slot, and watch it come up."""
        self._generation += 1
        instance = _Instance(slot=slot, generation=self._generation, spec=spec)
        self._owned[instance.name] = instance
        self._registry.add(spec, ready=ready, owned=True)
        # Watched even when it is already serving: a replica adopted from a
        # previous run used to get no supervisor at all, and neither did any
        # replica once it had been promoted.
        self._watchers[instance.name] = asyncio.create_task(
            self._watch_instance(instance.name, instance.generation, ready=ready)
        )
        return instance

    def _poll_step(self, deadline: float) -> float:
        """How often a watcher looks, for a deadline of this size.

        Never slower than 5s (a model load is minutes), never faster than 50ms,
        and never coarser than the deadline itself, so a short deadline is
        honoured rather than rounded up to the next poll.
        """
        return max(0.05, min(5.0, self._config.health_interval_seconds, deadline))

    def _is_current(self, name: str, generation: int) -> bool:
        """Is this still the instance the watcher was started for?

        A watcher asleep in its poll when its replica is replaced would wake to
        find the replacement unready -- it is loading -- and declare *it* dead.
        """
        instance = self._owned.get(name)
        return instance is not None and instance.generation == generation

    async def _watch_instance(self, name: str, generation: int, *, ready: bool) -> None:
        """Supervise one instance for as long as it is the one on its slot.

        Two phases. Until it first reports ready this is the launch timeout: a
        replica that never comes up is stopped so its devices come back, and it
        is deliberately *not* replaced -- a launch that never succeeded is a bad
        configuration or a bad checkpoint, and relaunching it is a crash loop by
        construction.

        After that comes the case nothing watched before: a replica that was
        serving and stopped. That one is replaced, because capacity returning
        without a human is the point of owning the launch at all.
        """
        if not ready and not await self._await_ready(name, generation):
            return
        await self._await_death(name, generation)

    async def _await_ready(self, name: str, generation: int) -> bool:
        """Wait for a launch to report ready. False if it never did."""
        deadline = self._config.launch_timeout_seconds
        step = self._poll_step(deadline)
        waited = 0.0
        while waited < deadline:
            if not self._is_current(name, generation):
                return False
            state = self._registry.state(name)
            if state is None:
                return False  # removed by someone else
            if state.ready:
                logger.info("replica %s reported ready after %.0fs", name, waited)
                return True
            await asyncio.sleep(step)
            waited += step
        logger.error(
            "replica %s never became ready within the %.0fs launch timeout; stopping it and "
            "freeing its device(s). If the model was still loading -- DeepSeek V4 needs well "
            "over 1200s from cold -- raise --launch-timeout; %s holds the startup output",
            name, deadline, self._pool.log_path_for(name) or "the host's log_dir",
        )
        await self.stop(name, drain=False)
        return False

    def _death_dwell(self) -> float:
        """How long unroutable means dead rather than briefly unwell.

        Floored at two probe intervals, because a dwell shorter than that cannot
        see a recovery: it would pay a whole model load for a replica whose next
        probe would have answered 200.
        """
        return max(
            self._config.replace_after_seconds, 2 * self._config.health_interval_seconds
        )

    async def _await_death(self, name: str, generation: int) -> None:
        """Replace this instance once it has been unroutable for the dwell.

        The registry is polled rather than hooked, because the health poller is
        not the only thing that demotes a replica: the proxy does it too, on any
        request it could not deliver, with no threshold at all. That is the
        earlier and more certain signal, and only the registry sees both.
        """
        if not self._config.replace_dead:
            return
        dwell = self._death_dwell()
        step = self._poll_step(dwell)
        unready_for = 0.0
        ready_for = 0.0
        while unready_for < dwell:
            if not self._is_current(name, generation):
                return
            state = self._registry.state(name)
            if state is None or state.draining:
                return  # someone is already stopping it
            if state.ready:
                unready_for = 0.0
                ready_for += step
                if ready_for >= self._config.launch_timeout_seconds:
                    # Up long enough to be a healthy replica rather than one
                    # more turn of a loop. Without this, a replica that dies
                    # once a month would eventually exhaust a cap meant to
                    # catch a replica dying once a minute.
                    self._restarts.pop(name, None)
            else:
                unready_for += step
            await asyncio.sleep(step)

        instance = self._owned.get(name)
        if instance is None or instance.generation != generation:
            return
        logger.error(
            "replica %s has been unroutable for %.0fs; retiring it and refilling slot %s. "
            "%s holds its output",
            name, dwell, name, instance.slot.log_path,
        )
        await self._replace(instance)

    async def _replace(self, instance: _Instance) -> None:
        """Retire a dead instance, then start another on its slot.

        Stop first and launch second, never overlapped. A replacement started
        before the old one is confirmed gone would need a second device group,
        and two serving processes on one card is the thing this module exists to
        prevent.
        """
        name = instance.name
        self._replacing[name] = True
        try:
            if not await self.stop(name, drain=True):
                # The stop could not be confirmed, so the slot is reserved and a
                # human has to free it. Launching onto it could put a second
                # process on cards the first may still hold.
                return
            if self._closing or not self._replacing.get(name):
                return
            record = self._restarts.setdefault(name, _Restarts())
            record.count += 1
            if record.count > MAX_RESTARTS:
                record.given_up = True
                logger.error(
                    "slot %s has lost %d replicas in a row; not launching another. The slot "
                    "is free, so POST /replicas or a restart will use it again once whatever "
                    "is wrong with it is fixed; %s holds the last one's output",
                    name, MAX_RESTARTS, instance.slot.log_path,
                )
                return
            slot = self._pool.reserve(name)
            if slot is None:
                logger.warning(
                    "slot %s was taken while its replacement was being prepared", name
                )
                return
            # Reserved before the wait, so the slot is honestly accounted for --
            # out of free_slots and reported as replacing -- rather than looking
            # available to a concurrent launch for the whole backoff.
            await asyncio.sleep(self._backoff(record.count))
            if self._closing or not self._replacing.get(name):
                self._pool.release(name)
                return
            await self._launch_on(slot)
        except Exception:  # noqa: BLE001 - a failed replacement must not kill the watcher
            logger.exception("replacing replica %s failed", name)
        finally:
            self._replacing.pop(name, None)

    def _backoff(self, attempt: int) -> float:
        return min(RESTART_BACKOFF_CAP_SECONDS, self._restart_backoff * 2 ** (attempt - 1))

    def cancel_replacement(self, name: str) -> bool:
        """Call off a replacement that has not launched yet.

        Returns whether there was one. Without this a DELETE arriving between
        the stop and the launch answers 404 -- the registry entry is already
        gone -- and the replica the operator just retired comes back anyway.
        """
        if name not in self._replacing:
            return False
        self._replacing[name] = False
        return True

    async def launch_initial(self) -> list[ReplicaSpec]:
        """Start the replicas the configuration asks for at boot.

        Concurrent, because N model loads in parallel cost one load's wall clock
        rather than N of them.
        """
        # Replicas adopted from a previous run already count towards the request.
        # Without this the fleet grows by initial_replicas on *every* restart
        # until it hits the ceiling -- a card per restart on a shared cluster,
        # each one a fresh model load -- and then reports a spurious
        # "initial launch failed: PoolExhausted" for a perfectly healthy adoption.
        wanted = self._config.initial_replicas - len(self._owned)
        if wanted <= 0:
            return []
        results = await asyncio.gather(
            *(self.launch_one() for _ in range(wanted)), return_exceptions=True
        )
        launched: list[ReplicaSpec] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error("initial launch failed: %s", result)
            else:
                launched.append(result)
        return launched

    # --- shrinking --------------------------------------------------------

    async def stop(self, name: str, *, drain: bool = True) -> bool:
        """Drain (optionally) and stop a replica this router owns.

        Every caller asking to stop the same instance shares one retirement.
        Running one each is what lost slots: two concurrent DELETEs both passed
        the ownership check, the second found the registry entry already gone,
        read that as proof of exit, and released the slot while the first was
        still waiting for the endpoint to go quiet. If a replacement had been
        launched onto that slot in between, the first one's cleanup then retired
        *it* -- leaving a live replica nobody owned on devices the pool was free
        to hand out again.
        """
        instance = self._owned.get(name)
        if instance is None:
            return False
        self._cancel_watcher(name)
        if instance.retirement is None or instance.retirement.cancelled():
            instance.retirement = asyncio.create_task(self._retire(instance, drain=drain))
        # Shielded: a client that disconnects mid-DELETE cancels this coroutine,
        # and a retirement abandoned after the stop was signalled would leave a
        # dead replica owning its slot with nothing left to release it.
        return await asyncio.shield(instance.retirement)

    def _cancel_watcher(self, name: str) -> None:
        """Drop the launch watcher of a replica that is going away.

        Never cancelled from inside itself: the watcher calls stop() when a
        launch overruns, so cancelling it there would deliver the CancelledError
        to the very coroutine asking for the retirement.
        """
        watcher = self._watchers.pop(name, None)
        if watcher is None or watcher.done() or watcher is asyncio.current_task():
            return
        watcher.cancel()

    async def _retire(self, instance: _Instance, *, drain: bool) -> bool:
        """Take one instance out of rotation and stop it. One per instance."""
        name = instance.name
        try:
            if drain:
                await self._drain(name)
            state = self._registry.remove(name)
            spec = state.spec if state is not None else instance.spec
            signalled = await self._pool.signal_stop(instance.slot)
            # Only now is the device really free. Releasing the slot on the
            # signal alone would let the next launch bind a port the old replica
            # still holds, and that failure would look like a bad configuration.
            gone = await self._wait_gone(spec)
        except Exception:  # noqa: BLE001 - whatever failed, the devices are unknown
            self._registry.remove(name)
            logger.exception("retiring replica %s failed", name)
            signalled = gone = False
        return self._finish_retirement(instance, signalled=signalled, gone=gone)

    def _finish_retirement(self, instance: _Instance, *, signalled: bool, gone: bool) -> bool:
        name = instance.name
        if self._owned.get(name) is not instance:
            # Checked rather than assumed. The slot stays taken in the pool for
            # the whole retirement, so nothing should be able to take it in the
            # meantime -- but acting on a replacement is precisely the failure
            # this instance identity exists to make impossible, so it is worth a
            # branch instead of a comment.
            logger.warning(
                "the retirement of %s (generation %d) finished after its slot changed hands; "
                "leaving the current owner alone",
                name, instance.generation,
            )
            return signalled and gone

        self._owned.pop(name, None)
        if not (signalled and gone):
            # The replica is out of rotation either way, but its devices are in
            # an unknown state: the stop command failed, or something is still
            # answering. Reusing them could overlap two instances on the same
            # cards, so the slot stays reserved and the retirement is reported
            # as incomplete. Freeing it is a human decision for now.
            self._unconfirmed[name] = _Reserved(instance.slot, STOP_UNCONFIRMED)
            self._write_state()
            logger.error(
                "replica %s could not be confirmed stopped (signalled=%s, gone=%s); "
                "slot %s stays reserved",
                name, signalled, gone, instance.slot.name,
            )
            return False

        self._pool.release(name)
        self._write_state()
        logger.info("replica %s stopped; slot %s is free", name, instance.slot.name)
        return True

    async def _wait_gone(self, spec: ReplicaSpec, timeout: float | None = None) -> bool:
        """Wait until nothing answers at a stopped replica's address.

        The question here is liveness, not readiness: *any* HTTP answer means
        something is still there, 503 included. serving answers /health with 503
        whenever its engine is not ready, so taking a 503 for an exit would
        release the devices of a process that is still holding them -- and a
        worker that died while its HTTP server stayed up is exactly when that
        answer appears.

        A silent endpoint is still weak evidence -- it says the HTTP server is
        gone, not that the serving process and its device workers have exited --
        but a noisy one is conclusive: something is there, so the devices are
        not free.

        TODO: a stronger check needs a verifiable identity for what was launched
        (a pid, or the broker task id where there is a broker) instead of an
        address. Worth a configurable per-host probe, like ``stop_command``, if
        the endpoint test proves too weak in practice.
        """
        # The readiness probe is the weaker answer to this question -- it calls a
        # 503 "gone" -- so it is a fallback for callers that supply only one, not
        # the default the router runs with.
        probe = self._alive_probe or self._probe
        if probe is None:
            return True
        timeout = self._stop_grace if timeout is None else timeout
        waited = 0.0
        while waited < timeout:
            if not await probe(spec):
                return True
            await asyncio.sleep(0.5)
            waited += 0.5
        logger.error(
            "%s still answered %s after %.0fs; keeping its slot reserved",
            spec.name, spec.base_url, timeout,
        )
        return False

    async def _drain(self, name: str) -> None:
        """Stop routing to a replica and wait for its in-flight work.

        Bounded: a stuck request must not hold a device forever, but the bound
        is generous because cutting a user's generation short to reclaim a card
        is the worse trade.
        """
        self._registry.set_draining(name, True)
        deadline = self._config.drain_timeout_seconds
        waited = 0.0
        while waited < deadline:
            state = self._registry.state(name)
            if state is None or state.outstanding <= 0:
                return
            await asyncio.sleep(0.25)
            waited += 0.25
        state = self._registry.state(name)
        remaining = state.outstanding if state else 0
        logger.warning(
            "replica %s still had %d request(s) in flight after %.0fs; stopping anyway",
            name, remaining, deadline,
        )

    async def stop_all(self) -> None:
        """Stop every owned replica. Called on an explicit shutdown, not a crash."""
        # Before anything else: a replacement between its stop and its launch is
        # no longer in _owned, so the snapshot below cannot see it and the
        # lifespan would exit leaving a fresh process holding a card.
        self._closing = True
        names = list(self._owned)
        if not names:
            return
        logger.info("stopping %d replica(s) this router launched", len(names))
        await asyncio.gather(*(self.stop(name) for name in names), return_exceptions=True)

    # --- crash recovery ---------------------------------------------------

    @staticmethod
    def _state_entry(slot: Slot, *, state: str, reason: str | None = None) -> dict:
        entry = {
            "slot": slot.name,
            "host": slot.host.name,
            "devices": list(slot.devices),
            "address": slot.host.address(),
            "port": slot.port,
            "state": state,
        }
        if reason is not None:
            entry["reason"] = reason
        return entry

    def _write_state(self) -> None:
        if self._state_path is None:
            return
        # Unresolved reservations are written too. Recording only what is owned
        # dropped a slot from the file at exactly the moment it mattered -- a
        # stop that could not be confirmed -- so the next run saw that device
        # group as free and could launch onto cards the old process may still
        # have held.
        payload = {
            "replicas": [
                self._state_entry(instance.slot, state="owned")
                for instance in self._owned.values()
            ] + [
                self._state_entry(held.slot, state="unconfirmed", reason=held.reason)
                for held in self._unconfirmed.values()
            ]
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            # Temp file plus os.replace, because write_text truncates before it
            # writes: a crash mid-write would leave a half-written file, and
            # surviving a crash is the only reason this file exists.
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:
            logger.warning("could not write fleet state to %s: %s", self._state_path, exc)

    def _hold(self, slot: Slot, reason: str, detail: str) -> None:
        """Keep a slot reserved with nothing of ours on it, and say why."""
        self._unconfirmed[slot.name] = _Reserved(slot, reason)
        logger.error(
            "slot %s stays reserved (%s): %s. Free it by stopping whatever holds "
            "%s device(s) %s and restarting the router",
            slot.name, reason, detail, slot.host.name, list(slot.devices),
        )

    async def adopt_previous(self, probe=None) -> list[str]:
        """Re-take replicas a previous run of this router left behind.

        A crash runs no shutdown hook, so its replicas keep serving. Throwing
        them away would cost a fresh model load each; probing and re-adopting
        the ones that answer costs a round trip.

        The recorded reservation is restored *before* anything is probed. A
        replica that was still loading its model when the router died answers
        nothing yet -- uvicorn binds only once the engine is up -- and so does a
        healthy one behind a blip in the network. Classifying first and
        reserving only on success let launch_initial() put a second replica on
        those same devices.
        """
        if self._state_path is None or not self._state_path.exists():
            return []
        try:
            recorded = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ignoring unreadable fleet state %s: %s", self._state_path, exc)
            return []

        # Everything below is parsed defensively. The file is operator-visible
        # (--state-file) and may be hand-edited or written by another version, and
        # adopt_previous is awaited from the lifespan with no handler: an
        # AttributeError or ValueError here does not degrade to "no adoption", it
        # stops the router from starting at all.
        if not isinstance(recorded, dict):
            logger.warning("ignoring fleet state %s: not a JSON object", self._state_path)
            return []
        entries = recorded.get("replicas", [])
        if not isinstance(entries, list):
            logger.warning("ignoring fleet state %s: 'replicas' is not a list", self._state_path)
            return []

        adopted: list[str] = []
        for entry in entries:
            name = self._recorded_name(entry)
            if name is None:
                continue
            slot = self._pool.reserve(name)
            if slot is None:
                logger.warning(
                    "slot %s is no longer declared in the config, or is already taken; "
                    "not adopting", name,
                )
                continue
            if entry.get("state") == "unconfirmed":
                # The previous run could not confirm this one stopped. Nothing
                # since then has, either. The recorded reason is matched against
                # the known ones rather than echoed: the file is hand-editable
                # and capacity() is an API response.
                reason = entry.get("reason")
                self._hold(
                    slot,
                    reason if reason in (STOP_UNCONFIRMED, UNREACHABLE_AT_STARTUP)
                    else STOP_UNCONFIRMED,
                    "the previous run could not confirm it stopped",
                )
                continue
            if await self._adopt_recorded(slot, entry, probe):
                adopted.append(name)

        if adopted:
            logger.info("adopted %d replica(s) from the previous run: %s",
                        len(adopted), ", ".join(adopted))
        self._write_state()
        return adopted

    def _recorded_name(self, entry) -> str | None:
        """The slot name of a state-file entry worth acting on, if it has one."""
        if not isinstance(entry, dict):
            logger.warning("ignoring a malformed entry in %s: %r", self._state_path, entry)
            return None
        name = entry.get("slot")
        if not isinstance(name, str) or not name:
            return None
        if self._registry.state(name) is not None:
            return None
        return name

    async def _adopt_recorded(self, slot: Slot, entry: dict, probe) -> bool:
        """Classify one recorded replica on its restored slot.

        Ready, still starting, or unreachable -- and the slot stays reserved for
        all three. Only the first two are adopted into the registry.
        """
        try:
            port = int(entry.get("port", 0))
        except (TypeError, ValueError):
            port = 0
        address = entry.get("address", "")
        if not isinstance(address, str) or not address or not port:
            self._hold(slot, UNREACHABLE_AT_STARTUP,
                       f"its recorded address is unusable ({address!r}:{entry.get('port')!r})")
            return False

        spec = ReplicaSpec(name=slot.name, host=address, port=port)
        ready = probe if probe is not None else self._probe
        if ready is not None and await ready(spec):
            self._adopt_instance(slot, spec, ready=True)
            return True
        # Not serving. If anything still answers there it is ours and it is
        # coming up, so adopt it unroutable and let the health poller promote it.
        # It is watched like a fresh launch, so one that never comes up is
        # stopped at --launch-timeout and gives its devices back.
        alive = self._alive_probe
        if alive is not None and await alive(spec):
            logger.info(
                "replica %s from the previous run answers at %s but is not ready; "
                "adopting it as still starting",
                slot.name, spec.base_url,
            )
            self._adopt_instance(slot, spec, ready=False)
            return True
        self._hold(slot, UNREACHABLE_AT_STARTUP,
                   f"it did not answer at {spec.base_url}")
        return False


__all__ = ["MAX_RESTARTS", "RESTART_BACKOFF_SECONDS", "FleetManager", "PoolExhausted"]
