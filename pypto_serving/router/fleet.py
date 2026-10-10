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
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from pypto_serving.router.config import ReplicaSpec, RouterConfig
from pypto_serving.router.launcher import (
    HostPool,
    LaunchError,
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
    ) -> None:
        self._config = config
        self._registry = registry
        self._pool = HostPool(config, transport_factory=transport_factory)
        self._state_path = state_path
        # Answers "is something serving at this address?". Used both to adopt a
        # previous run's replicas and to confirm a stopped one is really gone.
        self._probe = probe
        # name -> slot, for every replica this router started.
        self._owned: dict[str, Slot] = {}
        self._watchers: dict[str, asyncio.Task] = {}
        self._draining: dict[str, asyncio.Task] = {}

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
        }

    # --- growing ----------------------------------------------------------

    async def launch_one(self) -> ReplicaSpec:
        """Reserve a slot, start a replica on it, and register it unroutable.

        Returns as soon as the launch command has been accepted. The replica is
        not usable yet; the health poller promotes it when its model is up.
        """
        slot = self._pool.allocate()  # synchronous: no await may split this
        try:
            launched = await self._pool.launch(slot)
        except LaunchError:
            # launch() already released the slot.
            raise
        self._owned[slot.name] = slot
        self._registry.add(launched.spec, ready=False, owned=True)
        self._write_state()
        self._watchers[slot.name] = asyncio.create_task(self._watch_launch(slot.name))
        return launched.spec

    async def _watch_launch(self, name: str) -> None:
        """Stop a replica that never becomes ready, so its device comes back."""
        deadline = self._config.launch_timeout_seconds
        waited = 0.0
        # Never slower than 5s (a model load is minutes), never faster than 50ms,
        # and never coarser than the deadline itself, so a short timeout is
        # honoured rather than rounded up to the next poll.
        step = max(0.05, min(5.0, self._config.health_interval_seconds, deadline))
        try:
            while waited < deadline:
                state = self._registry.state(name)
                if state is None:
                    return  # removed by someone else
                if state.ready:
                    logger.info("replica %s reported ready after %.0fs", name, waited)
                    return
                await asyncio.sleep(step)
                waited += step
        except asyncio.CancelledError:
            raise
        logger.error(
            "replica %s never became ready within the %.0fs launch timeout; stopping it and "
            "freeing its device(s). If the model was still loading -- DeepSeek V4 needs well "
            "over 1200s from cold -- raise --launch-timeout; %s holds the startup output",
            name, deadline, self._pool.log_path_for(name) or "the host's log_dir",
        )
        await self.stop(name, drain=False)

    async def launch_initial(self) -> list[ReplicaSpec]:
        """Start the replicas the configuration asks for at boot.

        Concurrent, because N model loads in parallel cost one load's wall clock
        rather than N of them.
        """
        wanted = self._config.initial_replicas
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
        """Drain (optionally) and stop a replica this router owns."""
        slot = self._owned.get(name)
        if slot is None:
            return False

        watcher = self._watchers.pop(name, None)
        if watcher is not None and not watcher.done():
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

        if drain:
            await self._drain(name)
        state = self._registry.remove(name)
        await self._pool.signal_stop(slot)
        # Only now is the device really free. Releasing the slot on the signal
        # alone would let the next launch bind a port the old replica still
        # holds, and that failure would look like a bad configuration.
        if state is not None:
            await self._wait_gone(state.spec)
        self._pool.release(name)
        self._owned.pop(name, None)
        self._write_state()
        logger.info("replica %s stopped; slot %s is free", name, slot.name)
        return True

    async def _wait_gone(self, spec: ReplicaSpec, timeout: float = STOP_GRACE_SECONDS) -> None:
        """Wait until nothing answers at a stopped replica's address."""
        if self._probe is None:
            return
        waited = 0.0
        while waited < timeout:
            if not await self._probe(spec):
                return
            await asyncio.sleep(0.5)
            waited += 0.5
        logger.warning(
            "%s still answered %s after %.0fs; reusing its slot anyway",
            spec.name, spec.base_url, timeout,
        )

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
        names = list(self._owned)
        if not names:
            return
        logger.info("stopping %d replica(s) this router launched", len(names))
        await asyncio.gather(*(self.stop(name) for name in names), return_exceptions=True)

    # --- crash recovery ---------------------------------------------------

    def _write_state(self) -> None:
        if self._state_path is None:
            return
        payload = {
            "replicas": [
                {
                    "slot": name,
                    "host": slot.host.name,
                    "devices": list(slot.devices),
                    "address": slot.host.address(),
                    "port": slot.port,
                }
                for name, slot in self._owned.items()
            ]
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not write fleet state to %s: %s", self._state_path, exc)

    async def adopt_previous(self, probe=None) -> list[str]:
        """Re-take replicas a previous run of this router left behind.

        A crash runs no shutdown hook, so its replicas keep serving. Throwing
        them away would cost a fresh model load each; probing and re-adopting
        the ones that answer costs a round trip. Slots whose replica does not
        answer are reported, because something is still holding those devices.
        """
        if self._state_path is None or not self._state_path.exists():
            return []
        try:
            recorded = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ignoring unreadable fleet state %s: %s", self._state_path, exc)
            return []

        adopted: list[str] = []
        for entry in recorded.get("replicas", []):
            name = entry.get("slot")
            if not isinstance(name, str) or self._registry.state(name) is not None:
                continue
            spec = ReplicaSpec(
                name=name, host=entry.get("address", ""), port=int(entry.get("port", 0)),
            )
            if not spec.host or not spec.port:
                continue
            checker = probe if probe is not None else self._probe
            if checker is None or not await checker(spec):
                logger.warning(
                    "replica %s from the previous run does not answer at %s; if a process "
                    "is still holding %s device(s) %s, stop it by hand",
                    name, spec.base_url, entry.get("host"), entry.get("devices"),
                )
                continue
            slot = self._pool.reserve(name)
            if slot is None:
                logger.warning("slot %s is no longer declared in the config; not adopting", name)
                continue
            self._owned[name] = slot
            self._registry.add(spec, ready=True, owned=True)
            adopted.append(name)

        if adopted:
            logger.info("adopted %d replica(s) from the previous run: %s",
                        len(adopted), ", ".join(adopted))
            self._write_state()
        return adopted


__all__ = ["FleetManager", "PoolExhausted"]
