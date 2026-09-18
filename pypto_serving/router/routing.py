# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Where a conversation goes, and whether its replica is still alive.

The serving prefix cache is per replica, and its block hashes cover the prompt
at admission, so turn N's answer enters the cache as part of turn N+1's prompt.
Sending consecutive turns to the same replica is what makes turn N+1 re-prefill
only the new text.

Session expiry drops the *pin*, never KV: blocks are keyed by content and shared
between conversations, so there is nothing per-session to evict.

Load is counted as outstanding requests the router itself dispatched -- it never
tokenizes, so it has no token estimate, and it sees every request.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from pypto_serving.router.config import ReplicaSpec, RouterConfig

logger = logging.getLogger(__name__)

# Consecutive failed probes before a replica leaves rotation. One blip should not
# depin every session; recovery is immediate on the first success.
UNHEALTHY_THRESHOLD = 2

# Session ids are client-supplied, so the directory is an unbounded map keyed by
# untrusted input until its entries expire. Cap it and evict least-recently-used
# pins: losing a pin costs one prefill, unbounded growth costs the process.
DEFAULT_MAX_SESSIONS = 100_000


class NoReplicaAvailable(RuntimeError):
    """Raised when every replica in the table is unroutable."""


def new_session_id() -> str:
    return uuid.uuid4().hex


class SessionDirectory:
    """Maps a conversation id to the replica holding its KV.

    The clock is injectable so expiry is testable without sleeping.
    """

    def __init__(self, ttl_seconds: float, *, clock: Callable[[], float] = time.monotonic,
                 max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self._ttl = ttl_seconds
        self._clock = clock
        self._max = max_sessions
        # Ordered by recency of pin, so eviction is a popitem from the front.
        self._pins: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._pins)

    def lookup(self, session_id: str) -> str | None:
        """Return the pinned replica name, dropping the pin if it has expired."""
        pin = self._pins.get(session_id)
        if pin is None:
            return None
        name, last_seen = pin
        if self._clock() - last_seen > self._ttl:
            del self._pins[session_id]
            return None
        return name

    def pin(self, session_id: str, replica_name: str) -> None:
        self._pins[session_id] = (replica_name, self._clock())
        self._pins.move_to_end(session_id)
        if len(self._pins) > self._max:
            self.sweep()
        while len(self._pins) > self._max:
            evicted, _ = self._pins.popitem(last=False)
            logger.debug("session directory full; evicted the least-recent pin %s", evicted)

    def forget(self, session_id: str) -> None:
        self._pins.pop(session_id, None)

    def sweep(self) -> int:
        """Drop every expired pin. Lookup only expires pins it touches, so this
        bounds the memory held by sessions that are never seen again."""
        now = self._clock()
        expired = [sid for sid, (_, seen) in self._pins.items() if now - seen > self._ttl]
        for session_id in expired:
            del self._pins[session_id]
        return len(expired)


@dataclass
class ReplicaState:
    """Mutable routing state for one replica."""

    spec: ReplicaSpec
    # Carried explicitly: the rotating tiebreak needs it, and looking it up by
    # value would rely on dataclass equality.
    index: int = 0
    outstanding: int = 0
    routed: int = 0
    # Replicas start routable: a launch still loading refuses the connection
    # anyway, and the first probe corrects an optimistic guess quickly.
    ready: bool = True
    failures: int = 0

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class RoutingDecision:
    replica: ReplicaSpec
    affinity_hit: bool


class ReplicaRegistry:
    """Owns replica state, picks a replica per request, and counts what happened."""

    def __init__(self, config: RouterConfig, sessions: SessionDirectory) -> None:
        self._config = config
        self._sessions = sessions
        self._states = [ReplicaState(spec=spec, index=i) for i, spec in enumerate(config.replicas)]
        self._by_name = {state.name: state for state in self._states}
        # Rotating tiebreak, matching AsyncLLMEngine._select_replica: equal-load
        # replicas are taken in turn instead of always the lowest index.
        self._route_counter = 0
        self.affinity_hits = 0
        self.rejected = 0

    @property
    def states(self) -> tuple[ReplicaState, ...]:
        return tuple(self._states)

    def state(self, name: str) -> ReplicaState | None:
        return self._by_name.get(name)

    def ready_count(self) -> int:
        return sum(1 for state in self._states if state.ready)

    def total_routed(self) -> int:
        return sum(state.routed for state in self._states)

    def set_ready(self, name: str, ready: bool) -> None:
        state = self._by_name.get(name)
        if state is None or state.ready == ready:
            return
        state.ready = ready
        logger.info("replica %s is now %s", name, "routable" if ready else "unroutable")

    def acquire(self, name: str) -> None:
        state = self._by_name.get(name)
        if state is not None:
            state.outstanding += 1

    def release(self, name: str) -> None:
        state = self._by_name.get(name)
        if state is not None:
            state.outstanding = max(0, state.outstanding - 1)

    def select(self, session_id: str) -> RoutingDecision:
        """Pick a replica for this session and record the pin.

        Affinity is a preference, not an invariant: prefix blocks are evictable,
        so a hit is never guaranteed, and the one prefill it saves is worth less
        than an unbounded wait behind a saturated replica.
        """
        candidates = [state for state in self._states if state.ready]
        if not candidates:
            self.rejected += 1
            raise NoReplicaAvailable("no routable replica")

        least = self._least_loaded(candidates)
        pinned = self._by_name.get(self._sessions.lookup(session_id) or "")
        chosen, affinity_hit = least, False

        if pinned is not None and pinned.ready:
            if pinned.outstanding > least.outstanding + self._config.affinity_slack:
                logger.info(
                    "session %s leaves %s (outstanding=%d) for %s (outstanding=%d): slack %d exceeded",
                    session_id, pinned.name, pinned.outstanding,
                    least.name, least.outstanding, self._config.affinity_slack,
                )
            else:
                chosen, affinity_hit = pinned, True

        if not affinity_hit:
            # Only advance the rotation when load decided the route, so a stream
            # of affine requests does not skew the tiebreak.
            self._route_counter = (chosen.index + 1) % len(self._states)

        self._sessions.pin(session_id, chosen.name)
        chosen.routed += 1
        self.affinity_hits += affinity_hit
        return RoutingDecision(replica=chosen.spec, affinity_hit=affinity_hit)

    def _least_loaded(self, candidates: list[ReplicaState]) -> ReplicaState:
        count = len(self._states)
        return min(
            candidates,
            key=lambda s: (s.outstanding, (s.index - self._route_counter) % count, s.index),
        )


class HealthMonitor:
    """Polls each replica's /health and keeps the registry in step.

    A replica still loading refuses the connection outright (uvicorn binds only
    after the engine starts), while a 503 means the process is up but its worker
    or engine loop is gone. Both are unroutable.
    """

    def __init__(self, config: RouterConfig, registry: ReplicaRegistry,
                 sessions: SessionDirectory, client) -> None:
        self._config = config
        self._registry = registry
        self._sessions = sessions
        self._client = client
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _loop(self) -> None:
        while True:
            try:
                await self.probe_once()
                self._sessions.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the poller must outlive one bad cycle
                logger.exception("health poll cycle failed")
            await asyncio.sleep(self._config.health_interval_seconds)

    async def probe_once(self) -> None:
        await asyncio.gather(
            *(self._probe(state) for state in self._registry.states), return_exceptions=True
        )

    async def _probe(self, state: ReplicaState) -> None:
        healthy = False
        try:
            response = await self._client.get(
                f"{state.spec.base_url}/health", timeout=self._config.connect_timeout_seconds
            )
            healthy = response.status_code == 200
        except Exception as exc:  # noqa: BLE001 - refused and timeout both mean unroutable
            logger.debug("health probe of %s failed: %s", state.name, exc)

        if healthy:
            state.failures = 0
            self._registry.set_ready(state.name, True)
            return
        state.failures += 1
        if state.failures >= UNHEALTHY_THRESHOLD:
            self._registry.set_ready(state.name, False)
