# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Crash-only fixed-pair recovery state machine for the external Router."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import Enum
import time
from typing import Awaitable, Callable, Protocol

from pypto_serving.serving.pd.protocol import HandoffKey


class RecoveryPhase(str, Enum):
    RUNNING = "RUNNING"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    RETIRING = "RETIRING"
    RESTARTING = "RESTARTING"
    VALIDATING = "VALIDATING"
    REPLAYING = "REPLAYING"


@dataclass(frozen=True)
class RuntimeGeneration:
    data_generation: int
    control_incarnation: int


@dataclass(frozen=True)
class ReplayableRequest:
    request_id: str
    request_kind: str
    request_json: bytes
    output_published: bool = False


class FixedPairRuntimeManager(Protocol):
    async def retire(self, current: RuntimeGeneration, reason: str) -> None: ...

    async def confirm_dead(self, current: RuntimeGeneration) -> bool: ...

    async def restart(self, next_generation: RuntimeGeneration) -> None: ...

    async def validate_ready(self, expected: RuntimeGeneration) -> bool: ...


class CallbackFixedPairRuntimeManager:
    """Bind the Router state machine to a deployment-owned process supervisor.

    The callbacks are deliberately semantic rather than shell-oriented.  The
    deployment layer keeps ownership of PID/session identities and launch
    mechanics; the Router only accepts a positive death barrier and an exact
    generation-ready validation before replay is possible.
    """

    def __init__(
        self,
        *,
        retire: Callable[[RuntimeGeneration, str], Awaitable[None]],
        confirm_dead: Callable[[RuntimeGeneration], Awaitable[bool]],
        restart: Callable[[RuntimeGeneration], Awaitable[None]],
        validate_ready: Callable[[RuntimeGeneration], Awaitable[bool]],
    ) -> None:
        self._retire = retire
        self._confirm_dead = confirm_dead
        self._restart = restart
        self._validate_ready = validate_ready

    async def retire(self, current: RuntimeGeneration, reason: str) -> None:
        await self._retire(current, reason)

    async def confirm_dead(self, current: RuntimeGeneration) -> bool:
        return bool(await self._confirm_dead(current))

    async def restart(self, next_generation: RuntimeGeneration) -> None:
        await self._restart(next_generation)

    async def validate_ready(self, expected: RuntimeGeneration) -> bool:
        return bool(await self._validate_ready(expected))


class FixedPairRecoveryController:
    """Gate admission and orchestrate a new-generation re-prefill recovery.

    The controller intentionally does not know process names, shell commands,
    NPU addresses, or provider envelopes.  A launcher-owned runtime manager
    must prove the exact old runtime group dead before a new arena generation
    is allowed to start.
    """

    def __init__(self, generation: int, control_incarnation: int) -> None:
        if generation < 1 or control_incarnation < 1:
            raise ValueError("recovery generations must be positive")
        self.current = RuntimeGeneration(generation, control_incarnation)
        self.phase = RecoveryPhase.RUNNING
        self.reason = ""
        self.affected: tuple[HandoffKey, ...] = ()
        self._lock = asyncio.Lock()
        self._history: deque[dict[str, object]] = deque(maxlen=128)

    def assert_admission(self) -> None:
        if self.phase is not RecoveryPhase.RUNNING:
            raise RuntimeError(
                f"PD Router admission is stopped: {self.phase.value} ({self.reason})"
            )

    def require(self, reason: str, affected: tuple[HandoffKey, ...] = ()) -> None:
        if not reason:
            raise ValueError("recovery reason must not be empty")
        if self.phase is RecoveryPhase.RUNNING:
            self.phase = RecoveryPhase.RECOVERY_REQUIRED
            self.reason = reason[:128]
            self.affected = affected
            self._append("RECOVERY_REQUIRED")

    async def recover(
        self,
        manager: FixedPairRuntimeManager,
        replayable: tuple[ReplayableRequest, ...],
        replay: Callable[[ReplayableRequest], Awaitable[None]],
    ) -> RuntimeGeneration:
        async with self._lock:
            if self.phase is not RecoveryPhase.RECOVERY_REQUIRED:
                raise RuntimeError("fixed-pair recovery was not requested")
            old = self.current
            try:
                self.phase = RecoveryPhase.RETIRING
                self._append("RETIRING")
                await manager.retire(old, self.reason)
                if not await manager.confirm_dead(old):
                    raise RuntimeError("old fixed P/D runtime group is not confirmed dead")

                next_generation = RuntimeGeneration(
                    old.data_generation + 1,
                    old.control_incarnation + 1,
                )
                self.phase = RecoveryPhase.RESTARTING
                self._append("RESTARTING", generation=next_generation)
                await manager.restart(next_generation)
                self.phase = RecoveryPhase.VALIDATING
                self._append("VALIDATING", generation=next_generation)
                if not await manager.validate_ready(next_generation):
                    raise RuntimeError("new fixed P/D runtime group failed validation")

                # The old runtime is already dead and the replacement has been
                # validated.  Replay identities must therefore use the new
                # generation; rolling back to the old identity is impossible
                # and would reopen the ABA window this controller fences.
                self.current = next_generation
                self.phase = RecoveryPhase.REPLAYING
                self._append("REPLAYING", generation=next_generation)
                for request in replayable:
                    if request.output_published:
                        continue
                    await replay(request)
            except BaseException as exc:
                self.phase = RecoveryPhase.RECOVERY_REQUIRED
                self.reason = type(exc).__name__
                self._append("RECOVERY_FAILED")
                raise
            self.phase = RecoveryPhase.RUNNING
            self.reason = ""
            self.affected = ()
            self._append("RECOVERY_COMPLETED", generation=next_generation)
            return next_generation

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase": self.phase.value,
            "reason": self.reason,
            "data_generation": self.current.data_generation,
            "control_incarnation": self.current.control_incarnation,
            "affected_handoffs": len(self.affected),
            "history": list(self._history),
        }

    def _append(
        self,
        event: str,
        *,
        generation: RuntimeGeneration | None = None,
    ) -> None:
        generation = generation or self.current
        self._history.append(
            {
                "event": event,
                "timestamp_ns": time.time_ns(),
                "data_generation": generation.data_generation,
                "control_incarnation": generation.control_incarnation,
            }
        )
