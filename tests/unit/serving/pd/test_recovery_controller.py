# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import asyncio

import pytest

from pypto_serving.router.recovery import (
    FixedPairRecoveryController,
    RecoveryPhase,
    ReplayableRequest,
    RuntimeGeneration,
)
from pypto_serving.serving.pd.protocol import HandoffKey


class _Manager:
    def __init__(self, dead: bool = True, ready: bool = True) -> None:
        self.dead = dead
        self.ready = ready
        self.calls = []

    async def retire(self, current, reason):
        self.calls.append(("retire", current, reason))

    async def confirm_dead(self, current):
        self.calls.append(("dead", current))
        return self.dead

    async def restart(self, next_generation):
        self.calls.append(("restart", next_generation))

    async def validate_ready(self, expected):
        self.calls.append(("ready", expected))
        return self.ready


def test_recovery_requires_death_barrier_and_replays_only_unpublished() -> None:
    async def exercise() -> None:
        controller = FixedPairRecoveryController(3, 7)
        key = HandoffKey("request", "handoff", 3, 1, 7)
        controller.require("UNKNOWN_TRANSFER", (key,))
        with pytest.raises(RuntimeError, match="admission is stopped"):
            controller.assert_admission()

        manager = _Manager()
        replayed = []
        generation = await controller.recover(
            manager,
            (
                ReplayableRequest("safe", "completion", b"{}"),
                ReplayableRequest(
                    "published",
                    "completion",
                    b"{}",
                    output_published=True,
                ),
            ),
            lambda request: _append(replayed, request.request_id),
        )
        assert generation == RuntimeGeneration(4, 8)
        assert replayed == ["safe"]
        assert controller.phase is RecoveryPhase.RUNNING
        controller.assert_admission()
        assert [call[0] for call in manager.calls] == [
            "retire",
            "dead",
            "restart",
            "ready",
        ]

    async def _append(values, value):
        values.append(value)

    asyncio.run(exercise())


def test_recovery_stays_fail_closed_when_old_runtime_is_not_dead() -> None:
    async def exercise() -> None:
        controller = FixedPairRecoveryController(1, 1)
        controller.require("CONTROL_EOF")
        with pytest.raises(RuntimeError, match="not confirmed dead"):
            await controller.recover(_Manager(dead=False), (), lambda _request: None)
        assert controller.phase is RecoveryPhase.RECOVERY_REQUIRED
        assert controller.current == RuntimeGeneration(1, 1)

    asyncio.run(exercise())


def test_replay_uses_new_generation_and_failure_cannot_roll_back_identity() -> None:
    async def exercise() -> None:
        controller = FixedPairRecoveryController(5, 9)
        controller.require("OWNER_DIED")

        async def fail_replay(_request):
            assert controller.current == RuntimeGeneration(6, 10)
            raise RuntimeError("replay failed")

        with pytest.raises(RuntimeError, match="replay failed"):
            await controller.recover(
                _Manager(),
                (ReplayableRequest("safe", "completion", b"{}"),),
                fail_replay,
            )
        assert controller.current == RuntimeGeneration(6, 10)
        assert controller.phase is RecoveryPhase.RECOVERY_REQUIRED
        with pytest.raises(RuntimeError, match="admission is stopped"):
            controller.assert_admission()

    asyncio.run(exercise())


def test_recovery_stays_fail_closed_when_replacement_is_not_ready() -> None:
    async def exercise() -> None:
        controller = FixedPairRecoveryController(2, 4)
        controller.require("CONTROL_EOF")
        with pytest.raises(RuntimeError, match="failed validation"):
            await controller.recover(
                _Manager(ready=False),
                (),
                lambda _request: None,
            )
        assert controller.current == RuntimeGeneration(2, 4)
        assert controller.phase is RecoveryPhase.RECOVERY_REQUIRED
        with pytest.raises(RuntimeError, match="admission is stopped"):
            controller.assert_admission()

    asyncio.run(exercise())
