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

from pypto_serving.serving.pd.admission import (
    FairHandoffAdmission,
    PDBackpressureError,
)


def test_fair_handoff_admission_runs_up_to_active_limit_in_fifo_order() -> None:
    async def exercise() -> None:
        admission = FairHandoffAdmission(active_limit=2, total_limit=4)
        release = asyncio.Event()
        entered = []

        async def hold(index: int) -> None:
            async with admission.admit():
                entered.append(index)
                if index < 2:
                    await release.wait()

        tasks = [asyncio.create_task(hold(index)) for index in range(4)]
        await asyncio.sleep(0)
        assert entered == [0, 1]
        assert admission.active == 2
        assert admission.queued == 2
        with pytest.raises(PDBackpressureError):
            async with admission.admit():
                pass
        release.set()
        await asyncio.gather(*tasks)
        assert entered == [0, 1, 2, 3]
        assert admission.count == 0

    asyncio.run(exercise())


def test_fair_handoff_admission_removes_cancelled_waiter_without_reordering() -> None:
    async def exercise() -> None:
        admission = FairHandoffAdmission(active_limit=1, total_limit=4)
        release = asyncio.Event()
        entered = []

        async def hold(index: int) -> None:
            async with admission.admit():
                entered.append(index)
                if index == 0:
                    await release.wait()

        first = asyncio.create_task(hold(0))
        await asyncio.sleep(0)
        cancelled = asyncio.create_task(hold(1))
        second = asyncio.create_task(hold(2))
        third = asyncio.create_task(hold(3))
        await asyncio.sleep(0)
        assert admission.active == 1
        assert admission.queued == 3

        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert admission.active == 1
        assert admission.queued == 2

        release.set()
        await asyncio.gather(first, second, third)
        assert entered == [0, 2, 3]
        assert admission.count == 0

    asyncio.run(exercise())


def test_fair_handoff_admission_observes_queue_before_it_is_admitted() -> None:
    async def exercise() -> None:
        states = []
        admission = FairHandoffAdmission(
            active_limit=1,
            total_limit=2,
            state_observer=lambda active, queued: states.append((active, queued)),
        )
        release = asyncio.Event()

        async def hold() -> None:
            async with admission.admit():
                await release.wait()

        first = asyncio.create_task(hold())
        await asyncio.sleep(0)
        second = asyncio.create_task(hold())
        await asyncio.sleep(0)
        assert (1, 1) in states

        release.set()
        await asyncio.gather(first, second)
        assert states[-1] == (0, 0)

    asyncio.run(exercise())


def test_handoff_budget_can_be_released_before_stream_with_idempotent_cleanup():
    async def run():
        admission = FairHandoffAdmission(1, 2)
        async with admission.admit() as ready:
            assert admission.active == 1
            await ready()
            await ready()
            assert admission.active == 0
            async with admission.admit():
                assert admission.active == 1
        assert admission.active == 0 and admission.queued == 0
    asyncio.run(run())
