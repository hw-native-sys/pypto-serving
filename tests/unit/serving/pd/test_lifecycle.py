# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shutdown ordering without native engines, sockets, or device resets."""

import asyncio
from types import SimpleNamespace

import pytest

from pypto_serving.serving.pd.config import PDRole
from pypto_serving.serving.pd.lifecycle import TransferLifecycle
from pypto_serving.serving.pd.worker import PDWorkerRuntime


@pytest.mark.parametrize("initiator", ["p", "d", "both"])
def test_peer_shutdown_drains_then_releases_sender_before_receiver(initiator):
    async def run():
        events = []
        idle = asyncio.Event()

        async def release(role):
            assert idle.is_set()
            events.append(role)

        p = TransferLifecycle(PDRole.PREFILL, idle=idle.is_set, release=lambda: release("p"), timeout=1)
        d = TransferLifecycle(PDRole.DECODE, idle=idle.is_set, release=lambda: release("d"), timeout=1)
        p.bind(SimpleNamespace(send=d.receive))
        d.bind(SimpleNamespace(send=p.receive))
        for node in ({"p": [p], "d": [d], "both": [p, d]}[initiator]):
            node.begin()
        await asyncio.sleep(.02)
        assert not events and p.draining and d.draining
        idle.set()
        await asyncio.gather(p.close(), d.close())
        assert events == ["p", "d"]
        await asyncio.gather(p.close(), d.close())
        assert events == ["p", "d"]
    asyncio.run(run())


def test_peer_eof_is_not_sender_release_proof():
    async def run():
        released = []
        async def send(message):
            raise EOFError("peer exited")
        async def release():
            released.append(True)
        d = TransferLifecycle(PDRole.DECODE, idle=lambda: True, release=release, timeout=.05)
        d.bind(SimpleNamespace(send=send))
        with pytest.raises(EOFError):
            await d.close()
        with pytest.raises(EOFError):
            await d.close()
        assert not released
    asyncio.run(run())


def test_sender_release_failure_does_not_authorize_receiver():
    async def run():
        released = []
        async def fail():
            raise RuntimeError("unregister failed")
        async def receiver():
            released.append(True)
        p = TransferLifecycle(PDRole.PREFILL, idle=lambda: True, release=fail, timeout=.05)
        d = TransferLifecycle(PDRole.DECODE, idle=lambda: True, release=receiver, timeout=.05)
        p.bind(SimpleNamespace(send=d.receive))
        d.bind(SimpleNamespace(send=p.receive))
        result = await asyncio.gather(p.close(), d.close(), return_exceptions=True)
        assert isinstance(result[0], RuntimeError) and isinstance(result[1], asyncio.TimeoutError)
        assert not released
    asyncio.run(run())


def test_worker_keeps_failed_owner_and_never_stops_it_after_unregister_failure():
    events = []
    class Bridge:
        def __init__(self, rank):
            self.owner = f"owner-{rank}"
            self.rank = rank
            self.fail = rank == 1
        def release(self):
            events.append((self.rank, "release"))
            if self.fail:
                raise RuntimeError("unregister failed")
        def close(self):
            events.append((self.rank, "close"))
    runtime = PDWorkerRuntime(SimpleNamespace(role=PDRole.PREFILL), SimpleNamespace(ranks=2))
    bridges = tuple(Bridge(rank) for rank in range(2))
    runtime._pd_owner_bridges = bridges
    with pytest.raises(RuntimeError, match="rank=1.*stage=unregister"):
        runtime.close()
    assert runtime._pd_owner_bridges is bridges
    assert runtime._pd_retired_ranks == {0}
    assert (1, "close") not in events
    bridges[1].fail = False
    runtime.close()
    runtime.close()
    assert events.count((0, "release")) == events.count((0, "close")) == 1
    assert not runtime._pd_owner_bridges


def test_worker_receiver_does_not_unregister_without_peer_release_proof():
    runtime = PDWorkerRuntime(SimpleNamespace(role=PDRole.DECODE), SimpleNamespace(ranks=1))
    runtime._pd_owner_bridges = (object(),)
    runtime._pd_owner_envelopes = (object(),)
    with pytest.raises(RuntimeError, match="confirmed sender"):
        runtime.close()
    assert runtime._pd_owner_bridges


def test_service_concurrent_close_runs_retirement_once():
    from pypto_serving.serving.pd.service import PDServingService

    async def exercise():
        service = object.__new__(PDServingService)
        service._closed, service._close_task = False, None
        calls = []
        async def close():
            calls.append(True)
            await asyncio.sleep(0)
            service._closed = True
        service._close = close
        await asyncio.gather(service.close(), service.close())
        await service.close()
        assert calls == [True]
    asyncio.run(exercise())
