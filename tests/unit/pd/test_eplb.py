# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PD write admission, native completion and local EPLB maintenance together."""

import asyncio
from threading import Event
from types import SimpleNamespace
import socket

import msgspec
import pytest

from pypto_serving.config.types import EplbConfig
from pypto_serving.pd import worker as worker_module
from pypto_serving.pd.protocol import Completion, digest
from pypto_serving.pd.service import Service
from pypto_serving.pd.transfer import Region, TransferError
from pypto_serving.pd.worker import Progress, Services, WorkerRuntime, WorkerSettings
from pypto_serving.serving.server.serving_worker import WorkerProcess
from tests.unit.pd.test_service import handoff


class NativeOwner:
    def __init__(self, *_args):
        self.retained = []
        self.write = None

    def call(self, operation, **payload):
        assert operation == "write"
        return self.write(payload)

    def close(self):
        pass


def runtime_pair(monkeypatch, case, write):
    monkeypatch.setattr(worker_module, "Owner", NativeOwner)
    runtimes, observers = [], []
    for node in case.nodes:
        parent, child = socket.socketpair()
        runtime = WorkerRuntime(WorkerSettings(node.boot, "local", "mooncake", 8, 2, 2),
                                SimpleNamespace(ranks=node.topology.ranks), child)
        runtime.start_progress()
        runtimes.append(runtime)
        observers.append(Progress(parent, 2, 8))
    source, destination = runtimes
    p, d = case.nodes
    source.registry_digest = p.registry
    source.peer_fingerprints[d.boot] = d.registry
    source.peers[d.boot] = ()
    source.local_regions = {(rank, g.name): Region(p.boot, rank, g.name,
        len(g.layers) * g.blocks * g.page_bytes, {})
        for rank in range(p.topology.ranks) for g in case.source_geometry}
    source.peer_regions[d.boot] = {(rank, g.name): Region(d.boot, rank, g.name,
        len(g.layers) * g.blocks * g.page_bytes, {})
        for rank in range(d.topology.ranks) for g in case.destination.geometry}
    for owner in source.owners:
        owner.write = write
    return runtimes, observers


def compute_worker(transfer, name, events, *, fail=False):
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.model_record = SimpleNamespace(runtime_model=object())
    worker._services = Services.__new__(Services)
    worker._services.extension = SimpleNamespace(transfer=transfer)
    worker.executor = SimpleNamespace(eplb_enabled=True,
        eplb_config=EplbConfig(maintenance_timeout_seconds=2),
        take_ready_eplb_plan=lambda _: name, validate_eplb_plan=lambda *_: None)
    def install(*_args):
        assert not transfer.gate.permits
        events.append(name)
        if fail:
            raise RuntimeError("partial installation")
    worker.executor.apply_eplb_plan = install
    return worker


@pytest.mark.parametrize("p_hit,d_hit", [(0, 0), (0, 128), (128, 0), (128, 128)])
def test_handoff_and_both_maintenance_lanes_keep_progress_and_cache_ownership(monkeypatch, p_hit, d_hit):
    async def run():
        case = handoff(257, p_hit=p_hit, d_hit=d_hit, d_tp=4, compact=True)
        case.entry.chunk = case.entry.end = 0
        started, finish = Event(), Event()
        copied, events = [], []
        def write(payload):
            started.set()
            assert finish.wait(3)
            copied.extend(payload["copies"])
        runtimes, observers = runtime_pair(monkeypatch, case, write)
        source, destination = runtimes
        case.destination.progress = observers[1]
        source_service = Service(SimpleNamespace(handoff_timeout=2, capacity=8, prefill=(0,), decode=(0,)),
            SimpleNamespace(node_id="p"), "prefill", None,
            SimpleNamespace(kv_cache_manager=case.p), observers[0])
        tasks = []
        try:
            for index, plan in enumerate(case.plans):
                async def control(operation, **payload):
                    return await case.destination.control(case.nodes[0], case.source_geometry, operation,
                        msgspec.to_builtins(dict(binding=plan.binding, **payload)))
                entry = SimpleNamespace(cancelled=False)
                await source_service.authorize_plan(entry, control, plan, case.p_hit, digest(plan))
                job = await source_service.worker("submit", plan) if plan.copies else None
                if job is not None:
                    assert await asyncio.to_thread(started.wait, 1)
                if job is not None and not events and not tasks:
                    for runtime, name in zip(runtimes, ("P:g3", "D:g9")):
                        worker = compute_worker(runtime, name, events)
                        tasks.append(asyncio.create_task(asyncio.to_thread(worker._maintain_eplb)))
                    for runtime in runtimes:
                        for _ in range(100):
                            with runtime.gate.condition:
                                closed = runtime.gate.maintenance
                            if closed:
                                break
                            await asyncio.sleep(.001)
                        assert closed and not runtime.gate.reserve("other-request")
                    assert not events
                finish.set()
                completion = (await observers[0].wait(job) if job is not None
                              else Completion(digest(plan), plan.units))
                if plan.copies:
                    assert "in:" + digest(plan) in destination.gate.permits
                await control("complete", completion=completion)
                if tasks:
                    await asyncio.wait_for(asyncio.gather(*tasks), 3)
                    tasks.clear()
                    assert set(events) == {"P:g3", "D:g9"}
            assert case.entry.transferred.is_set()
            assert case.destination.cache.group_cache_reservation(case.entry.binding.request).state.value == "READY"
            assert len(copied) == sum(len(plan.copies) for plan in case.plans)
            assert all(not runtime.gate.permits for runtime in runtimes)
            assert case.destination.node.registry == case.nodes[1].registry
        finally:
            finish.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            for observer in observers:
                await observer.close()
            for runtime in runtimes:
                runtime.close()
    asyncio.run(run())


def test_busy_receiver_returns_source_permit_and_defers_before_installation(monkeypatch):
    async def run():
        case = handoff(128, d_tp=4)
        case.entry.chunk = case.entry.end = 0
        runtimes, observers = runtime_pair(monkeypatch, case, lambda _: None)
        source, destination = runtimes
        case.destination.progress = observers[1]
        service = Service(SimpleNamespace(handoff_timeout=2, capacity=8, prefill=(0,), decode=(0,)),
            SimpleNamespace(node_id="p"), "prefill", None,
            SimpleNamespace(kv_cache_manager=case.p), observers[0])
        plan = case.plans[0]
        busy = Event()
        assert destination.gate.enter(0)
        async def control(operation, **payload):
            result = await case.destination.control(case.nodes[0], case.source_geometry, operation,
                msgspec.to_builtins(dict(binding=plan.binding, **payload)))
            busy.set()
            return result
        task = asyncio.create_task(service.authorize_plan(SimpleNamespace(cancelled=False), control,
                                                         plan, 0, digest(plan)))
        try:
            assert await asyncio.to_thread(busy.wait, 1)
            for _ in range(100):
                if not source.gate.permits:
                    break
                await asyncio.sleep(.001)
            assert not source.gate.permits and not case.entry.reservation.write_authorized
            destination.gate.leave()
            await asyncio.wait_for(task, 2)
            case.entry.pending = None
            await case.destination.release_transfer_permit(case.entry)
            await service.worker("release_write", dict(key="out:" + digest(plan)))
            assert not source.gate.permits and not destination.gate.permits
            events = []
            worker = compute_worker(source, "never-installed", events)
            worker.executor.eplb_config = EplbConfig(maintenance_timeout_seconds=0)
            deferred = []
            worker.executor.defer_eplb_plan = lambda *args: deferred.append(args)
            assert source.gate.reserve("in-use")
            worker._maintain_eplb()
            assert deferred and not events and not source.gate.maintenance
            source.gate.release("in-use")
        finally:
            destination.gate.leave()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for observer in observers:
                await observer.close()
            for runtime in runtimes:
                runtime.close()
    asyncio.run(run())


def test_partial_installation_stops_compute_and_new_dma_but_allows_confirmed_retirement(monkeypatch):
    async def run():
        case = handoff(128, d_tp=4)
        runtimes, observers = runtime_pair(monkeypatch, case, lambda _: None)
        try:
            worker = compute_worker(runtimes[1], "D:g9", [], fail=True)
            with pytest.raises(RuntimeError, match="partial installation"):
                worker._maintain_eplb()
            with pytest.raises(TransferError, match="maintenance"):
                runtimes[1].gate.reserve("later")
            result = msgspec.msgpack.decode(await observers[1].call("reserve_write",
                msgspec.msgpack.encode(dict(key="in:" + "0" * 64))))
            assert result["uncertain"] and "error" in result
            result = msgspec.msgpack.decode(await observers[1].call("release_write",
                msgspec.msgpack.encode(dict(key="in:" + "0" * 64))))
            assert result is None
        finally:
            for observer in observers:
                await observer.close()
            for runtime in runtimes:
                runtime.uncertain = False  # No native writes were submitted in this fixture.
                runtime.close()
    asyncio.run(run())


def test_uncertain_dma_keeps_receiver_permit_and_quarantined_cache_until_stop(monkeypatch):
    async def run():
        case = handoff(128, d_tp=4)
        case.entry.chunk = case.entry.end = 0
        def unknown(_):
            raise TransferError("native completion unknown", uncertain=True)
        runtimes, observers = runtime_pair(monkeypatch, case, unknown)
        source, destination = runtimes
        case.destination.progress = observers[1]
        async def call_worker(operation, payload):
            return destination.handle(operation, payload)
        destination.model.handle = lambda *_: None
        case.destination.core.call_worker = call_worker
        case.destination.core.scheduler = SimpleNamespace(requests={})
        plan = case.plans[0]
        source_service = Service(SimpleNamespace(handoff_timeout=2, capacity=8, prefill=(0,), decode=(0,)),
            SimpleNamespace(node_id="p"), "prefill", None,
            SimpleNamespace(kv_cache_manager=case.p), observers[0])
        async def control(operation, **payload):
            return await case.destination.control(case.nodes[0], case.source_geometry, operation,
                msgspec.to_builtins(dict(binding=plan.binding, **payload)))
        try:
            await source_service.authorize_plan(SimpleNamespace(cancelled=False), control, plan, 0, digest(plan))
            job = await source_service.worker("submit", plan)
            with pytest.raises(TransferError) as error:
                await observers[0].wait(job)
            assert error.value.uncertain
            await control("abort", stopped=False)
            assert case.destination.cache.group_cache_reservation(plan.binding.request).state.value == "QUARANTINED"
            assert destination.gate.permits and not destination.gate.enter(0)
            await control("abort", stopped=True)
            assert not destination.gate.permits
            assert case.destination.cache.group_cache_reservation(plan.binding.request) is None
        finally:
            for observer in observers:
                await observer.close()
            # The fake native owners have stopped; no registered device memory exists here.
            source.jobs.clear()
            source.gate.permits.clear()
            source.uncertain = False
            for runtime in runtimes:
                runtime.close()
    asyncio.run(run())
