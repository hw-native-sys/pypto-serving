# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real Linux file I/O and CPU-backed owner copies; no NPU is initialized."""

import asyncio
import ctypes as C
import json
from mmap import mmap, PAGESIZE
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from pypto_serving.serving.memory.tiering.archive import SSDArchive
from pypto_serving.serving.memory.tiering.backend import SimplerBackend, BATCH_SPANS
from pypto_serving.serving.memory.tiering.cache import TieredCache
from pypto_serving.serving.memory.tiering.manager import CacheTier
from pypto_serving.serving.memory.tiering.pool import host_views
from pypto_serving.serving.memory.tiering.simpler import NativeCopy, Span
from pypto_serving.serving.memory.tiering.ssd import FileStore
from pypto_serving.serving.memory.tiering.trace import OffloadTrace
from pypto_serving.serving.memory.kv_cache import KVCacheRestoreError
from tests.unit.serving.memory._copy_stub import helper as helper  # Actual C++ planner, CPU ACL stubs.


def eventually(check, progress=lambda: None):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        progress()
        if check():
            return
        time.sleep(.002)
    raise AssertionError("Host operation did not finish")


class Backing:
    def __init__(self, size):
        self.buffer = mmap(-1, size)
        self.base = C.addressof(C.c_ubyte.from_buffer(self.buffer))
        self.nbytes = size

    def views(self, value, size=PAGESIZE):
        view, = host_views(self, ((0, size),), writable=True)
        view[:] = bytes([value]) * size
        return (view,)

    def close(self):
        self.buffer.close()


def test_real_file_lru_pins_and_cross_class_capacity(tmp_path):
    source, target = Backing(4 * PAGESIZE), Backing(4 * PAGESIZE)
    store = FileStore(tmp_path, 4 * PAGESIZE)
    try:
        first = store.write("first", source.views(11, 2 * PAGESIZE), 2 * PAGESIZE)
        first.future.result(3)
        lease = store.hold("first")
        second = store.write("second", source.views(22, 2 * PAGESIZE), 2 * PAGESIZE)
        second.future.result(3)
        third = store.write("third", source.views(33, 3 * PAGESIZE), 3 * PAGESIZE)
        eventually(lambda: store.lookup("second") is None)
        assert not third.future.done() and store.lookup("first") is first
        store.release(lease)
        third.future.result(3)
        lease = store.hold("third")
        store.read(lease, target.views(0, 3 * PAGESIZE)).result(3)
        store.release(lease)
        assert target.buffer[:3 * PAGESIZE] == bytes([33]) * (3 * PAGESIZE)
        assert store.used == 3 * PAGESIZE
        assert sum(Path(store.directory.name, f"cache-{i}.data").stat().st_blocks * 512
                   for i in range(len(store.classes))) <= store.capacity
        with pytest.raises(OSError):
            store.read(first, target.views(0, 2 * PAGESIZE))
    finally:
        store.close()
        source.close()
        target.close()
    assert not list(tmp_path.iterdir())


def test_cancel_queued_write_and_refuse_close_with_reader(tmp_path):
    source = Backing(PAGESIZE)
    store = FileStore(tmp_path, PAGESIZE)
    try:
        store.write("hot", source.views(19), 19).future.result(3)
        lease = store.hold("hot")
        queued = store.write("queued", source.views(20), 20)
        assert queued.future.cancel()
        eventually(lambda: not store.contains("queued"))
        with pytest.raises(RuntimeError, match="readers"):
            store.close()
        assert store.lookup("hot") is lease
        store.release(lease)
    finally:
        store.close()
        source.close()
    assert not list(tmp_path.iterdir())


def test_short_write_fails_without_publishing_and_next_write_works(tmp_path, monkeypatch):
    import os
    source = Backing(PAGESIZE)
    store = FileStore(tmp_path, 2 * PAGESIZE)
    original = os.pwritev
    try:
        monkeypatch.setattr(os, "pwritev", lambda *args: 17)
        failed = store.write("bad", source.views(7), 7)
        with pytest.raises(OSError, match="aligned"):
            failed.future.result(3)
        assert store.lookup("bad") is None and store.used == 0
        monkeypatch.setattr(os, "pwritev", original)
        good = store.write("good", source.views(9), 9)
        good.future.result(3)
        assert store.lookup("good") is good
    finally:
        store.close()
        source.close()


class CPUOwner:
    def __init__(self, helper, pool, regions):
        self.helper, self.pool, self.regions = helper, pool, regions
        self.calls = []
        self.parent = SimpleNamespace(close=lambda: None)

    def call(self, operation, payload=None):
        if operation == "stop":
            self.calls.append((operation, 0))
            return
        direction, pages = payload
        native = NativeCopy.__new__(NativeCopy)
        native.thread = threading.get_ident()
        native.closed = native.poisoned = False
        native.pool, native.regions = self.pool, self.regions
        native.commands = (Span * BATCH_SPANS)()
        native.native, native.stream = self.helper, None
        result = native.copy_batch(direction, pages)
        self.calls.append((direction, len(pages)))
        return result


def stack(tmp_path, helper, tp=1, *, cache=None, capacity=2 << 20, trace=None):
    pool = Backing(capacity)
    names = ("ori", "cmp_c128", "cmp_c4", "idx", "hca_state", "csa_state", "csa_inner_state")
    geometry, regions, buffers = {}, {}, []
    partitions = 1 if cache is None else cache.group_partition_count
    for name in names:
        blocks = 128 if cache is None else len(cache._group_pools[name].blocks) + 1
        components = ((name, 2, blocks, 13),) if name != "idx" else (
            ("idx_k", 2, blocks, 128), ("idx_scale", 2, blocks, 4))
        geometry[name] = components
        for component, layers, count, width in components:
            buffer = (C.c_ubyte * (layers * count * width))()
            C.memset(C.addressof(buffer), 0xA7, C.sizeof(buffer))
            regions[component] = ("pin", C.addressof(buffer), C.sizeof(buffer), len(regions))
            buffers.append(buffer)
    helper.inject(0, 0)
    owners = tuple(CPUOwner(helper, pool, regions) for _ in range(tp * partitions))
    backend = SimplerBackend(owners, dict(tp=tp, ranks=len(owners), groups=geometry),
                             capacity, pool=pool, trace=trace)
    pins = []
    if cache is None:
        tier = CacheTier(capacity, backend, pins.append, trace=trace)
    else:
        cache.tiering = TieredCache(cache, backend, capacity, trace=trace)
        tier = cache.tiering.tier
    archive = SSDArchive(tier, pool, tmp_path, min(16 << 20, 1_000_000_000))
    if cache is not None:
        cache.tiering.archive = archive
    return pool, backend, tier, archive, owners, regions, pins, buffers


def drain(tier, archive):
    archive.pump()
    tier.pump()


def close_stack(pool, backend, tier, archive):
    archive.begin_close()
    tier.begin_close()
    eventually(lambda: tier.close_ready() and archive.idle(), lambda: drain(tier, archive))
    archive.close()
    backend.close()
    tier.finish_close()
    assert not backend.records and not backend.extents.allocations and not tier.used
    pool.close()


@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("traced", [False, True])
def test_eight_components_ssd_only_restore_native_batches_and_host_promotion(tmp_path, helper, tp, traced):
    trace = OffloadTrace(tmp_path.parent / (tmp_path.name + "-trace")) if traced else None
    pool, backend, tier, archive, owners, regions, pins, buffers = stack(tmp_path, helper, tp, trace=trace)
    names = tuple(backend.geometry["groups"])
    try:
        for index, name in enumerate(names):
            tier.capture((0, name, bytes([index])), (name, 0, 2), backend.size((name, 0, 2)), name,
                         charge=backend.footprint((name, 0, 2)))
        tier.publish("joint", tuple(tier.pages))
        eventually(lambda: tier.counters["stores"] == 7 and not archive.writers,
                   lambda: drain(tier, archive))
        assert set(pins) == set(names) and archive.units
        assert archive.files.counters["writes"] == 7
        saved = {key: archive.lookup(key) for key in tier.pages}
        while tier._evict():
            tier._reap_retired()
        assert not tier.pages and all(archive.lookup(key) is page for key, page in saved.items())
        for buffer in buffers:
            C.memset(C.addressof(buffer), 0x5B, C.sizeof(buffer))
        jobs = [(page, (key[1], 0, 3)) for key, page in saved.items()]
        with archive.lock:
            futures = [archive.load(page, destination, request="restore") for page, destination in jobs]
        eventually(lambda: all(f.done() for f in futures), lambda: drain(tier, archive))
        for (page, destination), future in zip(jobs, futures):
            CacheTier.completed(future, page.size)
            for component, layers, blocks, width in backend.geometry["groups"][destination[0]]:
                _, address, _, _ = regions[component]
                for layer in range(layers):
                    start = address + (layer * blocks + 3) * width
                    assert C.string_at(start, width) == bytes([0xA7]) * width
                    assert C.string_at(start - width, width) == bytes([0x5B]) * width
                    assert C.string_at(start + width, width) == bytes([0x5B]) * width
        eventually(lambda: all(tier.lookup(key) for key in saved), lambda: drain(tier, archive))
        assert sum(n for owner in owners for direction, n in owner.calls if direction == "store") == 7 * tp
        assert archive.files.counters["reads"] == 7
        assert max(n for owner in owners for direction, n in owner.calls if direction == "load") > 1
        assert tier.error is None
    finally:
        close_stack(pool, backend, tier, archive)
        if trace is not None:
            trace.close()
    if trace is not None:
        rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
        loads = [row for row in rows if row["event"] == "batch" and row["direction"] == "load"]
        assert loads and all(row["requests"] == ["restore"] for row in loads)
        assert sum(row["pages"] for row in rows if row["event"] == "store_reaped") == 7
    assert not list(tmp_path.iterdir())


def test_slow_ssd_write_holds_only_ddr_reader_not_hbm_pin(tmp_path, helper, monkeypatch):
    pool, backend, tier, archive, _, _, pins, buffers = stack(tmp_path, helper)
    gate, entered = threading.Event(), threading.Event()
    original = archive.files._io
    def delayed(operation, entry, views):
        if operation == "write":
            entered.set()
            assert gate.wait(5)
        return original(operation, entry, views)
    monkeypatch.setattr(archive.files, "_io", delayed)
    try:
        source = "ori", 0, 2
        tier.capture((0, "ori", b"slow"), source, backend.size(source), "hbm-pin",
                     charge=backend.footprint(source))
        eventually(entered.is_set, lambda: drain(tier, archive))
        assert pins == ["hbm-pin"] and next(iter(tier.pages.values())).readers == 1
        assert not tier._evict()
        gate.set()
        eventually(lambda: not archive.writers, lambda: drain(tier, archive))
        assert tier._evict()
    finally:
        gate.set()
        close_stack(pool, backend, tier, archive)


@pytest.mark.parametrize("partitions,tp,length", [(1, 4, 513), (4, 1, 8205), (16, 1, 513)])
def test_real_cache_ssd_only_mixed_prefix_waiters_and_d_reservation(tmp_path, helper, partitions, tp, length):
    from tests.unit.serving.test_grouped_prefix_cache import manager, prompt, _prefill_scheduler, _finish_prefill
    from tests.unit.serving.memory.test_cache_offload import evict
    from pypto_serving.serving.sched.scheduler import Request
    from pypto_serving.pd.adapter import DeepSeekDSpark
    cache = manager(slots=5, partitions=partitions, checkpoint_state=True)
    pool, backend, tier, archive, owners, regions, pins, buffers = stack(
        tmp_path, helper, tp, cache=cache, capacity=32 << 20)
    view = cache.tiering
    tokens = prompt(0, length)
    scheduler = _prefill_scheduler(cache, speculative=7)
    schedule = scheduler.schedule
    def progress(**kwargs):
        view.progress()
        time.sleep(.002)
        return schedule(**kwargs)
    scheduler.schedule = progress
    try:
        for group in range(8):
            boundary = _finish_prefill(scheduler, Request(f"warm-{group}", prompt(group, length), max_new_tokens=512))
            eventually(lambda: not tier.queued and not tier.inflight and not archive.writers, view.progress)
        assert archive.units and boundary > 0
        evict(cache, names=("ori", "csa_state"))
        hashes = cache.compute_group_block_hashes(tokens)
        baseline = archive.files.counters["reads"]
        assert not view.prepare("mixed", hashes, length, maximum=boundary)
        eventually(lambda: view.prepare("mixed", hashes, length, maximum=boundary), view.progress)
        assert archive.files.counters["reads"] == baseline  # DDR beats SSD.
        assert cache.acquire_group_prefix_blocks("mixed", hashes, max_cache_hit_tokens=boundary)[1] == boundary
        cache.release_all_group_requests("mixed")
        evict(cache)
        while tier._evict():
            tier._reap_retired()
        assert not tier.pages and archive.units
        assert not view.prepare("cancelled", hashes, length, maximum=boundary)
        assert all(archive.files.lookup(key).readers > 0 for key, load in view.loads.items() if load.disk)
        assert not view.prepare("survivor", hashes, length, maximum=boundary)
        physical_loads = len(view.loads)
        view.cancel("cancelled")
        eventually(lambda: view.prepare("survivor", hashes, length, maximum=boundary), view.progress)
        assert archive.files.counters["reads"] - baseline == physical_loads
        assert cache.acquire_group_prefix_blocks("survivor", hashes, max_cache_hit_tokens=boundary)[1] == boundary
        cache.release_all_group_requests("survivor")
        evict(cache)
        groups = tuple(DeepSeekDSpark().prefix_spec(tokens, cache).hashes)
        d_hashes = cache.compute_group_block_hashes(tokens, group_names=groups)
        eventually(lambda: view.prepare("d", d_hashes, length, maximum=length, groups=groups), view.progress)
        result = cache.reserve_group_cache("d", "d", length + 131072, prompt_token_count=length,
            prefix_block_hashes=d_hashes, shareable_group_names=groups)
        assert result.prefix_hit_tokens > 0
        assert all(not table for name, table in result.shared_block_ids_by_group.items() if name.endswith("state"))
        cache.release_group_cache("d", confirmed_stopped=True)
        # Losing a required lower-tier state member removes the full replay candidate.
        evict(cache)
        eventually(lambda: not tier.inflight and not archive.writers and archive.idle(), view.progress)
        while tier._evict():
            tier._reap_retired()
        identity, keys = next(iter(archive.units.items()))
        state = next(key for key in keys if key[1].endswith("state"))
        with archive.files.lock:
            record = archive.files.entries.pop(state)
            record.ready = False
        archive.removed(state)
        assert identity not in archive.units
        assert not view.pending and not view.loads and tier.error is None
    finally:
        view.begin_close()
        eventually(view.close_ready, view.progress)
        view.close_storage()
        tier.finish_close()
        assert not backend.records and not backend.work and not backend.extents.allocations
        assert all(block.ref_cnt == 0 for group in cache._group_pools.values() for block in group.blocks)
        pool.close()
    assert not list(tmp_path.iterdir())



def test_failed_background_promotion_keeps_restored_hbm_and_ssd_healthy(tmp_path, helper, monkeypatch):
    pool, backend, tier, archive, _, _, _, buffers = stack(tmp_path, helper)
    try:
        source = "ori", 0, 2
        tier.capture((0, "ori", b"warm-failure"), source, backend.size(source), "pin",
                     charge=backend.footprint(source))
        eventually(lambda: tier.counters["stores"] == 1 and not archive.writers,
                   lambda: drain(tier, archive))
        page, = tier.pages.values()
        while tier._evict():
            tier._reap_retired()
        def fail(*args):
            raise OSError("injected Host copy failure")
        monkeypatch.setattr(archive, "_copy_host", fail)
        result = archive.load(page, ("ori", 0, 3))
        eventually(result.done, lambda: drain(tier, archive))
        CacheTier.completed(result, page.size)
        eventually(lambda: archive.idle() and not tier.inflight, lambda: drain(tier, archive))
        assert not tier.pages and tier.error is None and archive.lookup(page.key) is page
    finally:
        close_stack(pool, backend, tier, archive)


def test_unknown_work_completion_retains_ranges_and_refuses_owner_stop(tmp_path, helper):
    pool = Backing(16 * PAGESIZE)
    calls = []
    class FailingOwner:
        parent = SimpleNamespace(close=lambda: None)
        def call(self, operation, payload=None):
            calls.append(operation)
            if operation == "copy_batch":
                raise RuntimeError("native completion unknown")
    backend = SimplerBackend((FailingOwner(),), dict(tp=1, groups={"ori": (("ori", 1, 4, 13),)}),
                             pool.nbytes, pool=pool)
    try:
        with pytest.raises(ValueError, match="in flight"):
            backend.load_host_batch(((1, ("ori", 0, 1), ((0, PAGESIZE),), None),
                                     (1, ("ori", 0, 2), ((PAGESIZE, PAGESIZE),), None)))
        assert not backend.work and not calls
        result, = backend.load_host_batch(((1, ("ori", 0, 1), ((0, PAGESIZE),), None),))
        with pytest.raises(RuntimeError, match="unknown"):
            result.result(3)
        with pytest.raises(RuntimeError, match="uncertain"):
            backend.close()
        assert backend.work and "stop" not in calls
    finally:
        # The injected CPU owner never submits DMA; all RPC threads have stopped.
        pool.close()



def test_actual_simpler_host_pool_direct_file_roundtrip(tmp_path):
    from pypto_serving.serving.memory.tiering.pool import allocate_pool
    pool = allocate_pool(4 << 20)
    store = FileStore(tmp_path, 1 << 20)
    try:
        source, target = host_views(pool, ((0, 90112), (131072, 90112)), writable=True)
        source[:] = bytes([71]) * len(source)
        target[:] = bytes(len(target))
        store.write("idx", (source.toreadonly(),), 88704).future.result(3)
        entry = store.hold("idx")
        store.read(entry, (target,)).result(3)
        store.release(entry)
        assert source == target and entry.valid_bytes == 88704
    finally:
        store.close()
        pool.close()


@pytest.mark.parametrize("outcome", ["success", "file_error", "cancel"])
def test_ssd_events_progress_without_model_ticks(tmp_path, helper, monkeypatch, outcome):
    from tests.unit.serving.test_grouped_prefix_cache import manager, prompt, _prefill_scheduler, _finish_prefill
    from tests.unit.serving.memory.test_cache_offload import attach, evict
    from pypto_serving.serving.sched.scheduler import Request
    async def run():
        async def until(check):
            while not check():
                await asyncio.sleep(.001)
        cache = manager(slots=5, partitions=1, checkpoint_state=True)
        pool, backend, tier, archive, owners, _, _, _ = stack(tmp_path, helper, cache=cache)
        view, session = cache.tiering, attach(cache.tiering)
        model = asyncio.get_running_loop().create_future()
        gate, entered, failed = threading.Event(), threading.Event(), set()
        task = None
        try:
            tokens = prompt(0, 513)
            boundary = _finish_prefill(_prefill_scheduler(cache, speculative=7),
                                       Request("warm", tokens, max_new_tokens=512))
            await asyncio.wait_for(until(lambda: not tier.queued and not tier.inflight
                                               and not archive.writers), 10)
            assert all(not page.readers for page in tier.pages.values()) and not model.done()
            evict(cache)
            while tier._evict():
                tier._reap_retired()
            original = archive.files._io
            def io(operation, entry, views):
                if operation == "read" and outcome == "file_error":
                    failed.add(entry.key)
                    raise OSError("injected SSD read error")
                if operation == "read" and outcome == "cancel":
                    entered.set()
                    assert gate.wait(10)
                return original(operation, entry, views)
            monkeypatch.setattr(archive.files, "_io", io)
            hashes = cache.compute_group_block_hashes(tokens)
            task = asyncio.create_task(view.restore("restore", hashes, len(tokens), maximum=boundary))
            if outcome == "cancel":
                await asyncio.wait_for(until(entered.is_set), 10)
                view.cancel("restore")
                with pytest.raises(ValueError, match="cancelled"):
                    await asyncio.wait_for(task, 10)
                gate.set()
            elif outcome == "file_error":
                with pytest.raises(KVCacheRestoreError):
                    await asyncio.wait_for(task, 10)
            else:
                await asyncio.wait_for(task, 10)
                assert cache.acquire_group_prefix_blocks("use", hashes, max_cache_hit_tokens=boundary)[1] == boundary
                cache.release_all_group_requests("use")
            await asyncio.wait_for(until(lambda: not view.loads and archive.idle() and not tier.inflight), 10)
            assert not view.pending and tier.error is None and session.core._running and not model.done()
            if failed:
                await asyncio.wait_for(until(lambda: all(key not in archive.pages for key in failed)), 10)
                assert not any(op == "load" for owner in owners for op, _ in owner.calls)
        finally:
            gate.set()
            view.cancel("restore")
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await session.close()
            pool.close()
            model.cancel()
    asyncio.run(run())


def test_failed_disk_retirement_reports_error_instead_of_waiting_forever(tmp_path, monkeypatch):
    source = Backing(PAGESIZE)
    store = FileStore(tmp_path, PAGESIZE)
    try:
        store.write("old", source.views(1), 1).future.result(3)
        def fail(*args):
            raise OSError("injected disk retirement failure")
        monkeypatch.setattr(store, "_punch", fail)
        entry = store.write("new", source.views(2), 2)
        with pytest.raises(OSError, match="retirement"):
            entry.future.result(3)
        with pytest.raises(OSError, match="unavailable"):
            store.write("later", source.views(3), 3)
    finally:
        with pytest.raises(OSError, match="retirement"):
            store.close()
        source.close()
    assert not list(tmp_path.iterdir())
