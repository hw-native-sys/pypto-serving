# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real Linux file I/O and CPU-backed owner copies; no NPU is initialized."""
"""Basic file, restore and lifecycle coverage using real I/O and CPU owner copies."""

import asyncio
import ctypes as C
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
from tests.unit.serving.memory import _copy_stub

helper = _copy_stub.helper


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
        with pytest.raises(RuntimeError, match="readers"):
            store.close()
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


def test_failed_write_does_not_publish_a_cache_entry(tmp_path, monkeypatch):
    import os
    source = Backing(PAGESIZE)
    store = FileStore(tmp_path, PAGESIZE)
    try:
        monkeypatch.setattr(os, "pwritev", lambda *args: 0)
        entry = store.write("bad", source.views(7), PAGESIZE)
        with pytest.raises(OSError, match="aligned"):
            entry.future.result(3)
        assert store.lookup("bad") is None and store.used == 0
    finally:
        store.close()
        source.close()


class CPUOwner:
    _copy_lock = threading.Lock()

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
        with self._copy_lock:
            result = native.copy_batch(direction, pages)
        self.calls.append((direction, len(pages)))
        return result


def stack(tmp_path, helper, tp=1, *, cache=None, capacity=2 << 20):
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
                             capacity, pool=pool)
    pins = []
    if cache is None:
        tier = CacheTier(capacity, backend, pins.append)
    else:
        cache.tiering = TieredCache(cache, backend, capacity)
        tier = cache.tiering.tier
    try:
        archive = SSDArchive(tier, pool, tmp_path, min(16 << 20, 1_000_000_000))
    except BaseException:
        backend.close()
        pool.close()
        if cache is not None:
            cache.tiering = None
        raise
    if cache is not None:
        cache.tiering.add_store(archive)
    else:
        tier.on_store, tier.on_remove = archive.stored, archive.removed
    return pool, backend, tier, archive, owners, regions, pins, buffers


@pytest.mark.parametrize("grouped", [False, True])
def test_failed_stack_initialization_retires_backend_and_pool(tmp_path, monkeypatch, grouped):
    from tests.unit.serving.test_grouped_prefix_cache import manager
    created, backend_factory = [], SimplerBackend
    def backend(*args, **kwargs):
        value = backend_factory(*args, **kwargs)
        created.append(value)
        return value
    def fail_probe(self):
        raise OSError(22, "injected unsupported direct I/O")
    monkeypatch.setitem(globals(), "SimplerBackend", backend)
    monkeypatch.setattr(FileStore, "_probe", fail_probe)
    cache = manager(slots=5, checkpoint_state=True) if grouped else None
    with pytest.raises(OSError, match="unsupported direct I/O"):
        stack(tmp_path, SimpleNamespace(inject=lambda *args: None), cache=cache)
    assert len(created) == 1 and not created[0].thread.is_alive()
    assert created[0].pool.buffer.closed and not list(tmp_path.iterdir())
    if grouped:
        assert cache.tiering is None


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
def test_eight_components_store_restore_and_release(tmp_path, helper, monkeypatch, tp):
    pool, backend, tier, archive, _, regions, pins, buffers = stack(tmp_path, helper, tp)
    gate, io = threading.Event(), archive.files._io
    def blocked(operation, *args):
        if operation == "write":
            assert gate.wait(5)
        return io(operation, *args)
    monkeypatch.setattr(archive.files, "_io", blocked)
    try:
        names = tuple(backend.geometry["groups"])
        for name in names:
            source = name, 0, 2
            tier.capture((0, name, name.encode()), source, backend.size(source), name,
                         charge=backend.footprint(source))
        eventually(lambda: len(pins) == len(names), lambda: drain(tier, archive))
        assert archive.writers and not tier._evict()  # SSD writers borrow DDR, not HBM.
        gate.set()
        eventually(archive.idle, lambda: drain(tier, archive))
        assert tier.capacity + archive.work_bytes == pool.nbytes
        saved = tuple(tier.pages.values())
        while tier._evict():
            tier._reap_retired()
        for buffer in buffers:
            C.memset(C.addressof(buffer), 0x5B, C.sizeof(buffer))
        for page in saved:
            result = archive.load(page, (page.key[1], 0, 3))
            eventually(result.done, lambda: drain(tier, archive))
            CacheTier.completed(result, page.size)
            for component, layers, blocks, width in backend.geometry["groups"][page.key[1]]:
                address = regions[component][1]
                for layer in range(layers):
                    start = address + (layer * blocks + 3) * width
                    assert C.string_at(start, width) == bytes([0xA7]) * width
                    assert C.string_at(start - width, width) == bytes([0x5B]) * width
        assert archive.files.counters["reads"] == len(names) and tier.error is None
    finally:
        gate.set()
        close_stack(pool, backend, tier, archive)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("tp,partitions", [(4, 1), (1, 4)])
def test_prefix_restore_shared_cancel_and_decode_reservation(tmp_path, helper, monkeypatch, tp, partitions):
    from pypto_serving.pd.adapter import DeepSeekDSpark
    from pypto_serving.serving.sched.scheduler import Request
    from tests.unit.serving.test_grouped_prefix_cache import manager, prompt, _prefill_scheduler, _finish_prefill
    from tests.unit.serving.memory.test_cache_offload import attach, evict
    async def run():
        cache = manager(slots=5, partitions=partitions, checkpoint_state=True)
        pool, backend, tier, archive, *_ = stack(tmp_path, helper, tp, cache=cache, capacity=32 << 20)
        view, session = cache.tiering, attach(cache.tiering)
        gate = threading.Event()
        try:
            tokens = prompt(0, 513)
            boundary = _finish_prefill(_prefill_scheduler(cache, speculative=7),
                                       Request("warm", tokens, max_new_tokens=512))
            await asyncio.to_thread(eventually, lambda: not tier.queued and not tier.inflight and archive.idle())
            hashes = cache.compute_group_block_hashes(tokens)
            evict(cache, ("ori", "csa_state"))
            await view.restore("ddr", hashes, len(tokens), maximum=boundary)
            assert not archive.files.counters["reads"]  # DDR precedes SSD.
            evict(cache)
            while tier._evict():
                tier._reap_retired()
            io = archive.files._io
            def blocked(operation, *args):
                if operation == "read":
                    assert gate.wait(5)
                return io(operation, *args)
            monkeypatch.setattr(archive.files, "_io", blocked)
            assert not view.prepare("cancelled", hashes, len(tokens), maximum=boundary)
            assert not view.prepare("survivor", hashes, len(tokens), maximum=boundary)
            pages = len(view.loads)
            view.cancel("cancelled")
            gate.set()
            await view.restore("survivor", hashes, len(tokens), maximum=boundary)
            assert archive.files.counters["reads"] == pages
            assert cache.acquire_group_prefix_blocks("use", hashes, max_cache_hit_tokens=boundary)[1] == boundary
            cache.release_all_group_requests("use")
            groups = tuple(DeepSeekDSpark().prefix_spec(tokens, cache).hashes)
            d_hashes = cache.compute_group_block_hashes(tokens, group_names=groups)
            await view.restore("d", d_hashes, len(tokens), maximum=len(tokens), groups=groups)
            reservation = cache.reserve_group_cache("d", "d", len(tokens) + 131072,
                prompt_token_count=len(tokens), prefix_block_hashes=d_hashes, shareable_group_names=groups)
            assert reservation.prefix_hit_tokens > 0
            assert all(not blocks for name, blocks in reservation.shared_block_ids_by_group.items()
                       if name.endswith("state"))
            cache.release_group_cache("d", confirmed_stopped=True)
        finally:
            gate.set()
            view.cancel("cancelled")
            view.cancel("survivor")
            await session.close()
            assert all(block.ref_cnt == 0 for group in cache._group_pools.values() for block in group.blocks)
            pool.close()
        assert not list(tmp_path.iterdir())
    asyncio.run(asyncio.wait_for(run(), 10))


def test_unknown_work_completion_retains_ranges_and_refuses_owner_stop(tmp_path):
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
        result, = backend.load_host_batch(((1, ("ori", 0, 1), ((0, PAGESIZE),), None),))
        with pytest.raises(RuntimeError, match="unknown"):
            result.result(3)
        with pytest.raises(RuntimeError, match="uncertain"):
            backend.close()
        assert backend.work and "stop" not in calls
    finally:
        # The injected CPU owner never submits DMA; all RPC threads have stopped.
        pool.close()
