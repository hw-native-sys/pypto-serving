# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner command geometry and native copy guards, no ACL library required."""

from mmap import PAGESIZE
import asyncio
import threading
from types import SimpleNamespace

import pytest

from pypto_serving.model.deepseek_dspark.cache_offload import CacheOffloadExtension
from pypto_serving.serving.memory.tiering.backend import SimplerBackend
from pypto_serving.serving.memory.tiering.integration import Session
from pypto_serving.serving.memory.tiering.simpler import CopyService, NativeCopy, rectangles
from tests.unit.serving.memory.test_tiering import setup


class Owner:
    def __init__(self):
        self.calls = []
        self.parent = SimpleNamespace(close=lambda: None)

    def call(self, operation, payload=None):
        self.calls.append((operation, payload, threading.get_ident()))
        if operation == "copy_batch":
            _, pages = payload
            return dict(pages=tuple(dict(handle=h, bytes=sum(s[3] * s[4] for s in spans),
                                         stopped=True, native_completed=True)
                                    for h, _, _, spans in pages),
                        started_ns=1, submitted_ns=2, completed_ns=3)


@pytest.mark.parametrize("tp", [1, 4])
def test_index_scale_is_one_job_and_tp_copies_are_charged_to_one_instance_budget(tp):
    owners = tuple(Owner() for _ in range(2 * tp))
    geometry = dict(tp=tp, groups={"idx": (("k", 2, 7, 128), ("scale", 2, 7, 4))})
    backend = SimplerBackend(owners, geometry, 16 * PAGESIZE)
    source = "idx", 1, 3
    assert backend.size(source) == tp * 2 * 132
    assert backend.footprint(source) == tp * PAGESIZE
    try:
        assert backend.reserve(1, source)
        result = backend.store(1, source).result(2)
        assert result == dict(bytes=tp * 264, stopped=True, native_completed=True)
        assert backend.load(1, ("idx", 1, 5)).result(2) == result
        assert not any(owner.calls for owner in owners[:tp])
        for owner in owners[tp:]:
            store, load = owner.calls
            assert store[0] == load[0] == "copy_batch"
            assert store[1][0] == "store" and load[1][0] == "load"
            assert store[1][1][0][3] == (("k", 384, 0, 128, 2, 896), ("scale", 12, 256, 4, 2, 28))
            assert load[1][1][0][3][0] == ("k", 640, 0, 128, 2, 896)
            assert store[2] != threading.get_ident()
        backend.release(1).result(2)
    finally:
        backend.close()
    assert all(owner.calls[-1][0] == "stop" for owner in owners)


@pytest.mark.parametrize("failed", [False, True])
def test_close_retains_pool_on_unknown_batch_and_otherwise_stops_once(failed):
    owners = (Owner(), Owner())
    backend = SimplerBackend(owners, dict(tp=1, groups={"kv": (("k", 1, 8, 4),)}), 8 * PAGESIZE)
    if failed:
        original = owners[0].call
        def call(operation, payload=None):
            if operation == "copy_batch":
                raise RuntimeError("owner completion unknown")
            return original(operation, payload)
        owners[0].call = call
    assert backend.reserve(1, ("kv", 0, 0))
    future = backend.store(1, ("kv", 0, 0))
    if failed:
        with pytest.raises(RuntimeError, match="unknown"):
            future.result(2)
        with pytest.raises(RuntimeError, match="uncertain"):
            backend.close()
        assert backend.records and not any(c[0] == "stop" for owner in owners for c in owner.calls)
    else:
        future.result(2)
        backend.close()
        assert not backend.records
        assert all([c[0] for c in owner.calls].count("stop") == 1 for owner in owners)


def test_actual_allocation_alignment_limits_residency_not_native_payload():
    tier, released = setup(PAGESIZE)
    tier.capture("a", 2, 2, "a", charge=PAGESIZE)
    tier.pump()
    tier.pump()
    assert tier.used == PAGESIZE and tier.counters["stored_bytes"] == 2
    tier.capture("b", 2, 2, "b", charge=PAGESIZE)
    for _ in range(4):
        tier.pump()
    assert tier.used == PAGESIZE and tier.lookup("a") is None and tier.lookup("b")
    assert sorted(released) == ["a", "b"]


@pytest.mark.parametrize("span", [("k", -1, 0, 1, 1, 1), ("k", 15, 0, 2, 1, 2),
                                   ("k", 0, 15, 2, 1, 2), ("k", 0, 0, 0, 1, 1),
                                   ("k", 0, 0, 4, 0, 4), ("k", 0, 0, 4, 2, 3),
                                   ("k", 0, 0, 4, 2, 16)])
def test_invalid_spans_never_enter_native_submission(span):
    native = NativeCopy.__new__(NativeCopy)
    native.thread = threading.get_ident()
    native.closed = native.poisoned = False
    native.pool = SimpleNamespace(nbytes=16, base=100)
    native.regions = {"k": ("token", 200, 16)}
    native._call = lambda *args: pytest.fail("invalid span reached native DMA")
    with pytest.raises(ValueError):
        native.copy_batch("store", ((1, 0, 16, (span,)),))


@pytest.mark.parametrize("store", [True, False])
def test_one_pitched_command_preserves_layer_offsets_and_dense_ddr_packing(store):
    native = NativeCopy.__new__(NativeCopy)
    native.thread = threading.get_ident()
    native.closed = native.poisoned = False
    native.pool = SimpleNamespace(nbytes=32, base=100)
    native.device = SimpleNamespace(value=0)
    native.regions = {"k": ("token", 200, 2048)}
    native.stream = "stream"
    calls = []
    def copy(commands, count, kind, attributes, stream, timing):
        calls.extend((c.dst, c.src, c.width, c.height, c.dst_pitch, c.src_pitch) for c in commands)
        timing[:] = [1, 2, 3]
        return 0
    native.native = SimpleNamespace(copy_cache_batch=copy)
    proof = native.copy_batch("store" if store else "load", ((1, 0, 32, (("k", 16, 0, 4, 3, 128),)),))
    assert calls == ([(100, 216, 4, 3, 4, 128)] if store else [(216, 100, 4, 3, 128, 4)])
    assert proof["pages"] == (dict(handle=1, bytes=12, stopped=True, native_completed=True),)


def test_offload_extension_composes_existing_pd_factory_and_hooks():
    factory = lambda context: context

    class Existing:
        def __init__(self, runner):
            self.runner = runner

        def worker_options(self):
            return dict(chip_service_factories={0: factory}, untouched="setting")

        def wrap_dispatch(self, dispatch):
            return ("wrapped", dispatch)

    extension = CacheOffloadExtension("runner", (object(), object()), object(), {}, previous=Existing)
    options = extension.worker_options()
    assert options["untouched"] == "setting"
    assert options["chip_service_factories"][0].args[2] is factory
    assert options["chip_service_factories"][1].args[2] is None
    assert extension.wrap_dispatch("compute") == ("wrapped", "compute")


def test_progress_failure_wakes_clients_without_release_or_fake_completion():
    queue = asyncio.Queue()
    context = SimpleNamespace(queue=queue)
    core = SimpleNamespace(_request_contexts={"r": context})

    def failed():
        raise RuntimeError("native result unknown")

    session = Session.__new__(Session)
    session.core, session.cache = core, SimpleNamespace(progress=failed)
    with pytest.raises(RuntimeError, match="unknown"):
        session.progress()
    assert isinstance(queue.get_nowait(), RuntimeError)
    assert core._request_contexts == {"r": context}


def test_offload_configuration_rejects_prefix_disabled_before_creating_owners(monkeypatch):
    from pypto_serving.serving.memory.tiering import integration
    monkeypatch.setattr(integration, "endpoints", lambda *args: pytest.fail("created an owner endpoint"))
    config = SimpleNamespace(executor_cls="PyptoDeepSeekV4DSparkExecutor", enable_prefix_cache=False,
                             runtime_config=SimpleNamespace(kv_offload_backend="simpler"))
    with pytest.raises(ValueError, match="prefix"):
        Session(SimpleNamespace(config=config))


@pytest.mark.parametrize("progressing", [True, False])
def test_shutdown_bounds_inactivity_without_timing_out_a_progressing_drain(monkeypatch, progressing):
    from pypto_serving.serving.memory.tiering import integration
    clock, calls, closed = [0], [0], []
    monkeypatch.setattr(integration.time, "monotonic", lambda: clock[0])
    tier = SimpleNamespace(used=4, inflight={}, retiring={})

    async def tick(_):
        clock[0] += 70  # The whole drain exceeds 120s; every ACK is prompt.

    monkeypatch.setattr(integration.asyncio, "sleep", tick)

    def ready():
        calls[0] += 1
        if progressing:
            tier.used -= 1
        return tier.used == 0

    cache = SimpleNamespace(tier=tier, loads={}, begin_close=lambda: None, close_ready=ready,
                            backend=SimpleNamespace(close=lambda: closed.append(True)))
    session = Session.__new__(Session)
    session.cache = cache
    session.pool = SimpleNamespace(close=lambda: None)
    session.copy_library = SimpleNamespace(cleanup=lambda: None)
    session.core = SimpleNamespace(kv_cache_manager=SimpleNamespace(tiering=cache))
    tier.counters = {}
    tier.finish_close = lambda: closed.append("metadata")
    if progressing:
        asyncio.run(session.close())
        assert clock[0] > 120 and closed == [True, "metadata"]
        assert session.core.kv_cache_manager.tiering is None
    else:
        with pytest.raises(RuntimeError, match="drain incomplete"):
            asyncio.run(session.close())
        assert not closed and session.core.kv_cache_manager.tiering is cache


@pytest.mark.parametrize("failed", [False, True])
def test_shutdown_forgets_residency_only_after_backend_ack(failed):
    tier, _ = setup()
    tier.capture("page", 2, 2, "source")
    tier.pump()
    tier.pump()

    def close():
        assert tier.pages and tier.used == 2
        if failed:
            raise RuntimeError("owner completion unknown")
        tier.backend.live.clear()

    tier.backend.close = close
    cache = SimpleNamespace(tier=tier, backend=tier.backend, loads={},
                            begin_close=tier.begin_close, close_ready=tier.close_ready)
    session = Session.__new__(Session)
    session.pool = SimpleNamespace(close=lambda: None)
    session.copy_library = SimpleNamespace(cleanup=lambda: None)
    session.cache, session.core = cache, SimpleNamespace(kv_cache_manager=SimpleNamespace(tiering=cache))
    if failed:
        with pytest.raises(RuntimeError, match="unknown"):
            asyncio.run(session.close())
        assert tier.pages and tier.used == 2 and session.core.kv_cache_manager.tiering is cache
        with pytest.raises(RuntimeError, match="quarantined"):
            tier.pump()
    else:
        asyncio.run(session.close())
        assert not tier.pages and tier.used == 0 and session.core.kv_cache_manager.tiering is None


@pytest.mark.parametrize("has_backing", [False, True])
def test_owner_teardown_can_close_an_unused_service_but_keeps_live_backing(monkeypatch, has_backing):
    from pypto_serving.serving.memory.tiering import simpler

    class Function:
        def __call__(self, *args):
            return 0

    closed = []

    class Provider:
        def __init__(self, context):
            pass

        def initialize_pool(self, pool):
            pass

        def close(self):
            if has_backing:
                raise RuntimeError("backing still live")
            closed.append(True)

    monkeypatch.setattr(simpler.ctypes, "CDLL", lambda _: SimpleNamespace(aclrtSetCurrentContext=Function()))
    monkeypatch.setattr(simpler, "NativeCopy", Provider)
    service = CopyService.__new__(CopyService)
    service.context = service.acl_context = service.pool = None
    service.channel = SimpleNamespace(poll=lambda _: False, close=lambda: None)
    service.started, service.close_requested = threading.Event(), threading.Event()
    service.stopped, service.error = False, None
    service.thread = threading.Thread(target=service._run)
    service.thread.start()
    assert service.started.wait(2)
    if has_backing:
        with pytest.raises(RuntimeError, match="safely retired"):
            service.close()
        assert not closed and not service.stopped
    else:
        service.close()
        assert closed == [True] and service.stopped


@pytest.mark.parametrize("copies", [1, 4])
def test_pool_atomic_reservation_fragmentation_and_generation_reuse(copies):
    from pypto_serving.serving.memory.tiering.pool import Extents
    pool = Extents(8 * PAGESIZE + 13)
    first = pool.reserve(1, copies, 1)
    assert len(first) == copies and len(set(first)) == copies
    before = list(pool.free)
    assert pool.reserve(2, 9, 1) is None
    assert pool.free == before and set(pool.allocations) == {1}
    pool.release(1)
    assert pool.free == [(0, 8 * PAGESIZE)]
    assert pool.reserve(2, copies, 1) == first
    with pytest.raises(KeyError):
        pool.release(1)
    pool.release(2)
    a = pool.reserve(3, 4, 1)
    pool.reserve(4, 1, 4 * PAGESIZE)
    pool.release(3)
    assert pool.reserve(5, 1, 5 * PAGESIZE) is None
    assert sum(size for _, size in a) == 4 * PAGESIZE


def test_waiting_store_does_not_hold_load_progress_and_native_users_prevent_reuse():
    owner = Owner()
    original, entered, proceed = owner.call, threading.Event(), threading.Event()
    def call(operation, payload=None):
        if operation == "copy_batch" and payload[0] == "store":
            entered.set()
            assert proceed.wait(2)
        return original(operation, payload)
    owner.call = call
    backend = SimplerBackend((owner,), dict(tp=1, groups={"kv": (("k", 1, 16, 4),)}), 2 * PAGESIZE)
    try:
        assert backend.reserve(1, ("kv", 0, 0))
        store = backend.store(1, ("kv", 0, 0))
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="native users"):
            backend.release(1)
        assert backend.reserve(2, ("kv", 0, 1))
        assert not backend.reserve(3, ("kv", 0, 2))
        proceed.set()
        store.result(2)
        result = backend.load(1, ("kv", 0, 8)).result(2)
        assert result["native_completed"]
        backend.release(1).result(2)
        assert backend.reserve(3, ("kv", 0, 2))
    finally:
        proceed.set()
        backend.close()


@pytest.mark.parametrize("failed", [False, True])
def test_pool_placement_precedes_physical_touch_and_startup_failure_closes_backing(monkeypatch, failed):
    from pypto_serving.serving.memory.tiering import pool
    import simpler.buffer
    calls = []
    buffer = SimpleNamespace(base=4096, nbytes=8192, close=lambda: calls.append("close"))
    monkeypatch.setattr(simpler.buffer, "create_host_shared_buffer", lambda *args: buffer)
    monkeypatch.setattr(pool.Path, "read_text", lambda *args, **kwargs: "Mems_allowed_list:\t0-7\n")
    class Bind:
        def __call__(self, base, size, mode, mask, maximum, flags):
            assert base == 4096 and size == 8192 and mode == 3
            assert mask[0] == 255 and maximum == 9 and flags == 0
            calls.append("policy")
            return -1 if failed else 0
    class Advice:
        def __call__(self, base, size, flag):
            assert base == 4096 and size == 8192 and flag == 14
            calls.append("advice")
            return 0
    monkeypatch.setattr(pool.ctypes, "CDLL", lambda *args, **kwargs:
                        SimpleNamespace(mbind=Bind(), madvise=Advice()))
    monkeypatch.setattr(pool.ctypes, "memset", lambda *args: calls.append("touch"))
    if failed:
        with pytest.raises(OSError, match="interleave"):
            pool.allocate_pool(8192)
        assert calls == ["advice", "policy", "close"]
    else:
        assert pool.allocate_pool(8192) is buffer
        assert calls == ["advice", "policy", "touch"]


def test_copy_rectangles_preserve_all_pages_and_never_cross_allocations():
    entries = [("a", layer*512*32 + page*32, 100000+page*43*32+layer*32, 32)
               for layer in range(43) for page in range(190)]
    entries += [("b", 0, 900000, 4), ("b", 8, 900016, 4)]
    planned = rectangles(entries)
    expanded = sorted((d+i*dp, h+i*hp, w) for d,h,w,n,dp,hp in planned for i in range(n))
    assert expanded == sorted((d,h,w) for name,d,h,w in entries)
    assert len(planned) == 44
    assert all(dp >= w and hp >= w for d,h,w,n,dp,hp in planned)


def test_native_copy_error_cannot_publish_or_retire():
    native = NativeCopy.__new__(NativeCopy)
    native.thread = threading.get_ident()
    native.closed = native.poisoned = False
    native.pool = SimpleNamespace(nbytes=4096, base=100)
    native.regions = {"k": ("pin", 10000, 4096)}
    native.device, native.stream = SimpleNamespace(value=0), None
    native.native = SimpleNamespace(copy_cache_batch=lambda *args: 7)
    with pytest.raises(RuntimeError, match="failed"):
        native.copy_batch("store", ((1, 0, 4096, (("k", 0, 0, 4, 1, 4),)),))
    assert native.poisoned
    with pytest.raises(RuntimeError, match="unavailable"):
        native.close()


@pytest.mark.parametrize("store", [False, True])
def test_fragmented_pages_keep_layer_rows_together(store):
    native = NativeCopy.__new__(NativeCopy)
    native.thread = threading.get_ident()
    native.closed = native.poisoned = False
    native.pool = SimpleNamespace(nbytes=4096, base=1000)
    native.regions = {"k": ("pin", 10000, 1024)}
    native.device, native.stream = SimpleNamespace(value=0), None
    captured = []
    def copy(commands, count, kind, attributes, stream, timing):
        captured.extend((v.dst, v.src, v.width, v.height, v.dst_pitch, v.src_pitch) for v in commands)
        timing[:] = [1, 2, 3]
        return 0
    native.native = SimpleNamespace(copy_cache_batch=copy)
    native.copy_batch("store" if store else "load", tuple(
        (i+1, i*12, 12, (("k", p*4, 0, 4, 3, 64),)) for i, p in enumerate((2, 0, 5))))
    expected = sorted((10000+p*4+j*64, 1000+i*12+j*4, 4)
                      for i, p in enumerate((2, 0, 5)) for j in range(3))
    actual = sorted((s+j*sp, d+j*dp, w) if store else (d+j*dp, s+j*sp, w)
                    for d, s, w, rows, dp, sp in captured for j in range(rows))
    assert len(captured) == 3 and actual == expected


@pytest.mark.parametrize("failed", [False, True])
def test_copy_helper_builds_before_owners_and_cleans_failed_build(monkeypatch, failed):
    from pathlib import Path
    import subprocess
    from pypto_serving.serving.memory.tiering import native

    directories = []
    monkeypatch.setenv("ASCEND_HOME_PATH", "/toolkit")
    def build(command, **kwargs):
        output = Path(command[-1])
        directories.append(output.parent)
        assert command[-2] == "-o" and output.name == "libcache_copy.so"
        assert "-I/toolkit/include" in command and "-lascendcl" in command
        if failed:
            raise subprocess.CalledProcessError(1, command, stderr="compiler failure")
        output.write_bytes(b"library")
    monkeypatch.setattr(native.subprocess, "run", build)
    if failed:
        with pytest.raises(RuntimeError, match="compiler failure"):
            native.build_copy_library()
    else:
        library = native.build_copy_library()
        assert (Path(library.name) / "libcache_copy.so").read_bytes() == b"library"
        library.cleanup()
    assert len(directories) == 1 and not directories[0].exists()
