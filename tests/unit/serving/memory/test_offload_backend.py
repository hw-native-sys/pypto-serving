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
from pypto_serving.serving.memory.tiering.simpler import CopyService, NativeCopy
from tests.unit.serving.memory.test_tiering import setup


class Owner:
    def __init__(self):
        self.calls = []
        self.parent = SimpleNamespace(close=lambda: None)

    def call(self, operation, payload=None):
        self.calls.append((operation, payload, threading.get_ident()))
        if operation in ("store", "load"):
            return dict(bytes=sum(span[3] * span[4] for span in payload[1]), stopped=True, native_completed=True)


@pytest.mark.parametrize("tp", [1, 4])
def test_index_scale_is_one_job_and_tp_copies_are_charged_to_one_instance_budget(tp):
    owners = tuple(Owner() for _ in range(2 * tp))
    geometry = dict(tp=tp, groups={"idx": (("k", 2, 7, 128), ("scale", 2, 7, 4))})
    backend = SimplerBackend(owners, geometry)
    source = "idx", 1, 3
    assert backend.size(source) == tp * 2 * 132
    assert backend.footprint(source) == tp * PAGESIZE
    try:
        result = backend.store(1, source).result(2)
        assert result == dict(bytes=tp * 264, stopped=True, native_completed=True)
        assert backend.load(1, ("idx", 1, 5)).result(2) == result
        assert not any(owner.calls for owner in owners[:tp])
        for owner in owners[tp:]:
            reserve, store, load = owner.calls
            assert reserve[:2] == ("reserve", (1, PAGESIZE))
            assert store[1][1] == (("k", 384, 0, 128, 2, 896), ("scale", 12, 256, 4, 2, 28))
            assert load[1][1][0] == ("k", 640, 0, 128, 2, 896)
            assert store[2] != threading.get_ident()
        backend.release(1).result(2)
    finally:
        backend.close()
    assert all(owner.calls[-1][0] == "stop" for owner in owners)


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
    native.buffers = {1: (SimpleNamespace(nbytes=16), 100)}
    native.regions = {"k": ("token", 200, 16)}
    native._call = lambda *args: pytest.fail("invalid span reached native DMA")
    with pytest.raises(ValueError):
        native.copy(1, (span,), store=True)


@pytest.mark.parametrize("store", [True, False])
def test_one_pitched_command_preserves_layer_offsets_and_dense_ddr_packing(store):
    native = NativeCopy.__new__(NativeCopy)
    native.thread = threading.get_ident()
    native.closed = native.poisoned = False
    native.buffers = {1: (SimpleNamespace(nbytes=32), 100)}
    native.regions = {"k": ("token", 200, 2048)}
    native.stream = "stream"
    calls = []
    native._call = lambda *args: calls.append(args)
    proof = native.copy(1, (("k", 16, 0, 4, 3, 128),), store=store)
    expected = (100, 4, 216, 128) if store else (216, 128, 100, 4)
    assert calls == [("aclrtMemcpy2dAsync", *expected, 4, 3, 2 if store else 1, "stream"),
                     ("aclrtSynchronizeStream", "stream")]
    assert proof["bytes"] == 12 and proof["stopped"] and proof["native_completed"]


def test_offload_extension_composes_existing_pd_factory_and_hooks():
    factory = lambda context: context

    class Existing:
        def __init__(self, runner):
            self.runner = runner

        def worker_options(self):
            return dict(chip_service_factories={0: factory}, untouched="setting")

        def wrap_dispatch(self, dispatch):
            return ("wrapped", dispatch)

    extension = CacheOffloadExtension("runner", (object(), object()), object(), previous=Existing)
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
    session.core = SimpleNamespace(kv_cache_manager=SimpleNamespace(tiering=cache))
    tier.counters = {}
    if progressing:
        asyncio.run(session.close())
        assert clock[0] > 120 and closed == [True]
        assert session.core.kv_cache_manager.tiering is None
    else:
        with pytest.raises(RuntimeError, match="drain incomplete"):
            asyncio.run(session.close())
        assert not closed and session.core.kv_cache_manager.tiering is cache


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

        def close(self):
            if has_backing:
                raise RuntimeError("backing still live")
            closed.append(True)

    monkeypatch.setattr(simpler.ctypes, "CDLL", lambda _: SimpleNamespace(aclrtSetCurrentContext=Function()))
    monkeypatch.setattr(simpler, "NativeCopy", Provider)
    service = CopyService.__new__(CopyService)
    service.context = service.acl_context = None
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
