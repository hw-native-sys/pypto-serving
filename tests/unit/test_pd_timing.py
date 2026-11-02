# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Diagnostics must keep existing results, wire bytes and uncertainty unchanged."""

import logging
import struct
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.pd.protocol import Copy
from pypto_serving.pd.transfer import Mooncake, Region, TransferError, _send
from pypto_serving.tools import observe


def test_profile_only_phase_measures_wall_and_thread_cpu_without_logging(monkeypatch):
    events, logged = [], []
    wall, cpu = iter((1000000, 4000000)), iter((100000, 600000))
    monkeypatch.setattr(observe.time, "perf_counter_ns", lambda: next(wall))
    monkeypatch.setattr(observe.time, "thread_time_ns", lambda: next(cpu))
    monkeypatch.setattr(observe, "profile_duration", lambda name, **kw: events.append((name, kw)))
    logger = SimpleNamespace(info=lambda *args: logged.append(args))
    with observe.span(logger, "lower", thread_cpu=True, log=False, request="request") as timing:
        pass
    assert timing == {"duration_ms": 3.0, "thread_cpu_ms": 0.5} and not logged
    assert events[0][1]["args"] == {"request": "request", "thread_cpu_ms": 0.5, "outcome": "completed"}


def test_normal_span_does_not_measure_thread_cpu_or_mask_original_errors(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("diagnostic sink failed")
    monkeypatch.setattr(observe, "profile_duration", broken)
    monkeypatch.setattr(observe.time, "thread_time_ns", lambda: pytest.fail("async wall span measured CPU"))
    with pytest.raises(ValueError, match="execution failure"):
        with observe.span(SimpleNamespace(info=broken), "rpc"):
            raise ValueError("execution failure")


def test_owner_encoding_preserves_exact_wire_and_ignores_failed_profiler(monkeypatch):
    monkeypatch.setattr(observe, "profile_duration", lambda *a, **kw: (_ for _ in ()).throw(OSError("closed")))
    packet = dict(sequence=1, boot="boot", operation="write", copies=[Copy("kv", 0, 1, 0, 0, 64)])
    sent = []
    _send(SimpleNamespace(sendall=sent.append), packet)
    payload = msgspec.msgpack.encode(packet)
    assert sent == [struct.pack("!I", len(payload)) + payload]


@pytest.mark.parametrize("failure", ["none", "bounds", "native"])
def test_provider_lowering_and_native_timings_keep_validation_and_failure_semantics(monkeypatch, failure):
    events, writes, logged = [], [], []
    monkeypatch.setattr(observe, "profile_duration", lambda name, **kw: events.append((name, kw)))
    monkeypatch.setattr(logging.getLogger("pypto_serving.pd.transfer"), "info", lambda *a: logged.append(a))
    def write(*args):
        writes.append(args)
        return 1 if failure == "native" else 0
    engine = SimpleNamespace(initialize=lambda *a: 0, get_rpc_port=lambda: 1234,
        register_memory=lambda *a: 0, batch_transfer_sync_write=write)
    provider = Mooncake("host", lambda: engine)
    provider.register("source", 0, "kv", 4096, 1024)
    provider.install((Region("destination", 1, "kv", 1024, {"endpoint": "destination:1", "address": 8192}),))
    copy = Copy("kv", 0, 1, 1000 if failure == "bounds" else 0, 0, 64)
    if failure == "none":
        provider.write("destination", (copy,))
    else:
        with pytest.raises(ValueError if failure == "bounds" else TransferError) as error:
            provider.write("destination", (copy,))
        if failure == "native":
            assert error.value.uncertain and provider.poisoned
    assert not logged  # New detail spans are only exported during profiling.
    assert events[0][0] == "pd.provider.lower" and "thread_cpu_ms" in events[0][1]["args"]
    if failure == "bounds":
        assert not writes and not provider.poisoned and len(events) == 1
    else:
        assert writes == [("destination:1", [4096], [8192], [64])]
        assert events[1][0] == "MooncakeTransferEngine.batch_transfer_sync_write"
        assert events[1][1]["args"] == {"segments": 1, "bytes": 64,
                                        "outcome": "error" if failure == "native" else "completed"}
