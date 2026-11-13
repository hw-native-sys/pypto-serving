# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Trace coverage without changing cache or native-completion semantics."""

import json
from mmap import PAGESIZE
from queue import Full

import pytest

from pypto_serving.serving.memory.tiering.backend import SimplerBackend
from pypto_serving.serving.memory.tiering.trace import OffloadTrace
from tests.unit.serving.memory.test_cache_offload import evict, fixture, prompt, warm
from tests.unit.serving.memory.test_offload_backend import Owner


def records(trace):
    trace.close()
    assert not trace.thread.is_alive()
    return [json.loads(line) for line in trace.path.read_text().splitlines()]


def test_trace_is_opt_in_and_overflow_is_reported_without_waiting(tmp_path, monkeypatch):
    monkeypatch.delenv("PYPTO_SERVING_KV_OFFLOAD_TRACE_DIR", raising=False)
    assert OffloadTrace.from_env() is None
    trace = OffloadTrace(tmp_path)
    trace.emit("phase_probe", request_id="small")
    with monkeypatch.context() as patch:
        patch.setattr(trace.queue, "put_nowait", lambda _: (_ for _ in ()).throw(Full()))
        trace.emit("dropped")
    trace.emit("phase_probe", request_id="full")
    rows = records(trace)
    assert [r["request_id"] for r in rows if r["event"] == "phase_probe"] == ["small", "full"]
    assert len({r["session"] for r in rows}) == 1
    assert rows[-1]["event"] == "end" and rows[-1]["dropped_events"] == 1
    assert not rows[-1]["complete"]


@pytest.mark.parametrize("missing_diagnostics", [False, True])
def test_batch_trace_preserves_native_proof_and_tp_byte_accounting(tmp_path, missing_diagnostics):
    trace = OffloadTrace(tmp_path)
    owners = (Owner(), Owner())
    if missing_diagnostics:
        for owner in owners:
            original = owner.call
            def call(operation, payload=None, original=original):
                result = original(operation, payload)
                if operation == "copy_batch":
                    del result["prepare_start_ns"]
                return result
            owner.call = call
    backend = SimplerBackend(owners, dict(tp=2, groups={"idx": (("k", 2, 7, 128),
        ("scale", 2, 7, 4))}), 8*PAGESIZE, trace=trace)
    try:
        assert backend.reserve(1, ("idx", 0, 0))
        stored = backend.store(1, ("idx", 0, 0)).result(2)
        loaded = backend.load(1, ("idx", 0, 1), request="request").result(2)
        assert stored == loaded == dict(bytes=528, stopped=True, native_completed=True)
        assert not backend.records[1].uncertain
        backend.release(1).result(2)
    finally:
        backend.close()
    rows = records(trace)
    assert not any(r["event"] == "batch_failed" for r in rows)
    if missing_diagnostics:
        assert any(r["event"] == "diagnostic_failed" for r in rows)
    else:
        loads = [r for r in rows if r["event"] == "batch" and r["direction"] == "load"]
        assert {r["owner"] for r in loads} == {0, 1}
        assert sum(r["bytes"] for r in loads) == loaded["bytes"]
        assert all(r["requests"] == ["request"] and r["groups"] == {"idx": 1} for r in loads)


@pytest.mark.parametrize("unknown", [False, True])
def test_restore_trace_distinguishes_shared_cancel_ready_and_unknown(tmp_path, unknown):
    cache, tier, backend = fixture()
    tokens = prompt(0, 513)
    boundary = warm(cache, tier, tokens)
    evict(cache)
    trace = OffloadTrace(tmp_path)
    tier.trace = tier.tier.trace = trace
    backend.hold = True
    hashes = cache.compute_group_block_hashes(tokens)
    assert not tier.prepare("cancel", hashes, len(tokens), maximum=boundary)
    assert not tier.prepare("keep", hashes, len(tokens), maximum=boundary)
    tier.cancel("cancel")
    for future in backend.waiting:
        future.set_result(dict(bytes=1, stopped=not unknown, native_completed=not unknown))
    if unknown:
        with pytest.raises(RuntimeError, match="proof"):
            tier.progress()
        assert all(load.block.ref_cnt and load.page.readers for load in tier.loads.values())
    else:
        tier.progress()
        assert tier.prepare("keep", hashes, len(tokens), maximum=boundary)
        assert not tier.loads and not tier.pending
    rows = records(trace)
    shared = next(r for r in rows if r["event"] == "restore_start" and r["request_id"] == "keep")
    assert shared["new_pages"] == 0 and shared["shared_with"] == ["cancel"]
    cancelled = next(r for r in rows if r["event"] == "restore_cancelled")
    assert cancelled["unconfirmed_pages"] > 0
    assert any(r["event"] == ("restore_failed" if unknown else "restore_ready") for r in rows)
    assert not unknown or not any(r["event"] == "restore_ready" for r in rows)
