# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise native instrumentation without changing publication or ownership."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pypto_serving.model.deepseek_lowlat.profiling import operation
from pypto_serving.tools.profile import configure_profiler, create_profile_config, merge_profile
from test_native_pipeline import accepted_pair, publish, runner


@pytest.fixture
def recording(tmp_path):
    configure_profiler(create_profile_config(enabled=True, output=tmp_path, levels="verbose"))
    yield tmp_path
    configure_profiler(create_profile_config(enabled=False))


def test_real_pipeline_spans_preserve_two_handles_and_capture_before_frame_reuse(recording):
    instance, batch, hosts = runner()
    ids = iter((101, 102))
    handles = []
    def submit(*args, **kwargs):
        identity = next(ids)
        native = SimpleNamespace(_run_id=identity)
        handle = SimpleNamespace(_native_handle=native, _dispatch_id=identity,
                                 result=Mock(side_effect=lambda: setattr(native, "_run_id", None)))
        handles.append(handle)
        return handle
    instance.native.frames.worker.submit = Mock(side_effect=submit)
    first, second = accepted_pair(instance, batch)
    packet = publish(hosts, 0, instance.native.state)
    publish(hosts, 1, packet.state)
    a = instance.reclaim_prepared_decode(first)
    b = instance.reclaim_prepared_decode(second)
    assert len(a.accepted_token_ids[0]) == len(b.accepted_token_ids[0]) == 12
    assert instance.native.frames.idle()
    assert all(h.result.call_count == 1 for h in handles)
    merge_profile()
    spans = [e for e in json.loads((recording/"trace.json").read_text())["traceEvents"] if e.get("ph") == "X"]
    submitted = [e for e in spans if e["name"] == "lowlat.native.submit"]
    assert [e["args"]["runtime_run_id"] for e in submitted] == [101, 102]
    waits = [e for e in spans if e["name"] == "lowlat.native.wait"]
    assert [e["args"]["runtime_run_id"] for e in waits] == [101, 102]
    captures = [e for e in spans if e["name"] == "lowlat.native.capture"]
    assert [e["args"]["emitted"] for e in captures] == [12, 12]
    assert all(e["args"]["advances"] == [6, 6] for e in captures)
    retired = [e for e in spans if e["name"] == "lowlat.native.retire"]
    assert [e["args"]["reservation_id"] for e in retired] == [1, 2]
    assert all(e["args"]["success"] is True for e in spans)


def test_failed_wait_still_retains_accepted_handle_and_poison_state(recording):
    instance, batch, _ = runner()
    pending = instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    pending.handle.result.side_effect = RuntimeError("failed wait")
    with pytest.raises(RuntimeError, match="failed wait"):
        instance.reclaim_prepared_decode(pending)
    assert pending in instance.state.pending_handles and instance.native.frames.poisoned
    merge_profile()
    events = json.loads((recording/"trace.json").read_text())["traceEvents"]
    assert any(e.get("args", {}).get("success") is False for e in events)


def test_disabled_instrumentation_does_not_inspect_owner_or_arguments(monkeypatch):
    monkeypatch.setattr("pypto_serving.model.deepseek_lowlat.profiling.is_enabled", lambda *a: False)
    class Opaque:
        def __getattribute__(self, key):
            raise AssertionError("unexpected profiling inspection")
    token = object()
    @operation("test")
    def f(owner, value):
        return value
    assert f(Opaque(), token) is token
