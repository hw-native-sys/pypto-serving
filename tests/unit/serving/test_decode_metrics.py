# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from types import SimpleNamespace

import pytest

from pypto_serving.observability import decode_metrics
from pypto_serving.observability.decode_metrics import DecodeStepRecorder
from pypto_serving.serving.server.serving_worker import WorkerProcess, _PendingDecodeOutput, _PreparedDecodeWork


def test_disabled_metrics_do_not_create_a_sink_or_read_clock(monkeypatch, tmp_path):
    monkeypatch.delenv("PYPTO_SERVING_DECODE_METRICS_DIR", raising=False)
    monkeypatch.setattr(decode_metrics.time, "perf_counter_ns", lambda: pytest.fail("clock read"))
    assert DecodeStepRecorder.from_env() is None
    assert list(tmp_path.iterdir()) == []


def test_records_raw_accepted_counts_and_one_duration_per_batch(monkeypatch, tmp_path):
    recorder = DecodeStepRecorder(str(tmp_path))
    ticks = iter((1_000_000, 6_000_000))
    monkeypatch.setattr(decode_metrics.time, "perf_counter_ns", lambda: next(ticks))
    recorder.record(step_id=4, started_ns=recorder.start(), request_ids=["a", "b"],
                    new_tokens={"a": [1, 2, 3], "b": [7], "prefill": [8]},
                    num_draft_tokens={"a": 7}, mode="sync", prefill_requests=1)
    row = json.loads(recorder.path.read_text())
    assert row["step_time_ms"] == 5
    assert row["requests"] == [dict(request_id="a", accepted_tokens=3, proposed_drafts=7),
                               dict(request_id="b", accepted_tokens=1, proposed_drafts=0)]
    assert row["prefill_requests"] == 1
    assert row["worker_id"] and row["step_id"] == 4
    recorder.close()
    with pytest.raises(FileExistsError):
        DecodeStepRecorder(str(tmp_path))


def test_sync_metrics_exclude_prefill_duration(monkeypatch, tmp_path):
    recorder = DecodeStepRecorder(str(tmp_path))
    clock = [0]
    monkeypatch.setattr(decode_metrics.time, "perf_counter_ns", lambda: clock[0])
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._decode_metrics = recorder
    worker.model_record = SimpleNamespace(runtime_model=object())
    expected_batch = object()
    def prefill(scheduled, model, tokens):
        clock[0] += 100_000_000
        tokens["p"] = [9]
    def prepare(*args, **kwargs):
        clock[0] += 30_000_000
        return object()
    def decode(*args):
        clock[0] += 5_000_000
        return object()
    def consume(scheduled, result, tokens, *, batch):
        assert batch is expected_batch
        tokens["d"] = [1, 2, 3]
        return {"d": 7}
    def batch(*args, **kwargs):
        clock[0] += 20_000_000
        return expected_batch
    worker.executor = SimpleNamespace(max_prefill_batch_size=None, prepare_decode=prepare,
                                      run_prepared_decode=decode)
    worker._batch_prefill, worker._batch_builder = prefill, batch
    worker._consume_decode_result = consume
    worker._allow_device_sampled_ids = worker._allow_device_topk_sampling = lambda scheduled, *, phase: phase == "decode"
    worker._record_last_tokens = lambda *args: None
    command = SimpleNamespace(step_id=7, prefill_requests=[SimpleNamespace(request_id="p")],
                              decode_requests=[SimpleNamespace(request_id="d")])
    result = worker._execute_step(command, _PreparedDecodeWork(prepared=None, buffer_slot=0))
    row = json.loads(recorder.path.read_text())
    assert result.new_tokens == {"p": [9], "d": [1, 2, 3]}
    assert row["step_time_ms"] == 5
    assert row["start_ns"] == 150_000_000
    assert row["mode"] == "sync_prepared" and row["prefill_requests"] == 1
    recorder.close()


def test_async_metrics_include_the_dispatch_to_reclaim_wait(monkeypatch, tmp_path):
    recorder = DecodeStepRecorder(str(tmp_path))
    monkeypatch.setattr(decode_metrics.time, "perf_counter_ns", lambda: 151_000_000)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._decode_metrics = recorder
    worker.executor = SimpleNamespace(reclaim_prepared_decode=lambda pending: object())
    expected_batch = object()
    def consume(scheduled, result, tokens, *, batch):
        assert batch is expected_batch
        tokens.update(a=[1, 2], b=[3, 4, 5])
        return dict(a=7, b=7)
    worker._consume_decode_result = consume
    worker._record_last_tokens = lambda *args: None
    work = _PendingDecodeOutput(cmd=SimpleNamespace(step_id=8),
        scheduled=(SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")),
        pending=object(), buffer_slot=0, batch=expected_batch, started_ns=1_000_000)
    assert worker._reclaim_pending_decode(work).error is None
    row = json.loads(recorder.path.read_text())
    assert row["step_time_ms"] == 150
    assert row["mode"] == "async" and len(row["requests"]) == 2
    recorder.close()
