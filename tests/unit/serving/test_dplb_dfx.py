# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DPLB diagnostics preserve decisions, async ownership and live-output accounting."""

import asyncio
import json
import queue
import threading
from pathlib import Path

import pytest

from pypto_serving.cli.main import _build_dplb_dfx_config, build_parser
from pypto_serving.observability.dplb_dfx import DplbDfxConfig, DplbDfxRecorder, _Sink, command_load
from pypto_serving.observability.dplb_dfx_report import analyze_capture
from pypto_serving.observability.logger import InMemoryStatLogger
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
from pypto_serving.serving.server.ipc import (DecodeRequest, ProfileCommand, StepCommand, StepResult,
                                              decode_command, encode_command)
from pypto_serving.serving.server.serving_worker import WorkerProcess
from pypto_serving.serving.server import serving_worker
from pypto_serving.tools.profile import create_profile_config
from tests.unit.serving.test_dplb import allocate, cache


def recorder(tmp_path, **kwargs):
    result = DplbDfxRecorder(DplbDfxConfig(level="load", **kwargs), tmp_path, "engine", 0)
    result.start("abc123")
    return result


def events(tmp_path):
    return [json.loads(line) for p in tmp_path.rglob("*.jsonl") for line in p.read_text().splitlines()]


def command(step, token=None):
    return StepCommand([], [], [DecodeRequest("request", -1, 100, [], cache_partition=1)],
                       [], step_id=step, dplb_dfx=token)


def test_cli_opt_in_and_unsupported_timing():
    parser = build_parser()
    assert _build_dplb_dfx_config(parser.parse_args(["--model", "/model"])).level == "off"
    with pytest.raises(ValueError, match="requires --profile"):
        _build_dplb_dfx_config(parser.parse_args(["--model", "/model", "--dplb-dfx-level", "load"]))
    with pytest.raises(ValueError, match="semantic device timing"):
        DplbDfxConfig(level="timing")
    with pytest.raises(ValueError, match="positive"):
        DplbDfxConfig(sample_every=0)


def test_disabled_creates_no_files_and_ipc_roundtrip(tmp_path):
    core = ReplicaEngineCore(EngineConfig(), None)
    assert core._dplb_dfx is None
    assert core.kv_cache_manager.dplb_dfx is None
    assert not list(tmp_path.iterdir())
    assert decode_command(encode_command(command(1))) == command(1)


def test_sampling_caps_and_late_outcome_keep_capture(tmp_path):
    dfx = recorder(tmp_path, sample_every=2, max_steps=2)
    assert dfx.sample(10)["step_id"] == 10
    assert dfx.sample(11) is None
    token = dfx.sample(12)
    assert token["capture_id"] == "abc123"
    assert dfx.sample(13) is None
    assert dfx.sample(14) is None
    dfx.stop()
    assert dfx.sample(15) is None
    dfx.outcome(12, status="applied")
    dfx.close()
    rows = events(tmp_path)
    assert rows[-2]["kind"] == "capture_stop"
    assert rows[-1]["step_id"] == 12
    assert rows[-1]["capture_id"] == "abc123"


def test_deadline_stops_admissions_but_retains_results(tmp_path):
    dfx = recorder(tmp_path)
    dfx.sample(1)
    dfx.started -= 100
    dfx.emit("must_not_record")
    assert dfx.sample(2) is None
    dfx.outcome(1, status="applied")
    dfx.close()
    assert "must_not_record" not in [e["kind"] for e in events(tmp_path)]
    assert any(e["kind"] == "step_outcome" for e in events(tmp_path))


def test_queue_pressure_does_not_block_and_reports_loss(tmp_path, monkeypatch):
    entered, unblock = threading.Event(), threading.Event()
    real_open = Path.open

    class SlowFile:
        def __init__(self, file):
            self.file = file
        def write(self, data):
            entered.set()
            assert unblock.wait(5)
            return self.file.write(data)
        def flush(self):
            self.file.flush()
        def close(self):
            self.file.close()

    def slow_open(path, *args, **kwargs):
        file = real_open(path, *args, **kwargs)
        return SlowFile(file) if path.suffix == ".jsonl" else file

    monkeypatch.setattr(Path, "open", slow_open)
    sink = _Sink(tmp_path / "data.jsonl", DplbDfxConfig(queue_size=1))
    sink.put({"n": 0})
    assert entered.wait(5)
    for n in range(20):
        sink.put({"n": n})
    assert sink.queue.qsize() == 1
    assert sink.dropped == 19
    assert not sink.close(timeout=0)
    unblock.set()
    assert sink.close(timeout=5)
    assert json.loads((tmp_path / "data.manifest.json").read_text())["dropped"] == 19


def test_disk_failure_does_not_raise_into_serving(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x")
    sink = _Sink(blocked / "data.jsonl", DplbDfxConfig())
    assert sink.done.wait(5)
    sink.put({"event": "after failure"})
    assert sink.error
    assert sink.dropped == 1


def test_output_size_limit_is_explicit(tmp_path):
    sink = _Sink(tmp_path / "data.jsonl", DplbDfxConfig(max_bytes=20, max_event_bytes=16))
    sink.put({"too_large": "x" * 50})
    for i in range(5):
        sink.put({"n": i})
    assert sink.close(5)
    assert sink.bytes_written <= 20
    assert sink.dropped > 0


def test_recording_preserves_choices_lengths_and_epochs(tmp_path):
    plain, observed = cache(), cache()
    dfx = recorder(tmp_path)
    observed.dplb_dfx = dfx
    for manager in (plain, observed):
        manager.ensure_group_blocks("long", 32, load_tokens=2048)
        allocate(manager, "short", 32)
        allocate(manager, "incoming", 1536)
        manager.release_all_group_requests("long")
        allocate(manager, "long", 64)
    assert observed._group_request_partitions == plain._group_request_partitions
    assert observed._group_request_tokens == plain._group_request_tokens
    dfx.close()
    rows = events(tmp_path)
    admissions = [e for e in rows if e["kind"] == "ownership" and e["action"] == "admit"]
    assert [e["admission_epoch"] for e in admissions if e["request_id"] == "long"] == [1, 2]
    assert admissions[0]["logical_tokens"] == 2048
    assert any(e["kind"] == "placement_rank" and e["scores"] for e in rows)


def test_prefix_observation_uses_existing_lookup_and_longest_hit(tmp_path):
    manager = cache()
    dfx = recorder(tmp_path)
    manager.dplb_dfx = dfx
    for part, length in ((0, 128), (1, 256)):
        allocate(manager, "seed", length, part)
        hashes = manager.compute_group_block_hashes(list(range(length)))
        manager.cache_group_blocks("seed", hashes, length, {})
        manager.release_all_group_requests("seed")
    _, hit, part = manager.acquire_group_prefix_blocks(
        "new", manager.compute_group_block_hashes(list(range(256))), max_cache_hit_tokens=256)
    assert (hit, part) == (256, 1)
    dfx.close()
    lookup = next(e for e in events(tmp_path) if e["kind"] == "prefix_lookup")
    assert lookup["hits"] == {"0": 128, "1": 256}


def test_two_slots_late_reclaim_and_slot_reuse(tmp_path):
    dfx = recorder(tmp_path, sample_every=1)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._dplb_dfx = dfx
    worker._dfx_pending = {}
    worker._dfx_slot_generations = {}
    for step, slot in ((1, 1), (2, 0), (3, 1)):
        token = dict(capture_id="abc123", step_id=step, group_count=2, epochs={"request": 1})
        worker._dfx_prepare(command(step, token), slot)
        worker._dfx_phase(step, "worker_command_begin")
    dfx.stop()
    for step in (2, 1, 3):
        worker._dfx_complete(StepResult({"request": [9]}, step_id=step))
    dfx.close()
    rows = [e for e in events(tmp_path) if e["kind"] == "worker_command_reclaimed"]
    assert [(e["step_id"], e["buffer_slot"], e["slot_generation"]) for e in rows] == [
        (2, 0, 1), (1, 1, 1), (3, 1, 2)]
    assert not worker._dfx_pending
    groups = command_load(command(1), 2)
    assert groups[0]["decode_count"] == 0
    assert groups[1]["sum_seq_len"] == 100


def test_topology_maps_logical_groups_to_four_members(tmp_path):
    core = ReplicaEngineCore(EngineConfig(device_ids=tuple(range(8))), None)
    core.kv_cache_manager = cache()
    assert core._dfx_topology() == [
        dict(group=0, tp_ranks=[0, 1, 2, 3], devices=[0, 1, 2, 3]),
        dict(group=1, tp_ranks=[4, 5, 6, 7], devices=[4, 5, 6, 7])]


def test_report_uses_unfinished_multitoken_metrics_and_marks_missing(tmp_path):
    metrics = InMemoryStatLogger("model", engine_indexes=(0,))
    metrics.start_request(0, "unfinished", arrival_monotonic=0, num_prompt_tokens=10)
    metrics.record_output(0, "unfinished", completion_tokens=1, timestamp=1)
    dfx = DplbDfxRecorder(DplbDfxConfig(level="load"), tmp_path, "engine", 0)
    dfx.start("abc123", counters=metrics.snapshot())
    metrics.record_output(0, "unfinished", completion_tokens=4, timestamp=7)
    token = dfx.sample(1)
    dfx.emit("scheduled_step", token=token, step_id=1, groups=command_load(command(1), 2))
    dfx.stop(counters=metrics.snapshot())
    dfx.close()
    summary, trace = analyze_capture(tmp_path / "dplb" / "abc123")
    assert summary["observed_generation_tokens"] == 3
    assert summary["observed_itl_count"] == 3
    assert summary["observed_token_weighted_tpot_ms"] == 2000
    assert summary["host_capture_status"] == "partial"
    assert summary["missing"]["step_outcome"] == 1
    assert summary["device_timing"] == "unavailable"
    assert trace["traceEvents"]
    assert all(isinstance(e["pid"], int) and isinstance(e["tid"], int) for e in trace["traceEvents"])



def test_enabled_control_ack_drains_late_results_and_seals_files(tmp_path, monkeypatch):
    class Profiler:
        active = False
        def start(self):
            self.active = True
        def stop(self):
            self.active = False

    # Retain state across start/stop as the real process-local profiler does.
    profiler = Profiler()
    monkeypatch.setattr(serving_worker, "get_profiler", lambda **kw: profiler)
    config = EngineConfig(dplb_dfx_config=DplbDfxConfig(level="load", sample_every=1),
                          profile_config=create_profile_config(enabled=True, output=tmp_path))
    acks = queue.Queue()
    worker = WorkerProcess(config, None, None, acks)
    core = ReplicaEngineCore(config, None)
    core.kv_cache_manager = cache()
    core.kv_cache_manager.dplb_dfx = core._dplb_dfx
    core.set_stat_logger(InMemoryStatLogger("model", [0]), 0)

    class Input:
        def put(self, raw):
            worker._handle_profile_command(decode_command(raw))
    core._input_queue = Input()
    core._profile_output_queue = acks

    async def run():
        await core.start_profile()
        token = core._dplb_dfx.sample(1)
        token["group_count"] = 2
        core._dplb_dfx.emit("scheduled_step", token=token, step_id=1, groups=command_load(command(1), 2))
        worker._dfx_prepare(command(1, token), 1)
        worker._dfx_phase(1, "worker_command_begin")
        worker._dfx_complete(StepResult({}, step_id=1))
        async def apply_late():
            await asyncio.sleep(0.05)
            core._dplb_dfx.outcome(1, status="applied", token_counts={})
        apply_task = asyncio.create_task(apply_late())
        await core.stop_profile()
        await apply_task
        assert core._dplb_dfx.sink is None
        assert worker._dplb_dfx.sink is None
    asyncio.run(run())
    capture = next((tmp_path / "dplb").iterdir())
    summary, _ = analyze_capture(capture)
    assert summary["host_capture_status"] == "complete"
    assert summary["sampled_steps"] == 1
    assert not summary["missing"]
    assert all(m["closed"] and m["dropped"] == 0 for m in summary["manifests"])



def test_prepare_lane_can_record_before_device_profile_barrier(tmp_path):
    config = EngineConfig(dplb_dfx_config=DplbDfxConfig(level="load"),
                          profile_config=create_profile_config(enabled=True, output=tmp_path))
    worker = WorkerProcess(config, None, None)
    worker._dfx_start_capture(ProfileCommand(True, capture_id="abc123"), boundary="input_control_received")
    token = dict(capture_id="abc123", step_id=1, group_count=2)
    worker._dfx_prepare(command(1, token), 1)
    worker._dplb_dfx.close()
    assert any(e["kind"] == "worker_prepared" for e in events(tmp_path))
    assert events(tmp_path)[0]["boundary"] == "input_control_received"
