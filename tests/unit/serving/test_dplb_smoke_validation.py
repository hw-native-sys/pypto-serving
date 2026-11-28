# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the hardware-smoke verdict against actual Host allocator captures."""

import json
from pathlib import Path
import runpy

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.observability.dplb_dfx import DplbDfxConfig, DplbDfxRecorder
from pypto_serving.serving.memory.kv_cache import KvCacheManager


@pytest.fixture
def smoke():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "manual_dplb_dfx_smoke.py"))


@pytest.fixture
def completed(tmp_path):
    manager = KvCacheManager(enable_dplb=True)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                "kv",
                (0,),
                KVCacheSpec(16, 16, 1),
                256,
                num_partitions=2,
            ),
        ),
        max_batch_size=8,
        primary_num_blocks=256,
    )
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load"), tmp_path)
    manager.dplb_dfx = recorder
    for window in ("first", "second"):
        assert recorder.start(window, topology=[dict(group_id=0), dict(group_id=1)])
        manager.dfx_baseline()
        for name, length in (("long", 2048), ("short", 32)):
            manager.ensure_group_blocks(name, 16, load_tokens=length)
        for name in ("long", "short"):
            manager.release_all_group_requests(name)
        assert recorder.close()
    return tmp_path


def test_smoke_accepts_two_actual_captures(smoke, completed):
    result = smoke["validate_capture"](completed, 2)
    assert result["status"] == "PASS_HOST_CAPTURE_SMOKE"
    assert len(result["captures"]) == 2
    assert all(row["groups"] == [0, 1] and row["terminals"] == 2 for row in result["captures"])


def test_smoke_does_not_claim_prefix_coverage_without_a_hit(smoke, completed):
    with pytest.raises(RuntimeError, match="no actual prefix hit"):
        smoke["validate_capture"](completed, 2, require_prefix_hit=True)


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("loss", "incomplete capture"),
        ("identity", "without an observed terminal"),
        ("load", "did not return to zero"),
        ("sequence", "sequence gaps"),
    ],
)
def test_smoke_rejects_incomplete_or_inconsistent_evidence(smoke, completed, fault, reason):
    status_path = next(completed.glob("dplb/first/*.manifest.json"))
    data_path = next(completed.glob("dplb/first/*.jsonl"))
    status = json.loads(status_path.read_text())
    events = [json.loads(line) for line in data_path.read_text().splitlines()]
    if fault == "loss":
        status.update(complete=False, dropped=dict(queue_full=1))
    elif fault == "identity":
        for event in events:
            if event["event"] == "request_terminal":
                event["request_uid"] = "unrelated"
    elif fault == "load":
        releases = [event for event in events if event.get("action") == "released"]
        releases[-1]["groups"][0]["max_logical_tokens"] = 32
    else:
        events[-1]["event_seq"] += 1
    data_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    status["bytes_written"] = data_path.stat().st_size
    status_path.write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match=reason):
        smoke["validate_capture"](completed, 2)


def test_smoke_server_configuration_preserves_requested_diagnostic_mode(smoke, tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(model=tmp_path, output_dir=tmp_path, ep=8, dplb=False, dfx_level="off")
    command = smoke["server_command"](args, tuple(range(8, 16)), 8000)
    assert command[command.index("--devices") + 1] == "8,9,10,11,12,13,14,15"
    assert command[command.index("--dp") + 1] == "2"
    assert command[command.index("--dplb-dfx-level") + 1] == "off"
    assert "--no-enable-dplb" in command
    assert "--enable-prefix-caching" in command


@pytest.fixture
def scheduled_events(tmp_path):
    from types import SimpleNamespace
    from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
    from pypto_serving.serving.sched.scheduler import Request, ScheduledRequest, SchedulerOutput
    from pypto_serving.serving.server.ipc import StepCommand

    manager = KvCacheManager()
    manager.init_groups((KVCacheGroupSpec("kv", (0,), KVCacheSpec(16, 16, 1), 16,
                                         num_partitions=2),), max_batch_size=4)
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load"), tmp_path)
    manager.dplb_dfx = recorder
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.config = EngineConfig(device_ids=(4, 5, 6, 7))
    core.kv_cache_manager = manager
    core.scheduler = SimpleNamespace(running=[])
    core._dplb_dfx = recorder
    assert recorder.start("m2")
    block_ids = manager.ensure_group_blocks("r", 32, partition=1)
    request = Request("r", [1, 2], max_new_tokens=8)
    scheduled = ScheduledRequest(request, 1, False, num_computed_tokens=32,
                                 num_speculative_tokens=2, cache_partition=1,
                                 block_ids_by_group=block_ids)
    output = SchedulerOutput(scheduled_requests=[scheduled], num_decode_tokens=3)
    command = StepCommand([], [], [], [], step_id=7)
    core._emit_dfx_step_plan(output, command, dispatch_ordinal=3)
    core._emit_dfx_step_outcome(7, output, {"r": [9, 10]}, {"r": 2})
    assert recorder.close()
    return [json.loads(line) for line in recorder.sink.path.read_text().splitlines()], core._dfx_topology()


def test_m2_smoke_requires_scheduled_evidence(smoke, completed):
    with pytest.raises(RuntimeError, match="missing M2"):
        smoke["validate_capture"](completed, 2, require_scheduled_load=True)


def test_m2_validator_accepts_matching_events(smoke, scheduled_events):
    events, topology = scheduled_events
    summary = smoke["validate_scheduled_load"](events, topology)
    assert summary["planned_steps"] == summary["paired_steps"] == 1
    assert summary["worker_accepted_draft_tokens"] == 1


@pytest.mark.parametrize("fault,reason", [
    ("ordinal", "matching submitted step"),
    ("identity", "identity mismatch"),
    ("count", "active count mismatch"),
    ("tokens", "decode token mismatch"),
    ("group", "group aggregate mismatch"),
    ("rows", "unobserved runner rows"),
    ("outcome", "boundary explanation"),
    ("duplicate", "duplicate scheduled step"),
])
def test_m2_validator_rejects_corrupted_evidence(smoke, scheduled_events, fault, reason):
    events, topology = scheduled_events
    plan = next(event for event in events if event["event"] == "step_plan")
    outcome = next(event for event in events if event["event"] == "step_outcome")
    if fault == "ordinal":
        outcome["dispatch_ordinal"] += 1
    elif fault == "identity":
        outcome["requests"][0]["request_uid"] = "wrong"
    elif fault == "count":
        plan["active_count"] += 1
    elif fault == "tokens":
        plan["decode_tokens"] += 1
    elif fault == "group":
        plan["groups"][1]["participating_request_count"] += 1
    elif fault == "rows":
        plan["requests"][0]["row_extent"]["valid_rows"] = 128
    elif fault == "outcome":
        extra = dict(plan, step_id=8, dispatch_ordinal=4)
        events.insert(events.index(outcome), extra)
    elif fault == "duplicate":
        events.insert(events.index(outcome), plan)
    with pytest.raises(RuntimeError, match=reason):
        smoke["validate_scheduled_load"](events, topology)


def test_m2_validator_explains_pending_boundary_steps(smoke, scheduled_events):
    events, topology = scheduled_events
    plan = next(event for event in events if event["event"] == "step_plan")
    events.insert(-1, dict(plan, step_id=8, dispatch_ordinal=4))
    events[-1]["in_flight_step_ids"] = [8]
    result = smoke["validate_scheduled_load"](events, topology)
    assert result["pending_at_capture_end"] == [8]
