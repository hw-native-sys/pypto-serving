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
        ("unflushed", "manifest record accounting mismatch"),
        ("tail_missing", "manifest record accounting mismatch"),
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
    elif fault == "sequence":
        events[-1]["event_seq"] += 1
    elif fault == "unflushed":
        status["queued_bytes"] = 1
    elif fault == "tail_missing":
        status["written"].pop(events.pop()["event"])
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
