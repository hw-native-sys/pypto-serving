# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device timing joins preserve Serving identity and incomplete coverage."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import DispatchInvocation
from pypto_serving.observability.dplb_device_timing import (
    analyze_invocation,
    build_invocation_manifest,
    summarize_capture,
)
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin


def capture(rank=0, offset=0):
    identity = dict(
        owner_instance_id="aa" * 8,
        chip_rank=rank,
        run_id=2,
        endpoint_dispatch_id=rank + 1,
        task_slot=0,
        group_index=0,
        group_size=1,
        pipeline_slot=0,
        pipeline_generation=1,
        local_capture_index=31 + rank,
    )
    dispatch = {key: value for key, value in identity.items() if key != "local_capture_index"}
    dispatch.update(status="identified", capture_index=identity["local_capture_index"])
    collection = dict(
        dispatch=dispatch,
        run_epoch=7,
        processing_complete=True,
        metadata_complete=True,
        verdict="published",
        not_received_buffers=0,
        unpublished_loss=0,
        transport_retired=0,
        transport_retired_records=0,
        unattributable_handoffs=0,
        cut_failed_queues=0,
    )
    raw = dict(
        chip_swimlane_level=1,
        metadata=dict(collection=collection, clock_freq_hz=1_000_000),
        aicore_tasks=[
            [0, 123, 0, 100 + offset, 120 + offset, 0, 7],
            [1, 124, 0, 110 + offset, 130 + offset, 0, 7],
            [0, 125, 1, 140 + offset, 150 + offset, 0, 7],
        ],
    )
    return raw, identity


def test_parallel_cores_are_not_added():
    raw, identity = capture()
    result = summarize_capture(raw, identity)
    assert result["busy_union_us"] == 40
    assert result["envelope_us"] == 50


def test_raw_swimlane_uses_sidecar_identity_and_marks_coverage_unknown():
    raw, identity = capture()
    raw["metadata"].pop("collection")
    result = summarize_capture(raw, identity)
    assert result["status"] == "observed"
    assert result["coverage_state"] == "unavailable"
    assert result["busy_union_us"] == 40


@pytest.mark.parametrize("frequency", [None, 0, -1, True])
def test_no_guessed_device_frequency(frequency):
    raw, identity = capture()
    raw["metadata"]["clock_freq_hz"] = frequency
    with pytest.raises(ValueError, match="frequency"):
        summarize_capture(raw, identity)


def test_serving_invocation_is_retained_in_manifest():
    invocation = DispatchInvocation("engine-a:7", step_id=7, execution_slot=1, pipeline_generation=4)
    manifest = build_invocation_manifest(
        invocation,
        scope="attention_block_v1",
        logical_dp_groups={"0": [0, 1]},
        captures=[],
        semantic={"attention_kinds": ["swa", "hca"], "boundary": "attention_block_v1"},
    )
    assert manifest["invocation_id"] == "engine-a:7"
    assert manifest["serving_step_id"] == 7
    assert manifest["execution_slot"] == 1
    assert manifest["pipeline_generation"] == 4
    assert manifest["scope"] == "attention_block_v1"


def test_l3_profile_carries_explicit_attention_semantics():
    args = L3DispatchMixin._semantic_span_args(SimpleNamespace(
        semantic_scope="attention_block_v1",
        attention_kinds=("swa", "hca"),
        layer_ids=(0, 1),
    ))
    assert args == {
        "semantic_scope": "attention_block_v1",
        "attention_kinds": ["swa", "hca"],
        "layer_ids": [0, 1],
    }


def manifest(tmp_path):
    entries = []
    for rank, offset in ((0, 0), (1, 1_000_000_000)):
        raw, identity = capture(rank, offset)
        filename = f"arbitrary-capture-{19-rank}.json"
        (tmp_path / filename).write_text(json.dumps(raw))
        entries.append(dict(records=filename, identity=identity))
    return build_invocation_manifest(
        DispatchInvocation("engine-a:7", step_id=7, execution_slot=1, pipeline_generation=4),
        scope="attention_block_v1",
        logical_dp_groups={"0": [0, 1]},
        captures=entries,
        semantic={"attention_kinds": ["swa", "hca"]},
    )


def test_group_uses_durations_across_unaligned_device_clocks(tmp_path):
    result = analyze_invocation(manifest(tmp_path), tmp_path)
    assert result["status"] == "complete"
    assert result["invocation_id"] == "engine-a:7"
    assert result["serving_step_id"] == 7
    group = result["logical_dp_groups"]["0"]
    assert group["attention_envelope_max_us"] == 50
    assert group["attention_busy_union_max_us"] == 40


@pytest.mark.parametrize("failure", ["missing", "duplicate", "different_run", "unidentified", "extra_rank"])
def test_group_never_hides_incomplete_or_mixed_identity(tmp_path, failure):
    data = manifest(tmp_path)
    if failure == "missing":
        data["captures"].pop()
    elif failure == "duplicate":
        data["captures"].append(deepcopy(data["captures"][0]))
    elif failure == "different_run":
        data["captures"][1]["identity"]["run_id"] = 3
        path = tmp_path / data["captures"][1]["records"]
        raw = json.loads(path.read_text())
        raw["metadata"]["collection"]["dispatch"]["run_id"] = 3
        path.write_text(json.dumps(raw))
    elif failure == "unidentified":
        path = tmp_path / data["captures"][1]["records"]
        raw = json.loads(path.read_text())
        raw["metadata"]["collection"]["dispatch"]["status"] = "unidentified"
        path.write_text(json.dumps(raw))
    else:
        entry = deepcopy(data["captures"][0])
        entry["identity"]["chip_rank"] = 9
        data["captures"].append(entry)
    result = analyze_invocation(data, tmp_path)
    assert result["status"] == "partial"
    assert result["logical_dp_groups"]["0"]["attention_envelope_max_us"] is None
