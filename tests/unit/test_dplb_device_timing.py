# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device timing joins preserve Serving identity and incomplete coverage."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import json
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import DecodeResult, DispatchInvocation
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor
from pypto_serving.model.common.runner import l3_dispatch
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin
from pypto_serving.model.deepseek_v41 import npu_runner as v41_runner
from pypto_serving.model.deepseek_v41.npu_executor import DeepSeekV41PyptoExecutor
from pypto_serving.observability.dplb_device_timing import (
    analyze_invocation,
    build_invocation_manifest,
    main,
    summarize_capture,
)
from pypto_serving.tools.profile import invocation_profile_args


def capture(rank=0, offset=0, **dispatch_fields):
    identity = dict(
        owner_instance_id="aa" * 8,
        chip_rank=rank,
        run_id=2,
        endpoint_dispatch_id=rank + 1,
        task_slot=rank,
        group_index=0,
        group_size=1,
        pipeline_slot=0,
        pipeline_generation=1,
        local_capture_index=31,
        callable_digest="a1" * 32,
    )
    identity.update(dispatch_fields)
    dispatch = {key: value for key, value in identity.items() if key not in ("local_capture_index", "callable_digest")}
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


def test_l3_dispatch_keeps_step_identity_during_next_step_prepare_and_reclaim(monkeypatch):
    dispatch_entered = Event()
    prepare_finished = Event()
    spans = {}

    @contextmanager
    def record_span(name, *, args, **kwargs):
        spans[name] = args
        yield

    monkeypatch.setattr(l3_dispatch, "profile_span", record_span)
    spec = SimpleNamespace(name="decode", compiled=object(), aicpu_thread_num=1, block_dim=None, dispatch_args=())

    class Runner(L3DispatchMixin):
        def _shared_l3_worker(self):
            return SimpleNamespace(submit=lambda *args, **kwargs: SimpleNamespace(result=lambda: None))

        def prepare_decode(self, model, batch, *, buffer_slot):
            return invocation_profile_args()

        def dispatch_prepared_decode(self, model, batch, prepared):
            dispatch_entered.set()
            assert prepare_finished.wait(5)
            return self._submit_l3(spec)

        def reclaim_prepared_decode(self, pending):
            pending.wait()
            return invocation_profile_args()

    runner = Runner()
    runner._init_l3_dispatch(stacked=False)
    executor = SimpleNamespace(_runners={"model": runner})
    model = SimpleNamespace(config=SimpleNamespace(model_id="model"))
    first = DispatchInvocation("engine-a:7", 7, 1, 4)
    second = DispatchInvocation("engine-a:8", 8, 0, 4)
    first_batch = SimpleNamespace(request_ids=[], invocation=first)
    second_batch = SimpleNamespace(request_ids=[], invocation=second)

    with ThreadPoolExecutor(max_workers=1) as pool:
        dispatch = pool.submit(PyptoExecutor.dispatch_prepared_decode, executor, model, first_batch, None)
        try:
            assert dispatch_entered.wait(5)
            prepared = PyptoExecutor.prepare_decode(executor, model, second_batch, buffer_slot=0)
            assert prepared == second.as_profile_args()
        finally:
            prepare_finished.set()
        pending = dispatch.result(timeout=5)
        assert pool.submit(invocation_profile_args).result(timeout=5) == {}

    assert PyptoExecutor.reclaim_prepared_decode(executor, pending) == first.as_profile_args()
    assert invocation_profile_args() == {}
    for name in ("decode.worker_submit", "decode.worker_wait"):
        assert spans[name] == {**first.as_profile_args(), "aicpu_thread_num": 1}


def test_v41_dspark_dispatch_span_has_serving_step(monkeypatch):
    spans = []

    @contextmanager
    def record_span(name, *, args=None, **kwargs):
        if name == "v41.worker.run":
            spans.append(args)
        yield

    monkeypatch.setattr(v41_runner, "profile_span", record_span)
    resources = object.__new__(v41_runner.V41ResidentResources)
    resources.closed = resources.failed = False
    resources.worker = SimpleNamespace(run=lambda *args, **kwargs: None)
    resources.run_config = None
    resources.dspark_programs = {"dspark_markov": SimpleNamespace(compiled=object())}
    resources.dspark_param_names = {"dspark_markov": ()}
    invocation = DispatchInvocation("engine-a:7", 7, 1, 4)

    def dispatch(batch):
        resources.dispatch_dspark("dspark_markov", {})
        return DecodeResult(hidden_states=None, logits=None)

    executor = SimpleNamespace(runners={"v41": SimpleNamespace(run_dspark_decode_batch=dispatch)})
    DeepSeekV41PyptoExecutor.run_dspark_decode_batch(executor, SimpleNamespace(invocation=invocation))
    assert invocation_profile_args() == {}
    assert spans == [invocation.as_profile_args()]


def manifest(tmp_path, *, grouped=False, legacy=False):
    entries = []
    for rank, offset in ((0, 0), (1, 1_000_000_000)):
        dispatch_fields = (
            dict(task_slot=7, group_size=2, group_index=rank, local_capture_index=31 + rank)
            if grouped else {}
        )
        raw, identity = capture(rank, offset, **dispatch_fields)
        directory = tmp_path / f"rank{rank}" / "d31"
        if legacy:
            directory.mkdir(parents=True)
            raw["metadata"].pop("collection")
            filename = str((directory / "chip_swimlane_records.json").relative_to(tmp_path))
            sidecar = directory / "dispatch_identity.json"
        else:
            filename = f"arbitrary-capture-{19-rank}.json"
            sidecar = tmp_path / f"identity-{rank}.json"
        (tmp_path / filename).write_text(json.dumps(raw))
        sidecar.write_text(json.dumps(identity))
        entry = dict(records=filename, identity=identity)
        if not legacy:
            entry["identity_sidecar"] = sidecar.name
        entries.append(entry)
    return build_invocation_manifest(
        DispatchInvocation("engine-a:7", step_id=7, execution_slot=1, pipeline_generation=4),
        scope="attention_block_v1",
        logical_dp_groups={"0": [0, 1]},
        captures=entries,
        semantic={"attention_kinds": ["swa", "hca"]},
    )


@pytest.mark.parametrize("grouped", [False, True])
def test_group_uses_durations_across_unaligned_device_clocks(tmp_path, grouped):
    result = analyze_invocation(manifest(tmp_path, grouped=grouped), tmp_path)
    assert result["status"] == "complete"
    assert result["one_native_dispatch"] is True
    assert result["invocation_id"] == "engine-a:7"
    assert result["serving_step_id"] == 7
    group = result["logical_dp_groups"]["0"]
    assert group["dispatch_envelope_max_us"] == 50
    assert group["dispatch_busy_union_max_us"] == 40


@pytest.mark.parametrize("field", ["invocation_id", "serving_step_id", "execution_slot", "pipeline_generation"])
@pytest.mark.parametrize("invalid", [None, True])
def test_manifest_cli_rejects_missing_or_invalid_serving_identity(tmp_path, monkeypatch, capsys, field, invalid):
    data = manifest(tmp_path)
    if field == "invocation_id":
        del data["prototype_invocation_id"]
    if invalid is None:
        del data[field]
    else:
        data[field] = invalid
    source = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(sys, "argv", ["dplb_device_timing", str(source), "--output", str(output)])

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert field in capsys.readouterr().err
    assert not output.exists()


def test_manifest_accepts_legacy_invocation_alias(tmp_path):
    data = manifest(tmp_path)
    del data["invocation_id"]
    assert analyze_invocation(data, tmp_path)["invocation_id"] == data["prototype_invocation_id"]
    data["invocation_id"] = "wrong-step"
    with pytest.raises(ValueError, match="disagrees"):
        analyze_invocation(data, tmp_path)


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
    assert result["logical_dp_groups"]["0"]["dispatch_envelope_max_us"] is None


@pytest.mark.parametrize(("grouped", "sidecar_field", "native_field", "value"), [
    (False, "task_slot", "task_slot", 9),
    (False, "local_capture_index", "capture_index", 32),
    (False, "group_size", "group_size", 2),
    (True, "task_slot", "task_slot", 9),
    (True, "group_index", "group_index", 0),
])
def test_same_run_still_requires_one_dispatch(tmp_path, grouped, sidecar_field, native_field, value):
    data = manifest(tmp_path, grouped=grouped)
    entry = data["captures"][1]
    entry["identity"][sidecar_field] = value
    path = tmp_path / entry["records"]
    raw = json.loads(path.read_text())
    raw["metadata"]["collection"]["dispatch"][native_field] = value
    path.write_text(json.dumps(raw))
    (tmp_path / entry["identity_sidecar"]).write_text(json.dumps(entry["identity"]))

    summary = analyze_invocation(data, tmp_path)
    assert summary["one_native_run"] is True
    assert summary["one_native_dispatch"] is False
    assert summary["status"] == "partial"
    assert summary["logical_dp_groups"]["0"]["dispatch_busy_union_max_us"] is None


@pytest.mark.parametrize("digest_evidence", ["mixed", "missing", "manifest_only"])
def test_independent_submissions_require_matching_runtime_callable_digests(tmp_path, digest_evidence):
    data = manifest(tmp_path)
    first, second = data["captures"]
    if digest_evidence == "manifest_only":
        del second["identity_sidecar"]
    else:
        if digest_evidence == "mixed":
            second["identity"]["callable_digest"] = "b2" * 32
            for entry, program in ((first, "prefill"), (second, "decode")):
                entry["program_sidecar"] = f"{program}.json"
                entry["program"] = program
                (tmp_path / entry["program_sidecar"]).write_text(json.dumps({"program": program}))
        else:
            del second["identity"]["callable_digest"]
        (tmp_path / second["identity_sidecar"]).write_text(json.dumps(second["identity"]))

    summary = analyze_invocation(data, tmp_path)
    assert all(rank["status"] == "complete" for rank in summary["ranks"].values())
    assert summary["one_native_run"] is True
    assert summary["one_native_dispatch"] is False
    assert summary["status"] == "partial"
    assert summary["logical_dp_groups"]["0"]["dispatch_busy_union_max_us"] is None


def test_native_submit_group_does_not_require_callable_sidecars(tmp_path):
    data = manifest(tmp_path, grouped=True)
    for entry in data["captures"]:
        del entry["identity_sidecar"]
        del entry["identity"]["callable_digest"]
    assert analyze_invocation(data, tmp_path)["status"] == "complete"


def test_legacy_capture_accepts_adjacent_runtime_identity(tmp_path):
    data = manifest(tmp_path, legacy=True)
    summary = analyze_invocation(data, tmp_path)
    assert summary["status"] == "observed"
    assert summary["logical_dp_groups"]["0"]["dispatch_busy_union_max_us"] == 40


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink"])
def test_raw_capture_can_only_be_assigned_to_one_rank(tmp_path, alias):
    data = manifest(tmp_path, legacy=True)
    original_path = tmp_path / data["captures"][0]["records"]
    duplicate = data["captures"][1]
    duplicate_path = tmp_path / duplicate["records"]
    if alias == "same_path":
        duplicate["records"] = data["captures"][0]["records"]
    elif alias == "symlink":
        duplicate_path.unlink()
        duplicate_path.symlink_to(original_path)
    else:
        duplicate_path.unlink()
        duplicate_path.hardlink_to(original_path)

    summary = analyze_invocation(data, tmp_path)
    assert summary["status"] == "partial"
    assert "more than one rank" in summary["ranks"]["1"]["error"]
    assert summary["logical_dp_groups"]["0"]["dispatch_busy_union_max_us"] is None


@pytest.mark.parametrize("use_foreign_sidecar", [False, True])
def test_legacy_capture_rejects_a_foreign_identity(tmp_path, use_foreign_sidecar):
    data = manifest(tmp_path, legacy=True)
    entry = data["captures"][0]
    records = tmp_path / entry["records"]
    sidecar = records.with_name("dispatch_identity.json")
    if use_foreign_sidecar:
        foreign_sidecar = tmp_path / "dispatch_identity.json"
        sidecar.rename(foreign_sidecar)
        entry["identity_sidecar"] = foreign_sidecar.name
    else:
        sidecar.write_text(json.dumps(data["captures"][1]["identity"]))

    summary = analyze_invocation(data, tmp_path)
    assert summary["status"] == "partial"
    assert "sidecar" in summary["ranks"]["0"]["error"]


def test_dispatch_metric_names_do_not_claim_an_attention_breakdown(tmp_path):
    data = manifest(tmp_path)
    data["scope"] = "moe_only"
    summary = analyze_invocation(data, tmp_path)
    assert summary["scope"] == "moe_only"
    assert summary["status"] == "complete"
    group = summary["logical_dp_groups"]["0"]
    assert "attention_busy_union_max_us" not in group
    assert group["dispatch_busy_union_max_us"] == 40
