# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host decisions and loss semantics; all allocator fixtures are metadata-only."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.observability import dplb_dfx
from pypto_serving.observability.dplb_dfx import DplbDfxConfig, DplbDfxRecorder
from pypto_serving.serving.memory.kv_cache import KVCacheCapacityError, KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


def cache(enabled=True, *, blocks=256, request_limit=None):
    manager = KvCacheManager(enable_dplb=enabled)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                "kv",
                (0,),
                KVCacheSpec(16, 16, 1),
                256,
                num_partitions=2,
                max_requests_per_partition=request_limit,
            ),
        ),
        max_batch_size=8,
        primary_num_blocks=blocks,
    )
    return manager


@pytest.fixture
def capture(tmp_path):
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load"), tmp_path)
    assert recorder.start("first")
    yield recorder
    assert recorder.close()


def records(recorder):
    assert recorder.close()
    return [json.loads(line) for line in recorder.sink.path.read_text().splitlines()]


def manifest(recorder):
    assert recorder.close()
    return json.loads(recorder.sink.path.with_suffix(".manifest.json").read_text())


def test_default_off_does_not_construct_collector_or_observations(monkeypatch):
    from pypto_serving.serving.engine import async_engine

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled capture must not construct diagnostic state")

    monkeypatch.setattr(async_engine, "DplbDfxRecorder", forbidden)
    core = async_engine.ReplicaEngineCore(async_engine.EngineConfig(), SimpleNamespace(eos_token_id=None))
    assert core._dplb_dfx is None
    manager = cache()
    monkeypatch.setattr(manager, "_dfx_groups", forbidden)
    manager.ensure_group_blocks("plain", 32)
    manager.release_all_group_requests("plain")


@pytest.mark.parametrize("enabled", [False, True])
def test_capture_preserves_choices_tables_and_rotation(capture, enabled):
    observed, plain = cache(enabled), cache(enabled)
    observed.dplb_dfx = capture
    for name, length, partition in (
        ("long", 2048, 0),
        ("short", 32, 1),
        ("incoming", 1536, None),
        ("tiny", 16, None),
    ):
        assert observed.ensure_group_blocks(name, length, partition=partition) == plain.ensure_group_blocks(
            name,
            length,
            partition=partition,
        )
        assert observed._group_request_partitions == plain._group_request_partitions
        assert observed._next_group_partition == plain._next_group_partition
    observed.ensure_group_blocks("tiny", 256)
    observed.release_all_group_requests("long")
    events = records(capture)
    decision = next(e for e in events if e["event"] == "placement_commit" and e["request_id"] == "incoming")
    assert decision["reason"] == ("long_logical_load" if enabled else "request_count")
    assert decision["selected_group"] == (1 if enabled else 0)
    assert decision["groups"][0]["max_logical_tokens"] == 2048
    assert decision["groups"][1]["sum_logical_tokens"] == 32
    assert decision["scores"][str(decision["ranked"][0])] == min(decision["scores"].values())
    assert decision["admission_epoch"] == 1
    assert manifest(capture)["complete"]


def test_full_prompt_max_reduction_and_empty_group(capture):
    manager = cache()
    manager.dplb_dfx = capture
    manager.ensure_group_blocks("long", 16, partition=0, load_tokens=2048)
    manager.ensure_group_blocks("short", 16, partition=0, load_tokens=128)
    manager.release_all_group_requests("long", reason="finished_length")
    manager.dfx_terminal("long", "finished_length")
    events = records(capture)
    acquired = next(e for e in events if e["event"] == "owner_change" and e["request_id"] == "long")
    assert acquired["logical_tokens"] == 2048
    assert acquired["allocation_tokens"] == 16
    released = next(e for e in events if e.get("action") == "released")
    assert released["groups"][0]["max_logical_tokens"] == 128
    assert released["groups"][0]["request_count"] == 1
    assert released["groups"][1]["request_count"] == 0


def test_filtered_candidates_explain_capacity_and_request_limit(capture):
    manager = cache(request_limit=1)
    manager.ensure_group_blocks("full", 4096, partition=0)
    manager.dplb_dfx = capture
    manager.dfx_baseline()
    manager.ensure_group_blocks("incoming", 64)
    with pytest.raises(KVCacheCapacityError):
        manager.ensure_group_blocks("blocked", 32)
    events = records(capture)
    decision = next(e for e in events if e["event"] == "placement_commit")
    excluded = decision["groups"][0]
    assert excluded["exclusion_reasons"] == ["request_limit", "kv_capacity:kv"]
    assert excluded["capacity"]["kv"]["required_blocks"] == 4
    assert excluded["capacity"]["kv"]["free_blocks"] == 0
    assert decision["ranked"] == [1]
    assert set(decision["scores"]) == {"1"}
    failure = next(e for e in events if e["event"] == "placement_failed")
    assert not failure["placement_committed"]
    assert not failure["owner_present_on_failure"]
    assert "blocked" not in manager._group_request_tokens


def test_preemption_epoch_and_reused_request_identity(capture):
    manager = cache()
    manager.dplb_dfx = capture
    manager.ensure_group_blocks("request", 32)
    manager.release_all_group_requests("request", reason="preempted")
    manager.ensure_group_blocks("request", 64)
    manager.release_all_group_requests("request")
    manager.ensure_group_blocks("request", 16)
    admitted = [e for e in records(capture) if e["event"] == "placement_commit"]
    assert [e["admission_epoch"] for e in admitted] == [1, 2, 1]
    assert admitted[0]["request_uid"] == admitted[1]["request_uid"]
    assert admitted[2]["request_uid"] != admitted[1]["request_uid"]


def test_scheduler_cancellation_and_rejection_are_terminal(capture):
    manager = cache()
    manager.dplb_dfx = capture
    scheduler = Scheduler(SchedulerConfig(max_seq_len=4096, enable_prefix_cache=False), manager)
    scheduler.add_request(Request("cancel", list(range(32)), max_new_tokens=32))
    scheduler.schedule()
    scheduler.abort_request("cancel")
    with pytest.raises(ValueError):
        scheduler.add_request(Request("reject", list(range(4097)), max_new_tokens=1))
    terminals = [e for e in records(capture) if e["event"] == "request_terminal"]
    assert [(e["request_id"], e["reason"]) for e in terminals] == [
        ("cancel", "finished_aborted"),
        ("reject", "rejected"),
    ]
    assert not manager._group_request_tokens


def test_longest_prefix_and_capacity_fallback_keep_attempt_identity(capture):
    manager = cache()
    hashes = manager.compute_group_block_hashes(list(range(128)))
    manager.ensure_group_blocks("seed", 128, partition=0)
    manager.cache_group_blocks("seed", hashes, 128, {})
    manager.release_all_group_requests("seed")
    manager.ensure_group_blocks("busy", 2048, partition=0)
    manager.dplb_dfx = capture
    _, hit, selected = manager.acquire_group_prefix_blocks(
        "incoming", hashes, max_cache_hit_tokens=128, load_tokens=4096
    )
    assert (hit, selected) == (128, 0)
    with pytest.raises(KVCacheCapacityError):
        manager.ensure_group_blocks("incoming", 4096)
    manager.release_all_group_requests("incoming", reason="prefix_fallback")
    manager.ensure_group_blocks("incoming", 4096)
    events = records(capture)
    commits = [e for e in events if e["event"] == "placement_commit"]
    assert [e["selected_group"] for e in commits] == [0, 1]
    assert [e["admission_epoch"] for e in commits] == [1, 2]
    assert commits[0]["request_uid"] == commits[1]["request_uid"]
    assert commits[0]["reason"] == "longest_prefix"
    assert next(e for e in events if e.get("action") == "released")["reason"] == "prefix_fallback"


def test_shared_prefix_selector_remains_read_only_and_capture_searches_once(capture, monkeypatch):
    manager = cache()
    manager.dplb_dfx = capture
    hashes = manager.compute_group_block_hashes(list(range(128)))
    manager.ensure_group_blocks("producer", 128, partition=0)
    manager.cache_group_blocks("producer", hashes, 128, {})
    manager.release_all_group_requests("producer")
    rotation = manager._next_group_partition
    free = [manager.group_num_free_blocks("kv", group) for group in (0, 1)]

    # DDR offload also uses this selector with a custom lookup. Probing cannot
    # acquire ownership or publish a placement for a nonexistent request.
    _, hit, group = manager.select_group_prefix_blocks(
        hashes, max_cache_hit_tokens=128,
        lookup=lambda pool, owner, value: pool.hash_to_block.get((owner, value)),
    )
    assert (hit, group) == (128, 0)
    assert not manager._group_request_partitions
    assert not manager._group_request_tokens
    assert manager._next_group_partition == rotation
    assert [manager.group_num_free_blocks("kv", group) for group in (0, 1)] == free

    selector = manager.select_group_prefix_blocks
    calls = []

    def select(*args, **kwargs):
        calls.append(1)
        return selector(*args, **kwargs)

    monkeypatch.setattr(manager, "select_group_prefix_blocks", select)
    _, hit, group = manager.acquire_group_prefix_blocks(
        "incoming", hashes, max_cache_hit_tokens=128, load_tokens=256,
    )
    assert (hit, group) == (128, 0)
    assert len(calls) == 1
    events = records(capture)
    assert len([event for event in events if event["event"] == "prefix_lookup"]) == 1
    commits = [event for event in events if event["event"] == "placement_commit"]
    assert [event["request_id"] for event in commits] == ["producer", "incoming"]
    assert commits[-1]["reason"] == "longest_prefix"


def test_reservation_adoption_is_not_another_admission(capture):
    manager = cache()
    manager.dplb_dfx = capture
    manager.reserve_group_cache("lease", "request", 256, prompt_token_count=32)
    manager.authorize_group_cache_write("lease")
    manager.commit_group_cache("lease")
    manager.adopt_group_cache("lease")
    manager.release_group_cache("lease")
    events = records(capture)
    assert len([e for e in events if e["event"] == "placement_commit"]) == 1
    changes = [e for e in events if e["event"] == "reservation_change"]
    assert [e["action"] for e in changes] == ["reserved", "adopted", "released"]
    assert all(e["admission_epoch"] == 1 for e in changes)
    assert not manager._group_request_tokens
    assert not capture._requests


def test_reservation_preemption_keeps_identity(capture):
    manager = cache()
    manager.dplb_dfx = capture
    manager.reserve_group_cache("lease", "request", 64)
    manager.release_group_cache("lease", reason="preempted")
    manager.ensure_group_blocks("request", 64)
    events = records(capture)
    commits = [e for e in events if e["event"] == "placement_commit"]
    assert [e["admission_epoch"] for e in commits] == [1, 2]
    assert commits[0]["request_uid"] == commits[1]["request_uid"]
    assert not any(e["event"] == "request_terminal" for e in events)


def test_unexpected_reservation_failure_rolls_back_without_committed_load(capture, monkeypatch):
    manager = cache()
    manager.dplb_dfx = capture
    take = manager._take_group_block
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected allocation failure")
        return take(*args)

    monkeypatch.setattr(manager, "_take_group_block", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        manager.reserve_group_cache("lease", "request", 128)
    events = records(capture)
    assert not any(e["event"] == "placement_commit" for e in events)
    assert next(e for e in events if e["event"] == "request_terminal")["reason"] == "reservation_rollback"
    assert not manager._group_request_tokens
    assert not manager._group_request_partitions
    assert manager.group_num_free_blocks("kv", 0) == 256


def test_repeated_capture_baseline_and_stale_timer(capture):
    manager = cache()
    manager.ensure_group_blocks("live", 32)
    manager.dplb_dfx = capture
    manager.dfx_baseline()
    old = records(capture)
    assert next(e for e in old if e["event"] == "owner_change")["identity_status"] == "preexisting"
    assert capture.start("second")
    capture.stop(reason="duration", expected_capture_id="first")
    assert capture.capturing
    manager.dfx_baseline()
    assert all(e["capture_id"] == "second" for e in records(capture))


def test_bounded_queue_reports_loss_without_blocking_capture(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    write = dplb_dfx._Sink._write

    def gated(sink):
        entered.set()
        assert release.wait(5)
        write(sink)

    monkeypatch.setattr(dplb_dfx._Sink, "_write", gated)
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load", queue_size=1), tmp_path)
    try:
        assert recorder.start("pressure")
        assert entered.wait(2)
        assert not recorder.emit("extra")
        recorder.stop()
        assert not recorder.sink.done.is_set()
    finally:
        release.set()
        assert recorder.close()
    status = manifest(recorder)
    assert status["dropped"]["queue_full"] >= 1
    assert not status["complete"]
    assert status["queued_bytes"] == 0


@pytest.mark.parametrize(
    "limits,reason",
    [
        ({"queue_bytes": 1}, "queue_bytes"),
        ({"max_event_bytes": 1}, "event_bytes"),
        ({"max_bytes": 1}, "output_bytes"),
    ],
)
def test_byte_limits_are_reported(tmp_path, limits, reason):
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load", **limits), tmp_path)
    recorder.start("bounded")
    status = manifest(recorder)
    assert status["dropped"][reason] >= 1
    assert not status["complete"]
    assert status["bytes_written"] <= recorder.config.max_bytes


def test_disk_failure_does_not_change_allocator(tmp_path):
    (tmp_path / "dplb").write_text("not a directory")
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load"), tmp_path)
    recorder.start("disk")
    assert recorder.sink.done.wait(2)
    assert recorder.sink.error
    manager = cache()
    manager.dplb_dfx = recorder
    manager.ensure_group_blocks("still_served", 32)
    assert manager.group_request_partition("still_served") == 0
    assert not recorder.capturing


def test_request_identity_limit_is_bounded_and_visible(tmp_path):
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load", max_requests=1), tmp_path)
    recorder.start("identities")
    recorder.request_fields("one")
    assert recorder.request_fields("two")["identity_status"] == "request_limit"
    assert len(recorder._requests) == 1
    status = manifest(recorder)
    assert status["untracked_requests"] == 1
    assert not status["complete"]


def test_immediate_writer_failure_does_not_leave_timer(tmp_path, monkeypatch):
    (tmp_path / "dplb").write_text("not a directory")
    sink_class = dplb_dfx._Sink

    def failed_sink(*args):
        sink = sink_class(*args)
        assert sink.done.wait(2)
        return sink

    monkeypatch.setattr(dplb_dfx, "_Sink", failed_sink)
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load"), tmp_path)
    recorder.start("failed")
    assert not recorder.capturing
    assert recorder._timer is None
    assert recorder.close()


def test_deadline_closes_without_traffic(tmp_path, monkeypatch):
    now = [0]
    monkeypatch.setattr(dplb_dfx, "time", SimpleNamespace(monotonic_ns=lambda: now[0]))
    recorder = DplbDfxRecorder(DplbDfxConfig(level="load", max_seconds=1), tmp_path)
    recorder.start("deadline")
    now[0] = 2_000_000_000
    assert not recorder.capturing
    assert manifest(recorder)["stop_reason"] == "duration"


def test_profile_controls_keep_host_capture_and_worker_ack_separate(tmp_path):
    from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
    from pypto_serving.tools.profile import create_profile_config

    config = EngineConfig(
        profile_config=create_profile_config(enabled=True, output=tmp_path),
        dplb_dfx_config=DplbDfxConfig(level="load"),
        device_ids=tuple(range(8)),
    )
    core = ReplicaEngineCore(config, SimpleNamespace(eos_token_id=None))
    core.kv_cache_manager = cache()
    core.kv_cache_manager.dplb_dfx = core._dplb_dfx
    calls = []

    async def acknowledge(active):
        calls.append(active)

    core._set_worker_profile_active = acknowledge

    async def run():
        await core.start_profile("control")
        core.kv_cache_manager.ensure_group_blocks("live", 32)
        await core.start_profile("ignored")
        await core.stop_profile()

    asyncio.run(run())
    events = records(core._dplb_dfx)
    assert calls == [True, True, False]
    assert len([e for e in events if e["event"] == "capture_start"]) == 1
    assert events[0]["topology"][1]["tp_ranks"] == [4, 5, 6, 7]
    assert all(e["capture_id"] == "control" for e in events)
    assert "counters_start" in manifest(core._dplb_dfx)
    assert "counters_end" in manifest(core._dplb_dfx)


def test_profile_failure_seals_capture_and_can_restart(tmp_path):
    from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
    from pypto_serving.tools.profile import create_profile_config

    config = EngineConfig(
        profile_config=create_profile_config(enabled=True, output=tmp_path),
        dplb_dfx_config=DplbDfxConfig(level="load"),
    )
    core = ReplicaEngineCore(config, SimpleNamespace(eos_token_id=None))

    async def fail(_active):
        raise RuntimeError("worker profile failed")

    core._set_worker_profile_active = fail
    with pytest.raises(RuntimeError, match="worker profile failed"):
        asyncio.run(core.start_profile("failed"))
    assert not core._dplb_dfx.capturing
    assert manifest(core._dplb_dfx)["stop_reason"] == "profile_control_failed"
    assert not manifest(core._dplb_dfx)["complete"]
    assert core._dplb_dfx.start("retry")
    assert core._dplb_dfx.close()


def test_replica_budget_is_shared_and_capture_id_is_common(tmp_path):
    from pypto_serving.config.parallel import ParallelConfig
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig
    from pypto_serving.tools.profile import create_profile_config

    class Core:
        def __init__(self, config, **_kwargs):
            self.config = config
            self._dplb_dfx = object()
            self.capture_id = None

        async def start_profile(self, capture_id):
            self.capture_id = capture_id

    config = EngineConfig(
        profile_config=create_profile_config(enabled=True, output=tmp_path),
        dplb_dfx_config=DplbDfxConfig(level="load", max_bytes=5),
        parallel_config=ParallelConfig(data_parallel_size=2, devices=(0, 1)),
    )
    engine = AsyncLLMEngine(
        config, tokenizer=SimpleNamespace(eos_token_id=None, bos_token_id=None), core_factory=Core
    )
    asyncio.run(engine.start_profile())
    assert [core.config.dplb_dfx_config.max_bytes for core in engine._cores] == [2, 2]
    assert len({core.capture_id for core in engine._cores}) == 1
    assert engine._cores[0].capture_id is not None


@pytest.mark.parametrize("length", [-1, 4097])
def test_invalid_allocation_has_same_error_with_capture(capture, length):
    plain, observed = cache(), cache()
    observed.dplb_dfx = capture
    with pytest.raises((ValueError, KVCacheCapacityError)) as expected:
        plain.ensure_group_blocks("invalid", length)
    with pytest.raises(type(expected.value)) as actual:
        observed.ensure_group_blocks("invalid", length)
    assert str(actual.value) == str(expected.value)
    assert not observed._group_request_tokens
    assert not any(e["event"] == "placement_commit" for e in records(capture))


@pytest.mark.parametrize(
    "level,profile,error", [("load", False, "requires --profile"), ("timing", True, "timing is unavailable")]
)
def test_cli_rejects_unsupported_dfx_modes(level, profile, error):
    import pypto_serving.cli.main as cli

    args = cli.build_parser().parse_args(
        ["--model", "model", "--dplb-dfx-level", level] + (["--profile"] if profile else [])
    )
    with pytest.raises(ValueError, match=error):
        cli._build_dplb_dfx_config(args)


@pytest.mark.parametrize(
    "limits", [{"max_seconds": float("nan")}, {"queue_size": True}, {"max_bytes": 0}, {"max_seconds": -1}]
)
def test_invalid_limits_are_rejected(limits):
    with pytest.raises(ValueError):
        DplbDfxConfig(**limits)
