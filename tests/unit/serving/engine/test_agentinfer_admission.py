# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import time

import pytest

from pypto_serving.config.types import GenerateConfig
from pypto_serving.serving.agentinfer_admission import (
    MAX_ADMISSION_WAIT_SECONDS, AgentInferAdmission, AgentInferOptions,
)
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
from pypto_serving.serving.memory.kv_cache import KVCacheCapacityObservation, KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


def _admission() -> AgentInferAdmission:
    pytest.importorskip("agentinfer.scheduling")
    manager = KvCacheManager(num_blocks=64, block_size=16)
    native = Scheduler(SchedulerConfig(), manager)
    return AgentInferAdmission(0, native, manager)


def test_enforce_tracks_immediate_native_request_and_completion() -> None:
    admission = _admission()
    request = Request("first", [1, 2, 3], max_new_tokens=1)
    assert admission.arrive(request, "session-a")
    admission.native.add_request(request)
    assert admission.retained_request_ids == ()
    step = admission.native.schedule()
    admission.native_scheduled(step.scheduled_requests[0])
    admission.native.update_from_output(step, {"first": 4})
    admission.progress(request)
    admission.complete(request)
    assert admission.policy.retained_request_count == 0
    assert admission._sessions == {}


def test_retained_request_is_released_once_by_policy_cycle() -> None:
    admission = _admission()
    request = Request("held", [1, 2, 3], max_new_tokens=1)
    policy = SimpleNamespace(
        on_request_arrival=lambda *args: False,
        run_due_lightweight_checks=lambda *args: None,
        needs_schedule_cycle=lambda *args: True,
        schedule_cycle=lambda *args: None,
        consume_admitted_requests=lambda: ((SimpleNamespace(request_id="held"), None),),
        cancel_request=lambda *args: None,
    )
    admission.policy = policy
    assert not admission.arrive(request, "session-a")
    assert "held" not in admission.native.requests
    assert admission.retained_request_ids == ("held",)
    assert admission.poll() == [request]
    assert admission.poll() == []
    assert admission.retained_request_ids == ()
    admission.native.add_request(request)
    assert tuple(admission.native.requests) == ("held",)


def test_capacity_recovery_releases_retained_request_without_one_second_poll_delay(monkeypatch) -> None:
    pytest.importorskip("agentinfer.scheduling")
    manager = KvCacheManager(num_blocks=1337, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16), manager)
    admission = AgentInferAdmission(0, native, manager)
    assert admission.policy.strategy.config.resume_capacity_ratio == 1.0
    assert admission.policy._schedule_interval_seconds == 0.02
    monkeypatch.setattr(
        manager, "observe_capacity",
        lambda: KVCacheCapacityObservation(total_tokens=1024, used_tokens=0, free_tokens=1024),
    )
    request = Request("held", list(range(1200)), max_new_tokens=16)
    assert not admission.arrive(request, "session-a")
    assert admission.poll() == []
    monkeypatch.setattr(
        admission.kv_cache_manager, "observe_capacity",
        lambda: KVCacheCapacityObservation(total_tokens=100_000, used_tokens=0, free_tokens=100_000),
    )
    time.sleep(0.1)
    assert admission.poll() == [request]
    assert admission.retained_request_ids == ()


def test_small_kv_pool_keeps_policy_retention_window() -> None:
    admission = _admission()
    assert admission.policy._schedule_interval_seconds == 1.0
    assert admission.policy.strategy.config.resume_capacity_ratio == 1.0


def test_medium_kv_pool_reserves_one_context_and_schedules_promptly() -> None:
    pytest.importorskip("agentinfer.scheduling")
    manager = KvCacheManager(num_blocks=82, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16, max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager)
    assert admission.policy.strategy.config.resume_capacity_ratio == (82 * 128 - 4096) / (82 * 128)
    assert admission.policy._schedule_interval_seconds == 0.02
    assert admission.policy.strategy.config.shared_prefix_freshness_warmup_seconds == 0.0


def test_medium_kv_pool_refreshes_warm_prefix_before_next_request() -> None:
    pytest.importorskip("agentinfer.scheduling")
    manager = KvCacheManager(num_blocks=82, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16, max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager)
    first = Request("cold", list(range(1192)), max_new_tokens=16)
    assert admission.arrive(first, "session-a")
    admission.policy.on_prefix_cache_observation(first.request_id, 0, hbm_cached_prefix_tokens=0)
    admission.complete(first)
    admission.poll()

    second = Request("warm", list(range(1192)), max_new_tokens=16)
    assert admission.arrive(second, "session-a")
    admission.policy.on_prefix_cache_observation(second.request_id, 1152, hbm_cached_prefix_tokens=1152)
    assert admission.policy.registry.views()[0].tokens.shared_prefix_tokens == 1152


def test_retained_cancel_and_timeout_do_not_enter_native_queue(monkeypatch) -> None:
    admission = _admission()
    cancelled: list[str] = []
    admission.policy = SimpleNamespace(
        on_request_arrival=lambda *args: False,
        run_due_lightweight_checks=lambda *args: None,
        needs_schedule_cycle=lambda *args: True,
        schedule_cycle=lambda *args: None,
        consume_admitted_requests=lambda: (),
        cancel_request=cancelled.append,
    )
    request = Request("cancelled", [1, 2], max_new_tokens=1)
    assert not admission.arrive(request, "session-a")
    assert admission.cancel("cancelled")
    assert not admission.cancel("cancelled")
    assert cancelled == ["cancelled"]
    assert not admission.native.requests

    timeout = Request("timeout", [1, 2], max_new_tokens=1)
    assert not admission.arrive(timeout, "session-a")
    admission._retained["timeout"].arrived_at -= MAX_ADMISSION_WAIT_SECONDS + 1
    assert admission.poll() == [timeout]
    assert cancelled == ["cancelled", "timeout"]
    assert admission.retained_request_ids == ()
    assert admission._sessions == {}


def test_prefix_observation_is_not_overwritten_by_decode_or_later_prefill() -> None:
    admission = _admission()
    request = Request("prefix", [1, 2, 3, 4], max_new_tokens=2)
    assert admission.arrive(request, "session-a")
    observations: list[tuple[tuple, dict]] = []
    admission.policy.on_prefix_cache_observation = lambda *args, **kwargs: observations.append((args, kwargs))
    admission.native_scheduled(SimpleNamespace(request=request, is_prefill=True, num_computed_tokens=2))
    admission.native_scheduled(SimpleNamespace(request=request, is_prefill=True, num_computed_tokens=4))
    admission.native_scheduled(SimpleNamespace(request=request, is_prefill=False, num_computed_tokens=4))
    assert observations == [(('prefix', 2), {"hbm_cached_prefix_tokens": 2})]
    admission.cancel("prefix")
    assert admission._prefix_observed == set()


def test_configured_force_resume_uses_original_policy_before_fallback(monkeypatch) -> None:
    pytest.importorskip("agentinfer.scheduling")
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    manager = KvCacheManager(num_blocks=64, block_size=16)
    native = Scheduler(SchedulerConfig(), manager)
    admission = AgentInferAdmission(0, native, manager, AgentInferOptions(
        force_resume_timeout_min_seconds=1, force_resume_timeout_max_seconds=3,
    ))
    request = Request("held", list(range(1200)), max_new_tokens=16)
    assert not admission.arrive(request, "session-a")
    assert admission.poll() == []
    now[0] = 102.0
    assert admission.poll() == []
    now[0] = 103.1
    assert admission.poll() == [request]
    assert admission.poll() == []
    # A policy resume keeps tracking, unlike the adapter's timeout cancellation.
    assert admission._sessions == {"held": "session-a"}
    assert admission.retained_request_ids == ()
    admission.cancel("held")


def test_retained_prefix_refresh_releases_request_without_acquiring_blocks(monkeypatch) -> None:
    pytest.importorskip("agentinfer.scheduling")
    manager = KvCacheManager(num_blocks=82, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16, max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager)
    tokens = list(range(1192))
    cold = Request("donor", tokens, max_new_tokens=1)
    assert admission.arrive(cold, "donor-session")
    native.add_request(cold)
    step = native.schedule()
    admission.native_scheduled(step.scheduled_requests[0])
    native.update_from_output(step, {"donor": 42})
    admission.complete(cold)
    assert manager.observe_prefix(tokens).cached_tokens == 1152
    monkeypatch.setattr(manager, "observe_capacity", lambda: KVCacheCapacityObservation(10496, 4500, 5996))
    request = Request("held", tokens, max_new_tokens=1)
    assert not admission.arrive(request, "session-a")
    # Cold demand cannot fit alongside the donor's acting reservation.
    refs = [block.ref_cnt for block in manager.blocks]
    free = manager.num_free_blocks
    # Admission must refresh the estimate without acquiring KV references.
    assert admission.poll() == [request]
    assert manager.num_free_blocks == free
    assert [block.ref_cnt for block in manager.blocks] == refs
    held = next(view for view in admission.policy.registry.views() if view.ref.program_id == "session-a")
    assert held.tokens.shared_prefix_tokens == 1152
    assert admission.poll() == []
    admission.cancel("held")
    assert admission.policy.prefix_cache_candidates() == ()


@pytest.mark.parametrize("values", [
    {"resume_capacity_ratio": 0}, {"resume_capacity_ratio": float("nan")},
    {"resume_capacity_ratio": True}, {"force_resume_timeout_min_seconds": 4,
                                     "force_resume_timeout_max_seconds": 3},
    {"unknown": 1},
])
def test_invalid_admission_settings_are_rejected(values) -> None:
    with pytest.raises(ValueError):
        AgentInferOptions.from_mapping(values)


def test_full_admission_pool_uses_native_queue_without_binding(monkeypatch) -> None:
    admission = _admission()
    monkeypatch.setattr("pypto_serving.serving.agentinfer_admission.MAX_RETAINED_REQUESTS", 0)
    request = Request("overflow", [1, 2], max_new_tokens=1)
    assert admission.arrive(request, "session-a")
    assert admission._sessions == {}
    assert admission.retained_request_ids == ()


def test_prefix_ownership_is_captured_before_each_request_acquires_references():
    admission = _admission()
    manager, native = admission.kv_cache_manager, admission.native
    tokens = list(range(33))
    donor = Request("donor", tokens, max_new_tokens=1)
    assert admission.arrive(donor, "donor-session")
    native.add_request(donor)
    first = native.schedule()
    admission.native_scheduled(first.scheduled_requests[0])
    native.update_from_output(first, {"donor": 100})
    admission.complete(donor)
    for name in ("one", "two"):
        request = Request(name, tokens, max_new_tokens=1)
        assert admission.arrive(request, f"session-{name}")
        native.add_request(request)
    step = native.schedule()
    refs = [block.ref_cnt for block in manager.blocks]
    free = manager.num_free_blocks
    for scheduled in step.scheduled_requests:
        admission.native_scheduled(scheduled)
    one = admission.policy._request_bindings["one"]
    two = admission.policy._request_bindings["two"]
    assert (one.cached_prefix_tokens, one.shared_prefix_tokens, one.hbm_hit_ref_zero_tokens) == (32, 0, 32)
    assert (two.cached_prefix_tokens, two.shared_prefix_tokens, two.hbm_hit_ref_zero_tokens) == (32, 32, 0)
    assert [block.ref_cnt for block in manager.blocks] == refs
    assert manager.num_free_blocks == free


def test_terminal_lifecycle_releases_only_idle_program_and_preserves_live_request():
    admission = _admission()
    first = Request("first", [1, 2], max_new_tokens=1)
    assert admission.arrive(first, "session-a")
    assert not admission.observe_lifecycle("session-a", "terminal")
    assert admission._sessions == {"first": "session-a"}
    admission.complete(first)
    assert not admission.observe_lifecycle("session-a", "continue")
    assert admission.policy.registry.current_ref("session-a") is not None
    assert admission.observe_lifecycle("session-a", "terminal")
    assert not admission.observe_lifecycle("session-a", "terminal")
    assert admission.policy.registry.current_ref("session-a") is None
    next_request = Request("next", [1, 2], max_new_tokens=1)
    assert admission.arrive(next_request, "session-a")
    assert not admission.observe_lifecycle("session-a", "terminal")
    assert admission._sessions == {"next": "session-a"}
    assert admission.policy.registry.current_ref("session-a") is not None


def test_native_waiting_demand_uses_current_context_without_future_decode_budget():
    admission = _admission()
    tracked = Request("agent", [1, 2, 3], max_new_tokens=100)
    plain = Request("plain", [4, 5], max_new_tokens=200, output_token_ids=[6])
    assert admission.arrive(tracked, "session-a")
    admission.native.add_request(tracked)
    admission.native.add_request(plain)
    rank = admission._backend_pool().backends[0].dp_ranks[0]
    assert rank.waiting_requests == 2
    assert rank.waiting_hbm_kv_tokens == 6


def test_enforce_requires_qwen_prefix_cache() -> None:
    with pytest.raises(ValueError, match="Qwen with prefix caching"):
        ReplicaEngineCore(
            EngineConfig(agentinfer_mode="enforce", enable_prefix_cache=False),
            SimpleNamespace(eos_token_id=0),
        )


def test_enforce_reports_missing_scheduler_before_worker_start(monkeypatch) -> None:
    monkeypatch.setattr("pypto_serving.serving.engine.async_engine.importlib.util.find_spec", lambda _: None)
    with pytest.raises(RuntimeError, match="pinned agentinfer.scheduling"):
        ReplicaEngineCore(
            EngineConfig(agentinfer_mode="enforce"),
            SimpleNamespace(eos_token_id=0),
        )


def test_core_abort_of_retained_request_never_releases_worker() -> None:
    async def run() -> None:
        core = ReplicaEngineCore.__new__(ReplicaEngineCore)
        core.config = EngineConfig(agentinfer_mode="enforce")
        core.tokenizer = SimpleNamespace(eos_token_id=0)
        native_adds: list[str] = []
        native_aborts: list[str] = []
        core.scheduler = SimpleNamespace(
            add_request=lambda request: native_adds.append(request.request_id),
            abort_request=native_aborts.append,
        )
        cancelled: list[str] = []
        core._agentinfer_admission = SimpleNamespace(
            arrive=lambda request, session_id: False,
            cancel=lambda request_id: cancelled.append(request_id) or True,
        )
        core._stat_logger = None
        core._request_contexts = {}
        core._pending_free_ids = []
        queued = asyncio.Event()

        async def collect():
            return [output async for output in core.add_request(
                "held", "prompt", GenerateConfig(max_new_tokens=1),
                prompt_token_ids=[1, 2], agent_session_id="session-a",
                on_queued=queued.set,
            )]

        task = asyncio.create_task(collect())
        await queued.wait()
        await core.abort_request("held")
        outputs = await task
        assert len(outputs) == 1
        assert outputs[0].finish_reason == "FINISHED_ABORTED"
        assert native_adds == native_aborts == core._pending_free_ids == []
        assert cancelled == ["held"]
        assert core._request_contexts == {}

    asyncio.run(run())


def test_core_stop_aborts_retained_requests_before_worker_shutdown() -> None:
    async def run() -> None:
        core = ReplicaEngineCore.__new__(ReplicaEngineCore)
        core._request_contexts = {"held": SimpleNamespace(queue=asyncio.Queue())}
        core._pending_free_ids = []
        core._stat_logger = None
        core._loop_task = None
        core._running = True
        native_aborts: list[str] = []
        core.scheduler = SimpleNamespace(abort_request=native_aborts.append)
        retained = {"held"}
        closed: list[bool] = []

        def cancel(request_id: str) -> bool:
            retained.remove(request_id)
            return True

        core._agentinfer_admission = SimpleNamespace(
            retained_request_ids=("held",),
            cancel=cancel,
            close=lambda: closed.append(True),
        )
        core._shutdown_worker = lambda *, timeout: None
        queue = core._request_contexts["held"].queue

        await core.stop()

        assert queue.get_nowait().finish_reason == "FINISHED_ABORTED"
        assert retained == set()
        assert closed == [True]
        assert native_aborts == core._pending_free_ids == []

    asyncio.run(run())
