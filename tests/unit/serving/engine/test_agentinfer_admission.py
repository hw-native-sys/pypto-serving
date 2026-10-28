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

from pypto_serving.config.types import GenerateConfig, KVCacheGroupSpec, KVCacheSpec, RuntimeConfig
from pypto_serving.serving.agentinfer_admission import (
    MAX_ADMISSION_WAIT_SECONDS, AgentInferAdmission, AgentInferOptions,
)
from pypto_serving.serving.agentinfer_core.scheduling.identity import AgentIdentity
from pypto_serving.serving.agentinfer_core.scheduling.program_registry import ProgramRegistry
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
from pypto_serving.serving.memory.kv_cache import KVCacheCapacityObservation, KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


def _admission() -> AgentInferAdmission:
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


def test_admission_receives_full_agent_lineage() -> None:
    admission = _admission()
    observed = []
    admission.policy.on_request_arrival = lambda request_id, identity, *args: observed.append(identity) or True
    identity = AgentIdentity(
        program_id="run-a:worker", task_id="run-a", session_id="run-a",
        agent_id="worker", parent_program_id="run-a:lead", blocks_parent=True,
        expected_resume=False,
    )
    assert admission.arrive(Request("child", [1, 2], max_new_tokens=1), identity)
    assert observed == [identity]
    assert admission._sessions == {"child": "run-a:worker"}


def test_retained_request_is_released_once_by_policy_cycle() -> None:
    admission = _admission()
    request = Request("held", [1, 2, 3], max_new_tokens=1)
    policy = SimpleNamespace(
        on_request_arrival=lambda *args: False,
        run_due_lightweight_checks=lambda *args: None,
        needs_schedule_cycle=lambda *args: True,
        schedule_cycle=lambda *args: None,
        prefix_cache_candidates=lambda: (),
        on_prefix_cache_observation=lambda *args: None,
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
    manager = KvCacheManager(num_blocks=82, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16, max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager)
    assert admission.policy.strategy.config.resume_capacity_ratio == (82 * 128 - 4096) / (82 * 128)
    assert admission.policy._schedule_interval_seconds == 0.02
    assert admission.policy.strategy.config.shared_prefix_freshness_warmup_seconds == 0.0


def test_optional_wait_capacity_relax_reaches_policy() -> None:
    manager = KvCacheManager(num_blocks=128, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16, max_seq_len=4096), manager)
    options = AgentInferOptions.from_mapping({
        "resume_wait_relax_seconds": 8,
        "resume_wait_growth_reserve_fraction": 0.06,
        "resume_wait_extra_headroom_tokens": 512,
        "resume_wait_preserve_cached_blocks": True,
    })
    config = AgentInferAdmission(0, native, manager, options).policy.strategy.config
    assert config.resume_wait_relax_seconds == 8
    assert config.resume_wait_growth_reserve_fraction == 0.06
    assert config.resume_wait_extra_headroom_tokens == 512
    assert config.resume_wait_preserve_cached_blocks


def test_medium_kv_pool_refreshes_warm_prefix_before_next_request() -> None:
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
        prefix_cache_candidates=lambda: (),
        on_prefix_cache_observation=lambda *args: None,
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


@pytest.mark.parametrize("completion_pending", [True, False])
def test_retained_prefix_refresh_releases_request_without_acquiring_blocks(
    monkeypatch, completion_pending,
) -> None:
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
    if not completion_pending:
        admission.poll()
        assert not admission._prefix_refresh_pending
    monkeypatch.setattr(manager, "observe_capacity", lambda: KVCacheCapacityObservation(10496, 4500, 5996))
    request = Request("held", tokens, max_new_tokens=1)
    assert not admission.arrive(request, "session-a")
    # Cold demand cannot fit alongside the donor's acting reservation.
    refs = [block.ref_cnt for block in manager.blocks]
    free = manager.num_free_blocks
    # Admission must refresh the estimate without acquiring KV references.
    admitted = admission.poll()
    if not completion_pending:
        assert admitted == []
        time.sleep(0.05)
        admitted = admission.poll()
    assert admitted == [request]
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
    {"resume_wait_relax_seconds": -1}, {"resume_wait_relax_seconds": True},
    {"resume_wait_growth_reserve_fraction": 1.1},
    {"resume_wait_growth_reserve_fraction": 0.06},
    {"resume_wait_extra_headroom_tokens": -1}, {"resume_wait_extra_headroom_tokens": True},
    {"resume_wait_extra_headroom_tokens": 1.5}, {"resume_wait_extra_headroom_tokens": 512},
    {"resume_wait_preserve_cached_blocks": 1}, {"resume_wait_preserve_cached_blocks": True},
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
    assert not admission.observe_lifecycle("session-a", "first", "terminal")
    assert admission._sessions == {"first": "session-a"}
    admission.complete(first)
    assert not admission.observe_lifecycle("session-a", "first", "continue")
    assert admission.policy.registry.current_ref("session-a") is not None
    assert admission.observe_lifecycle("session-a", "first", "terminal")
    assert not admission.observe_lifecycle("session-a", "first", "terminal")
    assert admission.policy.registry.current_ref("session-a") is None
    next_request = Request("next", [1, 2], max_new_tokens=1)
    assert admission.arrive(next_request, "session-a")
    assert not admission.observe_lifecycle("session-a", "first", "terminal")
    assert admission._sessions == {"next": "session-a"}
    assert admission.policy.registry.current_ref("session-a") is not None


def test_late_terminal_from_previous_round_cannot_release_completed_new_round():
    admission = _admission()
    first = Request("first", [1, 2], max_new_tokens=1)
    second = Request("second", [1, 2], max_new_tokens=1)
    assert admission.arrive(first, "session-a")
    admission.complete(first)
    ref = admission.policy.registry.current_ref("session-a")
    assert admission.arrive(second, "session-a")
    admission.complete(second)
    assert admission.policy.registry.current_ref("session-a") == ref
    assert not admission.observe_lifecycle("session-a", "first", "terminal")
    assert not admission.observe_lifecycle("session-a", "unknown", "terminal")
    assert admission.policy.registry.current_ref("session-a") == ref
    assert admission.observe_lifecycle("session-a", "second", "terminal")
    assert admission.policy.registry.current_ref("session-a") is None


def test_released_program_generations_do_not_accumulate_per_session():
    registry = ProgramRegistry()
    first, created = registry.materialize(AgentIdentity(program_id="session-0"))
    assert created
    assert registry.release(first.ref) is first
    for index in range(1, 1001):
        program, created = registry.materialize(AgentIdentity(program_id=f"session-{index}"))
        assert created
        assert registry.release(program.ref) is program
    assert not registry.has_programs
    assert all(not value for value in vars(registry).values() if isinstance(value, dict))
    current, created = registry.materialize(AgentIdentity(program_id="session-0"))
    assert created
    assert current.ref != first.ref
    assert registry.get(first.ref) is None
    assert registry.release(first.ref) is None
    assert registry.current_ref("session-0") == current.ref


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


def test_enforce_requires_supported_model_and_prefix_cache() -> None:
    with pytest.raises(ValueError, match="prefix caching enabled"):
        ReplicaEngineCore(
            EngineConfig(agentinfer_mode="enforce", enable_prefix_cache=False),
            SimpleNamespace(eos_token_id=0),
        )


def test_deepseek_k1_grouped_cache_reaches_agentinfer_policy() -> None:
    groups = (KVCacheGroupSpec(
        name="eagle", layer_indices=(0,),
        spec=KVCacheSpec(block_size=2, page_size_bytes=1),
        max_blocks_per_seq=4, num_blocks=8, num_partitions=2, is_eagle_group=True,
    ),)
    runtime = RuntimeConfig(page_size=2, max_seq_len=8, num_speculative_tokens=1,
                            kv_cache_groups=groups)
    core = ReplicaEngineCore(
        EngineConfig(executor_cls="PyptoDeepSeekV4Executor", runtime_config=runtime,
                     agentinfer_mode="enforce"),
        SimpleNamespace(eos_token_id=0),
    )
    core.kv_cache_manager.init_groups(groups, max_batch_size=2)
    admission = AgentInferAdmission(0, core.scheduler, core.kv_cache_manager)
    assert admission.grouped
    rank = admission._backend_pool().backends[0].dp_ranks[0]
    assert (rank.total_hbm_kv_tokens, rank.used_hbm_kv_tokens) == (32, 0)
    request = Request("deepseek", [1, 2, 3], max_new_tokens=1)
    assert admission.arrive(request, "session-deepseek")
    assert admission.policy.strategy._capacity_cost(1) == 2
    core.scheduler.add_request(request)
    scheduled = core.scheduler.schedule().scheduled_requests[0]
    admission.native_scheduled(scheduled)
    assert request.cache_partition is not None

    with pytest.raises(ValueError, match="MTP K=1"):
        ReplicaEngineCore(
            EngineConfig(executor_cls="PyptoDeepSeekV4Executor", runtime_config=RuntimeConfig(
                page_size=2, max_seq_len=8, num_speculative_tokens=2, kv_cache_groups=groups,
            ), agentinfer_mode="enforce"),
            SimpleNamespace(eos_token_id=0),
        )


def test_enforce_uses_bundled_scheduler_without_external_package() -> None:
    from pypto_serving.serving.agentinfer_core.scheduling.runtime import ProgramScheduler

    core = ReplicaEngineCore(
        EngineConfig(agentinfer_mode="enforce"),
        SimpleNamespace(eos_token_id=0),
    )
    assert ProgramScheduler.__module__.startswith("pypto_serving.serving.agentinfer_core")
    assert core._agentinfer_admission is None


@pytest.mark.parametrize("prompt_tokens,max_new_tokens,min_tokens,error", [
    (5, 1, 0, "exceeds max_seq_len"),
    (4, 1, 0, "leaves no room for generation"),
    (3, 2, 2, "min_tokens"),
])
def test_invalid_agent_request_is_rejected_before_admission(
    prompt_tokens, max_new_tokens, min_tokens, error,
) -> None:
    async def run() -> None:
        manager = KvCacheManager(num_blocks=8, block_size=2)
        core = ReplicaEngineCore.__new__(ReplicaEngineCore)
        core.config = EngineConfig(agentinfer_mode="enforce")
        core.tokenizer = SimpleNamespace(eos_token_id=0)
        core.scheduler = Scheduler(SchedulerConfig(max_seq_len=4), manager)
        arrivals: list[str] = []
        core._agentinfer_admission = SimpleNamespace(
            arrive=lambda request, session_id: arrivals.append(request.request_id) or False,
            cancel=lambda request_id: False,
        )
        core._stat_logger = None
        core._request_contexts = {}
        core._pending_free_ids = []

        with pytest.raises(ValueError, match=error):
            async for _ in core.add_request(
                "invalid", "prompt", GenerateConfig(max_new_tokens=max_new_tokens, min_tokens=min_tokens),
                prompt_token_ids=list(range(prompt_tokens)), agent_session_id="session-a",
            ):
                pass
        assert arrivals == []
        assert core._request_contexts == core.scheduler.requests == {}
        assert core._pending_free_ids == []

    asyncio.run(run())


def test_released_request_validation_failure_does_not_stop_engine_loop() -> None:
    async def run() -> None:
        core = ReplicaEngineCore.__new__(ReplicaEngineCore)
        core.config = EngineConfig(engine_loop_interval=0)
        core._async_scheduling = False
        core._running = True
        core._batch_queue = []
        core._max_in_flight = 1
        bad = Request("bad", [1, 2], max_new_tokens=1)
        good = Request("good", [3], max_new_tokens=1)
        bad_queue = asyncio.Queue()
        core._request_contexts = {
            "bad": SimpleNamespace(queue=bad_queue),
            "good": SimpleNamespace(queue=asyncio.Queue()),
        }
        added: list[str] = []
        cancelled: list[str] = []

        def add_request(request: Request) -> None:
            if request.request_id == "bad":
                raise ValueError("invalid released request")
            added.append(request.request_id)

        core.scheduler = SimpleNamespace(add_request=add_request, has_work=lambda: False)
        core._agentinfer_admission = SimpleNamespace(
            poll=lambda: [bad, good], cancel=cancelled.append,
        )

        async def finish_loop() -> None:
            core._running = False

        core._flush_pending_frees = finish_loop
        await core._engine_loop()

        assert isinstance(bad_queue.get_nowait(), ValueError)
        assert added == ["good"]
        assert cancelled == ["bad"]
        assert tuple(core._request_contexts) == ("good",)

    asyncio.run(run())


def test_core_abort_of_retained_request_never_releases_worker() -> None:
    async def run() -> None:
        core = ReplicaEngineCore.__new__(ReplicaEngineCore)
        core.config = EngineConfig(agentinfer_mode="enforce")
        core.tokenizer = SimpleNamespace(eos_token_id=0)
        native_adds: list[str] = []
        native_aborts: list[str] = []
        core.scheduler = SimpleNamespace(
            add_request=lambda request: native_adds.append(request.request_id),
            validate_request=lambda request: None,
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
