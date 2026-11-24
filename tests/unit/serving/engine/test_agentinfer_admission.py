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
from pypto_serving.serving.agentinfer_core.scheduling.domain import ProgramStatus
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
    def record_arrival(request_id, identity, *args, **kwargs):
        observed.append(identity)
        return True

    admission.policy.on_request_arrival = record_arrival
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
        on_request_arrival=lambda *args, **kwargs: False,
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


@pytest.mark.parametrize("max_new_tokens", [1, 128])
def test_decode_buffer_respects_context_limit_in_idle_two_context_pool(max_new_tokens) -> None:
    manager = KvCacheManager(num_blocks=64, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16, max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager)
    request = Request("long", list(range(4095)), max_new_tokens=max_new_tokens)
    native.validate_request(request)

    assert request.max_new_tokens == 1
    assert admission.arrive(request, "session-a")
    assert admission.retained_request_ids == ()
    native.add_request(request)
    for _ in range(2):
        step = native.schedule()
        assert len(step.scheduled_requests) == 1
        admission.native_scheduled(step.scheduled_requests[0])
        outputs = native.update_from_output(step, {request.request_id: 42})

    assert len(outputs) == 1 and outputs[0].finished
    assert len(request.output_token_ids) == 1
    admission.complete(request)
    assert not native.has_work()
    assert not admission._sessions
    assert manager.num_free_blocks == manager.num_blocks


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


def test_retained_cancel_removes_request_without_native_admission() -> None:
    admission = _admission()
    request = Request("cancelled", list(range(1200)), max_new_tokens=1)
    assert not admission.arrive(request, "session-a")
    assert admission.cancel("cancelled")
    assert not admission.cancel("cancelled")
    assert not admission.native.requests
    assert admission.retained_request_ids == ()
    assert admission._sessions == {}


def test_admission_timeout_hands_off_program_waiters_with_their_bindings() -> None:
    manager = KvCacheManager(num_blocks=16, block_size=128)
    native = Scheduler(SchedulerConfig(max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager, AgentInferOptions(resume_capacity_ratio=0.85))
    first = Request("first", list(range(1800)), max_new_tokens=1)
    latest = Request("latest", list(range(1800)), max_new_tokens=1)
    assert not admission.arrive(first, "session-a")
    assert not admission.arrive(latest, "session-a")
    ref = admission.policy.registry.current_ref("session-a")
    assert ref is not None

    admission._retained[first.request_id].arrived_at -= MAX_ADMISSION_WAIT_SECONDS + 1
    released = admission.poll()
    assert released == [first, latest]
    assert admission.poll() == []
    assert admission.retained_request_ids == ()
    for request in released:
        native.add_request(request)
    assert admission.policy.registry.current_ref("session-a") == ref

    native.abort_request(first.request_id)
    assert not admission.cancel(first.request_id)
    assert admission.policy.registry.current_ref("session-a") == ref
    step = native.schedule()
    scheduled, = step.scheduled_requests
    admission.native_scheduled(scheduled)
    output, = native.update_from_output(step, {latest.request_id: 42})
    assert output.finished
    admission.complete(latest)
    assert admission.observe_lifecycle("session-a", latest.request_id, "terminal")
    assert admission.policy.registry.current_ref("session-a") is None
    assert not admission._sessions


@pytest.mark.parametrize("state", ["admitted", "retained", "resumed"])
def test_cancel_last_request_releases_program_and_capacity(monkeypatch, state) -> None:
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    admission = _admission()
    if state == "resumed":
        previous = Request("previous", [1, 2], max_new_tokens=1)
        assert admission.arrive(previous, "session-a")
        admission.complete(previous)
    request = Request("cancelled", list(range(1200 if state == "retained" else 500)), max_new_tokens=1)
    assert admission.arrive(request, "session-a") is (state != "retained")
    ref = admission.policy.registry.current_ref("session-a")
    assert ref is not None

    assert admission.cancel(request.request_id) is (state == "retained")

    assert admission.policy.registry.get(ref) is None
    assert admission.policy.strategy_factors.for_program(ref) is None
    assert admission.retained_request_ids == ()
    now[0] += 4000
    assert admission.poll() == []
    capacity = admission.kv_cache_manager.observe_capacity()
    assert capacity.free_tokens == capacity.total_tokens == 1024
    assert admission.arrive(Request("next", list(range(800)), max_new_tokens=1), "session-b")


def test_cancelling_one_request_preserves_overlapping_program() -> None:
    admission = _admission()
    first = Request("first", [1, 2], max_new_tokens=1)
    second = Request("second", [1, 2], max_new_tokens=1)
    assert admission.arrive(first, "session-a")
    assert admission.arrive(second, "session-a")
    ref = admission.policy.registry.current_ref("session-a")

    admission.cancel(first.request_id)

    assert admission.policy.registry.current_ref("session-a") == ref
    assert admission.policy.registry.require(ref).status is ProgramStatus.REASONING
    assert admission._sessions == {"second": "session-a"}
    admission.complete(second)
    assert admission.observe_lifecycle("session-a", second.request_id, "terminal")
    assert not admission.policy.registry.has_programs


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


def test_plain_completion_refreshes_retained_agent_prefix_without_binding_or_acquiring_blocks(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    manager = KvCacheManager(num_blocks=64, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16, max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager, AgentInferOptions(resume_capacity_ratio=0.4))
    tokens = list(range(3500))
    donor = Request("plain", tokens, max_new_tokens=1)
    native.add_request(donor)
    held = Request("held", tokens, max_new_tokens=1)
    assert not admission.arrive(held, "session-a")
    assert admission.poll() == []

    for _ in range(2):
        step = native.schedule()
        assert len(step.scheduled_requests) == 1
        admission.native_scheduled(step.scheduled_requests[0])
        outputs = native.update_from_output(step, {donor.request_id: 42})
    assert len(outputs) == 1 and outputs[0].finished
    admission.complete(donor)
    assert manager.observe_prefix(tokens).cached_tokens == 3456

    refs = [block.ref_cnt for block in manager.blocks]
    free = manager.num_free_blocks
    now[0] += 0.05
    assert admission.poll() == [held]
    assert admission.poll() == []
    assert [block.ref_cnt for block in manager.blocks] == refs
    assert manager.num_free_blocks == free
    programs = admission.policy.registry.views()
    assert len(programs) == 1 and programs[0].ref.program_id == "session-a"
    assert programs[0].tokens.shared_prefix_tokens == 3456

    native.add_request(held)
    step = native.schedule()
    assert len(step.scheduled_requests) == 1
    assert step.scheduled_requests[0].num_computed_tokens == 3456
    admission.native_scheduled(step.scheduled_requests[0])
    outputs = native.update_from_output(step, {held.request_id: 43})
    assert len(outputs) == 1 and outputs[0].finished
    admission.complete(held)
    assert not native.has_work()
    assert not admission._sessions
    assert manager.num_free_blocks == manager.num_blocks


@pytest.mark.parametrize("arrival_offsets", [(0.0,), (0.0, 50.0)])
def test_completion_refresh_waits_for_each_retained_prefix_freshness_deadline(monkeypatch, arrival_offsets):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    manager = KvCacheManager(num_blocks=16, block_size=128)
    native = Scheduler(SchedulerConfig(max_num_running_reqs=16, max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager, AgentInferOptions(resume_capacity_ratio=0.85))
    tokens = list(range(1800))
    donor = Request("plain", tokens, max_new_tokens=1)
    native.add_request(donor)
    held = []
    for index, offset in enumerate(arrival_offsets):
        now[0] = 100.0 + offset
        request = Request(f"held-{index}", tokens, max_new_tokens=1)
        assert not admission.arrive(request, f"session-{index}")
        held.append(request)
        assert admission.poll() == []

    step = native.schedule()
    outputs = native.update_from_output(step, {donor.request_id: 42})
    assert len(outputs) == 1 and outputs[0].finished
    admission.complete(donor)
    assert manager.observe_prefix(tokens).cached_tokens == 1792
    refs = [block.ref_cnt for block in manager.blocks]
    free = manager.num_free_blocks
    observations = []
    observe_prefix = manager.observe_prefix

    def record_observation(prompt_token_ids):
        observations.append(now[0])
        return observe_prefix(prompt_token_ids)

    monkeypatch.setattr(manager, "observe_prefix", record_observation)
    now[0] += 0.1
    assert admission.poll() == []
    assert admission._prefix_refresh_pending
    for index, offset in enumerate(arrival_offsets):
        now[0] = 199.9 + offset
        assert admission.poll() == []
        assert len(observations) == index
        now[0] = 201.0 + offset
        assert admission.poll() == [held[index]]
        assert len(observations) == index + 1
        view = next(view for view in admission.policy.registry.views()
                    if view.ref.program_id == f"session-{index}")
        assert view.tokens.shared_prefix_tokens == 1792
        assert admission._prefix_refresh_pending is (index + 1 < len(held))
        assert admission.poll() == []
    assert manager.num_free_blocks == free
    assert [block.ref_cnt for block in manager.blocks] == refs
    for request in held:
        admission.cancel(request.request_id)


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


def test_full_admission_pool_keeps_overlapping_native_request_bound(monkeypatch) -> None:
    manager = KvCacheManager(num_blocks=16, block_size=128)
    native = Scheduler(SchedulerConfig(max_seq_len=4096), manager)
    admission = AgentInferAdmission(0, native, manager, AgentInferOptions(resume_capacity_ratio=0.85))
    monkeypatch.setattr("pypto_serving.serving.agentinfer_admission.MAX_RETAINED_REQUESTS", 1)
    earlier = Request("earlier", [1, 2], max_new_tokens=2)
    latest = Request("latest", [1, 2], max_new_tokens=1)
    for request in (earlier, latest):
        assert admission.arrive(request, "session-a")
        native.add_request(request)
    ref = admission.policy.registry.current_ref("session-a")
    assert ref is not None

    held = Request("held", list(range(1800)), max_new_tokens=1)
    assert not admission.arrive(held, "other-session")
    overflow = Request("overflow", [1, 2], max_new_tokens=2)
    assert admission.arrive(overflow, "session-a")
    native.add_request(overflow)
    assert admission.retained_request_ids == (held.request_id,)
    assert admission.policy.retained_request_ids == (held.request_id,)

    step = native.schedule()
    outputs = native.update_from_output(step, {request.request_id: 42 for request in (earlier, latest, overflow)})
    assert [out.request_id for out in outputs if out.finished] == [latest.request_id]
    admission.complete(latest)
    assert not admission.observe_lifecycle("session-a", latest.request_id, "terminal")

    native.abort_request(earlier.request_id)
    assert not admission.cancel(earlier.request_id)
    assert overflow.request_id in native.requests
    assert admission.policy.registry.current_ref("session-a") == ref

    step = native.schedule()
    output, = native.update_from_output(step, {overflow.request_id: 42})
    assert output.finished
    admission.complete(overflow)
    assert admission.observe_lifecycle("session-a", overflow.request_id, "terminal")
    assert admission.policy.registry.current_ref("session-a") is None


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


@pytest.mark.parametrize("finish_earlier", ["complete", "cancel"])
def test_latest_terminal_is_deferred_until_overlapping_request_finishes(finish_earlier):
    admission = _admission()
    earlier = Request("earlier", [1, 2], max_new_tokens=1)
    latest = Request("latest", [1, 2], max_new_tokens=1)
    assert admission.arrive(earlier, "session-a")
    assert admission.arrive(latest, "session-a")
    admission.complete(latest)
    assert not admission.observe_lifecycle("session-a", latest.request_id, "terminal")
    assert admission.policy.registry.has_programs

    if finish_earlier == "complete":
        admission.complete(earlier)
    else:
        admission.cancel(earlier.request_id)
    assert not admission.policy.registry.has_programs
    assert not admission._sessions
    assert not admission.observe_lifecycle("session-a", latest.request_id, "terminal")


def test_new_arrival_invalidates_terminal_deferred_by_overlapping_request():
    admission = _admission()
    earlier = Request("earlier", [1, 2], max_new_tokens=1)
    latest = Request("latest", [1, 2], max_new_tokens=1)
    new = Request("new", [1, 2], max_new_tokens=1)
    assert admission.arrive(earlier, "session-a")
    assert admission.arrive(latest, "session-a")
    ref = admission.policy.registry.current_ref("session-a")
    admission.complete(latest)
    assert not admission.observe_lifecycle("session-a", latest.request_id, "terminal")
    assert admission.arrive(new, "session-a")
    admission.complete(earlier)
    assert admission.policy.registry.current_ref("session-a") == ref
    admission.complete(new)
    assert admission.policy.registry.current_ref("session-a") == ref
    assert not admission.observe_lifecycle("session-a", latest.request_id, "terminal")
    assert admission.observe_lifecycle("session-a", new.request_id, "terminal")


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


def test_deepseek_k1_grouped_cache_reaches_agentinfer_policy(monkeypatch) -> None:
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
    selected = 0
    select_group_prefix = core.kv_cache_manager.select_group_prefix_blocks

    def count_prefix_selection(*args, **kwargs):
        nonlocal selected
        selected += 1
        return select_group_prefix(*args, **kwargs)

    monkeypatch.setattr(core.kv_cache_manager, "select_group_prefix_blocks", count_prefix_selection)
    scheduled = core.scheduler.schedule().scheduled_requests[0]
    admission.native_scheduled(scheduled)
    assert request.cache_partition is not None
    assert selected == 1

    with pytest.raises(ValueError, match="MTP K=1"):
        ReplicaEngineCore(
            EngineConfig(executor_cls="PyptoDeepSeekV4Executor", runtime_config=RuntimeConfig(
                page_size=2, max_seq_len=8, num_speculative_tokens=2, kv_cache_groups=groups,
            ), agentinfer_mode="enforce"),
            SimpleNamespace(eos_token_id=0),
        )


@pytest.mark.parametrize("num_speculative_tokens", [0, 7])
def test_dspark_grouped_cache_reaches_agentinfer_policy(num_speculative_tokens: int) -> None:
    groups = tuple(
        KVCacheGroupSpec(
            name=name,
            layer_indices=(0,),
            spec=KVCacheSpec(block_size=block_size, page_size_bytes=1),
            max_blocks_per_seq=16,
            num_partitions=1,
            sliding_window=sliding_window,
        )
        for name, block_size, sliding_window in (
            ("ori", 32, 128),
            ("cmp_c128", 128, None),
            ("cmp_c4", 128, None),
            ("idx", 128, None),
            ("hca_state", 8, 128),
            ("csa_state", 2, 8),
            ("csa_inner_state", 2, 8),
        )
    )
    runtime = RuntimeConfig(
        page_size=32,
        max_seq_len=128,
        num_speculative_tokens=num_speculative_tokens,
        speculative_prefix_cache_replay_tokens=128 if num_speculative_tokens else 0,
        kv_cache_groups=groups,
    )
    core = ReplicaEngineCore(
        EngineConfig(
            executor_cls="PyptoDeepSeekV4DSparkExecutor",
            runtime_config=runtime,
            agentinfer_mode="enforce",
        ),
        SimpleNamespace(eos_token_id=0),
    )
    core.kv_cache_manager.init_groups(groups, max_batch_size=4)

    admission = AgentInferAdmission(
        0, core.scheduler, core.kv_cache_manager, executor_cls="PyptoDeepSeekV4DSparkExecutor",
    )

    assert admission.grouped
    assert admission.policy.strategy._capacity_cost(1) > 0


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
        core._step_timeout = 1
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
