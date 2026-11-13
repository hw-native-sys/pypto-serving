# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from collections import deque

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import (
    Request, RequestStatus, Scheduler, SchedulerConfig, SchedulerOutput,
)


def _output_budget_scheduler(*, grouped=False, **options):
    config = {
        "max_num_scheduled_tokens": 128, "long_prefill_token_threshold": 32,
        "max_seq_len": 512, "async_scheduling": True, "enable_prefix_cache": False,
    }
    manager = KvCacheManager(num_blocks=None if grouped else 128, block_size=128,
                             enable_prefix_cache=False)
    if grouped:
        manager.init_groups((KVCacheGroupSpec(
            name="test", layer_indices=(0,), spec=KVCacheSpec(block_size=128, page_size_bytes=1),
            max_blocks_per_seq=4, num_blocks=128,
        ),), max_batch_size=4)
        config.update(max_num_running_reqs=4, max_prefill_tokens_per_request=32,
                      requires_homogeneous_prefill_decode=True)
    config.update(options)
    return Scheduler(SchedulerConfig(**config), manager)


def _output_budget_request(scheduler, *, outputs=6, maximum=8):
    request = Request(
        request_id="r", prompt_token_ids=[1, 2], max_new_tokens=maximum,
        output_token_ids=list(range(100, 100 + outputs)),
        num_computed_tokens=2 + outputs - 1, status=RequestStatus.RUNNING,
    )
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    return request


def test_async_output_budget_keeps_final_reservation_pending():
    scheduler = _output_budget_scheduler()
    request = _output_budget_request(scheduler, outputs=7)
    final = scheduler.schedule()
    scheduler.advance_after_schedule(final)
    blocks = list(request.allocated_block_ids)
    assert request.num_output_placeholders == 1
    assert request.num_remaining_output_tokens == 0
    assert scheduler.schedule().is_empty
    assert request.status is RequestStatus.RUNNING
    assert request.allocated_block_ids == blocks
    scheduler.update_from_output(final, {request.request_id: 107})
    assert request.output_token_ids == list(range(100, 108))
    assert request.num_output_placeholders == 0
    assert request.status is RequestStatus.FINISHED_LENGTH


def test_async_output_budget_preserves_two_valid_decode_dispatches():
    scheduler = _output_budget_scheduler()
    request = _output_budget_request(scheduler)
    first = scheduler.schedule()
    scheduler.advance_after_schedule(first)
    second = scheduler.schedule()
    assert not second.is_empty
    scheduler.advance_after_schedule(second)
    assert request.num_output_placeholders == 2
    assert scheduler.schedule().is_empty
    scheduler.update_from_output(first, {request.request_id: 106})
    assert scheduler.schedule().is_empty
    scheduler.update_from_output(second, {request.request_id: 107})
    assert request.output_token_ids[-2:] == [106, 107]
    assert request.status is RequestStatus.FINISHED_LENGTH


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("concurrency", [1, 2, 4])
@pytest.mark.parametrize("maximum", [1, 2, 8])
def test_output_budget_counts_terminal_prefill_token(async_mode, concurrency, maximum):
    scheduler = _output_budget_scheduler(async_scheduling=async_mode)
    requests = [Request(str(index), list(range(128)), max_new_tokens=maximum, ignore_eos=True)
                for index in range(concurrency)]
    for request in requests:
        scheduler.add_request(request)
    pending = deque()
    submitted = 0
    sampled = {request.request_id: 0 for request in requests}
    depth = 2 if async_mode else 1
    for _ in range(100):
        dispatched = False
        if len(pending) < depth and scheduler.has_work():
            output = scheduler.schedule(allow_reject_stalled=not pending)
            if not output.is_empty:
                tokens = {}
                for item in output.scheduled_requests:
                    if item.is_prefill:
                        assert item.num_new_tokens == 32
                        if item.num_computed_tokens + item.num_new_tokens < item.request.num_prompt_tokens:
                            continue
                    sampled[item.request.request_id] += 1
                    tokens[item.request.request_id] = 1000 + sampled[item.request.request_id]
                scheduler.advance_after_schedule(output)
                pending.append((output, tokens))
                submitted += 1
                dispatched = True
        if pending and (len(pending) >= depth or not dispatched):
            output, tokens = pending.popleft()
            scheduler.update_from_output(output, tokens)
        elif not pending and not scheduler.has_work():
            break
        elif not pending and not dispatched:
            pytest.fail("generation budget stalled without in-flight work")
    else:
        pytest.fail("generation budget did not finish")
    assert submitted == 4 + maximum - 1
    for request in requests:
        assert request.output_token_ids == list(range(1001, 1001 + maximum))
        assert request.num_output_placeholders == 0
        assert request.status is RequestStatus.FINISHED_LENGTH


def test_grouped_output_budget_keeps_other_prefill_runnable():
    scheduler = _output_budget_scheduler(grouped=True)
    request = _output_budget_request(scheduler, outputs=7)
    final = scheduler.schedule()
    scheduler.advance_after_schedule(final)
    assert request.num_remaining_output_tokens == 0
    other = Request("other", list(range(32)), max_new_tokens=2)
    scheduler.add_request(other)
    output = scheduler.schedule()
    assert [item.request.request_id for item in output.scheduled_requests] == ["other"]
    assert output.scheduled_requests[0].is_prefill
    assert output.scheduled_requests[0].num_new_tokens == 32


def test_async_output_budget_reopens_after_speculative_shortfall():
    scheduler = _output_budget_scheduler(num_speculative_tokens=2)
    request = _output_budget_request(scheduler, outputs=1, maximum=4)
    first = scheduler.schedule()
    scheduler.advance_after_schedule(first)
    assert request.num_output_placeholders == 3
    assert request.num_remaining_output_tokens == 0
    assert scheduler.schedule().is_empty
    scheduler.update_from_output(first, {request.request_id: [101]})
    assert request.num_output_placeholders == 0
    assert request.num_remaining_output_tokens == 2
    assert not scheduler.schedule().is_empty


def test_nonterminal_prefill_keeps_output_budget_available():
    scheduler = _output_budget_scheduler()
    request = Request("r", list(range(81)), max_new_tokens=1)
    scheduler.add_request(request)
    first = scheduler.schedule()
    scheduler.advance_after_schedule(first)
    second = scheduler.schedule()
    scheduler.advance_after_schedule(second)
    assert request.num_computed_tokens == 64
    assert request.num_output_placeholders == 0
    assert request.num_remaining_output_tokens == 1
    assert [first.scheduled_requests[0].num_new_tokens, second.scheduled_requests[0].num_new_tokens] == [32, 32]


@pytest.mark.parametrize("discard_failed_step", [False, True])
def test_output_budget_cancel_and_failure_do_not_charge_new_request(discard_failed_step):
    scheduler = _output_budget_scheduler()
    request = _output_budget_request(scheduler, outputs=7)
    old = scheduler.schedule()
    scheduler.advance_after_schedule(old)
    if discard_failed_step:
        scheduler.discard_scheduled_request(old.scheduled_requests[0])
    scheduler.abort_request(request.request_id)
    replacement = Request(request.request_id, [3], max_new_tokens=1)
    scheduler.add_request(replacement)
    assert replacement.num_remaining_output_tokens == 1
    fresh = scheduler.schedule()
    scheduler.advance_after_schedule(fresh)
    free_blocks = scheduler.kv_cache_manager.num_free_blocks
    scheduler.update_from_output(old, {request.request_id: 12345})
    assert replacement.output_token_ids == []
    assert replacement.num_output_placeholders == 1
    assert scheduler.kv_cache_manager.num_free_blocks == free_blocks
    scheduler.update_from_output(fresh, {replacement.request_id: 7})
    assert replacement.output_token_ids == [7]
    assert request.status is RequestStatus.FINISHED_ABORTED


def test_async_preemption_releases_output_budget_reservation():
    scheduler = _output_budget_scheduler()
    request = _output_budget_request(scheduler)
    old = scheduler.schedule()
    scheduler.advance_after_schedule(old)
    exclude = Request("exclude", [1], max_new_tokens=1)
    preempted = scheduler._preempt_lowest_priority(exclude, set(), {}, SchedulerOutput())
    assert preempted["request"] is request
    assert request.status is RequestStatus.PREEMPTED
    assert request.num_output_placeholders == 0
    assert request.num_remaining_output_tokens == 2
    scheduler.update_from_output(old, {request.request_id: 12345})
    assert len(request.output_token_ids) == 6
    readmitted = scheduler.schedule()
    assert [item.request.request_id for item in readmitted.scheduled_requests] == [request.request_id]
