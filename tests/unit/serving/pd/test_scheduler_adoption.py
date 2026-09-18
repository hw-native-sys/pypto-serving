# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig

from .helpers import make_cache_manager


def _scheduler(manager, *, prefill_handoff: bool = False) -> Scheduler:
    return Scheduler(
        SchedulerConfig(
            max_num_running_reqs=4,
            max_num_scheduled_tokens=128,
            long_prefill_token_threshold=128,
            max_seq_len=512,
            enable_prefix_cache=False,
            enable_chunk_prefill=True,
            requires_homogeneous_prefill_decode=True,
            prefill_handoff=prefill_handoff,
        ),
        manager,
    )


def test_prefill_role_stops_before_decode_and_keeps_requested_partition() -> None:
    manager = make_cache_manager()
    scheduler = _scheduler(manager, prefill_handoff=True)
    request = Request("request", list(range(33)), 8, cache_partition=2)
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    assert scheduled.scheduled_requests[0].cache_partition == 2
    outputs = scheduler.update_from_output(scheduled, {"request": 7})
    assert outputs[0].handoff_ready
    assert request.handoff_pending
    assert scheduler.schedule().is_empty
    scheduler.mark_prefill_chunk_transfer_pending("request")
    scheduler.complete_prefill_handoff("request")
    assert manager.group_request_partition("request") is None


def test_decode_adoption_skips_prefill_and_handles_first_token_boundary() -> None:
    manager = make_cache_manager()
    scheduler = _scheduler(manager)
    reservation = manager.reserve_group_cache("reservation", "request", 41, partition=1)
    manager.authorize_group_cache_write(reservation.reservation_id)
    manager.commit_group_cache(reservation.reservation_id, "manifest")
    request, first = scheduler.adopt_handoff(
        reservation_id="reservation",
        request_id="request",
        prompt_token_ids=list(range(33)),
        first_token=7,
        max_new_tokens=8,
    )
    assert not first.finished
    assert request.num_computed_tokens == 33
    assert request.output_token_ids == [7]
    assert not scheduler.waiting
    step = scheduler.schedule()
    scheduled = step.scheduled_requests[0]
    assert not scheduled.is_prefill
    assert scheduled.num_computed_tokens == 33
    assert scheduled.cache_partition == 1


def test_max_new_tokens_one_finishes_without_fabricating_decode() -> None:
    manager = make_cache_manager()
    scheduler = _scheduler(manager)
    reservation = manager.reserve_group_cache("reservation", "request", 34)
    manager.authorize_group_cache_write(reservation.reservation_id)
    manager.commit_group_cache(reservation.reservation_id, "manifest")
    _, first = scheduler.adopt_handoff(
        reservation_id="reservation",
        request_id="request",
        prompt_token_ids=list(range(33)),
        first_token=7,
        max_new_tokens=1,
    )
    assert first.finished
    assert scheduler.schedule().is_empty
