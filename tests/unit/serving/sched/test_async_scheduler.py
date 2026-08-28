# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto_serving.config.types import (
    KVCacheGroupSpec,
    KVCacheSpec,
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    ScheduledRequest,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)


def test_scheduler_rejects_speculative_depth_larger_than_token_budget():
    with pytest.raises(ValueError, match="one decode token"):
        SchedulerConfig(max_num_scheduled_tokens=4, num_speculative_tokens=4)


def test_scheduler_speculative_output_counts_only_tokens_retained_before_eos():
    manager = KvCacheManager(num_blocks=4, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=False), manager)
    request = Request(
        request_id="speculative",
        prompt_token_ids=[1],
        max_new_tokens=4,
        eos_token_id=7,
        num_computed_tokens=1,
        status=RequestStatus.RUNNING,
    )
    scheduler.running.append(request)
    scheduler.requests[request.request_id] = request
    scheduled = SchedulerOutput(
        scheduled_requests=[ScheduledRequest(request=request, num_new_tokens=1, is_prefill=False)]
    )

    outputs = scheduler.update_from_output(scheduled, {request.request_id: [7, 8]})

    assert request.output_token_ids == [7]
    assert request.num_computed_tokens == 2
    assert request.status is RequestStatus.FINISHED_EOS
    assert [(output.new_token_id, output.finished) for output in outputs] == [(7, True)]


def _running_decode_request(req_id="r", prompt=(1, 2), first_output=99):
    """A RUNNING request that finished prefill and has one decoded token, i.e.
    ready to schedule its next decode step (num_new_tokens_needed == 1)."""
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=8,
        num_computed_tokens=len(prompt),
        output_token_ids=[first_output],
        status=RequestStatus.RUNNING,
    )


def _prefill_scheduler(*, num_blocks=8, block_size=128, prefix_cache=False, **config):
    manager = KvCacheManager(num_blocks=num_blocks, block_size=block_size, enable_prefix_cache=prefix_cache)
    return Scheduler(SchedulerConfig(enable_prefix_cache=prefix_cache, **config), manager)


def _scheduled_tuple(item):
    return item.request.request_id, item.num_computed_tokens, item.num_new_tokens


def _drain_prefills(scheduler):
    waves = []
    first_waiting = []
    while scheduler.has_work():
        output = scheduler.schedule()
        assert output.scheduled_requests
        if not waves:
            first_waiting = [request.request_id for request in scheduler.waiting]
        waves.append([_scheduled_tuple(item) for item in output.scheduled_requests])
        sampled = {
            item.request.request_id: [7]
            for item in output.scheduled_requests
            if item.num_computed_tokens + item.num_new_tokens >= item.request.num_prompt_tokens
        }
        scheduler.update_from_output(output, sampled)
    return waves, first_waiting


def _assert_prefill_rejected(prompt_len, match, *, budget=512, **config):
    scheduler = _prefill_scheduler(
        max_num_scheduled_tokens=budget,
        max_prefill_tokens_per_request=128,
        max_seq_len=512,
        **config,
    )
    with pytest.raises(ValueError, match=match):
        scheduler.add_request(Request("too-long", list(range(prompt_len)), max_new_tokens=1))


def _scheduled_prefill_chunks(
    prompt_len: int,
    *,
    max_scheduled_tokens: int = 512,
    threshold: int = 2048,
    model_limit: int | None = 128,
    num_speculative_tokens: int = 1,
) -> list[tuple[int, int]]:
    max_seq_len = max(512, prompt_len + 1)
    scheduler = _prefill_scheduler(
        num_blocks=(max_seq_len + 127) // 128,
        max_num_scheduled_tokens=max_scheduled_tokens,
        long_prefill_token_threshold=threshold,
        max_prefill_tokens_per_request=model_limit,
        max_seq_len=max_seq_len,
        num_speculative_tokens=num_speculative_tokens,
        supports_chunked_prefill_with_speculation=True,
    )
    request = Request("chunked", list(range(prompt_len)), max_new_tokens=1, temperature=0.0)
    scheduler.add_request(request)
    waves, _ = _drain_prefills(scheduler)
    assert all(len(wave) == 1 for wave in waves)
    return [(wave[0][1], wave[0][2]) for wave in waves]


_DYNAMIC_PREFILL_OPTIONS = {"max_scheduled_tokens": 8192, "threshold": 8192, "model_limit": 8192}
_DYNAMIC_AR_PREFILL_OPTIONS = _DYNAMIC_PREFILL_OPTIONS | {"num_speculative_tokens": 0}


@pytest.mark.parametrize(
    ("prompt_len", "options", "expected"),
    [
        (127, {}, [(0, 127)]),
        (128, {}, [(0, 128)]),
        (129, {}, [(0, 128), (128, 1)]),
        (257, {}, [(0, 128), (128, 128), (256, 1)]),
        (129, {"threshold": 64}, [(0, 64), (64, 64), (128, 1)]),
        (129, {"model_limit": None, "num_speculative_tokens": 0}, [(0, 129)]),
        (257, _DYNAMIC_AR_PREFILL_OPTIONS, [(0, 257)]),
        (8192, _DYNAMIC_AR_PREFILL_OPTIONS, [(0, 8192)]),
        (8191, _DYNAMIC_PREFILL_OPTIONS, [(0, 8191)]),
        (8192, _DYNAMIC_PREFILL_OPTIONS, [(0, 8192)]),
        (8193, _DYNAMIC_PREFILL_OPTIONS, [(0, 8192), (8192, 1)]),
    ],
)
def test_scheduler_prefill_chunking_modes(prompt_len, options, expected):
    assert _scheduled_prefill_chunks(prompt_len, **options) == expected


@pytest.mark.parametrize(("budget", "prompt_len", "expected_limit"), [(512, 129, 128), (64, 80, 64)])
def test_scheduler_rejects_no_chunk_prefill_over_effective_limit(budget, prompt_len, expected_limit):
    _assert_prefill_rejected(
        prompt_len,
        f"single-dispatch prefill limit {expected_limit}",
        budget=budget,
        enable_chunk_prefill=False,
    )


def test_scheduler_rejects_multi_chunk_prefill_when_speculation_is_unsupported():
    _assert_prefill_rejected(
        129,
        "not supported with speculative decoding",
        num_speculative_tokens=1,
        supports_chunked_prefill_with_speculation=False,
    )


def test_single_dispatch_limit_counts_only_uncached_prompt_suffix():
    scheduler = _prefill_scheduler(
        num_blocks=8,
        block_size=2,
        prefix_cache=True,
        max_num_scheduled_tokens=4,
        max_prefill_tokens_per_request=4,
        max_seq_len=8,
        enable_chunk_prefill=False,
    )
    manager = scheduler.kv_cache_manager
    cached_block = manager.allocate_blocks(1)[0]
    manager.cache_block(cached_block, manager.compute_block_hashes([1, 2])[0])
    manager.release(cached_block)

    accepted = Request("accepted", [1, 2, 3, 4, 5, 6], max_new_tokens=1)
    scheduler.add_request(accepted)
    accepted_output = scheduler.schedule()

    assert [_scheduled_tuple(item) for item in accepted_output.scheduled_requests] == [
        ("accepted", 2, 4)
    ]
    scheduler.abort_request(accepted.request_id)

    rejected = Request("rejected", [1, 2, 3, 4, 5, 6, 7], max_new_tokens=1)
    scheduler.add_request(rejected)
    rejected_output = scheduler.schedule()

    assert rejected_output.is_empty
    assert "uncached prompt length 5" in rejected_output.rejected_requests[rejected.request_id]
    assert rejected.request_id not in scheduler.requests
    assert cached_block.ref_cnt == 0


@pytest.mark.parametrize(
    ("config_overrides", "expected_waves", "expected_first_waiting"),
    [
        pytest.param(
            {"num_speculative_tokens": 1, "supports_chunked_prefill_with_speculation": False},
            [[("first", 0, 80)], [("second", 0, 80)]],
            ["second"],
            id="speculation-requires-single-dispatch",
        ),
        pytest.param(
            {"enable_chunk_prefill": False, "long_prefill_token_threshold": 32},
            [[("first", 0, 80)], [("second", 0, 80)]],
            ["second"],
            id="chunk-prefill-disabled",
        ),
        pytest.param(
            {},
            [[("first", 0, 80), ("second", 0, 48)], [("second", 48, 32)]],
            [],
            id="chunk-prefill-enabled",
        ),
    ],
)
def test_scheduler_residual_budget_respects_single_dispatch_mode(
    config_overrides, expected_waves, expected_first_waiting
):
    scheduler = _prefill_scheduler(
        max_num_scheduled_tokens=128,
        max_prefill_tokens_per_request=128,
        max_seq_len=256,
        **config_overrides,
    )
    for request_id in ("first", "second"):
        scheduler.add_request(Request(request_id, list(range(80)), max_new_tokens=1, temperature=0.0))
    waves, first_waiting = _drain_prefills(scheduler)
    assert first_waiting == expected_first_waiting
    assert waves == expected_waves


def test_scheduler_releases_prefix_hit_when_no_chunk_prefill_is_deferred():
    scheduler = _prefill_scheduler(
        num_blocks=16,
        block_size=2,
        prefix_cache=True,
        max_num_scheduled_tokens=4,
        max_prefill_tokens_per_request=4,
        max_seq_len=8,
        enable_chunk_prefill=False,
    )
    manager = scheduler.kv_cache_manager
    cached_prompt = [1, 2, 3, 4]
    cached_block = manager.allocate_blocks(1)[0]
    manager.cache_block(cached_block, manager.compute_block_hashes(cached_prompt)[0])
    manager.release(cached_block)
    initial_free_blocks = manager.num_free_blocks

    first = Request("first", [9, 8, 7], max_new_tokens=1)
    deferred = Request("deferred", cached_prompt, max_new_tokens=1)
    scheduler.add_request(first)
    scheduler.add_request(deferred)

    first_output = scheduler.schedule()

    assert [_scheduled_tuple(item) for item in first_output.scheduled_requests] == [("first", 0, 3)]
    assert cached_block.ref_cnt == 0
    assert deferred.cached_block_ids == []
    assert deferred.num_computed_tokens == 0
    assert deferred.num_blocks_cached == 0

    scheduler.update_from_output(first_output, {"first": [7]})
    deferred_output = scheduler.schedule()

    assert [_scheduled_tuple(item) for item in deferred_output.scheduled_requests] == [("deferred", 2, 2)]

    scheduler.update_from_output(deferred_output, {"deferred": [7]})

    assert cached_block.ref_cnt == 0
    assert manager.num_free_blocks == initial_free_blocks


def test_async_reconciliation_matches_sync_end_state():
    """Driving N decode steps through the async path (schedule -> advance ->
    update_from_output) yields the same request state as the sync path."""

    def run(async_mode: bool):
        manager = KvCacheManager(num_blocks=16, block_size=2, enable_prefix_cache=False)
        scheduler = Scheduler(
            SchedulerConfig(enable_prefix_cache=False, async_scheduling=async_mode),
            manager,
        )
        request = _running_decode_request()
        scheduler.running.append(request)
        scheduler.requests[request.request_id] = request

        collected = []
        for step_token in (10, 11, 12):
            out = scheduler.schedule()
            if not out.scheduled_requests:
                break
            if async_mode:
                scheduler.advance_after_schedule(out)
            outs = scheduler.update_from_output(out, {request.request_id: [step_token]})
            collected.extend(o.new_token_id for o in outs if o.new_token_id is not None)
        return request.output_token_ids, request.num_computed_tokens, collected

    sync_out, sync_comp, sync_tokens = run(async_mode=False)
    async_out, async_comp, async_tokens = run(async_mode=True)

    assert async_out == sync_out == [99, 10, 11, 12]
    assert async_comp == sync_comp
    assert async_tokens == sync_tokens == [10, 11, 12]


def _mtp_scheduler(
    async_mode: bool,
    *,
    num_speculative_tokens: int = 1,
    max_seq_len: int = 4096,
):
    """Scheduler configured like an MTP (speculative) decoder."""
    manager = KvCacheManager(num_blocks=32, block_size=2, enable_prefix_cache=False)
    return Scheduler(
        SchedulerConfig(
            enable_prefix_cache=False,
            async_scheduling=async_mode,
            num_speculative_tokens=num_speculative_tokens,
            max_seq_len=max_seq_len,
        ),
        manager,
    )


def _mtp_request():
    """A greedy (temperature 0) decode-ready request — MTP only runs greedy."""
    request = _running_decode_request()
    request.temperature = 0.0
    return request


def test_async_mtp_reserves_max_tokens_per_step():
    """A speculative step can emit 1+num_speculative_tokens, so the optimistic
    advance must reserve that upper bound (block allocation already did)."""
    scheduler = _mtp_scheduler(async_mode=True, num_speculative_tokens=1)
    request = _mtp_request()
    scheduler.running.append(request)
    scheduler.requests[request.request_id] = request

    out = scheduler.schedule()
    assert out.scheduled_requests
    scheduler.advance_after_schedule(out)

    # Upper bound reserved: 1 base + 1 speculative.
    assert request.num_output_placeholders == 2
    # computed advanced by num_new_tokens (1) + the extra speculative slot (1).
    assert request.num_computed_tokens == 4

    boundary_scheduler = _mtp_scheduler(
        async_mode=True,
        num_speculative_tokens=7,
        max_seq_len=4,
    )
    boundary_request = _mtp_request()
    boundary_scheduler.running.append(boundary_request)
    boundary_scheduler.requests[boundary_request.request_id] = boundary_request

    boundary_out = boundary_scheduler.schedule()
    boundary_scheduler.advance_after_schedule(boundary_out)

    # Only one extra target-input position remains after the current token.
    assert boundary_request.num_output_placeholders == 2
    assert boundary_request.num_computed_tokens == 4


def test_async_mtp_matches_sync_when_all_tokens_accepted():
    """Both MTP tokens accepted: async end-state must equal the sync path."""

    def run(async_mode: bool):
        scheduler = _mtp_scheduler(async_mode)
        request = _mtp_request()
        scheduler.running.append(request)
        scheduler.requests[request.request_id] = request
        collected = []
        for pair in ([10, 11], [12, 13]):
            out = scheduler.schedule()
            if not out.scheduled_requests:
                break
            if async_mode:
                scheduler.advance_after_schedule(out)
            outs = scheduler.update_from_output(out, {request.request_id: pair})
            collected.extend(o.new_token_id for o in outs if o.new_token_id is not None)
        return request.output_token_ids, request.num_computed_tokens, collected, request

    sync_out, sync_comp, sync_tok, _ = run(False)
    async_out, async_comp, async_tok, async_req = run(True)

    assert async_out == sync_out == [99, 10, 11, 12, 13]
    assert async_comp == sync_comp
    assert async_tok == sync_tok
    assert async_req.num_output_placeholders == 0  # fully released


def test_async_mtp_subtracts_shortfall_on_rejection():
    """When the speculative token is REJECTED (only 1 token returned), the
    optimistically-advanced position must be given back so async == sync."""

    def run(async_mode: bool):
        scheduler = _mtp_scheduler(async_mode)
        request = _mtp_request()
        scheduler.running.append(request)
        scheduler.requests[request.request_id] = request
        for tok in ([10], [11]):  # 1 token per step = draft rejected
            out = scheduler.schedule()
            if not out.scheduled_requests:
                break
            if async_mode:
                scheduler.advance_after_schedule(out)
            scheduler.update_from_output(out, {request.request_id: tok})
        return request.output_token_ids, request.num_computed_tokens, request

    sync_out, sync_comp, _ = run(False)
    async_out, async_comp, async_req = run(True)

    assert async_out == sync_out == [99, 10, 11]
    # The rejected speculative slot was reclaimed — no permanent desync.
    assert async_comp == sync_comp
    assert async_req.num_output_placeholders == 0


def test_async_completing_prefill_keeps_its_computed_tokens():
    """A prefill chunk's own KV work must never be reverted by the shortfall.

    Regression: the shortfall reclaimed `reserved - retained`, which for a
    completing prefill chunk that returned no token clawed back the chunk's own
    num_new_tokens. The chunk was then re-scheduled and prefilled twice, and the
    model sampled the same token twice (seen on device as duplicated tokens with
    chunked prefill).
    """
    for returned_tokens in ([], [100]):
        manager = KvCacheManager(num_blocks=64, block_size=2, enable_prefix_cache=False)
        scheduler = Scheduler(
            SchedulerConfig(
                enable_prefix_cache=False,
                async_scheduling=True,
                long_prefill_token_threshold=2,
                enable_chunk_prefill=True,
            ),
            manager,
        )
        # 4 of 5 prompt tokens already computed: this chunk completes the prompt.
        request = Request(
            request_id="r",
            prompt_token_ids=[1, 2, 3, 4, 5],
            max_new_tokens=4,
            num_computed_tokens=4,
            temperature=0.0,
            status=RequestStatus.RUNNING,
        )
        scheduler.running.append(request)
        scheduler.requests[request.request_id] = request

        out = scheduler.schedule()
        assert out.scheduled_requests and out.scheduled_requests[0].is_prefill
        scheduler.advance_after_schedule(out)
        assert request.num_computed_tokens == 5  # prompt fully computed

        payload = {request.request_id: returned_tokens} if returned_tokens else {}
        scheduler.update_from_output(out, payload)

        # The chunk's KV work is retained either way — never reverted to 4.
        assert request.num_computed_tokens == 5, (
            f"completing prefill reverted its own computed tokens (returned_tokens={returned_tokens})"
        )
        assert request.num_output_placeholders == 0
        # And it is NOT re-scheduled as prefill again.
        again = scheduler.schedule()
        if again.scheduled_requests:
            assert not again.scheduled_requests[0].is_prefill


def test_async_terminal_prefill_blocks_first_decode_until_confirmed():
    manager = KvCacheManager(num_blocks=16, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(
        SchedulerConfig(
            enable_prefix_cache=False,
            async_scheduling=True,
            num_speculative_tokens=1,
        ),
        manager,
    )
    request = Request(
        request_id="r",
        prompt_token_ids=[1, 2],
        max_new_tokens=4,
    )
    scheduler.add_request(request)

    terminal_prefill = scheduler.schedule()
    assert terminal_prefill.scheduled_requests[0].is_prefill
    scheduler.advance_after_schedule(terminal_prefill)

    assert request.terminal_prefill_in_flight
    assert scheduler.schedule().is_empty

    scheduler.update_from_output(terminal_prefill, {request.request_id: [10]})

    assert not request.terminal_prefill_in_flight
    first_decode = scheduler.schedule()
    assert len(first_decode.scheduled_requests) == 1
    assert not first_decode.scheduled_requests[0].is_prefill


def test_async_terminal_prefill_barrier_does_not_block_ready_requests():
    manager = KvCacheManager(num_blocks=16, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(
        SchedulerConfig(
            enable_prefix_cache=False,
            async_scheduling=True,
            num_speculative_tokens=1,
        ),
        manager,
    )
    pending = _running_decode_request(req_id="pending")
    pending.terminal_prefill_in_flight = True
    ready = _running_decode_request(req_id="ready")
    scheduler.running.extend((pending, ready))
    scheduler.requests.update({request.request_id: request for request in scheduler.running})

    output = scheduler.schedule()

    assert [scheduled.request.request_id for scheduled in output.scheduled_requests] == ["ready"]


def test_async_mtp_shortfall_on_eos_mid_pair():
    """EOS in the first of two returned tokens: the second is dropped (as in the
    sync path) and its optimistic position reclaimed."""
    scheduler = _mtp_scheduler(async_mode=True)
    request = _mtp_request()
    request.eos_token_id = 7
    scheduler.running.append(request)
    scheduler.requests[request.request_id] = request

    out = scheduler.schedule()
    scheduler.advance_after_schedule(out)
    assert request.num_output_placeholders == 2

    # Worker returns [EOS, extra]: only EOS is retained.
    scheduler.update_from_output(out, {request.request_id: [7, 8]})

    assert request.output_token_ids == [99, 7]  # 8 dropped after EOS
    assert request.status is RequestStatus.FINISHED_EOS
    assert request.num_output_placeholders == 0


def test_async_discards_stale_result_for_preempted_request():
    """A request preempted while its step is in flight must NOT have that step's
    result applied: preemption reset its computed/placeholder state, so appending
    the stale token would corrupt bookkeeping and emit a spurious output."""
    manager = KvCacheManager(num_blocks=8, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=False, async_scheduling=True), manager)
    request = _running_decode_request()
    scheduler.running.append(request)
    scheduler.requests[request.request_id] = request

    out = scheduler.schedule()
    scheduler.advance_after_schedule(out)  # step N in flight

    # Preemption (as _preempt_lowest_priority does) resets state and marks the
    # request PREEMPTED before step N's result returns.
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    request.num_output_placeholders = 0

    outputs = scheduler.update_from_output(out, {request.request_id: [42]})

    # Stale token discarded: no output emitted, state untouched by reconcile.
    assert outputs == []
    assert request.output_token_ids == [99]  # unchanged (42 not appended)
    assert request.num_computed_tokens == 0  # reset preserved
    assert request.num_output_placeholders == 0


def test_async_defers_prefix_cache_publish_until_confirmed():
    """Prefix-cache blocks must be published only after the worker confirms the
    step, not optimistically at dispatch — otherwise a failed step leaves hashes
    for uncomputed KV that a later same-prompt request could hit."""
    manager = KvCacheManager(num_blocks=16, block_size=2, enable_prefix_cache=True)
    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=True, async_scheduling=True), manager)
    # Fresh prompt long enough to complete >=1 cache block on prefill.
    prompt = [5, 6, 7, 8]
    request = Request(
        request_id="p",
        prompt_token_ids=prompt,
        max_new_tokens=4,
        status=RequestStatus.WAITING,
    )
    scheduler.add_request(request)

    out = scheduler.schedule()
    assert out.scheduled_requests and out.scheduled_requests[0].is_prefill
    scheduler.advance_after_schedule(out)

    # advance_after_schedule advanced computed tokens but must NOT have published
    # any prefix-cache blocks yet.
    assert scheduler.kv_cache_manager.get_computed_blocks(prompt) == []
    assert request.num_blocks_cached == 0

    # After the worker confirms, blocks are published.
    scheduler.update_from_output(out, {request.request_id: [42]})
    assert request.num_blocks_cached >= 1


def test_grouped_cache_preemption_removes_victim_from_running_queue():
    manager = KvCacheManager(block_size=1, enable_prefix_cache=False)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                name="test",
                layer_indices=(0,),
                spec=KVCacheSpec(block_size=1, page_size_bytes=1),
                max_blocks_per_seq=3,
                num_blocks=3,
            ),
        ),
        max_batch_size=2,
    )
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_scheduled_tokens=4,
            enable_prefix_cache=False,
            num_speculative_tokens=1,
        ),
        manager,
    )
    requests = [
        Request(
            request_id=request_id,
            prompt_token_ids=[1],
            max_new_tokens=5,
            arrival_time=arrival_time,
            status=RequestStatus.RUNNING,
            num_computed_tokens=1,
            output_token_ids=[2],
            temperature=0.0,
        )
        for request_id, arrival_time in (("older", 1.0), ("newer", 2.0))
    ]
    for request in requests:
        request.allocated_group_block_ids = manager.ensure_group_blocks(request.request_id, 1)
        request.cache_partition = 0
        scheduler.requests[request.request_id] = request
    scheduler.running = requests

    output = scheduler.schedule()

    assert [request.request_id for request in output.preempted_requests] == ["newer"]
    assert [request.request_id for request in scheduler.running] == ["older"]
    assert [request.request_id for request in scheduler.waiting] == ["newer"]


def test_grouped_cache_capacity_scales_from_device_reported_primary_pool():
    manager = KvCacheManager(block_size=1, enable_prefix_cache=False)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                name="primary",
                layer_indices=(0,),
                spec=KVCacheSpec(block_size=1, page_size_bytes=4),
                max_blocks_per_seq=3,
            ),
            KVCacheGroupSpec(
                name="compressed",
                layer_indices=(1,),
                spec=KVCacheSpec(block_size=1, page_size_bytes=2),
                max_blocks_per_seq=2,
            ),
        ),
        max_batch_size=8,
        primary_num_blocks=6,
    )

    assert manager.group_num_blocks("primary") == 6
    assert manager.group_num_blocks("compressed") == 4


def test_eagle_group_reuses_every_page_with_a_known_boundary_token():
    manager = KvCacheManager(block_size=2, enable_prefix_cache=True)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                name="eagle",
                layer_indices=(0,),
                spec=KVCacheSpec(block_size=2, page_size_bytes=1),
                max_blocks_per_seq=4,
                num_blocks=4,
                is_eagle_group=True,
            ),
        ),
        max_batch_size=1,
    )
    prompt = [1, 2, 3, 4, 5]
    hashes = manager.compute_group_block_hashes(prompt)
    manager.ensure_group_blocks("warm", len(prompt), partition=0)
    published = manager.cache_group_blocks("warm", hashes, len(prompt), {})
    manager.release_all_group_requests("warm")

    blocks, hit_tokens, partition = manager.acquire_group_prefix_blocks(
        "hit",
        hashes,
        max_cache_hit_tokens=len(prompt) - 1,
    )

    assert published == {"eagle": 2}
    assert hit_tokens == 4
    assert partition == 0
    assert len(blocks["eagle"]) == 2


def test_grouped_prefix_hit_falls_back_to_an_idle_partition_when_suffix_does_not_fit():
    manager = KvCacheManager(block_size=2, enable_prefix_cache=True)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                name="test",
                layer_indices=(0,),
                spec=KVCacheSpec(block_size=2, page_size_bytes=1),
                max_blocks_per_seq=2,
                num_blocks=2,
                num_partitions=2,
            ),
        ),
        max_batch_size=2,
    )
    prompt = [1, 2, 3, 4]
    hashes = manager.compute_group_block_hashes(prompt)
    manager.ensure_group_blocks("warm", 2, partition=0)
    manager.cache_group_blocks("warm", hashes, 2, {})
    manager.release_all_group_requests("warm")
    manager.ensure_group_blocks("partition-0-blocker", 2, partition=0)
    _, probe_hit_tokens, probe_partition = manager.acquire_group_prefix_blocks(
        "probe",
        hashes,
        max_cache_hit_tokens=len(prompt) - 1,
    )
    assert probe_hit_tokens == 2
    assert probe_partition == 0
    manager.release_all_group_requests("probe")

    scheduler = Scheduler(
        SchedulerConfig(
            max_num_scheduled_tokens=4,
            max_seq_len=16,
            enable_prefix_cache=True,
        ),
        manager,
    )
    request = Request(
        request_id="fallback",
        prompt_token_ids=prompt,
        max_new_tokens=1,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert len(output.scheduled_requests) == 1
    assert output.scheduled_requests[0].num_computed_tokens == 0
    assert output.scheduled_requests[0].num_new_tokens == len(prompt)
    assert request.cache_partition == 1
