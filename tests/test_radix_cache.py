# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pypto_serving.cli.main import build_parser
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.memory.prefix_cache import RadixCache, RadixKey
from pypto_serving.serving.memory.request_kv_pool import RequestKVPool
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


PAGE_SIZE = 4


def _slots(*page_ids: int) -> list[int]:
    return [page_id * PAGE_SIZE + offset for page_id in page_ids for offset in range(PAGE_SIZE)]


def _scheduler(
    backend: str,
    *,
    num_blocks: int = 64,
    in_batch_dedup: bool = True,
) -> Scheduler:
    manager = KvCacheManager(num_blocks=num_blocks, block_size=PAGE_SIZE, enable_prefix_cache=True)
    config = SchedulerConfig(
        max_num_running_reqs=16,
        max_num_scheduled_tokens=256,
        long_prefill_token_threshold=256,
        max_seq_len=256,
        prefix_cache_backend=backend,
        enable_radix_in_batch_dedup=in_batch_dedup,
    )
    return Scheduler(config, manager)


def _run_generation(scheduler: Scheduler, request: Request, output_tokens: list[int]) -> None:
    scheduler.add_request(request)
    output_index = 0
    while request.request_id in {running.request_id for running in scheduler.running} or any(
        waiting.request_id == request.request_id for waiting in scheduler.waiting
    ):
        scheduled = scheduler.schedule()
        assert len(scheduled.scheduled_requests) == 1
        token_map: dict[str, int] = {}
        scheduled_request = scheduled.scheduled_requests[0]
        reaches_decode_head = (
            scheduled_request.num_computed_tokens + scheduled_request.num_new_tokens
            >= request.num_prompt_tokens
        )
        if reaches_decode_head:
            token_map[request.request_id] = output_tokens[output_index]
            output_index += 1
        scheduler.update_from_output(scheduled, token_map)
    assert request.output_token_ids == output_tokens


def test_radix_tree_matches_splits_namespaces_and_page_alignment() -> None:
    retained: list[int] = []
    released: list[int] = []
    cache = RadixCache(
        PAGE_SIZE,
        retain_pages=lambda pages: retained.extend(pages),
        release_pages=lambda pages: released.extend(pages),
    )
    tokens = list(range(12))
    inserted = cache.insert(RadixKey.from_tokens(tokens, extra_key=("model-a",)), _slots(3, 4, 5))

    assert inserted.inserted_len == 12
    assert retained == [3, 4, 5]
    assert cache.match_prefix(
        RadixKey.from_tokens(tokens[:8] + [100, 101, 102, 103], extra_key=("model-a",))
    ).device_indices == tuple(_slots(3, 4))
    assert len(
        cache.match_prefix(RadixKey.from_tokens(tokens[:10], extra_key=("model-a",))).device_indices
    ) == 8
    assert not cache.match_prefix(
        RadixKey.from_tokens(tokens, extra_key=("model-b",))
    ).device_indices

    cache.reset()
    assert sorted(released) == [3, 4, 5]
    assert cache.total_size() == 0


def test_radix_lock_protects_matched_path_and_lru_evicts_leaf() -> None:
    cache = RadixCache.create_simulated(PAGE_SIZE)
    tokens = list(range(12))
    cache.insert(RadixKey.from_tokens(tokens), _slots(0, 1, 2))
    match = cache.match_prefix(RadixKey.from_tokens(tokens[:8]))

    cache.inc_lock_ref(match.last_device_node)
    assert cache.protected_size() == 8
    assert cache.evict(12).num_tokens_evicted == 4
    assert cache.total_size() == 8

    cache.dec_lock_ref(match.last_device_node)
    assert cache.protected_size() == 0
    assert cache.evict(8).num_tokens_evicted == 8
    assert cache.total_size() == 0


def test_request_kv_pool_maps_pages_to_logical_slots() -> None:
    pool = RequestKVPool(PAGE_SIZE)
    pool.set_pages("request", [3, 7])
    pool.extend_pages("request", [9])

    assert pool.capacity("request") == 12
    assert pool.slot_indices("request", 6) == [12, 13, 14, 15, 28, 29]
    assert pool.page_ids_from_slots(_slots(3, 7)) == [3, 7]
    assert pool.blocks_needed("request", 13) == 1
    assert pool.free("request") == [3, 7, 9]


def test_manager_uses_separate_request_and_radix_ownership() -> None:
    manager = KvCacheManager(num_blocks=3, block_size=PAGE_SIZE)
    pages = manager.allocate_block_ids(2)
    assert pages == [0, 1]
    manager.insert_radix_prefix(list(range(8)), _slots(0, 1))

    manager.release_pages_from_request(pages)
    assert manager.num_free_blocks == 1
    assert [manager.blocks[index].radix_ref_cnt for index in pages] == [1, 1]

    replacement = manager.allocate_block_ids(2)
    assert replacement is not None
    assert manager.radix_cache.total_size() == 0
    assert manager.num_free_blocks == 1


def test_scheduler_reuses_page_aligned_radix_prefix() -> None:
    scheduler = _scheduler("radix")
    cold = Request("cold", list(range(10)), 1, model_id="model")
    _run_generation(scheduler, cold, [100])

    warm = Request("warm", list(range(10)), 1, model_id="model")
    scheduler.add_request(warm)
    scheduled = scheduler.schedule()
    warm_step = scheduled.scheduled_requests[0]

    assert warm_step.num_computed_tokens == 8
    assert warm_step.num_new_tokens == 2
    assert warm.cached_block_ids == [0, 1]


def test_chunked_prefill_publishes_only_completed_pages() -> None:
    manager = KvCacheManager(num_blocks=16, block_size=PAGE_SIZE)
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=4,
            max_num_scheduled_tokens=3,
            long_prefill_token_threshold=3,
            max_seq_len=64,
            prefix_cache_backend="radix",
        ),
        manager,
    )
    request = Request("chunked", list(range(10)), 1, model_id="model")
    scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.update_from_output(first, {})
    assert manager.radix_cache.total_size() == 0
    assert request.cache_protected_len == 0
    assert len(request.prefix_indices) == 3

    second = scheduler.schedule()
    scheduler.update_from_output(second, {})
    assert manager.radix_cache.total_size() == 4
    assert request.cache_protected_len == 4
    assert len(request.prefix_indices) == 6

    third = scheduler.schedule()
    scheduler.update_from_output(third, {})
    assert manager.radix_cache.total_size() == 8

    final = scheduler.schedule()
    scheduler.update_from_output(final, {"chunked": 100})
    assert manager.radix_cache.total_size() == 8
    assert manager.radix_cache.protected_size() == 0
    assert manager.num_free_blocks == 14

    warm = Request("chunked-warm", list(range(10)), 1, model_id="model")
    scheduler.add_request(warm)
    warm_step = scheduler.schedule().scheduled_requests[0]
    assert warm_step.num_computed_tokens == 8


def test_async_radix_publishes_only_worker_confirmed_pages() -> None:
    manager = KvCacheManager(num_blocks=16, block_size=PAGE_SIZE)
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=4,
            max_num_scheduled_tokens=3,
            long_prefill_token_threshold=3,
            max_seq_len=64,
            prefix_cache_backend="radix",
            async_scheduling=True,
        ),
        manager,
    )
    request = Request("async-chunked", list(range(10)), 1, model_id="model")
    scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.advance_after_schedule(first)
    second = scheduler.schedule()
    scheduler.advance_after_schedule(second)

    # The second chunk is already reflected in the optimistic scheduler state,
    # but neither worker result has returned, so no page is visible yet.
    assert request.num_computed_tokens == 6
    assert manager.radix_cache.total_size() == 0

    # Confirming only the first three tokens still cannot publish a full page.
    scheduler.update_from_output(first, {})
    assert manager.radix_cache.total_size() == 0

    # Once the second result arrives, exactly the first complete page is safe.
    scheduler.update_from_output(second, {})
    assert manager.radix_cache.total_size() == PAGE_SIZE
    assert request.kv_committed_len == PAGE_SIZE


def test_async_radix_defers_follower_until_producer_page_is_confirmed() -> None:
    manager = KvCacheManager(num_blocks=16, block_size=PAGE_SIZE)
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=4,
            max_num_scheduled_tokens=8,
            long_prefill_token_threshold=4,
            max_seq_len=64,
            prefix_cache_backend="radix",
            async_scheduling=True,
        ),
        manager,
    )
    producer = Request("producer", list(range(10)), 1, model_id="model")
    scheduler.add_request(producer)

    first = scheduler.schedule()
    scheduler.advance_after_schedule(first)
    assert producer.num_computed_tokens == PAGE_SIZE
    assert producer.kv_committed_len == 0

    follower = Request("follower", list(range(8)) + [100, 101], 1, model_id="model")
    scheduler.add_request(follower)
    second = scheduler.schedule()

    # The producer has optimistically advanced one page, but its first worker
    # result is still pending. The follower must wait rather than recompute it.
    assert [step.request.request_id for step in second.scheduled_requests] == ["producer"]
    assert scheduler.prefix_cache_stats.in_batch_deferred == 1
    assert [request.request_id for request in scheduler.waiting] == ["follower"]


def test_async_radix_publishes_retained_speculative_tokens() -> None:
    manager = KvCacheManager(num_blocks=16, block_size=PAGE_SIZE)
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=4,
            max_num_scheduled_tokens=16,
            long_prefill_token_threshold=16,
            max_seq_len=64,
            prefix_cache_backend="radix",
            num_speculative_tokens=2,
            async_scheduling=True,
        ),
        manager,
    )
    request = Request(
        "async-speculative",
        list(range(6)),
        5,
        model_id="model",
        temperature=0.0,
    )
    scheduler.add_request(request)

    prefill = scheduler.schedule()
    scheduler.advance_after_schedule(prefill)
    scheduler.update_from_output(prefill, {request.request_id: [100]})
    assert request.kv_committed_len == PAGE_SIZE

    decode = scheduler.schedule()
    decode_step = decode.scheduled_requests[0]
    assert decode_step.num_computed_tokens == 6
    scheduler.advance_after_schedule(decode)
    scheduler.update_from_output(decode, {request.request_id: [101, 102]})

    # The base decode position plus one accepted speculative position are now
    # confirmed, extending the reusable generated history to two full pages.
    assert request.num_computed_tokens == 8
    assert request.kv_committed_len == 2 * PAGE_SIZE
    assert manager.radix_cache.total_size() == 2 * PAGE_SIZE


def test_radix_reuses_committed_generated_output_beyond_hash_prompt_cache() -> None:
    prompt = list(range(6))
    generated = [100, 101, 102, 103, 104]

    radix = _scheduler("radix")
    _run_generation(radix, Request("radix-source", prompt, len(generated), model_id="model"), generated)
    radix_followup = Request("radix-followup", prompt + generated, 1, model_id="model")
    radix.add_request(radix_followup)
    radix_match = radix.schedule().scheduled_requests[0].num_computed_tokens

    hashed = _scheduler("hash")
    _run_generation(hashed, Request("hash-source", prompt, len(generated), model_id="model"), generated)
    hash_followup = Request("hash-followup", prompt + generated, 1, model_id="model")
    hashed.add_request(hash_followup)
    hash_match = hashed.schedule().scheduled_requests[0].num_computed_tokens

    assert radix_match == 8
    assert hash_match == 4
    assert radix_match > hash_match


def test_radix_in_batch_producer_reduces_cold_shared_prefix_prefill() -> None:
    prompts = [list(range(12)) + [100 + index] for index in range(4)]

    hashed = _scheduler("hash")
    for index, prompt in enumerate(prompts):
        hashed.add_request(Request(f"hash-{index}", prompt, 1, model_id="model"))
    hash_first_step = hashed.schedule()
    hash_prefill_tokens = hash_first_step.num_prefill_tokens

    radix = _scheduler("radix")
    for index, prompt in enumerate(prompts):
        radix.add_request(Request(f"radix-{index}", prompt, 1, model_id="model"))
    radix_first_step = radix.schedule()
    assert len(radix_first_step.scheduled_requests) == 1
    radix.update_from_output(radix_first_step, {"radix-0": 200})
    radix_second_step = radix.schedule()
    radix_prefill_tokens = radix_first_step.num_prefill_tokens + radix_second_step.num_prefill_tokens

    assert len(radix_second_step.scheduled_requests) == 3
    assert all(step.num_computed_tokens == 12 for step in radix_second_step.scheduled_requests)
    assert radix.prefix_cache_stats.in_batch_deferred == 3
    assert radix_prefill_tokens == 16
    assert hash_prefill_tokens == 52
    assert radix_prefill_tokens * 2 < hash_prefill_tokens


def test_radix_canonicalizes_duplicate_pages_after_concurrent_prefill() -> None:
    scheduler = _scheduler("radix", num_blocks=20, in_batch_dedup=False)
    scheduler.add_request(Request("left", [1, 2, 3, 4, 10, 11, 12, 13], 1, model_id="model"))
    scheduler.add_request(Request("right", [1, 2, 3, 4, 20, 21, 22, 23], 1, model_id="model"))
    scheduled = scheduler.schedule()
    original_pages = [page for step in scheduled.scheduled_requests for page in step.block_ids]
    assert len(set(original_pages)) == 4

    scheduler.update_from_output(scheduled, {"left": 100, "right": 101})

    manager = scheduler.kv_cache_manager
    assert scheduler.prefix_cache_stats.duplicate_pages_freed == 1
    assert manager.radix_cache.total_size() == 12
    assert len(set(slot // PAGE_SIZE for slot in manager.radix_cache.all_values_flatten())) == 3
    assert manager.num_free_blocks == 17
    assert all(block.active_ref_cnt == 0 for block in manager.blocks)


def test_abort_releases_private_radix_pages_and_path_lock() -> None:
    scheduler = _scheduler("radix", num_blocks=8)
    request = Request("abort", list(range(10)), 2, model_id="model")
    scheduler.add_request(request)
    scheduler.schedule()
    assert scheduler.kv_cache_manager.num_free_blocks == 5

    scheduler.abort_request(request.request_id)

    assert scheduler.kv_cache_manager.num_free_blocks == 8
    assert scheduler.kv_cache_manager.radix_cache.protected_size() == 0
    assert not scheduler.request_kv_pool.has_request(request.request_id)


def test_cli_exposes_hash_compatible_default_and_radix_opt_in() -> None:
    parser = build_parser()

    assert parser.parse_args(["--model", "/tmp/model"]).prefix_cache_backend == "hash"
    assert parser.parse_args(
        ["--model", "/tmp/model", "--prefix-cache-backend", "radix"]
    ).prefix_cache_backend == "radix"
