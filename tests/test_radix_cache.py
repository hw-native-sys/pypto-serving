# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from python.core.radix_cache import RadixKey, RadixPrefixCache
from python.core.kv_cache import KvCacheManager
from python.core.scheduler import Request, Scheduler, SchedulerConfig


def _key(tokens: list[int], extra_key: tuple[str, ...] = ("model",)) -> RadixKey:
    return RadixKey.from_tokens(tokens, extra_key=extra_key)


def test_radix_cache_matches_inserted_prefix():
    cache = RadixPrefixCache(page_size=2)

    result = cache.insert(_key([1, 2, 3, 4]), [10, 11])
    match = cache.match(_key([1, 2, 3, 4, 5, 6]))

    assert result.existing_prefix_len == 0
    assert result.inserted_len == 4
    assert match.prefix_len == 4
    assert match.page_ids == [10, 11]


def test_radix_cache_splits_node_on_page_aligned_partial_match():
    cache = RadixPrefixCache(page_size=2)
    cache.insert(_key([1, 2, 3, 4, 5, 6]), [10, 11, 12])

    match = cache.match(_key([1, 2, 3, 4, 9, 9]))
    insert = cache.insert(_key([1, 2, 3, 4, 9, 9]), [10, 11, 13])
    branch_match = cache.match(_key([1, 2, 3, 4, 9, 9, 7, 7]))
    original_match = cache.match(_key([1, 2, 3, 4, 5, 6]))

    assert match.prefix_len == 4
    assert match.page_ids == [10, 11]
    assert insert.existing_prefix_len == 4
    assert insert.inserted_len == 2
    assert branch_match.prefix_len == 6
    assert branch_match.page_ids == [10, 11, 13]
    assert original_match.prefix_len == 6
    assert original_match.page_ids == [10, 11, 12]


def test_radix_cache_page_aligns_keys_and_values():
    cache = RadixPrefixCache(page_size=4)

    result = cache.insert(_key([1, 2, 3, 4, 5, 6]), [20, 21])
    match = cache.match(_key([1, 2, 3, 4, 7, 8]))

    assert result.inserted_len == 4
    assert cache.total_pages() == 1
    assert match.prefix_len == 4
    assert match.page_ids == [20]


def test_radix_cache_extra_key_isolates_prefixes():
    cache = RadixPrefixCache(page_size=2)

    cache.insert(_key([1, 2, 3, 4], ("model-a",)), [10, 11])

    assert cache.match(_key([1, 2, 3, 4], ("model-a",))).page_ids == [10, 11]
    assert cache.match(_key([1, 2, 3, 4], ("model-b",))).page_ids == []


def test_radix_cache_retain_and_release_callbacks():
    retained: list[int] = []
    released: list[int] = []
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=lambda page_ids: retained.extend(page_ids),
        release_pages=lambda page_ids: released.extend(page_ids),
    )

    cache.insert(_key([1, 2, 3, 4]), [10, 11])
    match = cache.match(_key([1, 2, 3, 4]))

    assert retained == [10, 11]
    assert cache.protected_pages() == 2
    assert cache.evict_pages(1) == 0

    cache.dec_lock_ref(match.last_node)

    assert cache.evict_pages(1) == 2
    assert released == [10, 11]
    assert cache.total_pages() == 0


def test_radix_cache_evicts_parent_after_leaf_batch():
    released: list[int] = []
    cache = RadixPrefixCache(page_size=2, release_pages=lambda page_ids: released.extend(page_ids))
    cache.insert(_key([1, 2, 3, 4]), [10, 11])
    cache.insert(_key([1, 2, 5, 6]), [10, 12])

    assert cache.evict_pages(3) == 3
    assert sorted(released) == [10, 11, 12]
    assert cache.total_pages() == 0


def test_radix_cache_rejects_missing_page_ids():
    cache = RadixPrefixCache(page_size=2)

    with pytest.raises(ValueError, match="need at least 2"):
        cache.insert(_key([1, 2, 3, 4]), [10])


def test_radix_owned_pages_are_evicted_for_new_allocations():
    manager = KvCacheManager(num_blocks=2, block_size=2)
    cached_pages = manager.allocate_block_ids(2)
    assert cached_pages is not None
    manager.insert_radix_prefix("model", [1, 2, 3, 4], cached_pages)
    manager.release_pages_from_request(cached_pages)

    assert manager.num_free_blocks == 0

    allocated = manager.allocate_block_ids(1)

    assert allocated is not None
    assert len(allocated) == 1
    assert manager.blocks[allocated[0]].ref_cnt == 1
    assert manager.radix_cache.total_pages() == 0


def test_scheduler_uses_radix_prefix_for_suffix_prefill():
    manager = KvCacheManager(num_blocks=8, block_size=2)
    cached_pages = manager.allocate_block_ids(2)
    assert cached_pages is not None
    manager.insert_radix_prefix("model", [1, 2, 3, 4], cached_pages)
    manager.release_pages_from_request(cached_pages)

    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=1,
            max_num_scheduled_tokens=16,
            max_seq_len=8,
            enable_prefix_cache=True,
            prefix_cache_backend="radix",
        ),
        manager,
    )
    request = Request(
        request_id="req-0",
        model_id="model",
        prompt_token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=1,
    )

    scheduler.add_request(request)
    scheduled = scheduler.schedule()

    assert len(scheduled.scheduled_requests) == 1
    sr = scheduled.scheduled_requests[0]
    assert sr.num_computed_tokens == 4
    assert sr.num_new_tokens == 1
    assert sr.block_ids[:2] == cached_pages
    assert request.cached_block_ids == cached_pages
    assert len(request.allocated_block_ids) == 1
    assert manager.blocks[cached_pages[0]].ref_cnt == 2

    scheduler.abort_request(request.request_id)

    assert manager.blocks[cached_pages[0]].ref_cnt == 1


def test_scheduler_allocation_can_evict_unlocked_radix_pages():
    manager = KvCacheManager(num_blocks=2, block_size=2)
    cached_pages = manager.allocate_block_ids(2)
    assert cached_pages is not None
    manager.insert_radix_prefix("model", [1, 2, 3, 4], cached_pages)
    manager.release_pages_from_request(cached_pages)

    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=1,
            max_num_scheduled_tokens=16,
            max_seq_len=8,
            enable_prefix_cache=True,
            prefix_cache_backend="radix",
        ),
        manager,
    )
    request = Request(
        request_id="req-0",
        model_id="model",
        prompt_token_ids=[9, 9],
        max_new_tokens=1,
    )

    scheduler.add_request(request)
    scheduled = scheduler.schedule()

    assert len(scheduled.scheduled_requests) == 1
    assert request.allocated_block_ids
    assert manager.radix_cache.total_pages() == 0

    scheduler.abort_request(request.request_id)


def test_scheduler_default_hash_backend_ignores_radix_cache():
    manager = KvCacheManager(num_blocks=8, block_size=2)
    cached_pages = manager.allocate_block_ids(2)
    assert cached_pages is not None
    manager.insert_radix_prefix("model", [1, 2, 3, 4], cached_pages)
    manager.release_pages_from_request(cached_pages)

    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=1,
            max_num_scheduled_tokens=16,
            max_seq_len=8,
            enable_prefix_cache=True,
        ),
        manager,
    )
    request = Request(
        request_id="req-0",
        model_id="model",
        prompt_token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=1,
    )

    scheduler.add_request(request)
    scheduled = scheduler.schedule()

    assert scheduler.config.prefix_cache_backend == "hash"
    assert len(scheduled.scheduled_requests) == 1
    sr = scheduled.scheduled_requests[0]
    assert sr.num_computed_tokens == 0
    assert sr.num_new_tokens == 5
    assert request.cached_block_ids == []
    assert sr.block_ids[:2] != cached_pages

    scheduler.abort_request(request.request_id)

    assert manager.blocks[cached_pages[0]].ref_cnt == 1


def test_scheduler_radix_insert_uses_only_completed_full_pages_for_chunked_prefill():
    manager = KvCacheManager(num_blocks=8, block_size=2)
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=1,
            max_num_scheduled_tokens=16,
            long_prefill_token_threshold=3,
            max_seq_len=8,
            enable_prefix_cache=True,
            enable_chunk_prefill=True,
            prefix_cache_backend="radix",
        ),
        manager,
    )
    request = Request(
        request_id="req-0",
        model_id="model",
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
        max_new_tokens=1,
    )

    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    scheduler.update_from_output(scheduled, {})

    assert request.num_computed_tokens == 3
    assert manager.radix_cache.total_pages() == 1
    match = manager.radix_cache.match(_key([1, 2, 3, 4]))
    assert match.prefix_len == 2
    assert len(match.page_ids) == 1

    manager.radix_cache.dec_lock_ref(match.last_node)
    scheduler.abort_request(request.request_id)


def test_scheduler_releases_radix_match_when_suffix_allocation_fails():
    manager = KvCacheManager(num_blocks=2, block_size=2)
    cached_pages = manager.allocate_block_ids(2)
    assert cached_pages is not None
    manager.insert_radix_prefix("model", [1, 2, 3, 4], cached_pages)
    manager.release_pages_from_request(cached_pages)

    scheduler = Scheduler(
        SchedulerConfig(
            max_num_running_reqs=1,
            max_num_scheduled_tokens=16,
            max_seq_len=8,
            enable_prefix_cache=True,
            prefix_cache_backend="radix",
        ),
        manager,
    )
    request = Request(
        request_id="req-0",
        model_id="model",
        prompt_token_ids=[1, 2, 3, 4, 5],
        max_new_tokens=1,
    )

    scheduler.add_request(request)
    scheduled = scheduler.schedule()

    assert scheduled.is_empty
    assert request.cached_block_ids == []
    assert request.num_computed_tokens == 0
    assert manager.blocks[cached_pages[0]].ref_cnt == 1
    assert manager.radix_cache.protected_pages() == 0
