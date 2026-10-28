# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.serving.memory.kv_cache import (
    KVCacheCapacityObservation,
    PrefixCacheObservation,
    KvCacheManager,
)
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


def test_read_only_observation_tracks_shared_and_reclaimable_prefix() -> None:
    manager = KvCacheManager(num_blocks=4, block_size=2)
    prompt = [1, 2, 3, 4, 5]
    blocks = manager.allocate_blocks(2)
    assert blocks is not None
    for block, block_hash in zip(blocks, manager.compute_block_hashes(prompt)):
        manager.cache_block(block, block_hash)
    manager.release(blocks[1])

    free_before = manager.num_free_blocks
    refs_before = [block.ref_cnt for block in manager.blocks]
    hashes_before = dict(manager.hash_to_block)
    queue_before = (manager.free_queue.head, manager.free_queue.tail)

    assert manager.observe_capacity() == KVCacheCapacityObservation(8, 2, 6)
    assert manager.observe_prefix(prompt) == PrefixCacheObservation(4, 2, 2)
    assert manager.observe_prefix(prompt, max_cache_hit_tokens=2) == PrefixCacheObservation(2, 2, 0)
    assert manager.num_free_blocks == free_before
    assert [block.ref_cnt for block in manager.blocks] == refs_before
    assert manager.hash_to_block == hashes_before
    assert (manager.free_queue.head, manager.free_queue.tail) == queue_before

    acquired = manager.get_computed_blocks(prompt)
    assert len(acquired) * manager.block_size == manager.observe_prefix(prompt).cached_tokens
    manager.release_cached_blocks(acquired)

    evicted = manager.allocate_blocks(3)
    assert evicted is not None
    assert manager.observe_prefix(prompt) == PrefixCacheObservation(2, 2, 0)


def test_uncached_free_head_stops_before_cached_prefix_blocks() -> None:
    manager = KvCacheManager(num_blocks=4, block_size=2)
    assert manager.observe_uncached_free_head_tokens() == 8
    blocks = manager.allocate_blocks(2)
    assert blocks is not None
    manager.cache_block(blocks[0], manager.compute_block_hashes([1, 2])[0])
    manager.release(blocks[0])
    assert manager.observe_uncached_free_head_tokens() == 4
    assert manager.num_free_blocks == 3
    assert manager.allocate_blocks(2) is not None
    assert manager.observe_uncached_free_head_tokens() == 0
    assert manager.num_free_blocks == 1


def test_read_only_observation_matches_native_scheduler_prefix_hit() -> None:
    manager = KvCacheManager(num_blocks=8, block_size=2)
    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=True), manager)
    cold = Request(request_id="cold", prompt_token_ids=[1, 2, 3, 4], max_new_tokens=1)
    scheduler.add_request(cold)
    scheduler.update_from_output(scheduler.schedule(), {"cold": 9})

    observed = manager.observe_prefix([1, 2, 3, 4])
    warm = Request(request_id="warm", prompt_token_ids=[1, 2, 3, 4], max_new_tokens=1)
    scheduler.add_request(warm)
    scheduled = scheduler.schedule()
    assert observed.cached_tokens == scheduled.prefix_cache_hits == 2


def test_observation_respects_disabled_cache_and_uninitialized_capacity() -> None:
    manager = KvCacheManager(enable_prefix_cache=False)
    assert manager.observe_capacity() == KVCacheCapacityObservation(0, 0, 0)
    assert manager.observe_prefix([1, 2, 3]) == PrefixCacheObservation(0, 0, 0)


def test_grouped_cache_requires_partition_aware_observation() -> None:
    manager = KvCacheManager(block_size=2)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                name="primary",
                layer_indices=(0,),
                spec=KVCacheSpec(block_size=2, page_size_bytes=1),
                max_blocks_per_seq=4,
                num_blocks=4,
            ),
        ),
        max_batch_size=1,
    )
    with pytest.raises(NotImplementedError, match="per-partition"):
        manager.observe_capacity()
    with pytest.raises(NotImplementedError, match="per-partition"):
        manager.observe_uncached_free_head_tokens()
    with pytest.raises(NotImplementedError, match="rank-partition"):
        manager.observe_prefix([1, 2, 3])


def test_grouped_capacity_uses_each_partitions_bottleneck_group() -> None:
    manager = KvCacheManager(block_size=2)
    manager.init_groups((
        KVCacheGroupSpec(
            name="eagle", layer_indices=(0,),
            spec=KVCacheSpec(block_size=2, page_size_bytes=1),
            max_blocks_per_seq=4, num_blocks=8, num_partitions=2, is_eagle_group=True,
        ),
        KVCacheGroupSpec(
            name="state", layer_indices=(1,),
            spec=KVCacheSpec(block_size=4, page_size_bytes=1),
            max_blocks_per_seq=2, num_blocks=4, num_partitions=2,
        ),
    ), max_batch_size=2)

    assert manager.observe_group_capacity(8) == KVCacheCapacityObservation(32, 0, 32)
    assert manager.group_capacity_cost_tokens(1, 8) == 4
    assert manager.group_capacity_cost_tokens(8, 8) == 8
    assert manager.observe_group_uncached_free_head_tokens(8) == 32

    manager.ensure_group_blocks("first", 2, partition=0)
    assert manager.observe_group_capacity(8) == KVCacheCapacityObservation(32, 4, 28)
    assert manager.observe_group_uncached_free_head_tokens(8) == 28
    manager.ensure_group_blocks("second", 8, partition=1)
    assert manager.observe_group_capacity(8) == KVCacheCapacityObservation(32, 12, 20)


def test_grouped_prefix_observation_matches_acquisition_without_pinning() -> None:
    manager = KvCacheManager(block_size=2, enable_prefix_cache=True)
    manager.init_groups((KVCacheGroupSpec(
        name="eagle", layer_indices=(0,),
        spec=KVCacheSpec(block_size=2, page_size_bytes=1),
        max_blocks_per_seq=4, num_blocks=4, num_partitions=2, is_eagle_group=True,
    ),), max_batch_size=1)
    prompt = [1, 2, 3, 4, 5]
    hashes = manager.compute_group_block_hashes(prompt)
    manager.ensure_group_blocks("warm", len(prompt), partition=1)
    manager.cache_group_blocks("warm", hashes, len(prompt), {})
    manager.release_all_group_requests("warm")
    pool = manager._group_pools["eagle"]
    before = (tuple(block.ref_cnt for block in pool.blocks),
              tuple(queue.count for queue in pool.free_queues),
              dict(pool.request_partitions))

    assert manager.observe_group_prefix(hashes, max_cache_hit_tokens=4) == PrefixCacheObservation(4, 0, 0)
    assert before == (tuple(block.ref_cnt for block in pool.blocks),
                      tuple(queue.count for queue in pool.free_queues),
                      dict(pool.request_partitions))
    _, hit, partition = manager.acquire_group_prefix_blocks(
        "next", hashes, max_cache_hit_tokens=4,
    )
    assert (hit, partition) == (4, 1)


def test_grouped_scheduler_observes_prefix_before_acquiring_references() -> None:
    manager = KvCacheManager(block_size=2, enable_prefix_cache=True)
    manager.init_groups((KVCacheGroupSpec(
        name="eagle", layer_indices=(0,),
        spec=KVCacheSpec(block_size=2, page_size_bytes=1),
        max_blocks_per_seq=4, num_blocks=4, is_eagle_group=True,
    ),), max_batch_size=1)
    prompt = [1, 2, 3, 4, 5]
    hashes = manager.compute_group_block_hashes(prompt)
    manager.ensure_group_blocks("warm", len(prompt), partition=0)
    manager.cache_group_blocks("warm", hashes, len(prompt), {})
    manager.release_all_group_requests("warm")

    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=True, num_speculative_tokens=1), manager)
    request = Request("next", prompt, max_new_tokens=1, observe_prefix_cache=True)
    scheduler.add_request(request)
    scheduled = scheduler.schedule().scheduled_requests[0]
    assert request.prefix_cache_observation == PrefixCacheObservation(4, 0, 0)
    assert scheduled.num_computed_tokens == 4
