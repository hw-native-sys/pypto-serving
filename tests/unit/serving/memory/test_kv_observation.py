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
    with pytest.raises(NotImplementedError, match="rank-partition"):
        manager.observe_prefix([1, 2, 3])
