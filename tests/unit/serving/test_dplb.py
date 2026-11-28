# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouped DP admission balances logical KV length and sequence count."""

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.serving.memory.kv_cache import KvCacheManager, KVCacheCapacityError
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


def cache(enabled=True):
    manager = KvCacheManager(enable_prefix_cache=True, enable_dplb=enabled)
    manager.init_groups((KVCacheGroupSpec(
        "kv", (0,), KVCacheSpec(16, 16, 1), 256,
        num_partitions=2,
    ),), max_batch_size=8)
    return manager


def allocate(manager, name, length, partition=None):
    manager.ensure_group_blocks(name, length, partition=partition)
    return manager.group_request_partition(name)


@pytest.mark.parametrize("enabled,expected", [(True, 1), (False, 0)])
def test_long_request_uses_kv_load_instead_of_rotation(enabled, expected):
    manager = cache(enabled)
    allocate(manager, "long", 2048, 0)
    allocate(manager, "short", 32, 1)
    assert allocate(manager, "incoming", 1536) == expected


def test_short_request_fills_concurrency_gap_then_larger_kv_load():
    manager = cache()
    allocate(manager, "long", 2048, 1)
    allocate(manager, "short", 32, 0)
    assert allocate(manager, "incoming", 16) == 1
    assert allocate(manager, "next", 16) == 0


def test_chunked_prefill_counts_complete_prompt_and_keeps_owner():
    manager = cache()
    scheduler = Scheduler(SchedulerConfig(
        max_num_running_reqs=8, max_num_scheduled_tokens=32,
        max_prefill_tokens_per_request=16, long_prefill_token_threshold=16,
        max_seq_len=4096, enable_prefix_cache=False,
    ), manager)
    scheduler.add_request(Request("long", list(range(2048)), max_new_tokens=10))
    scheduler.add_request(Request("short", list(range(32)), max_new_tokens=10))
    scheduler.schedule()
    assert manager._group_request_tokens == {"long": 2048, "short": 32}
    assert allocate(manager, "incoming", 1536) == 1
    allocate(manager, "long", 2064)
    assert manager.group_request_partition("long") == 0
    manager.release_all_group_requests("long")
    assert "long" not in manager._group_request_tokens
    assert allocate(manager, "replacement", 1024) == 0


def test_allocation_failure_does_not_leak_load():
    manager = cache()
    for partition in (0, 1):
        for index in range(8):
            allocate(manager, f"full-{partition}-{index}", 4096, partition)
    with pytest.raises(KVCacheCapacityError):
        allocate(manager, "blocked", 1024)
    assert "blocked" not in manager._group_request_tokens
    manager.release_all_group_requests("full-0-0")
    assert allocate(manager, "retry", 1024) == 0


def test_equal_prefix_hits_use_dplb_and_longer_hit_still_wins():
    manager = cache()
    tokens = list(range(128))
    hashes = manager.compute_group_block_hashes(tokens)
    for partition in (0, 1):
        name = f"prefix-{partition}"
        allocate(manager, name, 128, partition)
        manager.cache_group_blocks(name, hashes, 128, {})
        manager.release_all_group_requests(name)
    allocate(manager, "long", 2048, 0)
    allocate(manager, "short", 32, 1)
    _, hit, partition = manager.acquire_group_prefix_blocks(
        "hit", hashes, max_cache_hit_tokens=128, load_tokens=1536,
    )
    assert (hit, partition) == (128, 1)
    manager.release_all_group_requests("hit")
    longer = list(range(256))
    allocate(manager, "longer-prefix", 256, 0)
    manager.cache_group_blocks("longer-prefix", manager.compute_group_block_hashes(longer), 256, {})
    manager.release_all_group_requests("longer-prefix")
    _, hit, partition = manager.acquire_group_prefix_blocks(
        "longer-hit", manager.compute_group_block_hashes(longer),
        max_cache_hit_tokens=256, load_tokens=1536,
    )
    assert (hit, partition) == (256, 0)


def test_decode_growth_updates_load_and_cancellation_removes_it():
    manager = cache()
    allocate(manager, "growing", 32, 0)
    allocate(manager, "steady", 512, 1)
    allocate(manager, "growing", 2048)
    assert allocate(manager, "incoming", 1536) == 1
    scheduler = Scheduler(SchedulerConfig(), manager)
    request = Request("growing", list(range(32)), max_new_tokens=10)
    scheduler.add_request(request)
    request.allocated_group_block_ids = manager.ensure_group_blocks("growing", 2048)
    request.cache_partition = manager.group_request_partition("growing")
    scheduler.abort_request("growing")
    assert "growing" not in manager._group_request_tokens
    assert allocate(manager, "replacement", 1536) == 0


def test_equal_prefix_loads_rotate_after_release():
    manager = cache()
    hashes = manager.compute_group_block_hashes(list(range(128)))
    for partition in (0, 1):
        allocate(manager, "seed", 128, partition)
        manager.cache_group_blocks("seed", hashes, 128, {})
        manager.release_all_group_requests("seed")
    selected = []
    for _ in range(4):
        _, hit, partition = manager.acquire_group_prefix_blocks(
            "hit", hashes, max_cache_hit_tokens=128,
        )
        assert hit == 128
        selected.append(partition)
        manager.release_all_group_requests("hit")
    assert selected == [0, 1, 0, 1]
