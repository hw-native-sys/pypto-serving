# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression coverage for V4.1 cache ownership across batches."""

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.model.deepseek_v41.task_args import RequestLedger
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


def test_concurrent_prefix_hits_reuse_retired_window_page():
    cache = KvCacheManager(block_size=128, enable_prefix_cache=True)
    cache.init_groups((
        KVCacheGroupSpec(
            "window", (0,), KVCacheSpec(128, 16), 3, num_blocks=6,
            max_requests_per_partition=2, sliding_window=128, prefix_cacheable=False,
        ),
        KVCacheGroupSpec(
            "c2a", (1,), KVCacheSpec(128, 16, 2), 7, num_blocks=14, max_requests_per_partition=2,
        ),
        KVCacheGroupSpec(
            "c1a", (2,), KVCacheSpec(128, 16), 7, num_blocks=14, max_requests_per_partition=2,
        ),
    ), max_batch_size=2)
    prompt = list(range(766))
    cache.ensure_group_blocks("warm", 512, partition=0)
    cache.cache_group_blocks("warm", cache.compute_group_block_hashes(prompt), 512, {})
    cache.release_all_group_requests("warm")

    scheduler = Scheduler(SchedulerConfig(
        max_num_scheduled_tokens=256, max_prefill_tokens_per_request=256,
        max_seq_len=896, enable_prefix_cache=True, prefix_cache_replay_before_hit=True,
        speculative_prefix_cache_replay_tokens=128, prefill_chunk_alignment=128,
    ), cache)
    ledger = RequestLedger(max_requests=2, max_seq_len=896, partitions=1, sliding_groups=("window",))
    for name in ("a", "b"):
        scheduler.add_request(Request(name, prompt, max_new_tokens=1))

    def begin(output):
        items = output.scheduled_requests
        return ledger.begin_prefill([
            (item.request.request_id, item.cache_partition, item.num_computed_tokens,
             prompt[item.num_computed_tokens:item.num_computed_tokens + item.num_new_tokens],
             len(prompt), item.block_ids_by_group)
            for item in items
        ], prefix_cache_hits={item.request.request_id: item.request.prefix_cache_hit_tokens for item in items})

    first = scheduler.schedule()
    ledger.commit(begin(first))
    scheduler.update_from_output(first, {})
    previous = set(ledger.owners["a"].pages["window"])

    second = scheduler.schedule()
    a, b = second.scheduled_requests
    assert (a.request.request_id, b.request.request_id) == ("a", "b")
    assert a.request.prefix_cache_hit_tokens == b.request.prefix_cache_hit_tokens == 512
    a_pages, b_pages = a.block_ids_by_group, b.block_ids_by_group
    assert set(b_pages["window"]) & (previous - set(a_pages["window"]))
    assert not set(a_pages["window"]) & set(b_pages["window"])
    assert a_pages["c2a"][:4] == b_pages["c2a"][:4]

    ledger.commit(begin(second))
    assert ledger.owners["a"].pages["window"] == tuple(a_pages["window"])
    assert ledger.owners["b"].pages["window"] == tuple(b_pages["window"])


@pytest.mark.parametrize("schedule_owner", [False, True])
def test_live_window_page_cannot_be_reused(schedule_owner):
    ledger = RequestLedger(max_requests=2, max_seq_len=256, partitions=1, sliding_groups=("window",))
    first = ledger.begin_prefill([("a", 0, 0, [1], 3, {"window": [5]})])
    ledger.commit(first)

    requests = [("b", 0, 0, [1], 3, {"window": [5]})]
    if schedule_owner:
        requests.insert(0, ("a", 0, 1, [2], 3, {"window": [5]}))

    with pytest.raises(ValueError, match="cache page belongs to another live request"):
        ledger.begin_prefill(requests)

    assert ledger.pending is None
    assert list(ledger.owners) == ["a"]
