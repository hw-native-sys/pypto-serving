# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from dataclasses import replace
from unittest.mock import Mock

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.pd.scheduler import PrefillScheduler
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.memory.tiering.cache import TieredCache
from pypto_serving.serving.sched.scheduler import Request, RequestStatus, SchedulerConfig
from tests.unit.serving.memory.test_cache_offload import Backend, evict, fixture, warm
from tests.unit.serving.test_grouped_prefix_cache import _prefill_scheduler, manager, prompt


def scheduler(cache, *, window=5, asynchronous=False):
    config = replace(_prefill_scheduler(cache, asynchronous=asynchronous).config,
                     max_num_scheduled_tokens=128 * cache.group_partition_count)
    return PrefillScheduler(config=config, kv_cache_manager=cache, waiting_window=window)


def add(sched, name, tokens):
    request = Request(name, tokens, max_new_tokens=16)
    sched.add_request(request)
    return request


def running(sched, length, partition=0):
    request = add(sched, "running", list(range(length)))
    sched.waiting.remove(request)
    request.status = RequestStatus.RUNNING
    request.cache_partition = partition
    sched.running.append(request)
    return request


def discard(sched, output):
    for item in output.scheduled_requests:
        sched.discard_scheduled_request(item)
        sched.abort_request(item.request.request_id)


@pytest.mark.parametrize("window,limit", [(1, 0), (5, 4)])
def test_window_charges_admissions_across_steps_and_resets_on_head_exit(monkeypatch, window, limit):
    cache = manager(enabled=False, partitions=4)
    sched = scheduler(cache, window=window)
    head = add(sched, "head", prompt(0, 96))
    for index in range(6):
        add(sched, str(index), prompt(index + 1, 96))
    choose = sched._choose_first_chunk
    missing = {head.request_id}
    monkeypatch.setattr(sched, "_choose_first_chunk",
                        lambda r, budget, **kw: (None, 0) if r.request_id in missing else choose(r, budget, **kw))

    first = sched.schedule()
    assert len(first.scheduled_requests) == min(limit, 4)
    assert sched.waiting[0] is head
    discard(sched, first)
    assert sched.schedule().is_empty
    assert sched.waiting[0] is head

    sched.abort_request(head.request_id)
    missing.add(sched.waiting[0].request_id)
    output = sched.schedule()
    assert len(output.scheduled_requests) == min(limit, 5 - limit)
    discard(sched, output)


def test_tail_slack_sets_the_new_requests_actual_first_chunk():
    sched = scheduler(manager(enabled=False, partitions=1))
    old = running(sched, 32)
    new = add(sched, "new", prompt(2, 256))
    output = sched.schedule()
    assert [(item.request, item.num_new_tokens) for item in output.scheduled_requests] == [(old, 32), (new, 96)]
    discard(sched, output)


@pytest.mark.parametrize("long", [False, True])
def test_hot_group_waits_two_rounds_or_restores_on_another_group(monkeypatch, long):
    tokens = prompt(0, 513)
    cache, tier, backend = fixture(4)
    hit = warm(cache, tokens)
    sched = scheduler(cache, asynchronous=True)
    old = running(sched, 1024 if long else 320)
    request = add(sched, "next", tokens)
    name = "cmp_c128"
    value = request.group_block_hashes[name][0]
    pool = cache._group_pools[name]
    # The mixed destination needs one fewer LOAD than the pure-DDR alternatives.
    cache._cache_group_block(pool, 2, pool.block_from_local_id(2, 1), value)
    scan = Mock(wraps=tier.matching_checkpoint_boundaries)
    monkeypatch.setattr(tier, "matching_checkpoint_boundaries", scan)
    backend.hold = True
    output = sched.schedule()
    sched.advance_after_schedule(output)
    assert [item.request for item in output.scheduled_requests] == [old]
    assert scan.call_count == int(long)
    assert (request.request_id in tier.pending) == long
    if long:
        assert request.cache_partition == 2
        assert {(2, *key) for (end, _), keys in tier.tier.units.items() if end == hit
                for key in keys if key != (name, value)} == set(tier.loads)
        reserved = request.allocated_group_block_ids
        for future in backend.waiting:
            future.set_result(dict(bytes=1, stopped=True, native_completed=True))
        tier.progress()
        assert not sched._prepare_waiting_group(request, 0, head=True)
        assert request.allocated_group_block_ids is reserved
        assert tier.pending[request.request_id].remaining == 0
        step, = [item for item in sched.schedule().scheduled_requests if item.request is request]
        assert step.num_computed_tokens == hit and step.cache_partition == 2
        assert request.allocated_group_block_ids is reserved
        sched.discard_scheduled_request(step)
    else:
        assert request.cache_partition is None
        assert not tier.tier.counters["loads"]
        discard(sched, output)
        step, = sched.schedule().scheduled_requests
        assert step.request is request and step.cache_partition == 0
        assert step.num_computed_tokens == hit
        sched.discard_scheduled_request(step)
    sched.abort_request(request.request_id)
    discard(sched, output)


def test_empty_ddr_checkpoint_lookup_is_shared_across_groups(monkeypatch):
    cache, tier, _ = fixture(4)
    warm(cache, prompt(0, 513))
    sched = scheduler(cache)
    request = add(sched, "cold", prompt(1, 513))
    scan = Mock(wraps=tier.matching_checkpoint_boundaries)
    monkeypatch.setattr(tier, "matching_checkpoint_boundaries", scan)

    output = sched.schedule()
    step, = output.scheduled_requests
    assert step.request is request and step.num_computed_tokens == 0
    assert scan.call_count == 1
    assert not tier.loads
    discard(sched, output)


@pytest.mark.parametrize("first_chunk_fits", [False, True])
def test_load_needs_room_for_the_first_chunk_and_retains_native_pins(first_chunk_fits):
    cache = KvCacheManager()
    spec = KVCacheGroupSpec("history", (0,), KVCacheSpec(128, 1), 4)
    cache.init_groups((spec,), max_batch_size=1)
    backend = Backend()
    tier = cache.tiering = TieredCache(cache, backend, 1024)
    tokens = prompt(4, 513)
    config = SchedulerConfig(max_num_scheduled_tokens=128, long_prefill_token_threshold=128,
                             max_prefill_tokens_per_request=128, max_seq_len=1024)
    seed = PrefillScheduler(config=config, kv_cache_manager=cache)
    prefix = add(seed, "seed", tokens[:512])
    for index in range(4):
        output = seed.schedule()
        seed.update_from_output(output, {prefix.request_id: [11]} if index == 3 else {})
    seed.finish_prefilled_request(prefix.request_id)
    tier.progress()
    tier.progress()
    evict(cache)
    # Three prefix pages fit. The fourth is needed for the first uncached position.
    if not first_chunk_fits:
        cache.ensure_group_blocks("competitor", 128)
    sched = PrefillScheduler(config=config, kv_cache_manager=cache)
    request = add(sched, "restore", tokens[:385])
    backend.hold = True
    assert sched.schedule().is_empty
    assert (request.request_id in tier.pending) == first_chunk_fits
    if first_chunk_fits:
        assert cache.group_num_free_blocks("history", 0) == 0
        sched.abort_request(request.request_id)
        assert cache.group_num_free_blocks("history", 0) == 1
        again = add(sched, "again", tokens[:385])
        loads = len(backend.waiting)
        assert sched.schedule().is_empty
        assert again.request_id in tier.pending and len(backend.waiting) == loads
        for future in backend.waiting:
            future.set_result(dict(bytes=1, stopped=True, native_completed=True))
        tier.progress()
        restored = sched.schedule()
        assert restored.scheduled_requests[0].num_computed_tokens == 384
        discard(sched, restored)
        assert cache.group_num_free_blocks("history", 0) == 4
    else:
        assert not backend.waiting


def test_blocked_retry_skips_lookup_until_group_state_changes(monkeypatch):
    cache = manager(enabled=False, partitions=2)
    sched = scheduler(cache)
    request = add(sched, "waiting", prompt(2, 1024))
    choose = sched._choose_first_chunk
    checked = []
    def blocked(*args, **kwargs):
        checked.append(True)
        return None, 0
    monkeypatch.setattr(sched, "_choose_first_chunk", blocked)
    assert sched.schedule().is_empty
    assert sched.schedule().is_empty
    assert len(checked) == 1

    cache.ensure_group_blocks("other", 128, partition=0)
    cache.release_all_group_requests("other")
    monkeypatch.setattr(sched, "_choose_first_chunk", choose)
    assert sched.schedule().scheduled_requests[0].request is request
