# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real DSpark page/window geometry; metadata-only, no SDK/device allocation."""

import pytest
from dataclasses import replace

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


TARGET = ("ori", "cmp_c128", "cmp_c4", "idx")
GEOMETRY = (
    ("ori", 32, 1, 9, 128), ("cmp_c128", 128, 128, 5120, None),
    ("cmp_c4", 128, 4, 5120, None), ("idx", 128, 4, 5120, None),
    ("hca_state", 8, 1, 33, 128), ("csa_state", 2, 1, 69, 8),
    ("csa_inner_state", 2, 1, 69, 8),
)


def manager(enabled=True, slots=3, partitions=4):
    # Exact max_seq_len=655360, prefill=128, four partitions and 3 capacity
    # slots. Tensor widths are irrelevant to the Host page/ref-count contract.
    groups = tuple(KVCacheGroupSpec(name, (0,), KVCacheSpec(tokens, tokens // ratio, ratio),
        blocks, num_partitions=partitions, sliding_window=window)
        for name, tokens, ratio, blocks, window in GEOMETRY)
    cache = KvCacheManager(enable_prefix_cache=enabled)
    cache.init_groups(groups, max_batch_size=slots, primary_num_blocks=9 * slots)
    return cache


def prompt(group, length):
    return [1000000 + group] + list(range(1, length - 81)) + list(range(2000000, 2000081))


def query(cache, tokens, groups=None):
    _, hit, _ = cache.acquire_group_prefix_blocks("query", cache.compute_group_block_hashes(
        tokens, group_names=groups), max_cache_hit_tokens=len(tokens) - 1, shareable_group_names=groups)
    if hit:
        cache.release_all_group_requests("query")
    return hit


def commit(cache, request):
    cache.authorize_group_cache_write(request)
    cache.commit_group_cache(request)


def _prefill_scheduler(cache, *, speculative=0, asynchronous=False, replay=128):
    return Scheduler(SchedulerConfig(max_num_running_reqs=8, max_num_scheduled_tokens=8192,
        long_prefill_token_threshold=128, max_prefill_tokens_per_request=128, max_seq_len=655360,
        num_speculative_tokens=speculative, speculative_prefix_cache_replay_tokens=replay,
        enable_prefix_cache=cache.enable_prefix_cache, async_scheduling=asynchronous,
        stop_after_prefill=not asynchronous), cache)


def _finish_prefill(scheduler, request, *, expected_hit=None):
    if scheduler.config.async_scheduling:
        request.max_new_tokens = 1
    scheduler.add_request(request)
    boundary = max(0, len(request.prompt_token_ids)
        - max(1, scheduler.config.speculative_prefix_cache_replay_tokens)) // 128 * 128
    pending = []
    started = False
    for _ in range(4 * len(request.prompt_token_ids) // 128 + 12):
        output = scheduler.schedule()
        if output.scheduled_requests:
            step, = output.scheduled_requests
            if not started and expected_hit is not None:
                assert step.num_computed_tokens == expected_hit
            started = True
            end = step.num_computed_tokens + step.num_new_tokens
            tokens = {request.request_id: [11]} if end == request.num_prompt_tokens else {}
            if scheduler.config.async_scheduling:
                scheduler.advance_after_schedule(output)
            pending.append((output, tokens, end))
        if pending and (not scheduler.config.async_scheduling or len(pending) == 2
                        or not output.scheduled_requests):
            confirmed, tokens, end = pending.pop(0)
            scheduler.update_from_output(confirmed, tokens)
            if boundary and end >= boundary and scheduler.config.enable_prefix_cache:
                checkpoints = scheduler.kv_cache_manager._group_checkpoints
                identity = tuple(sorted((group.name,
                    request.group_block_hashes[group.name][boundary // group.spec.token_capacity - 1])
                    for group in scheduler.kv_cache_manager.group_specs))
                key = confirmed.scheduled_requests[0].cache_partition, boundary, identity
                assert key in checkpoints.checkpoints
                if not request.status.is_finished:
                    assert checkpoints.owners[request.request_id] == key
        if request.prefill_complete or request.status.is_finished:
            assert not pending
            if request.prefill_complete:
                scheduler.finish_prefilled_request(request.request_id)
            return boundary
    pytest.fail("bounded Prefill did not finish")


@pytest.mark.parametrize("partitions", [4, 16])
@pytest.mark.parametrize("length", [8192, 8193, 8205, 8319, 8320])
@pytest.mark.parametrize("speculative,asynchronous", [(0, False), (7, False), (7, True)],
    ids=["prefill-target-only", "replica-k7-sync", "replica-k7-async"])
def test_request_checkpoint_matches_full_prompt_replay_boundary(partitions, length, speculative, asynchronous):
    cache = manager(partitions=partitions)
    scheduler = _prefill_scheduler(cache, speculative=speculative, asynchronous=asynchronous)
    for group in range(8):
        # Keep two documents per local partition, including in the larger topology.
        cache._next_group_partition = group % 4
        boundary = _finish_prefill(scheduler, Request(str(group), prompt(group, length), max_new_tokens=512))
    for group in range(8):
        request = Request(f"hot-{group}", prompt(group, length), max_new_tokens=512)
        scheduler.add_request(request)
        output = scheduler.schedule()
        assert output.prefix_cache_hits == boundary
        step, = output.scheduled_requests
        assert step.num_computed_tokens == boundary
        scheduler.abort_request(request.request_id)
    assert not cache._group_checkpoints.owners
    assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)


@pytest.mark.parametrize("name", ["ori", "hca_state", "csa_state", "csa_inner_state"])
def test_request_checkpoint_loses_joint_validity_when_a_member_is_evicted(name):
    cache = manager(partitions=1)
    scheduler = _prefill_scheduler(cache, speculative=7, asynchronous=True)
    tokens = prompt(0, 513)
    boundary = _finish_prefill(scheduler, Request("warm", tokens, max_new_tokens=512))
    checkpoints = cache._group_checkpoints
    key, members = next(iter(checkpoints.checkpoints.items()))
    block = next(block for group, block in members if group == name)
    pool = cache._group_pools[name]
    cache._invalidate_group_block(pool, key[0], pool.blocks[block])
    assert key not in checkpoints.checkpoints
    _, hit, _ = cache.acquire_group_prefix_blocks("query", cache.compute_group_block_hashes(tokens),
        max_cache_hit_tokens=boundary)
    if hit:
        cache.release_all_group_requests("query")
    assert hit < boundary
    assert cache._group_pools["cmp_c128"].hash_to_block


def paged_manager(partitions=4, enabled=True):
    from pypto_serving.model.deepseek_dspark.cache_capacity import cache_pool_bytes, plan_cache_pages
    cache = manager(partitions=partitions)
    # Same real ring/table geometry, lightweight per-page widths. Index data
    # and scale bytes retain their actual 128:4 allocation ratio.
    groups = tuple(replace(g, spec=replace(g.spec, page_size_bytes=132 if g.name == "idx" else 128))
                   for g in cache.group_specs)
    budget = cache_pool_bytes(groups, {g.name: g.max_blocks_per_seq for g in groups}, filler_pages=64) * 3 // 2
    counts = plan_cache_pages(groups, budget, active_requests=max(1, 8 // partitions), step_tokens=8,
                             filler_pages=64)
    cache = KvCacheManager(enable_prefix_cache=enabled)
    cache.init_groups(tuple(replace(g, num_blocks=counts[g.name]) for g in groups), max_batch_size=8)
    return cache


@pytest.mark.parametrize("partitions", [4, 16])
@pytest.mark.parametrize("enabled", [False, True], ids=["prefix-off", "prefix-on"])
def test_paged_decode_keeps_depth_two_and_eight_group_prefixes(partitions, enabled):
    cache = paged_manager(partitions, enabled)
    scheduler = Scheduler(SchedulerConfig(max_num_running_reqs=8, max_num_scheduled_tokens=8192,
        max_seq_len=655360, num_speculative_tokens=7, speculative_prefix_cache_replay_tokens=128,
        async_scheduling=True, enable_prefix_cache=enabled), cache)
    depth_two = 0
    for turn in range(3):
        length = 8205 + turn * 512
        requests = []
        for group in range(8):
            # Accumulated history preserves every previous input token. The
            # adversarial prompt() helper instead rewrites its last 81 tokens.
            rid = f"paged-{group}-{turn}"
            tokens = [1000000 + group] + list(range(1, length))
            lease = cache.reserve_group_cache(rid, rid, length + 131072, prompt_token_count=length,
                prefix_block_hashes=cache.compute_group_block_hashes(tokens, group_names=TARGET),
                shareable_group_names=TARGET)
            expected = (length - 512 - 1) // 128 * 128 if turn and enabled else 0
            assert lease.prefix_hit_tokens == expected
            commit(cache, rid)
            request, _ = scheduler.admit_prefilled(reservation_id=rid, request_id=rid,
                prompt_token_ids=tokens, first_token=11, max_new_tokens=512)
            requests.append(request)
        pending = []
        for _ in range(256):
            output = scheduler.schedule(allow_reject_stalled=not pending)
            assert not output.rejected_requests and not output.preempted_requests
            if output.scheduled_requests:
                scheduler.advance_after_schedule(output)
                pending.append(output)
            if len(pending) == 2:
                depth_two += 1
            if pending and (len(pending) == 2 or not output.scheduled_requests):
                confirmed = pending.pop(0)
                scheduler.update_from_output(confirmed,
                    {s.request.request_id: [11] * 8 for s in confirmed.scheduled_requests})
            if all(request.status.is_finished for request in requests):
                for confirmed in pending:
                    scheduler.update_from_output(confirmed, {})
                break
        else:
            pytest.fail("paged Decode did not drain")
        assert all(len(request.output_token_ids) == 512 for request in requests)
        for request in requests:
            cache.retire_group_cache_reservation(request.request_id)
        assert all(block.ref_cnt == 0 for pool in cache._group_pools.values() for block in pool.blocks)
    assert depth_two > 100


def test_external_decode_capacity_waits_for_snapshot_then_fails_without_local_prefill():
    cache = KvCacheManager(enable_prefix_cache=False)
    cache.init_groups((KVCacheGroupSpec("history", (0,), KVCacheSpec(1, 4), 32, num_blocks=32),),
                      max_batch_size=8)
    lease = cache.reserve_group_cache("external", "external", 32, prompt_token_count=2)
    commit(cache, "external")
    scheduler = Scheduler(SchedulerConfig(max_seq_len=32, async_scheduling=True, enable_prefix_cache=False), cache)
    request, _ = scheduler.admit_prefilled(reservation_id="external", request_id="external",
        prompt_token_ids=[1, 2], first_token=11, max_new_tokens=30)
    # A detached in-flight table occupies the last pages. No rejection until
    # its FIFO completion has had the opportunity to return those references.
    held = cache.ensure_group_blocks("inflight", 30)
    cache.retain_group_block_snapshot(held, 0)
    cache.release_all_group_requests("inflight")
    waiting = scheduler.schedule(allow_reject_stalled=False)
    assert not waiting.rejected_requests and request in scheduler.running and not scheduler.waiting
    cache.release_group_block_snapshot(held, 0)
    resumed = scheduler.schedule()
    assert [s.request.request_id for s in resumed.scheduled_requests] == ["external"]
    scheduler.advance_after_schedule(resumed)
    scheduler.update_from_output(resumed, {"external": [11]})
    # Fill the last page with another admitted owner. The external request is
    # never a preemption/recompute candidate for a competing ordinary request.
    cache.ensure_group_blocks("holder", 29)
    failed = scheduler.schedule()
    assert "external Prefill" in failed.rejected_requests["external"]
    assert not failed.preempted_requests and not scheduler.waiting
    assert cache.group_cache_reservation("external").state.name == "RELEASED"
    cache.release_all_group_requests("holder")
    assert all(block.ref_cnt == 0 for pool in cache._group_pools.values() for block in pool.blocks)


@pytest.mark.parametrize("uncertain", [False, True], ids=["cancel", "unknown"])
def test_paged_receive_failure_keeps_healthy_shared_prefix(uncertain):
    cache = paged_manager(16)
    tokens = list(range(8205))
    def receive(request, prompt):
        return cache.reserve_group_cache(request, request, len(prompt) + 131072, prompt_token_count=len(prompt),
            prefix_block_hashes=cache.compute_group_block_hashes(prompt, group_names=TARGET),
            shareable_group_names=TARGET)
    receive("warm", tokens)
    commit(cache, "warm")
    cache.release_group_cache("warm")
    lease = receive("failed", tokens + [91] * 512)
    assert lease.prefix_hit_tokens == 8192
    for name in TARGET:
        assert set(lease.writable_block_ids_by_group[name]).isdisjoint(lease.shared_block_ids_by_group[name])
    if uncertain:
        cache.authorize_group_cache_write("failed")
        quarantined = cache.quarantine_group_cache("failed")
        assert quarantined.quarantined_block_ids_by_group == lease.writable_block_ids_by_group
        with pytest.raises(RuntimeError, match="native writer"):
            cache.release_group_cache("failed")
    assert query(cache, tokens, TARGET) == 8192
    cache.release_group_cache("failed", confirmed_stopped=uncertain)
    assert query(cache, tokens, TARGET) == 8192
    assert all(block.ref_cnt == 0 for pool in cache._group_pools.values() for block in pool.blocks)


def test_partial_ring_tail_does_not_publish_an_overwritten_checkpoint():
    group = KVCacheGroupSpec("state", (0,), KVCacheSpec(128, 16, 128), 2, num_blocks=8,
                             sliding_window=128)
    cache = KvCacheManager(enable_prefix_cache=True)
    cache.init_groups((group,), max_batch_size=1)
    tokens = list(range(257))
    tables = cache.ensure_group_blocks("producer", 257)
    hashes = cache.compute_group_block_hashes(tokens)
    cache.retain_group_block_snapshot(tables, 0)
    cache.cache_group_blocks_from_snapshot(hashes, 257, {}, tables, 0, request_id="producer")
    cache.release_group_block_snapshot(tables, 0)
    # The ring contains the 256 endpoint and a mutable 257 tail. The 128 page
    # has already been displaced and must not be registered under its old hash.
    assert query(cache, tokens[:129]) == 0
    assert query(cache, tokens) == 256
