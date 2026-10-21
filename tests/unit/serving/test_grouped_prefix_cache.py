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

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.serving.memory.kv_cache import KvCacheManager, KVCacheCapacityError
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


TARGET = ("ori", "cmp_c128", "cmp_c4", "idx")
GEOMETRY = (
    ("ori", 32, 1, 9, 128), ("cmp_c128", 128, 128, 5120, None),
    ("cmp_c4", 128, 4, 5120, None), ("idx", 128, 4, 5120, None),
    ("hca_state", 8, 1, 33, 128), ("csa_state", 2, 1, 69, 8),
    ("csa_inner_state", 2, 1, 69, 8),
)


def manager(enabled=True, slots=3):
    # Exact max_seq_len=655360, prefill=128, four partitions and 3 capacity
    # slots. Tensor widths are irrelevant to the Host page/ref-count contract.
    groups = tuple(KVCacheGroupSpec(name, (0,), KVCacheSpec(tokens, tokens // ratio, ratio),
        blocks, num_partitions=4, sliding_window=window)
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


def prefill(cache, request, tokens, partition, snapshot=False):
    hashes = cache.compute_group_block_hashes(tokens)
    published = {}
    for end in (*range(128, len(tokens), 128), len(tokens)):
        tables = cache.ensure_group_blocks(request, end, partition=partition)
        if snapshot:
            cache.retain_group_block_snapshot(tables, partition)
            published = cache.cache_group_blocks_from_snapshot(hashes, end, published, tables,
                partition, request_id=request)
            cache.release_group_block_snapshot(tables, partition)
        else:
            published = cache.cache_group_blocks(request, hashes, end, published)
    cache.release_all_group_requests(request)


def reserve(cache, request, tokens, partition=None):
    return cache.reserve_group_cache(request, request, len(tokens) + 512,
        prompt_token_count=len(tokens), partition=partition,
        prefix_block_hashes=cache.compute_group_block_hashes(tokens, group_names=TARGET),
        shareable_group_names=TARGET)


def commit(cache, request):
    cache.authorize_group_cache_write(request)
    cache.commit_group_cache(request)


@pytest.mark.parametrize("length", [512, 20480])
@pytest.mark.parametrize("snapshot", [False, True])
def test_prefill_retains_joint_cache_and_state_for_eight_groups(length, snapshot):
    cache = manager()
    # Two independent documents per partition; adversarial, non-interleaved
    # order used to evict the first document's short CSA states.
    for group in (4, 5, 6, 7, 0, 1, 2, 3):
        prefill(cache, str(group), prompt(group, length), group % 4, snapshot)
    for group in range(8):
        assert query(cache, prompt(group, length + 512)) == (length - 81) // 128 * 128
    assert not cache._group_checkpoints.owners
    assert len(cache._group_checkpoints.checkpoints) == 8


def test_cold_reservation_allocates_each_ring_destination_once(monkeypatch):
    cache = manager()
    reserve(cache, "previous", prompt(0, 20480), 0)
    commit(cache, "previous")
    cache.release_group_cache("previous")
    takes = []
    original = cache._take_group_block
    def take(pool, partition):
        takes.append(pool.spec.name)
        return original(pool, partition)
    monkeypatch.setattr(cache, "_take_group_block", take)
    reserve(cache, "cold", prompt(4, 20480), 0)
    assert takes.count("ori") == 9
    assert takes.count("hca_state") == 33
    assert takes.count("csa_state") == 69
    assert query(cache, prompt(0, 22528), TARGET) == 20352
    cache.release_group_cache("cold")  # No write was authorized.


@pytest.mark.parametrize("initial,growth,turns", [(512, 512, 20), (20480, 2048, 16)])
def test_decode_eight_groups_keep_prefix_after_512_outputs(initial, growth, turns):
    cache = manager()
    for turn in range(turns):
        length = initial + turn * growth
        for group in range(8):
            request = f"{group}-{turn}"
            lease = reserve(cache, request, prompt(group, length))
            expected = (initial + (turn - 1) * growth - 81) // 128 * 128 if turn else 0
            assert lease.prefix_hit_tokens == expected
            for name in TARGET:
                assert set(lease.shared_block_ids_by_group[name]).isdisjoint(lease.writable_block_ids_by_group[name])
            commit(cache, request)
            assert not cache.group_cache_reservation(request).shared_prefix_pinned
        for offset in range(8, 513, 8):
            for group in range(8):
                cache.ensure_group_blocks(f"{group}-{turn}", length + offset)
        for group in range(8):
            request = f"{group}-{turn}"
            cache.release_group_cache(request)
            cache.retire_group_cache_reservation(request)
        assert not cache._group_checkpoints.owners


def test_evicted_state_retires_the_joint_checkpoint_not_a_partial_hit():
    cache = manager()
    prefill(cache, "source", prompt(0, 512), 0)
    checkpoints = cache._group_checkpoints
    key, members = next(iter(checkpoints.checkpoints.items()))
    assert key[1] == 384
    state = next(block for name, block in members if name == "csa_state")
    pool = cache._group_pools["csa_state"]
    cache._invalidate_group_block(pool, 0, pool.blocks[state])
    assert key not in checkpoints.checkpoints
    assert all(key not in owners for owners in checkpoints.pages.values())
    assert query(cache, prompt(0, 1024)) != 384
    # Full-history KV can still be useful to a different valid checkpoint; it
    # must not be presented as a complete cache/state hit on its own.
    assert cache._group_pools["cmp_c128"].hash_to_block


def test_full_history_eviction_also_retires_the_cache_state_checkpoint():
    cache = manager()
    prefill(cache, "source", prompt(0, 512), 0)
    pool = cache._group_pools["cmp_c128"]
    block = next(iter(pool.hash_to_block.values()))
    cache._invalidate_group_block(pool, 0, block)
    cache.ensure_group_blocks("other", 128, partition=1)
    assert not cache._group_checkpoints.checkpoints
    assert not cache._group_checkpoints.pages
    assert query(cache, prompt(0, 1024)) == 0
    cache.release_all_group_requests("other")


@pytest.mark.parametrize("length", [127, 128, 129, 255, 256, 257, 511, 512, 513])
def test_replayed_cache_and_state_use_one_common_boundary(length):
    cache = manager()
    tokens = prompt(0, length)
    prefill(cache, "source", tokens, 0)
    assert query(cache, tokens) == (length - 1) // 128 * 128


def test_readonly_shared_prefix_survives_unknown_suffix_write_and_rollback():
    cache = manager()
    tokens = prompt(0, 20480)
    reserve(cache, "source", tokens, 0)
    commit(cache, "source")
    cache.release_group_cache("source")
    lease = reserve(cache, "receiver", prompt(0, 22528))
    assert lease.prefix_hit_tokens == 20352 and lease.shared_prefix_pinned
    assert all(set(lease.shared_block_ids_by_group[n]).isdisjoint(lease.writable_block_ids_by_group[n]) for n in TARGET)
    cache.authorize_group_cache_write("receiver")
    cache.quarantine_group_cache("receiver")
    with pytest.raises(RuntimeError, match="native writer"):
        cache.release_group_cache("receiver")
    assert query(cache, prompt(0, 22528), TARGET) == 20352
    cache.release_group_cache("receiver", confirmed_stopped=True)
    assert query(cache, prompt(0, 22528), TARGET) == 20352


def test_reservation_failure_releases_all_readonly_references(monkeypatch):
    cache = manager()
    reserve(cache, "source", prompt(0, 512), 0)
    commit(cache, "source")
    cache.release_group_cache("source")
    before = [[b.ref_cnt for b in p.blocks] for p in cache._group_pools.values()]
    def fail(*args):
        raise KVCacheCapacityError("forced capacity failure")
    monkeypatch.setattr(cache, "_reserve_group_decode_capacity", fail)
    with pytest.raises(KVCacheCapacityError):
        reserve(cache, "failed", prompt(0, 1024))
    assert [[b.ref_cnt for b in p.blocks] for p in cache._group_pools.values()] == before
    assert "failed" not in cache._group_request_partitions


@pytest.mark.parametrize("next_length", [2048, 4096], ids=["decode-spares", "writable-suffix"])
@pytest.mark.parametrize("partition", [None, 0], ids=["pooled-placement", "fixed-placement"])
def test_busy_prefix_partition_can_use_cold_capacity(next_length, partition):
    cache = manager()
    reserve(cache, "source", prompt(0, 2048), 0)
    commit(cache, "source")
    cache.release_group_cache("source")
    # Two in-flight destinations leave nine free ori pages. Matching the
    # immutable four-page tail also needs a writable ring / Decode spares;
    # that hit cannot fit, but a cold allocation can still be admitted.
    for group in (1, 2):
        reserve(cache, f"live-{group}", prompt(group, 2048), 0)
    before = {name: list(pool.request_blocks["live-1"]) for name, pool in cache._group_pools.items()}
    lease = reserve(cache, "receiver", prompt(0, next_length), partition)
    assert lease.prefix_hit_tokens == 0 and not lease.shared_prefix_pinned
    assert lease.partition == (0 if partition == 0 else 1)
    assert all(not ids for ids in lease.shared_block_ids_by_group.values())
    assert all(pool.request_blocks["live-1"] == before[name] for name, pool in cache._group_pools.items())
    for request in ("receiver", "live-1", "live-2"):
        cache.release_group_cache(request)
    assert all(block.ref_cnt == 0 for pool in cache._group_pools.values() for block in pool.blocks)


def test_disabled_prefix_uses_no_checkpoint_metadata():
    cache = manager(enabled=False)
    prefill(cache, "source", prompt(0, 512), 0)
    assert cache._group_checkpoints is None


def test_checkpoint_pressure_remains_bounded_and_never_blocks_live_allocations():
    cache = manager(slots=1)
    for group in range(12):
        prefill(cache, str(group), prompt(group, 512), 0)
        for pool in cache._group_pools.values():
            assert pool.num_free_blocks_in(0) == pool.blocks_per_partition
        assert not cache._group_checkpoints.owners
    assert len(cache._group_checkpoints.checkpoints) <= 2


def test_duplicate_document_checkpoints_do_not_accumulate_request_identities():
    cache = manager()
    for index in range(32):
        prefill(cache, str(index), prompt(0, 512), 0)
    assert len(cache._group_checkpoints.checkpoints) == 1
    assert not cache._group_checkpoints.owners


def test_inflight_snapshot_keeps_cache_state_aligned_before_next_chunk_confirms():
    cache = manager()
    hashes = cache.compute_group_block_hashes(prompt(0, 512))
    first = cache.ensure_group_blocks("source", 128, partition=0)
    cache.retain_group_block_snapshot(first, 0)
    second = cache.ensure_group_blocks("source", 256)
    cache.retain_group_block_snapshot(second, 0)
    published = cache.cache_group_blocks_from_snapshot(hashes, 128, {}, first, 0, request_id="source")
    cache.release_group_block_snapshot(first, 0)
    cache.cache_group_blocks_from_snapshot(hashes, 256, published, second, 0, request_id="source")
    cache.release_group_block_snapshot(second, 0)
    cache.release_all_group_requests("source")
    assert query(cache, prompt(0, 1024)) == 256


@pytest.mark.parametrize("external", [False, True], ids=["ordinary-replica", "external-decode"])
def test_eight_groups_with_depth_two_snapshots_keep_only_complete_checkpoints(external):
    cache = manager()
    scheduler = Scheduler(SchedulerConfig(max_num_running_reqs=8, max_num_scheduled_tokens=8192,
        long_prefill_token_threshold=128, max_prefill_tokens_per_request=128, max_seq_len=655360,
        num_speculative_tokens=7, speculative_prefix_cache_replay_tokens=128,
        async_scheduling=True), cache)
    for turn in range(3):
        length = 2048 + turn * 2048
        requests = []
        for group in (2, 7, 5, 1, 3, 0, 4, 6):
            rid, tokens = f"{group}-{turn}", prompt(group, length)
            if external:
                reserve(cache, rid, tokens)
                commit(cache, rid)
                request, _ = scheduler.admit_prefilled(reservation_id=rid, request_id=rid,
                    prompt_token_ids=tokens, first_token=11, max_new_tokens=512)
            else:
                request = Request(rid, tokens, max_new_tokens=512)
                scheduler.add_request(request)
            requests.append(request)
        pending = []
        for _ in range(1500):
            output = scheduler.schedule()
            if output.scheduled_requests:
                tokens = {s.request.request_id: ([11] if s.is_prefill else [11] * 8)
                    for s in output.scheduled_requests
                    if s.num_computed_tokens + s.num_new_tokens >= s.request.num_prompt_tokens}
                scheduler.advance_after_schedule(output)
                pending.append((output, tokens))
            if pending and (len(pending) == 2 or not output.scheduled_requests):
                scheduler.update_from_output(*pending.pop(0))
            if all(request.status.is_finished for request in requests):
                for item in pending:
                    scheduler.update_from_output(*item)
                break
        else:
            pytest.fail("bounded eight-group async workload did not drain")
        assert all(len(request.output_token_ids) == 512 for request in requests)
        if external:
            for request in requests:
                cache.retire_group_cache_reservation(request.request_id)
        # Two live snapshots add pressure beyond a sequential allocation test.
        # A bounded cache may retire a checkpoint, but never claim a hit with
        # only its KV half or leave snapshot/readonly references behind.
        hits = [query(cache, prompt(group, length + 2048), TARGET if external else None)
            for group in range(8)]
        assert any(hits)
        assert all(0 <= hit <= (length - 81) // 128 * 128 and hit % 128 == 0 for hit in hits)
        assert all(block.ref_cnt == 0 for pool in cache._group_pools.values() for block in pool.blocks)


@pytest.mark.parametrize("stop_after_prefill,async_scheduling", [(False, False), (False, True), (True, False)],
    ids=["replica-k7-sync", "replica-k7-async", "prefill"])
def test_scheduler_reuses_the_same_cache_state_checkpoint(stop_after_prefill, async_scheduling):
    cache = manager()
    scheduler = Scheduler(SchedulerConfig(max_num_running_reqs=8, max_num_scheduled_tokens=8192,
        long_prefill_token_threshold=128, max_prefill_tokens_per_request=128, max_seq_len=655360,
        num_speculative_tokens=0 if stop_after_prefill else 7,
        speculative_prefix_cache_replay_tokens=0 if stop_after_prefill else 128,
        async_scheduling=async_scheduling, stop_after_prefill=stop_after_prefill), cache)
    cold = Request("cold", prompt(0, 512), max_new_tokens=512)
    scheduler.add_request(cold)
    pending = []
    for _ in range(600):
        output = scheduler.schedule()
        if output.scheduled_requests:
            step, = output.scheduled_requests
            final = step.num_computed_tokens + step.num_new_tokens >= cold.num_prompt_tokens
            tokens = {"cold": [10] if step.is_prefill else [11, 12, 13]} if final else {}
            if async_scheduling:
                scheduler.advance_after_schedule(output)
            pending.append((output, tokens))
        assert pending
        if not async_scheduling or len(pending) == 2 or not output.scheduled_requests:
            scheduler.update_from_output(*pending.pop(0))
        if cold.prefill_complete:
            scheduler.finish_prefilled_request("cold")
        if cold.status.is_finished:
            for item in pending:
                scheduler.update_from_output(*item)
            break
    else:
        pytest.fail("Scheduler did not finish the bounded warmup request")
    assert len(cold.output_token_ids) == (1 if stop_after_prefill else 512)

    hot = Request("hot", prompt(0, 1024), max_new_tokens=512)
    scheduler.add_request(hot)
    output = scheduler.schedule()
    assert output.prefix_cache_hits == 384
    step, = output.scheduled_requests
    assert step.num_computed_tokens == 384
    assert set(step.block_ids_by_group) == {name for name, *_ in GEOMETRY}
    assert all(step.block_ids_by_group.values())
    if async_scheduling:
        scheduler.advance_after_schedule(output)
    scheduler.update_from_output(output, {})
    scheduler.abort_request("hot")
    assert not cache._group_checkpoints.owners


@pytest.mark.parametrize("p_enabled,d_enabled", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("p_length,d_length", [(512, 1024), (1024, 512), (512, 512)])
def test_prefill_compute_hit_and_decode_network_hit_are_independent(p_enabled, d_enabled, p_length, d_length):
    p_cache, d_cache = manager(p_enabled), manager(d_enabled)
    prefill(p_cache, "warm-p", prompt(0, p_length), 0)
    reserve(d_cache, "warm-d", prompt(0, d_length), 0)
    commit(d_cache, "warm-d")
    d_cache.release_group_cache("warm-d")
    tokens = prompt(0, 1536)
    _, p_hit, _ = p_cache.acquire_group_prefix_blocks("p", p_cache.compute_group_block_hashes(tokens),
        max_cache_hit_tokens=len(tokens) - 128)
    lease = reserve(d_cache, "d", tokens)
    assert p_hit == ((p_length - 81) // 128 * 128 if p_enabled else 0)
    assert lease.prefix_hit_tokens == ((d_length - 81) // 128 * 128 if d_enabled else 0)
    for name in ("hca_state", "csa_state", "csa_inner_state"):
        assert not lease.shared_block_ids_by_group[name]
        assert lease.writable_block_ids_by_group[name]  # Request-local final state is never reused.
    p_cache.release_all_group_requests("p")
    d_cache.release_group_cache("d")


def test_full_decode_kv_hit_still_plans_all_final_states_at_real_ring_geometry():
    from pypto_serving.pd.adapter import COMPONENTS, DeepSeekDSpark
    from pypto_serving.pd.integration import Chunk
    from pypto_serving.pd.planner import Geometry, plan_chunk
    from pypto_serving.pd.protocol import Binding, Node, Placement
    p_cache, d_cache = manager(), manager()
    tokens = prompt(0, 512)
    prefill(p_cache, "warm-p", tokens, 0)
    reserve(d_cache, "warm-d", tokens, 0)
    commit(d_cache, "warm-d")
    d_cache.release_group_cache("warm-d")
    adapter = DeepSeekDSpark()
    groups = {g.name: g for g in p_cache.group_specs}
    geometry = tuple(Geometry(c.name, (0,), p_cache.group_num_blocks(c.group) + 4,
        groups[c.group].spec.storage_block_size, 64, "uint8") for c in COMPONENTS)
    contract = adapter.contract("weights", geometry, (16, 4))
    p = Node("p", "prefill", "p-boot", contract, "p-registry")
    d = Node("d", "decode", "d-boot", contract, "d-registry")
    binding = Binding.create(p, d, {"prompt": "test"})
    _, p_hit, _ = p_cache.acquire_group_prefix_blocks(binding.request,
        p_cache.compute_group_block_hashes(tokens), max_cache_hit_tokens=len(tokens) - 1)
    lease = reserve(d_cache, binding.request, tokens)
    assert p_hit == 384 and lease.prefix_hit_tokens == 512
    tables = p_cache.ensure_group_blocks(binding.request, len(tokens))
    partition = p_cache.group_request_partition(binding.request)
    chunk = Chunk(binding.request, 0, p_hit, len(tokens), True, 17,
        {name: tuple(ids) for name, ids in tables.items()}, partition)
    placement = Placement(binding.request, lease.prefix_hit_tokens, lease.partition,
        {name: tuple(ids) for name, ids in lease.block_ids_by_group.items()})
    plan = plan_chunk(binding, chunk, placement, contract, p_cache.group_specs, geometry, geometry,
        p_hit, adapter.mapping(contract.topology, partition, lease.partition), "continuation")
    assert {copy.component for copy in plan.copies} == {"hca_state", "csa_state", "csa_inner_state"}
    p_cache.release_all_group_requests(binding.request)
    d_cache.release_group_cache(binding.request)
