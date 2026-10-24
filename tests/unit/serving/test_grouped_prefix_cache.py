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
from pypto_serving.serving.memory.kv_cache import KvCacheManager, KVCacheCapacityError
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
    def fail(*args, **kwargs):
        raise KVCacheCapacityError("forced capacity failure")
    monkeypatch.setattr(cache, "ensure_group_blocks", fail)
    with pytest.raises(KVCacheCapacityError):
        reserve(cache, "failed", prompt(0, 1024))
    assert [[b.ref_cnt for b in p.blocks] for p in cache._group_pools.values()] == before
    assert "failed" not in cache._group_request_partitions


@pytest.mark.parametrize("next_length", [2048, 4096], ids=["no-receive-suffix", "writable-suffix"])
@pytest.mark.parametrize("partition", [None, 0], ids=["pooled-placement", "fixed-placement"])
def test_busy_prefix_partition_can_use_cold_capacity(next_length, partition):
    cache = manager()
    reserve(cache, "source", prompt(0, 2048), 0)
    commit(cache, "source")
    cache.release_group_cache("source")
    # Two in-flight destinations leave nine free ori pages. Matching the
    # immutable four-page tail leaves five pages. An identical prompt fits
    # without reserving future Decode pages; a long receive suffix does not.
    for group in (1, 2):
        reserve(cache, f"live-{group}", prompt(group, 2048), 0)
    before = {name: list(pool.request_blocks["live-1"]) for name, pool in cache._group_pools.items()}
    lease = reserve(cache, "receiver", prompt(0, next_length), partition)
    if next_length == 2048:
        assert lease.prefix_hit_tokens == 2048 and lease.shared_prefix_pinned
        assert lease.partition == 0
    else:
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


@pytest.mark.parametrize("d_tp", [1, 4])
def test_full_decode_kv_hit_still_plans_all_final_states_at_real_ring_geometry(d_tp):
    from pypto_serving.pd.adapter import CACHE_COMPONENTS, DeepSeekDSpark
    from pypto_serving.pd.integration import Chunk
    from pypto_serving.pd.planner import Geometry, plan_chunk
    from pypto_serving.pd.protocol import Binding, Node, Placement, Topology
    p_cache, d_cache = manager(), manager(partitions=16 // d_tp)
    tokens = prompt(0, 512)
    prefill(p_cache, "warm-p", tokens, 0)
    reserve(d_cache, "warm-d", tokens, 0)
    commit(d_cache, "warm-d")
    d_cache.release_group_cache("warm-d")
    adapter = DeepSeekDSpark()
    groups = {g.name: g for g in p_cache.group_specs}
    geometry = tuple(Geometry(c.name, (0,), p_cache.group_num_blocks(c.group) + 4,
        groups[c.group].spec.storage_block_size, 64, "uint8") for c in CACHE_COMPONENTS) + (
            Geometry("bootstrap_hidden", (0,), 64, 128, 64, "uint8"),)
    contract = adapter.contract("weights", geometry)
    p = Node("p", "prefill", "p-boot", contract, "p-registry", Topology.for_tp(16, 4))
    d = Node("d", "decode", "d-boot", contract, "d-registry", Topology.for_tp(16, d_tp))
    binding = Binding.create(p, d, {"prompt": "test"})
    _, p_hit, _ = p_cache.acquire_group_prefix_blocks(binding.request,
        p_cache.compute_group_block_hashes(tokens), max_cache_hit_tokens=len(tokens) - 1)
    lease = reserve(d_cache, binding.request, tokens)
    assert p_hit == 384 and lease.prefix_hit_tokens == 512
    tables = p_cache.ensure_group_blocks(binding.request, len(tokens))
    partition = p_cache.group_request_partition(binding.request)
    chunk = Chunk(binding.request, 0, p_hit, len(tokens), True, 17,
        {name: tuple(ids) for name, ids in tables.items()}, partition, {"bootstrap": 3})
    placement = Placement(binding.request, lease.prefix_hit_tokens, lease.partition,
        {name: tuple(ids) for name, ids in lease.block_ids_by_group.items()}, {"bootstrap": 5})
    plan = plan_chunk(binding, chunk, placement, contract, p_cache.group_specs, geometry, geometry,
        p_hit, adapter.mapping(p.topology, d.topology, partition, lease.partition), "continuation")
    assert {copy.component for copy in plan.copies} == {
        "hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"}
    assert len(plan.units) == 9 * d_tp
    p_cache.release_all_group_requests(binding.request)
    d_cache.release_group_cache(binding.request)


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


@pytest.mark.parametrize("asynchronous", [False, True], ids=["prefill", "replica-k7"])
def test_growing_eight_groups_reuse_the_previous_joint_checkpoint(asynchronous):
    cache = manager()
    scheduler = _prefill_scheduler(cache, speculative=7 if asynchronous else 0, asynchronous=asynchronous)
    for turn in range(3):
        length = 8205 + 512 * turn
        expected = (length - 512 - 128) // 128 * 128 if turn else 0
        for group in range(8):
            cache._next_group_partition = group % 4
            _finish_prefill(scheduler,
                Request(f"{group}-{turn}", prompt(group, length), max_new_tokens=512), expected_hit=expected)
    assert not cache._group_checkpoints.owners
    assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)


@pytest.mark.parametrize("length", [127, 128, 129, 255, 256, 257])
@pytest.mark.parametrize("replay", [0, 128])
def test_short_prompt_checkpoint_never_exceeds_its_query_limit(length, replay):
    cache = manager(partitions=1)
    scheduler = _prefill_scheduler(cache, replay=replay)
    boundary = _finish_prefill(scheduler, Request("short", prompt(0, length), max_new_tokens=512))
    checkpoints = cache._group_checkpoints
    assert (sorted(key[1] for key in checkpoints.checkpoints) if checkpoints else []) == (
        [boundary] if boundary else [])


@pytest.mark.parametrize("aborted", [False, True], ids=["disabled-prefix", "failed-inflight"])
def test_unconfirmed_or_disabled_prefix_has_no_request_checkpoint(aborted):
    cache = manager(enabled=aborted, partitions=1)
    scheduler = _prefill_scheduler(cache, speculative=7, asynchronous=True)
    request = Request("unfinished", prompt(0, 513), max_new_tokens=512)
    scheduler.add_request(request)
    first = scheduler.schedule()
    scheduler.advance_after_schedule(first)
    second = scheduler.schedule()
    scheduler.advance_after_schedule(second)
    scheduler.update_from_output(first, {})
    third = scheduler.schedule()
    scheduler.advance_after_schedule(third)
    scheduler.update_from_output(second, {})
    # The target is 384, while only the first 256 tokens are confirmed. Even
    # queued work must not publish a target checkpoint before it completes.
    assert cache._group_checkpoints is None
    scheduler.discard_scheduled_request(third.scheduled_requests[0])
    scheduler.abort_request(request.request_id)
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


@pytest.mark.parametrize("partitions", [4, 16])
def test_local_replay_checkpoint_does_not_limit_decode_network_hit(partitions):
    tokens = prompt(0, 8205)
    p_cache, d_cache = manager(), manager(partitions=partitions)
    scheduler = _prefill_scheduler(p_cache)
    _finish_prefill(scheduler, Request("warm-p", tokens, max_new_tokens=512))
    scheduler.add_request(Request("hot-p", tokens, max_new_tokens=512))
    p_hit = scheduler.schedule().prefix_cache_hits
    scheduler.abort_request("hot-p")
    reserve(d_cache, "warm-d", tokens)
    commit(d_cache, "warm-d")
    d_cache.release_group_cache("warm-d")
    lease = reserve(d_cache, "next", tokens)
    assert p_hit == 8064 and lease.prefix_hit_tokens == 8192
    assert all(not lease.shared_block_ids_by_group[name]
        for name in ("hca_state", "csa_state", "csa_inner_state"))
    d_cache.release_group_cache("next")


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
@pytest.mark.parametrize("length", [17, 128, 8205])
def test_large_logical_output_only_reserves_received_pages_and_early_eos_frees_them(partitions, length):
    cache = paged_manager(partitions)
    tokens = prompt(0, length) if length >= 81 else list(range(length))
    lease = cache.reserve_group_cache("early", "early", length + 131072, prompt_token_count=length)
    assert lease.token_capacity == length + 131072
    for name, pool in cache._group_pools.items():
        expected = (length + pool.spec.spec.token_capacity - 1) // pool.spec.spec.token_capacity
        assert pool.request_logical_blocks["early"] == expected
        assert len(lease.block_ids_by_group[name]) == min(expected, pool.spec.max_blocks_per_seq)
    commit(cache, "early")
    scheduler = Scheduler(SchedulerConfig(max_seq_len=655360), cache)
    request, first = scheduler.admit_prefilled(reservation_id="early", request_id="early",
        prompt_token_ids=tokens, first_token=11, max_new_tokens=131072, eos_token_id=11)
    assert first.finished and request.status.name == "FINISHED_EOS"
    assert not cache._group_request_partitions
    assert all(block.ref_cnt == 0 for pool in cache._group_pools.values() for block in pool.blocks)


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
