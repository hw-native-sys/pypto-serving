# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resolved Host pins preserve grouped page ownership through asynchronous use."""

import gc
import math
import weakref
from collections import Counter
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.pd.planner import SourcePins
from pypto_serving.serving.memory.kv_cache import KVCacheCapacityError, KvCacheManager, _GroupBlockPool
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig
from pypto_serving.serving.server.ipc import decode_command, encode_command
from tests.unit.serving.engine.test_async_pipeline import _async_pipeline_core
from tests.unit.serving.test_grouped_prefix_cache import GEOMETRY


def cache(partitions=2, enabled=True):
    result = KvCacheManager(enable_prefix_cache=enabled)
    result.init_groups(tuple(KVCacheGroupSpec(name, (0,), KVCacheSpec(2, 4), ring,
        num_blocks=256 if window is None else 64, num_partitions=partitions, sliding_window=window)
        for name, ring, window in (("history", 128, None), ("state", 3, 4))), max_batch_size=2)
    return result


def refs(manager):
    return tuple(b.ref_cnt for p in manager._group_pools.values() for b in p.blocks)


def assert_free(manager):
    assert not any(refs(manager))
    for pool in manager._group_pools.values():
        for partition, queue in enumerate(pool.free_queues):
            pages, previous, block = [], None, queue.head
            while block is not None:
                assert block.prev_free is previous and block.ref_cnt == 0
                assert block.block_id // pool.blocks_per_partition == partition
                pages.append(block.block_id)
                assert len(pages) <= pool.blocks_per_partition
                previous, block = block, block.next_free
            assert previous is queue.tail
            assert len(set(pages)) == queue.count == pool.blocks_per_partition


@pytest.mark.parametrize("partition", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_resolved_pin_survives_input_mutation_and_releases_without_lookup(monkeypatch, partition, enabled):
    manager = cache(enabled=enabled)
    tables = manager.ensure_group_blocks("r", 7, partition=partition)
    expected = sum(map(len, tables.values()))
    original = _GroupBlockPool.block_from_local_id
    lookups = []
    def resolve(pool, part, block):
        lookups.append((pool.spec.name, part, block))
        return original(pool, part, block)
    monkeypatch.setattr(_GroupBlockPool, "block_from_local_id", resolve)
    pin = manager.retain_group_block_snapshot(tables, partition)
    assert len(lookups) == expected
    for ids in tables.values():
        ids[:] = [999999]
    manager.release_all_group_requests("r")
    assert sum(refs(manager)) == expected
    # Unpin must use the resolved physical pages, never a request's latest table.
    lookups.clear()
    manager.release_group_block_snapshot(pin)
    assert not lookups and pin._owner is None and not pin._groups
    assert_free(manager)


@pytest.mark.parametrize("bad", ["missing_group", "extra_group", "duplicate", "negative", "bounds", "unowned",
                                 "negative_partition", "partition_bounds"])
def test_failed_retain_is_atomic_across_groups(bad):
    manager = cache()
    tables = manager.ensure_group_blocks("r", 4, partition=0)
    # Fail in the last group, after the history pages have been resolved.
    if bad == "missing_group":
        del tables["state"]
    elif bad == "extra_group":
        tables["extra"] = []
    elif bad == "duplicate":
        tables["state"] *= 2
    elif bad in ("negative", "bounds", "unowned"):
        tables["state"] = [{"negative": -1, "bounds": 64, "unowned": 63}[bad]]
    partition = {"negative_partition": -1, "partition_bounds": 2}.get(bad, 0)
    before = refs(manager)
    with pytest.raises((ValueError, RuntimeError)):
        manager.retain_group_block_snapshot(tables, partition)
    assert refs(manager) == before
    manager.release_all_group_requests("r")
    assert_free(manager)


@pytest.mark.parametrize("tables", [{}, {"history": [], "state": []}])
def test_empty_snapshot_and_double_or_foreign_release(tables):
    manager, other = cache(), cache()
    pin = manager.retain_group_block_snapshot(tables, 0)
    with pytest.raises(RuntimeError, match="another manager"):
        other.release_group_block_snapshot(pin)
    manager.release_group_block_snapshot(pin)
    with pytest.raises(RuntimeError, match="already released"):
        manager.release_group_block_snapshot(pin)
    assert_free(manager)
    assert_free(other)


def test_foreign_and_duplicate_release_cannot_remove_a_live_pin():
    manager, other = cache(), cache()
    pin = manager.retain_group_block_snapshot(manager.ensure_group_blocks("r", 4, partition=0), 0)
    before = refs(manager)
    with pytest.raises(RuntimeError, match="another manager"):
        other.release_group_block_snapshot(pin)
    assert refs(manager) == before
    manager.release_group_block_snapshot(pin)
    before = refs(manager)
    with pytest.raises(RuntimeError, match="already released"):
        manager.release_group_block_snapshot(pin)
    assert refs(manager) == before
    manager.release_all_group_requests("r")
    assert_free(manager)
    assert_free(other)


@pytest.mark.parametrize("partition", [0, 1])
def test_late_release_does_not_touch_reused_request_id(partition):
    manager = cache()
    old = manager.ensure_group_blocks("r", 8, partition=partition)
    pin = manager.retain_group_block_snapshot(old, partition)
    manager.release_all_group_requests("r")
    new = manager.ensure_group_blocks("r", 8, partition=partition)
    assert all(set(old[name]).isdisjoint(new[name]) for name in old)
    manager.release_group_block_snapshot(pin)
    for name, ids in new.items():
        assert all(manager._group_pools[name].block_from_local_id(partition, b).ref_cnt == 1 for b in ids)
    manager.release_all_group_requests("r")
    assert_free(manager)


@pytest.mark.parametrize("length", [1, 31, 32, 33, 127, 128, 129])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_depth_two_keeps_old_ring_pages_through_cow_and_request_release(length, order):
    manager = cache()
    first = manager.ensure_group_blocks("r", length, partition=1)
    pins = [manager.retain_group_block_snapshot(first, 1)]
    second = manager.ensure_group_blocks("r", length + 4, partition=1)
    assert first["state"] != second["state"]
    pins.append(manager.retain_group_block_snapshot(second, 1))
    # Source or scheduler retirement does not remove either in-flight reference.
    manager.release_all_group_requests("r")
    manager.release_group_block_snapshot(pins[order[0]])
    assert any(refs(manager))
    manager.release_group_block_snapshot(pins[order[1]])
    assert_free(manager)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32, 64])
def test_release_resolution_count_does_not_grow_with_batch(monkeypatch, batch):
    manager = cache(partitions=16)
    tables = [manager.ensure_group_blocks(str(i), 8, partition=i % 16) for i in range(batch)]
    pins = [manager.retain_group_block_snapshot(table, i % 16) for i, table in enumerate(tables)]
    for i in range(batch):
        manager.release_all_group_requests(str(i))
    def unexpected(*args):
        pytest.fail("release performed an ID-to-block lookup")
    monkeypatch.setattr(_GroupBlockPool, "block_from_local_id", unexpected)
    for pin in reversed(pins):
        manager.release_group_block_snapshot(pin)
    assert_free(manager)


@pytest.mark.parametrize("threshold", [128, 2048, 8192])
@pytest.mark.parametrize("partitions", [4, 16])
@pytest.mark.parametrize("enabled", [False, True])
def test_eight_groups_keep_both_snapshots_with_real_window_geometry(threshold, partitions, enabled):
    manager = KvCacheManager(enable_prefix_cache=enabled)
    rows = math.ceil(8 / partitions)
    specs = []
    for name, page, ratio, _, window in GEOMETRY:
        ring = math.ceil(655360 / page) if window is None else math.ceil((window - 1 + threshold) / page) + 1
        capacity = math.ceil(8320 / page) if window is None else ring
        specs.append(KVCacheGroupSpec(name, (0,), KVCacheSpec(page, page // ratio, ratio), ring,
            num_blocks=max(ring, rows * capacity * 3), num_partitions=partitions, sliding_window=window))
    manager.init_groups(tuple(specs), max_batch_size=8)
    pins = []
    for length in (8205, 8320):
        for group in range(8):
            tables = manager.ensure_group_blocks(str(group), length, partition=group % partitions)
            pins.append(manager.retain_group_block_snapshot(tables, group % partitions))
    expected = Counter(id(block) for pin in pins for _, blocks in pin._groups for block in blocks)
    for group in range(8):
        manager.release_all_group_requests(str(group))
    for pool in manager._group_pools.values():
        assert all(block.ref_cnt == expected[id(block)] for block in pool.blocks)
    for pin in reversed(pins):
        manager.release_group_block_snapshot(pin)
    assert_free(manager)


def test_source_pins_use_one_handle_and_hold_original_pages():
    manager = cache()
    tables = manager.ensure_group_blocks("r", 4, partition=0)
    pins = SourcePins(manager)
    chunk = SimpleNamespace(request="r", partition=0, blocks={k: tuple(v) for k, v in tables.items()})
    pins.retain(chunk)
    before = refs(manager)
    with pytest.raises(RuntimeError, match="one transfer"):
        pins.retain(chunk)
    assert refs(manager) == before
    manager.ensure_group_blocks("r", 8, partition=0)
    manager.release_all_group_requests("r")
    assert any(refs(manager))
    handle = pins.active["r"]
    pins.release("r")
    assert not pins.active and handle._owner is None
    assert_free(manager)


@pytest.mark.parametrize("ending", ["finish", "abort", "discard", "publish_error"])
def test_scheduler_releases_exact_handle_and_never_sends_it_over_ipc(monkeypatch, ending):
    manager = cache(enabled=False)
    scheduler = Scheduler(SchedulerConfig(async_scheduling=True, enable_prefix_cache=False,
        max_num_scheduled_tokens=128, max_seq_len=256), manager)
    request = Request("r", list(range(7)), max_new_tokens=1)
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduled, = output.scheduled_requests
    scheduler.advance_after_schedule(output)
    pin = scheduled.group_block_snapshot
    assert pin is not None
    core, _ = _async_pipeline_core()
    core.scheduler, core.kv_cache_manager = scheduler, manager
    # Building a real wire command with a retained schedule cannot serialize the handle.
    command = core._build_step_command(output, [], step_id=1)
    assert decode_command(encode_command(command)) == command
    if ending == "abort":
        scheduler.abort_request("r")
        scheduler.update_from_output(output, {"r": [11]})
    elif ending == "discard":
        scheduler.abort_request("r")
        scheduler.discard_scheduled_request(scheduled)
    elif ending == "publish_error":
        def fail(*args, **kwargs):
            raise ValueError("publication failed")
        monkeypatch.setattr(manager, "cache_group_blocks_from_snapshot", fail)
        with pytest.raises(ValueError, match="publication failed"):
            scheduler.update_from_output(output, {"r": [11]})
        scheduler.abort_request("r")
    else:
        scheduler.update_from_output(output, {"r": [11]})
    assert scheduled.group_block_snapshot is None and pin._owner is None
    scheduler.discard_scheduled_request(scheduled)  # Cleanup remains idempotent.
    assert_free(manager)


@pytest.mark.parametrize("ending", ["commit", "abort", "unknown", "allocation_error", "cold_retry"])
def test_shared_prefix_reservation_releases_handle_only_after_stop_proof(monkeypatch, ending):
    manager = cache()
    tokens = list(range(9))
    hashes = manager.compute_group_block_hashes(tokens)
    manager.ensure_group_blocks("warm", 4, partition=0)
    manager.cache_group_blocks("warm", hashes, 4, {})
    manager.release_all_group_requests("warm")
    acquired, allocate = [], manager.ensure_group_blocks
    retain = manager.retain_group_block_snapshot
    def retain_pin(*args):
        pin = retain(*args)
        acquired.append(pin)
        return pin
    monkeypatch.setattr(manager, "retain_group_block_snapshot", retain_pin)
    def fail_once(*args, **kwargs):
        monkeypatch.setattr(manager, "ensure_group_blocks", allocate)
        raise (KVCacheCapacityError("retry") if ending == "cold_retry" else ValueError("allocation failed"))
    if ending in ("allocation_error", "cold_retry"):
        monkeypatch.setattr(manager, "ensure_group_blocks", fail_once)
    if ending == "allocation_error":
        with pytest.raises(ValueError, match="allocation failed"):
            manager.reserve_group_cache("r", "r", 100, prompt_token_count=9, prefix_block_hashes=hashes,
                                        shareable_group_names=tuple(hashes))
    else:
        reservation = manager.reserve_group_cache("r", "r", 100, prompt_token_count=9,
            prefix_block_hashes=hashes, shareable_group_names=tuple(hashes))
        if ending == "cold_retry":
            assert reservation.prefix_hit_tokens == 0 and reservation.shared_prefix_snapshot is None
        else:
            assert reservation.prefix_hit_tokens == 4 and reservation.shared_prefix_snapshot is acquired[0]
        if ending == "commit":
            manager.authorize_group_cache_write("r")
            ready = manager.commit_group_cache("r")
            assert ready.shared_prefix_snapshot is None and acquired[0]._owner is None
        if ending == "unknown":
            manager.authorize_group_cache_write("r")
            manager.quarantine_group_cache("r")
            before = refs(manager)
            with pytest.raises(RuntimeError, match="native writer"):
                manager.release_group_cache("r")
            assert refs(manager) == before and acquired[0]._owner is manager
        manager.release_group_cache("r", confirmed_stopped=ending == "unknown")
        manager.release_group_cache("r")
        manager.retire_group_cache_reservation("r")
    assert acquired and all(pin._owner is None and not pin._groups for pin in acquired)
    assert_free(manager)


def test_retired_handles_do_not_retain_manager_or_accumulate_pages():
    manager = cache()
    retired = []
    for turn in range(256):
        request = str(turn % 8)
        tables = manager.ensure_group_blocks(request, 8, partition=turn % 2)
        pin = manager.retain_group_block_snapshot(tables, turn % 2)
        manager.release_all_group_requests(request)
        manager.release_group_block_snapshot(pin)
        retired.append(pin)
    assert_free(manager)
    reference = weakref.ref(manager)
    del manager
    gc.collect()
    assert reference() is None
    assert all(pin._owner is None and pin._groups == () for pin in retired)
