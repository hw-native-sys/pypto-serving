# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise real state address readers across retained inputs and compact outputs."""

import ast
import math
from dataclasses import replace
from functools import lru_cache

import pytest
import torch

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.model.deepseek_dspark.cache_capacity import cache_pool_bytes, plan_cache_pages
from pypto_serving.model.deepseek_dspark.npu_runner import build_dspark_cache_group_specs
from pypto_serving.serving.memory.kv_cache import KvCacheManager, aligned_prefix_cache_boundary
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig
from tests.unit.model.deepseek_dspark.test_state_checkpoint import LIB, RATIOS, STATE, _CPU, publish


class CPU(_CPU):
    INT64 = int


@lru_cache(None)
def address_program(name, prefix):
    tree = ast.parse((LIB / STATE[name][0]).read_text())
    tile = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.endswith("_tile"))
    variable = prefix + "_state_row_raw"
    for parent in ast.walk(tile):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for i, n in enumerate(body):
            if (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == variable):
                # Execute the production address selection, without reimplementing it.
                return compile(ast.fix_missing_locations(ast.Module(body[i:i + 2], type_ignores=[])),
                               str(LIB / STATE[name][0]), "exec")
    raise AssertionError(variable)


def address(name, q, *, start, tile, read_table, write_map, packed=0, prefix=None):
    prefix = prefix or ("pool" if name == "hca_state" else "prev")
    page = STATE[name][3]
    env = dict(pl=CPU, request_start=packed, request_position=start, tile_first_position=tile,
               pool_abs=q, pool_source=packed + q - start, prev_abs=q, cur_abs=q,
               HCA_STATE_BLOCK_SIZE=page, CSA_STATE_BLOCK_SIZE=page, INNER_STATE_BLOCK_SIZE=page,
               compress_state_block_table=read_table, inner_compress_state_block_table=read_table,
               state_slot_mapping=write_map, inner_state_slot_mapping=write_map)
    exec(address_program(name, prefix), env)
    return env[prefix + "_state_row_raw"]


def state_cache(name, *, prefix=True, extra=0):
    _, history, _, page = STATE[name]
    ring = math.ceil((history - 1 + 1024) / page) + 1
    entry = (history + 2 * page - 2) // page
    count = 2 * ring + 2 * entry + history // page + extra
    spec = KVCacheGroupSpec(name, (0,), KVCacheSpec(page, 16 * page), ring,
                           num_blocks=count, sliding_window=history,
                           is_checkpoint_state=True, preserve_prefill_history=True)
    cache = KvCacheManager(enable_prefix_cache=prefix)
    cache.init_groups((spec,), max_batch_size=1)
    return cache, torch.full((count * page, 4), float("nan"))


def row_value(q, name):
    ratio = 128 if name == "hca_state" else 4
    return [q, q + 1, q + 100 + q % ratio, q + 101 + q % ratio]


def read_table_for(name, read, end):
    page = STATE[name][3]
    table = torch.full(((end + page - 1) // page + 1,), -1, dtype=torch.int32)
    for q, physical in read.get(name, {}).items():
        table[q] = physical
    return table


def execute_chunk(name, buffer, ids, read, *, start, end, checkpoint, packed=0):
    page = STATE[name][3]
    table = torch.tensor(ids)
    positions = torch.arange(start, end)
    mapping = table[positions // page % len(table)] * page + positions % page
    mapping = torch.cat((torch.full((packed,), -1, dtype=torch.int64), mapping))
    entry_table = read_table_for(name, read, end)
    ratio = 128 if name == "hca_state" else 4
    window = 128 if name == "hca_state" else 8
    for tile in range(start, end, 512):
        stop = min(tile + 512, end)
        first = (tile // ratio + 1) * ratio - 1
        if first < stop:
            # The first compression in a tile covers every earlier state row
            # any compression in that tile can read. Current rows use scratch.
            for q in range(max(0, first + 1 - window), min(tile, first + 1)):
                prefixes = ("pool",) if name == "hca_state" else ("prev", "cur")
                for prefix in prefixes:
                    row = address(name, q, start=start, tile=tile, read_table=entry_table,
                                  write_map=mapping, packed=packed, prefix=prefix)
                    assert row >= 0, (name, start, tile, q, read)
                    assert buffer[row].tolist() == row_value(q, name), (name, start, tile, q)
        if name != "hca_state":
            for prefix in ("prev", "cur"):
                assert address(name, tile, start=start, tile=tile, read_table=entry_table,
                               write_map=mapping, packed=packed, prefix=prefix) == -1
        publish(name, buffer, table, start=tile, rows=stop - tile,
                checkpoint=checkpoint, packed_offset=packed)
    return mapping


def seed_history(cache, buffer, name, start):
    ids = cache.ensure_group_blocks("request", start)[name]
    if start:
        publish(name, buffer, torch.tensor(ids), start=max(0, start - 512), rows=min(start, 512), checkpoint=0)
    return ids


@pytest.mark.parametrize("name", STATE)
@pytest.mark.parametrize("start,length", [(0, 8192), (1, 31), (3, 7), (31, 33), (32, 128),
    (33, 513), (127, 8192), (128, 513), (129, 1024), (511, 8), (512, 513), (513, 8192),
    (8192, 128), (8205, 8192), (8319, 513), (8320, 8192), (1040001, 8192), (1048445, 131)])
@pytest.mark.parametrize("prefix", [False, True])
def test_real_readers_keep_entry_history_through_multiple_wraps(name, start, length, prefix):
    cache, buffer = state_cache(name, prefix=prefix)
    seed_history(cache, buffer, name, start)
    read, pin = cache.retain_group_prefill_history("request", start)
    before = {(q, p): buffer[p * STATE[name][3]:(p + 1) * STATE[name][3]].clone()
              for q, p in read.get(name, {}).items()}
    ids = cache.ensure_group_blocks("request", start + length)[name]
    checkpoint = aligned_prefix_cache_boundary(start + length, 128, replay_tokens=128)
    execute_chunk(name, buffer, ids, read, start=start, end=start + length, checkpoint=checkpoint, packed=17)
    page = STATE[name][3]
    for q in range(max(0, start - STATE[name][1]), start):
        physical = read[name][q // page]
        # Only previously valid rows are immutable; unused partial-page rows
        # are not part of the input history contract.
        assert buffer[physical * page + q % page].tolist() == before[(q // page, physical)][q % page].tolist()
    if pin is not None:
        cache.release_group_block_snapshot(pin)
    cache.release_all_group_requests("request")
    assert all(b.ref_cnt == 0 for b in cache._group_pools[name].blocks)


@pytest.mark.parametrize("name", STATE)
def test_all_alignment_phases_use_the_real_state_readers(name):
    for phase in range(128):
        start = 8192 + phase
        cache, buffer = state_cache(name)
        seed_history(cache, buffer, name, start)
        read, pin = cache.retain_group_prefill_history("request", start)
        ids = cache.ensure_group_blocks("request", start + 1024)[name]
        execute_chunk(name, buffer, ids, read, start=start, end=start + 1024,
                      checkpoint=(start + 896) // 128 * 128, packed=phase % 32)
        cache.release_group_block_snapshot(pin)
        cache.release_all_group_requests("request")
        assert all(b.ref_cnt == 0 for b in cache._group_pools[name].blocks)


@pytest.mark.parametrize("name", STATE)
def test_prerotation_without_an_entry_view_really_loses_history(name):
    start = 8205
    cache, buffer = state_cache(name)
    seed_history(cache, buffer, name, start)
    ids = cache.ensure_group_blocks("request", start + 8192)[name]
    page = STATE[name][3]
    assert any(torch.isnan(buffer[ids[q // page % len(ids)] * page + q % page]).all()
               for q in range(start - STATE[name][1], start))
    cache.release_all_group_requests("request")


@pytest.mark.parametrize("name", STATE)
def test_missing_history_is_rejected_before_any_pin_is_acquired(name):
    cache = grouped_cache()
    ids = cache.ensure_group_blocks("request", 8192, partition=0)[name]
    pool = cache._group_pools[name]
    slot = (8191 // STATE[name][3]) % len(ids)
    block = pool.request_blocks["request"][slot]
    pool.request_blocks["request"][slot] = None
    before = {n: [b.ref_cnt for b in p.blocks] for n, p in cache._group_pools.items()}
    with pytest.raises(RuntimeError, match="Missing Prefill history"):
        cache.retain_group_prefill_history("request", 8192)
    assert before == {n: [b.ref_cnt for b in p.blocks] for n, p in cache._group_pools.items()}
    pool.request_blocks["request"][slot] = block
    cache.release_all_group_requests("request")
    assert all(b.ref_cnt == 0 for p in cache._group_pools.values() for b in p.blocks)


def grouped_cache(*, partitions=1, active=2, prefix=True, max_seq_len=32768, chunk=8192):
    groups = build_dspark_cache_group_specs(43, RATIOS, max_seq_len=max_seq_len,
        max_prefill_tokens=chunk, partitions=partitions, prefill_only=True)
    groups = tuple(replace(g, spec=replace(g.spec, page_size_bytes=132 if g.name == "idx" else 128))
                   for g in groups)
    def counts_for(g):
        p = g.spec.token_capacity
        if g.preserve_prefill_history:
            return active * (2 * g.max_blocks_per_seq + 2 * ((g.sliding_window + 2 * p - 2) // p)
                             + g.sliding_window // p)
        if g.sliding_window:
            return active * (g.max_blocks_per_seq + math.ceil(chunk / p) + g.sliding_window // p)
        return max(g.max_blocks_per_seq, active * math.ceil(chunk / p))
    minimum = {g.name: counts_for(g) for g in groups}
    budget = cache_pool_bytes(groups, minimum, filler_pages=64)
    counts = plan_cache_pages(groups, budget, active_requests=active, step_tokens=chunk, filler_pages=64)
    for g in groups:
        if g.sliding_window:
            assert counts[g.name] == minimum[g.name]
    cache = KvCacheManager(enable_prefix_cache=prefix)
    cache.init_groups(tuple(replace(g, num_blocks=counts[g.name]) for g in groups), max_batch_size=active * partitions)
    return cache


def scheduler_for(cache, *, asynchronous=True, chunk=8192):
    return Scheduler(SchedulerConfig(max_num_running_reqs=8, max_num_scheduled_tokens=8192,
        long_prefill_token_threshold=chunk, max_prefill_tokens_per_request=chunk,
        max_seq_len=32768, num_speculative_tokens=7, speculative_prefix_cache_replay_tokens=128,
        async_scheduling=asynchronous, stop_after_prefill=not asynchronous,
        enable_prefix_cache=cache.enable_prefix_cache), cache)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_minimum_group_budget_keeps_two_steps_inputs_outputs_and_checkpoint(asynchronous, prefix):
    cache = grouped_cache(prefix=prefix)
    scheduler = scheduler_for(cache, asynchronous=asynchronous)
    request = Request("pipeline", list(range(24621)), max_new_tokens=1)
    scheduler.add_request(request)
    buffers = {(0, name): torch.full((cache.group_num_blocks(name) * STATE[name][3], 4), float("nan"))
               for name in STATE}
    pending = []
    max_depth = 0
    for _ in range(12):
        output = scheduler.schedule()
        if output.scheduled_requests:
            step, = output.scheduled_requests
            if step.num_computed_tokens:
                assert step.group_read_snapshot is not None
                assert set(step.read_block_ids_by_group) == set(STATE)
            if asynchronous:
                scheduler.advance_after_schedule(output)
            pending.append(output)
            max_depth = max(max_depth, len(pending))
        if pending and (not asynchronous or len(pending) == 2 or not output.scheduled_requests):
            batch = pending.pop(0)
            step, = batch.scheduled_requests
            start, end = step.num_computed_tokens, step.num_computed_tokens + step.num_new_tokens
            for name in STATE:
                execute_chunk(name, buffers[(0, name)], step.block_ids_by_group[name], step.read_block_ids_by_group,
                              start=start, end=end, checkpoint=request.group_checkpoint_boundary or 0)
            scheduler.update_from_output(batch, {request.request_id: [11]} if end == request.num_prompt_tokens else {})
            assert step.group_read_snapshot is None and step.group_block_snapshot is None
        if request.prefill_complete or request.status.is_finished:
            break
    else:
        pytest.fail("Pipeline did not finish with the minimum page budget")
    assert max_depth == (2 if asynchronous else 1)
    assert not pending
    if request.prefill_complete:
        scheduler.finish_prefilled_request(request.request_id)
    assert all(b.ref_cnt == 0 for p in cache._group_pools.values() for b in p.blocks)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("chunk", [128, 1024, 8192])
def test_eight_groups_replay_grow_and_fork_with_real_checkpoint_contents(asynchronous, prefix, chunk):
    cache = grouped_cache(partitions=4, active=2, prefix=prefix, chunk=chunk)
    scheduler = scheduler_for(cache, asynchronous=asynchronous, chunk=chunk)
    buffers = {(part, name): torch.full((cache.group_num_blocks(name) * STATE[name][3], 4), float("nan"))
               for part in range(4) for name in STATE}
    hits = []
    for turn, length in enumerate((8205, 8205, 8733, 9001)):
        requests = []
        first_seen = set()
        for group in range(8):
            tokens = [1000000 + group] + list(range(1, length))
            if turn == 3:
                tokens[8192:] = [2000000 + group] * (length - 8192)
            request = Request(f"group-{group}-{turn}", tokens, max_new_tokens=1)
            requests.append(request)
            scheduler.add_request(request)
        pending = []
        for _ in range(100):
            output = scheduler.schedule()
            if output.scheduled_requests:
                for step in output.scheduled_requests:
                    assert step.is_prefill
                    if step.request.request_id not in first_seen:
                        first_seen.add(step.request.request_id)
                        hits.append(step.num_computed_tokens)
                        if prefix and turn:
                            assert step.num_computed_tokens >= 8064
                        else:
                            assert step.num_computed_tokens == 0
                if asynchronous:
                    scheduler.advance_after_schedule(output)
                pending.append(output)
            if pending and (not asynchronous or len(pending) == 2 or not output.scheduled_requests):
                batch = pending.pop(0)
                returned = {}
                for ordinal, step in enumerate(batch.scheduled_requests):
                    start, end = step.num_computed_tokens, step.num_computed_tokens + step.num_new_tokens
                    for name in STATE:
                        execute_chunk(name, buffers[(step.cache_partition, name)], step.block_ids_by_group[name],
                                      step.read_block_ids_by_group, start=start, end=end,
                                      checkpoint=step.request.group_checkpoint_boundary or 0, packed=ordinal * 7)
                    if end == step.request.num_prompt_tokens:
                        returned[step.request.request_id] = [11]
                scheduler.update_from_output(batch, returned)
            if all(r.prefill_complete or r.status.is_finished for r in requests):
                assert not pending
                break
        else:
            pytest.fail("Eight groups stalled with the minimum pool budget")
        for request in requests:
            if request.prefill_complete:
                scheduler.finish_prefilled_request(request.request_id)
            assert request.output_token_ids == [11]
            if prefix:
                boundary = request.group_checkpoint_boundary
                _, hit, partition = cache.acquire_group_prefix_blocks("inspect", request.group_block_hashes,
                    max_cache_hit_tokens=boundary)
                assert hit == boundary
                read, pin = cache.retain_group_prefill_history("inspect", hit)
                for name in STATE:
                    page, history = STATE[name][3], STATE[name][1]
                    for q in range(max(0, hit - history), hit):
                        row = read[name][q // page] * page + q % page
                        assert buffers[(partition, name)][row].tolist() == row_value(q, name)
                cache.release_group_block_snapshot(pin)
                cache.release_all_group_requests("inspect")
        assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)
    assert len(hits) == 32


@pytest.mark.parametrize("name", STATE)
@pytest.mark.parametrize("unexpected", [False, True])
def test_failed_allocation_rolls_back_only_new_entry_pins(name, unexpected, monkeypatch):
    from pypto_serving.serving.sched.scheduler import RequestStatus, ScheduledRequest
    cache, buffer = state_cache(name)
    ids = seed_history(cache, buffer, name, 8205)
    output_pin = cache.retain_group_block_snapshot({name: ids}, 0)
    cache.ensure_group_blocks("blocker", 8192, partition=0)
    request = Request("request", list(range(20000)), max_new_tokens=1,
                      status=RequestStatus.RUNNING, num_computed_tokens=8205,
                      allocated_group_block_ids={name: ids}, cache_partition=0)
    scheduler = scheduler_for(cache)
    scheduled = ScheduledRequest(request, 8192, True, num_computed_tokens=8205)
    before = [b.ref_cnt for b in cache._group_pools[name].blocks]
    if unexpected:
        monkeypatch.setattr(cache, "ensure_group_blocks", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad-view")))
        with pytest.raises(ValueError, match="bad-view"):
            scheduler._try_allocate_request_blocks(request, 8192, scheduled=scheduled)
    else:
        assert not scheduler._try_allocate_request_blocks(request, 8192, scheduled=scheduled)
    assert before == [b.ref_cnt for b in cache._group_pools[name].blocks]
    assert scheduled.group_read_snapshot is None and not scheduled.read_block_ids_by_group
    cache.release_group_block_snapshot(output_pin)
    cache.release_all_group_requests("blocker")
    cache.release_all_group_requests("request")
    assert all(b.ref_cnt == 0 for b in cache._group_pools[name].blocks)


@pytest.mark.parametrize("dispatched", [False, True])
def test_cancel_keeps_read_pins_until_the_matching_step_is_discarded(dispatched):
    cache = grouped_cache()
    scheduler = scheduler_for(cache)
    request = Request("cancel", list(range(25000)), max_new_tokens=1)
    scheduler.add_request(request)
    first = scheduler.schedule()
    scheduler.advance_after_schedule(first)
    second = scheduler.schedule()
    step, = second.scheduled_requests
    assert step.group_read_snapshot is not None
    if dispatched:
        scheduler.advance_after_schedule(second)
    snapshot = step.group_read_snapshot
    pages = [b for _, blocks in snapshot._groups for b in blocks]
    scheduler.abort_request(request.request_id)
    assert all(b.ref_cnt > 0 for b in pages)
    scheduler.discard_scheduled_request(first.scheduled_requests[0])
    assert all(b.ref_cnt > 0 for b in pages)
    scheduler.discard_scheduled_request(step)
    scheduler.discard_scheduled_request(step)  # Idempotent terminal cleanup.
    assert all(b.ref_cnt == 0 for p in cache._group_pools.values() for b in p.blocks)


@pytest.mark.parametrize("name", STATE)
@pytest.mark.parametrize("length", [1, 7, 8, 31, 128, 8192])
def test_transfer_pin_outlives_request_and_cannot_alias_new_prefill(name, length):
    cache, buffer = state_cache(name)
    seed_history(cache, buffer, name, 8205)
    read, entry_pin = cache.retain_group_prefill_history("request", 8205)
    end = 8205 + length
    ids = cache.ensure_group_blocks("request", end)[name]
    execute_chunk(name, buffer, ids, read, start=8205, end=end, checkpoint=max(0, end - 128) // 128 * 128)
    page = STATE[name][3]
    for q in range(end - STATE[name][1], end):
        assert buffer[ids[q // page % len(ids)] * page + q % page].tolist() == row_value(q, name)
    transfer_pin = cache.retain_group_block_snapshot({name: ids}, 0)
    cache.release_group_block_snapshot(entry_pin)
    cache.release_all_group_requests("request")
    saved = buffer.clone()
    next_ids = cache.ensure_group_blocks("next", 8192, partition=0)[name]
    assert set(next_ids).isdisjoint(ids)
    execute_chunk(name, buffer, next_ids, {}, start=0, end=8192, checkpoint=8064)
    for physical in ids:
        lo = physical * STATE[name][3]
        torch.testing.assert_close(buffer[lo:lo + STATE[name][3]], saved[lo:lo + STATE[name][3]], equal_nan=True)
    cache.release_all_group_requests("next")
    # An UNKNOWN transfer retains this same pin; release only after stop ACK.
    assert all(b.ref_cnt > 0 for _, blocks in transfer_pin._groups for b in blocks)
    cache.release_group_block_snapshot(transfer_pin)
    assert all(b.ref_cnt == 0 for b in cache._group_pools[name].blocks)


@pytest.mark.parametrize("chunk", [8, 32, 128, 512, 1024, 2048, 8192])
@pytest.mark.parametrize("max_seq_len", [32, 512, 1048576])
@pytest.mark.parametrize("partitions,active,utilization", [(4, 1, .85), (4, 4, .9), (16, 5, .95)])
def test_prefill_geometry_and_budget_remain_safe_for_deployment_parameters(chunk, max_seq_len,
                                                                         partitions, active, utilization):
    groups = build_dspark_cache_group_specs(43, RATIOS, max_seq_len=max_seq_len,
        max_prefill_tokens=chunk, partitions=partitions, prefill_only=True)
    mixed = build_dspark_cache_group_specs(43, RATIOS, max_seq_len=max_seq_len,
        max_prefill_tokens=chunk, partitions=partitions)
    decode = build_dspark_cache_group_specs(43, RATIOS, max_seq_len=max_seq_len,
        max_prefill_tokens=chunk, partitions=partitions, decode_only=True)
    for g, ordinary, d in zip(groups, mixed, decode):
        if not g.is_checkpoint_state:
            assert g == ordinary == d
        else:
            page, history = g.spec.token_capacity, g.sliding_window
            span = min(max(8, min(chunk, max_seq_len)), 1024)
            assert g.max_blocks_per_seq == math.ceil((history - 1 + span) / page) + 1
            assert g.preserve_prefill_history and not ordinary.preserve_prefill_history
            assert not d.preserve_prefill_history
    budget = int((16 << 30) * utilization)
    counts = plan_cache_pages(groups, budget, active_requests=active, step_tokens=chunk, filler_pages=64)
    assert cache_pool_bytes(groups, counts, filler_pages=64) <= budget
    for g in groups:
        if g.preserve_prefill_history:
            page = g.spec.token_capacity
            per_request = 2 * g.max_blocks_per_seq + 2 * ((g.sliding_window + 2 * page - 2) // page) + g.sliding_window // page
            assert counts[g.name] == active * per_request
    # The original mixed geometry is not silently selected as a capacity fallback.
    minimum = plan_cache_pages(groups, budget, active_requests=1, step_tokens=chunk, filler_pages=64)
    base = {g.name: (max(g.max_blocks_per_seq, math.ceil(chunk / g.spec.token_capacity))
            if g.sliding_window is None else minimum[g.name]) for g in groups}
    required = cache_pool_bytes(groups, base, filler_pages=64)
    with pytest.raises(RuntimeError, match="minimum"):
        plan_cache_pages(groups, required - 1, active_requests=active, step_tokens=chunk, filler_pages=64)


def test_prefill_geometry_rejects_conflicting_roles():
    with pytest.raises(ValueError, match="both Prefill-only and Decode-only"):
        build_dspark_cache_group_specs(43, RATIOS, prefill_only=True, decode_only=True)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_publication_failure_still_releases_confirmed_step_read_pins(asynchronous, monkeypatch):
    cache = grouped_cache()
    scheduler = scheduler_for(cache, asynchronous=asynchronous)
    request = Request("failed-publish", list(range(25000)), max_new_tokens=1)
    scheduler.add_request(request)
    first = scheduler.schedule()
    if asynchronous:
        scheduler.advance_after_schedule(first)
    scheduler.update_from_output(first, {})
    second = scheduler.schedule()
    step, = second.scheduled_requests
    assert step.group_read_snapshot is not None
    if asynchronous:
        scheduler.advance_after_schedule(second)
    def fail(*a, **k):
        raise ValueError("publication failed")
    monkeypatch.setattr(cache, "cache_group_blocks_from_snapshot" if asynchronous else "cache_group_blocks", fail)
    with pytest.raises(ValueError, match="publication failed"):
        scheduler.update_from_output(second, {})
    assert step.group_read_snapshot is None and step.group_block_snapshot is None
    scheduler.abort_request(request.request_id)
    assert all(b.ref_cnt == 0 for p in cache._group_pools.values() for b in p.blocks)


def test_preempting_an_unsubmitted_step_releases_its_entry_view():
    from pypto_serving.serving.sched.scheduler import RequestStatus, ScheduledRequest, SchedulerOutput
    cache = grouped_cache()
    scheduler = scheduler_for(cache)
    initial = cache.ensure_group_blocks("victim", 8205, partition=0)
    victim = Request("victim", list(range(25000)), max_new_tokens=1, arrival_time=2,
        status=RequestStatus.RUNNING, num_computed_tokens=8205, allocated_group_block_ids=initial, cache_partition=0)
    scheduler.requests[victim.request_id] = victim
    scheduler.running.append(victim)
    step = ScheduledRequest(victim, 8192, True, num_computed_tokens=8205)
    assert scheduler._try_allocate_request_blocks(victim, 8192, scheduled=step)
    step.block_ids_by_group = victim.allocated_group_block_ids
    output = SchedulerOutput(scheduled_requests=[step], num_prefill_tokens=8192)
    exclude = Request("older", [1], max_new_tokens=1, arrival_time=1, cache_partition=0)
    assert scheduler._preempt_lowest_priority(exclude, {victim.request_id}, {victim.request_id: 8192}, output)
    assert not output.scheduled_requests and step.group_read_snapshot is None
    assert victim.status is RequestStatus.PREEMPTED
    assert all(b.ref_cnt == 0 for p in cache._group_pools.values() for b in p.blocks)


@pytest.mark.parametrize("failure", ["build", "encode"])
def test_pre_dispatch_host_failure_releases_new_read_pins(failure, monkeypatch):
    from pypto_serving.serving.engine import async_engine
    from pypto_serving.serving.sched.scheduler import RequestStatus
    from tests.unit.serving.engine.test_async_pipeline import _async_pipeline_core
    cache = grouped_cache()
    scheduler = scheduler_for(cache)
    ids = cache.ensure_group_blocks("request", 8205, partition=0)
    request = Request("request", list(range(25000)), max_new_tokens=1, status=RequestStatus.RUNNING,
                      num_computed_tokens=8205, allocated_group_block_ids=ids, cache_partition=0)
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    core, commands = _async_pipeline_core()
    core.scheduler, core.kv_cache_manager = scheduler, cache
    def fail(*a, **k):
        raise ValueError("host encoding failed")
    if failure == "build":
        monkeypatch.setattr(core, "_build_step_command", fail)
    else:
        monkeypatch.setattr(async_engine, "encode_command", fail)
    with pytest.raises(ValueError, match="host encoding failed"):
        core._try_dispatch_step()
    assert not commands and not core._batch_queue
    scheduler.abort_request(request.request_id)
    assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)


def test_entry_read_view_crosses_engine_codec_worker_and_packed_batch():
    from types import SimpleNamespace
    from pypto_serving.serving.sched.scheduler import RequestStatus
    from pypto_serving.serving.server.ipc import decode_command, encode_command
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    from tests.unit.serving.engine.test_async_pipeline import _async_pipeline_core
    from tests.unit.serving.device_sampling_fakes import _model
    cache = grouped_cache()
    scheduler = scheduler_for(cache)
    ids = cache.ensure_group_blocks("request", 8205, partition=0)
    request = Request("request", list(range(25000)), max_new_tokens=1, status=RequestStatus.RUNNING,
                      num_computed_tokens=8205, allocated_group_block_ids=ids, cache_partition=0)
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    output = scheduler.schedule()
    core, _ = _async_pipeline_core()
    core.scheduler, core.kv_cache_manager = scheduler, cache
    decoded = decode_command(encode_command(core._build_step_command(output, [], step_id=9)))
    step, = output.scheduled_requests
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._req_cache = {r.request_id: r for r in decoded.new_requests}
    worker.executor = SimpleNamespace(supports_device_embedding=True)
    worker._allow_device_sampled_ids = lambda *a, **k: False
    worker._allow_device_topk_sampling = lambda *a, **k: False
    worker._constraint_states = {}
    batch = worker._make_prefill_batch(decoded.prefill_requests, _model(1, max_seq_len=32768))
    assert batch.read_block_ids_by_group == [step.read_block_ids_by_group]
    assert batch.chunk_starts == [8205]
    assert batch.block_ids_by_group == [step.block_ids_by_group]
    scheduler.discard_scheduled_request(step)
    scheduler.abort_request(request.request_id)
    assert all(b.ref_cnt == 0 for pool in cache._group_pools.values() for b in pool.blocks)


@pytest.mark.parametrize("missing", ["ori", "hca_state", "csa_state", "csa_inner_state"])
def test_compact_checkpoint_loses_joint_validity_after_required_page_eviction(missing):
    cache = grouped_cache(active=1)
    scheduler = scheduler_for(cache, asynchronous=False)
    request = Request("checkpoint", list(range(8205)), max_new_tokens=1)
    scheduler.add_request(request)
    buffers = {name: torch.full((cache.group_num_blocks(name) * STATE[name][3], 4), float("nan"))
               for name in STATE}
    for _ in range(4):
        output = scheduler.schedule()
        step, = output.scheduled_requests
        start, end = step.num_computed_tokens, step.num_computed_tokens + step.num_new_tokens
        for name in STATE:
            execute_chunk(name, buffers[name], step.block_ids_by_group[name], step.read_block_ids_by_group,
                          start=start, end=end, checkpoint=request.group_checkpoint_boundary)
        scheduler.update_from_output(output, {request.request_id: [11]} if end == request.num_prompt_tokens else {})
        if request.prefill_complete:
            scheduler.finish_prefilled_request(request.request_id)
            break
    else:
        pytest.fail("Checkpoint Prefill did not finish")
    boundary = request.group_checkpoint_boundary
    pool = cache._group_pools[missing]
    page = pool.spec.spec.token_capacity
    key = 0, request.group_block_hashes[missing][boundary // page - 1]
    block = pool.hash_to_block[key]
    assert cache._group_checkpoints.checkpoints
    cache._invalidate_group_block(pool, 0, block)
    _, hit, _ = cache.acquire_group_prefix_blocks("query", request.group_block_hashes,
                                                 max_cache_hit_tokens=boundary)
    assert hit < boundary
    if hit:
        cache.release_all_group_requests("query")
    assert cache._group_pools["cmp_c128"].hash_to_block
    assert all(b.ref_cnt == 0 for p in cache._group_pools.values() for b in p.blocks)
