# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retained entry state survives compact Prefill output-ring rotation."""

import ast
import math
from functools import lru_cache

import pytest
import torch

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.serving.memory.kv_cache import KvCacheManager, aligned_prefix_cache_boundary
from tests.unit.model.deepseek_dspark.test_state_checkpoint import LIB, STATE, _CPU, publish


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


def state_cache(name, *, prefix=True):
    _, history, _, page = STATE[name]
    ring = math.ceil((history - 1 + 1024) / page) + 1
    entry = (history + 2 * page - 2) // page
    count = 2 * ring + 2 * entry + history // page
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


@pytest.mark.parametrize("name", STATE)
@pytest.mark.parametrize("start,length", [(0, 8192), (31, 33), (129, 1024), (513, 8192),
                                         (8205, 8192), (1048445, 131)])
@pytest.mark.parametrize("prefix", [False, True])
def test_real_readers_keep_entry_history_through_multiple_wraps(name, start, length, prefix):
    cache, buffer = state_cache(name, prefix=prefix)
    initial = cache.ensure_group_blocks("request", start)[name]
    if start:
        publish(name, buffer, torch.tensor(initial), start=max(0, start - 512), rows=min(start, 512), checkpoint=0)
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
def test_missing_history_is_rejected_before_any_pin_is_acquired(name):
    cache, _ = state_cache(name)
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
