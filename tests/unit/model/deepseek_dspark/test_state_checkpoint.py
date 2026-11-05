# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute the actual compressor publication statements on poisoned CPU rings."""

import ast
import math
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from pypto_serving.config.types import RuntimeConfig
from pypto_serving.model.deepseek_dspark.cache_capacity import cache_pool_bytes, plan_cache_pages
from pypto_serving.model.deepseek_dspark.npu_runner import build_dspark_cache_group_specs
from pypto_serving.serving.memory.kv_cache import KvCacheManager, aligned_prefix_cache_boundary


STATE = {
    "hca_state": ("prefill_compressor_ratio128.py", 128, 136, 8),
    "csa_state": ("prefill_compressor_ratio4.py", 8, 8, 2),
    "csa_inner_state": ("prefill_indexer_compressor.py", 8, 8, 2),
}
RATIOS = (0, 0) + (4,) * 21 + (128,) * 20
LIB = Path(__file__).resolve().parents[4] / "pypto-lib/models/deepseek_v4_flash_dspark"


class _CPU:
    min = staticmethod(min)
    max = staticmethod(max)
    range = staticmethod(range)
    add = staticmethod(torch.add)
    INDEX = int

    @staticmethod
    def read(tensor, indices):
        return int(tensor[tuple(indices)])

    @staticmethod
    def cast(value, _type):
        return int(value)


class _SpmdOnCPU(ast.NodeTransformer):
    def visit_With(self, node):
        call = node.items[0].context_expr
        assert isinstance(call, ast.Call) and call.func.attr == "spmd"
        loop = ast.For(ast.Name("_lane", ast.Store()),
                       ast.Call(ast.Name("range", ast.Load()), [call.args[0]], []), node.body, [])
        return self.generic_visit(loop)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get_block_idx":
            return ast.Name("_lane", ast.Load())
        return self.generic_visit(node)


@lru_cache(None)
def _publication_program(name):
    filename, *_ = STATE[name]
    tree = ast.parse((LIB / filename).read_text())
    tile = next(f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name.endswith("_tile"))
    start_name = "tail_rows" if name == "hca_state" else "publish_rows"
    start = next(i for i, n in enumerate(tile.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == start_name)
    end = next(i for i in range(start, len(tile.body)) if isinstance(tile.body[i], ast.With))
    # Only lower the CPU execution construct; retain the production indices,
    # guards, projections, APE and tensor stores without a second implementation.
    program = ast.Module(tile.body[start:end + 1], type_ignores=[])
    program = ast.fix_missing_locations(_SpmdOnCPU().visit(program))
    return compile(program, str(LIB / filename), "exec")


def publish(name, buffer, table, *, start, rows, checkpoint, packed_offset=0):
    _, history, tail, page = STATE[name]
    positions = torch.arange(start, start + rows, dtype=torch.int32)
    mapping = table[positions.long() // page % len(table)] * page + positions % page
    projection = torch.stack((positions.float(), positions.float() + 1), dim=1)
    score = projection + 100
    ratio = 128 if name == "hca_state" else 4
    ape = torch.arange(ratio).float().unsqueeze(1).expand(-1, 2)
    positions = torch.cat((torch.zeros(packed_offset, dtype=torch.int32), positions))
    mapping = torch.cat((torch.full((packed_offset,), -1, dtype=torch.int64), mapping))
    env = dict(pl=_CPU, tile_base=packed_offset, tile_rows=rows, checkpoint_end=checkpoint,
               STATE_LEN=history, STATE_STORAGE_LEN=tail, OUT_DIM=2, OUT_TILE=2,
               COMPRESS_STATE_DIM=4, COMPRESS_RATIO=ratio, position_ids=positions,
               state_slot_mapping=mapping, inner_state_slot_mapping=mapping,
               compress_state_flat=buffer, inner_compress_state_flat=buffer,
               kv_proj_scratch=projection, score_proj_scratch=score, ape=ape)
    exec(_publication_program(name), env)


def assert_rows(name, buffer, table, positions):
    page = STATE[name][3]
    ratio = 128 if name == "hca_state" else 4
    for p in positions:
        row = int(table[p // page % len(table)]) * page + p % page
        assert buffer[row].tolist() == [p, p + 1, p + 100 + p % ratio, p + 101 + p % ratio]


@pytest.mark.parametrize("name", STATE)
@pytest.mark.parametrize("replay", [0, 128])
def test_publication_covers_checkpoint_at_every_tile_phase(name, replay):
    page = STATE[name][3]
    history, tail = STATE[name][1:3]
    ring = math.ceil((history - 1 + 8192) / page) + 1
    table = torch.arange(ring - 1, -1, -1)
    for phase in range(128):
        start = 7680 + phase
        end = start + 512
        checkpoint = aligned_prefix_cache_boundary(end, 128, replay_tokens=replay)
        buffer = torch.full((ring * page, 4), float("nan"))
        publish(name, buffer, table, start=start, rows=512, checkpoint=checkpoint)
        expected = set(range(end - tail, end)) | set(range(max(start, checkpoint - history), checkpoint))
        assert_rows(name, buffer, table, expected)
        # Neither preallocated pages nor gaps become valid by allocation alone.
        written = torch.isfinite(buffer[:, 0]).sum().item()
        assert written == len(expected)
        assert written <= tail + history


@pytest.mark.parametrize("name", STATE)
@pytest.mark.parametrize("length", [31, 32, 33, 127, 128, 129, 511, 512, 513,
                                    8192, 8205, 8319, 8320, 1048445])
def test_checkpoint_survives_tiles_and_non_aligned_chunk_ends(name, length):
    _, history, _, page = STATE[name]
    ring = math.ceil((history - 1 + 8192) / page) + 1
    table = torch.roll(torch.arange(ring), 17)
    buffer = torch.full((ring * page, 4), float("nan"))
    checkpoint = aligned_prefix_cache_boundary(length, 128, replay_tokens=128)
    # Bound the test to the final two chunks even for near-1M logical positions.
    start = max(0, (length // 8192 - 1) * 8192)
    for chunk in range(start, length, 8192):
        for tile in range(chunk, min(chunk + 8192, length), 512):
            publish(name, buffer, table, start=tile,
                    rows=min(512, length - tile), checkpoint=checkpoint)
    if checkpoint:
        assert_rows(name, buffer, table, range(max(0, checkpoint - history), checkpoint))
    assert_rows(name, buffer, table, range(max(start, length - history), length))


@pytest.mark.parametrize("name", STATE)
def test_no_checkpoint_keeps_only_the_existing_terminal_tail(name):
    _, history, tail, page = STATE[name]
    ring = math.ceil((history - 1 + 8192) / page) + 1
    table = torch.arange(ring)
    buffer = torch.full((ring * page, 4), float("nan"))
    publish(name, buffer, table, start=8205, rows=512, checkpoint=0)
    assert torch.isfinite(buffer[:, 0]).sum().item() == tail
    assert_rows(name, buffer, table, range(8717 - tail, 8717))


@pytest.mark.parametrize("name", STATE)
@pytest.mark.parametrize("length", [8205, 8320, 8333, 1048445])
def test_checkpoint_spanning_ragged_chunks_and_packed_offsets(name, length):
    _, history, _, page = STATE[name]
    ring = math.ceil((history - 1 + 8192) / page) + 1
    table = torch.roll(torch.arange(ring), 3)
    buffer = torch.full((ring * page, 4), float("nan"))
    checkpoint = aligned_prefix_cache_boundary(length, 128, replay_tokens=128)
    start = max(0, length - 8192)
    for end in (length - 600, length - 207, length):
        for tile in range(start, end, 512):
            publish(name, buffer, table, start=tile, rows=min(512, end - tile),
                    checkpoint=checkpoint, packed_offset=31)
        start = end
    assert_rows(name, buffer, table, range(checkpoint - history, checkpoint))


@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("layout", ["wrapped", "fragmented"])
@pytest.mark.parametrize("length", [1, 33, 129, 8192, 8205, 8319, 8320, 1048445])
def test_actual_published_state_reaches_compact_decode_history(d_tp, layout, length):
    from pypto_serving.pd.planner import plan_chunk
    from tests.unit.pd.test_plan_ranges import plan_case

    case = plan_case(8192, destination_threshold=2048, destination_state_tokens=144,
                     length=length, d_tp=d_tp, layout=layout)
    plan = plan_chunk(*case.args)
    case.service.validate_plan(case.entry, plan, case.geometry, 0, case.source.topology)
    geometry = {g.name: g for g in case.geometry}
    destination = {g.name: g for g in case.service.geometry}
    checkpoint = aligned_prefix_cache_boundary(length, 128, replay_tokens=128)
    source, received = {}, {}
    for name, (_, history, _, page) in STATE.items():
        source[name] = torch.full((geometry[name].blocks * page, 4), float("nan"))
        table = torch.tensor(case.chunk.blocks[name])
        for tile in range(max(0, length - 8192), length, 512):
            publish(name, source[name], table, start=tile, rows=min(512, length - tile),
                    checkpoint=checkpoint, packed_offset=7)
    # Probe four FP32 fields per logical row; the wire plan still uses the
    # production row widths. Whole-page padding may remain poisoned.
    for copy in plan.copies:
        name = copy.component
        if name not in STATE:
            continue
        page = STATE[name][3]
        target = received.setdefault((copy.destination_rank, name),
            torch.full((destination[name].blocks * page, 4), float("nan")))
        for offset in range(0, copy.length, geometry[name].page_bytes):
            sl, sp = divmod((copy.source_offset + offset) // geometry[name].page_bytes,
                            geometry[name].blocks)
            dl, dp = divmod((copy.destination_offset + offset) // destination[name].page_bytes,
                            destination[name].blocks)
            assert sl == dl
            target[dp * page:(dp + 1) * page] = source[name][sp * page:(sp + 1) * page]
    for (_, name), buffer in received.items():
        assert_rows(name, buffer, torch.tensor(case.placement.blocks[name]),
                    range(max(0, length - STATE[name][1]), length))


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("length", [8192, 8205, 8319, 8320])
def test_prefix_index_publishes_only_materialized_checkpoint_state(snapshot, length):
    groups = build_dspark_cache_group_specs(3, (0, 4, 128), max_seq_len=16384, max_prefill_tokens=8192,
                                           partitions=1)
    manager = KvCacheManager()
    manager.initialize(RuntimeConfig(max_batch_size=2, max_seq_len=16384, kv_cache_groups=groups),
                       num_blocks=2 * groups[0].max_blocks_per_seq)
    tokens = list(range(length))
    hashes = manager.compute_group_block_hashes(tokens)
    checkpoint = manager.group_prefix_cache_boundary(length, replay_tokens=128)
    blocks = manager.ensure_group_blocks("source", length)
    if snapshot:
        handle = manager.retain_group_block_snapshot(blocks, 0)
        manager.cache_group_blocks_from_snapshot(hashes, length, {}, blocks, 0, request_id="source",
                                                checkpoint_boundary=checkpoint)
        manager.release_group_block_snapshot(handle)
    else:
        manager.cache_group_blocks("source", hashes, length, {}, checkpoint_boundary=checkpoint)
    for group in groups:
        if not group.is_checkpoint_state:
            continue
        page = group.spec.token_capacity
        actual = {index for index, h in enumerate(hashes[group.name])
                  if (0, h) in manager._group_pools[group.name].hash_to_block}
        assert actual == set(range((checkpoint - group.sliding_window) // page, checkpoint // page))
    manager.release_all_group_requests("source")
    _, hit, _ = manager.acquire_group_prefix_blocks("replay", hashes, max_cache_hit_tokens=checkpoint)
    assert hit == checkpoint
    manager.release_all_group_requests("replay")
    # Lose one compulsory CSA state page while keeping KV: no false full hit.
    pool = manager._group_pools["csa_state"]
    member = pool.hash_to_block[(0, hashes["csa_state"][checkpoint // 2 - 1])]
    manager._invalidate_group_block(pool, 0, member)
    _, hit, _ = manager.acquire_group_prefix_blocks("missing", hashes, max_cache_hit_tokens=checkpoint)
    assert hit < checkpoint
    manager.release_all_group_requests("missing")
    assert all(b.ref_cnt == 0 for pool in manager._group_pools.values() for b in pool.blocks)


@pytest.mark.parametrize("tokens", [128, 2048, 8192])
@pytest.mark.parametrize("partitions", [4, 16])
def test_decode_state_rings_are_independent_of_prefill_budget(tokens, partitions):
    ordinary = build_dspark_cache_group_specs(43, RATIOS, max_prefill_tokens=tokens, partitions=partitions)
    decode = build_dspark_cache_group_specs(43, RATIOS, max_prefill_tokens=tokens, partitions=partitions,
                                          decode_only=True)
    assert {g.name: g.max_blocks_per_seq for g in decode if g.name in STATE} == {
        "hca_state": 35, "csa_state": 77, "csa_inner_state": 77}
    for a, b in zip(ordinary, decode):
        if a.name not in STATE:
            assert a == b
    assert sum(g.spec.page_size_bytes * g.max_blocks_per_seq for g in decode if g.name in STATE) == 56053760


@pytest.mark.parametrize("snapshot", [False, True])
def test_wrapped_snapshot_cannot_publish_expired_checkpoint_state(snapshot):
    groups = build_dspark_cache_group_specs(3, (0, 4, 128), max_seq_len=16384,
                                           max_prefill_tokens=2048, decode_only=True, partitions=1)
    manager = KvCacheManager()
    manager.initialize(RuntimeConfig(max_batch_size=3, max_seq_len=16384, kv_cache_groups=groups),
                       num_blocks=3 * groups[0].max_blocks_per_seq)
    hashes = manager.compute_group_block_hashes(list(range(8192)))
    manager.ensure_group_blocks("source", 8192)
    blocks = manager.ensure_group_blocks("source", 8704)
    if snapshot:
        handle = manager.retain_group_block_snapshot(blocks, 0)
        manager.cache_group_blocks_from_snapshot(hashes, 8704, {}, blocks, 0, request_id="source",
                                                checkpoint_boundary=8064)
        manager.release_group_block_snapshot(handle)
    else:
        manager.cache_group_blocks("source", hashes, 8704, {}, checkpoint_boundary=8064)
    assert all(not manager._group_pools[name].hash_to_block for name in STATE)
    manager.release_all_group_requests("source")
    assert all(b.ref_cnt == 0 for pool in manager._group_pools.values() for b in pool.blocks)


def test_decode_pool_accounts_for_unaligned_writes_and_filler():
    groups = build_dspark_cache_group_specs(43, RATIOS, decode_only=True, max_prefill_tokens=2048)
    work = {g.name: max((s + 8 + g.spec.token_capacity - 1) // g.spec.token_capacity
                       - s // g.spec.token_capacity for s in range(g.spec.token_capacity))
            if g.is_checkpoint_state else math.ceil(8 / g.spec.token_capacity)
            for g in groups if g.sliding_window}
    counts = {g.name: g.max_blocks_per_seq if g.sliding_window is None else
              5 * (g.max_blocks_per_seq + work[g.name] + g.sliding_window // g.spec.token_capacity)
              for g in groups}
    budget = cache_pool_bytes(groups, counts, filler_pages=64)
    planned = plan_cache_pages(groups, budget, active_requests=5, step_tokens=8, filler_pages=64)
    assert cache_pool_bytes(groups, planned, filler_pages=64) <= budget
    for g in groups:
        if g.sliding_window:
            assert planned[g.name] == counts[g.name]
    assert work["hca_state"] == 2
    assert work["csa_state"] == work["csa_inner_state"] == 5
