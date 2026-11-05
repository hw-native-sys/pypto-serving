# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compressor checkpoints contain materialized state across tile boundaries."""

import ast
import math
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from pypto_serving.serving.memory.kv_cache import aligned_prefix_cache_boundary

STATE = {
    "hca_state": ("prefill_compressor_ratio128.py", 128, 136, 8),
    "csa_state": ("prefill_compressor_ratio4.py", 8, 8, 2),
    "csa_inner_state": ("prefill_indexer_compressor.py", 8, 8, 2),
}
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
@pytest.mark.parametrize("length", [31, 32, 33, 127, 128, 129, 511, 512, 513,
                                    8192, 8205, 8319, 8320, 1048445])
def test_checkpoint_survives_tiles_and_non_aligned_chunk_ends(name, length):
    _, history, _, page = STATE[name]
    ring = math.ceil((history - 1 + 1024) / page) + 1
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
