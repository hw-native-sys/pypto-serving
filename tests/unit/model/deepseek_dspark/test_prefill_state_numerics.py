# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU golden equivalence for aliased large rings and distinct compact views."""

import ast
import math

import pytest
import torch

from tests.unit.model.deepseek_dspark.test_state_checkpoint import LIB, STATE


def fixture(name, start, length, *, separate):
    _, history, _, page = STATE[name]
    ratio = 128 if name == "hca_state" else 4
    out = 2 if name == "hca_state" else 4
    ring = math.ceil((history - 1 + (1024 if separate else length)) / page) + 1
    end = start + length
    logical = torch.arange((end + page - 1) // page + 1)
    write_table = logical % ring
    read_table = write_table.clone()
    required = list(range(max(0, start - history) // page, (start + page - 1) // page))
    extra = len(required) if separate else 0
    if separate:
        read_table.fill_(-1)
        for i, q in enumerate(required):
            read_table[q] = ring + i
    state = torch.full(((ring + extra) * page, out * 2), float("nan"))
    for q in range(max(0, start - history), start):
        row = int(read_table[q // page]) * page + q % page
        state[row] = torch.arange(out * 2).float() * .01 + (q % 17) * .03
    positions = torch.arange(start, end, dtype=torch.int32)
    write_map = write_table[positions.long() // page] * page + positions % page
    compressed_rows = (end + ratio - 1) // ratio
    generator = torch.Generator().manual_seed(13)
    tensors = dict(x=(torch.rand(length, 2, generator=generator) * .2 - .1).to(torch.bfloat16),
        query_start_loc=torch.tensor([0, length], dtype=torch.int32),
        state_checkpoint_ends=torch.tensor([max(0, end - 128) // 128 * 128], dtype=torch.int32),
        compress_state=state.reshape(ring + extra, page, out * 2),
        wkv=torch.randn(out, 2, generator=generator).to(torch.bfloat16) * .05,
        wgate=torch.randn(out, 2, generator=generator).to(torch.bfloat16) * .03,
        ape=torch.randn(ratio, out, generator=generator) * .05,
        norm_w=torch.tensor([.9, 1.1], dtype=torch.bfloat16),
        cmp_freqs_cos=torch.ones(length, 2, dtype=torch.bfloat16),
        cmp_freqs_sin=torch.zeros(length, 2, dtype=torch.bfloat16), position_ids=positions)
    slots = torch.where((positions + 1) % ratio == 0, positions.long() // ratio, -1)
    if name == "csa_inner_state":
        tensors.update(inner_compress_state_block_table=read_table.unsqueeze(0).to(torch.int32),
            inner_state_slot_mapping=write_map, idx_slot_mapping=slots,
            idx_kv_cache=torch.zeros(compressed_rows, 1, 1, 2, dtype=torch.int8),
            idx_kv_scale=torch.zeros(compressed_rows, 1, 1, 1), hadamard=torch.eye(2, dtype=torch.bfloat16))
    else:
        tensors.update(compress_state_block_table=read_table.unsqueeze(0).to(torch.int32),
            state_slot_mapping=write_map, cmp_slot_mapping=slots,
            cmp_kv=torch.zeros(compressed_rows, 1, 1, 2, dtype=torch.bfloat16))
    constants = dict(CMP_MAX_BLOCKS=compressed_rows, CMP_STORAGE_BLOCK_SIZE=1, BLOCK_SIZE=1,
        COMPRESS_RATIO=ratio, COMPRESS_STATE_DIM=out * 2, D=2, HEAD_DIM=2, OUT_DIM=out,
        HCA_STATE_BLOCK_NUM=ring + extra, HCA_STATE_BLOCK_SIZE=page, CSA_STATE_BLOCK_SIZE=page,
        INNER_STATE_BLOCK_SIZE=page, STATE_LEN=history, STATE_STORAGE_LEN=136, MAX_SEQ_LEN=1048576,
        PREFILL_STATE_TILE=512, NOPE_HEAD_DIM=0, ROPE_HEAD_DIM=2, ROPE_HALF=1,
        EPS=1e-6, INT8_AMAX_EPS=1e-10, INT8_SCALE_MAX=127.)
    return tensors, constants, write_table, required


def run_golden(name, tensors, constants):
    tree = ast.parse((LIB / STATE[name][0]).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("golden_"))
    env = dict(constants)
    exec(compile(ast.fix_missing_locations(ast.Module([fn], type_ignores=[])), str(LIB / STATE[name][0]), "exec"), env)
    env[fn.name](tensors)


@pytest.mark.parametrize("name", STATE)
@pytest.mark.parametrize("start,length", [(13, 513), (127, 2049)])
def test_compact_distinct_views_preserve_compressed_outputs_and_final_state(name, start, length):
    baseline, bconstants, btable, _ = fixture(name, start, length, separate=False)
    compact, constants, table, required = fixture(name, start, length, separate=True)
    page = STATE[name][3]
    entry_table = compact["inner_compress_state_block_table" if name == "csa_inner_state"
                          else "compress_state_block_table"][0]
    saved = compact["compress_state"].clone()
    run_golden(name, baseline, bconstants)
    run_golden(name, compact, constants)
    for output in (("idx_kv_cache", "idx_kv_scale") if name == "csa_inner_state" else ("cmp_kv",)):
        torch.testing.assert_close(compact[output], baseline[output], rtol=0, atol=0)
    boundary = int(compact["state_checkpoint_ends"][0])
    positions = set(range(start + length - STATE[name][1], start + length))
    positions |= set(range(max(start, boundary - STATE[name][1]), boundary))
    cstate = compact["compress_state"].flatten(0, 1)
    bstate = baseline["compress_state"].flatten(0, 1)
    for q in positions:
        row = int(table[q // page]) * page + q % page
        brow = int(btable[q // page]) * page + q % page
        torch.testing.assert_close(cstate[row], bstate[brow], rtol=0, atol=0)
        assert torch.isfinite(cstate[row]).all()
    for q in required:
        physical = int(entry_table[q])
        torch.testing.assert_close(compact["compress_state"][physical], saved[physical], equal_nan=True)
