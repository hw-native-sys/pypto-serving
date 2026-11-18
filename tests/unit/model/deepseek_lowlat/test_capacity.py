# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check capacity arithmetic against independent byte and context budgets."""

import pytest
import torch

from pypto_serving.model.deepseek_lowlat.capacity import LowlatCapacityPlan
from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile
from pypto_serving.model.deepseek_lowlat.speculation import validate_native_capacity


@pytest.mark.parametrize("context", [16384, 16385, 131072, 262144, 524288, 1048576])
def test_logical_compile_and_physical_capacities_are_separate(context):
    plan = LowlatCapacityPlan(context, 512, "rolling")
    assert context <= plan.compiled_context < context + 128
    groups = {g.name: g for g in plan.groups}
    assert groups["ori"].active_ring_blocks == 6
    assert groups["hca_state"].active_ring_blocks == 18
    assert groups["cmp_c4"].physical_pool_blocks == (context + 127) // 128
    assert groups["csa_state"].logical_table_blocks == (context + 3) // 4
    assert plan.rope_bytes_per_rank == plan.compiled_context * 512


def test_million_token_memory_ledger_and_prefix_reserve():
    base = LowlatCapacityPlan(1048576, 512, "rolling")
    assert base.cache_bytes_per_rank == 6580633600
    prefix = LowlatCapacityPlan(1048576, 512, "rolling", True)
    assert prefix.cache_bytes_per_rank == base.cache_bytes_per_rank + 49053696 + 17842176
    for group in prefix.groups:
        spec = next(s for s in prefix.cache_specs() if s.name == group.name)
        assert spec.max_blocks_per_seq == group.active_ring_blocks
        assert spec.num_blocks == group.physical_pool_blocks


@pytest.mark.parametrize(
    "context,chunk", [(True, 512), (0, 512), (1048577, 512), (16384, False), (16384, 8193)]
)
def test_invalid_sizes_fail_on_cpu(context, chunk):
    with pytest.raises(ValueError):
        LowlatCapacityPlan(context, chunk, "rolling")


def test_extended_full_layout_is_rejected_before_allocating():
    with pytest.raises(ValueError, match="rolling"):
        LowlatCapacityPlan(1048576)


@pytest.mark.parametrize("prompt,output", [(1000000, 64), (1048506, 64), (1046522, 2048)])
def test_native_logical_context_includes_full_lookahead(prompt, output):
    validate_native_capacity(prompt_tokens=prompt, max_output_tokens=output, physical_capacity=1048576)
    with pytest.raises(ValueError):
        validate_native_capacity(prompt_tokens=1048576, max_output_tokens=output, physical_capacity=1048576)


def test_actual_lib_import_binds_capacity_and_restores_globals():
    with isolated_profile(context_capacity=1048576):
        from models.deepseek_v4_flash_lowlat.config import FLASH
        from models.deepseek_v4_flash_lowlat import decode_shapes

        assert FLASH.max_position_embeddings == 1048576
        assert decode_shapes.KV_ORI_TABLE_MAX_BLOCKS == 8192
    with isolated_profile(context_capacity=16384):
        from models.deepseek_v4_flash_lowlat.config import FLASH

        assert FLASH.max_position_embeddings == 16384


def test_streamed_rope_matches_independent_absolute_position_oracle():
    with isolated_profile(context_capacity=131072):
        from models.deepseek_v4_flash_lowlat.utils import precompute_freqs_cos_sin

        cos, sin = precompute_freqs_cos_sin(64, 131072, 0, 10000.0, 1.0, 32, 1)
    positions = torch.tensor([0, 65535, 65536, 100000, 131071], dtype=torch.float32)
    freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = positions[:, None] * freq[None, :]
    assert torch.equal(cos[positions.long(), :32], angles.cos().bfloat16())
    assert torch.equal(sin[positions.long(), :32], angles.sin().bfloat16())
    assert torch.equal(cos[:, :32], cos[:, 32:])


@pytest.mark.parametrize("capacity", [16384, 131072, 1048576])
def test_actual_drafter_model_capacity_covers_each_target_rope_profile(capacity):
    from pypto_serving.model.deepseek_lowlat.kernel_loader import validate_shared_rope

    with isolated_profile(context_capacity=capacity):
        from models.deepseek_v4_flash_lowlat.config import FLASH as target
        from models.deepseek_v4_flash_lowlat.drafter.config import FLASH as draft

        validate_shared_rope(target, draft, capacity)
        assert draft.max_position_embeddings == 1048576
        assert target.max_position_embeddings == capacity
