# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check logical-page boundaries and shared producer storage for current V4.1 Lib."""

from types import SimpleNamespace

from pypto_serving.model.deepseek_v41.npu_runner import (
    SegmentTopology, _cache_source_shapes, plan_decode_cache_pools, prepare_compressed_metadata,
)
from pypto_serving.model.deepseek_v41.task_args import ForwardStep, RequestSlice, build_cache_groups
from pypto_serving.model.deepseek_v41.weight_loader import LayerPlan


def test_c2a_publication_crosses_a_logical_page_at_position_128():
    request = RequestSlice("request", 0, 0, 126, (10, 11, 12, 13), 130, {"c2a": (5, 9)})
    metadata = prepare_compressed_metadata(
        ForwardStep("prefill", (request,), 1), SegmentTopology.for_prefill(dp=4, max_tokens=80),
        ratio=2, compressed_group="c2a", cache_pages=10, max_requests=1, state_blocks=1,
    )
    assert metadata.compressed_slots[0, :4].tolist() == [-1, 383, -1, 576]
    assert metadata.compressed_rope_positions[0, :4].tolist() == [-1, 126, -1, 128]


def test_cache_pool_reserves_two_units_for_the_c1a_producer():
    layers = []
    for layer in range(40):
        if layer < 2:
            layers.append(LayerPlan(layer, "swa", None, None, None))
        elif layer < 20:
            source = max(value for value in (2, 8, 14) if value <= layer)
            mode = "c2a_full" if layer == source else "c2a_reuse"
            layers.append(LayerPlan(layer, mode, source, source, None))
        else:
            index = max(value for value in (20, 24, 28, 32, 36) if value <= layer)
            mode = "c1a_full" if layer == 20 else "c1a_reindex" if layer == index else "c1a_reuse"
            layers.append(LayerPlan(layer, mode, 20, index, 20))
    groups = build_cache_groups(tuple(layers), 384, dp_size=4)
    assert [group.max_blocks_per_seq for group in groups] == [3, 3, 3]
    assert [group.spec.storage_block_size for group in groups] == [128, 64, 128]
    abi = SimpleNamespace(
        EP_SIZE=4, N_LAYERS=40, KV_SOURCE_COUNT=4, C2A_SOURCE_COUNT=3, INDEX_SOURCE_COUNT=8,
        BLOCK_SIZE=128, HEAD_DIM=512, INDEX_DIM=128, STATE_CAPACITY=8, STATE_WIDTH=1024,
        INDEX_CACHE_GROUP=32, WINDOW_CACHE_BYTES=544, COMPRESSED_CACHE_BYTES=272,
        COMPRESSED_POOL_UNIT_ROWS=64,
    )
    plan = plan_decode_cache_pools(tuple(layers), groups, max_seq_len=384,
                                   primary_num_blocks=3, request_slots=1, abi=abi)
    pool = next(spec for spec in plan.buffers if spec.name == "compressed_cache_pool")
    assert pool.shape == (4, 20, 64, 1, 272)
    assert _cache_source_shapes(plan, pool) == (
        (4, 64, 1, 272), (4, 64, 1, 272), (4, 64, 1, 272), (4, 128, 1, 272),
    )
