# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for the V4.1 TP1/DP4/EP4 serving placement."""

from types import SimpleNamespace

from pypto_serving.model.deepseek_v41.npu_runner import (
    SegmentTopology,
    plan_decode_cache_pools,
    selected_decode_topology,
)
from pypto_serving.model.deepseek_v41.task_args import build_cache_groups
from pypto_serving.model.deepseek_v41.weight_loader import LayerPlan, RankPlacement
from pypto_serving.model.deepseek_v41.weight_loader import DecodeWeightLayout


def _layers():
    kv_sources = (2, 8, 14, 20)
    index_sources = (2, 4, 6, 8, 10, 12, 14, 20)
    return tuple(
        LayerPlan(
            layer,
            "swa" if layer < 2 else
            "c1a_full" if layer == 20 else
            "c2a_full" if layer in kv_sources else
            "c2a_reuse" if layer < 20 else "c1a_reuse",
            max((source for source in kv_sources if source <= layer), default=None),
            max((source for source in index_sources if source <= layer), default=None),
            20 if layer >= 20 else None,
        )
        for layer in range(40)
    )


def test_tp1_dp4_topology_and_cache_groups():
    topology = SegmentTopology.for_decode(dp=4)
    assert (topology.tp, topology.dp, topology.world) == (1, 4, 4)
    assert topology.capacity == 80
    assert [RankPlacement(rank, tp_size=1, dp_size=4, ep_size=4).dp_rank
            for rank in range(4)] == [0, 1, 2, 3]
    groups = build_cache_groups(_layers(), 8192, dp_size=4)
    assert {group.num_partitions for group in groups} == {4}


def test_tp1_dp4_cache_pool_uses_four_device_axis():
    layers = _layers()
    groups = build_cache_groups(layers, 8192, dp_size=4)
    abi = SimpleNamespace(
        TP_SIZE=1, EP_SIZE=4, N_LAYERS=40, KV_SOURCE_COUNT=4,
        C2A_SOURCE_COUNT=3, INDEX_SOURCE_COUNT=8, BLOCK_SIZE=128,
        HEAD_DIM=512, INDEX_DIM=128, STATE_CAPACITY=2, STATE_WIDTH=1024,
        WINDOW_CACHE_GROUP=32, COMPRESSED_CACHE_GROUP=16, INDEX_CACHE_GROUP=32,
    )
    plan = plan_decode_cache_pools(
        layers, groups, max_seq_len=8192,
        primary_num_blocks=groups[0].max_blocks_per_seq,
        request_slots=1, abi=abi,
    )
    assert all(spec.shape[0] == 4 for spec in plan.buffers)
    assert plan.physical_request_slots == 2


def test_tp1_dp4_cache_pool_accepts_decode_abi_without_tp_constant():
    """The V4.1 decode module omits TP_SIZE because DSpark is TP1."""
    layers = _layers()
    groups = build_cache_groups(layers, 8192, dp_size=4)
    abi = SimpleNamespace(
        EP_SIZE=4, N_LAYERS=40, KV_SOURCE_COUNT=4,
        C2A_SOURCE_COUNT=3, INDEX_SOURCE_COUNT=8, BLOCK_SIZE=128,
        HEAD_DIM=512, INDEX_DIM=128, STATE_CAPACITY=2, STATE_WIDTH=1024,
        WINDOW_CACHE_GROUP=32, COMPRESSED_CACHE_GROUP=16, INDEX_CACHE_GROUP=32,
    )
    plan = plan_decode_cache_pools(
        layers, groups, max_seq_len=8192,
        primary_num_blocks=groups[0].max_blocks_per_seq,
        request_slots=1, abi=abi,
    )
    assert plan.physical_request_slots == 2


def test_decode_weight_layout_accepts_decode_abi_without_tp_constant():
    abi = SimpleNamespace(
        N_LAYERS=40, KV_SOURCE_COUNT=4, INDEX_SOURCE_COUNT=8,
        C2A_SOURCE_COUNT=3, EP_SIZE=4,
    )
    layout = DecodeWeightLayout.from_lib(abi)
    assert (layout.world, layout.tp_size) == (4, 1)


def test_tp1_dp4_topology_reads_current_lib_config(tmp_path):
    config = tmp_path / "models" / "deepseek_v4_1_flash" / "config.py"
    config.parent.mkdir(parents=True)
    config.write_text(
        "MAX_BATCH_PER_DP = 12\nDSPARK_SPEC_TOKENS = 5\n"
        "DECODE_ROWS_PER_REQUEST = DSPARK_SPEC_TOKENS + 1\n", encoding="utf-8",
    )
    assert selected_decode_topology(tmp_path, dp_size=4).world == 4
