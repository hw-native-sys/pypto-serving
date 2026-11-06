# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side contract for Qwen3.8-27B's two scheduler cache families."""

import pytest

from pypto_serving.model.qwen3_8_27b import build_qwen38_cache_group_specs
from pypto_serving.serving.memory.kv_cache import KvCacheManager


def test_qwen38_groups_match_kernel_pools():
    kv, state = build_qwen38_cache_group_specs(8192)

    assert kv.name == "attn_kv"
    assert kv.layer_indices == tuple(range(3, 64, 4))
    assert kv.spec.block_size == 128
    # 16 layers x K/V x 128 positions x 4 heads x 256 columns x BF16.
    assert kv.spec.page_size_bytes == 8_388_608
    assert kv.max_blocks_per_seq == 64

    assert state.name == "gdn_state"
    assert state.layer_indices == tuple(layer for layer in range(64) if layer % 4 != 3)
    assert state.spec.block_size == 8192
    assert state.max_blocks_per_seq == 1
    # 48 layers x ([3, 10240] BF16 convolution + [48, 128, 128] FP32 state).
    assert state.spec.page_size_bytes == 153_944_064
    assert state.num_partitions == kv.num_partitions == 1


@pytest.mark.parametrize("max_seq_len,kv_pages", [(1, 1), (128, 1), (129, 2), (8192, 64)])
def test_qwen38_groups_cover_context_with_one_state_slot(max_seq_len, kv_pages):
    kv, state = build_qwen38_cache_group_specs(max_seq_len)
    manager = KvCacheManager(enable_prefix_cache=True)
    manager.init_groups((kv, state), max_batch_size=2, primary_num_blocks=2 * kv_pages)

    assert kv.max_blocks_per_seq == kv_pages
    assert manager.required_group_block_counts(max_seq_len) == {
        "attn_kv": kv_pages,
        "gdn_state": 1,
    }
    if max_seq_len == 8192:
        assert manager.group_prefix_cache_alignment == 8192
        assert manager.published_group_block_counts(128) == {
            "attn_kv": 1,
            "gdn_state": 0,
        }
    first = manager.ensure_group_blocks("first", 1)
    first_state_id = first["gdn_state"][0]
    first = manager.ensure_group_blocks("first", max_seq_len)
    second = manager.ensure_group_blocks("second", max_seq_len)
    assert len(first["attn_kv"]) == len(second["attn_kv"]) == kv_pages
    assert first["gdn_state"] == [first_state_id]
    assert len(second["gdn_state"]) == 1
    assert second["gdn_state"] != first["gdn_state"]
    assert not manager.can_ensure_group_blocks("third", 1)


@pytest.mark.parametrize("invalid", [0, -1, 262145, 1.5, True])
def test_qwen38_groups_reject_invalid_context(invalid):
    with pytest.raises(ValueError, match="max_seq_len"):
        build_qwen38_cache_group_specs(invalid)
