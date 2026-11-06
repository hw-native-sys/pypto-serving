# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scheduler cache groups for Qwen3.8-27B's attention and Gated DeltaNet layers.

The layer sets and widths match ``pypto-lib/models/qwen3_8_27b/config.py``.
The kernels expose separate K/V and convolution/recurrent tensors; each pair
uses the same physical block IDs within its group. One GDN block ID selects a
48-row sequence range in both state tensors, while one attention block ID
selects a page in K/V for every attention layer.
"""

from __future__ import annotations

import math

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec


QWEN38_NUM_LAYERS = 64
QWEN38_FULL_ATTENTION_INTERVAL = 4
QWEN38_MAX_POSITION_EMBEDDINGS = 262144
QWEN38_KV_PAGE_TOKENS = 128
QWEN38_KV_HEADS = 4
QWEN38_KV_HEAD_DIM = 256
QWEN38_LINEAR_KEY_HEADS = 16
QWEN38_LINEAR_VALUE_HEADS = 48
QWEN38_LINEAR_HEAD_DIM = 128
QWEN38_CONV_KERNEL = 4

QWEN38_ATTENTION_LAYERS = tuple(
    range(QWEN38_FULL_ATTENTION_INTERVAL - 1, QWEN38_NUM_LAYERS, QWEN38_FULL_ATTENTION_INTERVAL)
)
QWEN38_GDN_LAYERS = tuple(
    layer for layer in range(QWEN38_NUM_LAYERS) if layer not in QWEN38_ATTENTION_LAYERS
)

# One physical attention page contains K and V for all attention layers.
QWEN38_KV_PAGE_BYTES = (
    len(QWEN38_ATTENTION_LAYERS) * 2 * QWEN38_KV_PAGE_TOKENS
    * QWEN38_KV_HEADS * QWEN38_KV_HEAD_DIM * 2
)

# Each GDN sequence owns one row in each of two physical tensors. The
# convolution window is BF16; the recurrent [H, V, K] state is FP32.
QWEN38_CONV_ROW_BYTES = (
    (QWEN38_CONV_KERNEL - 1)
    * (2 * QWEN38_LINEAR_KEY_HEADS + QWEN38_LINEAR_VALUE_HEADS)
    * QWEN38_LINEAR_HEAD_DIM * 2
)
QWEN38_RECURRENT_ROW_BYTES = (
    QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_HEAD_DIM**2 * 4
)
QWEN38_GDN_PAGE_BYTES = len(QWEN38_GDN_LAYERS) * (
    QWEN38_CONV_ROW_BYTES + QWEN38_RECURRENT_ROW_BYTES
)


def build_qwen38_cache_group_specs(max_seq_len: int) -> tuple[KVCacheGroupSpec, KVCacheGroupSpec]:
    """Declare independent attention KV and per-sequence GDN state pools.

    The attention group reports the primary page capacity. The state group
    needs one block per live sequence, covering the full configured context;
    mutable state must not be shared as a partial prefix-cache page.
    """
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
        raise ValueError("max_seq_len must be a non-boolean integer")
    if not 0 < max_seq_len <= QWEN38_MAX_POSITION_EMBEDDINGS:
        raise ValueError(
            f"max_seq_len must be in [1, {QWEN38_MAX_POSITION_EMBEDDINGS}]"
        )

    return (
        KVCacheGroupSpec(
            name="attn_kv",
            layer_indices=QWEN38_ATTENTION_LAYERS,
            spec=KVCacheSpec(
                block_size=QWEN38_KV_PAGE_TOKENS,
                page_size_bytes=QWEN38_KV_PAGE_BYTES,
            ),
            max_blocks_per_seq=math.ceil(max_seq_len / QWEN38_KV_PAGE_TOKENS),
        ),
        KVCacheGroupSpec(
            name="gdn_state",
            layer_indices=QWEN38_GDN_LAYERS,
            spec=KVCacheSpec(block_size=max_seq_len, page_size_bytes=QWEN38_GDN_PAGE_BYTES),
            max_blocks_per_seq=1,
        ),
    )
