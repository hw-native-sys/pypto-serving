# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scheduler cache-family contract for the complete V4.1 text backbone.

The compressed pools have different source-token capacities per physical page:
C2A stores one row per two tokens, while C1A stores one row per token. Their
page IDs may be jointly lowered with index keys inside each family, but cannot
share one scheduler group with a single block-size declaration.
"""

from dataclasses import dataclass

from pypto_serving.config.types import KVCacheGroupSpec

from .execution_plan import LayerPlan


@dataclass(frozen=True)
class V41CacheGroups:
    window: str
    c2a: str
    c1a: str


def validate_cache_groups(
    layers: tuple[LayerPlan, ...], groups: tuple[KVCacheGroupSpec, ...], max_seq_len: int,
) -> V41CacheGroups:
    """Resolve the three full-history page families before device allocation."""
    if len(layers) != 40 or tuple(layer.layer_id for layer in layers) != tuple(range(40)):
        raise ValueError("V4.1 cache contract requires the complete ordered 40-layer backbone")
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("V4.1 cache contract requires a positive sequence capacity")
    if (len(groups) != 3 or any(not isinstance(group, KVCacheGroupSpec) for group in groups)
            or len({group.name for group in groups}) != 3):
        raise ValueError("V4.1 requires separate window, C2A and C1A cache groups")

    expected = {
        "window": (tuple(range(40)), 128, 1),
        "c2a": (tuple(layer.layer_id for layer in layers
                      if layer.mode == "c2a_full" and layer.kv_source == layer.layer_id), 256, 2),
        "c1a": (tuple(layer.layer_id for layer in layers
                      if layer.mode == "c1a_full" and layer.kv_source == layer.layer_id), 128, 1),
    }
    if not expected["c2a"][0] or not expected["c1a"][0]:
        raise ValueError("V4.1 cache contract requires C2A and C1A KV producers")
    resolved = {}
    for group in groups:
        matches = [family for family, (owners, _, _) in expected.items()
                   if tuple(group.layer_indices) == owners]
        if len(matches) != 1 or matches[0] in resolved:
            raise ValueError("V4.1 cache groups must match window and KV producer layers")
        family = matches[0]
        _, block_size, ratio = expected[family]
        if (group.spec.block_size != block_size or group.spec.compress_ratio != ratio
                or group.spec.storage_block_size != 128 or group.num_partitions != 2
                or group.sliding_window is not None or group.is_eagle_group):
            raise ValueError(f"V4.1 {family} cache page layout disagrees with the lib ABI")
        required = (max_seq_len + block_size - 1) // block_size
        if (group.max_blocks_per_seq < required
                or (group.num_blocks is not None and group.num_blocks < required)):
            raise ValueError(f"V4.1 {family} cache cannot cover max_seq_len")
        resolved[family] = group.name
    return V41CacheGroups(**resolved)
