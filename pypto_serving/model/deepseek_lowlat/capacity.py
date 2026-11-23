# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure CPU capacity planning shared by admission, compilation and allocation."""

from dataclasses import dataclass
from math import ceil

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec

MODEL_CONTEXT = 1_048_576
MAX_PREFILL = 8192
REPLAY_TOKENS = 128

# group: (source tokens/page, bytes/page/rank, layer IDs, compression)
CACHE_GROUPS = {
    "ori": (128, 5636096, tuple(range(43)), 1),
    "cmp_c128": (128, 20480, tuple(range(3, 43, 2)), 128),
    "cmp_c4": (128, 688128, tuple(range(2, 43, 2)), 4),
    "idx": (128, 88704, tuple(range(2, 43, 2)), 4),
    "hca_state": (8, 655360, tuple(range(3, 43, 2)), 1),
    "csa_state": (4, 688128, tuple(range(2, 43, 2)), 1),
    "csa_inner_state": (4, 172032, tuple(range(2, 43, 2)), 1),
}


@dataclass(frozen=True)
class LowlatGroupCapacity:
    name: str
    source_tokens_per_page: int
    bytes_per_page: int
    logical_table_blocks: int
    active_ring_blocks: int
    physical_pool_blocks: int
    sliding_window: int | None


@dataclass(frozen=True)
class LowlatCapacityPlan:
    logical_context: int = 16384
    chunk_tokens: int = 512
    layout: str = "full"
    prefix_cache: bool = False

    def __post_init__(self):
        if type(self.logical_context) is not int or not 1 <= self.logical_context <= MODEL_CONTEXT:
            raise ValueError("lowlat context must be an integer in [1, 1048576]")
        if type(self.chunk_tokens) is not int or not 1 <= self.chunk_tokens <= MAX_PREFILL:
            raise ValueError("lowlat prefill chunk must be an integer in [1, 8192]")
        if self.layout not in ("full", "rolling") or type(self.prefix_cache) is not bool:
            raise ValueError("invalid lowlat cache policy")
        if self.prefix_cache and self.layout != "rolling":
            raise ValueError("lowlat prefix caching requires rolling cache layout")
        if self.logical_context > 16384 and self.layout != "rolling":
            raise ValueError("lowlat extended context requires rolling cache layout")

    @property
    def compiled_context(self):
        return ceil(self.logical_context / 128) * 128

    @property
    def padded_chunk_tokens(self):
        return ceil(self.chunk_tokens / 8) * 8

    @property
    def rope_bytes_per_rank(self):
        # Two profiles, cos/sin, 64 BF16 columns.
        return self.compiled_context * 2 * 2 * 64 * 2

    @property
    def groups(self):
        active = {
            "ori": ceil((self.padded_chunk_tokens + 127) / 128) + 1,
            "hca_state": 18,
            "csa_state": 4,
            "csa_inner_state": 4,
        }
        windows = {"ori": 128, "hca_state": 128, "csa_state": 8, "csa_inner_state": 8}
        result = []
        for name, (tokens, nbytes, _, _) in CACHE_GROUPS.items():
            logical = ceil(self.logical_context / tokens)
            count = min(logical, active.get(name, logical)) if self.layout == "rolling" else logical
            window = (
                min(windows[name], count * tokens) if self.layout == "rolling" and name in windows else None
            )
            # A retained checkpoint and a transition source must not alias the
            # active writable ring. Full-history capacity stays model-bounded.
            reserve = count + ceil(windows[name] / tokens) if self.prefix_cache and window else 0
            result.append(LowlatGroupCapacity(name, tokens, nbytes, logical, count, count + reserve, window))
        return tuple(result)

    @property
    def cache_bytes_per_rank(self):
        return sum(g.physical_pool_blocks * g.bytes_per_page for g in self.groups)

    def cache_specs(self):
        return tuple(
            KVCacheGroupSpec(
                name=g.name,
                layer_indices=CACHE_GROUPS[g.name][2],
                spec=KVCacheSpec(
                    block_size=g.source_tokens_per_page,
                    page_size_bytes=g.bytes_per_page,
                    compress_ratio=CACHE_GROUPS[g.name][3],
                ),
                max_blocks_per_seq=g.active_ring_blocks,
                num_blocks=g.physical_pool_blocks,
                sliding_window=g.sliding_window,
                num_partitions=1,
                max_requests_per_partition=1,
            )
            for g in self.groups
        )
