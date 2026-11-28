# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KV load observed within one serving replica."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypto_serving.serving.memory.kv_cache import KVCacheCapacityObservation


@dataclass(frozen=True)
class ReplicaKVLoad:
    rank: int
    capacity: KVCacheCapacityObservation
    waiting_tokens: int
    waiting_requests: int
    uncached_free_head_tokens: int | None

    def __post_init__(self) -> None:
        total = self.capacity.total_tokens
        if not 0 <= self.capacity.used_tokens <= total:
            raise ValueError("used KV tokens must be in [0, total KV tokens]")
        if self.uncached_free_head_tokens is not None and not 0 <= self.uncached_free_head_tokens <= total:
            raise ValueError("uncached KV headroom must be in [0, total KV tokens]")
        if min(self.rank, self.waiting_tokens, self.waiting_requests) < 0:
            raise ValueError("rank and native waiting demand must be non-negative")

    @property
    def replica_id(self) -> str:
        return f"pypto-replica-{self.rank}"
