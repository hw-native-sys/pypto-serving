# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .radix import RadixKey, TreeNode


@dataclass(frozen=True)
class MatchResult:
    """The canonical KV locations for a matched token prefix."""

    device_indices: tuple[int, ...]
    last_device_node: "TreeNode"
    last_host_node: "TreeNode"
    best_match_node: "TreeNode"
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    cache_protected_len: int | None = None


@dataclass(frozen=True)
class InsertResult:
    """Result of publishing a token-to-KV mapping into a prefix cache."""

    prefix_len: int
    total_len: int = 0
    last_device_node: "TreeNode | None" = None
    inserted_len: int = 0


@dataclass(frozen=True)
class EvictResult:
    """Number of logical tokens removed from device KV storage."""

    num_tokens_evicted: int = 0


@dataclass
class PrefixCacheStats:
    """Low-cost counters used by Stage 1 tests and A/B measurements."""

    lookups: int = 0
    matched_tokens: int = 0
    inserts: int = 0
    inserted_tokens: int = 0
    duplicate_pages_freed: int = 0
    evicted_tokens: int = 0
    in_batch_deferred: int = 0
    scheduled_prefill_tokens: int = 0
    extra: dict[str, int] = field(default_factory=dict)


class BasePrefixCache(ABC):
    """Dense-prefix cache protocol adapted from SGLang's public semantics."""

    page_size: int
    disable: bool

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def match_prefix(self, key: "RadixKey") -> MatchResult:
        raise NotImplementedError

    @abstractmethod
    def insert(self, key: "RadixKey", value: list[int], *, priority: int = 0) -> InsertResult:
        raise NotImplementedError

    @abstractmethod
    def evict(self, num_tokens: int) -> EvictResult:
        raise NotImplementedError

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> int:
        raise NotImplementedError

    @abstractmethod
    def dec_lock_ref(self, node: Any) -> int:
        raise NotImplementedError

    @abstractmethod
    def evictable_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def protected_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def total_size(self) -> int:
        raise NotImplementedError
