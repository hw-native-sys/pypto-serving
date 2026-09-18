# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import hashlib
import struct
from dataclasses import dataclass, field

import torch

from pypto_serving.config.types import KVCacheGroupSpec, KvAllocation, ModelConfig, RuntimeConfig


_PREFIX_HASH_DOMAIN = b"pypto-prefix-cache-v1\0"
NONE_HASH = int.from_bytes(hashlib.sha256(_PREFIX_HASH_DOMAIN + b"root").digest(), "big")


class KVCacheCapacityError(RuntimeError):
    """Raised when a cache allocation cannot fit in the physical pools."""


def hash_block_tokens(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    """Return a deterministic chained digest for one full token block.

    Python's built-in ``hash`` is randomized per interpreter and cannot name
    cache objects shared across workers or recovered after a restart. Keep the
    integer representation used by the in-memory indexes while deriving it
    from a stable, versioned byte encoding.
    """
    if parent_hash < 0 or parent_hash.bit_length() > 256:
        raise ValueError("parent_hash must be an unsigned 256-bit integer")
    digest = hashlib.sha256()
    digest.update(_PREFIX_HASH_DOMAIN)
    digest.update(parent_hash.to_bytes(32, "big"))
    digest.update(struct.pack(">Q", len(token_ids)))
    for token_id in token_ids:
        if token_id < 0 or token_id > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("token IDs must be unsigned 64-bit integers")
        digest.update(struct.pack(">Q", token_id))
    return int.from_bytes(digest.digest(), "big")


@dataclass(slots=True)
class KVCacheBlock:
    """Metadata for one physical KV cache page/block."""

    block_id: int
    ref_cnt: int = 0
    block_hash: int | None = None
    prev_free: "KVCacheBlock | None" = field(default=None, repr=False)
    next_free: "KVCacheBlock | None" = field(default=None, repr=False)


@dataclass(frozen=True)
class KVCacheBlocks:
    """Scheduler-facing KV blocks grouped by cache group."""

    blocks: tuple[list[KVCacheBlock], ...]

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple([block.block_id for block in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        if len(self.blocks) != 1:
            raise ValueError("get_unhashed_block_ids requires one KV cache group")
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]


class FreeKVCacheBlockQueue:
    """Doubly-linked free block queue in eviction order."""

    def __init__(self) -> None:
        self.head: KVCacheBlock | None = None
        self.tail: KVCacheBlock | None = None
        self.count: int = 0

    def append(self, block: KVCacheBlock) -> None:
        block.prev_free = self.tail
        block.next_free = None
        if self.tail is not None:
            self.tail.next_free = block
        else:
            self.head = block
        self.tail = block
        self.count += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.append(block)

    def popleft(self) -> KVCacheBlock | None:
        if self.head is None:
            return None
        block = self.head
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        if block != self.head and block != self.tail and block.prev_free is None and block.next_free is None:
            return
        prev_b = block.prev_free
        next_b = block.next_free
        if prev_b is not None:
            prev_b.next_free = next_b
        else:
            self.head = next_b
        if next_b is not None:
            next_b.prev_free = prev_b
        else:
            self.tail = prev_b
        block.prev_free = None
        block.next_free = None
        self.count -= 1

    def __len__(self) -> int:
        return self.count


@dataclass
class _CachePool:
    """Paged KV cache storage for one registered model.

    ``key_pages`` / ``value_pages`` are allocated lazily on first access
    via ``write_tokens`` or ``read_context``.  NPU serving paths never
    trigger the allocation — they manage KV cache on-device through the
    runner.
    """

    page_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_blocks_per_seq: int
    num_pages: int
    kv_dtype: torch.dtype
    key_pages: torch.Tensor | None = None
    value_pages: torch.Tensor | None = None


@dataclass
class _GroupBlockPool:
    """Rank-partitioned physical block namespaces for one cache family."""

    spec: KVCacheGroupSpec
    blocks: list[KVCacheBlock]
    blocks_per_partition: int
    free_queues: tuple[FreeKVCacheBlockQueue, ...]
    request_blocks: dict[str, list[KVCacheBlock | None]] = field(default_factory=dict)
    request_partitions: dict[str, int] = field(default_factory=dict)
    request_logical_blocks: dict[str, int] = field(default_factory=dict)
    hash_to_block: dict[tuple[int, int], KVCacheBlock] = field(default_factory=dict)

    def num_free_blocks_in(self, partition: int) -> int:
        return len(self.free_queues[partition])

    def local_block_id(self, block: KVCacheBlock) -> int:
        return block.block_id % self.blocks_per_partition

    def block_from_local_id(self, partition: int, block_id: int) -> KVCacheBlock:
        if not 0 <= partition < len(self.free_queues):
            raise ValueError(f"Invalid cache partition {partition}")
        if not 0 <= block_id < self.blocks_per_partition:
            raise ValueError(
                f"Invalid local block ID {block_id} for cache group {self.spec.name!r}"
            )
        return self.blocks[partition * self.blocks_per_partition + block_id]


class KvCacheManager:
    """Unified KV block metadata and paged KV tensor storage manager."""

    def __init__(
        self,
        *,
        num_blocks: int | None = None,
        block_size: int = 64,
        enable_prefix_cache: bool = True,
    ) -> None:
        """Create an empty registry of model-specific KV pools."""
        self._pools: dict[str, _CachePool] = {}
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks: list[KVCacheBlock] = []
        self.free_queue = FreeKVCacheBlockQueue()
        self.hash_to_block: dict[int, KVCacheBlock] = {}
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}
        self._group_pools: dict[str, _GroupBlockPool] = {}
        self._group_request_partitions: dict[str, int] = {}
        self._next_group_partition = 0
        if num_blocks is not None:
            self._init_blocks(num_blocks, block_size)

    @property
    def num_free_blocks(self) -> int:
        """Return the number of immediately allocatable KV blocks."""
        return self.free_queue.count

    @property
    def num_blocks(self) -> int:
        """Return the total number of physical KV blocks."""
        return len(self.blocks)

    def initialize(
        self,
        runtime: RuntimeConfig,
        *,
        num_blocks: int,
    ) -> None:
        """Initialize scheduler-visible pools from the runtime cache topology.

        ``num_blocks`` is the device-reported capacity of the primary cache
        group, or the complete capacity for a generic single cache pool.
        """
        num_blocks = int(num_blocks)
        if num_blocks <= 0:
            raise RuntimeError(
                f"Worker reported invalid KV cache block count: {num_blocks}"
            )
        if runtime.kv_cache_groups:
            self.init_groups(
                runtime.kv_cache_groups,
                max_batch_size=runtime.max_batch_size,
                primary_num_blocks=num_blocks,
            )
            return
        self._init_blocks(num_blocks, runtime.page_size)

    def _init_blocks(self, num_blocks: int, block_size: int) -> None:
        if self.blocks:
            if len(self.blocks) != num_blocks or self.block_size != block_size:
                raise ValueError("KV block pool is already initialized with different dimensions")
            return
        self.block_size = block_size
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        for block in self.blocks:
            self.free_queue.append(block)

    def register_model(
        self,
        model_id: str,
        config: ModelConfig,
        runtime: RuntimeConfig,
        *,
        num_pages: int | None = None,
    ) -> None:
        """Register model metadata using the device-reported page capacity when supplied."""
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        if num_pages is None:
            num_pages = runtime.total_kv_pages
            if num_pages is None:
                num_pages = runtime.max_batch_size * max_blocks_per_seq
        if num_pages <= 0:
            raise ValueError(f"KV cache page count must be positive, got {num_pages}")
        if model_id in self._pools:
            if self._pools[model_id].num_pages != num_pages:
                raise ValueError(
                    f"Model {model_id} is already registered with "
                    f"{self._pools[model_id].num_pages} KV pages, not {num_pages}"
                )
            return
        self._init_blocks(num_pages, runtime.page_size)
        kv_dtype = getattr(torch, runtime.kv_dtype)
        self._pools[model_id] = _CachePool(
            page_size=runtime.page_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_blocks_per_seq=max_blocks_per_seq,
            num_pages=num_pages,
            kv_dtype=kv_dtype,
        )

    def allocate_for_prompt(self, model_id: str, request_id: str, prompt_len: int) -> KvAllocation:
        """Allocate enough KV pages to store a prompt of ``prompt_len`` tokens."""
        pool = self._pool(model_id)
        num_pages = max(1, math.ceil(prompt_len / pool.page_size))
        blocks = self.allocate_blocks(num_pages)
        if blocks is None:
            raise RuntimeError("Insufficient KV cache blocks.")
        self.request_blocks[request_id] = blocks
        page_ids = [block.block_id for block in blocks]
        return KvAllocation(
            request_id=request_id,
            model_id=model_id,
            page_ids=page_ids,
            tokens_capacity=len(page_ids) * pool.page_size,
            tokens_used=0,
        )

    def allocate_blocks(self, num_blocks: int) -> list[KVCacheBlock] | None:
        """Allocate physical KV blocks, evicting stale prefix hashes as needed."""
        if num_blocks <= 0:
            return []
        if self.num_free_blocks < num_blocks:
            return None
        blocks: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self.free_queue.popleft()
            if block is None:
                for allocated in blocks:
                    self.release(allocated)
                return None
            if block.block_hash is not None:
                self.hash_to_block.pop(block.block_hash, None)
                block.block_hash = None
            block.ref_cnt = 1
            blocks.append(block)
        return blocks

    def allocate_block_ids(self, num_blocks: int) -> list[int] | None:
        """Allocate physical KV blocks and return their IDs."""
        blocks = self.allocate_blocks(num_blocks)
        if blocks is None:
            return None
        return [block.block_id for block in blocks]

    def release_blocks_by_ids(self, *block_id_groups: list[int]) -> None:
        """Release request references for one or more groups of physical block IDs."""
        for block_ids in block_id_groups:
            for block_id in block_ids:
                self.release(self.blocks[block_id])

    def release_cached_blocks(self, blocks: list[KVCacheBlock]) -> None:
        """Release cached block objects returned by ``get_computed_blocks``."""
        for block in blocks:
            self.release(block)

    def release_request(self, request_id: str) -> None:
        """Release all blocks tracked for a request."""
        blocks = self.request_blocks.pop(request_id, [])
        for block in blocks:
            self.release(block)

    def get_cached_block(self, block_hash: int) -> KVCacheBlock | None:
        """Return and reference a cached block for one block hash."""
        if not self.enable_prefix_cache:
            return None
        block = self.hash_to_block.get(block_hash)
        if block is None:
            return None
        if block.ref_cnt == 0:
            self.free_queue.remove(block)
        block.ref_cnt += 1
        return block

    def cache_block(self, block: KVCacheBlock, block_hash: int) -> None:
        """Publish a full block to the prefix cache."""
        if not self.enable_prefix_cache:
            return
        if block.block_hash is not None and block.block_hash in self.hash_to_block:
            del self.hash_to_block[block.block_hash]
        block.block_hash = block_hash
        self.hash_to_block[block_hash] = block

    def cache_block_ids(self, block_ids: list[int], block_hashes: list[int], start: int, end: int) -> None:
        """Publish a range of full blocks to the prefix cache."""
        if not self.enable_prefix_cache:
            return
        for idx in range(start, end):
            if idx >= len(block_hashes) or idx >= len(block_ids):
                break
            self.cache_block(self.blocks[block_ids[idx]], block_hashes[idx])

    def release(self, block: KVCacheBlock) -> None:
        """Release one request reference to a block."""
        if block.ref_cnt <= 0:
            return
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self.free_queue.append(block)

    @staticmethod
    def _iter_token_block_hashes(token_ids: list[int], block_size: int):
        """Yield chained hashes for full ``block_size`` token chunks."""
        parent_hash = NONE_HASH
        num_full_blocks = len(token_ids) // block_size
        for i in range(num_full_blocks):
            start = i * block_size
            block_tokens = tuple(token_ids[start : start + block_size])
            parent_hash = hash_block_tokens(parent_hash, block_tokens)
            yield i, parent_hash

    @staticmethod
    def _iter_eagle_token_block_hashes(token_ids: list[int], block_size: int):
        """Yield hashes for shifted EAGLE/MTP pages with one-token look-ahead.

        MTP page ``i`` stores rows whose embedding inputs end at token
        ``(i + 1) * block_size``. Including that boundary token in the page
        hash prevents two equal prompt pages with different following tokens
        from deduplicating to incompatible physical MTP KV.
        """
        parent_hash = NONE_HASH
        num_full_blocks = max(0, (len(token_ids) - 1) // block_size)
        for i in range(num_full_blocks):
            start = i * block_size
            block_tokens = tuple(token_ids[start : start + block_size + 1])
            parent_hash = hash_block_tokens(parent_hash, block_tokens)
            yield i, parent_hash

    def _iter_block_hashes(self, token_ids: list[int]):
        """Yield hashes at the generic cache pool's block granularity."""
        yield from self._iter_token_block_hashes(token_ids, self.block_size)

    def get_computed_blocks(
        self,
        token_ids: list[int],
        *,
        max_cache_hit_tokens: int | None = None,
    ) -> list[KVCacheBlock]:
        """Find the longest full-block cached prefix for the token sequence."""
        if not self.enable_prefix_cache:
            return []
        if max_cache_hit_tokens is None:
            # Keep one token uncached so the scheduler can recompute logits
            # without writing into the final shared prefix-cache block.
            max_cache_hit_tokens = max(0, len(token_ids) - 1)
        max_hit_blocks = max(0, int(max_cache_hit_tokens)) // self.block_size
        hit_blocks: list[KVCacheBlock] = []
        for _, block_hash in self._iter_block_hashes(token_ids):
            if len(hit_blocks) >= max_hit_blocks:
                break
            block = self.get_cached_block(block_hash)
            if block is None:
                break
            hit_blocks.append(block)
        return hit_blocks

    def compute_block_hashes(self, token_ids: list[int]) -> list[int]:
        """Compute chained hashes for all full blocks in the token sequence."""
        return [block_hash for _, block_hash in self._iter_block_hashes(token_ids)]

    def ensure_one_more_slot(self, alloc: KvAllocation) -> int:
        """Ensure a request has capacity for one more token and return its slot."""
        pool = self._pool(alloc.model_id)
        if alloc.tokens_used >= alloc.tokens_capacity:
            blocks = self.allocate_blocks(1)
            if blocks is None:
                raise RuntimeError("Insufficient KV cache blocks.")
            self.request_blocks.setdefault(alloc.request_id, []).extend(blocks)
            alloc.page_ids.extend(block.block_id for block in blocks)
            alloc.tokens_capacity = len(alloc.page_ids) * pool.page_size
        return self.slot_mapping_for_request(alloc, alloc.tokens_used)

    def slot_mapping_for_request(self, alloc: KvAllocation, token_index: int | None = None) -> int:
        """Return the physical slot index for a request token."""
        pool = self._pool(alloc.model_id)
        logical_index = alloc.tokens_used if token_index is None else token_index
        page_idx = logical_index // pool.page_size
        offset = logical_index % pool.page_size
        return alloc.page_ids[page_idx] * pool.page_size + offset

    def slot_mapping_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        """Return current decode slot mappings for a batch."""
        return torch.tensor(
            [self.slot_mapping_for_request(alloc) for alloc in allocations],
            dtype=torch.int32,
        )

    def _ensure_host_pool(self, model_id: str) -> _CachePool:
        """Return the pool, allocating host-side tensors on first access.

        The host-side :class:`torch.Tensor` pool is only needed by the CPU
        executor (``write_tokens`` / ``read_context``).  NPU serving paths
        never call these methods, so the tensors are never allocated.
        """
        pool = self._pool(model_id)
        if pool.key_pages is None:
            key_pages = torch.zeros(
                pool.num_layers,
                pool.num_pages,
                pool.num_kv_heads,
                pool.page_size,
                pool.head_dim,
                dtype=pool.kv_dtype,
                device="cpu",
            )
            value_pages = torch.zeros_like(key_pages)
            pool.key_pages = key_pages
            pool.value_pages = value_pages
        return pool

    def write_tokens(
        self,
        layer_idx: int,
        alloc: KvAllocation,
        start_token_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write key/value rows for consecutive tokens into paged cache."""
        pool = self._ensure_host_pool(alloc.model_id)
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        for row in range(keys.shape[0]):
            token_index = start_token_index + row
            page_idx = token_index // pool.page_size
            offset = token_index % pool.page_size
            physical_page = alloc.page_ids[page_idx]
            pool.key_pages[layer_idx, physical_page, :, offset, :] = keys[row]
            pool.value_pages[layer_idx, physical_page, :, offset, :] = values[row]
        alloc.tokens_used = max(alloc.tokens_used, start_token_index + keys.shape[0])

    def read_context(
        self,
        layer_idx: int,
        alloc: KvAllocation,
        upto_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read contiguous K/V context for one request and layer."""
        pool = self._ensure_host_pool(alloc.model_id)
        token_count = alloc.tokens_used if upto_tokens is None else upto_tokens
        keys = torch.empty(
            token_count,
            pool.num_kv_heads,
            pool.head_dim,
            dtype=pool.key_pages.dtype,
            device=pool.key_pages.device,
        )
        values = torch.empty_like(keys)
        for token_index in range(token_count):
            page_idx = token_index // pool.page_size
            offset = token_index % pool.page_size
            physical_page = alloc.page_ids[page_idx]
            keys[token_index] = pool.key_pages[layer_idx, physical_page, :, offset, :]
            values[token_index] = pool.value_pages[layer_idx, physical_page, :, offset, :]
        return keys, values

    def materialize_single_layer_cache(
        self,
        model_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened K/V cache views for exactly one model layer.

        The returned tensors are zero-copy views over the selected layer of
        the paged cache, shaped ``[num_pages * num_kv_heads * page_size,
        head_dim]``. Use this API for kernels that receive one layer's cache
        at a time.
        """
        pool = self._ensure_host_pool(model_id)
        return (
            pool.key_pages[layer_idx].reshape(-1, pool.head_dim),
            pool.value_pages[layer_idx].reshape(-1, pool.head_dim),
        )

    def free(self, alloc: KvAllocation) -> None:
        """Return an allocation's pages to the model pool."""
        self.release_request(alloc.request_id)
        alloc.page_ids.clear()
        alloc.tokens_capacity = 0
        alloc.tokens_used = 0

    @property
    def has_groups(self) -> bool:
        """Return whether model-specific cache groups are configured."""
        return bool(self._group_pools)

    @property
    def group_names(self) -> tuple[str, ...]:
        """Return cache group names in stable allocation order."""
        return tuple(self._group_pools)

    @property
    def group_specs(self) -> tuple[KVCacheGroupSpec, ...]:
        """Return grouped-cache layouts in stable allocation order."""
        return tuple(pool.spec for pool in self._group_pools.values())

    def init_groups(
        self,
        group_specs: tuple[KVCacheGroupSpec, ...],
        *,
        max_batch_size: int,
        primary_num_blocks: int | None = None,
    ) -> None:
        """Initialize independent physical pools for model-specific caches.

        ``primary_num_blocks`` is the device-reported capacity of the first
        group. All groups are scaled to the same number of
        ``max_blocks_per_seq`` capacity slots, keeping heterogeneous physical
        pools in lockstep with the runner's allocation. Explicit ``num_blocks``
        values must agree with the device-reported capacity.
        """
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not group_specs:
            return
        names = [group.name for group in group_specs]
        if len(names) != len(set(names)):
            raise ValueError("KV cache group names must be unique")

        partition_counts = {group.num_partitions for group in group_specs}
        if len(partition_counts) != 1:
            raise ValueError("KV cache groups must use the same num_partitions")

        capacity_slots = None
        if primary_num_blocks is not None:
            primary_num_blocks = int(primary_num_blocks)
            primary_stride = group_specs[0].max_blocks_per_seq
            if primary_num_blocks <= 0 or primary_num_blocks % primary_stride:
                raise ValueError(
                    "primary_num_blocks must be a positive multiple of the first "
                    "KV cache group's max_blocks_per_seq"
                )
            capacity_slots = primary_num_blocks // primary_stride
        if capacity_slots is not None:
            requested_sizes = {
                group.name: capacity_slots * group.max_blocks_per_seq
                for group in group_specs
            }
            conflicting_sizes = [
                (
                    group.name,
                    group.num_blocks,
                    requested_sizes[group.name],
                )
                for group in group_specs
                if group.num_blocks is not None
                and group.num_blocks != requested_sizes[group.name]
            ]
            if conflicting_sizes:
                details = ", ".join(
                    f"{name} configured={configured}, device={device}"
                    for name, configured, device in conflicting_sizes
                )
                raise ValueError(
                    "KV cache group num_blocks conflicts with device-reported capacity: "
                    + details
                )
        else:
            requested_sizes = {
                group.name: (
                    group.num_blocks
                    if group.num_blocks is not None
                    else max_batch_size * group.max_blocks_per_seq
                )
                for group in group_specs
            }
        undersized = [
            group.name
            for group in group_specs
            if requested_sizes[group.name] < group.max_blocks_per_seq
        ]
        if undersized:
            raise ValueError(
                "KV cache groups cannot hold one maximum-length sequence: " + ", ".join(undersized)
            )
        if self._group_pools:
            existing = {
                name: (pool.spec, pool.blocks_per_partition)
                for name, pool in self._group_pools.items()
            }
            requested = {
                group.name: (group, requested_sizes[group.name])
                for group in group_specs
            }
            if existing != requested:
                raise ValueError("KV cache groups are already initialized with different specifications")
            return

        for group in group_specs:
            blocks_per_partition = requested_sizes[group.name]
            total_blocks = blocks_per_partition * group.num_partitions
            blocks = [KVCacheBlock(block_id=block_id) for block_id in range(total_blocks)]
            free_queues = tuple(FreeKVCacheBlockQueue() for _ in range(group.num_partitions))
            for partition, free_queue in enumerate(free_queues):
                start = partition * blocks_per_partition
                free_queue.append_n(blocks[start : start + blocks_per_partition])
            self._group_pools[group.name] = _GroupBlockPool(
                spec=group,
                blocks=blocks,
                blocks_per_partition=blocks_per_partition,
                free_queues=free_queues,
            )

    def required_group_block_counts(self, token_count: int) -> dict[str, int]:
        """Return the per-group allocation size needed for ``token_count``."""
        if token_count < 0:
            raise ValueError("token_count must not be negative")
        counts = {}
        for name, pool in self._group_pools.items():
            logical_blocks = math.ceil(token_count / pool.spec.spec.token_capacity)
            if pool.spec.sliding_window is not None:
                logical_blocks = min(logical_blocks, pool.spec.max_blocks_per_seq)
            counts[name] = logical_blocks
        return counts

    def completed_group_block_counts(self, token_count: int) -> dict[str, int]:
        """Return absolute full-page counts at a logical token boundary."""
        if token_count < 0:
            raise ValueError("token_count must not be negative")
        return {
            name: token_count // pool.spec.spec.token_capacity
            for name, pool in self._group_pools.items()
        }

    def published_group_block_counts(self, token_count: int) -> dict[str, int]:
        """Return the full-page counts eligible for prefix-cache publication."""
        if token_count < 0:
            raise ValueError("token_count must not be negative")
        counts = {}
        for name, pool in self._group_pools.items():
            safe_tokens = token_count
            if pool.spec.is_eagle_group:
                safe_tokens = max(0, safe_tokens - 1)
            counts[name] = safe_tokens // pool.spec.spec.token_capacity
        return counts

    @property
    def group_prefix_cache_alignment(self) -> int:
        """Return the source-token boundary shared by every cache group.

        Strict zero-copy reuse requires every referenced physical page to be
        immutable. A grouped hit is therefore aligned to the least common
        multiple of each group's full-page logical token capacity.
        """
        alignment = 1
        for pool in self._group_pools.values():
            alignment = math.lcm(alignment, pool.spec.spec.token_capacity)
        return alignment

    @property
    def has_eagle_groups(self) -> bool:
        """Return whether grouped lookup uses shifted EAGLE/MTP page hashes."""
        return any(pool.spec.is_eagle_group for pool in self._group_pools.values())

    def compute_group_block_hashes(self, token_ids: list[int]) -> dict[str, list[int]]:
        """Compute full-page prefix hashes independently for every cache group."""
        hashes = {}
        for name, pool in self._group_pools.items():
            iterator = (
                self._iter_eagle_token_block_hashes
                if pool.spec.is_eagle_group
                else self._iter_token_block_hashes
            )
            hashes[name] = [
                block_hash
                for _, block_hash in iterator(
                    token_ids,
                    pool.spec.spec.token_capacity,
                )
            ]
        return hashes

    def acquire_group_prefix_blocks(
        self,
        request_id: str,
        block_hashes: dict[str, list[int]],
        *,
        max_cache_hit_tokens: int,
    ) -> tuple[dict[str, list[int]], int, int | None]:
        """Attach the longest strict zero-copy grouped prefix to a request.

        Cache namespaces are rank-local, so every partition is evaluated and
        the partition with the longest hit is selected. Full-history groups
        match from the start. Rolling groups match only the immutable tail
        needed to resume at the candidate boundary. EAGLE/MTP page hashes
        already include the boundary token consumed by their last shifted KV
        row, so a matching page is self-validating. Partial pages are never
        shared. Sparse rolling rows are completed by ``ensure_group_blocks``
        before they are sent to a worker.
        """
        if not self.enable_prefix_cache or not self._group_pools:
            return {}, 0, None
        if request_id in self._group_request_partitions:
            raise ValueError(f"Request {request_id!r} already owns grouped KV cache blocks")
        expected_names = set(self._group_pools)
        if set(block_hashes) != expected_names:
            missing = sorted(expected_names - set(block_hashes))
            extra = sorted(set(block_hashes) - expected_names)
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("extra=" + ",".join(extra))
            raise ValueError("Grouped prefix hashes do not match cache groups (" + "; ".join(details) + ")")

        alignment = self.group_prefix_cache_alignment
        max_hit_tokens = max(0, int(max_cache_hit_tokens))
        # EAGLE hashes include one boundary token in the page being matched.
        # Do not also reserve or require another whole page here: publication
        # and hash availability already prove that the shifted page is safe.
        max_candidate = max_hit_tokens
        max_candidate -= max_candidate % alignment
        if max_candidate == 0:
            return {}, 0, None

        best_partition: int | None = None
        best_hit_tokens = 0
        best_blocks: dict[str, list[KVCacheBlock | None]] = {}
        for partition in range(self.group_partition_count):
            candidate = max_candidate
            for name, pool in self._group_pools.items():
                if pool.spec.sliding_window is not None:
                    continue
                token_capacity = pool.spec.spec.token_capacity
                max_blocks = min(
                    candidate // token_capacity,
                    pool.spec.max_blocks_per_seq,
                    len(block_hashes[name]),
                )
                num_hit_blocks = 0
                for block_hash in block_hashes[name][:max_blocks]:
                    block = pool.hash_to_block.get((partition, block_hash))
                    if block is None:
                        break
                    num_hit_blocks += 1
                candidate = min(candidate, num_hit_blocks * token_capacity)

            candidate -= candidate % alignment
            candidate_blocks: dict[str, list[KVCacheBlock | None]] = {}
            while candidate > 0:
                candidate_blocks = {}
                valid = True
                for name, pool in self._group_pools.items():
                    token_capacity = pool.spec.spec.token_capacity
                    end_block = candidate // token_capacity
                    if end_block > len(block_hashes[name]):
                        valid = False
                        break

                    if pool.spec.sliding_window is None:
                        if end_block > pool.spec.max_blocks_per_seq:
                            valid = False
                            break
                        selected: list[KVCacheBlock | None] = []
                        lookup_indices = range(end_block)
                    else:
                        table_size = min(end_block, pool.spec.max_blocks_per_seq)
                        selected = [None] * table_size
                        tail_blocks = min(
                            end_block,
                            pool.spec.sliding_window // token_capacity,
                        )
                        lookup_indices = range(end_block - tail_blocks, end_block)

                    for block_index in lookup_indices:
                        block_hash = block_hashes[name][block_index]
                        block = pool.hash_to_block.get((partition, block_hash))
                        if block is None:
                            valid = False
                            break
                        if pool.spec.sliding_window is None:
                            selected.append(block)
                        else:
                            selected[block_index % len(selected)] = block
                    if not valid:
                        break
                    candidate_blocks[name] = selected

                if valid:
                    break
                candidate -= alignment

            if candidate > best_hit_tokens:
                best_partition = partition
                best_hit_tokens = candidate
                best_blocks = candidate_blocks

        if best_partition is None or best_hit_tokens == 0:
            return {}, 0, None

        self._group_request_partitions[request_id] = best_partition
        for name, pool in self._group_pools.items():
            owned = best_blocks[name]
            pool.request_blocks[request_id] = list(owned)
            pool.request_partitions[request_id] = best_partition
            pool.request_logical_blocks[request_id] = (
                best_hit_tokens // pool.spec.spec.token_capacity
            )
            for block in owned:
                if block is None:
                    continue
                if block.ref_cnt == 0:
                    pool.free_queues[best_partition].remove(block)
                block.ref_cnt += 1

        return (
            {
                name: [
                    pool.local_block_id(block)
                    for block in best_blocks[name]
                    if block is not None
                ]
                for name, pool in self._group_pools.items()
            },
            best_hit_tokens,
            best_partition,
        )

    def cache_group_blocks(
        self,
        request_id: str,
        block_hashes: dict[str, list[int]],
        num_computed_tokens: int,
        already_cached: dict[str, int],
    ) -> dict[str, int]:
        """Publish newly completed grouped pages and return cached counts."""
        if not self.enable_prefix_cache:
            return dict(already_cached)
        partition = self._group_request_partitions.get(request_id)
        if partition is None:
            return dict(already_cached)

        cached_counts = dict(already_cached)
        for name, pool in self._group_pools.items():
            hashes = block_hashes.get(name, ())
            owned = pool.request_blocks.get(request_id, ())
            safe_computed_tokens = int(num_computed_tokens)
            if pool.spec.is_eagle_group:
                # MTP KV row p consumes token p + 1. The newest main-model row
                # is therefore pending until another prompt/generated token is
                # known, so a page ending at that row is not immutable yet.
                safe_computed_tokens = max(0, safe_computed_tokens - 1)
            completed = min(
                safe_computed_tokens // pool.spec.spec.token_capacity,
                len(hashes),
            )
            start = min(cached_counts.get(name, 0), completed)
            if pool.spec.sliding_window is not None:
                allocated = pool.request_logical_blocks.get(request_id, 0)
                start = max(start, allocated - len(owned))
            for index in range(start, completed):
                slot = index if pool.spec.sliding_window is None else index % len(owned)
                block = owned[slot]
                if block is None:
                    raise RuntimeError(
                        f"KV cache group {name!r} has an unallocated table slot"
                    )
                self._cache_group_block(pool, partition, block, hashes[index])
            cached_counts[name] = completed
        return cached_counts

    def retain_group_block_snapshot(
        self,
        block_ids_by_group: dict[str, list[int]],
        partition: int,
    ) -> None:
        """Pin one scheduled grouped block-table snapshot until its step settles.

        Async scheduling may advance a rolling request table before an older
        device step returns. The scheduled table must remain physically valid
        until that step is confirmed or discarded, matching vLLM's in-flight
        block lifetime fence.
        """
        if not block_ids_by_group:
            return
        if set(block_ids_by_group) != set(self._group_pools):
            raise ValueError("Grouped block snapshot does not match configured cache groups")

        blocks = []
        for name, block_ids in block_ids_by_group.items():
            pool = self._group_pools[name]
            if len(block_ids) != len(set(block_ids)):
                raise ValueError(f"Cache group {name!r} snapshot contains duplicate block IDs")
            for block_id in block_ids:
                block = pool.block_from_local_id(partition, block_id)
                if block.ref_cnt <= 0:
                    raise RuntimeError(
                        f"Cannot retain unowned grouped KV block {name}:{block_id}"
                    )
                blocks.append(block)

        for block in blocks:
            block.ref_cnt += 1

    def release_group_block_snapshot(
        self,
        block_ids_by_group: dict[str, list[int]],
        partition: int,
    ) -> None:
        """Release pins acquired by :meth:`retain_group_block_snapshot`."""
        for name, block_ids in block_ids_by_group.items():
            pool = self._group_pools[name]
            for block_id in block_ids:
                block = pool.block_from_local_id(partition, block_id)
                if block.ref_cnt <= 0:
                    raise RuntimeError(
                        f"Grouped KV snapshot block {name}:{block_id} has "
                        f"invalid ref_cnt={block.ref_cnt}"
                    )
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    pool.free_queues[partition].append(block)

    def cache_group_blocks_from_snapshot(
        self,
        block_hashes: dict[str, list[int]],
        num_computed_tokens: int,
        already_cached: dict[str, int],
        block_ids_by_group: dict[str, list[int]],
        partition: int,
    ) -> dict[str, int]:
        """Publish confirmed pages using the exact table used by one step.

        Unlike ``cache_group_blocks``, this method never consults the request's
        latest rolling table. That table may already describe a newer in-flight
        chunk when async scheduling depth is greater than one.
        """
        if not self.enable_prefix_cache:
            return dict(already_cached)
        if set(block_hashes) != set(self._group_pools):
            raise ValueError("Grouped prefix hashes do not match configured cache groups")
        if set(block_ids_by_group) != set(self._group_pools):
            raise ValueError("Grouped block snapshot does not match configured cache groups")

        cached_counts = dict(already_cached)
        for name, pool in self._group_pools.items():
            hashes = block_hashes[name]
            block_ids = block_ids_by_group[name]
            safe_computed_tokens = int(num_computed_tokens)
            if pool.spec.is_eagle_group:
                safe_computed_tokens = max(0, safe_computed_tokens - 1)
            completed = min(
                safe_computed_tokens // pool.spec.spec.token_capacity,
                len(hashes),
            )
            start = min(cached_counts.get(name, 0), completed)
            if pool.spec.sliding_window is not None:
                if completed and not block_ids:
                    raise RuntimeError(f"Cache group {name!r} has an empty scheduled table")
                start = max(start, completed - len(block_ids))

            for index in range(start, completed):
                slot = index if pool.spec.sliding_window is None else index % len(block_ids)
                if slot >= len(block_ids):
                    raise RuntimeError(
                        f"Cache group {name!r} scheduled table cannot address logical block {index}"
                    )
                block = pool.block_from_local_id(partition, block_ids[slot])
                self._cache_group_block(pool, partition, block, hashes[index])
            cached_counts[name] = completed
        return cached_counts

    @staticmethod
    def _cache_group_block(
        pool: _GroupBlockPool,
        partition: int,
        block: KVCacheBlock,
        block_hash: int,
    ) -> None:
        """Publish one immutable full page without replacing a live duplicate."""
        key = (partition, block_hash)
        existing = pool.hash_to_block.get(key)
        if existing is block:
            return
        if existing is not None:
            if block.block_hash is not None:
                old_key = (partition, block.block_hash)
                if pool.hash_to_block.get(old_key) is block:
                    del pool.hash_to_block[old_key]
                block.block_hash = None
            return
        if block.block_hash is not None:
            old_key = (partition, block.block_hash)
            if pool.hash_to_block.get(old_key) is block:
                del pool.hash_to_block[old_key]
        block.block_hash = block_hash
        pool.hash_to_block[key] = block

    @property
    def group_partition_count(self) -> int:
        """Return the number of independent grouped-cache namespaces."""
        if not self._group_pools:
            return 1
        return next(iter(self._group_pools.values())).spec.num_partitions

    def group_request_partition(self, request_id: str) -> int | None:
        """Return the stable cache partition assigned to a request."""
        return self._group_request_partitions.get(request_id)

    @staticmethod
    def _invalidate_group_block(
        pool: _GroupBlockPool,
        partition: int,
        block: KVCacheBlock,
    ) -> None:
        """Remove one block's old contents from its partition hash index."""
        if block.block_hash is None:
            return
        key = (partition, block.block_hash)
        if pool.hash_to_block.get(key) is block:
            del pool.hash_to_block[key]
        block.block_hash = None

    def _take_group_block(
        self,
        pool: _GroupBlockPool,
        partition: int,
    ) -> KVCacheBlock:
        """Take and initialize one evictable block from a group partition."""
        block = pool.free_queues[partition].popleft()
        if block is None:
            raise RuntimeError(
                f"KV cache group {pool.spec.name!r} free queue became inconsistent"
            )
        if block.ref_cnt != 0:
            raise RuntimeError(
                f"Free grouped KV block {block.block_id} has ref_cnt={block.ref_cnt}"
            )
        self._invalidate_group_block(pool, partition, block)
        block.ref_cnt = 1
        return block

    @staticmethod
    def _logical_group_blocks(pool: _GroupBlockPool, token_count: int) -> int:
        return math.ceil(token_count / pool.spec.spec.token_capacity)

    def _additional_group_blocks(
        self,
        pool: _GroupBlockPool,
        request_id: str,
        token_count: int,
    ) -> int:
        """Count new pages, including detach-on-write rolling destinations."""
        target = self._logical_group_blocks(pool, token_count)
        owned = pool.request_blocks.get(request_id, ())
        if pool.spec.sliding_window is None:
            if target > pool.spec.max_blocks_per_seq:
                return pool.blocks_per_partition + 1
            return max(0, target - len(owned))

        table_size = min(target, pool.spec.max_blocks_per_seq)
        missing = max(0, table_size - len(owned))
        missing += sum(block is None for block in owned[:table_size])
        previous = pool.request_logical_blocks.get(request_id, 0)
        shared_slots = set()
        for logical_block in range(previous, target):
            slot = (
                logical_block
                if logical_block < pool.spec.max_blocks_per_seq
                else logical_block % pool.spec.max_blocks_per_seq
            )
            if slot >= len(owned):
                continue
            block = owned[slot]
            if block is not None and block.ref_cnt > 1:
                shared_slots.add(slot)
        return missing + len(shared_slots)

    def _candidate_group_partitions(
        self,
        request_id: str,
        token_count: int,
    ) -> list[int]:
        assigned = self._group_request_partitions.get(request_id)
        candidates = [assigned] if assigned is not None else list(range(self.group_partition_count))
        return [
            partition
            for partition in candidates
            if all(
                self._additional_group_blocks(pool, request_id, token_count)
                <= pool.num_free_blocks_in(partition)
                for pool in self._group_pools.values()
            )
        ]

    def _select_group_partition(self, request_id: str, token_count: int) -> int | None:
        candidates = self._candidate_group_partitions(request_id, token_count)
        if not candidates:
            return None
        assigned = self._group_request_partitions.get(request_id)
        if assigned is not None:
            return assigned

        first_pool = next(iter(self._group_pools.values()))
        owned_counts = {
            partition: sum(
                request_partition == partition
                for request_partition in first_pool.request_partitions.values()
            )
            for partition in candidates
        }
        minimum = min(owned_counts.values())
        least_loaded = {partition for partition, count in owned_counts.items() if count == minimum}
        for offset in range(self.group_partition_count):
            partition = (self._next_group_partition + offset) % self.group_partition_count
            if partition in least_loaded:
                self._next_group_partition = (partition + 1) % self.group_partition_count
                return partition
        raise RuntimeError("group partition selection became inconsistent")

    def can_ensure_group_blocks(
        self,
        request_id: str,
        token_count: int,
        *,
        partition: int | None = None,
    ) -> bool:
        """Return whether every group can atomically grow this request."""
        if token_count < 0:
            raise ValueError("token_count must not be negative")
        if partition is not None:
            assigned = self._group_request_partitions.get(request_id)
            if assigned is not None and assigned != partition:
                return False
            if not 0 <= partition < self.group_partition_count:
                return False
            return partition in self._candidate_group_partitions(request_id, token_count)
        return bool(self._candidate_group_partitions(request_id, token_count))

    def ensure_group_blocks(
        self,
        request_id: str,
        token_count: int,
        *,
        partition: int | None = None,
    ) -> dict[str, list[int]]:
        """Atomically grow all cache groups for a request and return block IDs."""
        if not self._group_pools:
            return {}
        if token_count < 0:
            raise ValueError("token_count must not be negative")
        logical_counts = {
            name: self._logical_group_blocks(pool, token_count)
            for name, pool in self._group_pools.items()
        }
        oversized = [
            name
            for name, pool in self._group_pools.items()
            if pool.spec.sliding_window is None
            and logical_counts[name] > pool.spec.max_blocks_per_seq
        ]
        if oversized:
            raise KVCacheCapacityError(
                "Full-history KV cache table capacity exceeded (" + ", ".join(oversized) + ")"
            )
        assigned = self._group_request_partitions.get(request_id)
        if partition is not None and assigned is not None and partition != assigned:
            raise ValueError(
                f"Request {request_id!r} is assigned to cache partition {assigned}, got {partition}"
            )
        selected = (
            partition
            if partition is not None
            else self._select_group_partition(request_id, token_count)
        )
        if selected is None or not self.can_ensure_group_blocks(
            request_id,
            token_count,
            partition=selected,
        ):
            shortages = []
            for name, pool in self._group_pools.items():
                needed = self._additional_group_blocks(pool, request_id, token_count)
                if selected is not None and 0 <= selected < self.group_partition_count:
                    free = pool.num_free_blocks_in(selected)
                else:
                    free = max(
                        (
                            pool.num_free_blocks_in(candidate)
                            for candidate in range(self.group_partition_count)
                        ),
                        default=0,
                    )
                if needed > free:
                    shortages.append(f"{name}: need {needed}, free {free}")
            if not shortages:
                shortages.append("no cache partition can satisfy all groups atomically")
            raise KVCacheCapacityError(
                "Insufficient grouped KV cache blocks (" + "; ".join(shortages) + ")"
            )

        self._group_request_partitions.setdefault(request_id, selected)
        for name, pool in self._group_pools.items():
            owned = pool.request_blocks.setdefault(request_id, [])
            pool.request_partitions.setdefault(request_id, selected)
            target = logical_counts[name]
            table_size = (
                min(target, pool.spec.max_blocks_per_seq)
                if pool.spec.sliding_window is not None
                else target
            )
            owned.extend([None] * (table_size - len(owned)))
            # Prefix lookup leaves irrelevant rolling slots empty. Materialize
            # them without copying any cached KV before exposing the table.
            for slot in range(table_size):
                if owned[slot] is None:
                    owned[slot] = self._take_group_block(pool, selected)

            previous = pool.request_logical_blocks.get(request_id, 0)
            if pool.spec.sliding_window is not None:
                for logical_block in range(previous, target):
                    if logical_block < pool.spec.max_blocks_per_seq:
                        continue
                    slot = logical_block % pool.spec.max_blocks_per_seq
                    block = owned[slot]
                    if block is None:
                        raise RuntimeError(
                            f"KV cache group {name!r} has an unallocated ring slot"
                        )
                    if block.ref_cnt > 1:
                        # The expired source page stays immutable for its other
                        # owners. The replacement starts empty: no KV copy is
                        # needed because this request is about to overwrite it.
                        block.ref_cnt -= 1
                        owned[slot] = self._take_group_block(pool, selected)
                    else:
                        # Rotate through the eviction queue even for a sole
                        # owner. This preserves immutable rolling checkpoints
                        # whenever spare pages exist, while still falling back
                        # to the same physical page when the partition is
                        # saturated.
                        block.ref_cnt = 0
                        pool.free_queues[selected].append(block)
                        owned[slot] = self._take_group_block(pool, selected)
            pool.request_logical_blocks[request_id] = max(previous, target)

        result = {}
        for name, pool in self._group_pools.items():
            block_ids = []
            for block in pool.request_blocks.get(request_id, ()):
                if block is None:
                    raise RuntimeError(
                        f"KV cache group {name!r} has an unallocated table slot"
                    )
                block_ids.append(pool.local_block_id(block))
            result[name] = block_ids
        return result

    def release_all_group_requests(self, request_id: str) -> None:
        """Release every grouped block owned by a request."""
        for pool in self._group_pools.values():
            partition = pool.request_partitions.get(request_id)
            blocks = pool.request_blocks.get(request_id, [])
            if partition is None and any(block is not None for block in blocks):
                raise RuntimeError(f"Grouped request {request_id!r} has blocks without a partition")

            pool.request_partitions.pop(request_id, None)
            pool.request_logical_blocks.pop(request_id, None)
            pool.request_blocks.pop(request_id, None)
            if partition is None:
                continue
            for block in blocks:
                if block is None:
                    continue
                if block.ref_cnt <= 0:
                    raise RuntimeError(
                        f"Grouped KV block {block.block_id} has invalid ref_cnt={block.ref_cnt}"
                    )
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    pool.free_queues[partition].append(block)
        self._group_request_partitions.pop(request_id, None)

    def prepare_group_external_load(
        self,
        request_id: str,
        manifest,
    ) -> dict[tuple[str, int], int]:
        """Invalidate destination pages and map external objects to local IDs."""
        partition = self._group_request_partitions.get(request_id)
        if partition is None:
            raise ValueError(f"Request {request_id!r} has no grouped cache partition")
        if manifest.token_count % self.group_prefix_cache_alignment:
            raise ValueError("external checkpoint is not aligned to grouped cache pages")
        targets = {}
        for item in manifest.objects:
            pool = self._group_pools.get(item.group_name)
            if pool is None:
                raise ValueError(f"Unknown external cache group {item.group_name!r}")
            owned = pool.request_blocks.get(request_id, ())
            slot = (
                item.logical_block_index
                if pool.spec.sliding_window is None
                else item.logical_block_index % len(owned)
            )
            if slot >= len(owned) or owned[slot] is None:
                raise ValueError(
                    f"Request {request_id!r} has no destination for "
                    f"{item.group_name}:{item.logical_block_index}"
                )
            block = owned[slot]
            expected_hash = int(item.block_digest, 16)
            if block.block_hash == expected_hash:
                continue
            if block.ref_cnt != 1:
                raise RuntimeError(
                    f"External load would overwrite shared block {item.group_name}:"
                    f"{pool.local_block_id(block)}"
                )
            if block.block_hash is not None:
                key = (partition, block.block_hash)
                if pool.hash_to_block.get(key) is block:
                    del pool.hash_to_block[key]
            block.block_hash = None
            targets[(item.group_name, item.logical_block_index)] = pool.local_block_id(block)
        return targets

    def prepare_group_external_save(
        self,
        manifest,
        block_ids_by_group: dict[str, list[int]],
        partition: int,
    ) -> tuple[dict[tuple[str, int], int], dict[str, list[int]]]:
        """Validate and pin the exact physical pages referenced by a checkpoint."""
        if set(block_ids_by_group) != set(self._group_pools):
            raise ValueError("Grouped external save snapshot does not match cache groups")
        if not 0 <= partition < self.group_partition_count:
            raise ValueError("Grouped external save partition is out of range")

        physical_pages: dict[tuple[str, int], int] = {}
        retained: dict[str, list[int]] = {name: [] for name in self._group_pools}
        for item in manifest.objects:
            pool = self._group_pools.get(item.group_name)
            if pool is None:
                raise ValueError(f"Unknown grouped external save cache {item.group_name!r}")
            table = block_ids_by_group[item.group_name]
            if not table:
                raise RuntimeError(f"External save cache group {item.group_name!r} has no pages")
            if pool.spec.sliding_window is None:
                slot = item.logical_block_index
            else:
                slot = item.logical_block_index % len(table)
            if slot >= len(table):
                raise RuntimeError(
                    f"External save cache group {item.group_name!r} cannot address "
                    f"logical block {item.logical_block_index}"
                )
            block_id = table[slot]
            block = pool.block_from_local_id(partition, block_id)
            expected_hash = int(item.block_digest, 16)
            if block.block_hash != expected_hash:
                block = pool.hash_to_block.get((partition, expected_hash))
                if block is None:
                    raise RuntimeError(
                        f"External save page {item.group_name}:{item.logical_block_index} "
                        "does not contain the committed block hash"
                    )
                block_id = pool.local_block_id(block)
            physical_pages[(item.group_name, item.logical_block_index)] = block_id
            retained[item.group_name].append(block_id)

        for name, block_ids in retained.items():
            if len(block_ids) != len(set(block_ids)):
                raise RuntimeError(f"External save cache group {name!r} aliases checkpoint pages")
        for name, block_ids in retained.items():
            pool = self._group_pools[name]
            for block_id in block_ids:
                block = pool.block_from_local_id(partition, block_id)
                if block.ref_cnt == 0:
                    pool.free_queues[partition].remove(block)
                block.ref_cnt += 1
        return physical_pages, retained

    def commit_group_external_load(self, request_id: str, manifest) -> None:
        """Atomically publish every page after a complete external load."""
        partition = self._group_request_partitions.get(request_id)
        if partition is None:
            raise ValueError(f"Request {request_id!r} has no grouped cache partition")
        for item in manifest.objects:
            pool = self._group_pools[item.group_name]
            owned = pool.request_blocks[request_id]
            slot = (
                item.logical_block_index
                if pool.spec.sliding_window is None
                else item.logical_block_index % len(owned)
            )
            block = owned[slot]
            if block is None:
                raise RuntimeError("external cache destination disappeared before commit")
            expected_hash = int(item.block_digest, 16)
            if block.block_hash not in (None, expected_hash):
                raise RuntimeError("external cache destination changed before commit")
            self._cache_group_block(pool, partition, block, expected_hash)

    def group_num_blocks(self, group_name: str) -> int:
        """Return the rank-local physical block capacity of one cache group."""
        return self._group_pool(group_name).blocks_per_partition

    def _group_pool(self, group_name: str) -> _GroupBlockPool:
        try:
            return self._group_pools[group_name]
        except KeyError as exc:
            raise KeyError(f"KV cache group {group_name!r} is not registered") from exc

    def _pool(self, model_id: str) -> _CachePool:
        """Return the registered cache pool for a model."""
        if model_id not in self._pools:
            raise KeyError(f"Model {model_id} is not registered with the KV cache manager.")
        return self._pools[model_id]
