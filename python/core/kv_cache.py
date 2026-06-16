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
from dataclasses import dataclass, field

import torch

from .kv_offload import (
    CPULoadStoreSpec,
    KVBlockLocation,
    NPULoadStoreSpec,
    TransferJob,
    TransferResult,
)
from .types import KvAllocation, ModelConfig, RuntimeConfig


NONE_HASH = hash(("__none__",))


def hash_block_tokens(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    """Return a chained prefix-cache hash for one full token block."""
    return hash((parent_hash, token_ids))


@dataclass(slots=True)
class KVCacheBlock:
    """Metadata for one physical KV cache page/block."""

    block_id: int
    ref_cnt: int = 0
    block_hash: int | None = None
    location: KVBlockLocation = KVBlockLocation.NPU
    physical_page_id: int | None = None
    cpu_slot_id: int | None = None
    cpu_last_access: int = 0
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
    """Paged KV cache storage for one registered model."""

    page_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_blocks_per_seq: int
    key_pages: torch.Tensor
    value_pages: torch.Tensor


class KvCacheManager:
    """Unified KV block metadata and paged KV tensor storage manager."""

    def __init__(
        self,
        *,
        num_blocks: int | None = None,
        block_size: int = 64,
        enable_prefix_cache: bool = True,
        max_cpu_offload_blocks: int = 0,
    ) -> None:
        """Create an empty registry of model-specific KV pools."""
        self._pools: dict[str, _CachePool] = {}
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks: list[KVCacheBlock] = []
        self.free_queue = FreeKVCacheBlockQueue()
        self.hash_to_block: dict[int, KVCacheBlock] = {}
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}
        self.max_cpu_offload_blocks = int(max_cpu_offload_blocks)
        self._next_transfer_job_id = 0
        self._cpu_access_clock = 0
        self._free_cpu_slots: list[int] = list(range(self.max_cpu_offload_blocks))
        self._pending_transfer_blocks: dict[int, list[int]] = {}
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

    @property
    def num_free_cpu_slots(self) -> int:
        """Return the number of CPU offload slots available for new stores."""
        return len(self._free_cpu_slots)

    def _init_blocks(self, num_blocks: int, block_size: int) -> None:
        if self.blocks:
            if len(self.blocks) != num_blocks or self.block_size != block_size:
                raise ValueError("KV block pool is already initialized with different dimensions")
            return
        self.block_size = block_size
        self.blocks = [
            KVCacheBlock(block_id=i, physical_page_id=i)
            for i in range(num_blocks)
        ]
        for block in self.blocks:
            self.free_queue.append(block)

    def register_model(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> None:
        """Create the KV page pool for a model if it is not already registered."""
        if model_id in self._pools:
            return
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        num_pages = runtime.total_kv_pages
        if num_pages is None:
            num_pages = runtime.max_batch_size * max_blocks_per_seq
        self._init_blocks(num_pages, runtime.page_size)
        kv_dtype = getattr(torch, runtime.kv_dtype)
        key_pages = torch.zeros(
            config.num_hidden_layers,
            num_pages,
            config.num_key_value_heads,
            runtime.page_size,
            config.head_dim,
            dtype=kv_dtype,
            device=runtime.device,
        )
        value_pages = torch.zeros_like(key_pages)
        self._pools[model_id] = _CachePool(
            page_size=runtime.page_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_blocks_per_seq=max_blocks_per_seq,
            key_pages=key_pages,
            value_pages=value_pages,
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
        return self.allocate_preferred_blocks(num_blocks, [])

    def allocate_preferred_blocks(
        self,
        num_blocks: int,
        preferred_block_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        """Allocate blocks, trying specific free block IDs before FIFO fallback."""
        if num_blocks <= 0:
            return []
        if self.num_free_blocks < num_blocks:
            return None
        blocks: list[KVCacheBlock] = []
        used_preferred: set[int] = set()
        for block_id in preferred_block_ids:
            if len(blocks) >= num_blocks:
                break
            if block_id in used_preferred or block_id < 0 or block_id >= len(self.blocks):
                continue
            block = self.blocks[block_id]
            if not self._is_free(block):
                continue
            self.free_queue.remove(block)
            self._prepare_allocated_block(block)
            blocks.append(block)
            used_preferred.add(block_id)

        for _ in range(num_blocks):
            if len(blocks) >= num_blocks:
                break
            block = self.free_queue.popleft()
            if block is None:
                for allocated in blocks:
                    self.release(allocated)
                return None
            self._prepare_allocated_block(block)
            blocks.append(block)
        return blocks

    def allocate_block_ids(self, num_blocks: int) -> list[int] | None:
        """Allocate physical KV blocks and return their IDs."""
        blocks = self.allocate_blocks(num_blocks)
        if blocks is None:
            return None
        return [block.block_id for block in blocks]

    def allocate_preferred_block_ids(
        self,
        num_blocks: int,
        preferred_block_ids: list[int],
    ) -> list[int] | None:
        """Allocate physical KV blocks and return their IDs, preferring given IDs."""
        blocks = self.allocate_preferred_blocks(num_blocks, preferred_block_ids)
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
        if block.location == KVBlockLocation.CPU:
            self._touch_cpu_block(block)
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

    def build_cpu_store_job(self, block_ids: list[int], request_id: str | None = None) -> TransferJob:
        """Build a transfer job that stores resident NPU blocks into CPU slots."""
        blocks = [self.blocks[block_id] for block_id in block_ids]
        for block in blocks:
            if block.ref_cnt != 0:
                raise RuntimeError(f"Cannot offload KV block {block.block_id} with active references.")
        if len(blocks) > self.max_cpu_offload_blocks:
            raise RuntimeError("Insufficient CPU KV offload slots.")
        while len(self._free_cpu_slots) < len(blocks):
            if self._evict_lru_cpu_block() is None:
                raise RuntimeError("Insufficient CPU KV offload slots.")

        page_ids: list[int] = []
        slot_ids: list[int] = []
        for block in blocks:
            if block.location != KVBlockLocation.NPU or block.physical_page_id is None:
                raise RuntimeError(f"KV block {block.block_id} is not resident on NPU.")
            if block.ref_cnt == 0:
                self.free_queue.remove(block)
            slot_id = self._free_cpu_slots.pop(0)
            block.location = KVBlockLocation.TRANSFERRING
            block.cpu_slot_id = slot_id
            block.cpu_last_access = 0
            page_ids.append(block.physical_page_id)
            slot_ids.append(slot_id)

        job = TransferJob(
            self._next_job_id(),
            request_id,
            NPULoadStoreSpec(page_ids),
            CPULoadStoreSpec(slot_ids),
        )
        self._pending_transfer_blocks[job.job_id] = [block.block_id for block in blocks]
        return job

    def build_cpu_load_job(self, block_ids: list[int], request_id: str | None = None) -> TransferJob:
        """Build a transfer job that loads CPU-offloaded blocks back to NPU pages."""
        blocks = [self.blocks[block_id] for block_id in block_ids]
        page_ids: list[int] = []
        slot_ids: list[int] = []
        for block in blocks:
            if block.location != KVBlockLocation.CPU or block.cpu_slot_id is None:
                raise RuntimeError(f"KV block {block.block_id} is not resident on CPU.")
            block.location = KVBlockLocation.TRANSFERRING
            block.physical_page_id = block.block_id
            block.cpu_last_access = 0
            page_ids.append(block.physical_page_id)
            slot_ids.append(block.cpu_slot_id)

        job = TransferJob(
            self._next_job_id(),
            request_id,
            CPULoadStoreSpec(slot_ids),
            NPULoadStoreSpec(page_ids),
        )
        self._pending_transfer_blocks[job.job_id] = [block.block_id for block in blocks]
        return job

    def complete_transfer_result(self, result: TransferResult) -> None:
        """Apply a completed CPU offload transfer to block residency metadata."""
        block_ids = self._pending_transfer_blocks.pop(result.job_id, [])
        if not block_ids:
            return
        blocks = [self.blocks[block_id] for block_id in block_ids]
        if not result.success:
            if isinstance(result.src, NPULoadStoreSpec) and isinstance(result.dst, CPULoadStoreSpec):
                for block in blocks:
                    block.location = KVBlockLocation.NPU
                    if block.cpu_slot_id is not None:
                        self._free_cpu_slots.append(block.cpu_slot_id)
                        block.cpu_slot_id = None
                    block.cpu_last_access = 0
                    if block.ref_cnt == 0:
                        self.free_queue.append(block)
            elif isinstance(result.src, CPULoadStoreSpec) and isinstance(result.dst, NPULoadStoreSpec):
                for block in blocks:
                    block.location = KVBlockLocation.CPU
                    block.physical_page_id = None
                    block.cpu_last_access = 0
            else:
                raise TypeError("Unsupported KV offload transfer result")
            raise RuntimeError(result.error or "KV offload transfer failed")

        if isinstance(result.src, NPULoadStoreSpec) and isinstance(result.dst, CPULoadStoreSpec):
            for block, slot_id in zip(blocks, result.dst.slot_ids, strict=True):
                block.location = KVBlockLocation.CPU
                block.physical_page_id = None
                block.cpu_slot_id = slot_id
                self._touch_cpu_block(block)
                if block.ref_cnt == 0:
                    self.free_queue.append(block)
        elif isinstance(result.src, CPULoadStoreSpec) and isinstance(result.dst, NPULoadStoreSpec):
            for block, page_id in zip(blocks, result.dst.page_ids, strict=True):
                block.location = KVBlockLocation.NPU
                block.physical_page_id = page_id
                if block.cpu_slot_id is not None:
                    self._free_cpu_slots.append(block.cpu_slot_id)
                    block.cpu_slot_id = None
                    block.cpu_last_access = 0
        else:
            raise TypeError("Unsupported KV offload transfer result")

    def release(self, block: KVCacheBlock) -> None:
        """Release one request reference to a block."""
        if block.ref_cnt <= 0:
            return
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self.free_queue.append(block)

    def _next_job_id(self) -> int:
        job_id = self._next_transfer_job_id
        self._next_transfer_job_id += 1
        return job_id

    def _is_free(self, block: KVCacheBlock) -> bool:
        return (
            block.ref_cnt == 0
            and (
                block == self.free_queue.head
                or block == self.free_queue.tail
                or block.prev_free is not None
                or block.next_free is not None
            )
        )

    def _prepare_allocated_block(self, block: KVCacheBlock) -> None:
        if block.block_hash is not None:
            self.hash_to_block.pop(block.block_hash, None)
            block.block_hash = None
        if block.cpu_slot_id is not None:
            self._free_cpu_slots.append(block.cpu_slot_id)
        block.location = KVBlockLocation.NPU
        block.physical_page_id = block.block_id
        block.cpu_slot_id = None
        block.cpu_last_access = 0
        block.ref_cnt = 1

    def _touch_cpu_block(self, block: KVCacheBlock) -> None:
        self._cpu_access_clock += 1
        block.cpu_last_access = self._cpu_access_clock

    def _evict_lru_cpu_block(self) -> KVCacheBlock | None:
        candidates = [
            block
            for block in self.blocks
            if block.location == KVBlockLocation.CPU
            and block.ref_cnt == 0
            and block.cpu_slot_id is not None
        ]
        if not candidates:
            return None
        victim = min(candidates, key=lambda block: block.cpu_last_access)
        if victim.block_hash is not None:
            self.hash_to_block.pop(victim.block_hash, None)
            victim.block_hash = None
        self._free_cpu_slots.append(victim.cpu_slot_id)
        victim.cpu_slot_id = None
        victim.cpu_last_access = 0
        victim.location = KVBlockLocation.NPU
        victim.physical_page_id = victim.block_id
        return victim

    def _iter_block_hashes(self, token_ids: list[int]):
        """Yield (block_index, block_hash) for each full block in the token sequence."""
        parent_hash = NONE_HASH
        num_full_blocks = len(token_ids) // self.block_size
        for i in range(num_full_blocks):
            start = i * self.block_size
            block_tokens = tuple(token_ids[start : start + self.block_size])
            parent_hash = hash_block_tokens(parent_hash, block_tokens)
            yield i, parent_hash

    def get_computed_blocks(self, token_ids: list[int]) -> list[KVCacheBlock]:
        """Find the longest full-block cached prefix for the token sequence."""
        if not self.enable_prefix_cache:
            return []
        hit_blocks: list[KVCacheBlock] = []
        for _, block_hash in self._iter_block_hashes(token_ids):
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

    def write_tokens(
        self,
        layer_idx: int,
        alloc: KvAllocation,
        start_token_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Write key/value rows for consecutive tokens into paged cache."""
        pool = self._pool(alloc.model_id)
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
        pool = self._pool(alloc.model_id)
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
        pool = self._pool(model_id)
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

    def _pool(self, model_id: str) -> _CachePool:
        """Return the registered cache pool for a model."""
        if model_id not in self._pools:
            raise KeyError(f"Model {model_id} is not registered with the KV cache manager.")
        return self._pools[model_id]
