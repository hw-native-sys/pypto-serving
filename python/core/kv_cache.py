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
    OffloadKey,
    SSDLoadStoreSpec,
    TorchKVPageView,
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
    logical_block_id: int | None = None
    physical_page_id: int | None = None
    location: KVBlockLocation = KVBlockLocation.NPU
    cpu_slot_id: int | None = None
    ssd_slot_id: int | None = None
    pending_job_id: int | None = None
    offload_key: OffloadKey | None = None
    dirty: bool = False
    last_access_ts: float = 0.0
    prev_free: "KVCacheBlock | None" = field(default=None, repr=False)
    next_free: "KVCacheBlock | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.logical_block_id is None:
            self.logical_block_id = self.block_id
        if self.physical_page_id is None:
            self.physical_page_id = self.block_id


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
        if max_cpu_offload_blocks < 0:
            raise ValueError("max_cpu_offload_blocks must be non-negative")
        self._pools: dict[str, _CachePool] = {}
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.max_cpu_offload_blocks = max_cpu_offload_blocks
        self.blocks: list[KVCacheBlock] = []
        self.free_queue = FreeKVCacheBlockQueue()
        self.hash_to_block: dict[int, KVCacheBlock] = {}
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}
        self.page_id_to_pending_jobs: dict[int, set[int]] = {}
        self.cpu_slot_to_block_id: dict[int, int] = {}
        self.ssd_slot_to_block_id: dict[int, int] = {}
        self._num_physical_pages: int = 0
        self._free_physical_page_ids: list[int] = []
        self._free_cpu_slots: list[int] = []
        self._free_ssd_slots: list[int] = []
        self._next_cpu_slot_id: int = 0
        self._next_ssd_slot_id: int = 0
        self._next_transfer_job_id: int = 0
        if num_blocks is not None:
            self._init_blocks(num_blocks, block_size)

    @property
    def num_free_blocks(self) -> int:
        """Return the number of immediately allocatable KV blocks."""
        npu_free_blocks = 0
        block = self.free_queue.head
        while block is not None:
            if block.physical_page_id is not None:
                npu_free_blocks += 1
            block = block.next_free
        return npu_free_blocks + len(self._free_physical_page_ids)

    @property
    def num_blocks(self) -> int:
        """Return the total number of physical KV blocks."""
        return len(self.blocks)

    def _init_blocks(self, num_blocks: int, block_size: int) -> None:
        if self.blocks:
            if len(self.blocks) != num_blocks or self.block_size != block_size:
                raise ValueError("KV block pool is already initialized with different dimensions")
            return
        self.block_size = block_size
        self._num_physical_pages = num_blocks
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
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
        page_ids = [self._resident_physical_page_id(block) for block in blocks]
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
            if block is None and self._free_physical_page_ids:
                block = self._new_logical_block()
            if block is None:
                for allocated in blocks:
                    self.release(allocated)
                return None
            physical_page_id = block.physical_page_id
            if physical_page_id is None:
                physical_page_id = self._allocate_physical_page_id()
                if physical_page_id is None:
                    for allocated in blocks:
                        self.release(allocated)
                    self.free_queue.append(block)
                    return None
            if block.block_hash is not None:
                self.hash_to_block.pop(block.block_hash, None)
                block.block_hash = None
            if block.pending_job_id is not None:
                for allocated in blocks:
                    self.release(allocated)
                self.free_queue.append(block)
                return None
            if block.cpu_slot_id is not None:
                self._release_cpu_slot(block)
            if block.ssd_slot_id is not None:
                self._release_ssd_slot(block)
            block.location = KVBlockLocation.NPU
            block.physical_page_id = physical_page_id
            block.cpu_slot_id = None
            block.ssd_slot_id = None
            block.offload_key = None
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
        if block.ref_cnt == 0 and block.pending_job_id is None:
            self.free_queue.append(block)

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

    def resident_block_ids(self, block_ids: list[int]) -> list[int]:
        """Resolve logical block IDs to resident NPU physical page IDs."""
        physical_ids: list[int] = []
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.location != KVBlockLocation.NPU or block.physical_page_id is None:
                raise RuntimeError(f"KV block {block_id} is not resident on NPU: {block.location.value}")
            physical_ids.append(block.physical_page_id)
        return physical_ids

    def mark_blocks_moving_to_ssd(
        self,
        block_ids: list[int],
        ssd_slot_ids: list[int],
        job_id: int,
        *,
        offload_keys: list[OffloadKey] | None = None,
    ) -> None:
        """Mark resident NPU blocks as being stored to SSD by one transfer job."""
        if len(block_ids) != len(ssd_slot_ids):
            raise ValueError("block_ids and ssd_slot_ids must have the same length")
        if offload_keys is not None and len(offload_keys) != len(block_ids):
            raise ValueError("offload_keys and block_ids must have the same length")
        for idx, block_id in enumerate(block_ids):
            block = self.blocks[block_id]
            if block.location != KVBlockLocation.NPU or block.physical_page_id is None:
                raise RuntimeError(f"KV block {block_id} is not resident on NPU")
            if block.pending_job_id is not None:
                raise RuntimeError(f"KV block {block_id} already has pending job {block.pending_job_id}")
            if block.ref_cnt == 0:
                self.free_queue.remove(block)
            block.location = KVBlockLocation.MOVING_TO_SSD
            block.ssd_slot_id = ssd_slot_ids[idx]
            block.pending_job_id = job_id
            if offload_keys is not None:
                block.offload_key = offload_keys[idx]
            self.page_id_to_pending_jobs.setdefault(block.physical_page_id, set()).add(job_id)

    def mark_blocks_moving_to_cpu(
        self,
        block_ids: list[int],
        cpu_slot_ids: list[int],
        job_id: int,
        *,
        offload_keys: list[OffloadKey] | None = None,
    ) -> None:
        """Mark resident NPU blocks as being stored to CPU by one transfer job."""
        if len(block_ids) != len(cpu_slot_ids):
            raise ValueError("block_ids and cpu_slot_ids must have the same length")
        if offload_keys is not None and len(offload_keys) != len(block_ids):
            raise ValueError("offload_keys and block_ids must have the same length")
        for idx, block_id in enumerate(block_ids):
            block = self.blocks[block_id]
            if block.location != KVBlockLocation.NPU or block.physical_page_id is None:
                raise RuntimeError(f"KV block {block_id} is not resident on NPU")
            if block.pending_job_id is not None:
                raise RuntimeError(f"KV block {block_id} already has pending job {block.pending_job_id}")
            if block.ref_cnt == 0:
                self.free_queue.remove(block)
            block.location = KVBlockLocation.MOVING_TO_CPU
            block.cpu_slot_id = cpu_slot_ids[idx]
            block.pending_job_id = job_id
            if offload_keys is not None:
                block.offload_key = offload_keys[idx]
            self.page_id_to_pending_jobs.setdefault(block.physical_page_id, set()).add(job_id)

    def complete_cpu_store_job(self, job_id: int, *, success: bool = True) -> None:
        """Complete one NPU-to-CPU store job and publish or invalidate its blocks."""
        for block in self.blocks:
            if block.pending_job_id != job_id or block.location != KVBlockLocation.MOVING_TO_CPU:
                continue
            self._clear_pending_page_job(block, job_id)
            block.pending_job_id = None
            if success:
                block.location = KVBlockLocation.CPU
                self._release_physical_page_id(block.physical_page_id)
                block.physical_page_id = None
                block.dirty = False
                if block.ref_cnt == 0:
                    self.free_queue.append(block)
            else:
                block.location = KVBlockLocation.NPU
                block.offload_key = None
                self._release_cpu_slot(block)
                if block.ref_cnt == 0:
                    self.free_queue.append(block)

    def complete_store_job(self, job_id: int, *, success: bool = True) -> None:
        """Complete one NPU-to-SSD store job and publish or invalidate its blocks."""
        for block in self.blocks:
            if block.pending_job_id != job_id or block.location != KVBlockLocation.MOVING_TO_SSD:
                continue
            self._clear_pending_page_job(block, job_id)
            block.pending_job_id = None
            if success:
                block.location = KVBlockLocation.SSD
                self._release_physical_page_id(block.physical_page_id)
                block.physical_page_id = None
                block.dirty = False
                if block.ref_cnt == 0:
                    self.free_queue.append(block)
            else:
                block.location = KVBlockLocation.NPU
                block.offload_key = None
                self._release_ssd_slot(block)
                if block.ref_cnt == 0:
                    self.free_queue.append(block)

    def mark_blocks_moving_to_npu(
        self,
        block_ids: list[int],
        physical_page_ids: list[int],
        job_id: int,
    ) -> None:
        """Mark SSD blocks as being loaded back to NPU pages."""
        if len(block_ids) != len(physical_page_ids):
            raise ValueError("block_ids and physical_page_ids must have the same length")
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.location != KVBlockLocation.SSD:
                raise RuntimeError(f"KV block {block_id} is not stored on SSD")
            if block.pending_job_id is not None:
                raise RuntimeError(f"KV block {block_id} already has pending job {block.pending_job_id}")
        for idx, block_id in enumerate(block_ids):
            block = self.blocks[block_id]
            if block.ref_cnt == 0:
                self.free_queue.remove(block)
            self._reserve_physical_page_id(physical_page_ids[idx])
            block.location = KVBlockLocation.MOVING_TO_NPU
            block.physical_page_id = physical_page_ids[idx]
            block.pending_job_id = job_id
            self.page_id_to_pending_jobs.setdefault(physical_page_ids[idx], set()).add(job_id)

    def mark_cpu_blocks_moving_to_npu(
        self,
        block_ids: list[int],
        physical_page_ids: list[int],
        job_id: int,
    ) -> None:
        """Mark CPU-resident blocks as being loaded back to NPU pages."""
        if len(block_ids) != len(physical_page_ids):
            raise ValueError("block_ids and physical_page_ids must have the same length")
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.location != KVBlockLocation.CPU:
                raise RuntimeError(f"KV block {block_id} is not stored on CPU")
            if block.pending_job_id is not None:
                raise RuntimeError(f"KV block {block_id} already has pending job {block.pending_job_id}")
        for idx, block_id in enumerate(block_ids):
            block = self.blocks[block_id]
            if block.ref_cnt == 0:
                self.free_queue.remove(block)
            self._reserve_physical_page_id(physical_page_ids[idx])
            block.location = KVBlockLocation.MOVING_TO_NPU
            block.physical_page_id = physical_page_ids[idx]
            block.pending_job_id = job_id
            self.page_id_to_pending_jobs.setdefault(physical_page_ids[idx], set()).add(job_id)

    def complete_load_job(self, job_id: int, *, success: bool = True) -> None:
        """Complete one SSD-to-NPU load job and make loaded blocks resident."""
        for block in self.blocks:
            if block.pending_job_id != job_id or block.location != KVBlockLocation.MOVING_TO_NPU:
                continue
            self._clear_pending_page_job(block, job_id)
            block.pending_job_id = None
            if success:
                block.location = KVBlockLocation.NPU
                block.dirty = False
                if block.cpu_slot_id is not None:
                    self._release_cpu_slot(block)
                if block.ssd_slot_id is not None:
                    self._release_ssd_slot(block)
                if block.ref_cnt == 0:
                    self.free_queue.append(block)
            else:
                self._release_physical_page_id(block.physical_page_id)
                block.location = KVBlockLocation.CPU if block.cpu_slot_id is not None else KVBlockLocation.SSD
                block.physical_page_id = None

    def complete_cpu_load_job(self, job_id: int, *, success: bool = True) -> None:
        """Complete one CPU-to-NPU load job and make loaded blocks resident."""
        self.complete_load_job(job_id, success=success)

    def pending_jobs_for_page(self, page_id: int) -> set[int]:
        """Return transfer jobs that must be fenced before reusing one NPU page."""
        return set(self.page_id_to_pending_jobs.get(page_id, set()))

    def build_cpu_store_job(
        self,
        block_ids: list[int],
        *,
        request_id: str | None = None,
        offload_keys: list[OffloadKey] | None = None,
    ) -> TransferJob:
        """Create and mark one NPU-to-CPU transfer job for logical blocks."""
        job_id = self._next_job_id()
        page_ids = self.resident_block_ids(block_ids)
        cpu_slot_ids = self._allocate_cpu_slots(block_ids)
        try:
            self.mark_blocks_moving_to_cpu(
                block_ids,
                cpu_slot_ids,
                job_id,
                offload_keys=offload_keys,
            )
        except Exception:
            for slot_id in cpu_slot_ids:
                self.cpu_slot_to_block_id.pop(slot_id, None)
                self._free_cpu_slots.append(slot_id)
            raise
        return TransferJob(
            job_id=job_id,
            request_id=request_id,
            src=NPULoadStoreSpec(page_ids),
            dst=CPULoadStoreSpec(cpu_slot_ids),
            keys=tuple(offload_keys or ()),
        )

    def build_cpu_load_job(
        self,
        block_ids: list[int],
        physical_page_ids: list[int] | None = None,
        *,
        request_id: str | None = None,
    ) -> TransferJob:
        """Create and mark one CPU-to-NPU transfer job for logical blocks."""
        cpu_slot_ids: list[int] = []
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.location != KVBlockLocation.CPU or block.cpu_slot_id is None:
                raise RuntimeError(f"KV block {block_id} is not stored on CPU")
            cpu_slot_ids.append(block.cpu_slot_id)
        allocated_physical_pages = physical_page_ids is None
        if physical_page_ids is None:
            physical_page_ids = self._allocate_physical_page_ids(len(block_ids))
            if physical_page_ids is None:
                raise RuntimeError("Insufficient free NPU pages for CPU KV load")
        if len(block_ids) != len(physical_page_ids):
            raise ValueError("block_ids and physical_page_ids must have the same length")
        job_id = self._next_job_id()
        try:
            self.mark_cpu_blocks_moving_to_npu(block_ids, physical_page_ids, job_id)
        except Exception:
            if allocated_physical_pages:
                for page_id in physical_page_ids:
                    self._release_physical_page_id(page_id)
            raise
        return TransferJob(
            job_id=job_id,
            request_id=request_id,
            src=CPULoadStoreSpec(cpu_slot_ids),
            dst=NPULoadStoreSpec(physical_page_ids),
        )

    def build_ssd_store_job(
        self,
        block_ids: list[int],
        *,
        request_id: str | None = None,
        offload_keys: list[OffloadKey] | None = None,
    ) -> TransferJob:
        """Create and mark one NPU-to-SSD transfer job for logical blocks."""
        job_id = self._next_job_id()
        page_ids = self.resident_block_ids(block_ids)
        ssd_slot_ids = self._allocate_ssd_slots(block_ids)
        try:
            self.mark_blocks_moving_to_ssd(
                block_ids,
                ssd_slot_ids,
                job_id,
                offload_keys=offload_keys,
            )
        except Exception:
            for slot_id in ssd_slot_ids:
                self.ssd_slot_to_block_id.pop(slot_id, None)
                self._free_ssd_slots.append(slot_id)
            raise
        return TransferJob(
            job_id=job_id,
            request_id=request_id,
            src=NPULoadStoreSpec(page_ids),
            dst=SSDLoadStoreSpec(ssd_slot_ids),
            keys=tuple(offload_keys or ()),
        )

    def build_ssd_load_job(
        self,
        block_ids: list[int],
        physical_page_ids: list[int] | None = None,
        *,
        request_id: str | None = None,
    ) -> TransferJob:
        """Create and mark one SSD-to-NPU transfer job for logical blocks."""
        ssd_slot_ids: list[int] = []
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.location != KVBlockLocation.SSD or block.ssd_slot_id is None:
                raise RuntimeError(f"KV block {block_id} is not stored on SSD")
            ssd_slot_ids.append(block.ssd_slot_id)
        allocated_physical_pages = physical_page_ids is None
        if physical_page_ids is None:
            physical_page_ids = self._allocate_physical_page_ids(len(block_ids))
            if physical_page_ids is None:
                raise RuntimeError("Insufficient free NPU pages for SSD KV load")
        if len(block_ids) != len(physical_page_ids):
            raise ValueError("block_ids and physical_page_ids must have the same length")
        job_id = self._next_job_id()
        try:
            self.mark_blocks_moving_to_npu(block_ids, physical_page_ids, job_id)
        except Exception:
            if allocated_physical_pages:
                for page_id in physical_page_ids:
                    self._release_physical_page_id(page_id)
            raise
        return TransferJob(
            job_id=job_id,
            request_id=request_id,
            src=SSDLoadStoreSpec(ssd_slot_ids),
            dst=NPULoadStoreSpec(physical_page_ids),
        )

    def select_cpu_offload_candidates(
        self,
        num_blocks: int,
        *,
        excluded_block_ids: set[int] | None = None,
    ) -> list[int]:
        """Pick resident blocks that can be stored to CPU to free NPU pages."""
        if num_blocks <= 0:
            return []
        excluded_block_ids = excluded_block_ids or set()
        candidates = [
            block
            for block in self.blocks
            if block.block_id not in excluded_block_ids
            and block.location == KVBlockLocation.NPU
            and block.physical_page_id is not None
            and block.pending_job_id is None
        ]
        candidates.sort(key=lambda block: (block.ref_cnt > 0, block.last_access_ts, block.block_id))
        return [block.block_id for block in candidates[:num_blocks]]

    def non_resident_block_ids(self, block_ids: list[int]) -> list[int]:
        """Return blocks from the input list that are not currently resident on NPU."""
        return [
            block_id
            for block_id in block_ids
            if self.blocks[block_id].location != KVBlockLocation.NPU
            or self.blocks[block_id].physical_page_id is None
        ]

    def complete_transfer_result(self, result: TransferResult) -> None:
        """Apply one worker-reported KV transfer completion to block metadata."""
        self.complete_transfer_job(result.job_id, success=result.success)

    def complete_transfer_job(self, job_id: int, *, success: bool = True) -> None:
        """Complete one pending transfer job regardless of direction/medium."""
        matched = False
        for block in self.blocks:
            if block.pending_job_id != job_id:
                continue
            matched = True
            if block.location == KVBlockLocation.MOVING_TO_CPU:
                self.complete_cpu_store_job(job_id, success=success)
            elif block.location == KVBlockLocation.MOVING_TO_SSD:
                self.complete_store_job(job_id, success=success)
            elif block.location == KVBlockLocation.MOVING_TO_NPU:
                self.complete_load_job(job_id, success=success)
            break
        if not matched:
            return

    def ensure_one_more_slot(self, alloc: KvAllocation) -> int:
        """Ensure a request has capacity for one more token and return its slot."""
        pool = self._pool(alloc.model_id)
        if alloc.tokens_used >= alloc.tokens_capacity:
            blocks = self.allocate_blocks(1)
            if blocks is None:
                raise RuntimeError("Insufficient KV cache blocks.")
            self.request_blocks.setdefault(alloc.request_id, []).extend(blocks)
            alloc.page_ids.extend(self._resident_physical_page_id(block) for block in blocks)
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

    def materialize_page_view(self, model_id: str) -> TorchKVPageView:
        """Return a page-contiguous view over all K/V layers for CPU offload."""
        pool = self._pool(model_id)
        components: list[torch.Tensor] = []
        for layer_idx in range(pool.num_layers):
            components.append(pool.key_pages[layer_idx])
            components.append(pool.value_pages[layer_idx])
        return TorchKVPageView(components)

    def free(self, alloc: KvAllocation) -> None:
        """Return an allocation's pages to the model pool."""
        self.release_request(alloc.request_id)
        alloc.page_ids.clear()
        alloc.tokens_capacity = 0
        alloc.tokens_used = 0

    def _clear_pending_page_job(self, block: KVCacheBlock, job_id: int) -> None:
        page_id = block.physical_page_id
        if page_id is None:
            return
        pending_jobs = self.page_id_to_pending_jobs.get(page_id)
        if pending_jobs is None:
            return
        pending_jobs.discard(job_id)
        if not pending_jobs:
            del self.page_id_to_pending_jobs[page_id]

    def _next_job_id(self) -> int:
        job_id = self._next_transfer_job_id
        self._next_transfer_job_id += 1
        return job_id

    def _new_logical_block(self) -> KVCacheBlock:
        block = KVCacheBlock(block_id=len(self.blocks), physical_page_id=None)
        self.blocks.append(block)
        return block

    def _resident_physical_page_id(self, block: KVCacheBlock) -> int:
        if block.location != KVBlockLocation.NPU or block.physical_page_id is None:
            raise RuntimeError(f"KV block {block.block_id} is not resident on NPU")
        return block.physical_page_id

    def _allocate_physical_page_id(self) -> int | None:
        if not self._free_physical_page_ids:
            return None
        return self._free_physical_page_ids.pop()

    def _allocate_physical_page_ids(self, count: int) -> list[int] | None:
        if count <= 0:
            return []
        if len(self._free_physical_page_ids) < count:
            return None
        return [self._free_physical_page_ids.pop() for _ in range(count)]

    def _release_physical_page_id(self, page_id: int | None) -> None:
        if page_id is None:
            return
        if page_id < 0 or page_id >= self._num_physical_pages:
            raise ValueError(f"Invalid NPU page id {page_id}")
        if page_id in self._free_physical_page_ids:
            return
        self._free_physical_page_ids.append(page_id)

    def _reserve_physical_page_id(self, page_id: int) -> None:
        if page_id < 0 or page_id >= self._num_physical_pages:
            raise ValueError(f"Invalid NPU page id {page_id}")
        try:
            self._free_physical_page_ids.remove(page_id)
        except ValueError:
            pass

    def _allocate_cpu_slots(self, block_ids: list[int]) -> list[int]:
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.cpu_slot_id is not None:
                raise RuntimeError(f"KV block {block_id} already owns CPU slot {block.cpu_slot_id}")
        new_slots_needed = max(0, len(block_ids) - len(self._free_cpu_slots))
        if (
            self._next_cpu_slot_id + new_slots_needed > self.max_cpu_offload_blocks
        ):
            raise RuntimeError("CPU KV offload capacity exceeded")
        slots: list[int] = []
        for block_id in block_ids:
            if self._free_cpu_slots:
                slot_id = self._free_cpu_slots.pop()
            else:
                slot_id = self._next_cpu_slot_id
                self._next_cpu_slot_id += 1
            self.cpu_slot_to_block_id[slot_id] = block_id
            slots.append(slot_id)
        return slots

    def _release_cpu_slot(self, block: KVCacheBlock) -> None:
        slot_id = block.cpu_slot_id
        if slot_id is None:
            return
        self.cpu_slot_to_block_id.pop(slot_id, None)
        self._free_cpu_slots.append(slot_id)
        block.cpu_slot_id = None

    def _allocate_ssd_slots(self, block_ids: list[int]) -> list[int]:
        slots: list[int] = []
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.ssd_slot_id is not None:
                raise RuntimeError(f"KV block {block_id} already owns SSD slot {block.ssd_slot_id}")
            if self._free_ssd_slots:
                slot_id = self._free_ssd_slots.pop()
            else:
                slot_id = self._next_ssd_slot_id
                self._next_ssd_slot_id += 1
            self.ssd_slot_to_block_id[slot_id] = block_id
            slots.append(slot_id)
        return slots

    def _release_ssd_slot(self, block: KVCacheBlock) -> None:
        slot_id = block.ssd_slot_id
        if slot_id is None:
            return
        self.ssd_slot_to_block_id.pop(slot_id, None)
        self._free_ssd_slots.append(slot_id)
        block.ssd_slot_id = None

    def _pool(self, model_id: str) -> _CachePool:
        """Return the registered cache pool for a model."""
        if model_id not in self._pools:
            raise KeyError(f"Model {model_id} is not registered with the KV cache manager.")
        return self._pools[model_id]
