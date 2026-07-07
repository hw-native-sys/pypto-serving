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

from .types import KvAllocation, KvQuantConfig, ModelConfig, RuntimeConfig

# Lazy import to avoid hard dependency when quantization is disabled.
_KVCOMPRESSOR = None


def _get_kv_compressor_cls():
    global _KVCOMPRESSOR
    if _KVCOMPRESSOR is None:
        from .turboquant.compressor import KVCompressor
        _KVCOMPRESSOR = KVCompressor
    return _KVCOMPRESSOR


@dataclass
class _RequestQuantState:
    """Per-request tracking for KV quantization."""
    quant_page_count: int = 0  # how many compressed pages stored in quant_* buffers


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
    # TurboQuant KV cache quantization fields (None when quantization is disabled).
    kv_compressor: object | None = None
    kv_quant_config: KvQuantConfig | None = None
    quant_key_indices: torch.Tensor | None = None
    quant_key_norms: torch.Tensor | None = None
    quant_val_indices: torch.Tensor | None = None
    quant_val_norms: torch.Tensor | None = None
    request_quant_states: dict = field(default_factory=dict)


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

    def _init_blocks(self, num_blocks: int, block_size: int) -> None:
        if self.blocks:
            if len(self.blocks) != num_blocks or self.block_size != block_size:
                raise ValueError("KV block pool is already initialized with different dimensions")
            return
        self.block_size = block_size
        self.blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        for block in self.blocks:
            self.free_queue.append(block)

    def register_model(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> None:
        """Create the KV page pool for a model if it is not already registered."""
        if model_id in self._pools:
            return
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        # Determine number of bf16 pages.  When TurboQuant is enabled we only
        # need a small pool: one resident segment per batch slot (for the
        # residual window) plus enough workspace pages for one full-length
        # sequence (used temporarily during decode to hold decompressed old
        # tokens).
        #
        # NOTE: The workspace is shared across batch slots — it is only needed
        # during the restore→kernel→compress cycle.  For batch=1 this formula
        # is exact.  For batch>1, the caller must serialize the restore/compress
        # per request so that workspace pages are reused across slots, or set
        # total_kv_pages explicitly to cover simultaneous workspace demand.
        kv_quant_config = runtime.kv_quant_config
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
        # Initialize KV quantization if configured
        kv_compressor = None
        quant_key_indices = None
        quant_key_norms = None
        quant_val_indices = None
        quant_val_norms = None
        if kv_quant_config is not None and kv_quant_config.enabled:
            print(f"[KV-Quant] Creating KVCompressor for model {model_id} ...", flush=True)
            KVCompressor = _get_kv_compressor_cls()
            kv_compressor = KVCompressor(
                head_dim=config.head_dim,
                num_layers=config.num_hidden_layers,
                config=kv_quant_config,
                device=runtime.device,
            )
            print(f"[KV-Quant] KVCompressor for model {model_id} created", flush=True)

            # Compressed storage: nibble-packed uint8, two 4-bit indices per byte
            # (the NPU kernel packs byte c = idx[c+HALF]<<4 | idx[c]). Each head
            # therefore occupies head_dim // 2 bytes, not head_dim.
            # Shape: [num_layers, max_seq_pages, num_kv_heads, page_size, head_dim // 2]
            quant_key_indices = torch.zeros(
                config.num_hidden_layers, max_blocks_per_seq,
                config.num_key_value_heads, runtime.page_size, config.head_dim // 2,
                dtype=torch.uint8, device=runtime.device,
            )
            quant_key_norms = torch.zeros(
                config.num_hidden_layers, max_blocks_per_seq,
                config.num_key_value_heads, runtime.page_size, 1,
                dtype=torch.float32, device=runtime.device,
            )
            quant_val_indices = torch.zeros(
                config.num_hidden_layers, max_blocks_per_seq,
                config.num_key_value_heads, runtime.page_size, config.head_dim // 2,
                dtype=torch.uint8, device=runtime.device,
            )
            quant_val_norms = torch.zeros(
                config.num_hidden_layers, max_blocks_per_seq,
                config.num_key_value_heads, runtime.page_size, 1,
                dtype=torch.float32, device=runtime.device,
            )

        self._pools[model_id] = _CachePool(
            page_size=runtime.page_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_blocks_per_seq=max_blocks_per_seq,
            key_pages=key_pages,
            value_pages=value_pages,
            kv_compressor=kv_compressor,
            kv_quant_config=kv_quant_config,
            quant_key_indices=quant_key_indices,
            quant_key_norms=quant_key_norms,
            quant_val_indices=quant_val_indices,
            quant_val_norms=quant_val_norms,
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

    def block_table_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        """Return a dense ``[batch, max_blocks]`` block table for requests."""
        return block_table_from_block_ids([alloc.page_ids for alloc in allocations])

    def slot_mapping_for_request(self, alloc: KvAllocation, token_index: int | None = None) -> int:
        """Return the physical slot index for a request token."""
        pool = self._pool(alloc.model_id)
        logical_index = alloc.tokens_used if token_index is None else token_index
        return slot_mapping_for_decode(alloc.page_ids, logical_index, pool.page_size)

    def slot_mapping_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        """Return current decode slot mappings for a batch."""
        return torch.tensor(
            [self.slot_mapping_for_request(alloc) for alloc in allocations],
            dtype=torch.int32,
        )

    def slot_mapping_for_positions(
        self,
        alloc: KvAllocation,
        num_tokens: int,
        *,
        max_tokens: int | None = None,
    ) -> torch.Tensor:
        """Return per-position slot mappings, optionally padded with -1."""
        pool = self._pool(alloc.model_id)
        return slot_mapping_for_positions(alloc.page_ids, num_tokens, pool.page_size, max_tokens=max_tokens)

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

    def materialize_full_layer_cache(self, model_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened K/V cache views stacked across every model layer.

        Use this API for fused or L3 decode kernels that select layer i via
        an arithmetic offset (layer_idx * cache_rows_per_layer) on a single
        cache tensor. The pool is already laid out as
        ``[num_layers, num_pages, num_kv_heads, page_size, head_dim]`` so the
        flat view is zero-copy.

        Returns:
            (key_cache_all, value_cache_all) each shaped
            [num_layers * num_pages * num_kv_heads * page_size, head_dim].
        """
        pool = self._pool(model_id)
        return (
            pool.key_pages.reshape(-1, pool.head_dim),
            pool.value_pages.reshape(-1, pool.head_dim),
        )

    def materialize_single_layer_quant_cache(self, model_id: str, layer_idx: int):
        """Return flattened quant K/V cache views for one layer.

        Returns (quant_k, quant_v, quant_k_norms, quant_v_norms) each as 2D
        tensors suitable for kernel arguments. Returns (None, None, None, None)
        if quantization is not enabled.
        """
        pool = self._pool(model_id)
        if pool.quant_key_indices is None:
            return None, None, None, None
        rows_per_layer = pool.max_blocks_per_seq * pool.num_kv_heads * pool.page_size
        return (
            pool.quant_key_indices[layer_idx].reshape(-1, pool.head_dim // 2),
            pool.quant_val_indices[layer_idx].reshape(-1, pool.head_dim // 2),
            pool.quant_key_norms[layer_idx].reshape(-1, 1),
            pool.quant_val_norms[layer_idx].reshape(-1, 1),
        )

    def materialize_full_layer_quant_cache(self, model_id: str):
        """Return flattened quant K/V cache views across all layers.

        Returns (quant_k, quant_v, quant_k_norms, quant_v_norms) each as 2D
        tensors. Returns (None, None, None, None) if quantization is not enabled.
        """
        pool = self._pool(model_id)
        if pool.quant_key_indices is None:
            return None, None, None, None
        return (
            pool.quant_key_indices.reshape(-1, pool.head_dim // 2),
            pool.quant_val_indices.reshape(-1, pool.head_dim // 2),
            pool.quant_key_norms.reshape(-1, 1),
            pool.quant_val_norms.reshape(-1, 1),
        )

    def free(self, alloc: KvAllocation) -> None:
        """Return an allocation's pages to the model pool."""
        self.release_request(alloc.request_id)
        alloc.page_ids.clear()
        alloc.tokens_capacity = 0
        alloc.tokens_used = 0
        # Clean up per-request quantization state.
        self._pool(alloc.model_id).request_quant_states.pop(alloc.request_id, None)

    # ── KV cache quantization ──

    def is_quantization_enabled(self, model_id: str) -> bool:
        """Return whether KV cache quantization is enabled for a model."""
        pool = self._pool(model_id)
        return pool.kv_compressor is not None

    def get_compressed_block_info(self, model_id: str, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cmp_block_table and cmp_num_blocks for the decode_tq kernel.

        The TQ decode kernel reads ALL context blocks from the compressed
        (quantized) cache.  cmp_block_table maps logical block positions to
        physical page IDs in the quant cache (same layout as block_table).
        cmp_num_blocks is the number of context blocks per request
        (ceil(seq_len / page_size)).

        Returns:
            cmp_block_table: [batch * max_blocks_per_seq] INT32 — physical
                block IDs of compressed pages for each request.
            cmp_num_blocks: [batch] INT32 — number of compressed blocks per
                request.
        """
        import torch
        pool = self._pool(model_id)
        max_blocks = pool.max_blocks_per_seq
        actual_batch = len(batch.request_ids) if hasattr(batch, 'request_ids') else len(batch.kv_allocations)
        cmp_block_table = torch.zeros(actual_batch * max_blocks, dtype=torch.int32)
        cmp_num_blocks = torch.zeros(actual_batch, dtype=torch.int32)

        for i, alloc in enumerate(batch.kv_allocations):
            # All blocks written by prefill_tq are compressed.
            # Physical page IDs come from the allocation's page_ids.
            n_pages = len(alloc.page_ids)
            ctx_blocks = min(n_pages, max_blocks)
            cmp_num_blocks[i] = ctx_blocks
            for j in range(ctx_blocks):
                cmp_block_table[i * max_blocks + j] = alloc.page_ids[j]

        return cmp_block_table.share_memory_(), cmp_num_blocks.share_memory_()

    def get_quant_state(self, model_id: str, request_id: str) -> _RequestQuantState:
        """Get or create per-request quantization state."""
        pool = self._pool(model_id)
        if request_id not in pool.request_quant_states:
            pool.request_quant_states[request_id] = _RequestQuantState()
        return pool.request_quant_states[request_id]

    def _layer_key_bits(self, pool: _CachePool, layer_idx: int) -> int:
        """Return the effective key bits for a layer (respects protected layers)."""
        cfg = pool.kv_quant_config
        protected = cfg.protected_layers
        if layer_idx < protected or layer_idx >= pool.num_layers - protected:
            return cfg.protected_bits
        return cfg.key_bits

    def _layer_val_bits(self, pool: _CachePool, layer_idx: int) -> int:
        """Return the effective value bits for a layer (respects protected layers)."""
        cfg = pool.kv_quant_config
        protected = cfg.protected_layers
        if layer_idx < protected or layer_idx >= pool.num_layers - protected:
            return cfg.protected_bits
        return cfg.value_bits

    def compress_to_quant(self, model_id: str, alloc: KvAllocation, npu_runner=None) -> list[int]:
        """Quantize ALL tokens into quant_* buffers and free their BF16 pages.

        DEPRECATED: With prefill_tq writing directly to quant format, this method
        is no longer needed for the prefill path. Kept for potential decode-time
        residual window migration (not yet implemented).
        """
        pool = self._pool(model_id)
        if pool.kv_compressor is None or pool.kv_quant_config is None:
            return []
        if pool.quant_key_indices is None:
            return []
        return []

    def print_kv_cache_memory(self, model_id: str) -> None:
        """Calculate and print KV cache memory usage for a model."""
        pool = self._pool(model_id)

        single_tensor_bytes = pool.key_pages.numel() * pool.key_pages.element_size()
        total_kv_bytes = single_tensor_bytes * 2

        dtype_size = pool.key_pages.element_size()
        per_token_bytes = 2 * pool.num_layers * pool.num_kv_heads * pool.head_dim * dtype_size

        total_pages = pool.key_pages.shape[1]
        free_pages = self.num_free_blocks
        used_pages = total_pages - free_pages
        max_seq_len = pool.max_blocks_per_seq * pool.page_size

        def _fmt(b: int) -> str:
            if b >= 1024 ** 3:
                return f"{b / 1024 ** 3:.2f} GiB"
            return f"{b / 1024 ** 2:.2f} MiB"

        print(f"[KV Cache] model_id = {model_id}", flush=True)
        print(f"[KV Cache] dtype = {pool.key_pages.dtype}, element_size = {dtype_size} bytes", flush=True)
        print(f"[KV Cache] shape per tensor = {list(pool.key_pages.shape)} "
              f"[layers={pool.num_layers}, pages={total_pages}, "
              f"kv_heads={pool.num_kv_heads}, page_size={pool.page_size}, head_dim={pool.head_dim}]", flush=True)
        print(f"[KV Cache] key_pages   = {_fmt(single_tensor_bytes)}", flush=True)
        print(f"[KV Cache] value_pages = {_fmt(single_tensor_bytes)}", flush=True)
        print(f"[KV Cache] total KV cache memory = {_fmt(total_kv_bytes)}", flush=True)
        print(f"[KV Cache] per-token memory = {per_token_bytes} bytes ({per_token_bytes / 1024:.2f} KiB)", flush=True)
        print(f"[KV Cache] pages: total={total_pages}, free={free_pages}, used={used_pages}", flush=True)
        print(f"[KV Cache] max_seq_len per request = {max_seq_len} tokens "
              f"(max_blocks_per_seq={pool.max_blocks_per_seq}, page_size={pool.page_size})", flush=True)

        if pool.kv_compressor is not None:
            stats = self.quantization_stats(model_id)
            quant_bytes = stats.get("quant_buffer_bytes", 0)
            total_extra = quant_bytes
            print(f"[KV Cache] quant buffers = {_fmt(quant_bytes)}", flush=True)
            if total_kv_bytes > 0:
                print(f"[KV Cache] total (bf16 + quant) = {_fmt(total_kv_bytes + total_extra)}", flush=True)
            pages_per_seq_uncompressed = pool.max_blocks_per_seq
            print(
                f"[KV Cache] pages per request: pool={total_pages}, "
                f"uncompressed would need={pages_per_seq_uncompressed} per request",
                flush=True,
            )

    def quantization_stats(self, model_id: str) -> dict:
        """Return memory usage stats for compressed KV cache."""
        pool = self._pool(model_id)
        if pool.kv_compressor is None:
            return {"enabled": False}
        quant_bytes = 0
        if pool.quant_key_indices is not None:
            quant_bytes += pool.quant_key_indices.numel() * pool.quant_key_indices.element_size()
        if pool.quant_key_norms is not None:
            quant_bytes += pool.quant_key_norms.numel() * pool.quant_key_norms.element_size()
        if pool.quant_val_indices is not None:
            quant_bytes += pool.quant_val_indices.numel() * pool.quant_val_indices.element_size()
        if pool.quant_val_norms is not None:
            quant_bytes += pool.quant_val_norms.numel() * pool.quant_val_norms.element_size()
        return {
            "enabled": True,
            "quant_buffer_bytes": quant_bytes,
            "quant_requests": len(pool.request_quant_states),
        }

    def _pool(self, model_id: str) -> _CachePool:
        """Return the registered cache pool for a model."""
        if model_id not in self._pools:
            raise KeyError(f"Model {model_id} is not registered with the KV cache manager.")
        return self._pools[model_id]


# ---------------------------------------------------------------------------
# Standalone utility functions for block-table / slot-mapping computation.
# These are pure math (no allocation state) and can be used directly by
# the serving worker without a KvCacheManager instance.
# ---------------------------------------------------------------------------

def block_table_from_block_ids(block_ids_batch: list[list[int]]) -> torch.Tensor:
    """Return a dense ``[batch, max_blocks]`` int32 block table from physical block IDs."""
    max_blocks = max((len(bids) for bids in block_ids_batch), default=0)
    table = torch.full((len(block_ids_batch), max_blocks), -1, dtype=torch.int32)
    for row, block_ids in enumerate(block_ids_batch):
        if block_ids:
            table[row, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32)
    return table


def slot_mapping_from_block_ids(
    block_ids: list[int],
    token_indices: range | list[int],
    page_size: int,
) -> torch.Tensor:
    """Return physical slot IDs for absolute token indices in one request."""
    slots = []
    for token_index in token_indices:
        page_idx = token_index // page_size
        offset = token_index % page_size
        slots.append(block_ids[page_idx] * page_size + offset)
    return torch.tensor(slots, dtype=torch.int32)


def slot_mapping_from_block_ids_batch(
    block_ids_batch: list[list[int]],
    token_indices_batch: list[range | list[int]],
    page_size: int,
) -> torch.Tensor:
    """Return a padded ``[batch, max_tokens]`` int32 slot mapping from block IDs."""
    mappings = [
        slot_mapping_from_block_ids(block_ids, token_indices, page_size)
        for block_ids, token_indices in zip(block_ids_batch, token_indices_batch, strict=True)
    ]
    max_slots = max((mapping.numel() for mapping in mappings), default=0)
    table = torch.full((len(mappings), max_slots), -1, dtype=torch.int32)
    for row, mapping in enumerate(mappings):
        if mapping.numel() > 0:
            table[row, : mapping.numel()] = mapping
    return table


def slot_mapping_for_decode(page_ids: list[int], tokens_used: int, page_size: int) -> int:
    """Return the single physical slot for a decode step."""
    page_idx = tokens_used // page_size
    offset = tokens_used % page_size
    return page_ids[page_idx] * page_size + offset


def slot_mapping_for_positions(
    page_ids: list[int],
    num_tokens: int,
    page_size: int,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Return per-position slot mappings, optionally padded with -1."""
    size = num_tokens if max_tokens is None else max_tokens
    mapping = torch.full((size,), -1, dtype=torch.int32)
    for token_index in range(num_tokens):
        page_idx = token_index // page_size
        offset = token_index % page_size
        mapping[token_index] = page_ids[page_idx] * page_size + offset
    return mapping
