# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8 dispatch geometry shared by the executor and model runner."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import torch
from pypto.runtime import DeviceTensor

from pypto_serving.config.types import RuntimeConfig
from pypto_serving.model.qwen3_8_27b.cache import (
    QWEN38_ATTENTION_LAYERS,
    QWEN38_GDN_LAYERS,
    QWEN38_KV_PAGE_TOKENS,
    QWEN38_NUM_LAYERS,
)


QWEN38_HIDDEN_SIZE = 5120
QWEN38_VOCAB_SIZE = 248320
QWEN38_KV_WIDTH = 4 * 256
QWEN38_CONV_WIDTH = 10240
QWEN38_GDN_HEADS = 48
QWEN38_GDN_HEAD_DIM = 128
QWEN38_CONV_STATE_LEN = 3
QWEN38_DECODE_TILE = 16
QWEN38_DECODE_MAX_BATCH = 128


def resident_view(tensor: DeviceTensor, offset: int, shape: tuple[int, ...]) -> Any:
    """Name a contiguous subrange of a worker allocation without uploading it again.

    PyPTO dispatch accepts a simpler Tensor backed by the resident Buffer. Its
    byte offset is the only supported way to pass a layer slice of a single
    device allocation to a separately compiled L3 program.
    """
    if tensor.buffer is None:
        raise ValueError("Qwen3.8 resident views require a worker-owned Buffer")
    if offset < 0 or not shape or any(dim <= 0 for dim in shape):
        raise ValueError("Qwen3.8 resident view needs a non-negative offset and positive shape")
    if offset + prod(shape) > prod(tensor.shape):
        raise ValueError("Qwen3.8 resident view exceeds its device allocation")
    from simpler.task_interface import DataType  # noqa: PLC0415

    dtypes = {
        torch.bfloat16: DataType.BFLOAT16,
        torch.float16: DataType.FLOAT16,
        torch.float32: DataType.FLOAT32,
        torch.int8: DataType.INT8,
        torch.int32: DataType.INT32,
    }
    try:
        dtype = dtypes[tensor.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported Qwen3.8 resident view dtype: {tensor.dtype}") from exc
    return tensor.buffer.tensor(shape, dtype, byte_offset=offset * tensor.dtype.itemsize)


@dataclass(frozen=True)
class Qwen38Layout:
    """Shapes and offsets for the two physical cache pools on one NPU."""

    max_seq_len: int
    max_batch_size: int
    prefill_tokens: int
    num_kv_pages: int

    def __post_init__(self) -> None:
        if not 0 < self.max_batch_size <= QWEN38_DECODE_MAX_BATCH:
            raise ValueError("Qwen3.8 max_batch_size must be in [1, 128]")
        if not 0 < self.prefill_tokens <= self.max_seq_len:
            raise ValueError("Qwen3.8 prefill_tokens must be in [1, max_seq_len]")
        if self.num_kv_pages < self.max_blocks_per_seq:
            raise ValueError("Qwen3.8 KV pool cannot hold one maximum-length sequence")
        if self.num_kv_pages % self.max_blocks_per_seq:
            raise ValueError("Qwen3.8 KV pages must be a multiple of max_blocks_per_seq")

    @property
    def max_blocks_per_seq(self) -> int:
        return (self.max_seq_len + QWEN38_KV_PAGE_TOKENS - 1) // QWEN38_KV_PAGE_TOKENS

    @property
    def capacity_slots(self) -> int:
        return self.num_kv_pages // self.max_blocks_per_seq

    @property
    def max_prefill_rows(self) -> int:
        # Every owner must contribute at least one complete attention tile.
        return min(self.max_batch_size, self.prefill_tokens // 256)

    @property
    def decode_batch_pad(self) -> int:
        return ((self.max_batch_size + QWEN38_DECODE_TILE - 1) // QWEN38_DECODE_TILE) * QWEN38_DECODE_TILE

    @property
    def kv_layer_stride(self) -> int:
        # Inactive decode rows need distinct pages so parallel appends never
        # race with a live request or with one another.
        return (self.num_kv_pages + self.decode_batch_pad) * QWEN38_KV_PAGE_TOKENS

    @property
    def kv_pool_shape(self) -> tuple[int, int]:
        return (len(QWEN38_ATTENTION_LAYERS) * self.kv_layer_stride, QWEN38_KV_WIDTH)

    @property
    def conv_pool_shape(self) -> tuple[int, int, int]:
        return ((self.capacity_slots + 1) * len(QWEN38_GDN_LAYERS) + self.decode_batch_pad,
                QWEN38_CONV_STATE_LEN, QWEN38_CONV_WIDTH)

    @property
    def recurrent_pool_shape(self) -> tuple[int, int, int, int]:
        return ((self.capacity_slots + 1) * len(QWEN38_GDN_LAYERS) + self.decode_batch_pad,
                QWEN38_GDN_HEADS, QWEN38_GDN_HEAD_DIM, QWEN38_GDN_HEAD_DIM)

    @property
    def prefill_scratch_state_base(self) -> int:
        return self.capacity_slots * len(QWEN38_GDN_LAYERS)

    def decode_scratch_state_row(self, batch_row: int) -> int:
        if not 0 <= batch_row < self.decode_batch_pad:
            raise IndexError(f"decode row {batch_row} outside [0, {self.decode_batch_pad})")
        return self.prefill_scratch_state_base + len(QWEN38_GDN_LAYERS) + batch_row

    def decode_scratch_kv_page(self, batch_row: int) -> int:
        if not 0 <= batch_row < self.decode_batch_pad:
            raise IndexError(f"decode row {batch_row} outside [0, {self.decode_batch_pad})")
        return self.num_kv_pages + batch_row

    def state_row(self, block_id: int, global_layer: int) -> int:
        """Map one scheduler state block to the kernel's flattened layer row."""
        if not 0 <= block_id < self.capacity_slots:
            raise IndexError(f"state block {block_id} outside [0, {self.capacity_slots})")
        try:
            local = QWEN38_GDN_LAYERS.index(global_layer)
        except ValueError as exc:
            raise ValueError(f"layer {global_layer} is not a GDN layer") from exc
        return block_id * len(QWEN38_GDN_LAYERS) + local

    def attention_layer_base(self, global_layer: int) -> int:
        """Return the first K/V row for an attention layer's physical page pool."""
        try:
            local = QWEN38_ATTENTION_LAYERS.index(global_layer)
        except ValueError as exc:
            raise ValueError(f"layer {global_layer} is not a full-attention layer") from exc
        return local * self.kv_layer_stride


@dataclass(frozen=True)
class Qwen38CompiledKernels:
    """Kernel handles, staged weights and dimensions produced by the executor."""

    prefill_by_rows: dict[int, Any]
    decode_fwd_by_rows: dict[int, Any]
    decode_embedding: Any
    final_head: Any
    weights: dict[str, torch.Tensor]
    globals: dict[str, torch.Tensor]
    layout: Qwen38Layout
    kernel_dir: str
    prefill_ring_heap: tuple[int, ...] = ()


def layout_from_runtime(runtime: RuntimeConfig, *, num_kv_pages: int) -> Qwen38Layout:
    """Freeze dispatch shapes after the device has reported cache capacity."""
    if len(runtime.kv_cache_groups) != 2:
        raise ValueError("Qwen3.8 runtime requires the two declared cache groups")
    kv, state = runtime.kv_cache_groups
    if kv.name != "attn_kv" or state.name != "gdn_state":
        raise ValueError("Qwen3.8 cache groups must be attn_kv, gdn_state in that order")
    if kv.max_blocks_per_seq != (runtime.max_seq_len + 127) // 128 or state.max_blocks_per_seq != 1:
        raise ValueError("Qwen3.8 cache group extents disagree with max_seq_len")
    return Qwen38Layout(
        max_seq_len=runtime.max_seq_len,
        max_batch_size=runtime.max_batch_size,
        prefill_tokens=min(runtime.max_prefill_tokens_per_request or runtime.max_num_batched_tokens,
                           runtime.max_seq_len),
        num_kv_pages=num_kv_pages,
    )


assert len(QWEN38_GDN_LAYERS) + len(QWEN38_ATTENTION_LAYERS) == QWEN38_NUM_LAYERS
