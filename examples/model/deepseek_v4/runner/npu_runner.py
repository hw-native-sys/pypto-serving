# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from pypto.runtime import DeviceTensor

from examples.model.deepseek_v4.runner.weight_loader import DeepSeekV4WeightStore
from examples.model.deepseek_v4.runner.weight_loader import DeepSeekV4GlobalWeights
from examples.model.deepseek_v4.runner.weight_loader import DeepSeekV4StackedLayerWeights
from python.core.model_runner import ModelRunner
from python.core.types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)


DEEPSEEK_V4_RANKS = 8
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_BLOCK_SIZE = 128
DEEPSEEK_V4_DECODE_BATCH = 64
DEEPSEEK_V4_DECODE_SEQ = 2
DEEPSEEK_V4_DECODE_TOKENS = DEEPSEEK_V4_DECODE_BATCH * DEEPSEEK_V4_DECODE_SEQ
DEEPSEEK_V4_PREFILL_BATCH = 1
DEEPSEEK_V4_PREFILL_SEQ = 128
DEEPSEEK_V4_ORI_MAX_BLOCKS = 1
DEEPSEEK_V4_CMP_MAX_BLOCKS = 8
DEEPSEEK_V4_IDX_MAX_BLOCKS = 64
DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS = 64
DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_C128_STATE_BLOCK_SIZE = 8
DEEPSEEK_V4_C4_STATE_BLOCK_SIZE = 4
DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS = 2
DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS = 2
DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS = 1024
DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS = 2048
DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS = 2048
DEEPSEEK_V4_INDEX_TOPK = 512
DEEPSEEK_V4_PREFILL_SPARSE_TOPK = DEEPSEEK_V4_BLOCK_SIZE + DEEPSEEK_V4_INDEX_TOPK
DEEPSEEK_V4_HEAD_DIM = 512
DEEPSEEK_V4_IDX_HEAD_DIM = 128
DEEPSEEK_V4_HCA_MAIN_OUT_DIM = 512
DEEPSEEK_V4_CSA_MAIN_OUT_DIM = 1024
DEEPSEEK_V4_CSA_INNER_OUT_DIM = 256
DEEPSEEK_V4_HCA_STATE_DIM = 2 * DEEPSEEK_V4_HCA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_STATE_DIM = 2 * DEEPSEEK_V4_CSA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_INNER_STATE_DIM = 2 * DEEPSEEK_V4_CSA_INNER_OUT_DIM
DEEPSEEK_V4_RMS_NORM_EPS = 1e-6
DEEPSEEK_V4_HC_EPS = 1e-6
# Layer-stacking counts for the packed all-layer decode_fwd kernel.
DEEPSEEK_V4_FWD_NUM_LAYERS = 43
DEEPSEEK_V4_CSA_NUM_LAYERS = 21
DEEPSEEK_V4_HCA_NUM_LAYERS = 20


# Argument order for the packed all-43-layer ``l3_prefill_fwd`` kernel. This
# mirrors pypto-lib prefill_fwd.py ``prefill_fwd_rank_core`` / ``l3_prefill_fwd``
# host signature: every layer-stacked weight/state tensor in core-parameter order,
# then ``x_out`` and a trailing ``num_tokens`` scalar. The work caches
# (kv_cache/cmp_kv/idx_kv_cache) are kernel ``pl.Out`` tensors; weights and
# metadata are inputs.
_PREFILL_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_cmp_kv_state",
    "hca_cmp_score_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_cmp_kv_state",
    "csa_cmp_score_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_kv_state",
    "csa_inner_score_state",
    "csa_inner_compress_state_block_table",
    "kv_cache",
    "ori_block_table",
    "ori_slot_mapping",
    "cmp_kv",
    "cmp_block_table",
    "cmp_sparse_indices",
    "cmp_sparse_lens",
    "idx_kv_cache",
    "idx_block_table",
    "position_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "input_ids",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "x_out",
)

# Argument order for the packed all-43-layer ``l3_decode_fwd`` kernel. This
# mirrors pypto-lib decode_fwd.py ``build_tensor_specs`` ``ordered_names`` (the
# host signature follows it exactly, with ``x_out`` appended as the last output).
_DECODE_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_hadamard_idx",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "cmp_kv",
    "idx_kv_cache",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "freqs_cos",
    "freqs_sin",
    "block_table",
    "ori_slot_mapping",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "cmp_block_table",
    "idx_block_table",
    "input_ids",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "x_out",
)

_DECODE_INPUT_TENSOR_FIELDS = (
    "input_ids",
    "position_ids",
    "kv_seq_lens",
    "block_table",
    "ori_slot_mapping",
    "cmp_block_table",
    "idx_block_table",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
)


@dataclass(frozen=True)
class DeepSeekV4CacheLayout:
    """Static cache layout baked into the current DeepSeekV4 kernels."""

    ranks: int = DEEPSEEK_V4_RANKS
    hc_mult: int = DEEPSEEK_V4_HC_MULT
    block_size: int = DEEPSEEK_V4_BLOCK_SIZE
    decode_batch: int = DEEPSEEK_V4_DECODE_BATCH
    decode_seq: int = DEEPSEEK_V4_DECODE_SEQ
    decode_tokens: int = DEEPSEEK_V4_DECODE_TOKENS
    prefill_batch: int = DEEPSEEK_V4_PREFILL_BATCH
    prefill_seq: int = DEEPSEEK_V4_PREFILL_SEQ
    ori_max_blocks: int = DEEPSEEK_V4_ORI_MAX_BLOCKS
    cmp_max_blocks: int = DEEPSEEK_V4_CMP_MAX_BLOCKS
    idx_max_blocks: int = DEEPSEEK_V4_IDX_MAX_BLOCKS
    hca_state_max_blocks: int = DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS
    csa_state_max_blocks: int = DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS
    csa_inner_state_max_blocks: int = DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS
    c128_state_block_size: int = DEEPSEEK_V4_C128_STATE_BLOCK_SIZE
    c4_state_block_size: int = DEEPSEEK_V4_C4_STATE_BLOCK_SIZE
    prefill_cmp_max_blocks: int = DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS
    prefill_idx_max_blocks: int = DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS
    prefill_hca_state_max_blocks: int = DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS
    prefill_csa_state_max_blocks: int = DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS
    prefill_csa_inner_state_max_blocks: int = DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS
    prefill_sparse_topk: int = DEEPSEEK_V4_PREFILL_SPARSE_TOPK

    def validate_runtime(self, config: ModelConfig, runtime: RuntimeConfig, device_ids: Sequence[int]) -> None:
        """Validate serving/runtime options against kernel-fixed dimensions."""
        if len(device_ids) != self.ranks:
            raise ValueError(f"DeepSeekV4 requires exactly {self.ranks} devices, got {len(device_ids)}")
        if runtime.page_size != self.block_size:
            raise ValueError(f"DeepSeekV4 kernels require page_size={self.block_size}, got {runtime.page_size}")
        if runtime.max_batch_size > self.decode_batch:
            raise ValueError(
                f"DeepSeekV4 decode kernels support at most {self.decode_batch} active rows, "
                f"got max_batch_size={runtime.max_batch_size}"
            )
        decode_state_capacity = self.csa_state_max_blocks * self.c4_state_block_size
        if runtime.max_seq_len > decode_state_capacity:
            raise ValueError(
                "DeepSeekV4 pypto-lib decode CSA state tables currently support at most "
                f"max_seq_len={decode_state_capacity}, got {runtime.max_seq_len}. "
                "Increase the decode CSA state table depth in pypto-lib before serving longer contexts."
            )
        if self.decode_tokens != self.decode_batch * self.decode_seq:
            raise ValueError("DeepSeekV4 layout decode_tokens must equal decode_batch * decode_seq")
        expected = {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "vocab_size": 129280,
        }
        actual = {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
        }
        if actual != expected:
            mismatch = ", ".join(f"{name}={actual[name]} expected {value}" for name, value in expected.items())
            raise ValueError("DeepSeekV4 W8A8 kernels require Flash shape: " + mismatch)


@dataclass
class DeepSeekV4CacheManager:
    """Request-to-cache-slot mapping and table builders for DeepSeekV4 kernels."""

    layout: DeepSeekV4CacheLayout = field(default_factory=DeepSeekV4CacheLayout)
    _request_to_slot: dict[str, int] = field(default_factory=dict)
    _free_slots: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._free_slots:
            self._free_slots = list(range(self.layout.decode_batch))

    @property
    def active_slots(self) -> dict[str, int]:
        """Return a copy of currently assigned request slots."""
        return dict(self._request_to_slot)

    @property
    def free_count(self) -> int:
        """Return the number of unassigned decode slots."""
        return len(self._free_slots)

    def allocate(self, request_id: str) -> int | None:
        """Assign a stable decode slot to ``request_id``."""
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            return None
        slot = self._free_slots.pop(0)
        self._request_to_slot[request_id] = slot
        return slot

    def release(self, request_ids: Iterable[str]) -> None:
        """Release slots held by finished or aborted requests."""
        for request_id in request_ids:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is not None and slot not in self._free_slots:
                self._free_slots.append(slot)
        self._free_slots.sort()

    def slots_for_request_ids(self, request_ids: Sequence[str]) -> list[int]:
        """Return assigned slots for request ids, allocating missing slots."""
        slots = []
        for request_id in request_ids:
            slot = self.allocate(request_id)
            if slot is None:
                raise RuntimeError("DeepSeekV4 cache slots exhausted")
            slots.append(slot)
        return slots

    def block_table(self, slots: Sequence[int], *, max_blocks: int) -> torch.Tensor:
        """Build a row-major block table for request-owned physical block ranges."""
        table = torch.empty((len(slots), max_blocks), dtype=torch.int32)
        for row, slot in enumerate(slots):
            start = int(slot) * max_blocks
            table[row].copy_(torch.arange(start, start + max_blocks, dtype=torch.int32))
        return table

    def slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        block_size: int | None = None,
        compress_ratio: int = 1,
    ) -> torch.Tensor:
        """Map logical token positions to physical cache rows for each request slot."""
        block_size = self.layout.block_size if block_size is None else int(block_size)
        if compress_ratio <= 0:
            raise ValueError("compress_ratio must be positive")
        capacity = max_blocks * block_size
        max_tokens = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(slots), max_tokens), -1, dtype=torch.int64)
        for row, (slot, row_positions) in enumerate(zip(slots, positions, strict=True)):
            base = int(slot) * capacity
            for col, position in enumerate(row_positions):
                logical = int(position) // compress_ratio
                if logical >= capacity:
                    raise ValueError(
                        f"position {position} maps to logical cache row {logical}, "
                        f"but capacity is {capacity}"
                    )
                mapping[row, col] = base + logical
        return mapping

    def block_table_for_kernel_rows(
        self,
        slots: Sequence[int],
        *,
        max_blocks: int,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Build a fixed-row block table, replicating row 0 into inactive rows."""
        if not slots:
            raise ValueError("slots must not be empty")
        active = self.block_table(slots, max_blocks=max_blocks)
        return self.replicate_first_row(active, actual_rows=len(slots), kernel_rows=kernel_rows)

    def sliding_window_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Map absolute positions into the 128-token ori sliding-window cache."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        for row_idx, (slot, row_positions) in enumerate(rows):
            base = int(slot) * self.layout.ori_max_blocks * self.layout.block_size
            for col, position in enumerate(row_positions):
                window_slot = int(position) % self.layout.block_size
                mapping[row_idx, col] = base + window_slot
        return mapping

    def compressed_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        compress_ratio: int,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Map compression-boundary positions into a compressed KV cache."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        capacity = max_blocks * self.layout.block_size
        for row_idx, (slot, row_positions) in enumerate(rows):
            base = int(slot) * capacity
            for col, position in enumerate(row_positions):
                position = int(position)
                if (position + 1) % compress_ratio != 0:
                    continue
                logical = position // compress_ratio
                if logical >= capacity:
                    raise ValueError(
                        f"position {position} maps to compressed row {logical}, "
                        f"but capacity is {capacity}"
                    )
                mapping[row_idx, col] = base + logical
        return mapping

    def state_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        state_block_size: int,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Map absolute token positions into a compressor-state cache."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        capacity = max_blocks * state_block_size
        for row_idx, (slot, row_positions) in enumerate(rows):
            base = int(slot) * capacity
            for col, position in enumerate(row_positions):
                position = int(position)
                if position >= capacity:
                    raise ValueError(
                        f"position {position} exceeds compressor-state capacity {capacity} "
                        f"(max_blocks={max_blocks}, state_block_size={state_block_size})"
                    )
                mapping[row_idx, col] = base + position
        return mapping

    @staticmethod
    def _replicated_slots_and_positions(
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        kernel_rows: int,
    ) -> list[tuple[int, Sequence[int]]]:
        if not slots:
            raise ValueError("slots must not be empty")
        if len(slots) != len(positions):
            raise ValueError("slots and positions must have the same active row count")
        if len(slots) > kernel_rows:
            raise ValueError("active rows exceed kernel_rows")
        rows = [(int(slot), tuple(int(pos) for pos in row)) for slot, row in zip(slots, positions, strict=True)]
        rows.extend((rows[0][0], rows[0][1]) for _ in range(kernel_rows - len(rows)))
        return rows

    @staticmethod
    def replicate_first_row(tensor: torch.Tensor, *, actual_rows: int, kernel_rows: int) -> torch.Tensor:
        """Pad kernel inputs by replicating row 0 into inactive rows."""
        if actual_rows <= 0:
            raise ValueError("actual_rows must be positive")
        if kernel_rows < actual_rows:
            raise ValueError("kernel_rows must be >= actual_rows")
        if tensor.shape[0] < actual_rows:
            raise ValueError("tensor has fewer rows than actual_rows")
        out = torch.empty((kernel_rows, *tensor.shape[1:]), dtype=tensor.dtype)
        out[:actual_rows].copy_(tensor[:actual_rows])
        if actual_rows < kernel_rows:
            out[actual_rows:].copy_(tensor[0:1].expand(kernel_rows - actual_rows, *tensor.shape[1:]))
        return out


class DeepSeekV4InputBuilder:
    """Build fixed-shape host inputs for DeepSeekV4 HC-stack kernels."""

    def __init__(self, *, layout: DeepSeekV4CacheLayout, hidden_size: int) -> None:
        self.layout = layout
        self.hidden_size = int(hidden_size)

    def prefill_x_hc(self, embeddings: torch.Tensor, *, actual_tokens: int) -> torch.Tensor:
        """Build ``[ranks, 128, hc_mult, hidden]`` prefill HC input."""
        if embeddings.ndim != 2:
            raise ValueError(f"prefill embeddings must be rank-2, got shape={tuple(embeddings.shape)}")
        return self._x_hc_from_rows(
            embeddings,
            actual_tokens=actual_tokens,
            token_rows=self.layout.prefill_seq,
        )

    def decode_x_hc(self, embeddings: torch.Tensor, *, actual_batch: int) -> torch.Tensor:
        """Build ``[ranks, 128, hc_mult, hidden]`` decode HC input.

        Current DeepSeekV4 decode kernels use a fixed 64-token contract. Serving
        generation only samples from active request rows, but the fused decode
        layer still expects every fixed row to carry valid token data.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"decode embeddings must be rank-2, got shape={tuple(embeddings.shape)}")
        if actual_batch <= 0:
            raise ValueError("actual_batch must be positive")
        if actual_batch > self.layout.decode_batch:
            raise ValueError(
                f"actual_batch={actual_batch} exceeds decode batch capacity {self.layout.decode_batch}"
            )
        if embeddings.shape[0] < actual_batch:
            raise ValueError("decode embeddings has fewer rows than actual_batch")
        rows = torch.zeros(
            (self.layout.decode_tokens, self.hidden_size),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        for row in range(self.layout.decode_batch):
            source_row = row if row < actual_batch else 0
            start = row * self.layout.decode_seq
            rows[start : start + self.layout.decode_seq].copy_(
                embeddings[source_row : source_row + 1].expand(self.layout.decode_seq, self.hidden_size)
            )
        return self._expand_hc_and_ranks(rows)

    def _x_hc_from_rows(
        self,
        embeddings: torch.Tensor,
        *,
        actual_tokens: int,
        token_rows: int,
    ) -> torch.Tensor:
        if actual_tokens <= 0:
            raise ValueError("actual_tokens must be positive")
        if actual_tokens > token_rows:
            raise ValueError(f"actual_tokens={actual_tokens} exceeds token row capacity {token_rows}")
        if embeddings.shape[0] < actual_tokens:
            raise ValueError("embeddings has fewer rows than actual_tokens")
        if int(embeddings.shape[1]) != self.hidden_size:
            raise ValueError(f"embedding hidden size must be {self.hidden_size}, got {int(embeddings.shape[1])}")
        rows = torch.zeros((token_rows, self.hidden_size), dtype=embeddings.dtype, device=embeddings.device)
        rows[:actual_tokens].copy_(embeddings[:actual_tokens])
        return self._expand_hc_and_ranks(rows)

    def _expand_hc_and_ranks(self, rows: torch.Tensor) -> torch.Tensor:
        return (
            rows.unsqueeze(1)
            .expand(rows.shape[0], self.layout.hc_mult, self.hidden_size)
            .unsqueeze(0)
            .expand(self.layout.ranks, rows.shape[0], self.layout.hc_mult, self.hidden_size)
            .contiguous()
        )


@dataclass
class DeepSeekV4L3Callable:
    """Compiled HOST-dispatched DeepSeekV4 program."""

    compiled: object
    name: str


@dataclass
class _StaticDeviceTensor:
    """CPU tensor marker uploaded to the shared worker once."""

    tensor: torch.Tensor


@dataclass
class _TransientDeviceTensor:
    """CPU tensor marker uploaded for one layer dispatch and then freed."""

    tensor: torch.Tensor


@dataclass
class DeepSeekV4LayerCache:
    """Shared decode work-cache tensors for one DeepSeekV4 layer dispatch."""

    kv_cache: torch.Tensor
    cmp_kv: torch.Tensor
    idx_kv_cache: torch.Tensor
    hca_compress_state: torch.Tensor
    csa_compress_state: torch.Tensor
    csa_inner_compress_state: torch.Tensor


@dataclass
class DeepSeekV4LayerCacheSnapshot:
    """Compact parent-side cache snapshot captured after prefill for one layer."""

    tensors: dict[str, torch.Tensor]


@dataclass
class DeepSeekV4CompiledKernels:
    """Compiled-kernel placeholder and immutable DeepSeekV4 runtime metadata."""

    layout: DeepSeekV4CacheLayout
    model_dir: str
    weight_map: dict[str, str]
    weight_store: DeepSeekV4WeightStore
    compress_ratios: tuple[int, ...]
    layer_plan: tuple["DeepSeekV4LayerPlan", ...]
    kernel_dir: str
    prefill: DeepSeekV4L3Callable | None = None
    decode: DeepSeekV4L3Callable | None = None
    lm_head: DeepSeekV4L3Callable | None = None
    freqs_cos: torch.Tensor | None = None
    freqs_sin: torch.Tensor | None = None
    platform: str = "a2a3"
    device_id: int = 0
    n_routed_experts: int = 256
    num_hash_layers: int = 3

    def l3_callables(self) -> tuple[DeepSeekV4L3Callable, ...]:
        """Return every compiled L3 program that the shared worker may run."""
        callables: list[DeepSeekV4L3Callable] = []
        if self.prefill is not None:
            callables.append(self.prefill)
        if self.decode is not None:
            callables.append(self.decode)
        if self.lm_head is not None:
            callables.append(self.lm_head)
        return tuple(callables)


@dataclass(frozen=True)
class DeepSeekV4PreparedPrefillInputs:
    """Fixed-shape host tensors derived from one serving prefill chunk."""

    request_id: str
    slot: int
    actual_tokens: int
    x_hc: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    ori_block_table: torch.Tensor
    ori_slot_mapping: torch.Tensor
    cmp_block_table: torch.Tensor
    idx_block_table: torch.Tensor
    hca_compress_state_block_table: torch.Tensor
    csa_compress_state_block_table: torch.Tensor
    csa_inner_compress_state_block_table: torch.Tensor
    hca_cmp_slot_mapping: torch.Tensor
    hca_state_slot_mapping: torch.Tensor
    csa_cmp_slot_mapping: torch.Tensor
    csa_idx_slot_mapping: torch.Tensor
    csa_state_slot_mapping: torch.Tensor
    csa_inner_state_slot_mapping: torch.Tensor
    cmp_sparse_indices_by_ratio: dict[int, torch.Tensor]
    cmp_sparse_lens_by_ratio: dict[int, torch.Tensor]

    def sparse_inputs_for_ratio(self, compress_ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prefill sparse-attention inputs for one layer compression ratio."""
        ratio = int(compress_ratio)
        return self.cmp_sparse_indices_by_ratio[ratio], self.cmp_sparse_lens_by_ratio[ratio]


@dataclass(frozen=True)
class DeepSeekV4PreparedDecodeInputs:
    """Fixed-shape host tensors derived from one decode scheduler batch."""

    request_ids: tuple[str, ...]
    slots: tuple[int, ...]
    kernel_slots: tuple[int, ...]
    actual_batch: int
    x_hc: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_table: torch.Tensor
    ori_slot_mapping: torch.Tensor
    cmp_block_table: torch.Tensor
    idx_block_table: torch.Tensor
    hca_compress_state_block_table: torch.Tensor
    csa_compress_state_block_table: torch.Tensor
    csa_inner_compress_state_block_table: torch.Tensor
    hca_cmp_slot_mapping: torch.Tensor
    hca_state_slot_mapping: torch.Tensor
    csa_cmp_slot_mapping: torch.Tensor
    csa_idx_slot_mapping: torch.Tensor
    csa_state_slot_mapping: torch.Tensor
    csa_inner_state_slot_mapping: torch.Tensor


@dataclass
class _DeepSeekV4DecodeSharedBuffers:
    """Reusable decode shared-memory buffers inherited by the L3 chip workers."""

    x_hc_a: torch.Tensor
    x_hc_b: torch.Tensor
    x_out: torch.Tensor
    tensors: dict[str, torch.Tensor]


@dataclass
class _DeepSeekV4PrefillFwdSharedBuffers:
    """Reusable packed-prefill shared buffers inherited by the L3 chip workers.

    Every tensor is layer-stacked for the single ``l3_prefill_fwd`` dispatch: FWD
    metadata/work caches stack across all 43 hidden layers, CSA-group state across
    the 21 compress_ratio==4 layers, HCA-group state across the 20
    compress_ratio==128 layers. ``tensors`` is keyed by ``_PREFILL_FWD_TENSOR_ORDER``
    name (excluding the stacked weights, which live in ``_stacked_weight_buffers``,
    and ``freqs_*``/``x_hc``/``x_out`` which are tracked explicitly).
    """

    x_hc: torch.Tensor
    x_out: torch.Tensor
    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor
    tensors: dict[str, torch.Tensor]


@dataclass(frozen=True)
class DeepSeekV4LayerPlan:
    """Per-layer execution metadata for DeepSeekV4 serving."""

    layer_id: int
    compress_ratio: int
    attention_kind: str
    include_tid2eid: bool
    include_gate_bias: bool


def deepseek_v4_attention_kind(compress_ratio: int) -> str:
    """Return the DeepSeekV4 attention family for a compression ratio."""
    if compress_ratio == 0:
        return "swa"
    if compress_ratio == 128:
        return "hca"
    if compress_ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeekV4 attention compress ratio: {compress_ratio}")


def build_deepseek_v4_layer_plan(
    *,
    compress_ratios: Sequence[int],
    num_hidden_layers: int,
    num_hash_layers: int,
) -> tuple[DeepSeekV4LayerPlan, ...]:
    """Build the per-layer serving plan from config metadata."""
    if len(compress_ratios) < num_hidden_layers:
        raise ValueError("compress_ratios must include at least one entry per hidden layer")
    return tuple(
        DeepSeekV4LayerPlan(
            layer_id=layer_id,
            compress_ratio=int(compress_ratios[layer_id]),
            attention_kind=deepseek_v4_attention_kind(int(compress_ratios[layer_id])),
            include_tid2eid=layer_id < num_hash_layers,
            include_gate_bias=layer_id >= num_hash_layers,
        )
        for layer_id in range(num_hidden_layers)
    )


class DeepSeekV4ModelRunner(ModelRunner):
    """Runner boundary for DeepSeekV4 W8A8 kernels and model-specific caches."""

    def __init__(self, *, compiled: DeepSeekV4CompiledKernels) -> None:
        super().__init__()
        self._compiled = compiled
        self.cache_manager = DeepSeekV4CacheManager(layout=compiled.layout)
        self.input_builder: DeepSeekV4InputBuilder | None = None
        self._l3_worker: Any | None = None
        self._l3_static_tensors: dict[tuple[int, tuple[int, ...], torch.dtype], DeviceTensor] = {}
        self._decode_work_cache: DeepSeekV4LayerCache | None = None
        self._prefill_cache_snapshots: dict[int, DeepSeekV4LayerCacheSnapshot] = {}
        self._global_weights: DeepSeekV4GlobalWeights | None = None
        self._static_lm_head_weight: torch.Tensor | None = None
        self._static_freqs_cos: torch.Tensor | None = None
        self._static_freqs_sin: torch.Tensor | None = None
        self._prefill_fwd_buffers: _DeepSeekV4PrefillFwdSharedBuffers | None = None
        self._decode_buffers: _DeepSeekV4DecodeSharedBuffers | None = None
        self._stacked_weight_buffers: dict[str, torch.Tensor] | None = None
        self._hc_head_buffers: dict[str, torch.Tensor] | None = None
        self._lm_head_hidden_buffer: torch.Tensor | None = None
        self._lm_head_logits_buffer: torch.Tensor | None = None

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> None:
        """DeepSeekV4 owns several cache families, so generic KV allocation is bypassed."""
        self.input_builder = DeepSeekV4InputBuilder(
            layout=self._compiled.layout,
            hidden_size=config.hidden_size,
        )
        return None

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Release runner-owned cache slots for finished requests."""
        request_ids = tuple(request_ids)
        self.cache_manager.release(request_ids)
        if request_ids:
            self._prefill_cache_snapshots.clear()

    def load_packed_global_weights(self) -> DeepSeekV4GlobalWeights:
        """Load global tensors and pack the LM head for the 8-way TP kernel."""
        if self._global_weights is None:
            self._global_weights = self._compiled.weight_store.load_packed_global_weights(
                ranks=self._compiled.layout.ranks
            )
            self._ensure_shared_host_allocation_before_worker("lm_head_weight")
            self._static_lm_head_weight = self._static_device_tensor(self._global_weights.lm_head_weight)
        return self._global_weights

    def load_stacked_layer_weights(self) -> DeepSeekV4StackedLayerWeights:
        """Load and stack all hidden-layer weights for the packed decode_fwd kernel."""
        compress_ratios = tuple(int(layer.compress_ratio) for layer in self._compiled.layer_plan)
        return self._compiled.weight_store.load_stacked_layer_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=self._compiled.num_hash_layers,
        )

    def prepare_prefill_inputs(self, model: RuntimeModel, batch: PrefillBatch) -> DeepSeekV4PreparedPrefillInputs:
        """Build DeepSeekV4 prefill host inputs for the current scheduler chunk."""
        builder = self._require_input_builder()
        layout = self._compiled.layout
        if len(batch.request_ids) != layout.prefill_batch:
            raise ValueError(
                f"DeepSeekV4 prefill kernels support exactly {layout.prefill_batch} request per dispatch, "
                f"got {len(batch.request_ids)}"
            )
        request_id = batch.request_ids[0]
        slot = self.cache_manager.allocate(request_id)
        if slot is None:
            raise RuntimeError("DeepSeekV4 cache slots exhausted")

        actual_tokens = self._prefill_actual_tokens(batch)
        positions = self._prefill_positions(batch, actual_tokens)
        if positions[-1] >= model.runtime.max_seq_len:
            raise ValueError(
                f"prefill position {positions[-1]} exceeds max_seq_len={model.runtime.max_seq_len}"
            )
        embeddings = batch.input_embeddings[0, :actual_tokens].to(torch.bfloat16).cpu()
        token_ids = batch.token_ids[0, :actual_tokens].detach().cpu().to(torch.long)
        sparse_by_ratio = self._prefill_sparse_by_ratio(positions, actual_tokens)

        return DeepSeekV4PreparedPrefillInputs(
            request_id=request_id,
            slot=slot,
            actual_tokens=actual_tokens,
            x_hc=builder.prefill_x_hc(embeddings, actual_tokens=actual_tokens),
            input_ids=self._rank_stack(self._padded_vector(token_ids, layout.prefill_seq, dtype=torch.long)),
            position_ids=self._rank_stack(self._prefill_position_ids(positions, layout.prefill_seq)),
            ori_block_table=self._rank_stack(
                self.cache_manager.block_table([slot], max_blocks=layout.ori_max_blocks)[0]
            ),
            ori_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self.cache_manager.sliding_window_slot_mapping(
                        [slot],
                        [positions],
                        kernel_rows=layout.prefill_batch,
                    )[0],
                    layout.prefill_seq,
                )
            ),
            cmp_block_table=self._rank_stack(
                self.cache_manager.block_table([slot], max_blocks=layout.prefill_cmp_max_blocks)[0]
            ),
            idx_block_table=self._rank_stack(
                self.cache_manager.block_table([slot], max_blocks=layout.prefill_idx_max_blocks)[0]
            ),
            hca_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table([slot], max_blocks=layout.prefill_hca_state_max_blocks)[0]
            ),
            csa_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table([slot], max_blocks=layout.prefill_csa_state_max_blocks)[0]
            ),
            csa_inner_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table([slot], max_blocks=layout.prefill_csa_inner_state_max_blocks)[0]
            ),
            hca_cmp_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self.cache_manager.compressed_slot_mapping(
                        [slot],
                        [positions],
                        max_blocks=layout.prefill_cmp_max_blocks,
                        compress_ratio=128,
                        kernel_rows=layout.prefill_batch,
                    )[0],
                    layout.prefill_seq,
                )
            ),
            hca_state_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self.cache_manager.state_slot_mapping(
                        [slot],
                        [positions],
                        max_blocks=layout.prefill_hca_state_max_blocks,
                        state_block_size=layout.c128_state_block_size,
                        kernel_rows=layout.prefill_batch,
                    )[0],
                    layout.prefill_seq,
                )
            ),
            csa_cmp_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self.cache_manager.compressed_slot_mapping(
                        [slot],
                        [positions],
                        max_blocks=layout.prefill_cmp_max_blocks,
                        compress_ratio=4,
                        kernel_rows=layout.prefill_batch,
                    )[0],
                    layout.prefill_seq,
                )
            ),
            csa_idx_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self.cache_manager.compressed_slot_mapping(
                        [slot],
                        [positions],
                        max_blocks=layout.prefill_idx_max_blocks,
                        compress_ratio=4,
                        kernel_rows=layout.prefill_batch,
                    )[0],
                    layout.prefill_seq,
                )
            ),
            csa_state_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self.cache_manager.state_slot_mapping(
                        [slot],
                        [positions],
                        max_blocks=layout.prefill_csa_state_max_blocks,
                        state_block_size=layout.c4_state_block_size,
                        kernel_rows=layout.prefill_batch,
                    )[0],
                    layout.prefill_seq,
                )
            ),
            csa_inner_state_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self.cache_manager.state_slot_mapping(
                        [slot],
                        [positions],
                        max_blocks=layout.prefill_csa_inner_state_max_blocks,
                        state_block_size=layout.c4_state_block_size,
                        kernel_rows=layout.prefill_batch,
                    )[0],
                    layout.prefill_seq,
                )
            ),
            cmp_sparse_indices_by_ratio={
                ratio: self._rank_stack(indices)
                for ratio, (indices, _) in sparse_by_ratio.items()
            },
            cmp_sparse_lens_by_ratio={
                ratio: self._rank_stack(lens)
                for ratio, (_, lens) in sparse_by_ratio.items()
            },
        )

    def prepare_decode_inputs(self, model: RuntimeModel, batch: DecodeBatch) -> DeepSeekV4PreparedDecodeInputs:
        """Build DeepSeekV4 decode host inputs for the current scheduler batch."""
        builder = self._require_input_builder()
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if actual_batch <= 0:
            raise ValueError("decode batch must contain at least one request")
        if actual_batch > layout.decode_batch:
            raise ValueError(f"decode batch {actual_batch} exceeds kernel batch {layout.decode_batch}")
        slots = self.cache_manager.slots_for_request_ids(batch.request_ids)
        positions = self._decode_positions(batch, actual_batch)
        max_position = max(max(row) for row in positions)
        if max_position >= model.runtime.max_seq_len:
            raise ValueError(f"decode position {max_position} exceeds max_seq_len={model.runtime.max_seq_len}")

        token_ids = self._decode_token_rows(
            batch.token_ids.detach().cpu().to(torch.long),
            actual_batch,
            vocab_size=model.config.vocab_size,
        )
        x_hc = builder.decode_x_hc(batch.hidden_states.to(torch.bfloat16).cpu(), actual_batch=actual_batch)
        decode_slots = self._decode_kernel_slots(slots)
        decode_positions = (*positions, *((positions[0],) * (layout.decode_batch - actual_batch)))
        ori_slot_mapping = self.cache_manager.sliding_window_slot_mapping(
            decode_slots,
            decode_positions,
            kernel_rows=layout.decode_batch,
        )
        hca_cmp_slot_mapping = self.cache_manager.compressed_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.cmp_max_blocks,
            compress_ratio=128,
            kernel_rows=layout.decode_batch,
        )
        hca_state_slot_mapping = self.cache_manager.state_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.hca_state_max_blocks,
            state_block_size=layout.c128_state_block_size,
            kernel_rows=layout.decode_batch,
        )
        csa_cmp_slot_mapping = self.cache_manager.compressed_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.cmp_max_blocks,
            compress_ratio=4,
            kernel_rows=layout.decode_batch,
        )
        csa_idx_slot_mapping = self.cache_manager.compressed_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.idx_max_blocks,
            compress_ratio=4,
            kernel_rows=layout.decode_batch,
        )
        csa_state_slot_mapping = self.cache_manager.state_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.csa_state_max_blocks,
            state_block_size=layout.c4_state_block_size,
            kernel_rows=layout.decode_batch,
        )
        csa_inner_state_slot_mapping = self.cache_manager.state_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.csa_inner_state_max_blocks,
            state_block_size=layout.c4_state_block_size,
            kernel_rows=layout.decode_batch,
        )

        return DeepSeekV4PreparedDecodeInputs(
            request_ids=tuple(batch.request_ids),
            slots=tuple(slots),
            kernel_slots=decode_slots,
            actual_batch=actual_batch,
            x_hc=x_hc,
            input_ids=self._rank_stack(token_ids),
            position_ids=self._rank_stack(torch.tensor(decode_positions, dtype=torch.int32).reshape(-1)),
            kv_seq_lens=self._rank_stack(self._decode_kv_seq_lens(batch.seq_lens, actual_batch)),
            block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.ori_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            ori_slot_mapping=self._rank_stack(ori_slot_mapping.reshape(-1)),
            cmp_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.cmp_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            idx_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.idx_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            hca_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.hca_state_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            csa_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.csa_state_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            csa_inner_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.csa_inner_state_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            hca_cmp_slot_mapping=self._rank_stack(hca_cmp_slot_mapping.reshape(-1)),
            hca_state_slot_mapping=self._rank_stack(hca_state_slot_mapping.reshape(-1)),
            csa_cmp_slot_mapping=self._rank_stack(csa_cmp_slot_mapping.reshape(-1)),
            csa_idx_slot_mapping=self._rank_stack(csa_idx_slot_mapping.reshape(-1)),
            csa_state_slot_mapping=self._rank_stack(csa_state_slot_mapping.reshape(-1)),
            csa_inner_state_slot_mapping=self._rank_stack(csa_inner_state_slot_mapping.reshape(-1)),
        )

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceTensor:
        raise NotImplementedError("DeepSeekV4 uses model-specific cache pools, not generic KV tensors")

    def _free_kv_cache_tensor(self, tensor: DeviceTensor) -> None:
        return None

    def run_prefill(self, model, batch: PrefillBatch) -> PrefillResult:
        """Run all DeepSeekV4 hidden layers for one prefill chunk in a single packed call."""
        if self._compiled.prefill is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        self.load_packed_global_weights()
        self._static_freqs_cos_tensor()
        self._static_freqs_sin_tensor()
        self._ensure_lm_head_buffers()
        self._ensure_decode_buffers(model.config.hidden_size)
        self._ensure_decode_work_cache()
        # Pre-allocate the packed shared buffers (stacked layer weights, hc_head and
        # packed prefill caches/metadata) before the prefill dispatch forks the L3
        # worker, so the child inherits them.
        if self._stacked_weight_buffers is None:
            self._stage_stacked_weights(self.load_stacked_layer_weights())
        self._hc_head_tensors()
        self._ensure_prefill_fwd_buffers(model.config.hidden_size)
        inputs = self.prepare_prefill_inputs(model, batch)
        if inputs.slot != 0:
            raise RuntimeError(
                "DeepSeekV4 prefill currently supports the first active serving slot only. "
                "Run with one concurrent request until pypto-lib exposes a 64-slot prefill kernel."
            )
        self._stage_prefill_fwd_inputs(inputs)
        x_out = self._require_prefill_fwd_buffers().x_out
        x_out.zero_()
        args = self._prefill_fwd_args(x_out)
        self._debug_prefill_dispatch(inputs, args)
        try:
            self._run_l3(
                self._require_prefill_callable(),
                *args,
                self._int32_scalar(inputs.actual_tokens),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 packed prefill dispatch failed "
                f"(tokens={inputs.actual_tokens}, slot={inputs.slot})"
            ) from exc
        self._snapshot_prefill_fwd_caches()

        logits = self._logits_for_hidden(x_out, active_rows=(inputs.actual_tokens - 1,), label="prefill")
        return PrefillResult(last_hidden=None, logits=logits)

    def run_decode(self, model, batch: DecodeBatch) -> DecodeResult:
        """Run all DeepSeekV4 hidden layers for one decode batch in a single packed call."""
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        self.load_packed_global_weights()
        self._static_freqs_cos_tensor()
        self._static_freqs_sin_tensor()
        self._ensure_lm_head_buffers()
        self._ensure_decode_work_cache()
        inputs = self._stage_decode_inputs(self.prepare_decode_inputs(model, batch))
        if inputs.actual_batch != 1 or inputs.slots != (0,):
            raise RuntimeError(
                "DeepSeekV4 decode currently supports the first active serving slot only. "
                "Run with one concurrent request until the compact cache handoff supports multiple slots."
            )
        self._require_prefill_cache_snapshots()
        if self._stacked_weight_buffers is None:
            self._stage_stacked_weights(self.load_stacked_layer_weights())
        self._populate_decode_work_cache(inputs.kernel_slots)
        decode_buffers = self._require_decode_buffers()
        x_hc = decode_buffers.x_hc_a
        active_decode_tokens = inputs.actual_batch
        self._debug_tensor_stats("decode.input.initial.active", x_hc[:, :active_decode_tokens, :, :])

        x_out = self._require_decode_output_buffer(model.config.hidden_size)
        x_out.zero_()
        args = self._decode_fwd_args(inputs, x_hc, x_out)
        self._debug_decode_dispatch(inputs, args)
        try:
            self._run_l3(self._require_decode_callable(), *args)
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 packed decode dispatch failed "
                f"(actual_batch={inputs.actual_batch}, slots={inputs.slots})"
            ) from exc
        self._debug_tensor_stats(
            "decode.output.active",
            x_out[:, :active_decode_tokens, :],
            per_rank=True,
        )
        if self._debug_tensor_stats_enabled() and not self._tensor_is_finite(
            x_out[:, :active_decode_tokens, :]
        ):
            raise RuntimeError("DeepSeekV4 packed decode produced non-finite active hidden rows")

        active_rows = tuple(row * self._compiled.layout.decode_seq for row in range(inputs.actual_batch))
        logits = self._logits_for_hidden(x_out, active_rows=active_rows, label="decode")
        hidden = self._final_norm(x_out)[0, list(active_rows)].float()
        return DecodeResult(hidden_states=hidden, logits=logits)

    def _require_prefill_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.prefill is None:
            raise RuntimeError("DeepSeekV4 prefill kernel is not compiled")
        return self._compiled.prefill

    def _require_decode_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 decode kernel is not compiled")
        return self._compiled.decode

    def _require_lm_head_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.lm_head is None:
            raise RuntimeError("DeepSeekV4 LM-head kernel is not compiled")
        return self._compiled.lm_head

    def _prefill_fwd_args(self, x_out: torch.Tensor) -> tuple[Any, ...]:
        """Build the single packed ``l3_prefill_fwd`` argument tuple."""
        buffers = self._require_prefill_fwd_buffers()
        stacked = self._require_stacked_weights()
        values = dict(stacked.tensors)
        values.update(
            {
                "x_hc": buffers.x_hc,
                "freqs_cos": buffers.freqs_cos,
                "freqs_sin": buffers.freqs_sin,
                "x_out": x_out,
            }
        )
        values.update(buffers.tensors)
        return self._ordered_layer_args(values, _PREFILL_FWD_TENSOR_ORDER)

    def _decode_fwd_args(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        x_hc: torch.Tensor,
        x_out: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Build the single packed ``l3_decode_fwd`` argument tuple."""
        cache = self._require_decode_work_cache()
        stacked = self._require_stacked_weights()
        hc_head = self._hc_head_tensors()
        values = dict(stacked.tensors)
        values.update(
            {
                "x_hc": x_hc,
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": cache.kv_cache,
                "block_table": inputs.block_table,
                "ori_slot_mapping": inputs.ori_slot_mapping,
                "hca_cmp_slot_mapping": inputs.hca_cmp_slot_mapping,
                "hca_state_slot_mapping": inputs.hca_state_slot_mapping,
                "csa_cmp_slot_mapping": inputs.csa_cmp_slot_mapping,
                "csa_idx_slot_mapping": inputs.csa_idx_slot_mapping,
                "csa_state_slot_mapping": inputs.csa_state_slot_mapping,
                "csa_inner_state_slot_mapping": inputs.csa_inner_state_slot_mapping,
                "position_ids": inputs.position_ids,
                "kv_seq_lens": inputs.kv_seq_lens,
                "hca_compress_state": cache.hca_compress_state,
                "hca_compress_state_block_table": inputs.hca_compress_state_block_table,
                "csa_compress_state": cache.csa_compress_state,
                "csa_compress_state_block_table": inputs.csa_compress_state_block_table,
                "csa_inner_compress_state": cache.csa_inner_compress_state,
                "csa_inner_compress_state_block_table": inputs.csa_inner_compress_state_block_table,
                "cmp_kv": cache.cmp_kv,
                "cmp_block_table": inputs.cmp_block_table,
                "idx_kv_cache": cache.idx_kv_cache,
                "idx_block_table": inputs.idx_block_table,
                "input_ids": inputs.input_ids,
                "hc_head_fn": hc_head["hc_head_fn"],
                "hc_head_scale": hc_head["hc_head_scale"],
                "hc_head_base": hc_head["hc_head_base"],
                "x_out": x_out,
            }
        )
        return self._ordered_layer_args(values, _DECODE_FWD_TENSOR_ORDER)

    def _require_stacked_weights(self) -> DeepSeekV4StackedLayerWeights:
        if self._stacked_weight_buffers is None:
            raise RuntimeError("DeepSeekV4 stacked decode weights were not staged")
        return DeepSeekV4StackedLayerWeights(tensors=self._stacked_weight_buffers)

    def _ordered_layer_args(self, values: dict[str, Any], names: Sequence[str]) -> tuple[Any, ...]:
        missing = [name for name in names if name not in values]
        if missing:
            raise KeyError(f"DeepSeekV4 layer dispatch is missing tensors: {', '.join(missing)}")
        return tuple(values[name] for name in names)

    def _debug_prefill_dispatch(
        self,
        inputs: DeepSeekV4PreparedPrefillInputs,
        args: Sequence[Any],
    ) -> None:
        if os.getenv("PYPTO_DSV4_DEBUG") != "1":
            return
        named_args = dict(zip(_PREFILL_FWD_TENSOR_ORDER, args, strict=True))
        interesting = (
            "x_hc",
            "kv_cache",
            "cmp_kv",
            "idx_kv_cache",
            "ori_block_table",
            "cmp_block_table",
            "idx_block_table",
            "cmp_sparse_indices",
            "cmp_sparse_lens",
            "input_ids",
            "x_out",
        )
        tensor_names = [
            name
            for name, tensor in named_args.items()
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
        ]
        non_shared = [name for name in tensor_names if not named_args[name].is_shared()]
        parts = []
        for name in interesting:
            tensor = named_args[name]
            if isinstance(tensor, torch.Tensor):
                parts.append(f"{name}={tuple(tensor.shape)}/{tensor.dtype}/shared={tensor.is_shared()}")
            elif isinstance(tensor, DeviceTensor):
                parts.append(f"{name}=DeviceTensor")
            else:
                parts.append(f"{name}={type(tensor).__name__}")
        print(
            "DeepSeekV4 packed prefill dispatch "
            f"tokens={inputs.actual_tokens} slot={inputs.slot} "
            f"worker_started={self._l3_worker is not None} "
            f"cpu_tensor_args={len(tensor_names)} non_shared={non_shared} "
            + " ".join(parts),
            flush=True,
        )
        if os.getenv("PYPTO_DSV4_DEBUG_ARGS") == "1":
            for name in _PREFILL_FWD_TENSOR_ORDER:
                tensor = named_args[name]
                if isinstance(tensor, torch.Tensor):
                    print(
                        "DeepSeekV4 prefill arg "
                        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                        f"device={tensor.device} shared={tensor.is_shared()}",
                        flush=True,
                    )

    def _debug_decode_dispatch(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        args: Sequence[Any],
    ) -> None:
        if os.getenv("PYPTO_DSV4_DEBUG") != "1":
            return
        named_args = dict(zip(_DECODE_FWD_TENSOR_ORDER, args, strict=True))
        interesting = (
            "x_hc",
            "kv_cache",
            "block_table",
            "ori_slot_mapping",
            "cmp_kv",
            "cmp_block_table",
            "idx_kv_cache",
            "idx_block_table",
            "hca_compress_state",
            "hca_state_slot_mapping",
            "csa_compress_state",
            "csa_state_slot_mapping",
            "csa_inner_compress_state",
            "csa_inner_state_slot_mapping",
            "position_ids",
            "kv_seq_lens",
            "input_ids",
            "x_out",
        )
        tensor_names = [
            name
            for name, tensor in named_args.items()
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
        ]
        non_shared = [name for name in tensor_names if not named_args[name].is_shared()]
        parts = []
        for name in interesting:
            tensor = named_args[name]
            if isinstance(tensor, torch.Tensor):
                parts.append(f"{name}={tuple(tensor.shape)}/{tensor.dtype}/shared={tensor.is_shared()}")
            elif isinstance(tensor, DeviceTensor):
                parts.append(f"{name}=DeviceTensor")
            else:
                parts.append(f"{name}={type(tensor).__name__}")
        print(
            "DeepSeekV4 packed decode dispatch "
            f"actual_batch={inputs.actual_batch} active_tokens={inputs.actual_batch * self._compiled.layout.decode_seq} "
            f"slots={inputs.slots} "
            f"worker_started={self._l3_worker is not None} "
            f"cpu_tensor_args={len(tensor_names)} non_shared={non_shared} "
            + " ".join(parts),
            flush=True,
        )
        if os.getenv("PYPTO_DSV4_DEBUG_ARGS") == "1":
            for name in _DECODE_FWD_TENSOR_ORDER:
                tensor = named_args[name]
                if isinstance(tensor, torch.Tensor):
                    print(
                        "DeepSeekV4 decode arg "
                        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                        f"device={tensor.device} shared={tensor.is_shared()}",
                        flush=True,
                    )
                    self._debug_tensor_stats(f"dispatch.fwd.{name}", tensor)

    @staticmethod
    def _is_layer_weight_name(name: str) -> bool:
        runtime_names = {
            "x_hc",
            "freqs_cos",
            "freqs_sin",
            "hca_cmp_kv_state",
            "hca_cmp_score_state",
            "hca_compress_state_block_table",
            "csa_cmp_kv_state",
            "csa_cmp_score_state",
            "csa_compress_state_block_table",
            "csa_inner_kv_state",
            "csa_inner_score_state",
            "csa_inner_compress_state_block_table",
            "kv_cache",
            "ori_block_table",
            "block_table",
            "ori_slot_mapping",
            "cmp_kv",
            "cmp_block_table",
            "cmp_sparse_indices",
            "cmp_sparse_lens",
            "idx_kv_cache",
            "idx_block_table",
            "position_ids",
            "hca_cmp_slot_mapping",
            "hca_state_slot_mapping",
            "csa_cmp_slot_mapping",
            "csa_idx_slot_mapping",
            "csa_state_slot_mapping",
            "csa_inner_state_slot_mapping",
            "hca_compress_state",
            "csa_compress_state",
            "csa_inner_compress_state",
            "kv_seq_lens",
            "input_ids",
            "x_next",
        }
        return name not in runtime_names

    def _ensure_decode_buffers(self, hidden_size: int) -> _DeepSeekV4DecodeSharedBuffers:
        buffers = self._decode_buffers
        if buffers is None:
            self._ensure_shared_host_allocation_before_worker("decode inputs")
            layout = self._compiled.layout
            ranks = layout.ranks
            batch = layout.decode_batch
            tokens = layout.decode_tokens
            buffers = _DeepSeekV4DecodeSharedBuffers(
                x_hc_a=self._shared_empty(
                    (ranks, tokens, layout.hc_mult, int(hidden_size)),
                    torch.bfloat16,
                    name="decode_x_hc",
                ),
                x_hc_b=self._shared_empty(
                    (ranks, tokens, layout.hc_mult, int(hidden_size)),
                    torch.bfloat16,
                    name="decode_x_hc_next",
                ),
                x_out=self._shared_empty(
                    (ranks, tokens, int(hidden_size)),
                    torch.bfloat16,
                    name="decode_x_out",
                ),
                tensors={
                    "input_ids": self._shared_empty((ranks, tokens), torch.long, name="decode_input_ids"),
                    "position_ids": self._shared_empty((ranks, tokens), torch.int32, name="decode_position_ids"),
                    "kv_seq_lens": self._shared_empty((ranks, batch), torch.int32, name="decode_kv_seq_lens"),
                    "block_table": self._shared_empty(
                        (ranks, batch, layout.ori_max_blocks),
                        torch.int32,
                        name="decode_block_table",
                    ),
                    "ori_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_ori_slot_mapping",
                    ),
                    "cmp_block_table": self._shared_empty(
                        (ranks, batch, layout.cmp_max_blocks),
                        torch.int32,
                        name="decode_cmp_block_table",
                    ),
                    "idx_block_table": self._shared_empty(
                        (ranks, batch, layout.idx_max_blocks),
                        torch.int32,
                        name="decode_idx_block_table",
                    ),
                    "hca_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.hca_state_max_blocks),
                        torch.int32,
                        name="decode_hca_compress_state_block_table",
                    ),
                    "csa_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.csa_state_max_blocks),
                        torch.int32,
                        name="decode_csa_compress_state_block_table",
                    ),
                    "csa_inner_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.csa_inner_state_max_blocks),
                        torch.int32,
                        name="decode_csa_inner_compress_state_block_table",
                    ),
                    "hca_cmp_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_hca_cmp_slot_mapping",
                    ),
                    "hca_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_hca_state_slot_mapping",
                    ),
                    "csa_cmp_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_cmp_slot_mapping",
                    ),
                    "csa_idx_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_idx_slot_mapping",
                    ),
                    "csa_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_state_slot_mapping",
                    ),
                    "csa_inner_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_inner_state_slot_mapping",
                    ),
                },
            )
            self._decode_buffers = buffers
        return buffers

    def _stage_decode_inputs(self, inputs: DeepSeekV4PreparedDecodeInputs) -> DeepSeekV4PreparedDecodeInputs:
        buffers = self._ensure_decode_buffers(inputs.x_hc.shape[-1])
        self._copy_shared(buffers.x_hc_a, inputs.x_hc, name="decode_x_hc")
        staged_values: dict[str, torch.Tensor] = {}
        for name in _DECODE_INPUT_TENSOR_FIELDS:
            dst = buffers.tensors[name]
            self._copy_shared(dst, getattr(inputs, name), name=f"decode_{name}")
            staged_values[name] = dst
        return replace(inputs, x_hc=buffers.x_hc_a, **staged_values)

    def _ensure_prefill_fwd_buffers(self, hidden_size: int) -> _DeepSeekV4PrefillFwdSharedBuffers:
        """Allocate the layer-stacked shared buffers for the packed prefill dispatch."""
        buffers = self._prefill_fwd_buffers
        if buffers is not None:
            return buffers
        self._ensure_shared_host_allocation_before_worker("prefill_fwd buffers")
        layout = self._compiled.layout
        ranks = layout.ranks
        seq = layout.prefill_seq
        hidden = int(hidden_size)
        fwd = DEEPSEEK_V4_FWD_NUM_LAYERS
        csa = DEEPSEEK_V4_CSA_NUM_LAYERS
        hca = DEEPSEEK_V4_HCA_NUM_LAYERS
        rope_dim = self._compiled.freqs_cos.shape[-1] if self._compiled.freqs_cos is not None else 0
        max_seq_len = self._compiled.freqs_cos.shape[0] if self._compiled.freqs_cos is not None else 0

        def shared(shape, dtype, name):
            return self._shared_empty(shape, dtype, name=name)

        tensors: dict[str, torch.Tensor] = {
            # HCA-group prefill compressor state (x20).
            "hca_cmp_kv_state": shared(
                (ranks, hca * layout.prefill_hca_state_max_blocks, layout.c128_state_block_size, DEEPSEEK_V4_HCA_MAIN_OUT_DIM),
                torch.float32,
                "prefill_fwd_hca_cmp_kv_state",
            ),
            "hca_cmp_score_state": shared(
                (ranks, hca * layout.prefill_hca_state_max_blocks, layout.c128_state_block_size, DEEPSEEK_V4_HCA_MAIN_OUT_DIM),
                torch.float32,
                "prefill_fwd_hca_cmp_score_state",
            ),
            "hca_compress_state_block_table": shared(
                (ranks, hca * layout.prefill_hca_state_max_blocks), torch.int32, "prefill_fwd_hca_state_block_table"
            ),
            # CSA-group prefill compressor state (x21).
            "csa_cmp_kv_state": shared(
                (ranks, csa * layout.prefill_csa_state_max_blocks, layout.c4_state_block_size, DEEPSEEK_V4_CSA_MAIN_OUT_DIM),
                torch.float32,
                "prefill_fwd_csa_cmp_kv_state",
            ),
            "csa_cmp_score_state": shared(
                (ranks, csa * layout.prefill_csa_state_max_blocks, layout.c4_state_block_size, DEEPSEEK_V4_CSA_MAIN_OUT_DIM),
                torch.float32,
                "prefill_fwd_csa_cmp_score_state",
            ),
            "csa_compress_state_block_table": shared(
                (ranks, csa * layout.prefill_csa_state_max_blocks), torch.int32, "prefill_fwd_csa_state_block_table"
            ),
            "csa_inner_kv_state": shared(
                (ranks, csa * layout.prefill_csa_inner_state_max_blocks, layout.c4_state_block_size, DEEPSEEK_V4_CSA_INNER_OUT_DIM),
                torch.float32,
                "prefill_fwd_csa_inner_kv_state",
            ),
            "csa_inner_score_state": shared(
                (ranks, csa * layout.prefill_csa_inner_state_max_blocks, layout.c4_state_block_size, DEEPSEEK_V4_CSA_INNER_OUT_DIM),
                torch.float32,
                "prefill_fwd_csa_inner_score_state",
            ),
            "csa_inner_compress_state_block_table": shared(
                (ranks, csa * layout.prefill_csa_inner_state_max_blocks), torch.int32, "prefill_fwd_csa_inner_state_block_table"
            ),
            # FWD-stacked work caches (x43, explicit per-layer block dim).
            "kv_cache": shared(
                (ranks, fwd, layout.ori_max_blocks, layout.block_size, 1, DEEPSEEK_V4_HEAD_DIM),
                torch.bfloat16,
                "prefill_fwd_kv_cache",
            ),
            "cmp_kv": shared(
                (ranks, fwd, layout.prefill_cmp_max_blocks, layout.block_size, 1, DEEPSEEK_V4_HEAD_DIM),
                torch.bfloat16,
                "prefill_fwd_cmp_kv",
            ),
            "idx_kv_cache": shared(
                (ranks, fwd, layout.prefill_cmp_max_blocks, layout.block_size, 1, DEEPSEEK_V4_IDX_HEAD_DIM),
                torch.bfloat16,
                "prefill_fwd_idx_kv_cache",
            ),
            # FWD-stacked per-layer metadata (x43).
            "ori_block_table": shared((ranks, fwd * layout.ori_max_blocks), torch.int32, "prefill_fwd_ori_block_table"),
            "ori_slot_mapping": shared((ranks, fwd * seq), torch.long, "prefill_fwd_ori_slot_mapping"),
            "cmp_block_table": shared((ranks, fwd * layout.prefill_cmp_max_blocks), torch.int32, "prefill_fwd_cmp_block_table"),
            "cmp_sparse_indices": shared((ranks, fwd * seq, layout.prefill_sparse_topk), torch.int32, "prefill_fwd_cmp_sparse_indices"),
            "cmp_sparse_lens": shared((ranks, fwd * seq), torch.int32, "prefill_fwd_cmp_sparse_lens"),
            "idx_block_table": shared((ranks, fwd * layout.prefill_idx_max_blocks), torch.int32, "prefill_fwd_idx_block_table"),
            "position_ids": shared((ranks, fwd * seq), torch.int32, "prefill_fwd_position_ids"),
            "hca_cmp_slot_mapping": shared((ranks, fwd * seq), torch.long, "prefill_fwd_hca_cmp_slot_mapping"),
            "hca_state_slot_mapping": shared((ranks, fwd * seq), torch.long, "prefill_fwd_hca_state_slot_mapping"),
            "csa_cmp_slot_mapping": shared((ranks, fwd * seq), torch.long, "prefill_fwd_csa_cmp_slot_mapping"),
            "csa_idx_slot_mapping": shared((ranks, fwd * seq), torch.long, "prefill_fwd_csa_idx_slot_mapping"),
            "csa_state_slot_mapping": shared((ranks, fwd * seq), torch.long, "prefill_fwd_csa_state_slot_mapping"),
            "csa_inner_state_slot_mapping": shared((ranks, fwd * seq), torch.long, "prefill_fwd_csa_inner_state_slot_mapping"),
            "input_ids": shared((ranks, fwd * seq), torch.long, "prefill_fwd_input_ids"),
        }
        buffers = _DeepSeekV4PrefillFwdSharedBuffers(
            x_hc=shared((ranks, seq, layout.hc_mult, hidden), torch.bfloat16, "prefill_fwd_x_hc"),
            x_out=shared((ranks, seq, layout.hc_mult, hidden), torch.bfloat16, "prefill_fwd_x_out"),
            freqs_cos=shared((ranks, fwd * max_seq_len, rope_dim), torch.bfloat16, "prefill_fwd_freqs_cos"),
            freqs_sin=shared((ranks, fwd * max_seq_len, rope_dim), torch.bfloat16, "prefill_fwd_freqs_sin"),
            tensors=tensors,
        )
        self._prefill_fwd_buffers = buffers
        return buffers

    def _require_prefill_fwd_buffers(self) -> _DeepSeekV4PrefillFwdSharedBuffers:
        if self._prefill_fwd_buffers is None:
            raise RuntimeError("DeepSeekV4 packed prefill shared buffers were not staged")
        return self._prefill_fwd_buffers

    def _stage_prefill_fwd_inputs(self, inputs: DeepSeekV4PreparedPrefillInputs) -> None:
        """Copy one prefill chunk's stacked metadata/state into the packed buffers.

        The per-request metadata is layer-independent (one slot, one token chunk),
        so each FWD-stacked field tiles the single ``[ranks, dim]`` tensor x43 on a
        fused layer axis; the sparse tables are selected per layer by compress
        ratio. Compressor-state and work caches start zeroed and are produced by the
        kernel. Compacted block tables tile across the CSA (x21) / HCA (x20) groups.
        """
        buffers = self._require_prefill_fwd_buffers()
        layout = self._compiled.layout
        fwd = DEEPSEEK_V4_FWD_NUM_LAYERS
        plan = self._compiled.layer_plan

        # x_hc / output collapse weights.
        self._copy_shared(buffers.x_hc, inputs.x_hc, name="prefill_fwd_x_hc")
        self._copy_shared(
            buffers.freqs_cos,
            self._tile_layers(self._static_freqs_cos_table(), fwd),
            name="prefill_fwd_freqs_cos",
        )
        self._copy_shared(
            buffers.freqs_sin,
            self._tile_layers(self._static_freqs_sin_table(), fwd),
            name="prefill_fwd_freqs_sin",
        )

        # Layer-independent FWD metadata tiled x43 along the fused layer axis.
        fwd_tiled = {
            "ori_block_table": inputs.ori_block_table,
            "ori_slot_mapping": inputs.ori_slot_mapping,
            "cmp_block_table": inputs.cmp_block_table,
            "idx_block_table": inputs.idx_block_table,
            "position_ids": inputs.position_ids,
            "hca_cmp_slot_mapping": inputs.hca_cmp_slot_mapping,
            "hca_state_slot_mapping": inputs.hca_state_slot_mapping,
            "csa_cmp_slot_mapping": inputs.csa_cmp_slot_mapping,
            "csa_idx_slot_mapping": inputs.csa_idx_slot_mapping,
            "csa_state_slot_mapping": inputs.csa_state_slot_mapping,
            "csa_inner_state_slot_mapping": inputs.csa_inner_state_slot_mapping,
            "input_ids": inputs.input_ids,
        }
        for name, tensor in fwd_tiled.items():
            self._copy_shared(buffers.tensors[name], self._tile_layers(tensor, fwd), name=f"prefill_fwd_{name}")

        # Sparse tables vary by per-layer compression ratio.
        sparse_indices_layers = []
        sparse_lens_layers = []
        for layer in plan:
            indices, lens = inputs.sparse_inputs_for_ratio(layer.compress_ratio)
            sparse_indices_layers.append(indices)
            sparse_lens_layers.append(lens)
        self._copy_shared(
            buffers.tensors["cmp_sparse_indices"],
            self._concat_layers(sparse_indices_layers, axis=1),
            name="prefill_fwd_cmp_sparse_indices",
        )
        self._copy_shared(
            buffers.tensors["cmp_sparse_lens"],
            self._concat_layers(sparse_lens_layers, axis=1),
            name="prefill_fwd_cmp_sparse_lens",
        )

        # Compacted state block tables (CSA x21 / HCA x20).
        self._copy_shared(
            buffers.tensors["hca_compress_state_block_table"],
            self._tile_layers(inputs.hca_compress_state_block_table, DEEPSEEK_V4_HCA_NUM_LAYERS),
            name="prefill_fwd_hca_state_block_table",
        )
        self._copy_shared(
            buffers.tensors["csa_compress_state_block_table"],
            self._tile_layers(inputs.csa_compress_state_block_table, DEEPSEEK_V4_CSA_NUM_LAYERS),
            name="prefill_fwd_csa_state_block_table",
        )
        self._copy_shared(
            buffers.tensors["csa_inner_compress_state_block_table"],
            self._tile_layers(inputs.csa_inner_compress_state_block_table, DEEPSEEK_V4_CSA_NUM_LAYERS),
            name="prefill_fwd_csa_inner_state_block_table",
        )

        # Compressor-state and work caches start zeroed; the kernel populates them.
        for name in (
            "hca_cmp_kv_state",
            "hca_cmp_score_state",
            "csa_cmp_kv_state",
            "csa_cmp_score_state",
            "csa_inner_kv_state",
            "csa_inner_score_state",
            "kv_cache",
            "cmp_kv",
            "idx_kv_cache",
        ):
            buffers.tensors[name].zero_()

    @staticmethod
    def _tile_layers(tensor: torch.Tensor, count: int) -> torch.Tensor:
        """Tile a ``[ranks, d1, ...]`` tensor ``count`` times on a fused layer axis."""
        if tensor.ndim < 2:
            raise ValueError(f"layer-tiled tensor must be rank>=2, got shape={tuple(tensor.shape)}")
        expanded = tensor.unsqueeze(1).expand(tensor.shape[0], count, *tensor.shape[1:])
        return expanded.reshape(tensor.shape[0], count * tensor.shape[1], *tensor.shape[2:]).contiguous()

    @staticmethod
    def _concat_layers(per_layer: Sequence[torch.Tensor], *, axis: int) -> torch.Tensor:
        """Concatenate per-layer ``[ranks, d1, ...]`` tensors on the fused layer axis."""
        return torch.cat([tensor.contiguous() for tensor in per_layer], dim=axis).contiguous()

    def _static_freqs_cos_table(self) -> torch.Tensor:
        if self._compiled.freqs_cos is None:
            raise RuntimeError("DeepSeekV4 RoPE cosine table is not initialized")
        return self._rank_stack(self._compiled.freqs_cos)

    def _static_freqs_sin_table(self) -> torch.Tensor:
        if self._compiled.freqs_sin is None:
            raise RuntimeError("DeepSeekV4 RoPE sine table is not initialized")
        return self._rank_stack(self._compiled.freqs_sin)

    def _snapshot_prefill_fwd_caches(self) -> None:
        """Capture per-layer cache slices from the packed prefill Out caches."""
        buffers = self._require_prefill_fwd_buffers()
        csa_order = 0
        hca_order = 0
        for layer in self._compiled.layer_plan:
            fwd = int(layer.layer_id)
            snapshot: dict[str, torch.Tensor] = {
                "kv_cache": buffers.tensors["kv_cache"][:, fwd].detach().cpu().contiguous().clone(),
                "cmp_kv": buffers.tensors["cmp_kv"][:, fwd].detach().cpu().contiguous().clone(),
            }
            if layer.compress_ratio == 4:
                snapshot["idx_kv_cache"] = buffers.tensors["idx_kv_cache"][:, fwd].detach().cpu().contiguous().clone()
                snapshot["csa_cmp_kv_state"] = self._slice_layer_state(
                    buffers.tensors["csa_cmp_kv_state"], csa_order, self._compiled.layout.prefill_csa_state_max_blocks
                )
                snapshot["csa_cmp_score_state"] = self._slice_layer_state(
                    buffers.tensors["csa_cmp_score_state"], csa_order, self._compiled.layout.prefill_csa_state_max_blocks
                )
                snapshot["csa_inner_kv_state"] = self._slice_layer_state(
                    buffers.tensors["csa_inner_kv_state"], csa_order, self._compiled.layout.prefill_csa_inner_state_max_blocks
                )
                snapshot["csa_inner_score_state"] = self._slice_layer_state(
                    buffers.tensors["csa_inner_score_state"], csa_order, self._compiled.layout.prefill_csa_inner_state_max_blocks
                )
                csa_order += 1
            elif layer.compress_ratio == 128:
                snapshot["hca_cmp_kv_state"] = self._slice_layer_state(
                    buffers.tensors["hca_cmp_kv_state"], hca_order, self._compiled.layout.prefill_hca_state_max_blocks
                )
                snapshot["hca_cmp_score_state"] = self._slice_layer_state(
                    buffers.tensors["hca_cmp_score_state"], hca_order, self._compiled.layout.prefill_hca_state_max_blocks
                )
                hca_order += 1
            self._prefill_cache_snapshots[layer.layer_id] = DeepSeekV4LayerCacheSnapshot(snapshot)

    @staticmethod
    def _slice_layer_state(stacked: torch.Tensor, order: int, blocks_per_layer: int) -> torch.Tensor:
        start = int(order) * int(blocks_per_layer)
        return stacked[:, start : start + int(blocks_per_layer)].detach().cpu().contiguous().clone()

    def _stage_stacked_weights(self, weights: DeepSeekV4StackedLayerWeights) -> DeepSeekV4StackedLayerWeights:
        """Copy the layer-stacked decode_fwd weights into shared buffers once."""
        buffers = self._stacked_weight_buffers
        if buffers is None:
            self._ensure_shared_host_allocation_before_worker("stacked layer weights")
            buffers = {
                name: self._new_shared_like(tensor, name=f"stacked_weight[{name}]")
                for name, tensor in weights.tensors.items()
            }
            self._stacked_weight_buffers = buffers

        missing = sorted(set(weights.tensors) - set(buffers))
        if missing:
            raise KeyError(f"DeepSeekV4 shared stacked-weight buffers are missing: {', '.join(missing)}")

        for name, tensor in weights.tensors.items():
            self._copy_shared(buffers[name], tensor, name=f"stacked_weight[{name}]")
        return DeepSeekV4StackedLayerWeights(tensors=buffers)

    def _hc_head_tensors(self) -> dict[str, torch.Tensor]:
        """Return rank-replicated hc_head weights for the decode_fwd output collapse."""
        buffers = self._hc_head_buffers
        if buffers is not None:
            return buffers
        self._ensure_shared_host_allocation_before_worker("hc_head weights")
        global_weights = self.load_packed_global_weights()
        ranks = self._compiled.layout.ranks
        # The kernel hc_head_fn is [HC_MULT, HC_DIM]; the checkpoint stores it as
        # [HC_MULT, hidden*HC_MULT] (== [HC_MULT, HC_DIM]). Scale/base are scalars
        # per HC_MULT row, rank-replicated.
        hc_head_fn = global_weights.hc_head_fn.to(torch.float32).contiguous().cpu()
        hc_head_scale = global_weights.hc_head_scale.to(torch.float32).contiguous().cpu()
        hc_head_base = global_weights.hc_head_base.to(torch.float32).contiguous().cpu()
        buffers = {
            "hc_head_fn": self._static_device_tensor(self._rank_stack(hc_head_fn)),
            "hc_head_scale": self._static_device_tensor(self._rank_stack(hc_head_scale)),
            "hc_head_base": self._static_device_tensor(self._rank_stack(hc_head_base)),
        }
        self._hc_head_buffers = buffers
        return buffers

    def _ensure_lm_head_buffers(self) -> None:
        weights = self.load_packed_global_weights()
        layout = self._compiled.layout
        hidden_shape = (layout.ranks, layout.decode_tokens, weights.lm_head_layout.hidden_size)
        logits_shape = (layout.ranks, layout.decode_tokens, weights.lm_head_layout.vocab_size)
        if self._lm_head_hidden_buffer is None:
            self._ensure_shared_host_allocation_before_worker("lm_head_hidden")
            self._lm_head_hidden_buffer = self._shared_empty(hidden_shape, torch.bfloat16, name="lm_head_hidden")
        if self._lm_head_logits_buffer is None:
            self._ensure_shared_host_allocation_before_worker("lm_head_logits")
            self._lm_head_logits_buffer = self._shared_empty(logits_shape, torch.float32, name="lm_head_logits")

    def _require_decode_buffers(self) -> _DeepSeekV4DecodeSharedBuffers:
        if self._decode_buffers is None:
            raise RuntimeError("DeepSeekV4 decode shared buffers were not staged")
        return self._decode_buffers

    def _require_decode_output_buffer(self, hidden_size: int) -> torch.Tensor:
        return self._ensure_decode_buffers(int(hidden_size)).x_out

    def _static_freqs_cos_tensor(self) -> torch.Tensor:
        if self._static_freqs_cos is None:
            if self._compiled.freqs_cos is None:
                raise RuntimeError("DeepSeekV4 RoPE cosine table is not initialized")
            self._ensure_shared_host_allocation_before_worker("freqs_cos")
            self._static_freqs_cos = self._static_device_tensor(self._rank_stack(self._compiled.freqs_cos))
        return self._static_freqs_cos

    def _static_freqs_sin_tensor(self) -> torch.Tensor:
        if self._static_freqs_sin is None:
            if self._compiled.freqs_sin is None:
                raise RuntimeError("DeepSeekV4 RoPE sine table is not initialized")
            self._ensure_shared_host_allocation_before_worker("freqs_sin")
            self._static_freqs_sin = self._static_device_tensor(self._rank_stack(self._compiled.freqs_sin))
        return self._static_freqs_sin

    def _require_prefill_cache_snapshots(self) -> None:
        missing = [
            str(layer.layer_id)
            for layer in self._compiled.layer_plan
            if layer.layer_id not in self._prefill_cache_snapshots
        ]
        if missing:
            raise RuntimeError(
                "DeepSeekV4 decode requires prefill cache snapshots before decode; "
                "missing layers: " + ", ".join(missing)
            )

    def _populate_decode_work_cache(self, kernel_slots: Sequence[int]) -> None:
        """Fill the layer-stacked decode work cache from every prefill snapshot.

        The packed ``l3_decode_fwd`` kernel reads all 43 layers' KV/compress state
        in one call, so the full work cache is populated once before dispatch. Each
        layer's blocks live at a stacked offset on dim 1: FWD layers use the layer
        id (0..42), CSA-group state uses the CSA order index (0..20), and HCA-group
        state uses the HCA order index (0..19). Within a layer the slot offset is
        ``layer_offset * decode_batch + slot``.
        """
        cache = self._require_decode_work_cache()
        layout = self._compiled.layout
        batch = layout.decode_batch
        cache.kv_cache.zero_()
        cache.cmp_kv.zero_()
        cache.idx_kv_cache.zero_()
        cache.hca_compress_state.zero_()
        cache.csa_compress_state.zero_()
        cache.csa_inner_compress_state.zero_()

        csa_order = 0
        hca_order = 0
        for layer in self._compiled.layer_plan:
            snapshot = self._prefill_cache_snapshots.get(layer.layer_id)
            if snapshot is None:
                raise RuntimeError(f"DeepSeekV4 decode cache snapshot missing for layer {layer.layer_id}")
            tensors = snapshot.tensors
            fwd_offset = int(layer.layer_id)
            for slot in kernel_slots:
                self._copy_snapshot_blocks_to_work(
                    tensors["kv_cache"],
                    cache.kv_cache,
                    fwd_offset * batch + int(slot),
                    layout.ori_max_blocks,
                )
                self._copy_snapshot_blocks_to_work(
                    tensors["cmp_kv"],
                    cache.cmp_kv,
                    fwd_offset * batch + int(slot),
                    layout.cmp_max_blocks,
                )
            if layer.compress_ratio == 4:
                for slot in kernel_slots:
                    self._copy_snapshot_blocks_to_work(
                        tensors["idx_kv_cache"],
                        cache.idx_kv_cache,
                        csa_order * batch + int(slot),
                        layout.idx_max_blocks,
                    )
                    self._copy_split_state_to_work(
                        tensors["csa_cmp_kv_state"],
                        tensors["csa_cmp_score_state"],
                        cache.csa_compress_state,
                        csa_order * batch + int(slot),
                        layout.csa_state_max_blocks,
                        DEEPSEEK_V4_CSA_MAIN_OUT_DIM,
                    )
                    self._copy_split_state_to_work(
                        tensors["csa_inner_kv_state"],
                        tensors["csa_inner_score_state"],
                        cache.csa_inner_compress_state,
                        csa_order * batch + int(slot),
                        layout.csa_inner_state_max_blocks,
                        DEEPSEEK_V4_CSA_INNER_OUT_DIM,
                    )
                csa_order += 1
            elif layer.compress_ratio == 128:
                for slot in kernel_slots:
                    self._copy_split_state_to_work(
                        tensors["hca_cmp_kv_state"],
                        tensors["hca_cmp_score_state"],
                        cache.hca_compress_state,
                        hca_order * batch + int(slot),
                        layout.hca_state_max_blocks,
                        DEEPSEEK_V4_HCA_MAIN_OUT_DIM,
                    )
                hca_order += 1

        if csa_order != DEEPSEEK_V4_CSA_NUM_LAYERS:
            raise RuntimeError(
                f"DeepSeekV4 decode expected {DEEPSEEK_V4_CSA_NUM_LAYERS} CSA layers, found {csa_order}"
            )
        if hca_order != DEEPSEEK_V4_HCA_NUM_LAYERS:
            raise RuntimeError(
                f"DeepSeekV4 decode expected {DEEPSEEK_V4_HCA_NUM_LAYERS} HCA layers, found {hca_order}"
            )

    @staticmethod
    def _slot_block_slice(slot: int, blocks_per_slot: int) -> slice:
        if slot < 0:
            raise ValueError("slot must be non-negative")
        start = int(slot) * int(blocks_per_slot)
        return slice(start, start + int(blocks_per_slot))

    def _copy_snapshot_blocks_to_work(
        self,
        snapshot: torch.Tensor,
        work: torch.Tensor,
        slot: int,
        blocks_per_slot: int,
    ) -> None:
        del self
        slot_slice = DeepSeekV4ModelRunner._slot_block_slice(slot, blocks_per_slot)
        dst = work[:, slot_slice]
        dst.zero_()
        blocks = min(snapshot.shape[1], int(blocks_per_slot))
        dst[:, :blocks].copy_(snapshot[:, :blocks])

    def _copy_split_state_to_work(
        self,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        work: torch.Tensor,
        slot: int,
        blocks_per_slot: int,
        out_dim: int,
    ) -> None:
        del self
        slot_slice = DeepSeekV4ModelRunner._slot_block_slice(slot, blocks_per_slot)
        dst = work[:, slot_slice]
        dst.zero_()
        blocks = min(kv_state.shape[1], score_state.shape[1], int(blocks_per_slot))
        dst[:, :blocks, ..., :out_dim].copy_(kv_state[:, :blocks])
        dst[:, :blocks, ..., out_dim : 2 * out_dim].copy_(score_state[:, :blocks])

    def _logits_for_hidden(
        self,
        x_hc: torch.Tensor,
        *,
        active_rows: Sequence[int],
        label: str = "unknown",
    ) -> torch.Tensor:
        global_weights = self.load_packed_global_weights()
        self._ensure_lm_head_buffers()
        if self._static_lm_head_weight is None:
            self._static_lm_head_weight = self._static_device_tensor(global_weights.lm_head_weight)
        if x_hc.ndim == 3:
            # Decode output: the in-kernel hc_head already collapsed HC_MULT, so
            # only the final RMS norm remains before the LM head.
            hidden = self._final_norm(x_hc)
        else:
            hidden = self._final_hidden(x_hc)
        if self._lm_head_hidden_buffer is None or self._lm_head_logits_buffer is None:
            raise RuntimeError("DeepSeekV4 LM-head shared buffers were not initialized")
        rows = tuple(int(row) for row in active_rows)
        if not rows:
            raise ValueError("DeepSeekV4 LM-head requires at least one active row")
        if min(rows) < 0 or max(rows) >= hidden.shape[1]:
            raise ValueError(
                f"DeepSeekV4 LM-head active rows {rows} exceed hidden rows={hidden.shape[1]}"
            )
        if len(rows) > self._lm_head_hidden_buffer.shape[1]:
            raise ValueError(
                "DeepSeekV4 LM-head active row count exceeds kernel capacity: "
                f"rows={len(rows)} capacity={self._lm_head_hidden_buffer.shape[1]}"
            )
        row_list = list(rows)
        if self._debug_tensor_stats_enabled():
            print(f"DSV4_DEBUG lm_head.label={label} active_rows={rows}", flush=True)
            if x_hc.ndim == 4:
                self._debug_tensor_stats("lm_head.x_hc.active", x_hc[:, row_list, :, :])
            self._debug_tensor_stats("lm_head.hidden.active", hidden[:, row_list, :])
        self._lm_head_hidden_buffer.zero_()
        self._lm_head_hidden_buffer[:, : len(rows)].copy_(hidden[:, row_list, :])
        if self._debug_tensor_stats_enabled():
            self._debug_tensor_stats("lm_head.hidden_buffer.active", self._lm_head_hidden_buffer[:, : len(rows), :])
        self._lm_head_logits_buffer.zero_()
        self._run_l3(
            self._require_lm_head_callable(),
            self._lm_head_hidden_buffer,
            self._static_lm_head_weight,
            self._lm_head_logits_buffer,
        )
        active_logits = self._lm_head_logits_buffer[:, : len(rows), :]
        self._debug_tensor_stats("lm_head.logits.active", active_logits, per_rank=True)
        logits = self._lm_head_logits_buffer[0, : len(rows), :].contiguous()
        self._debug_tensor_stats("lm_head.logits.returned", logits)
        return logits

    @staticmethod
    def _debug_tensor_stats_enabled() -> bool:
        return os.getenv("PYPTO_DSV4_LOGIT_DEBUG") == "1"

    @staticmethod
    def _debug_tensor_stats(name: str, tensor: torch.Tensor, *, per_rank: bool = False) -> None:
        if not DeepSeekV4ModelRunner._debug_tensor_stats_enabled():
            return
        data = tensor.detach().cpu().to(torch.float32)
        finite = torch.isfinite(data)
        finite_count = int(finite.sum().item())
        total = data.numel()
        nan_count = int(torch.isnan(data).sum().item())
        pos_inf_count = int(torch.isposinf(data).sum().item())
        neg_inf_count = int(torch.isneginf(data).sum().item())
        if finite_count:
            finite_values = data[finite]
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            absmax_value = float(finite_values.abs().max().item())
        else:
            min_value = float("nan")
            max_value = float("nan")
            absmax_value = float("nan")
        print(
            "DSV4_DEBUG "
            f"{name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"finite={finite_count}/{total} nan={nan_count} "
            f"+inf={pos_inf_count} -inf={neg_inf_count} "
            f"min={min_value:.6g} max={max_value:.6g} absmax={absmax_value:.6g}",
            flush=True,
        )
        if per_rank and data.ndim >= 1:
            rank_view = data.reshape(data.shape[0], -1)
            rank_finite = torch.isfinite(rank_view)
            rank_counts = (rank_view.shape[1] - rank_finite.sum(dim=1)).tolist()
            print(f"DSV4_DEBUG {name} nonfinite_by_rank={rank_counts}", flush=True)

    @staticmethod
    def _tensor_is_finite(tensor: torch.Tensor) -> bool:
        return bool(torch.isfinite(tensor.detach().cpu().to(torch.float32)).all().item())

    def _final_hidden(self, x_hc: torch.Tensor) -> torch.Tensor:
        """Collapse a ``[ranks, T, HC_MULT, D]`` HC stack and apply the final norm."""
        weights = self.load_packed_global_weights()
        x_hc = x_hc.to(torch.bfloat16).cpu()
        x_float = x_hc.float()
        flat = x_float.flatten(2)
        rms = torch.sqrt(flat.double().square().mean(dim=-1, keepdim=True) + DEEPSEEK_V4_RMS_NORM_EPS)
        normed_flat = flat / rms.to(torch.float32)
        mixes = torch.matmul(normed_flat, weights.hc_head_fn.t())
        pre = torch.sigmoid(mixes * weights.hc_head_scale + weights.hc_head_base) + DEEPSEEK_V4_HC_EPS
        collapsed = torch.sum(pre.unsqueeze(-1).double() * x_float.double(), dim=2)
        return self._final_norm(collapsed)

    def _final_norm(self, collapsed: torch.Tensor) -> torch.Tensor:
        """Apply the final RMS norm to an already-collapsed ``[ranks, T, D]`` hidden.

        The packed ``l3_decode_fwd`` kernel collapses HC_MULT in-kernel via
        ``hc_head`` and returns the collapsed (pre-final-norm) hidden, so decode
        only needs the model's final RMS norm before the LM head.
        """
        collapsed = collapsed.cpu().double()
        weights = self.load_packed_global_weights()
        norm_inv = torch.rsqrt(collapsed.square().mean(dim=-1, keepdim=True) + DEEPSEEK_V4_RMS_NORM_EPS)
        normed = collapsed * norm_inv * weights.final_norm_weight.double()
        return normed.to(torch.float32).to(torch.bfloat16).contiguous()

    def _scope_stats_run_config(self) -> Any:
        """Optional per-dispatch RunConfig that captures device scope stats.

        Enabled with ``PYPTO_DSV4_SCOPE_STATS=1`` to dump per-scope
        heap / task_window / tensormap peaks under ``<dir>/dfx_outputs/``.
        """
        if os.getenv("PYPTO_DSV4_SCOPE_STATS") != "1":
            return None
        from pypto.runtime import RunConfig  # noqa: PLC0415

        out_dir = os.getenv("PYPTO_DSV4_SCOPE_STATS_DIR", "/data/liuxu/pypto-serving/dsv4_scope_stats")
        return RunConfig(
            platform=self._compiled.platform,
            device_id=self._compiled.device_id,
            enable_scope_stats=True,
            save_kernels=True,
            save_kernels_dir=out_dir,
        )

    def _run_l3(self, callable_spec: DeepSeekV4L3Callable, *args: Any) -> Any:
        if self._l3_worker is None:
            self._assert_l3_args_shared_before_worker(callable_spec, args)
        worker = self._shared_l3_worker()
        run_config = self._scope_stats_run_config()
        uploaded: list[DeviceTensor] = []
        try:
            l3_args = tuple(self._coerce_l3_arg(worker, arg, uploaded) for arg in args)
            if run_config is not None:
                return worker.run(callable_spec.compiled, *l3_args, config=run_config)
            return worker.run(callable_spec.compiled, *l3_args)
        finally:
            for tensor in uploaded:
                worker.free_tensor(tensor)

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if not tensor.is_shared():
            tensor = tensor.share_memory_()
        return tensor

    @staticmethod
    def _shared_empty(shape: Sequence[int], dtype: torch.dtype, *, name: str) -> torch.Tensor:
        del name
        return torch.empty(tuple(int(dim) for dim in shape), dtype=dtype).share_memory_()

    @staticmethod
    def _new_shared_like(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        return torch.empty_like(tensor.contiguous(), memory_format=torch.contiguous_format).share_memory_()

    @staticmethod
    def _copy_shared(dst: torch.Tensor, src: torch.Tensor, *, name: str) -> None:
        if src.device.type != "cpu":
            src = src.cpu()
        if not src.is_contiguous():
            src = src.contiguous()
        if tuple(dst.shape) != tuple(src.shape) or dst.dtype != src.dtype:
            raise ValueError(
                f"{name} shared buffer shape/dtype mismatch: "
                f"buffer shape={tuple(dst.shape)} dtype={dst.dtype}, "
                f"source shape={tuple(src.shape)} dtype={src.dtype}"
            )
        dst.copy_(src)

    @staticmethod
    def _int32_scalar(value: int) -> int:
        return int(value)

    def _ensure_shared_host_allocation_before_worker(self, name: str) -> None:
        if self._l3_worker is not None:
            raise RuntimeError(
                f"DeepSeekV4 shared host buffer '{name}' must be allocated before the L3 worker starts"
            )

    def _assert_l3_args_shared_before_worker(
        self,
        callable_spec: DeepSeekV4L3Callable,
        args: Sequence[Any],
    ) -> None:
        for index, arg in enumerate(args):
            self._assert_l3_arg_shared(arg, name=f"{callable_spec.name}[{index}]")

    def _assert_l3_arg_shared(self, arg: Any, *, name: str) -> None:
        if isinstance(arg, (_StaticDeviceTensor, _TransientDeviceTensor)):
            self._assert_l3_arg_shared(arg.tensor, name=f"{name}.tensor")
            return
        if isinstance(arg, torch.Tensor) and arg.device.type == "cpu" and not arg.is_shared():
            raise TypeError(
                "DeepSeekV4 L3 dispatch requires shared-memory CPU tensors allocated before "
                f"the L3 worker starts; got {name} shape={tuple(arg.shape)} dtype={arg.dtype}"
            )
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes, bytearray)):
            for index, item in enumerate(arg):
                self._assert_l3_arg_shared(item, name=f"{name}[{index}]")
            return
        if isinstance(arg, dict):
            for key, item in arg.items():
                self._assert_l3_arg_shared(item, name=f"{name}[{key!r}]")

    def _coerce_l3_arg(self, worker: Any, arg: Any, uploaded: list[DeviceTensor]) -> Any:
        if isinstance(arg, _StaticDeviceTensor):
            self._assert_l3_arg_shared(arg, name="static")
            return arg.tensor
        if isinstance(arg, _TransientDeviceTensor):
            tensor = arg.tensor
            self._assert_l3_arg_shared(arg, name="transient")
            dev = worker.alloc_tensor(tensor.shape, tensor.dtype, init=tensor)
            uploaded.append(dev)
            return dev
        if isinstance(arg, torch.Tensor) and arg.device.type == "cpu" and not arg.is_shared():
            raise TypeError(
                "DeepSeekV4 L3 dispatch requires shared-memory CPU tensors allocated before "
                f"the worker starts; got non-shared tensor shape={tuple(arg.shape)} dtype={arg.dtype}"
            )
        return arg

    def _shared_l3_worker(self) -> Any:
        worker = self._l3_worker
        if worker is None:
            compiled_callables = self._compiled.l3_callables()
            if not compiled_callables:
                raise RuntimeError("DeepSeekV4 L3 callables are not compiled")
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            worker = DistributedWorker([callable_spec.compiled for callable_spec in compiled_callables])
            self._l3_worker = worker
        return worker

    def _ensure_decode_work_cache(self) -> DeepSeekV4LayerCache:
        cache = self._decode_work_cache
        if cache is not None:
            return cache
        self._ensure_shared_host_allocation_before_worker("decode work cache")
        layout = self._compiled.layout
        fwd_layers = DEEPSEEK_V4_FWD_NUM_LAYERS
        csa_layers = DEEPSEEK_V4_CSA_NUM_LAYERS
        hca_layers = DEEPSEEK_V4_HCA_NUM_LAYERS
        cache = DeepSeekV4LayerCache(
            kv_cache=self._shared_empty(
                (
                    layout.ranks,
                    fwd_layers * layout.decode_batch * layout.ori_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_HEAD_DIM,
                ),
                torch.bfloat16,
                name="decode_work_kv_cache",
            ),
            cmp_kv=self._shared_empty(
                (
                    layout.ranks,
                    fwd_layers * layout.decode_batch * layout.cmp_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_HEAD_DIM,
                ),
                torch.bfloat16,
                name="decode_work_cmp_kv",
            ),
            idx_kv_cache=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.decode_batch * layout.idx_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_IDX_HEAD_DIM,
                ),
                torch.bfloat16,
                name="decode_work_idx_kv_cache",
            ),
            hca_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    hca_layers * layout.decode_batch * layout.hca_state_max_blocks,
                    layout.c128_state_block_size,
                    DEEPSEEK_V4_HCA_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_hca_compress_state",
            ),
            csa_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.decode_batch * layout.csa_state_max_blocks,
                    layout.c4_state_block_size,
                    DEEPSEEK_V4_CSA_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_csa_compress_state",
            ),
            csa_inner_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.decode_batch * layout.csa_inner_state_max_blocks,
                    layout.c4_state_block_size,
                    DEEPSEEK_V4_CSA_INNER_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_csa_inner_compress_state",
            ),
        )
        self._decode_work_cache = cache
        return cache

    def _require_decode_work_cache(self) -> DeepSeekV4LayerCache:
        if self._decode_work_cache is None:
            raise RuntimeError("DeepSeekV4 decode work cache was not allocated before the L3 worker started")
        return self._decode_work_cache

    @staticmethod
    def _static_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        return DeepSeekV4ModelRunner._share_cpu_tensor(tensor)

    def _reset_l3_worker(self) -> None:
        worker = self._l3_worker
        if worker is None:
            return
        try:
            for tensor in self._l3_static_tensors.values():
                worker.free_tensor(tensor)
            worker.close()
        finally:
            self._l3_worker = None
            self._l3_static_tensors.clear()

    def close(self) -> None:
        worker = self._l3_worker
        try:
            if worker is not None:
                for tensor in self._l3_static_tensors.values():
                    worker.free_tensor(tensor)
                worker.close()
        finally:
            self._l3_worker = None
            self._decode_work_cache = None
            self._prefill_cache_snapshots.clear()
            self._l3_static_tensors.clear()

    def _require_input_builder(self) -> DeepSeekV4InputBuilder:
        if self.input_builder is None:
            raise RuntimeError("DeepSeekV4 input builder is not initialized")
        return self.input_builder

    def _rank_stack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0).expand(self._compiled.layout.ranks, *tensor.shape).contiguous()

    @staticmethod
    def _padded_vector(values: torch.Tensor, length: int, *, dtype: torch.dtype) -> torch.Tensor:
        if values.numel() <= 0:
            raise ValueError("values must not be empty")
        if values.numel() > length:
            raise ValueError(f"values length {values.numel()} exceeds padded length {length}")
        out = torch.empty((length,), dtype=dtype)
        out[: values.numel()] = values.to(dtype=dtype)
        if values.numel() < length:
            out[values.numel() :] = out[0]
        return out

    @staticmethod
    def _prefill_position_ids(positions: Sequence[int], length: int) -> torch.Tensor:
        if len(positions) <= 0:
            raise ValueError("positions must not be empty")
        if len(positions) > length:
            raise ValueError(f"positions length {len(positions)} exceeds padded length {length}")
        out = torch.arange(length, dtype=torch.int32)
        out[: len(positions)] = torch.tensor(tuple(int(pos) for pos in positions), dtype=torch.int32)
        return out

    @staticmethod
    def _pad_prefill_mapping(mapping: torch.Tensor, length: int) -> torch.Tensor:
        if mapping.ndim != 1:
            raise ValueError(f"prefill mapping must be rank-1, got shape={tuple(mapping.shape)}")
        if mapping.numel() > length:
            raise ValueError(f"prefill mapping length {mapping.numel()} exceeds padded length {length}")
        out = torch.full((length,), -1, dtype=mapping.dtype)
        out[: mapping.numel()].copy_(mapping.to(dtype=mapping.dtype))
        return out

    @staticmethod
    def _prefill_actual_tokens(batch: PrefillBatch) -> int:
        if batch.positions is not None:
            valid = batch.positions[0].detach().cpu()
            valid = valid[valid >= 0]
            if valid.numel() <= 0:
                raise ValueError("prefill positions must include at least one token")
            return int(valid.numel())
        seq_len = int(batch.seq_lens[0].item())
        if seq_len <= 0:
            raise ValueError("prefill seq_len must be positive")
        return seq_len

    @staticmethod
    def _prefill_positions(batch: PrefillBatch, actual_tokens: int) -> list[int]:
        if batch.positions is None:
            positions = list(range(actual_tokens))
        else:
            raw = batch.positions[0, :actual_tokens].detach().cpu().to(torch.long)
            positions = [int(pos) for pos in raw.tolist()]
        if any(pos < 0 for pos in positions):
            raise ValueError("prefill positions must be non-negative")
        expected = list(range(positions[0], positions[0] + actual_tokens))
        if positions != expected:
            raise ValueError(
                "prefill positions must form one contiguous chunk: "
                f"positions={positions[:8]}{'...' if len(positions) > 8 else ''}"
            )
        return positions

    def _prefill_sparse_by_ratio(
        self,
        positions: Sequence[int],
        actual_tokens: int,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {
            ratio: self._prefill_sparse_indices(positions, actual_tokens, compress_ratio=ratio)
            for ratio in (0, 4, 128)
        }

    def _prefill_sparse_indices(
        self,
        positions: Sequence[int],
        actual_tokens: int,
        *,
        compress_ratio: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout = self._compiled.layout
        indices = torch.full(
            (layout.prefill_seq, layout.prefill_sparse_topk),
            -1,
            dtype=torch.int32,
        )
        lens = torch.zeros((layout.prefill_seq,), dtype=torch.int32)
        current = {int(pos): row for row, pos in enumerate(positions[:actual_tokens])}
        for row in range(actual_tokens):
            abs_pos = int(positions[row])
            window_valid = min(layout.block_size, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            cursor = 0
            for key_abs in range(key_start_abs, abs_pos + 1):
                overlay_row = current.get(key_abs)
                if overlay_row is not None and overlay_row <= row:
                    indices[row, cursor] = layout.block_size + overlay_row
                else:
                    indices[row, cursor] = key_abs % layout.block_size
                cursor += 1
            if compress_ratio > 0:
                compressed_visible = min(
                    (abs_pos + 1) // compress_ratio,
                    layout.cmp_max_blocks * layout.block_size,
                    layout.prefill_sparse_topk - layout.block_size,
                )
                for cmp_slot in range(compressed_visible):
                    if cursor >= layout.prefill_sparse_topk:
                        break
                    indices[row, cursor] = layout.block_size + layout.prefill_seq + cmp_slot
                    cursor += 1
            lens[row] = cursor
        return indices, lens

    def _decode_positions(self, batch: DecodeBatch, actual_batch: int) -> tuple[tuple[int, ...], ...]:
        positions = []
        for row in range(actual_batch):
            seq_len = int(batch.seq_lens[row].item())
            if seq_len <= 0:
                raise ValueError("decode seq_lens must be positive")
            first_position = seq_len - 1
            positions.append(tuple(first_position + offset for offset in range(self._compiled.layout.decode_seq)))
        return tuple(positions)

    def _decode_token_rows(self, token_ids: torch.Tensor, actual_batch: int, *, vocab_size: int) -> torch.Tensor:
        layout = self._compiled.layout
        if token_ids.ndim == 1:
            active = token_ids[:actual_batch].reshape(actual_batch, 1)
        else:
            active = token_ids[:actual_batch, :1]
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        rows = torch.arange(layout.decode_tokens, dtype=torch.long).reshape(
            layout.decode_batch,
            layout.decode_seq,
        )
        rows.remainder_(int(vocab_size))
        for row in range(actual_batch):
            rows[row].copy_(active[row, 0].expand(layout.decode_seq))
        return rows.reshape(layout.decode_tokens)

    def _decode_kernel_slots(self, active_slots: Sequence[int]) -> tuple[int, ...]:
        """Route padded fixed decode rows into scratch cache slots."""
        layout = self._compiled.layout
        slots = [int(slot) for slot in active_slots]
        if not slots:
            raise ValueError("decode must include at least one active slot")
        if len(set(slots)) != len(slots):
            raise ValueError(f"decode slots must be unique, got {slots}")
        if len(slots) > layout.decode_batch:
            raise ValueError(f"decode slots exceed kernel batch {layout.decode_batch}: {slots}")
        active_set = set(slots)
        for scratch_slot in range(layout.decode_batch):
            if len(slots) >= layout.decode_batch:
                break
            if scratch_slot not in active_set:
                slots.append(scratch_slot)
        if len(slots) != layout.decode_batch:
            raise RuntimeError(
                f"DeepSeekV4 decode needs {layout.decode_batch} kernel slots, built {len(slots)}"
            )
        return tuple(slots)

    def _decode_kv_seq_lens(self, seq_lens: torch.Tensor, actual_batch: int) -> torch.Tensor:
        layout = self._compiled.layout
        active = seq_lens[:actual_batch].detach().cpu().to(torch.int32) + (layout.decode_seq - 1)
        return DeepSeekV4CacheManager.replicate_first_row(
            active.reshape(actual_batch, 1),
            actual_rows=actual_batch,
            kernel_rows=layout.decode_batch,
        ).reshape(layout.decode_batch)
