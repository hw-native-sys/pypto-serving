# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runner for the DSpark DeepSeek-V4-Flash target kernels.

Serves ``l3_prefill_fwd`` and ``l3_decode_fwd`` from
``pypto-lib/models/deepseek_v4_flash_dspark`` on the canonical 16-card
TP4/DP4/EP16 topology:

* The 16 NPU ranks form 4 TP groups.  One group owns one request's prefill
  (all 4 ranks run the same prompt through context-parallel attention) and up
  to 64 requests' decode (each rank owns 16 requests' 8-row query tiles while
  the group's whole 512-row token stream is gathered to every rank).
* Cache pools are scheduler-visible as 4 partitions -- one per TP group --
  because the group's four ranks hold identical replicated caches that the
  decode kernels rebuild from the shared token stream every step.
* Both dispatch classes run their kernel-validated physical extents. Prefill
  pads each packed group only to its TP4 alignment; decode stages 16 requests
  per rank / 64 per TP group and fills inactive rows with noise tokens plus
  scratch cache metadata.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypto_serving.model.common.runner.task_args import TaskArgs

import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    KVCacheGroupSpec,
    KVCacheSpec,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from pypto_serving.model.common.runner.buffer_set import copy_shared
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.deepseek_dspark.weight_loader import (
    DSparkStackedLayerWeights,
    DSparkWeightStore,
)
from pypto_serving.tools.profile import profile_span

logger = logging.getLogger(__name__)


# ---- topology ----
DSPARK_RANKS = 16
DSPARK_TP_SIZE = 4
DSPARK_CACHE_PARTITIONS = DSPARK_RANKS // DSPARK_TP_SIZE

# ---- model dims (DeepSeek-V4-Flash) ----
DSPARK_HIDDEN_SIZE = 4096
DSPARK_HC_MULT = 4
DSPARK_VOCAB_SIZE = 129280
DSPARK_HEAD_DIM = 512
DSPARK_ROPE_HEAD_DIM = 64
DSPARK_IDX_HEAD_DIM = 128
DSPARK_HCA_MAIN_OUT_DIM = 512
DSPARK_CSA_MAIN_OUT_DIM = 1024
DSPARK_CSA_INNER_OUT_DIM = 256
DSPARK_HCA_STATE_DIM = 2 * DSPARK_HCA_MAIN_OUT_DIM
DSPARK_CSA_STATE_DIM = 2 * DSPARK_CSA_MAIN_OUT_DIM
DSPARK_CSA_INNER_STATE_DIM = 2 * DSPARK_CSA_INNER_OUT_DIM
DSPARK_FWD_NUM_LAYERS = 43
DSPARK_CSA_NUM_LAYERS = 21
DSPARK_HCA_NUM_LAYERS = 20
DSPARK_LM_HEAD_TP_SIZE = 4
DSPARK_NOISE_TOKEN_ID = 128799

# ---- ring heaps (sourced from the kernels' own runtime constants) ----
# decode_fwd.py pins DECODE_RING_HEAP = 1 GiB; prefill_fwd.py pins the
# rebalanced per-scope-depth (2, 2, 4, 8) GiB profile (pypto-lib#1073).  The
# two programs fault under each other's sizing, so each dispatch carries its
# own RunConfig instead of one process-wide value.
DSPARK_DECODE_RING_HEAP = int(os.environ.get("PYPTO_DSPARK_DECODE_RING_HEAP", 1 << 30))
DSPARK_PREFILL_RING_HEAP = (
    2 * 1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024,
    4 * 1024 * 1024 * 1024,
    8 * 1024 * 1024 * 1024,
)

# ---- paging ----
DSPARK_BLOCK_SIZE = 32
DSPARK_SLIDING_WINDOW = 128
DSPARK_C128_STATE_PAGE_TOKENS = 8
DSPARK_C4_STATE_PAGE_TOKENS = 2

# ---- decode tile (fixed at the device-validated shape) ----
DSPARK_DECODE_SEQ = 8
DSPARK_DECODE_BATCH = 64  # requests per TP group
DSPARK_DECODE_LOCAL_BATCH = DSPARK_DECODE_BATCH // DSPARK_TP_SIZE
DSPARK_DECODE_TOKENS = DSPARK_DECODE_BATCH * DSPARK_DECODE_SEQ
DSPARK_DECODE_LOCAL_TOKENS = DSPARK_DECODE_LOCAL_BATCH * DSPARK_DECODE_SEQ
DSPARK_MOE_TOKENS = 128
DSPARK_MAX_LOGIT_ROWS = DSPARK_DECODE_TOKENS
DSPARK_SAMPLED_IDS_PAD = 8
DSPARK_MAX_SEQ_LEN = 16384

# ---- decode metadata table depths (kernel-frozen) ----
# The decode kernels' table types freeze their depths at the 1M-context
# constants (decode_indexer.IDX_MAX_BLOCKS, decode_compressor_ratio4
# .CMP_MAX_BLOCKS, decode_hca.COMPRESS_STATE_MAX_BLOCKS) and the generated
# orchestration reshapes the tables with those depths baked in.  Staging a
# shallower table asserts on device (valid_reshape in simpler's tensormap
# tensor.h) and surfaces as an opaque AICore 507901 lane poison -- so the
# decode depths must match prefill's exactly.  Unused entries are -1, and only
# the leading per-request span is ever read.
DSPARK_DECODE_ORI_TABLE_BLOCKS = 512
DSPARK_DECODE_CMP_C4_TABLE_BLOCKS = 8192
DSPARK_DECODE_IDX_TABLE_BLOCKS = 8192
# The HCA cmp table's depth dim is dynamic (CMP_TABLE_BLOCKS_DYN); four pages
# cover the 16K decode context (16384 / 128-token compression = 4 blocks).
DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS = 4
DSPARK_DECODE_HCA_STATE_TABLE_BLOCKS = 131072
DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS = 8

# ---- prefill geometry ----
DSPARK_PREFILL_MAX_TOKENS = 8192
# The physical token extent is fixed: the kernel-internal staging tensors are
# device-resident, so their dynamic extents cannot shrink below allocation.
DSPARK_PREFILL_DISPATCH_TOKENS = DSPARK_PREFILL_MAX_TOKENS
DSPARK_PREFILL_LOCAL_TOKENS = DSPARK_PREFILL_DISPATCH_TOKENS // DSPARK_TP_SIZE
DSPARK_PREFILL_MAX_BATCH = DSPARK_CACHE_PARTITIONS
DSPARK_PREFILL_MAX_CONTEXT_TOKENS = 1_048_576
DSPARK_PREFILL_ORI_TABLE_BLOCKS = 32768
DSPARK_PREFILL_HCA_CMP_TABLE_BLOCKS = 256
DSPARK_PREFILL_CSA_CMP_TABLE_BLOCKS = 8192
DSPARK_PREFILL_IDX_TABLE_BLOCKS = 8192
DSPARK_PREFILL_HCA_STATE_TABLE_BLOCKS = 131072
# The kernel freezes the CSA state tables deeper than the HCA one
# (prefill_csa.CSA_STATE_MAX_BLOCKS / INNER_STATE_MAX_BLOCKS = 524288 vs
# prefill_hca.HCA_STATE_MAX_BLOCKS = 131072); the generated orchestration
# walks the frozen depth regardless of the staged extent.
DSPARK_PREFILL_CSA_STATE_TABLE_BLOCKS = 524288
DSPARK_PREFILL_CSA_INNER_STATE_TABLE_BLOCKS = 524288
# Packed-prefill request axis (pypto-lib#1095): the kernel takes per-request
# block tables plus a monotonic query_start_loc over the packed extent.
# Serving dispatches one request per TP group, so the staged request count is
# one; raising it only changes the slot shapes here and the staging below.
DSPARK_PREFILL_MAX_REQUESTS = 1

# Dynamic packed-prefill axes from pypto-lib's l3_prefill_fwd ABI. The slots
# retain their maximum backing allocation, but each dispatch binds only the
# TP-aligned prefix described by query_start_loc.
_PREFILL_GROUP_DYNAMIC_NAMES = frozenset(
    {
        "x_hc",
        "swa_freqs_cos",
        "swa_freqs_sin",
        "compressed_freqs_cos",
        "compressed_freqs_sin",
        "hca_cmp_freqs_cos",
        "hca_cmp_freqs_sin",
        "csa_cmp_freqs_cos",
        "csa_cmp_freqs_sin",
        "ori_slot_mapping_full",
        "position_ids_full",
        "hca_cmp_slot_mapping_full",
        "hca_state_slot_mapping_full",
        "csa_cmp_slot_mapping_full",
        "csa_idx_slot_mapping_full",
        "csa_state_slot_mapping_full",
        "csa_inner_state_slot_mapping_full",
        "attn_stage",
        "x_mixed",
        "post_ffn",
        "comb_ffn",
        "hidden_workspace",
        "x_out",
    }
)
_PREFILL_LOCAL_DYNAMIC_NAMES = frozenset(
    {"position_ids_local", "input_ids", "ffn_out"}
)

# ---- per-request ring sizes (scheduler-visible blocks per sequence) ----
# The raw-KV ring must cover the sliding window plus the in-flight decode rows
# crossing a page boundary; the HCA state ring covers one full 128-token
# compression window plus its 512-row prefill tile. The CSA working ring likewise
# preserves the prefill tile plus the eight rows needed by the next ratio-4 pool.
DSPARK_ORI_RING_BLOCKS = (
    math.ceil((DSPARK_SLIDING_WINDOW - 1 + DSPARK_DECODE_SEQ) / DSPARK_BLOCK_SIZE) + 1
)
DSPARK_HCA_STATE_RING_BLOCKS = 256
DSPARK_CSA_STATE_RING_BLOCKS = 260
# Decode keeps its eight-row mathematical CSA window and the eager S=8 writes
# in separate halves of a 16-row transaction ring.
DSPARK_CSA_DECODE_STATE_RING_TOKENS = (
    DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS * DSPARK_C4_STATE_PAGE_TOKENS
)
# Full-history groups at the 16K decode ceiling.
DSPARK_CMP_C128_BLOCKS_PER_SEQ = 4
DSPARK_CMP_C4_BLOCKS_PER_SEQ = 128
DSPARK_IDX_BLOCKS_PER_SEQ = 128

DSPARK_CACHE_GROUP_NAMES = (
    "ori",
    "cmp_c128",
    "cmp_c4",
    "idx",
    "hca_state",
    "csa_state",
    "csa_inner_state",
)


def build_dspark_cache_group_specs(
    num_hidden_layers: int,
    compress_ratios: Sequence[int] | None = None,
    *,
    max_seq_len: int = DSPARK_MAX_SEQ_LEN,
) -> tuple[KVCacheGroupSpec, ...]:
    """Describe the seven DSpark cache families as scheduler-visible groups.

    ``num_partitions`` is the TP-group count (4), not the rank count: the four
    ranks of one group hold identical replicated pools, so a block allocated in
    partition g exists -- with the same id -- on every rank of group g.
    """
    max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if max_seq_len > DSPARK_MAX_SEQ_LEN:
        raise ValueError(
            f"DSpark decode cache tables support at most max_seq_len={DSPARK_MAX_SEQ_LEN}, "
            f"got {max_seq_len}"
        )
    all_layers = tuple(range(int(num_hidden_layers)))
    ratios = tuple(int(ratio) for ratio in (compress_ratios or ()))[:num_hidden_layers]
    csa_layers = tuple(index for index, ratio in enumerate(ratios) if ratio == 4) or all_layers
    hca_layers = tuple(index for index, ratio in enumerate(ratios) if ratio == 128) or all_layers

    def group(
        name: str,
        layers: tuple[int, ...],
        *,
        block_size: int,
        element_bytes: int,
        row_width: int,
        max_blocks_per_seq: int,
        compress_ratio: int = 1,
        extra_row_bytes: int = 0,
        sliding_window: int | None = None,
    ) -> KVCacheGroupSpec:
        storage_rows = block_size // compress_ratio
        return KVCacheGroupSpec(
            name=name,
            layer_indices=layers,
            spec=KVCacheSpec(
                block_size=block_size,
                page_size_bytes=(
                    len(layers) * storage_rows * (row_width * element_bytes + extra_row_bytes)
                ),
                compress_ratio=compress_ratio,
            ),
            max_blocks_per_seq=int(max_blocks_per_seq),
            num_partitions=DSPARK_CACHE_PARTITIONS,
            sliding_window=sliding_window,
        )

    return (
        group(
            "ori",
            all_layers,
            block_size=DSPARK_BLOCK_SIZE,
            element_bytes=2,
            row_width=DSPARK_HEAD_DIM,
            max_blocks_per_seq=DSPARK_ORI_RING_BLOCKS,
            sliding_window=DSPARK_SLIDING_WINDOW,
        ),
        group(
            "cmp_c128",
            hca_layers,
            block_size=128 * DSPARK_BLOCK_SIZE,
            element_bytes=2,
            row_width=DSPARK_HEAD_DIM,
            max_blocks_per_seq=DSPARK_CMP_C128_BLOCKS_PER_SEQ,
            compress_ratio=128,
        ),
        group(
            "cmp_c4",
            csa_layers,
            block_size=4 * DSPARK_BLOCK_SIZE,
            element_bytes=2,
            row_width=DSPARK_HEAD_DIM,
            max_blocks_per_seq=DSPARK_CMP_C4_BLOCKS_PER_SEQ,
            compress_ratio=4,
        ),
        group(
            "idx",
            csa_layers,
            block_size=4 * DSPARK_BLOCK_SIZE,
            element_bytes=1,
            row_width=DSPARK_IDX_HEAD_DIM,
            max_blocks_per_seq=DSPARK_IDX_BLOCKS_PER_SEQ,
            compress_ratio=4,
            extra_row_bytes=4,
        ),
        group(
            "hca_state",
            hca_layers,
            block_size=DSPARK_C128_STATE_PAGE_TOKENS,
            element_bytes=4,
            row_width=DSPARK_HCA_STATE_DIM,
            max_blocks_per_seq=DSPARK_HCA_STATE_RING_BLOCKS,
            sliding_window=DSPARK_SLIDING_WINDOW,
        ),
        group(
            "csa_state",
            csa_layers,
            block_size=DSPARK_C4_STATE_PAGE_TOKENS,
            element_bytes=4,
            row_width=DSPARK_CSA_STATE_DIM,
            max_blocks_per_seq=DSPARK_CSA_STATE_RING_BLOCKS,
            sliding_window=DSPARK_C4_STATE_PAGE_TOKENS * DSPARK_CSA_STATE_RING_BLOCKS,
        ),
        group(
            "csa_inner_state",
            csa_layers,
            block_size=DSPARK_C4_STATE_PAGE_TOKENS,
            element_bytes=4,
            row_width=DSPARK_CSA_INNER_STATE_DIM,
            max_blocks_per_seq=DSPARK_CSA_STATE_RING_BLOCKS,
            sliding_window=DSPARK_C4_STATE_PAGE_TOKENS * DSPARK_CSA_STATE_RING_BLOCKS,
        ),
    )


def dspark_cache_blocks_for_slots(
    group_specs: Sequence[KVCacheGroupSpec],
    capacity_slots: int,
) -> dict[str, int]:
    """Return scheduler-visible blocks per partition for ``capacity_slots`` requests."""
    capacity_slots = int(capacity_slots)
    if capacity_slots <= 0:
        raise ValueError("DSpark cache capacity_slots must be positive")
    specs = {spec.name: spec for spec in group_specs}
    missing = [name for name in DSPARK_CACHE_GROUP_NAMES if name not in specs]
    if missing:
        raise ValueError("missing DSpark cache groups: " + ", ".join(missing))
    return {
        name: capacity_slots * specs[name].max_blocks_per_seq
        for name in DSPARK_CACHE_GROUP_NAMES
    }


@dataclass(frozen=True)
class DSparkCacheLayout:
    """Kernel-fixed execution dimensions and metadata table depths."""

    ranks: int = DSPARK_RANKS
    tp_size: int = DSPARK_TP_SIZE
    partitions: int = DSPARK_CACHE_PARTITIONS
    hc_mult: int = DSPARK_HC_MULT
    hidden_size: int = DSPARK_HIDDEN_SIZE
    block_size: int = DSPARK_BLOCK_SIZE
    sliding_window: int = DSPARK_SLIDING_WINDOW
    decode_batch: int = DSPARK_DECODE_BATCH
    decode_local_batch: int = DSPARK_DECODE_LOCAL_BATCH
    decode_seq: int = DSPARK_DECODE_SEQ
    decode_tokens: int = DSPARK_DECODE_TOKENS
    decode_local_tokens: int = DSPARK_DECODE_LOCAL_TOKENS
    moe_tokens: int = DSPARK_MOE_TOKENS
    max_logit_rows: int = DSPARK_MAX_LOGIT_ROWS
    prefill_tokens: int = DSPARK_PREFILL_DISPATCH_TOKENS
    prefill_local_tokens: int = DSPARK_PREFILL_LOCAL_TOKENS
    prefill_batch: int = DSPARK_PREFILL_MAX_BATCH

    def validate_runtime(
        self, config: ModelConfig, runtime: RuntimeConfig, device_ids: Sequence[int]
    ) -> None:
        """Validate serving options against the kernel-fixed dimensions."""
        if len(device_ids) != self.ranks:
            raise ValueError(f"DSpark requires exactly {self.ranks} devices, got {len(device_ids)}")
        if runtime.page_size != self.block_size:
            raise ValueError(
                f"DSpark kernels require page_size={self.block_size}, got {runtime.page_size}"
            )
        if runtime.max_seq_len > DSPARK_MAX_SEQ_LEN:
            raise ValueError(
                "DSpark decode cache tables support at most "
                f"max_seq_len={DSPARK_MAX_SEQ_LEN}, got {runtime.max_seq_len}"
            )
        global_decode_capacity = self.partitions * self.decode_batch
        if runtime.max_batch_size > global_decode_capacity:
            raise ValueError(
                f"DSpark decode supports at most {global_decode_capacity} global requests "
                f"({self.decode_batch} per TP group), got max_batch_size={runtime.max_batch_size}"
            )
        expected = {
            "hidden_size": DSPARK_HIDDEN_SIZE,
            "num_hidden_layers": DSPARK_FWD_NUM_LAYERS,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": DSPARK_HEAD_DIM,
            "vocab_size": DSPARK_VOCAB_SIZE,
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
            mismatch = ", ".join(
                f"{name}={actual[name]} expected {value}" for name, value in expected.items()
            )
            raise ValueError("DSpark W8A8 kernels require Flash shape: " + mismatch)


@dataclass(frozen=True)
class DSparkLayerPlan:
    """Per-layer execution metadata (shared with the DeepSeek V4 variant)."""

    layer_id: int
    compress_ratio: int
    attention_kind: str
    include_tid2eid: bool
    include_gate_bias: bool


def build_dspark_layer_plan(
    *,
    compress_ratios: Sequence[int],
    num_hidden_layers: int,
    num_hash_layers: int,
) -> tuple[DSparkLayerPlan, ...]:
    """Build the per-layer plan from config metadata."""
    from pypto_serving.model.deepseek.npu_runner import (  # noqa: PLC0415
        deepseek_v4_attention_kind,
    )

    if len(compress_ratios) < num_hidden_layers:
        raise ValueError("compress_ratios must include at least one entry per hidden layer")
    return tuple(
        DSparkLayerPlan(
            layer_id=layer_id,
            compress_ratio=int(compress_ratios[layer_id]),
            attention_kind=deepseek_v4_attention_kind(int(compress_ratios[layer_id])),
            include_tid2eid=layer_id < num_hash_layers,
            include_gate_bias=layer_id >= num_hash_layers,
        )
        for layer_id in range(num_hidden_layers)
    )


class DSparkCacheMetadataBuilder:
    """Vectorized host lowering from scheduler block IDs to kernel metadata.

    Mirrors the pypto-lib ``utils`` helpers (the per-kernel fixtures lower the
    same contract with Python loops); every routine here is a plain torch
    expression so a full 512-row decode step lowers in one pass.
    """

    def __init__(self, layout: DSparkCacheLayout = DSparkCacheLayout()) -> None:
        self.layout = layout

    @staticmethod
    def ring_table(
        block_ids: Sequence[int],
        *,
        depth: int,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """Expand one request's ring pages to a fixed-depth logical table."""
        ids = torch.tensor([int(block_id) for block_id in block_ids], dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError("ring table rows need at least one allocated block")
        if bool((ids < 0).any()):
            raise ValueError("ring table block IDs must not be negative")
        index = torch.arange(depth) % ids.numel()
        return ids.index_select(0, index).to(dtype)

    @staticmethod
    def trailing_ring_table(
        block_ids: Sequence[int],
        *,
        position: int,
        page_tokens: int,
        depth: int,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """Map a compact decode ring to the latest pages in a larger state ring."""
        ids = torch.tensor([int(block_id) for block_id in block_ids], dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError("trailing ring table rows need at least one allocated block")
        if bool((ids < 0).any()):
            raise ValueError("trailing ring table block IDs must not be negative")
        if page_tokens <= 0 or depth <= 0:
            raise ValueError("page_tokens and depth must be positive")

        last_page = max(int(position), 0) // int(page_tokens)
        first_page = max(last_page - depth + 1, 0)
        logical_pages = torch.arange(first_page, last_page + 1, dtype=torch.long)
        table = torch.full((depth,), -1, dtype=dtype)
        table[logical_pages % depth] = ids[logical_pages % ids.numel()].to(dtype)
        return table

    @staticmethod
    def absolute_table(
        block_ids: Sequence[int],
        *,
        depth: int,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """Place one request's full-history pages at their logical indices."""
        ids = torch.tensor([int(block_id) for block_id in block_ids], dtype=torch.long)
        if bool((ids < 0).any()):
            raise ValueError("absolute table block IDs must not be negative")
        if ids.numel() > depth:
            raise ValueError(f"request owns {ids.numel()} pages, table depth is {depth}")
        table = torch.full((depth,), -1, dtype=dtype)
        table[: ids.numel()] = ids.to(dtype)
        return table

    @staticmethod
    def _gather_table(table: torch.Tensor, logical: torch.Tensor) -> torch.Tensor:
        """Gather table rows with out-of-range logical indices clamped."""
        depth = table.shape[-1]
        clamped = logical.clamp(0, depth - 1)
        if table.ndim == 1:
            return table.index_select(0, clamped.reshape(-1)).reshape(logical.shape)
        rows = (
            torch.arange(table.shape[0], device=logical.device)
            .reshape((table.shape[0],) + (1,) * (logical.ndim - 1))
            .expand_as(logical)
            .reshape(-1)
        )
        return table.reshape(-1, depth)[rows, clamped.reshape(-1)].reshape(logical.shape)

    def paged_slot_mapping(
        self,
        positions: torch.Tensor,
        table: torch.Tensor,
        *,
        block_size: int,
    ) -> torch.Tensor:
        """Map absolute positions through paged tables; -1 where unmapped."""
        positions_i64 = positions.to(torch.int64)
        logical = positions_i64 // block_size
        depth = table.shape[-1]
        gathered = self._gather_table(table, logical)
        valid = (logical < depth) & (gathered >= 0)
        slot = gathered * block_size + positions_i64 % block_size
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def compressed_slot_mapping(
        self,
        positions: torch.Tensor,
        table: torch.Tensor,
        *,
        compress_ratio: int,
        commit_tokens: int | None = None,
    ) -> torch.Tensor:
        """Map compression-boundary positions into the compressed caches.

        With ``commit_tokens`` set, boundary writes past the committed prefix of
        each request's row window are masked to -1 so uncommitted (noise) rows
        cannot publish compressed-cache entries.
        """
        positions_i64 = positions.to(torch.int64)
        boundary = (positions_i64 + 1) % compress_ratio == 0
        if commit_tokens is not None:
            columns = torch.arange(positions.shape[-1], device=positions.device).unsqueeze(0)
            boundary = boundary & (columns < int(commit_tokens))
        cache_col = positions_i64 // compress_ratio
        logical = cache_col // self.layout.block_size
        depth = table.shape[-1]
        gathered = self._gather_table(table, logical)
        valid = boundary & (logical < depth) & (gathered >= 0)
        slot = gathered * self.layout.block_size + cache_col % self.layout.block_size
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def state_slot_mapping(
        self,
        positions: torch.Tensor,
        table: torch.Tensor,
        *,
        state_page_tokens: int,
    ) -> torch.Tensor:
        """Map absolute positions into ringed compressor-state pages."""
        return self.paged_slot_mapping(positions, table, block_size=state_page_tokens)

    def swa_window_indices_and_lens(
        self,
        positions: torch.Tensor,
        table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower visible raw-KV window rows for each query row.

        Every attention family consumes the same full window (the current
        chunk's rows are part of it, read through the raw-KV ring slots), so
        this is the one window lowering the decode step needs.
        """
        window = self.layout.sliding_window
        block_size = self.layout.block_size
        positions_i64 = positions.to(torch.int64)
        batch, seq = positions_i64.shape
        start = (positions_i64 - window + 1).clamp(min=0)
        offsets = torch.arange(window, device=positions.device)
        visible = start.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)
        valid = offsets.unsqueeze(0).unsqueeze(0) <= (positions_i64 - start).unsqueeze(-1)
        logical = visible // block_size
        depth = table.shape[-1]
        gathered = self._gather_table(table, logical)
        rows_valid = valid & (logical < depth) & (gathered >= 0)
        indices = torch.where(
            rows_valid,
            gathered * block_size + visible % block_size,
            torch.full_like(gathered, -1),
        ).to(torch.int32)
        lens = (positions_i64 - start + 1).clamp(min=0).to(torch.int32)
        return indices.reshape(batch * seq, window).contiguous(), lens.reshape(batch * seq)


@dataclass(frozen=True)
class DSparkRopeTables:
    """Position-indexed base RoPE tables for the four DSpark rope profiles."""

    max_position: int
    # Ratio-0 (uncompressed) profile, full rope width, BF16.
    swa_cos: torch.Tensor
    swa_sin: torch.Tensor
    # Ratio-4 YaRN profile, full rope width, BF16.
    ratio4_cos: torch.Tensor
    ratio4_sin: torch.Tensor
    # Ratio-128 YaRN profile, full rope width, BF16 (prefill "compressed").
    ratio128_cos: torch.Tensor
    ratio128_sin: torch.Tensor
    # Ratio-128 YaRN profile, half rope width, FP32 (HCA compressor).
    ratio128_half_cos: torch.Tensor
    ratio128_half_sin: torch.Tensor

    def gather(self, table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Collect rope rows for clamped absolute positions."""
        index = positions.to(torch.long).clamp(0, self.max_position - 1).reshape(-1)
        return table.index_select(0, index).reshape(*positions.shape, table.shape[-1])


@dataclass(frozen=True)
class DSparkPreparedPrefillInputs:
    """TP-aligned host tensors for one packed prefill dispatch."""

    request_ids: tuple[str, ...]
    groups: tuple[int, ...]
    actual_tokens: tuple[int, ...]
    physical_tokens: int
    chunk_starts: tuple[int, ...]
    # Per-request prompt-chunk embeddings ([tokens, hidden] FP32); staged
    # directly into the shared x_hc slot with a zero tail.
    embeddings: tuple[torch.Tensor, ...]
    input_ids: torch.Tensor
    position_ids_local: torch.Tensor
    position_ids_full: torch.Tensor
    # Packed request boundaries per rank: [0, chunk_len] for the group's one
    # request, [0, 0] for groups without a request this dispatch.
    query_start_loc: torch.Tensor
    rope_tables: dict[str, torch.Tensor]
    slot_mappings: dict[str, torch.Tensor]
    block_tables: dict[str, torch.Tensor]
    logit_row_indices: torch.Tensor
    sampled_slots: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DSparkPreparedDecodeInputs:
    """Host tensors for one full-tile decode dispatch."""

    request_ids: tuple[str, ...]
    groups: tuple[int, ...]
    group_ordinals: tuple[int, ...]
    anchor_positions: tuple[int, ...]
    input_ids: torch.Tensor
    position_ids_local: torch.Tensor
    position_ids: torch.Tensor
    logit_row_indices: torch.Tensor
    # (rank, logit entry row) per batch row, reading sampled_ids[rank, row, 0].
    sampled_slots: tuple[tuple[int, int], ...]
    buffer_slot: int = 0


@dataclass(frozen=True)
class _DSparkGroupAssignment:
    """Per-group placement of one decode batch's requests."""

    groups: tuple[int, ...]
    ordinals: tuple[int, ...]
    # group -> ((batch index, group stream slot) per active request).  The
    # stream slot is the rank-major request index every group-row tensor and
    # the rank-local tables slice through.
    active_by_group: tuple[tuple[tuple[int, int], ...], ...]


@dataclass
class DSparkCompiledKernels:
    """Compiled L3 programs and immutable DSpark runtime metadata."""

    layout: DSparkCacheLayout
    model_dir: str
    weight_map: dict[str, str]
    weight_store: DSparkWeightStore
    compress_ratios: tuple[int, ...]
    layer_plan: tuple[DSparkLayerPlan, ...]
    kernel_dir: str
    runtime_model: RuntimeModel | None = None
    prefill: Any | None = None
    decode: Any | None = None
    rope: DSparkRopeTables | None = None
    platform: str = "a2a3"
    device_id: int = 0
    device_ids: tuple[int, ...] = ()
    n_routed_experts: int = 256
    num_hash_layers: int = 3
    embedding_weight: torch.Tensor | None = None

    def l3_callables(self) -> tuple[Any, ...]:
        """Return every compiled L3 program the shared worker may run."""
        return tuple(program for program in (self.prefill, self.decode) if program is not None)


class DSparkModelRunner(L3DispatchMixin, ModelRunner):
    """Runner boundary for the DSpark target kernels."""

    def __init__(self, *, compiled: DSparkCompiledKernels) -> None:
        super().__init__()
        self._compiled = compiled
        self.cache_metadata = DSparkCacheMetadataBuilder(layout=compiled.layout)
        self._init_l3_dispatch(stacked=True)
        self._decode_run_config: Any = None
        self._cache_group_specs: tuple[KVCacheGroupSpec, ...] = ()
        self._cache_group_num_blocks: dict[str, int] = {}
        self._decode_device_cache: dict[str, StackedDeviceTensor] | None = None
        self._global_weights: Any | None = None
        self._static_final_norm_weight: torch.Tensor | None = None
        self._static_lm_head_weight: torch.Tensor | None = None
        self._hc_head_buffers: dict[str, torch.Tensor] | None = None
        self._stacked_host_weights: dict[str, torch.Tensor] | None = None
        self._stacked_prefill_host_weights: dict[str, torch.Tensor] | None = None
        self._stacked_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._stacked_prefill_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._embedding_device_weight: StackedDeviceTensor | None = None
        self._device_scratch: dict[str, StackedDeviceTensor] = {}
        self._prefill_task_args: TaskArgs | None = None
        self._decode_task_args: list[TaskArgs] = []
        self._l3_shared_buffers_ready = False

    # ------------------------------------------------------------------
    # cache topology
    # ------------------------------------------------------------------
    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> int:
        """Allocate the replicated group pools from the post-weight budget."""
        self._cache_group_specs = self._resolve_cache_group_specs(config, runtime)
        self._configure_l3_rings(runtime)
        from pypto.runtime import RunConfig  # noqa: PLC0415

        # Decode always runs at the kernel's own 1 GiB heap; the runtime /
        # CLI heap sizes prefill (the two profiles are mutually fatal).
        self._decode_run_config = RunConfig(ring_heap=DSPARK_DECODE_RING_HEAP)
        record = self._compiled.runtime_model
        if record is None or not self._compiled.l3_callables():
            self._cache_group_num_blocks = dspark_cache_blocks_for_slots(
                self._cache_group_specs, 1
            )
            return self._cache_group_num_blocks["ori"]

        logger.info("[init_kv_cache] preparing DSpark worker and resident weights ...")
        self._ensure_l3_shared_buffers(record)
        requested_slots = min(
            self._compute_kv_cache_capacity_slots(runtime),
            DSPARK_DECODE_BATCH,
        )
        allocated = self._alloc_kv_cache_with_retry(requested_slots)
        logger.info(
            "[init_kv_cache] allocated DSpark cache: slots=%d (requested=%d) per partition, "
            "ori_blocks=%d, max_seq_len=%d",
            allocated,
            requested_slots,
            self._cache_group_num_blocks["ori"],
            runtime.max_seq_len,
        )
        return self._cache_group_num_blocks["ori"]

    def _resolve_cache_group_specs(
        self, config: ModelConfig, runtime: RuntimeConfig
    ) -> tuple[KVCacheGroupSpec, ...]:
        specs = runtime.kv_cache_groups or build_dspark_cache_group_specs(
            config.num_hidden_layers,
            self._compiled.compress_ratios,
            max_seq_len=runtime.max_seq_len,
        )
        names = tuple(spec.name for spec in specs)
        if names != DSPARK_CACHE_GROUP_NAMES:
            raise ValueError(
                "DSpark KV cache groups must be ordered as "
                + ", ".join(DSPARK_CACHE_GROUP_NAMES)
                + f"; got {names}"
            )
        if any(spec.num_partitions != DSPARK_CACHE_PARTITIONS for spec in specs):
            raise ValueError(
                f"DSpark KV cache groups must use {DSPARK_CACHE_PARTITIONS} partitions"
            )
        return tuple(specs)

    def _compute_kv_cache_capacity_slots(self, runtime: RuntimeConfig) -> int:
        """Compute per-partition request slots from the per-device budget."""
        ori_spec = self._cache_group_specs[0]
        if runtime.total_kv_pages is not None:
            requested_pages = int(runtime.total_kv_pages)
            if requested_pages < ori_spec.max_blocks_per_seq:
                raise ValueError(
                    "DSpark total_kv_pages must hold at least one maximum ring: "
                    f"expected >= {ori_spec.max_blocks_per_seq}, got {requested_pages}"
                )
            return requested_pages // ori_spec.max_blocks_per_seq
        device_ids = self._compiled.device_ids or (self._compiled.device_id,)
        utilization = float(getattr(runtime, "npu_memory_utilization", 0.90))
        budgets = []
        for device_id in device_ids:
            free_bytes, total_bytes = torch.npu.mem_get_info(f"npu:{device_id}")
            peak_non_kv = int(total_bytes) - int(free_bytes)
            budgets.append(int(int(total_bytes) * utilization - peak_non_kv))
        bytes_per_slot = sum(
            spec.max_blocks_per_seq * spec.spec.page_size_bytes
            for spec in self._cache_group_specs
        )
        # Every kernel row needs one isolated scratch page per family for
        # filler requests; the pools are sized to hold them past the
        # allocator-visible blocks.
        scratch_bytes = sum(
            DSPARK_DECODE_BATCH * spec.spec.page_size_bytes
            for spec in self._cache_group_specs
        )
        kv_budget = min(budgets)
        if kv_budget < scratch_bytes + bytes_per_slot:
            raise RuntimeError(
                f"DSpark KV cache cannot fit one capacity slot within "
                f"npu_memory_utilization={utilization:.2f}: budget={min(budgets)} bytes, "
                f"requires at least {scratch_bytes + bytes_per_slot} bytes"
            )
        return (kv_budget - scratch_bytes) // bytes_per_slot

    def _alloc_kv_cache_with_retry(self, requested_slots: int) -> int:
        """Allocate every cache family atomically, halving capacity on OOM."""
        capacity_slots = max(int(requested_slots), 1)
        while capacity_slots >= 1:
            self._cache_group_num_blocks = dspark_cache_blocks_for_slots(
                self._cache_group_specs,
                capacity_slots,
            )
            try:
                self._materialize_decode_device_cache()
                return capacity_slots
            except (RuntimeError, MemoryError) as exc:
                self._free_device_caches()
                if capacity_slots == 1:
                    raise RuntimeError(
                        "DSpark KV cache allocation failed at the one-slot minimum"
                    ) from exc
                previous = capacity_slots
                capacity_slots = max(capacity_slots // 2, 1)
                logger.warning(
                    "DSpark KV cache allocation failed (%s); retrying slots %d -> %d",
                    exc,
                    previous,
                    capacity_slots,
                )
        raise RuntimeError("DSpark KV cache allocation failed")

    def _physical_cache_num_blocks(self, group_name: str) -> int:
        try:
            return self._cache_group_num_blocks[group_name] + DSPARK_DECODE_BATCH
        except KeyError as exc:
            raise RuntimeError("DSpark KV cache capacity is not initialized") from exc

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype):
        raise NotImplementedError("DSpark uses model-specific cache pools, not generic KV tensors")

    def _free_kv_cache_tensor(self, tensor) -> None:
        return None

    def preflight(self, record: ModelRecord) -> None:
        """Stage host buffers and allocate the resident cache before readiness."""
        self._ensure_l3_shared_buffers(record.runtime_model)
        self._materialize_decode_device_cache()

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------
    def load_packed_global_weights(self):
        """Load global tensors and shard the LM head across its TP ranks."""
        from pypto_serving.model.deepseek.npu_runner import (  # noqa: PLC0415
            DEEPSEEK_V4_LM_HEAD_TP_SIZE,
        )

        if self._global_weights is None:
            loaded = self._compiled.weight_store.load_packed_global_weights(
                ranks=DEEPSEEK_V4_LM_HEAD_TP_SIZE
            )
            embed_weight = loaded.embed_weight.to(
                device="cpu", dtype=torch.bfloat16
            ).contiguous()
            exact_weight = loaded.lm_head_weight[
                :, : loaded.lm_head_layout.vocab_per_rank, :
            ].contiguous()
            self._global_weights = replace(
                loaded,
                embed_weight=embed_weight,
                lm_head_weight=exact_weight,
            )
            self._compiled.embedding_weight = embed_weight
        return self._global_weights

    def load_stacked_layer_weights(self) -> DSparkStackedLayerWeights:
        """Load and stack all hidden-layer weights for both dispatch classes."""
        compress_ratios = tuple(int(layer.compress_ratio) for layer in self._compiled.layer_plan)
        return self._compiled.weight_store.load_stacked_layer_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=self._compiled.num_hash_layers,
        )

    def _retain_stacked_host_weights(self, weights: DSparkStackedLayerWeights) -> None:
        self._ensure_shared_host_allocation_before_worker("stacked layer weights")
        self._stacked_host_weights = dict(weights.tensors)
        self._stacked_prefill_host_weights = dict(weights.prefill_tensors)

    def _require_stacked_weights(self, *, prefill: bool = False):
        tensors = (
            (self._stacked_prefill_device_weights or self._stacked_prefill_host_weights)
            if prefill
            else (self._stacked_device_weights or self._stacked_host_weights)
        )
        if tensors is None:
            raise RuntimeError("DSpark stacked weights are not available")
        return tensors

    def lookup_embedding_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Prefill embedding lookup from the lazily loaded table."""
        embed = self._compiled.embedding_weight
        if embed is None:
            embed = self._compiled.weight_store.load_tensor("embed.weight").contiguous().cpu()
            self._compiled.embedding_weight = embed
        return embed.index_select(0, token_ids.detach().cpu().to(torch.long).reshape(-1))

    # ------------------------------------------------------------------
    # shared buffers
    # ------------------------------------------------------------------
    def _ensure_l3_shared_buffers(self, model: RuntimeModel) -> None:
        """Allocate every CPU tensor visible to the L3 worker before it forks."""
        if self._l3_shared_buffers_ready:
            return
        with profile_span("DSparkModelRunner.prepare.load_global_weights", cat="executor"):
            self.load_packed_global_weights()
        with profile_span("DSparkModelRunner.prepare.load_stacked_weights", cat="executor"):
            stacked = self.load_stacked_layer_weights()
            self._retain_stacked_host_weights(stacked)
            del stacked
        with profile_span("DSparkModelRunner.prepare.final_norm", cat="executor"):
            self._static_final_norm_weight_tensor()
        with profile_span("DSparkModelRunner.prepare.lm_head", cat="executor"):
            self._static_lm_head_weight_tensor()
        with profile_span("DSparkModelRunner.prepare.hc_head", cat="executor"):
            self._hc_head_tensors()
        with profile_span("DSparkModelRunner.prepare.prefill_task_args", cat="executor"):
            from pypto_serving.model.deepseek_dspark.task_args import (  # noqa: PLC0415
                prefill_task_args,
            )

            self._prefill_task_args = prefill_task_args(self)
            self._prefill_task_args.allocate_host_shared(None)
            # The padding tail of the embedding slab must read as zero for the
            # life of the worker (pypto-lib#1069 contract); zero it once here.
            self._prefill_task_args.tensors["x_hc"].zero_()
        with profile_span("DSparkModelRunner.prepare.decode_task_args", cat="executor"):
            from pypto_serving.model.deepseek_dspark.task_args import (  # noqa: PLC0415
                decode_task_args,
            )

            self._decode_task_args = []
            for _slot in (0, 1):
                task_args = decode_task_args(self)
                task_args.allocate_host_shared(None)
                self._decode_task_args.append(task_args)
        with profile_span("DSparkModelRunner.upload_resident_weights", cat="executor"):
            self._materialize_resident_weights()
        self._l3_shared_buffers_ready = True

    def _ensure_shared_host_allocation_before_worker(self, name: str) -> None:
        if self._l3_worker is not None:
            raise RuntimeError(
                f"DSpark shared host buffer '{name}' must be allocated before the L3 worker starts"
            )

    def _hc_head_tensors(self) -> dict[str, torch.Tensor]:
        """Rank-replicated hc_head weights for the output collapse."""
        if self._hc_head_buffers is not None:
            return self._hc_head_buffers
        self._ensure_shared_host_allocation_before_worker("hc_head weights")
        global_weights = self.load_packed_global_weights()
        ranks = self._compiled.layout.ranks

        def rank_stack(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.unsqueeze(0)
                .expand(ranks, *tensor.shape)
                .contiguous()
            )

        buffers = {
            "hc_head_fn": self._static_device_tensor(
                rank_stack(global_weights.hc_head_fn.to(torch.float32).contiguous().cpu())
            ),
            "hc_head_scale": self._static_device_tensor(
                rank_stack(global_weights.hc_head_scale.to(torch.float32).contiguous().cpu())
            ),
            "hc_head_base": self._static_device_tensor(
                rank_stack(global_weights.hc_head_base.to(torch.float32).contiguous().cpu())
            ),
        }
        self._hc_head_buffers = buffers
        return buffers

    def _static_weight(self, name: str) -> torch.Tensor:
        """Return one upload-once static weight shared by both dispatch classes."""
        if name == "hc_head_fn":
            return self._hc_head_tensors()[name]
        if name in ("hc_head_scale", "hc_head_base"):
            return self._hc_head_tensors()[name]
        if name == "final_norm_w":
            return self._static_final_norm_weight_tensor()
        if name == "lm_head_weight":
            return self._static_lm_head_weight_tensor()
        raise KeyError(name)

    def _static_final_norm_weight_tensor(self) -> torch.Tensor:
        if self._static_final_norm_weight is None:
            global_weights = self.load_packed_global_weights()
            self._ensure_shared_host_allocation_before_worker("final_norm_w")
            final_norm_w = global_weights.final_norm_weight.to(torch.bfloat16).contiguous().cpu()
            self._static_final_norm_weight = self._static_device_tensor(
                self._rank_stack(final_norm_w)
            )
        return self._static_final_norm_weight

    def _static_lm_head_weight_tensor(self) -> torch.Tensor:
        """One TP vocab shard per rank: rank r consumes shard ``r % tp``."""
        if self._static_lm_head_weight is None:
            global_weights = self.load_packed_global_weights()
            self._ensure_shared_host_allocation_before_worker("lm_head_weight")
            packed = global_weights.lm_head_weight.to(torch.bfloat16).contiguous().cpu()
            tp_size = packed.shape[0]
            ranks = self._compiled.layout.ranks
            rank_shards = [packed[rank % tp_size] for rank in range(ranks)]
            self._static_lm_head_weight = self._static_device_tensor(
                torch.stack(rank_shards, dim=0).contiguous()
            )
        return self._static_lm_head_weight

    @staticmethod
    def _static_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        if not tensor.is_shared():
            tensor = tensor.share_memory_()
        return tensor

    def _rank_stack(self, tensor: torch.Tensor) -> torch.Tensor:
        ranks = self._compiled.layout.ranks
        return tensor.unsqueeze(0).expand(ranks, *tensor.shape).contiguous()

    def _materialize_resident_weights(self) -> None:
        """Upload inherited weights once and release parent host references."""
        worker = self._shared_l3_worker()
        if self._stacked_device_weights is None:
            host_weights = self._stacked_host_weights
            if not host_weights:
                raise RuntimeError("DSpark stacked Host weights are not retained")
            with profile_span("DSparkModelRunner.upload_resident_weights", cat="executor"):
                self._stacked_device_weights = self._upload_weight_group(worker, host_weights)
            self._stacked_host_weights = None
        if self._stacked_prefill_device_weights is None:
            host_weights = self._stacked_prefill_host_weights
            if not host_weights:
                raise RuntimeError("DSpark prefill HC Host weights are not retained")
            with profile_span("DSparkModelRunner.upload_prefill_hc", cat="executor"):
                self._stacked_prefill_device_weights = self._upload_weight_group(
                    worker, host_weights
                )
            self._stacked_prefill_host_weights = None
        self._materialize_embedding_device_weight()
        for task_args in (self._prefill_task_args, *self._decode_task_args):
            if task_args is not None:
                task_args.allocate_device(worker, None)
        worker.release_inherited_host_tensor_refs()

    @staticmethod
    def _upload_weight_group(
        worker: Any,
        host_weights: dict[str, torch.Tensor],
    ) -> dict[str, StackedDeviceTensor]:
        device_weights: dict[str, StackedDeviceTensor] = {}
        try:
            for name, tensor in host_weights.items():
                device_weights[name] = worker.alloc_stacked_tensor(tensor)
        except Exception:
            for tensor in device_weights.values():
                worker.free_stacked_tensor(tensor)
            raise
        return device_weights

    def _inherited_host_weights(self) -> list[torch.Tensor]:
        """Return host weights that must be visible at worker fork."""
        tensors: list[torch.Tensor] = []
        if self._stacked_host_weights:
            tensors.extend(self._stacked_host_weights.values())
        if self._stacked_prefill_host_weights:
            tensors.extend(self._stacked_prefill_host_weights.values())
        global_weights = getattr(self, "_global_weights", None)
        if global_weights is not None:
            tensors.append(global_weights.embed_weight)
        return tensors

    def _materialize_embedding_device_weight(self) -> StackedDeviceTensor:
        """Upload one full embedding table to every rank."""
        stacked = self._embedding_device_weight
        if stacked is not None:
            return stacked
        source = self.load_packed_global_weights().embed_weight
        if (
            source.device.type != "cpu"
            or source.dtype != torch.bfloat16
            or not source.is_contiguous()
        ):
            raise ValueError(
                "DSpark embedding weight must be contiguous BF16 CPU storage before worker fork"
            )
        worker = self._shared_l3_worker()
        worker_ids = tuple(range(self._compiled.layout.ranks))
        shards = []
        try:
            for worker_id in worker_ids:
                shards.append(
                    worker.alloc_tensor(
                        source.shape, source.dtype, init=source, worker_id=worker_id
                    )
                )
        except Exception:
            for shard, worker_id in zip(shards, worker_ids, strict=False):
                worker.free_tensor(shard, worker_id=worker_id)
            raise
        stacked = StackedDeviceTensor(
            shards,
            (self._compiled.layout.ranks, *source.shape),
            worker_ids,
        )
        self._embedding_device_weight = stacked
        return stacked

    def _alloc_zeroed_stacked_tensor(
        self,
        name: str,
        full_shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        scope: str = "",
    ) -> StackedDeviceTensor:
        """Allocate one zero-initialized scratch buffer on every rank.

        ``scope`` separates the two dispatch classes: the generated host
        orchestration sub-slices these tensors at their bound dynamic extents
        (whole-shard only on a ``StackedDeviceTensor``), so a name shared by
        prefill and decode must not resolve to one buffer when their extents
        differ (8192-token prefill staging vs the 128-row decode tile).
        """
        key = (scope, name)
        stacked = self._device_scratch.get(key)
        if stacked is not None:
            if tuple(stacked.full_shape) != tuple(int(dim) for dim in full_shape):
                raise ValueError(
                    f"DSpark scratch buffer {name!r} in scope {scope!r} already allocated as "
                    f"{tuple(stacked.full_shape)}, requested {tuple(full_shape)}"
                )
            return stacked
        worker = self._shared_l3_worker()
        worker_ids = tuple(range(self._compiled.layout.ranks))
        shards = []
        try:
            for worker_id in worker_ids:
                shards.append(
                    worker.alloc_tensor(
                        full_shape[1:], dtype, init=torch.zeros(full_shape[1:], dtype=dtype),
                        worker_id=worker_id,
                    )
                )
        except Exception:
            for shard, worker_id in zip(shards, worker_ids, strict=False):
                worker.free_tensor(shard, worker_id=worker_id)
            raise
        stacked = StackedDeviceTensor(shards, full_shape, worker_ids)
        self._device_scratch[key] = stacked
        return stacked

    def _alloc_empty_stacked_tensor(
        self,
        full_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> StackedDeviceTensor:
        """Allocate an uninitialized shard directly on every chip worker."""
        worker = self._shared_l3_worker()
        worker_ids = tuple(range(self._compiled.layout.ranks))
        shards: list[Any] = []
        try:
            for worker_id in worker_ids:
                shards.append(worker.alloc_tensor(full_shape[1:], dtype, worker_id=worker_id))
        except Exception:
            for shard, worker_id in zip(shards, worker_ids, strict=False):
                worker.free_tensor(shard, worker_id=worker_id)
            raise
        return StackedDeviceTensor(shards, full_shape, worker_ids)

    def _device_cache_values(self) -> dict[str, StackedDeviceTensor]:
        """Return the worker-resident cache pools by kernel argument name.

        Both dispatch classes share the same physical pools under their own
        ABI names (prefill ``kv_cache``/``idx_kv_*`` vs decode
        ``raw_kv_pool``/``csa_idx_kv_*``), so the aliases all resolve here.
        """
        cache = self._materialize_decode_device_cache()
        return {
            "kv_cache": cache["kv_cache"],
            "raw_kv_pool": cache["kv_cache"],
            "hca_cmp_kv": cache["hca_cmp_kv"],
            "csa_cmp_kv": cache["csa_cmp_kv"],
            "idx_kv_cache": cache["idx_kv_cache"],
            "idx_kv_scale": cache["idx_kv_scale"],
            "csa_idx_kv_cache": cache["idx_kv_cache"],
            "csa_idx_kv_scale": cache["idx_kv_scale"],
            "hca_compress_state": cache["hca_compress_state"],
            "csa_compress_state": cache["csa_compress_state"],
            "csa_inner_compress_state": cache["csa_inner_compress_state"],
        }

    def _materialize_decode_device_cache(self) -> dict[str, StackedDeviceTensor]:
        """Allocate the replicated per-group cache shards on each NPU."""
        cache = self._decode_device_cache
        if cache is not None:
            return cache
        layout = self._compiled.layout

        def packed(name: str, layers: int, rows: int, tail: tuple[int, ...], dtype):
            return (
                layout.ranks,
                layers * self._physical_cache_num_blocks(name),
                rows,
                *tail,
            ), dtype

        shapes = {
            "kv_cache": packed(
                "ori", DSPARK_FWD_NUM_LAYERS, layout.block_size, (1, DSPARK_HEAD_DIM),
                torch.bfloat16,
            ),
            "hca_cmp_kv": packed(
                "cmp_c128", DSPARK_HCA_NUM_LAYERS, layout.block_size, (1, DSPARK_HEAD_DIM),
                torch.bfloat16,
            ),
            "csa_cmp_kv": packed(
                "cmp_c4", DSPARK_CSA_NUM_LAYERS, layout.block_size, (1, DSPARK_HEAD_DIM),
                torch.bfloat16,
            ),
            "idx_kv_cache": packed(
                "idx", DSPARK_CSA_NUM_LAYERS, layout.block_size, (1, DSPARK_IDX_HEAD_DIM),
                torch.int8,
            ),
            "idx_kv_scale": packed(
                "idx", DSPARK_CSA_NUM_LAYERS, layout.block_size, (1, 1), torch.float32
            ),
            "hca_compress_state": packed(
                "hca_state",
                DSPARK_HCA_NUM_LAYERS,
                DSPARK_C128_STATE_PAGE_TOKENS,
                (DSPARK_HCA_STATE_DIM,),
                torch.float32,
            ),
            "csa_compress_state": packed(
                "csa_state",
                DSPARK_CSA_NUM_LAYERS,
                DSPARK_C4_STATE_PAGE_TOKENS,
                (DSPARK_CSA_STATE_DIM,),
                torch.float32,
            ),
            "csa_inner_compress_state": packed(
                "csa_inner_state",
                DSPARK_CSA_NUM_LAYERS,
                DSPARK_C4_STATE_PAGE_TOKENS,
                (DSPARK_CSA_INNER_STATE_DIM,),
                torch.float32,
            ),
        }
        cache = {}
        try:
            for name, (shape, dtype) in shapes.items():
                cache[name] = self._alloc_empty_stacked_tensor(shape, dtype)
        except Exception:
            for tensor in cache.values():
                self._l3_worker.free_stacked_tensor(tensor)
            raise
        self._decode_device_cache = cache
        return cache

    def _free_device_caches(self) -> None:
        worker = self._l3_worker
        if worker is None:
            self._decode_device_cache = None
            return
        if self._decode_device_cache is not None:
            for tensor in self._decode_device_cache.values():
                worker.free_stacked_tensor(tensor)
        self._decode_device_cache = None

    def _shared_l3_worker(self) -> Any:
        worker = self._l3_worker
        if worker is None:
            compiled_callables = self._compiled.l3_callables()
            if not compiled_callables:
                raise RuntimeError("DSpark L3 callables are not compiled")
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            compiled = [callable_spec.compiled for callable_spec in compiled_callables]
            with profile_span(
                "DSparkModelRunner.create_persistent_l3_worker",
                cat="executor",
                args={"callable_count": len(compiled)},
            ):
                worker_kwargs: dict[str, Any] = {
                    "persistent": True,
                    "reset_persistent_windows": False,
                    "inherited_host_tensors": self._inherited_host_weights(),
                }
                run_config = getattr(self, "_l3_run_config", None)
                if run_config is not None:
                    # Prewarm the full prefill arena before KV sizing reads free HBM.
                    worker_kwargs["config"] = run_config
                worker = DistributedWorker(compiled, **worker_kwargs)
            self._l3_worker = worker
        return worker

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------
    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run one prefill chunk per TP group at its TP-aligned extent."""
        if self._compiled.prefill is None:
            raise RuntimeError("DSpark kernels were not compiled for this runner")
        if not batch.allow_device_greedy_sampling:
            raise RuntimeError(
                "DSpark serving currently supports greedy generation only "
                "(the kernels expose device greedy sampling; no temperature ABI yet)"
            )
        with profile_span("DSparkModelRunner.prefill", cat="executor"):
            self._ensure_l3_shared_buffers(model)
            inputs = self.prepare_prefill_inputs(model, batch)
            self._stage_prefill_inputs(inputs)
            self._prefill_task_args.clear_outputs()
            args = self._prefill_dispatch_args(inputs.physical_tokens)
            try:
                with profile_span(
                    "DSparkModelRunner.prefill.l3_dispatch",
                    cat="executor",
                    args={"actual_tokens": max(inputs.actual_tokens)},
                ):
                    self._run_l3(self._compiled.prefill, *args)
            except RuntimeError as exc:
                raise RuntimeError(
                    "DSpark packed prefill dispatch failed "
                    f"(tokens={inputs.actual_tokens}, groups={inputs.groups})"
                ) from exc
            sampled = self._prefill_task_args.tensors["sampled_ids"]
            tokens = [
                int(sampled[leader_rank, 0, 0].item())
                for leader_rank in (group * self._compiled.layout.tp_size for group in inputs.groups)
            ]
            return PrefillResult(
                last_hidden=None,
                logits=torch.zeros((len(tokens), 0)),
                sampled_token_ids=torch.tensor(tokens, dtype=torch.long),
            )

    def _prefill_kernel_tokens(self, actual_tokens: int, *, max_seq_len: int) -> int:
        """Return the TP-aligned packed extent for one logical chunk."""
        if actual_tokens <= 0 or actual_tokens > self._compiled.layout.prefill_tokens:
            raise ValueError(
                "DSpark prefill chunks must be in "
                f"[1, {self._compiled.layout.prefill_tokens}] tokens, got {actual_tokens}"
            )
        physical_tokens = (
            (actual_tokens + self._compiled.layout.tp_size - 1)
            // self._compiled.layout.tp_size
            * self._compiled.layout.tp_size
        )
        if physical_tokens > max_seq_len:
            raise ValueError(
                f"DSpark prefill physical extent {physical_tokens} exceeds max_seq_len={max_seq_len}"
            )
        return physical_tokens

    @staticmethod
    def _packed_host_prefix(tensor: torch.Tensor, rows: int) -> torch.Tensor:
        """Expose ``rows`` contiguous rows per rank from a max-sized shared slot."""
        if tensor.ndim < 2:
            raise ValueError(f"packed prefill tensor must have a rank and row axis, got {tensor.shape}")
        if rows <= 0 or rows > tensor.shape[1]:
            raise ValueError(f"packed prefill rows must be in [1, {tensor.shape[1]}], got {rows}")
        shape = (tensor.shape[0], rows, *tensor.shape[2:])
        numel = math.prod(shape)
        return tensor.reshape(-1)[:numel].view(shape)

    @staticmethod
    def _stacked_device_prefix(
        tensor: StackedDeviceTensor, rows: int
    ) -> StackedDeviceTensor:
        """Bind a compact logical row extent over existing per-rank device buffers."""
        tail = tensor.full_shape[1:]
        if len(tail) < 1 or rows <= 0 or rows > tail[0]:
            raise ValueError(
                f"device prefill rows must be in [1, {tail[0] if tail else 0}], got {rows}"
            )
        shard_shape = (rows, *tail[1:])
        shards = tuple(
            DeviceTensor(
                shard.data_ptr,
                shard_shape,
                shard.dtype,
                buffer=shard.buffer,
            )
            for shard in tensor.shards
        )
        return StackedDeviceTensor(
            shards,
            (tensor.full_shape[0], *shard_shape),
            tensor.worker_ids,
        )

    def _prefill_dispatch_args(self, physical_tokens: int) -> tuple[Any, ...]:
        """Build prefill args with the kernel's exact dynamic P/L descriptors."""
        task_args = self._prefill_task_args
        if task_args is None:
            raise RuntimeError("DSpark prefill TaskArgs are not staged")
        local_tokens = physical_tokens // self._compiled.layout.tp_size
        bounded: list[Any] = []
        for name, arg in zip(task_args.names, task_args.build(), strict=True):
            rows = None
            if name in _PREFILL_GROUP_DYNAMIC_NAMES:
                rows = physical_tokens
            elif name in _PREFILL_LOCAL_DYNAMIC_NAMES:
                rows = local_tokens
            if rows is None:
                bounded.append(arg)
            elif isinstance(arg, torch.Tensor):
                bounded.append(self._packed_host_prefix(arg, rows))
            elif isinstance(arg, StackedDeviceTensor):
                bounded.append(self._stacked_device_prefix(arg, rows))
            else:
                raise TypeError(
                    f"DSpark dynamic prefill arg {name!r} has unsupported type "
                    f"{type(arg).__name__}"
                )
        return tuple(bounded)

    def prepare_prefill_inputs(
        self, model: RuntimeModel, batch: PrefillBatch
    ) -> DSparkPreparedPrefillInputs:
        """Build TP-aligned host tensors for one packed prefill dispatch."""
        layout = self._compiled.layout
        request_count = len(batch.request_ids)
        if request_count <= 0 or request_count > layout.prefill_batch:
            raise ValueError(
                "DSpark prefill supports one request per TP group and at most "
                f"{layout.prefill_batch} per dispatch, got {request_count}"
            )
        if len(batch.cache_partitions) != request_count:
            raise ValueError("DSpark prefill requires one cache partition per request")
        groups = tuple(int(group) for group in batch.cache_partitions)
        if len(set(groups)) != request_count:
            raise ValueError("DSpark prefill accepts at most one request per TP group")
        if min(groups) < 0 or max(groups) >= layout.partitions:
            raise ValueError(
                f"DSpark prefill cache partitions must be in [0, {layout.partitions - 1}]"
            )
        if batch.input_embeddings is None:
            raise ValueError("DSpark prefill requires host input embeddings")
        if len(batch.chunk_lens) != request_count:
            raise ValueError("DSpark prefill requires one chunk length per request")

        builder = self.cache_metadata
        rope = self._require_rope_tables()
        tokens = self._prefill_kernel_tokens(
            max(int(length) for length in batch.chunk_lens),
            max_seq_len=model.runtime.max_seq_len,
        )
        local_tokens = tokens // layout.tp_size
        max_position = rope.max_position

        input_ids = torch.zeros((layout.ranks, local_tokens), dtype=torch.int64)
        position_ids_local = torch.zeros((layout.ranks, local_tokens), dtype=torch.int32)
        position_ids_full = torch.zeros((layout.ranks, tokens), dtype=torch.int32)
        # Packed-prefill boundaries (pypto-lib#1095): monotonic per-rank starts
        # ending at the group's logical length; [0, 0] leaves a group idle.
        query_start_loc = torch.zeros(
            (layout.ranks, DSPARK_PREFILL_MAX_REQUESTS + 1), dtype=torch.int32
        )
        slot_mappings = {
            name: torch.full((layout.ranks, tokens), -1, dtype=torch.int64)
            for name in (
                "ori_slot_mapping_full",
                "hca_cmp_slot_mapping_full",
                "hca_state_slot_mapping_full",
                "csa_cmp_slot_mapping_full",
                "csa_idx_slot_mapping_full",
                "csa_state_slot_mapping_full",
                "csa_inner_state_slot_mapping_full",
            )
        }
        block_tables = {
            name: torch.full(
                (layout.ranks, DSPARK_PREFILL_MAX_REQUESTS, depth), -1, dtype=torch.int32
            )
            for name, depth in (
                ("ori_block_table", DSPARK_PREFILL_ORI_TABLE_BLOCKS),
                ("hca_cmp_block_table", DSPARK_PREFILL_HCA_CMP_TABLE_BLOCKS),
                ("csa_cmp_block_table", DSPARK_PREFILL_CSA_CMP_TABLE_BLOCKS),
                ("idx_block_table", DSPARK_PREFILL_IDX_TABLE_BLOCKS),
                ("hca_compress_state_block_table", DSPARK_PREFILL_HCA_STATE_TABLE_BLOCKS),
                ("csa_compress_state_block_table", DSPARK_PREFILL_CSA_STATE_TABLE_BLOCKS),
                (
                    "csa_inner_compress_state_block_table",
                    DSPARK_PREFILL_CSA_INNER_STATE_TABLE_BLOCKS,
                ),
            )
        }
        logit_row_indices = torch.full(
            (layout.ranks, layout.max_logit_rows), -1, dtype=torch.int32
        )
        group_rows = self._normalize_group_block_ids(
            batch.block_ids_by_group, actual_batch=request_count
        )
        actual_tokens_by_group: list[int] = []
        chunk_starts: list[int] = []
        embeddings_by_request: list[torch.Tensor] = []
        # Per-rank rope rows: every TP group gathers its own table at its own
        # chunk positions (the kernel takes [N_RANKS, tokens, ROPE_HEAD_DIM]
        # and expects metadata identical only *within* a group, so a dispatch
        # with several groups must not share one request's rotary phase).
        rope_tables = {
            name: torch.zeros(
                (layout.ranks, tokens, DSPARK_ROPE_HEAD_DIM), dtype=torch.bfloat16
            )
            for name in (
                "swa_freqs_cos",
                "swa_freqs_sin",
                "compressed_freqs_cos",
                "compressed_freqs_sin",
                "csa_cmp_freqs_cos",
                "csa_cmp_freqs_sin",
                "hca_cmp_freqs_cos",
                "hca_cmp_freqs_sin",
            )
        }

        for index, group in enumerate(groups):
            actual_tokens = int(batch.chunk_lens[index])
            chunk_start = int(batch.chunk_starts[index])
            chunk_offset = int(batch.chunk_offsets[index])
            actual_tokens_by_group.append(actual_tokens)
            chunk_starts.append(chunk_start)
            if actual_tokens <= 0 or actual_tokens > DSPARK_PREFILL_MAX_TOKENS:
                raise ValueError(
                    f"DSpark prefill chunk must be in [1, {DSPARK_PREFILL_MAX_TOKENS}] tokens, "
                    f"got {actual_tokens}"
                )
            if chunk_start + tokens > model.runtime.max_seq_len:
                raise ValueError(
                    f"prefill physical positions [{chunk_start}, {chunk_start + tokens}) exceed "
                    f"max_seq_len={model.runtime.max_seq_len}"
                )
            ranks = tuple(
                range(group * layout.tp_size, (group + 1) * layout.tp_size)
            )
            positions = torch.arange(chunk_start, chunk_start + tokens, dtype=torch.int64)
            positions_c = positions.clamp(max=max_position - 1)
            group_rope_rows = {
                "swa_freqs_cos": rope.gather(rope.swa_cos, positions_c).to(torch.bfloat16),
                "swa_freqs_sin": rope.gather(rope.swa_sin, positions_c).to(torch.bfloat16),
                "compressed_freqs_cos": rope.gather(
                    rope.ratio128_cos, positions_c
                ).to(torch.bfloat16),
                "compressed_freqs_sin": rope.gather(
                    rope.ratio128_sin, positions_c
                ).to(torch.bfloat16),
            }
            cmp_positions = torch.where(
                (positions_c + 1) % 4 == 0,
                positions_c - 3,
                torch.zeros_like(positions_c),
            )
            group_rope_rows["csa_cmp_freqs_cos"] = rope.gather(
                rope.ratio4_cos, cmp_positions
            ).to(torch.bfloat16)
            group_rope_rows["csa_cmp_freqs_sin"] = rope.gather(
                rope.ratio4_sin, cmp_positions
            ).to(torch.bfloat16)
            hca_boundary = positions_c - positions_c % 128
            group_rope_rows["hca_cmp_freqs_cos"] = rope.gather(
                rope.ratio128_cos, hca_boundary
            ).to(torch.bfloat16)
            group_rope_rows["hca_cmp_freqs_sin"] = rope.gather(
                rope.ratio128_sin, hca_boundary
            ).to(torch.bfloat16)
            rank_lo = group * layout.tp_size
            for name, rows in group_rope_rows.items():
                rope_tables[name][rank_lo : rank_lo + layout.tp_size] = rows
            token_ids = (
                batch.token_ids[chunk_offset : chunk_offset + actual_tokens]
                .detach()
                .cpu()
                .to(torch.long)
            )
            embeddings_by_request.append(
                batch.input_embeddings[chunk_offset : chunk_offset + actual_tokens]
                .detach()
                .cpu()
                .to(torch.float32)
                .contiguous()
            )
            # pypto-lib#1069 padding contract: zero-padded input rows, natural
            # (non-aliasing) positions for the padding, -1 cache mappings.
            for rank in ranks:
                tp_rank = rank % layout.tp_size
                local_index = torch.arange(local_tokens, dtype=torch.int64)
                local_positions = positions_c[tp_rank * local_tokens + local_index]
                position_ids_local[rank] = local_positions.to(torch.int32)
                local_ids = torch.zeros(local_tokens, dtype=torch.int64)
                active_local = (
                    (tp_rank * local_tokens + local_index) < actual_tokens
                )
                local_ids[active_local] = token_ids[
                    (tp_rank * local_tokens + local_index)[active_local]
                ]
                input_ids[rank] = local_ids
                position_ids_full[rank] = positions_c.to(torch.int32)
                logit_row_indices[rank] = -1
            logit_row_indices[group * layout.tp_size, 0] = actual_tokens - 1

            blocks = group_rows[index]
            tables = {
                "ori_block_table": builder.ring_table(
                    blocks["ori"], depth=DSPARK_PREFILL_ORI_TABLE_BLOCKS
                ),
                "hca_cmp_block_table": builder.absolute_table(
                    blocks["cmp_c128"], depth=DSPARK_PREFILL_HCA_CMP_TABLE_BLOCKS
                ),
                "csa_cmp_block_table": builder.absolute_table(
                    blocks["cmp_c4"], depth=DSPARK_PREFILL_CSA_CMP_TABLE_BLOCKS
                ),
                "idx_block_table": builder.absolute_table(
                    blocks["idx"], depth=DSPARK_PREFILL_IDX_TABLE_BLOCKS
                ),
                "hca_compress_state_block_table": builder.ring_table(
                    blocks["hca_state"], depth=DSPARK_PREFILL_HCA_STATE_TABLE_BLOCKS
                ),
                "csa_compress_state_block_table": builder.ring_table(
                    blocks["csa_state"], depth=DSPARK_PREFILL_CSA_STATE_TABLE_BLOCKS
                ),
                "csa_inner_compress_state_block_table": builder.ring_table(
                    blocks["csa_inner_state"],
                    depth=DSPARK_PREFILL_CSA_INNER_STATE_TABLE_BLOCKS,
                ),
            }
            logical_positions = positions[:actual_tokens].reshape(1, -1)
            logical_positions_c = positions_c[:actual_tokens].reshape(1, -1)
            mappings = {
                "ori_slot_mapping_full": builder.paged_slot_mapping(
                    logical_positions_c, tables["ori_block_table"].unsqueeze(0),
                    block_size=layout.block_size,
                ),
                "hca_cmp_slot_mapping_full": builder.compressed_slot_mapping(
                    logical_positions_c,
                    tables["hca_cmp_block_table"].unsqueeze(0),
                    compress_ratio=128,
                ),
                "hca_state_slot_mapping_full": builder.state_slot_mapping(
                    logical_positions_c,
                    tables["hca_compress_state_block_table"].unsqueeze(0),
                    state_page_tokens=DSPARK_C128_STATE_PAGE_TOKENS,
                ),
                "csa_cmp_slot_mapping_full": builder.compressed_slot_mapping(
                    logical_positions_c,
                    tables["csa_cmp_block_table"].unsqueeze(0),
                    compress_ratio=4,
                ),
                "csa_idx_slot_mapping_full": builder.compressed_slot_mapping(
                    logical_positions_c,
                    tables["idx_block_table"].unsqueeze(0),
                    compress_ratio=4,
                ),
                "csa_state_slot_mapping_full": builder.state_slot_mapping(
                    logical_positions_c,
                    tables["csa_compress_state_block_table"].unsqueeze(0),
                    state_page_tokens=DSPARK_C4_STATE_PAGE_TOKENS,
                ),
                "csa_inner_state_slot_mapping_full": builder.state_slot_mapping(
                    logical_positions_c,
                    tables["csa_inner_compress_state_block_table"].unsqueeze(0),
                    state_page_tokens=DSPARK_C4_STATE_PAGE_TOKENS,
                ),
            }
            for rank in ranks:
                query_start_loc[rank, DSPARK_PREFILL_MAX_REQUESTS] = actual_tokens
                for name, table in tables.items():
                    block_tables[name][rank, 0] = table
                for name, mapping in mappings.items():
                    slot_mappings[name][rank, :actual_tokens] = mapping.reshape(-1)

        return DSparkPreparedPrefillInputs(
            request_ids=tuple(batch.request_ids),
            groups=groups,
            actual_tokens=tuple(actual_tokens_by_group),
            physical_tokens=tokens,
            chunk_starts=tuple(chunk_starts),
            embeddings=tuple(embeddings_by_request),
            input_ids=input_ids,
            position_ids_local=position_ids_local,
            position_ids_full=position_ids_full,
            query_start_loc=query_start_loc,
            rope_tables=rope_tables,
            slot_mappings=slot_mappings,
            block_tables=block_tables,
            logit_row_indices=logit_row_indices,
            sampled_slots=tuple(
                (group * layout.tp_size, 0) for group in groups
            ),
        )

    def _stage_prefill_inputs(self, inputs: DSparkPreparedPrefillInputs) -> None:
        """Pack one dispatch into compact views over max-sized shared buffers."""
        task_args = self._prefill_task_args
        if task_args is None:
            raise RuntimeError("DSpark prefill TaskArgs are not staged")
        tensors = task_args.tensors
        values: dict[str, torch.Tensor] = {
            "input_ids": inputs.input_ids,
            "position_ids_local": inputs.position_ids_local,
            "position_ids_full": inputs.position_ids_full,
            "query_start_loc": inputs.query_start_loc,
            "logit_row_indices": inputs.logit_row_indices,
        }
        values.update(inputs.rope_tables)
        values.update(inputs.slot_mappings)
        values.update(inputs.block_tables)
        # The inherited slot keeps its maximum allocation, while the dispatch
        # view packs P rows per rank contiguously at the front of that storage.
        # Repacking is required because a simple [:, :P] view retains the max-P
        # rank stride and cannot cross the address-free tensor wire ABI.
        layout = self._compiled.layout
        x_hc = self._packed_host_prefix(tensors["x_hc"], inputs.physical_tokens)
        x_hc.zero_()
        for group, embeddings in zip(inputs.groups, inputs.embeddings, strict=True):
            replicated = embeddings.unsqueeze(1).expand(-1, layout.hc_mult, -1)
            for rank in range(group * layout.tp_size, (group + 1) * layout.tp_size):
                x_hc[rank, : embeddings.shape[0]].copy_(replicated)
        for name, value in values.items():
            destination = tensors[name]
            if name in _PREFILL_GROUP_DYNAMIC_NAMES:
                destination = self._packed_host_prefix(destination, inputs.physical_tokens)
            elif name in _PREFILL_LOCAL_DYNAMIC_NAMES:
                destination = self._packed_host_prefix(
                    destination, inputs.physical_tokens // layout.tp_size
                )
            copy_shared(destination, value, name=f"dspark_prefill_{name}")

        self._mirror_inactive_prefill_groups(inputs, values)

    def _mirror_inactive_prefill_groups(
        self,
        inputs: DSparkPreparedPrefillInputs,
        values: dict[str, torch.Tensor],
    ) -> None:
        """Replay one real prefill into every otherwise idle TP partition.

        Block IDs are partition-local. Copying the active group's complete
        prefill metadata therefore populates the corresponding cache pages in
        each inactive partition, ready for the matching decode replay.
        """
        layout = self._compiled.layout
        active_groups = set(inputs.groups)
        if len(active_groups) == layout.partitions:
            return
        source_group = inputs.groups[0]
        tensors = self._prefill_task_args.tensors
        mirrored_groups = [
            group for group in range(layout.partitions) if group not in active_groups
        ]
        for name in ("x_hc", *values):
            if name == "logit_row_indices":
                continue
            tensor = tensors[name]
            if name in _PREFILL_GROUP_DYNAMIC_NAMES:
                tensor = self._packed_host_prefix(tensor, inputs.physical_tokens)
            elif name in _PREFILL_LOCAL_DYNAMIC_NAMES:
                tensor = self._packed_host_prefix(
                    tensor, inputs.physical_tokens // layout.tp_size
                )
            for group in mirrored_groups:
                for tp_rank in range(layout.tp_size):
                    source_rank = source_group * layout.tp_size + tp_rank
                    target_rank = group * layout.tp_size + tp_rank
                    tensor[target_rank].copy_(tensor[source_rank])
        logger.warning(
            "DSpark mirrored prefill TP group %d into inactive groups %s",
            source_group,
            mirrored_groups,
        )

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run one full-tile decode step and accept each anchor row."""
        if self._compiled.decode is None:
            raise RuntimeError("DSpark kernels were not compiled for this runner")
        if not batch.allow_device_greedy_sampling:
            raise RuntimeError(
                "DSpark serving currently supports greedy decoding only "
                "(the kernels expose device greedy sampling; no temperature ABI yet)"
            )
        with profile_span("DSparkModelRunner.decode", cat="executor"):
            self._ensure_l3_shared_buffers(model)
            buffer_slot = int(getattr(batch, "buffer_slot", 0)) or 0
            inputs = self.prepare_decode_inputs(model, batch, buffer_slot=buffer_slot)
            task_args = self._decode_task_args[inputs.buffer_slot]
            args = task_args.build()
            try:
                with profile_span(
                    "DSparkModelRunner.decode.l3_dispatch",
                    cat="executor",
                    args={"actual_batch": len(batch.request_ids)},
                ):
                    self._run_l3(
                        self._compiled.decode, *args, config=self._decode_run_config
                    )
            except RuntimeError as exc:
                raise RuntimeError(
                    "DSpark packed decode dispatch failed "
                    f"(actual_batch={len(batch.request_ids)})"
                ) from exc
            sampled = task_args.tensors["sampled_ids"]
            accepted = [
                [int(sampled[rank, row, 0].item())] for rank, row in inputs.sampled_slots
            ]
            return DecodeResult(
                hidden_states=None,
                logits=None,
                accepted_token_ids=accepted,
            )

    def _decode_assignment(self, batch: DecodeBatch) -> _DSparkGroupAssignment:
        """Assign batch rows to TP groups and rank-local request slots."""
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if actual_batch <= 0:
            raise ValueError("DSpark decode batch must not be empty")
        if len(batch.cache_partitions) != actual_batch:
            raise ValueError("DSpark decode requires one cache partition per request")
        groups = tuple(int(group) for group in batch.cache_partitions)
        if min(groups) < 0 or max(groups) >= layout.partitions:
            raise ValueError(
                f"DSpark decode cache partitions must be in [0, {layout.partitions - 1}]"
            )
        requests_by_group: list[list[int]] = [[] for _ in range(layout.partitions)]
        for request_index, group in enumerate(groups):
            ordinal = len(requests_by_group[group])
            if ordinal >= layout.decode_batch:
                raise ValueError(
                    f"DSpark TP group {group} decode batch exceeds local capacity "
                    f"{layout.decode_batch}"
                )
            requests_by_group[group].append(request_index)
        # The fused decode graph is validated at its fixed physical tile. In
        # particular, qkv_proj_rope's KV tail path faults below its 8-row
        # vector tile and several downstream kernels assume the full aligned
        # T/KV_T extents. Keep inactive rows benign instead of exposing a
        # smaller dynamic shape to the device ABI.
        local_batch = layout.decode_local_batch
        active_by_group: list[list[tuple[int, int]]] = [[] for _ in range(layout.partitions)]
        ordinals = [0] * actual_batch
        for group, request_indices in enumerate(requests_by_group):
            for ordinal, request_index in enumerate(request_indices):
                # The group stream is rank-major: slot ``s`` lives on TP rank
                # ``s // local_batch`` at its local row ``s % local_batch``.
                # Spreading ordinals round-robin over the ranks keeps every
                # rank's local rows dense while preserving that stream index.
                stream_slot = (
                    (ordinal % layout.tp_size) * local_batch
                    + ordinal // layout.tp_size
                )
                active_by_group[group].append((request_index, stream_slot))
                ordinals[request_index] = ordinal
        return _DSparkGroupAssignment(
            groups=tuple(groups),
            ordinals=tuple(ordinals),
            active_by_group=tuple(tuple(rows) for rows in active_by_group),
        )

    def prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        buffer_slot: int = 0,
    ) -> DSparkPreparedDecodeInputs:
        """Build packed metadata for one decode step."""
        layout = self._compiled.layout
        assignment = self._decode_assignment(batch)
        task_args = self._decode_task_args[buffer_slot]
        local_batch = layout.decode_local_batch
        staged = task_args.tensors
        group_batch = local_batch * layout.tp_size
        local_tokens = local_batch * layout.decode_seq
        builder = self.cache_metadata
        rope = self._require_rope_tables()
        max_position = rope.max_position
        actual_batch = len(batch.request_ids)

        anchors = [
            max(int(batch.seq_lens[index].item()) - 1, 0) for index in range(actual_batch)
        ]
        token_rows = (
            batch.token_ids[:actual_batch]
            .detach()
            .cpu()
            .to(torch.long)
            .reshape(actual_batch, -1)[:, 0]
        )

        # ---- per-group group-row and query-row positions ----
        # group_positions[g, slot, s]: the group stream row positions; filler
        # slots use a benign 0..7 window on scratch pages.
        filler_positions = torch.arange(layout.decode_seq, dtype=torch.int64)
        group_positions = filler_positions.view(1, 1, -1).expand(
            layout.partitions, group_batch, layout.decode_seq
        ).clone()
        group_tokens = torch.full(
            (layout.partitions, group_batch, layout.decode_seq),
            DSPARK_NOISE_TOKEN_ID,
            dtype=torch.int64,
        )
        group_anchor_flags = torch.zeros(
            (layout.partitions, group_batch), dtype=torch.bool
        )
        request_slot_of_group: dict[tuple[int, int], int] = {}
        for group in range(layout.partitions):
            for request_index, stream_slot in assignment.active_by_group[group]:
                anchor = anchors[request_index]
                positions = torch.arange(layout.decode_seq, dtype=torch.int64) + anchor
                positions = positions.clamp(max=max_position - 1)
                group_positions[group, stream_slot] = positions
                group_tokens[group, stream_slot, 0] = int(token_rows[request_index])
                group_anchor_flags[group, stream_slot] = True
                request_slot_of_group[(group, stream_slot)] = request_index

        group_rows = self._normalize_group_block_ids(
            batch.block_ids_by_group, actual_batch=actual_batch
        )
        request_blocks: dict[int, dict[str, tuple[int, ...]]] = {}
        for request_index in range(actual_batch):
            request_blocks[request_index] = group_rows[request_index]

        # ---- stage per-rank tensors ----
        logit_rows = staged["logit_row_indices"]
        logit_rows.fill_(-1)
        owner_token_counts = staged["num_tokens_per_owner"]
        owner_token_counts.zero_()
        sampled_slots: list[tuple[int, int]] = [(-1, -1)] * actual_batch
        scratch = self._scratch_blocks()

        for group in range(layout.partitions):
            ranks = tuple(range(group * layout.tp_size, (group + 1) * layout.tp_size))
            positions = group_positions[group]  # [decode_batch, seq]
            positions_flat = positions.reshape(-1)
            tokens_flat = group_tokens[group].reshape(-1)
            anchor_flags = group_anchor_flags[group]
            starts = positions[:, 0]
            # Per-request tables for the compact uniform group extent.
            ori_tables = torch.stack(
                [
                    builder.ring_table(
                        request_blocks[request_slot_of_group[(group, row)]]["ori"]
                        if anchor_flags[row]
                        else (scratch["ori"][row],),
                        depth=DSPARK_DECODE_ORI_TABLE_BLOCKS,
                    )
                    for row in range(group_batch)
                ]
            )
            hca_cmp_tables = torch.stack(
                [
                    builder.absolute_table(
                        request_blocks[request_slot_of_group[(group, row)]]["cmp_c128"]
                        if anchor_flags[row]
                        else (scratch["cmp_c128"][row],),
                        depth=DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS,
                    )
                    for row in range(group_batch)
                ]
            )
            csa_cmp_tables = torch.stack(
                [
                    builder.absolute_table(
                        request_blocks[request_slot_of_group[(group, row)]]["cmp_c4"]
                        if anchor_flags[row]
                        else (scratch["cmp_c4"][row],),
                        depth=DSPARK_DECODE_CMP_C4_TABLE_BLOCKS,
                    )
                    for row in range(group_batch)
                ]
            )
            idx_tables = torch.stack(
                [
                    builder.absolute_table(
                        request_blocks[request_slot_of_group[(group, row)]]["idx"]
                        if anchor_flags[row]
                        else (scratch["idx"][row],),
                        depth=DSPARK_DECODE_IDX_TABLE_BLOCKS,
                    )
                    for row in range(group_batch)
                ]
            )
            hca_state_tables = torch.stack(
                [
                    builder.ring_table(
                        request_blocks[request_slot_of_group[(group, row)]]["hca_state"]
                        if anchor_flags[row]
                        else (scratch["hca_state"][row],),
                        depth=DSPARK_DECODE_HCA_STATE_TABLE_BLOCKS,
                    )
                    for row in range(group_batch)
                ]
            )
            csa_state_tables = torch.stack(
                [
                    builder.trailing_ring_table(
                        request_blocks[request_slot_of_group[(group, row)]]["csa_state"],
                        position=int(starts[row].item()),
                        page_tokens=DSPARK_C4_STATE_PAGE_TOKENS,
                        depth=DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS,
                    )
                    if anchor_flags[row]
                    else builder.ring_table(
                        (scratch["csa_state"][row],),
                        depth=DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS,
                    )
                    for row in range(group_batch)
                ]
            )
            csa_inner_tables = torch.stack(
                [
                    builder.trailing_ring_table(
                        request_blocks[request_slot_of_group[(group, row)]]["csa_inner_state"],
                        position=int(starts[row].item()),
                        page_tokens=DSPARK_C4_STATE_PAGE_TOKENS,
                        depth=DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS,
                    )
                    if anchor_flags[row]
                    else builder.ring_table(
                        (scratch["csa_inner_state"][row],),
                        depth=DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS,
                    )
                    for row in range(group_batch)
                ]
            )
            commit = 1
            committed_rows = anchor_flags.unsqueeze(-1) & (
                torch.arange(positions.shape[-1]).unsqueeze(0) < commit
            )
            # The decode kernel addresses a 16-row transaction ring: eight historical
            # rows followed by the eager S=8 projection writes.
            # Commit-gate the seven unaccepted rows before mapping into it.
            csa_state_ring_positions = positions % DSPARK_CSA_DECODE_STATE_RING_TOKENS
            csa_state_slots = builder.paged_slot_mapping(
                csa_state_ring_positions, csa_state_tables,
                block_size=DSPARK_C4_STATE_PAGE_TOKENS,
            )
            csa_inner_state_slots = builder.paged_slot_mapping(
                csa_state_ring_positions, csa_inner_tables,
                block_size=DSPARK_C4_STATE_PAGE_TOKENS,
            )
            raw_slots = torch.where(
                committed_rows,
                builder.paged_slot_mapping(
                    positions, ori_tables, block_size=layout.block_size
                ),
                torch.full_like(positions, -1),
            ).reshape(-1)
            mappings = {
                "swa_slot_mapping": raw_slots,
                "hca_ori_slot_mapping": raw_slots,
                "csa_ori_slot_mapping": raw_slots,
                "hca_cmp_slot_mapping": builder.compressed_slot_mapping(
                    positions, hca_cmp_tables, compress_ratio=128,
                    commit_tokens=commit,
                ).reshape(group_batch, layout.decode_seq),
                "csa_cmp_slot_mapping": builder.compressed_slot_mapping(
                    positions, csa_cmp_tables, compress_ratio=4,
                    commit_tokens=commit,
                ).reshape(group_batch, layout.decode_seq),
                "csa_idx_slot_mapping": builder.compressed_slot_mapping(
                    positions, idx_tables, compress_ratio=4,
                    commit_tokens=commit,
                ).reshape(group_batch, layout.decode_seq),
                "hca_state_slot_mapping": torch.where(
                    committed_rows,
                    builder.state_slot_mapping(
                        positions,
                        hca_state_tables,
                        state_page_tokens=DSPARK_C128_STATE_PAGE_TOKENS,
                    ),
                    torch.full_like(positions, -1),
                ).reshape(-1),
                "csa_state_slot_mapping": torch.where(
                    committed_rows,
                    csa_state_slots,
                    torch.full_like(csa_state_slots, -1),
                ).reshape(-1),
                "csa_inner_state_slot_mapping": torch.where(
                    committed_rows,
                    csa_inner_state_slots,
                    torch.full_like(csa_inner_state_slots, -1),
                ).reshape(-1),
            }
            for name in (
                "hca_cmp_slot_mapping",
                "csa_cmp_slot_mapping",
                "csa_idx_slot_mapping",
            ):
                mappings[name] = torch.where(
                    anchor_flags.unsqueeze(-1),
                    mappings[name],
                    torch.full_like(mappings[name], -1),
                ).reshape(-1)
            kv_seq_lens = torch.where(
                anchor_flags,
                (starts + commit).to(torch.int32),
                torch.zeros_like(starts, dtype=torch.int32),
            )
            # Every attention family consumes the same raw-KV window lowering.
            swa_indices, swa_lens = builder.swa_window_indices_and_lens(positions, ori_tables)
            boundary_positions = (starts - starts % 128).clamp(min=0)
            hca_cmp_cos = rope.gather(rope.ratio128_half_cos, boundary_positions)
            hca_cmp_sin = rope.gather(rope.ratio128_half_sin, boundary_positions)
            cmp_positions_flat = torch.where(
                (positions_flat + 1) % 4 == 0,
                positions_flat - 3,
                torch.zeros_like(positions_flat),
            )
            group_cos = rope.gather(rope.swa_cos, positions_flat).to(torch.bfloat16)
            group_sin = rope.gather(rope.swa_sin, positions_flat).to(torch.bfloat16)
            compressed_group_cos = rope.gather(
                rope.ratio128_cos, positions_flat
            ).to(torch.bfloat16)
            compressed_group_sin = rope.gather(
                rope.ratio128_sin, positions_flat
            ).to(torch.bfloat16)
            csa_cmp_cos = rope.gather(
                rope.ratio4_cos, cmp_positions_flat.clamp(min=0)
            ).to(torch.bfloat16)
            csa_cmp_sin = rope.gather(
                rope.ratio4_sin, cmp_positions_flat.clamp(min=0)
            ).to(torch.bfloat16)
            for rank in ranks:
                tp_rank = rank % layout.tp_size
                local_tokens_slice = slice(
                    tp_rank * local_tokens,
                    (tp_rank + 1) * local_tokens,
                )
                local_requests_slice = slice(
                    tp_rank * local_batch,
                    (tp_rank + 1) * local_batch,
                )
                active_requests = int(anchor_flags[local_requests_slice].sum().item())
                active_tokens = active_requests * layout.decode_seq
                owner_token_counts[rank] = active_tokens
                for name, value in mappings.items():
                    staged[name][rank] = value.to(staged[name].dtype)
                staged["position_ids"][rank] = positions_flat.to(torch.int32)
                staged["freqs_cos"][rank] = group_cos
                staged["freqs_sin"][rank] = group_sin
                staged["compressed_freqs_cos"][rank] = compressed_group_cos
                staged["compressed_freqs_sin"][rank] = compressed_group_sin
                # The rank's query rows are its contiguous slice of the group
                # stream, so the local RoPE tables are the same slice.
                staged["freqs_cos_local"][rank] = group_cos[local_tokens_slice].contiguous()
                staged["freqs_sin_local"][rank] = group_sin[local_tokens_slice].contiguous()
                staged["compressed_freqs_cos_local"][rank] = compressed_group_cos[
                    local_tokens_slice
                ].contiguous()
                staged["compressed_freqs_sin_local"][rank] = compressed_group_sin[
                    local_tokens_slice
                ].contiguous()
                staged["position_ids_local"][rank] = positions_flat[local_tokens_slice].to(
                    torch.int32
                )
                local_input_ids = tokens_flat[local_tokens_slice].clone()
                local_input_ids[active_tokens:] = 0
                staged["input_ids"][rank] = local_input_ids
                staged["csa_cmp_freqs_cos"][rank] = csa_cmp_cos
                staged["csa_cmp_freqs_sin"][rank] = csa_cmp_sin
                staged["hca_cmp_freqs_cos"][rank] = hca_cmp_cos
                staged["hca_cmp_freqs_sin"][rank] = hca_cmp_sin
                # Rank-local request tables and lengths.
                staged["csa_cmp_block_table"][rank] = csa_cmp_tables[local_requests_slice]
                staged["csa_idx_block_table"][rank] = idx_tables[local_requests_slice]
                staged["hca_cmp_block_table"][rank] = hca_cmp_tables[local_requests_slice]
                staged["csa_kv_seq_lens"][rank] = kv_seq_lens[local_requests_slice]
                staged["hca_kv_seq_lens"][rank] = kv_seq_lens[local_requests_slice]
                for name, value in (
                    ("swa_indices", swa_indices),
                    ("swa_lens", swa_lens),
                    ("csa_window_swa_indices", swa_indices),
                    ("csa_window_swa_lens", swa_lens),
                    ("hca_window_swa_indices", swa_indices),
                    ("hca_window_swa_lens", swa_lens),
                ):
                    local_value = value[local_tokens_slice].clone()
                    if name.endswith("indices"):
                        local_value[active_tokens:] = -1
                    else:
                        local_value[active_tokens:] = 0
                    staged[name][rank] = local_value.to(staged[name].dtype)
                # Group-replicated state tables.
                staged["hca_compress_state_block_table"][rank] = hca_state_tables
                staged["csa_compress_state_block_table"][rank] = csa_state_tables
                staged["csa_inner_compress_state_block_table"][rank] = csa_inner_tables
                # Logit rows: one anchor entry per active request on this rank.
                entry = 0
                for local_index in range(local_batch):
                    stream_row = tp_rank * local_batch + local_index
                    if not bool(anchor_flags[stream_row]):
                        continue
                    logit_rows[rank, entry] = local_index * layout.decode_seq
                    request_index = request_slot_of_group[(group, stream_row)]
                    sampled_slots[request_index] = (rank, entry)
                    entry += 1

        return DSparkPreparedDecodeInputs(
            request_ids=tuple(batch.request_ids),
            groups=assignment.groups,
            group_ordinals=assignment.ordinals,
            anchor_positions=tuple(anchors),
            input_ids=staged["input_ids"],
            position_ids_local=staged["position_ids_local"],
            position_ids=staged["position_ids"],
            logit_row_indices=logit_rows,
            sampled_slots=tuple(sampled_slots),
            buffer_slot=buffer_slot,
        )

    def _scratch_blocks(self) -> dict[str, tuple[int, ...]]:
        """One isolated scratch page per kernel row and cache family."""
        scratch: dict[str, tuple[int, ...]] = {}
        for name in DSPARK_CACHE_GROUP_NAMES:
            base = self._cache_group_num_blocks[name]
            scratch[name] = tuple(base + row for row in range(DSPARK_DECODE_BATCH))
        return scratch

    def _normalize_group_block_ids(
        self,
        rows: Sequence[dict[str, Sequence[int]]],
        *,
        actual_batch: int,
    ) -> tuple[dict[str, tuple[int, ...]], ...]:
        """Validate and normalize grouped scheduler metadata for active rows."""
        if not rows or len(rows) != actual_batch:
            raise ValueError(
                f"grouped KV metadata has {len(rows) if rows else 0} rows, "
                f"expected batch {actual_batch}"
            )
        normalized = []
        for row_index, row in enumerate(rows):
            missing = [name for name in DSPARK_CACHE_GROUP_NAMES if not row.get(name)]
            if missing:
                raise ValueError(
                    f"row {row_index} is missing grouped KV blocks: {', '.join(missing)}"
                )
            entry = {}
            for name in DSPARK_CACHE_GROUP_NAMES:
                blocks = tuple(int(block_id) for block_id in row[name])
                if any(block_id < 0 or block_id >= self._cache_group_num_blocks[name] for block_id in blocks):
                    raise ValueError(
                        f"grouped KV block IDs for {name} must be in "
                        f"[0, {self._cache_group_num_blocks[name]}); "
                        f"[{self._cache_group_num_blocks[name]}, "
                        f"{self._cache_group_num_blocks[name] + DSPARK_DECODE_BATCH}) "
                        "is reserved for kernel padding"
                    )
                entry[name] = blocks
            normalized.append(entry)
        return tuple(normalized)

    def _require_rope_tables(self) -> DSparkRopeTables:
        if self._compiled.rope is None:
            raise RuntimeError("DSpark RoPE tables are not initialized")
        return self._compiled.rope

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """No request-local runner state in the target-only milestone."""
        del request_ids

    def close(self) -> None:
        worker = self._l3_worker
        try:
            if worker is not None:
                worker.close()
        finally:
            self._l3_worker = None
            self._cache_group_num_blocks.clear()
            self._stacked_host_weights = None
            self._stacked_prefill_host_weights = None
            self._stacked_device_weights = None
            self._stacked_prefill_device_weights = None
            self._embedding_device_weight = None
            self._device_scratch.clear()
            self._decode_device_cache = None
            self._global_weights = None
            self._hc_head_buffers = None
            self._l3_shared_buffers_ready = False
            self._l3_static_tensors.clear()
            if self._prefill_task_args is not None:
                self._prefill_task_args.close()
                self._prefill_task_args = None
            for task_args in self._decode_task_args:
                task_args.close()
            self._decode_task_args = []
