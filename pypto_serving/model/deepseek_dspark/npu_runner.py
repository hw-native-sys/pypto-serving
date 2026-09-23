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

* The 16 NPU ranks form 4 TP groups.  One group owns packed requests' prefill
  (all 4 ranks share the packed stream through context-parallel attention) and up
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

import json
import logging
import math
import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
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
    SamplingParams,
)
from pypto_serving.model.common.runner.buffer_set import copy_shared
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin, PendingL3Dispatch
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
# The target forwards tap layers 40/41/42 through one hc_head projection each
# and concatenate the three rows: dspark_target_hidden is [rows, 3*D] BF16.
DSPARK_MAIN_HIDDEN_DIM = 3 * DSPARK_HIDDEN_SIZE
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

# ---- per-dispatch ring heaps ----
def _parse_decode_ring_heap(value: str | None) -> tuple[int, ...]:
    """Parse per-depth byte counts; RunConfig validates the ring sizes."""
    if value is None:
        return (1 << 30, 1 << 30, 1 << 30, 4 << 30)
    sizes = tuple(int(part) for part in value.split(","))
    return sizes * 4 if len(sizes) == 1 else sizes


# At 256 HCA pages the deepest decode scope retains a 1 GiB FP32 partial-O
# tensor plus partial-M/L and stream state. Retained tensors across scopes
# can exceed 2 GiB; the shallower scopes retain their original sizes.
DSPARK_DECODE_RING_HEAP = _parse_decode_ring_heap(os.environ.get("PYPTO_DSPARK_DECODE_RING_HEAP"))
DSPARK_MARKOV_RING_HEAP = 1 << 30
DSPARK_PREFILL_RING_HEAP = (
    2 * 1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024,
    4 * 1024 * 1024 * 1024,
    8 * 1024 * 1024 * 1024,
)
# dspark_drafter.py pins (4 GiB,)*4 for its own scope depths.  EP8 doubles
# each rank's routed experts, and the 16 GiB pooled arena no longer fits next
# to the weights, so smaller worlds can shrink it (the profile is a generous
# bring-up default, not a kernel-validated minimum).
DSPARK_DRAFTER_RING_HEAP = tuple(
    int(value)
    for value in os.environ.get(
        "PYPTO_DSPARK_DRAFTER_RING_HEAP",
        "4294967296,4294967296,4294967296,4294967296",
    ).split(",")
)

# ---- speculative drafter (milestone 2) ----
# Kernel-fixed speculation constants (dspark_drafter.py / dspark_markov.py):
# K is DSPARK_QUERY_WIDTH, the per-rank drafter batch must be one of the
# supported paddings, and the decode tile already equals 1 + K rows.
DSPARK_SPECULATIVE_TOKENS = 7
DSPARK_DRAFTER_QUERY_WIDTH = 7
DSPARK_DRAFTER_ROPE_CANDIDATE_ROWS = 16
DSPARK_DRAFTER_BATCHES = (4, 8, 12, 16)
DSPARK_DRAFTER_MAX_BATCH = 16
DSPARK_DRAFT_LAYERS = 3
# Per-lease ring: a 128-deep sliding window plus the seven query rows fits in
# ceil((31 + 128 + 7) / 32) = 6 blocks for any window alignment.
DSPARK_DRAFTER_RING_BLOCKS = 6
DSPARK_DRAFTER_LEASES_PER_GROUP = 64
# 64 leases * 6 blocks per layer; the trailing shared range is a read-only,
# zero-initialized filler history that no live lease can reach.
DSPARK_DRAFTER_FILLER_BLOCK_BASE = (
    DSPARK_DRAFTER_LEASES_PER_GROUP * DSPARK_DRAFTER_RING_BLOCKS
)
# Block tables are [ranks, layers, batch, ORI_MAX_BLOCKS] with ORI_MAX_BLOCKS
# covering the 1M-position ceiling at 32-token pages.
DSPARK_DRAFTER_TABLE_BLOCKS = 32768
# Drafter-private SWA pools: KV_ORI_BLOCK_NUM = 512 blocks of 32 tokens per
# draft layer per rank (each rank holds a full group replica).
DSPARK_DRAFTER_KV_BLOCKS = 512
# Persistent K=7 device state, replicated over the four ranks of one TP group
# and indexed by the stable drafter lease.
_DSPARK_STATE_VALID = 0
_DSPARK_STATE_GENERATION = 1
_DSPARK_STATE_ANCHOR_POSITION = 2
_DSPARK_STATE_COMMITTED_COUNT = 3
_DSPARK_STATE_DRAFT_COUNT = 4
_DSPARK_STATE_POSITION_LIMIT = 5
_DSPARK_STATE_META_WIDTH = 6
_DSPARK_STATE_TOKEN_WIDTH = 1 + DSPARK_DRAFTER_QUERY_WIDTH
# Max per-rank context rows the drafter accepts: max(decode 16*8, prefill
# 512/4) -- both land on the same 128-row extent.
DSPARK_DRAFTER_CONTEXT_ROWS = 128
# The group-context tensors and block tables stage through one shared buffer
# per extent (their dynamic axis is not a per-rank storage prefix, so a view
# cannot cross the L3 wire).  Decode lands on batch*8 naturally; prefill
# seeding rounds its tail up to the next bucket.
DSPARK_DRAFTER_CONTEXT_BUCKETS = (32, 64, 96, 128)

# ---- paging ----
DSPARK_BLOCK_SIZE = 32
DSPARK_COMPRESSED_BLOCK_TOKENS = 128
DSPARK_HCA_CMP_STORAGE_BLOCK_SIZE = DSPARK_COMPRESSED_BLOCK_TOKENS // 128
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
# The LM-head / greedy-sampling windows cover one owner's rows per rank
# (pypto-lib#1182 right-sized them from the whole step's DECODE_TOKENS to
# MOE_TOKENS): decode packs at most local_batch * decode_seq = 128 logit
# rows per rank, prefill selects each request's last row, and markov at most 16 * 7.
DSPARK_MAX_LOGIT_ROWS = DSPARK_MOE_TOKENS

# The fused one-L2 kernel derives these values from persistent device state and
# allocates their storage inside the invocation, matching fused MTP.  They stay
# in the generic decode TaskArgs for the non-fused path, but must not be bound
# into the DSpark fused L3 ABI.
_DSPARK_FUSED_INTERNAL_PREPARE_NAMES = frozenset(
    {
        "freqs_cos",
        "freqs_sin",
        "compressed_freqs_cos",
        "compressed_freqs_sin",
        "swa_slot_mapping",
        "swa_indices",
        "swa_lens",
        "position_ids_local",
        "position_ids",
        "csa_cmp_freqs_cos",
        "csa_cmp_freqs_sin",
        "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "csa_ori_slot_mapping",
        "csa_window_swa_indices",
        "csa_window_swa_lens",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
        "csa_kv_seq_lens",
        "hca_cmp_freqs_cos",
        "hca_cmp_freqs_sin",
        "hca_compress_state_block_table",
        "hca_ori_slot_mapping",
        "hca_window_swa_indices",
        "hca_window_swa_lens",
        "hca_cmp_slot_mapping",
        "hca_state_slot_mapping",
        "hca_kv_seq_lens",
        "input_ids",
        "logit_row_indices",
    }
)
DSPARK_SAMPLED_IDS_PAD = 8
DSPARK_MAX_SEQ_LEN = 1_048_576

# ---- decode metadata table depths (kernel-frozen) ----
# The decode kernels' table types freeze their depths at the 1M-context
# constants (decode_indexer.IDX_MAX_BLOCKS, decode_compressor_ratio4
# .CMP_MAX_BLOCKS, decode_hca.COMPRESS_STATE_MAX_BLOCKS) and the generated
# orchestration reshapes the tables with those depths baked in.  Staging a
# shallower table asserts on device (valid_reshape in simpler's tensormap
# tensor.h) and surfaces as an opaque AICore 507901 lane poison -- so the
# decode depths must match prefill's exactly.  Unused entries are -1, and only
# the leading per-request span is ever read.
DSPARK_DECODE_ORI_TABLE_BLOCKS = 32768
DSPARK_DECODE_CMP_C4_TABLE_BLOCKS = 8192
DSPARK_DECODE_IDX_TABLE_BLOCKS = 8192
# HCA stores one compressed row per 128-source-token page, matching MTP.
# The table depth is dynamic (CMP_TABLE_BLOCKS_DYN).
DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS = DSPARK_MAX_SEQ_LEN // DSPARK_COMPRESSED_BLOCK_TOKENS
DSPARK_DECODE_HCA_STATE_TABLE_BLOCKS = 131072
DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS = 8

# ---- prefill geometry ----
DSPARK_PREFILL_MAX_TOKENS = 8192
# Maximum backing allocation; dispatches bind a compact TP-aligned prefix.
DSPARK_PREFILL_DISPATCH_TOKENS = DSPARK_PREFILL_MAX_TOKENS
DSPARK_PREFILL_LOCAL_TOKENS = DSPARK_PREFILL_DISPATCH_TOKENS // DSPARK_TP_SIZE
DSPARK_PREFILL_MAX_BATCH = DSPARK_CACHE_PARTITIONS * DSPARK_DECODE_BATCH
DSPARK_PREFILL_MAX_CONTEXT_TOKENS = 1_048_576
DSPARK_PREFILL_ORI_TABLE_BLOCKS = 32768
DSPARK_PREFILL_HCA_CMP_TABLE_BLOCKS = 8192
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
# Keep admission within the subsequent decode and drafter lease capacity.
DSPARK_PREFILL_MAX_REQUESTS = DSPARK_DECODE_BATCH

_PREFILL_REQUEST_DYNAMIC_NAMES = frozenset(
    {
        "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
        "hca_compress_state_block_table", "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
    }
)

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
        "x_out",
    }
)
# dspark_target_hidden shares the kernel's FWD_TOKENS_DYN axis with
# position_ids_local/input_ids/ffn_out: it holds each rank's OWNED prompt rows
# (pypto-lib#1084), not the gathered group stream that x_out carries.
_PREFILL_LOCAL_DYNAMIC_NAMES = frozenset(
    {"position_ids_local", "input_ids", "ffn_out", "dspark_target_hidden"}
)

# ---- per-request ring sizes (scheduler-visible blocks per sequence) ----
# Decode keeps its eight-row mathematical CSA window and the eager S=8 writes
# in separate halves of a 16-row transaction ring.
DSPARK_CSA_DECODE_STATE_RING_TOKENS = (
    DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS * DSPARK_C4_STATE_PAGE_TOKENS
)
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
    max_prefill_tokens: int = DSPARK_PREFILL_MAX_TOKENS,
    partitions: int = DSPARK_CACHE_PARTITIONS,
) -> tuple[KVCacheGroupSpec, ...]:
    """Describe the seven DSpark cache families as scheduler-visible groups.

    ``partitions`` is the TP-group count (one per scheduler cache partition),
    not the rank count: the four ranks of one group hold identical replicated
    pools, so a block allocated in partition g exists -- with the same id -- on
    every rank of group g.
    """
    for name, value in (("max_seq_len", max_seq_len), ("max_prefill_tokens", max_prefill_tokens)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be a non-boolean integer")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if max_seq_len > DSPARK_MAX_SEQ_LEN:
        raise ValueError(
            f"DSpark decode cache tables support at most max_seq_len={DSPARK_MAX_SEQ_LEN}, "
            f"got {max_seq_len}"
        )
    if not 0 < max_prefill_tokens <= DSPARK_PREFILL_MAX_TOKENS:
        raise ValueError(f"max_prefill_tokens must be in [1, {DSPARK_PREFILL_MAX_TOKENS}]")
    # Prefill publishes its entire chunk before reading the historical window.
    # Keep both resident, including a boundary page for unaligned chunks.
    in_flight_tokens = max(min(max_prefill_tokens, max_seq_len), DSPARK_DECODE_SEQ)

    def ring_blocks(history: int, page_tokens: int) -> int:
        return math.ceil((history - 1 + in_flight_tokens) / page_tokens) + 1

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
            num_partitions=partitions,
            sliding_window=sliding_window,
            supports_sparse_blocks=sliding_window is not None,
        )

    c128_blocks_per_seq = math.ceil(max_seq_len / DSPARK_COMPRESSED_BLOCK_TOKENS)
    c4_blocks_per_seq = math.ceil(max_seq_len / (4 * DSPARK_BLOCK_SIZE))

    return (
        group(
            "ori",
            all_layers,
            block_size=DSPARK_BLOCK_SIZE,
            element_bytes=2,
            row_width=DSPARK_HEAD_DIM,
            max_blocks_per_seq=ring_blocks(DSPARK_SLIDING_WINDOW, DSPARK_BLOCK_SIZE),
            sliding_window=DSPARK_SLIDING_WINDOW,
        ),
        group(
            "cmp_c128",
            hca_layers,
            block_size=DSPARK_COMPRESSED_BLOCK_TOKENS,
            element_bytes=2,
            row_width=DSPARK_HEAD_DIM,
            max_blocks_per_seq=c128_blocks_per_seq,
            compress_ratio=128,
        ),
        group(
            "cmp_c4",
            csa_layers,
            block_size=4 * DSPARK_BLOCK_SIZE,
            element_bytes=2,
            row_width=DSPARK_HEAD_DIM,
            max_blocks_per_seq=c4_blocks_per_seq,
            compress_ratio=4,
        ),
        group(
            "idx",
            csa_layers,
            block_size=4 * DSPARK_BLOCK_SIZE,
            element_bytes=1,
            row_width=DSPARK_IDX_HEAD_DIM,
            max_blocks_per_seq=c4_blocks_per_seq,
            compress_ratio=4,
            extra_row_bytes=4,
        ),
        group(
            "hca_state",
            hca_layers,
            block_size=DSPARK_C128_STATE_PAGE_TOKENS,
            element_bytes=4,
            row_width=DSPARK_HCA_STATE_DIM,
            max_blocks_per_seq=ring_blocks(DSPARK_SLIDING_WINDOW, DSPARK_C128_STATE_PAGE_TOKENS),
            sliding_window=DSPARK_SLIDING_WINDOW,
        ),
        group(
            "csa_state",
            csa_layers,
            block_size=DSPARK_C4_STATE_PAGE_TOKENS,
            element_bytes=4,
            row_width=DSPARK_CSA_STATE_DIM,
            max_blocks_per_seq=ring_blocks(8, DSPARK_C4_STATE_PAGE_TOKENS),
            sliding_window=8,
        ),
        group(
            "csa_inner_state",
            csa_layers,
            block_size=DSPARK_C4_STATE_PAGE_TOKENS,
            element_bytes=4,
            row_width=DSPARK_CSA_INNER_STATE_DIM,
            max_blocks_per_seq=ring_blocks(8, DSPARK_C4_STATE_PAGE_TOKENS),
            sliding_window=8,
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
    prefill_requests: int = DSPARK_PREFILL_MAX_REQUESTS

    @classmethod
    def for_ranks(cls, ranks: int) -> "DSparkCacheLayout":
        """Build the layout for one TP4 world of ``ranks`` ranks (8 or 16).

        Only the rank-derived axes move: the scheduler cache partitions
        (one per TP group) and the global packed-prefill request capacity that
        is one ``DSPARK_DECODE_BATCH`` per partition. Every per-group extent
        (decode batch, dispatch tokens, metadata table depths) is
        world-size-invariant.
        """
        ranks = int(ranks)
        if ranks not in (2 * DSPARK_TP_SIZE, 4 * DSPARK_TP_SIZE):
            raise ValueError(
                f"DSpark serving supports 8 or 16 ranks (TP4, EP=ranks), got {ranks}"
            )
        partitions = ranks // DSPARK_TP_SIZE
        return cls(
            ranks=ranks,
            partitions=partitions,
            prefill_batch=partitions * DSPARK_DECODE_BATCH,
        )

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
        if runtime.max_seq_len > config.max_position_embeddings:
            raise ValueError("DSpark max_seq_len exceeds checkpoint max_position_embeddings")
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
        if bool((ids < -1).any()):
            raise ValueError("ring table block IDs must be >= -1")
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
        if bool((ids < -1).any()):
            raise ValueError("trailing ring table block IDs must be >= -1")
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

    @staticmethod
    def ring_slot_mapping(
        positions: torch.Tensor,
        block_ids_by_row: Sequence[Sequence[int]],
        *,
        block_size: int,
    ) -> torch.Tensor:
        """Map absolute positions through compact per-request ring page lists."""
        rows = []
        for row_positions, block_ids in zip(positions, block_ids_by_row, strict=True):
            ids = torch.tensor(
                [int(block_id) for block_id in block_ids],
                dtype=torch.long,
                device=positions.device,
            )
            positions_i64 = row_positions.to(torch.int64)
            logical = positions_i64 // int(block_size)
            pages = ids.index_select(0, (logical % ids.numel()).reshape(-1)).reshape(
                logical.shape
            )
            slots = pages * int(block_size) + positions_i64 % int(block_size)
            rows.append(torch.where(pages >= 0, slots, torch.full_like(slots, -1)))
        return torch.stack(rows)

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
        cannot publish compressed-cache entries.  ``commit_tokens`` may be a
        per-request tensor over the batch axis.
        """
        positions_i64 = positions.to(torch.int64)
        boundary = (positions_i64 + 1) % compress_ratio == 0
        if commit_tokens is not None:
            columns = torch.arange(positions.shape[-1], device=positions.device).unsqueeze(0)
            if isinstance(commit_tokens, torch.Tensor):
                boundary = boundary & (columns < commit_tokens.reshape(-1, 1))
            else:
                boundary = boundary & (columns < int(commit_tokens))
        cache_col = positions_i64 // compress_ratio
        storage_block_size = DSPARK_COMPRESSED_BLOCK_TOKENS // compress_ratio
        logical = cache_col // storage_block_size
        depth = table.shape[-1]
        gathered = self._gather_table(table, logical)
        valid = boundary & (logical < depth) & (gathered >= 0)
        slot = gathered * storage_block_size + cache_col % storage_block_size
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

    def ring_swa_window_indices_and_lens(
        self,
        positions: torch.Tensor,
        block_ids_by_row: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower SWA windows directly from compact raw-KV ring page lists."""
        window = self.layout.sliding_window
        positions_i64 = positions.to(torch.int64)
        batch, seq = positions_i64.shape
        start = (positions_i64 - window + 1).clamp(min=0)
        offsets = torch.arange(window, device=positions.device)
        visible = start.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)
        valid = offsets.unsqueeze(0).unsqueeze(0) <= (positions_i64 - start).unsqueeze(-1)
        indices = self.ring_slot_mapping(
            visible,
            block_ids_by_row,
            block_size=self.layout.block_size,
        )
        indices = torch.where(valid, indices, torch.full_like(indices, -1)).to(torch.int32)
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
    packed_offsets: tuple[int, ...]
    # Per-request prompt-chunk embeddings ([tokens, hidden] FP32); staged
    # directly into the shared x_hc slot with a zero tail.
    embeddings: tuple[torch.Tensor, ...]
    input_ids: torch.Tensor
    position_ids_local: torch.Tensor
    position_ids_full: torch.Tensor
    # Packed request boundaries per rank; repeated terminal entries pad groups
    # with fewer requests to the dispatch's common request-axis extent.
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
    input_ids: torch.Tensor | None
    position_ids_local: torch.Tensor | None
    position_ids: torch.Tensor | None
    logit_row_indices: torch.Tensor | None
    # (rank, packed sampled row) per batch row, reading sampled_ids[rank,
    # row, 0]; the speculative readback extends to row+7.
    sampled_slots: tuple[tuple[int, int], ...]
    # Per batch row: whether the eight-row verify window carried drafts (the
    # readback then covers all eight rows and acceptance runs on-device greedy
    # samples; fallback rows keep the single-anchor milestone-1 contract).
    speculative_flags: tuple[bool, ...] = ()
    # Per batch row: the decode hidden-row base (local_index * decode_seq)
    # where this request's tap rows live in the backbone mirror.
    verify_hidden_rows: tuple[int, ...] = ()
    owner_ranks: tuple[int, ...] = ()
    owner_rows: tuple[int, ...] = ()
    buffer_slot: int = 0
    # Fully bound one-L2 arguments.  Keeping this tuple on the prepared slot
    # removes Python binding and all mutable Host staging from the device lane.
    dispatch_args: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class _DSparkDecodeGroupPlan:
    """Position-independent Host metadata for one TP group's decode tile."""

    request_indices: tuple[int | None, ...]
    anchor_flags: torch.Tensor
    ori_tables: torch.Tensor
    hca_cmp_tables: torch.Tensor
    csa_cmp_tables: torch.Tensor
    idx_tables: torch.Tensor
    # The split-L3 fallback still consumes Host-lowered transaction tables.
    # The one-L2 path derives all three on device from the physical rings.
    hca_state_tables: torch.Tensor | None
    csa_state_tables: torch.Tensor | None
    csa_inner_state_tables: torch.Tensor | None


@dataclass(frozen=True)
class DSparkPreparedDecodePlan:
    """Early Host plan whose mutable token/position state is bound at execution."""

    request_ids: tuple[str, ...]
    groups: tuple[int, ...]
    group_ordinals: tuple[int, ...]
    assignment: "_DSparkGroupAssignment"
    request_blocks: tuple[dict[str, tuple[int, ...]], ...]
    group_plans: tuple[_DSparkDecodeGroupPlan, ...]
    state_slot_ids: torch.Tensor
    state_generations: torch.Tensor
    group_state_slot_ids: torch.Tensor
    group_state_generations: torch.Tensor
    group_ori_block_tables: torch.Tensor
    group_hca_cmp_block_tables: torch.Tensor
    group_csa_cmp_block_tables: torch.Tensor
    group_idx_block_tables: torch.Tensor
    group_hca_state_block_tables: torch.Tensor
    group_csa_state_block_tables: torch.Tensor
    group_csa_inner_state_block_tables: torch.Tensor
    sampled_row_offsets: torch.Tensor
    hidden_row_offsets: torch.Tensor
    owner_ranks: tuple[int, ...]
    owner_rows: tuple[int, ...]
    buffer_slot: int
    dispatch_inputs: DSparkPreparedDecodeInputs | None = None


@dataclass(frozen=True)
class _DSparkPendingDecode:
    """Submitted one-L2 decode whose compact Host outputs await reclaim."""

    dispatch: PendingL3Dispatch
    inputs: DSparkPreparedDecodeInputs
    sampled_ids: torch.Tensor


@dataclass(frozen=True)
class _DSparkDeviceStateBuffers:
    """Ping-ponged Host descriptors and compact outputs for device-state programs."""

    state_slot_ids: torch.Tensor
    state_generations: torch.Tensor
    # Standalone state-accept scratch.  The one-L2 path keeps these values
    # invocation-local and therefore leaves the Host fields unallocated.
    sampled_row_offsets: torch.Tensor | None
    hidden_row_offsets: torch.Tensor | None
    accepted_token_ids: torch.Tensor
    accepted_counts: torch.Tensor
    context_positions: torch.Tensor | None
    context_valid: torch.Tensor | None
    last_sampled: torch.Tensor | None
    anchor_positions: torch.Tensor | None
    drafter_row_offsets: torch.Tensor | None
    draft_token_ids: torch.Tensor | None
    group_state_slot_ids: torch.Tensor | None = None
    group_state_generations: torch.Tensor | None = None
    group_ori_block_tables: torch.Tensor | None = None
    group_hca_cmp_block_tables: torch.Tensor | None = None
    group_csa_cmp_block_tables: torch.Tensor | None = None
    group_idx_block_tables: torch.Tensor | None = None
    group_hca_state_block_tables: torch.Tensor | None = None
    group_csa_state_block_tables: torch.Tensor | None = None
    group_csa_inner_state_block_tables: torch.Tensor | None = None


@dataclass(frozen=True)
class _DSparkGroupAssignment:
    """Per-group placement of one decode batch's requests."""

    groups: tuple[int, ...]
    ordinals: tuple[int, ...]
    # group -> ((batch index, group stream slot) per active request).  The
    # stream slot is the rank-major request index every group-row tensor and
    # the rank-local tables slice through.
    active_by_group: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True)
class DSparkDrafterRequestRow:
    """One dense drafter batch row, decoupled from persistent leases.

    ``hidden_row`` is the row offset in the rank-local backbone tap mirror
    where this request's ``valid_count`` context rows start; ``token_source``
    is the query row-0 token (the committed token in decode mode, the next
    prompt token when seeding).
    """

    request_id: str
    group: int
    lease: int
    anchor: int
    valid_count: int
    token_source: int
    hidden_row: int
    decode_mode: bool = True


@dataclass
class _DSparkDraftRequestState:
    """Per-request speculative state, keyed by a stable group-local lease."""

    group: int
    lease: int
    generation: int = 0
    device_state_initialized: bool = False
    current_token_id: int | None = None
    prompt_len: int = 0
    committed_count: int = 0
    # The seven proposals staged for the next target verify (empty between
    # prefill completion and the first drafter dispatch).
    pending_draft_tokens: list[int] = field(default_factory=list)
    pending_confidence: list[float] = field(default_factory=list)
    # Rolling prompt-tail capture for prefill seeding: rows are the rank-owned
    # backbone tap rows with their absolute positions and owning ranks.
    prefill_tail_rows: torch.Tensor | None = None
    prefill_tail_positions: torch.Tensor | None = None
    prefill_tail_ranks: torch.Tensor | None = None
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    verify_steps: int = 0
    matched_drafts: int = 0
    fallback_steps: int = 0


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
    drafter: Any | None = None
    markov: Any | None = None
    state_prepare: Any | None = None
    state_accept: Any | None = None
    state_commit: Any | None = None
    decode_device_state_fused: bool = False
    draft_device_state_fused: bool = False
    decode_full_fused: bool = False
    # K of the speculative chain: 0 keeps the milestone-1 target-only path
    # (no drafter weights, programs, or state are materialized), 7 enables it.
    num_speculative_tokens: int = 0
    rope: DSparkRopeTables | None = None
    platform: str = "a2a3"
    device_id: int = 0
    device_ids: tuple[int, ...] = ()
    n_routed_experts: int = 256
    num_hash_layers: int = 3
    embedding_weight: torch.Tensor | None = None

    def l3_callables(self) -> tuple[Any, ...]:
        """Return every compiled L3 program the shared worker may run."""
        return tuple(
            program
            for program in (
                self.prefill,
                self.decode,
                self.drafter,
                self.markov,
                self.state_prepare,
                self.state_accept,
                self.state_commit,
            )
            if program is not None
        )


def _accept_dspark_tokens(
    main: Sequence[int], draft: Sequence[int]
) -> tuple[list[int], int]:
    """Linear-chain acceptance: longest matching prefix plus the bonus token.

    ``main`` holds the target's greedy prediction for each verify row
    (``main[i]`` is the token following row ``i``); ``draft`` holds the K
    proposals staged into rows 1..K.  Acceptance stops at the first
    mismatch and always appends the target's own prediction at that point,
    so the result carries ``matched + 1`` tokens (1..K+1).
    """
    matched = 0
    for token in draft:
        if matched >= len(main):
            break
        if int(main[matched]) == int(token):
            matched += 1
        else:
            break
    if matched >= len(main):
        raise ValueError(
            "DSpark acceptance ran past the verify window: every row matched "
            "but a bonus prediction must remain"
        )
    return [int(token) for token in main[: matched + 1]], matched


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
        self._static_lm_head_device_weight: StackedDeviceTensor | None = None
        self._hc_head_buffers: dict[str, torch.Tensor] | None = None
        self._stacked_host_weights: dict[str, torch.Tensor] | None = None
        self._stacked_prefill_host_weights: dict[str, torch.Tensor] | None = None
        self._stacked_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._stacked_prefill_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._drafter_host_weights: dict[str, torch.Tensor] | None = None
        self._drafter_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._embedding_device_weight: StackedDeviceTensor | None = None
        self._device_scratch: dict[tuple[str, str], StackedDeviceTensor] = {}
        self._prefill_task_args: TaskArgs | None = None
        self._decode_task_args: list[TaskArgs] = []
        # Speculative drafter state (milestone 2): per-request leases, the
        # drafter/markov TaskArgs, their RunConfigs, and the D2H mirror for
        # the decode backbone tap.
        self._drafter_states: dict[str, _DSparkDraftRequestState] = {}
        self._drafter_free_leases: dict[int, list[int]] = {
            group: list(range(DSPARK_DRAFTER_LEASES_PER_GROUP))
            for group in range(compiled.layout.partitions)
        }
        self._drafter_lease_generations: list[list[int]] = [
            [0] * DSPARK_DRAFTER_LEASES_PER_GROUP
            for _ in range(compiled.layout.partitions)
        ]
        self._dspark_device_state_tokens: StackedDeviceTensor | None = None
        self._dspark_device_state_meta: StackedDeviceTensor | None = None
        self._dspark_rope_device_tables: dict[str, StackedDeviceTensor] | None = None
        self._dspark_state_buffers: list[_DSparkDeviceStateBuffers] = []
        self._drafter_task_args: TaskArgs | None = None
        self._markov_task_args: TaskArgs | None = None
        self._fused_drafter_task_args: list[TaskArgs] = []
        self._fused_markov_task_args: list[TaskArgs] = []
        self._drafter_context_staging: dict[int, dict[str, torch.Tensor]] = {}
        self._drafter_block_table_staging: dict[int, torch.Tensor] = {}
        self._drafter_rope_candidates: dict[str, torch.Tensor] = {}
        self._fused_drafter_batch: int | None = None
        self._fused_drafter_batches: list[int | None] = [None, None]
        self._fused_drafter_block_tables: list[dict[int, torch.Tensor]] = []
        self._fused_drafter_rope_candidates: list[dict[str, torch.Tensor]] = []
        self._drafter_run_config: Any = None
        self._markov_run_config: Any = None
        self._drafter_hidden_mirror: torch.Tensor | None = None
        self._active_drafter_target_hidden: StackedDeviceTensor | None = None
        self._acceptance_log_steps = 0
        self._pending_decode_dispatch_lock = threading.Lock()
        self._pending_decode_dispatches: dict[int, PendingL3Dispatch] = {}
        self._l3_shared_buffers_ready = False

    # ------------------------------------------------------------------
    # cache topology
    # ------------------------------------------------------------------
    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> int:
        """Allocate the replicated group pools from the post-weight budget."""
        self._cache_group_specs = self._resolve_cache_group_specs(config, runtime)
        self._configure_l3_rings(runtime)
        from pypto.runtime import RunConfig  # noqa: PLC0415

        # Fused decode includes the drafter; split target decode has its own
        # long-context profile. The runtime / CLI heap sizes prefill.
        decode_ring_heap = (
            DSPARK_DRAFTER_RING_HEAP
            if self._compiled.decode_full_fused
            else DSPARK_DECODE_RING_HEAP
        )
        # Keep expensive on-device DFX opt-in.  Device STRACE perturbs this
        # 16-rank workload enough to make prefill fail, while dependency
        # generation is useful for diagnosing the fused decode graph itself.
        decode_dep_gen = os.environ.get("PYPTO_DSPARK_ENABLE_DEP_GEN", "0") == "1"
        decode_chip_swimlane = int(os.environ.get("PYPTO_DSPARK_CHIP_SWIMLANE", "0"))
        self._decode_run_config = RunConfig(
            ring_heap=decode_ring_heap,
            enable_dep_gen=decode_dep_gen,
            enable_chip_swimlane=decode_chip_swimlane,
        )
        if self.speculative:
            # Markov does not allocate the target's HCA attention partials.
            self._drafter_run_config = RunConfig(ring_heap=DSPARK_DRAFTER_RING_HEAP)
            self._markov_run_config = RunConfig(ring_heap=DSPARK_MARKOV_RING_HEAP)
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
            max_prefill_tokens=min(
                runtime.max_prefill_tokens_per_request or DSPARK_PREFILL_MAX_TOKENS,
                runtime.max_num_batched_tokens,
            ),
            partitions=self._compiled.layout.partitions,
        )
        names = tuple(spec.name for spec in specs)
        if names != DSPARK_CACHE_GROUP_NAMES:
            raise ValueError(
                "DSpark KV cache groups must be ordered as "
                + ", ".join(DSPARK_CACHE_GROUP_NAMES)
                + f"; got {names}"
            )
        if any(spec.num_partitions != self._compiled.layout.partitions for spec in specs):
            raise ValueError(
                f"DSpark KV cache groups must use "
                f"{self._compiled.layout.partitions} partitions"
            )
        return tuple(specs)

    def _decode_state_table_depth(self, group_name: str) -> int:
        """Use the scheduler's physical ring period for device-side modulo."""
        specs = self._cache_group_specs or build_dspark_cache_group_specs(
            DSPARK_FWD_NUM_LAYERS, self._compiled.compress_ratios,
        )
        return next(spec.max_blocks_per_seq for spec in specs if spec.name == group_name)

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
        # Ranks are enumerated as logical Worker IDs on the shared L3 worker:
        # the query runs in the chip process that owns each device context.
        # Physical device IDs are never used to route the query (they may be
        # non-contiguous), and torch_npu is not a serving dependency, so the
        # free/total snapshot comes from the worker's device_memory_info.
        worker = self._shared_l3_worker()
        utilization = float(getattr(runtime, "npu_memory_utilization", 0.90))
        budgets = []
        for worker_id in range(self._compiled.layout.ranks):
            free_bytes, total_bytes = worker.device_memory_info(worker_id)
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

    def _track_pending_decode_dispatch(
        self,
        buffer_slot: int,
        dispatch: PendingL3Dispatch,
    ) -> None:
        """Prevent a ping-pong slot from being rebound before reclaim."""
        with self._pending_decode_dispatch_lock:
            if buffer_slot in self._pending_decode_dispatches:
                raise RuntimeError(
                    f"DSpark decode buffer slot {buffer_slot} was reused before completion"
                )
            self._pending_decode_dispatches[buffer_slot] = dispatch

    def _forget_pending_decode_dispatch(
        self,
        buffer_slot: int,
        dispatch: PendingL3Dispatch,
    ) -> None:
        """Drop a completed slot owner without removing a newer dispatch."""
        with self._pending_decode_dispatch_lock:
            if self._pending_decode_dispatches.get(buffer_slot) is dispatch:
                del self._pending_decode_dispatches[buffer_slot]

    def _wait_for_pending_decode_dispatches(self) -> None:
        """Fence shared prefill state behind earlier one-L2 decode work."""
        with self._pending_decode_dispatch_lock:
            pending = tuple(sorted(self._pending_decode_dispatches.items()))
        first_error: BaseException | None = None
        for buffer_slot, dispatch in pending:
            try:
                dispatch.wait()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self._forget_pending_decode_dispatch(buffer_slot, dispatch)
        if first_error is not None:
            raise first_error

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

    @property
    def speculative(self) -> bool:
        """Whether the K=7 drafter chain is enabled for this runner."""
        return self._compiled.num_speculative_tokens > 0

    @property
    def supports_async_decode_reclaim(self) -> bool:
        """The one-L2 path owns recurrent state and ping-ponged Host outputs."""
        return bool(self._compiled.decode_full_fused)

    @staticmethod
    def prepared_decode_requires_token(prepared: object) -> bool:
        """A fully staged one-L2 snapshot reads its next token from device state."""
        return not (
            isinstance(prepared, DSparkPreparedDecodePlan)
            and prepared.dispatch_inputs is not None
            and prepared.dispatch_inputs.dispatch_args is not None
        )

    def load_drafter_weights(self):
        """Load and pack the mtp.0/1/2 drafter banks (speculation only)."""
        return self._compiled.weight_store.load_drafter_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
        )

    def _require_drafter_weights(self):
        tensors = self._drafter_device_weights or self._drafter_host_weights
        if tensors is None:
            raise RuntimeError(
                "DSpark drafter weights are not available (speculation requires "
                "num_speculative_tokens=7)"
            )
        return tensors

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
        # Weight preparation can allocate these buffers before init_kv_cache.
        # Both phases must use the same configured physical ring periods.
        if not self._cache_group_specs:
            self._cache_group_specs = self._resolve_cache_group_specs(model.config, model.runtime)
        with profile_span("DSparkModelRunner.prepare.load_global_weights", cat="executor"):
            self.load_packed_global_weights()
        with profile_span("DSparkModelRunner.prepare.load_stacked_weights", cat="executor"):
            stacked = self.load_stacked_layer_weights()
            self._retain_stacked_host_weights(stacked)
            del stacked
        if self.speculative:
            # The drafter banks must be resident before the KV-capacity
            # snapshot below, so speculation's extra weights shrink the
            # measured free budget instead of silently overcommitting HBM.
            with profile_span("DSparkModelRunner.prepare.load_drafter_weights", cat="executor"):
                drafter = self.load_drafter_weights()
                self._ensure_shared_host_allocation_before_worker("drafter weights")
                self._drafter_host_weights = dict(drafter.tensors)
                del drafter
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
        if self.speculative:
            from pypto_serving.model.common.runner.buffer_set import (  # noqa: PLC0415
                shared_empty,
            )
            from pypto_serving.model.deepseek_dspark.task_args import (  # noqa: PLC0415
                drafter_task_args,
                markov_task_args,
            )

            with profile_span("DSparkModelRunner.prepare.drafter_task_args", cat="executor"):
                fused_slots = 2 if self._compiled.decode_full_fused else 1
                self._fused_drafter_task_args = []
                self._fused_markov_task_args = []
                for _slot in range(fused_slots):
                    drafter_args = drafter_task_args(self)
                    drafter_args.allocate_host_shared(None)
                    markov_args = markov_task_args(self)
                    markov_args.allocate_host_shared(None)
                    self._fused_drafter_task_args.append(drafter_args)
                    self._fused_markov_task_args.append(markov_args)
                # Standalone prefill seeding and split-L3 fallback keep using
                # slot zero.  The one-L2 decode path indexes the two lists so
                # a prepared slot remains immutable until output reclaim.
                self._drafter_task_args = self._fused_drafter_task_args[0]
                self._markov_task_args = self._fused_markov_task_args[0]
                # The mirror and staging buffers below must exist before the
                # L3 worker forks: _shared_l3_worker() creates it lazily at
                # the first device-side call, and host-shared tensors
                # allocated after that point never map into the children.
                self._ensure_shared_host_allocation_before_worker("drafter staging buffers")
                if not self._compiled.decode_full_fused:
                    self._drafter_hidden_mirror = shared_empty(
                        (
                            self._compiled.layout.ranks,
                            DSPARK_DRAFTER_CONTEXT_ROWS,
                            DSPARK_MAIN_HIDDEN_DIM,
                        ),
                        torch.bfloat16,
                        name="dspark_target_hidden_mirror",
                    )
                ranks = self._compiled.layout.ranks
                local_batch = self._compiled.layout.decode_local_batch
                local_tokens = local_batch * self._compiled.layout.decode_seq
                legacy_accept_scratch = not self._compiled.decode_full_fused
                self._dspark_state_buffers = [
                    _DSparkDeviceStateBuffers(
                        state_slot_ids=shared_empty(
                            (ranks, local_batch), torch.int32, name="dspark_state_slot_ids"
                        ),
                        state_generations=shared_empty(
                            (ranks, local_batch), torch.int32, name="dspark_state_generations"
                        ),
                        sampled_row_offsets=(
                            shared_empty(
                                (ranks, local_batch),
                                torch.int32,
                                name="dspark_sampled_rows",
                            )
                            if legacy_accept_scratch
                            else None
                        ),
                        hidden_row_offsets=(
                            shared_empty(
                                (ranks, local_batch),
                                torch.int32,
                                name="dspark_hidden_rows",
                            )
                            if legacy_accept_scratch
                            else None
                        ),
                        accepted_token_ids=shared_empty(
                            (ranks, local_batch, DSPARK_DECODE_SEQ),
                            torch.int32,
                            name="dspark_accepted_token_ids",
                        ),
                        accepted_counts=shared_empty(
                            (ranks, local_batch), torch.int32, name="dspark_accepted_counts"
                        ),
                        context_positions=(
                            shared_empty(
                                (ranks, local_tokens),
                                torch.int32,
                                name="dspark_context_positions",
                            )
                            if legacy_accept_scratch
                            else None
                        ),
                        context_valid=(
                            shared_empty(
                                (ranks, local_tokens),
                                torch.int32,
                                name="dspark_context_valid",
                            )
                            if legacy_accept_scratch
                            else None
                        ),
                        last_sampled=(
                            shared_empty(
                                (ranks, local_batch),
                                torch.long,
                                name="dspark_last_sampled",
                            )
                            if legacy_accept_scratch
                            else None
                        ),
                        anchor_positions=(
                            shared_empty(
                                (ranks, local_batch),
                                torch.int32,
                                name="dspark_anchor_positions",
                            )
                            if legacy_accept_scratch
                            else None
                        ),
                        drafter_row_offsets=(
                            shared_empty(
                                (ranks, local_batch),
                                torch.int32,
                                name="dspark_drafter_rows",
                            )
                            if legacy_accept_scratch
                            else None
                        ),
                        draft_token_ids=(
                            shared_empty(
                                (ranks, local_batch, DSPARK_DRAFTER_QUERY_WIDTH),
                                torch.int32,
                                name="dspark_state_draft_token_ids",
                            )
                            if legacy_accept_scratch
                            else None
                        ),
                        group_state_slot_ids=shared_empty(
                            (ranks, self._compiled.layout.decode_batch),
                            torch.int32,
                            name="dspark_group_state_slot_ids",
                        ),
                        group_state_generations=shared_empty(
                            (ranks, self._compiled.layout.decode_batch),
                            torch.int32,
                            name="dspark_group_state_generations",
                        ),
                        group_ori_block_tables=shared_empty(
                            (
                                ranks,
                                self._compiled.layout.decode_batch,
                                DSPARK_DECODE_ORI_TABLE_BLOCKS,
                            ),
                            torch.int32,
                            name="dspark_group_ori_block_tables",
                        ),
                        group_hca_cmp_block_tables=shared_empty(
                            (
                                ranks,
                                self._compiled.layout.decode_batch,
                                DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS,
                            ),
                            torch.int32,
                            name="dspark_group_hca_cmp_block_tables",
                        ),
                        group_csa_cmp_block_tables=shared_empty(
                            (
                                ranks,
                                self._compiled.layout.decode_batch,
                                DSPARK_DECODE_CMP_C4_TABLE_BLOCKS,
                            ),
                            torch.int32,
                            name="dspark_group_csa_cmp_block_tables",
                        ),
                        group_idx_block_tables=shared_empty(
                            (
                                ranks,
                                self._compiled.layout.decode_batch,
                                DSPARK_DECODE_IDX_TABLE_BLOCKS,
                            ),
                            torch.int32,
                            name="dspark_group_idx_block_tables",
                        ),
                        group_hca_state_block_tables=shared_empty(
                            (
                                ranks,
                                self._compiled.layout.decode_batch,
                                self._decode_state_table_depth("hca_state"),
                            ),
                            torch.int32,
                            name="dspark_group_hca_state_block_tables",
                        ),
                        group_csa_state_block_tables=shared_empty(
                            (
                                ranks,
                                self._compiled.layout.decode_batch,
                                self._decode_state_table_depth("csa_state"),
                            ),
                            torch.int32,
                            name="dspark_group_csa_state_block_tables",
                        ),
                        group_csa_inner_state_block_tables=shared_empty(
                            (
                                ranks,
                                self._compiled.layout.decode_batch,
                                self._decode_state_table_depth("csa_state"),
                            ),
                            torch.int32,
                            name="dspark_group_csa_inner_state_block_tables",
                        ),
                    )
                    for _slot in (0, 1)
                ]
                # One shared buffer per dynamic extent, allocated before the
                # worker fork like every other host-shared tensor.
                for extent in DSPARK_DRAFTER_CONTEXT_BUCKETS:
                    group_extent = 4 * extent
                    self._drafter_context_staging[extent] = {
                        "context_group_position_ids": shared_empty(
                            (ranks, group_extent),
                            torch.int32,
                            name=f"dspark_ctx_positions_{extent}",
                        ),
                        "context_group_slot_mapping": shared_empty(
                            (ranks, DSPARK_DRAFT_LAYERS, group_extent),
                            torch.int64,
                            name=f"dspark_ctx_slots_{extent}",
                        ),
                        "context_group_freqs_cos": shared_empty(
                            (ranks, group_extent, DSPARK_ROPE_HEAD_DIM),
                            torch.bfloat16,
                            name=f"dspark_ctx_cos_{extent}",
                        ),
                        "context_group_freqs_sin": shared_empty(
                            (ranks, group_extent, DSPARK_ROPE_HEAD_DIM),
                            torch.bfloat16,
                            name=f"dspark_ctx_sin_{extent}",
                        ),
                    }
                self._fused_drafter_block_tables = []
                self._fused_drafter_rope_candidates = []
                for slot in range(fused_slots):
                    block_tables = {
                        padded_batch: shared_empty(
                            (
                                ranks,
                                DSPARK_DRAFT_LAYERS,
                                padded_batch,
                                DSPARK_DRAFTER_TABLE_BLOCKS,
                            ),
                            torch.int32,
                            name=f"dspark_block_tables_s{slot}_{padded_batch}",
                        )
                        for padded_batch in DSPARK_DRAFTER_BATCHES
                    }
                    rope_candidates = (
                        {}
                        if self._compiled.decode_full_fused
                        else {
                            name: shared_empty(
                                (
                                    ranks,
                                    self._compiled.layout.decode_local_batch,
                                    DSPARK_DRAFTER_ROPE_CANDIDATE_ROWS,
                                    DSPARK_ROPE_HEAD_DIM,
                                ),
                                torch.bfloat16,
                                name=f"dspark_s{slot}_{name}",
                            )
                            for name in (
                                "rope_cos_candidates",
                                "rope_sin_candidates",
                            )
                        }
                    )
                    self._fused_drafter_block_tables.append(block_tables)
                    self._fused_drafter_rope_candidates.append(rope_candidates)
                self._drafter_block_table_staging = self._fused_drafter_block_tables[0]
                self._drafter_rope_candidates = self._fused_drafter_rope_candidates[0]
        with profile_span("DSparkModelRunner.upload_resident_weights", cat="executor"):
            self._materialize_resident_weights()
        # Decode preparation runs on the Host lane while the preceding mixed
        # prefill/decode step may still be executing.  Materialize every
        # decode workspace before the server becomes ready: TaskArgs binding
        # in the steady pipeline must only look up resident tensors, never
        # issue unordered device-control allocations beside an in-flight run.
        with profile_span("DSparkModelRunner.prepare.decode_scratch", cat="executor"):
            self._materialize_decode_scratch()
        if self.speculative:
            # Materialize the persistent drafter device buffers before the
            # KV-capacity free-memory snapshot: they are resident for the
            # worker's whole lifetime, so lazy first-draft allocation would
            # OOM outside the cache-allocation retry path.  This must run
            # after every host-shared allocation above: the first device
            # access creates and forks the L3 worker, and a staging buffer
            # allocated past that point is shm-backed yet unmapped in the
            # children -- the first drafter dispatch then SMMU-faults
            # reading it.
            from pypto_serving.model.deepseek_dspark.task_args import (  # noqa: PLC0415
                drafter_scratch_specs,
            )

            with profile_span("DSparkModelRunner.prepare.drafter_scratch", cat="executor"):
                for name, (shape, dtype) in drafter_scratch_specs(
                    self._compiled.layout.ranks
                ).items():
                    self._alloc_zeroed_stacked_tensor(name, shape, dtype, scope="drafter")
                self._materialize_dspark_device_state_tokens()
                self._materialize_dspark_device_state_meta()
                self._alloc_zeroed_stacked_tensor(
                    "device_state_target_hidden",
                    (
                        self._compiled.layout.ranks,
                        DSPARK_DRAFTER_CONTEXT_ROWS,
                        DSPARK_MAIN_HIDDEN_DIM,
                    ),
                    torch.bfloat16,
                    scope="drafter",
                )
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

    def _static_lm_head_weight_tensor(self) -> torch.Tensor | StackedDeviceTensor:
        """One TP vocab shard per rank: rank r consumes shard ``r % tp``."""
        if self._static_lm_head_device_weight is not None:
            return self._static_lm_head_device_weight
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

    def _materialize_lm_head_device_weight(self, worker: Any) -> StackedDeviceTensor:
        """Upload the shared TP LM-head shards once for target and drafter use."""
        stacked = self._static_lm_head_device_weight
        if stacked is not None:
            return stacked
        host = self._static_lm_head_weight
        if host is None:
            raise RuntimeError("DSpark LM-head Host shards are not staged")
        with profile_span("DSparkModelRunner.upload_lm_head", cat="executor"):
            stacked = worker.alloc_stacked_tensor(host)
        self._static_lm_head_device_weight = stacked
        self._static_lm_head_weight = None
        return stacked

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
        if self.speculative and self._drafter_device_weights is None:
            host_weights = self._drafter_host_weights
            if not host_weights:
                raise RuntimeError("DSpark drafter host weights are not retained")
            with profile_span("DSparkModelRunner.upload_drafter_weights", cat="executor"):
                self._drafter_device_weights = self._upload_weight_group(worker, host_weights)
            self._drafter_host_weights = None
        self._materialize_embedding_device_weight()
        self._materialize_lm_head_device_weight(worker)
        if self._compiled.decode_full_fused:
            self._materialize_dspark_rope_tables()
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
        if self._drafter_host_weights:
            tensors.extend(self._drafter_host_weights.values())
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

    def _materialize_dspark_rope_tables(self) -> dict[str, StackedDeviceTensor]:
        """Upload the three full RoPE profiles consumed by device-side prepare."""
        tables = self._dspark_rope_device_tables
        if tables is not None:
            return tables
        rope = self._require_rope_tables()
        sources = {
            "swa_rope_cos_table": rope.swa_cos,
            "swa_rope_sin_table": rope.swa_sin,
            "ratio4_rope_cos_table": rope.ratio4_cos,
            "ratio4_rope_sin_table": rope.ratio4_sin,
            "ratio128_rope_cos_table": rope.ratio128_cos,
            "ratio128_rope_sin_table": rope.ratio128_sin,
        }
        worker = self._shared_l3_worker()
        worker_ids = tuple(range(self._compiled.layout.ranks))
        materialized: dict[str, StackedDeviceTensor] = {}
        allocated: list[tuple[Any, int]] = []
        try:
            for name, source in sources.items():
                source = source.to(device="cpu", dtype=torch.bfloat16).contiguous()
                shards = []
                for worker_id in worker_ids:
                    shard = worker.alloc_tensor(
                        source.shape,
                        source.dtype,
                        init=source,
                        worker_id=worker_id,
                    )
                    shards.append(shard)
                    allocated.append((shard, worker_id))
                materialized[name] = StackedDeviceTensor(
                    shards,
                    (self._compiled.layout.ranks, *source.shape),
                    worker_ids,
                )
        except Exception:
            for shard, worker_id in allocated:
                worker.free_tensor(shard, worker_id=worker_id)
            raise
        self._dspark_rope_device_tables = materialized
        return materialized

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

    def _materialize_decode_scratch(self) -> None:
        """Allocate every decode workspace before asynchronous serving starts."""
        from pypto_serving.model.deepseek_dspark.task_args import (  # noqa: PLC0415
            decode_scratch_specs,
        )

        for name, (shape, dtype) in decode_scratch_specs(
            self._compiled.layout.ranks
        ).items():
            self._alloc_zeroed_stacked_tensor(name, shape, dtype, scope="decode")

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
                "cmp_c128", DSPARK_HCA_NUM_LAYERS, DSPARK_HCA_CMP_STORAGE_BLOCK_SIZE,
                (1, DSPARK_HEAD_DIM),
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
        """Run packed prefill chunks per TP group at their common TP-aligned extent."""
        if self._compiled.prefill is None:
            raise RuntimeError("DSpark kernels were not compiled for this runner")
        self._wait_for_pending_decode_dispatches()
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
            args = self._prefill_dispatch_args(
                inputs.physical_tokens, inputs.query_start_loc.shape[1] - 1
            )
            self._trace_prefill_chunk(inputs, status="started")
            try:
                with profile_span(
                    "DSparkModelRunner.prefill.l3_dispatch",
                    cat="executor",
                    args={
                        "actual_tokens": int(inputs.query_start_loc[:, -1].max()),
                        "requests_per_group": [
                            inputs.groups.count(group) for group in range(self._compiled.layout.partitions)
                        ],
                    },
                ):
                    self._run_l3(self._compiled.prefill, *args)
            except RuntimeError as exc:
                raise RuntimeError(
                    "DSpark packed prefill dispatch failed "
                    f"(tokens={inputs.actual_tokens}, groups={inputs.groups})"
                ) from exc
            if self.speculative:
                self._capture_prefill_tails(batch, inputs)
            sampled = self._prefill_task_args.tensors["sampled_ids"]
            tokens = [
                int(sampled[rank, row, 0].item()) for rank, row in inputs.sampled_slots
            ]
            self._trace_prefill_chunk(inputs, status="completed")
            return PrefillResult(
                last_hidden=None,
                logits=torch.zeros((len(tokens), 0)),
                sampled_token_ids=torch.tensor(tokens, dtype=torch.long),
            )

    def _trace_prefill_chunk(self, inputs: DSparkPreparedPrefillInputs, *, status: str) -> None:
        if os.environ.get("PYPTO_DSPARK_TRACE_PREFILL") != "1":
            return
        # Completion is emitted only after synchronous L3 execution and output readback.
        for request_id, group, start, actual in zip(
            inputs.request_ids, inputs.groups, inputs.chunk_starts, inputs.actual_tokens, strict=True
        ):
            logger.info("DSpark prefill chunk: %s", json.dumps({
                "request_id": request_id,
                "group": group,
                "start": start,
                "logical_tokens": actual,
                "physical_tokens": inputs.physical_tokens,
                "status": status,
            }, sort_keys=True))

    def _prefill_kernel_tokens(self, actual_tokens: int) -> int:
        """Return the TP-aligned packed extent, independent of individual context lengths."""
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

    def _prefill_dispatch_args(self, physical_tokens: int, request_rows: int) -> tuple[Any, ...]:
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
            elif name in _PREFILL_REQUEST_DYNAMIC_NAMES:
                rows = request_rows
            elif name == "query_start_loc":
                rows = request_rows + 1
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
                f"DSpark prefill supports at most {layout.prefill_batch} requests per dispatch, "
                f"got {request_count}"
            )
        if len(batch.cache_partitions) != request_count:
            raise ValueError("DSpark prefill requires one cache partition per request")
        groups = tuple(int(group) for group in batch.cache_partitions)
        if min(groups) < 0 or max(groups) >= layout.partitions:
            raise ValueError(
                f"DSpark prefill cache partitions must be in [0, {layout.partitions - 1}]"
            )
        if batch.input_embeddings is None:
            raise ValueError("DSpark prefill requires host input embeddings")
        if any(len(values) != request_count for values in (
            batch.chunk_lens, batch.chunk_starts, batch.chunk_offsets,
        )):
            raise ValueError("DSpark prefill requires one chunk length, start and offset per request")

        counts = [0] * layout.partitions
        group_lengths = [0] * layout.partitions
        packed_offsets = []
        request_ordinals = []
        for index, group in enumerate(groups):
            length = int(batch.chunk_lens[index])
            start = int(batch.chunk_starts[index])
            offset = int(batch.chunk_offsets[index])
            if length <= 0:
                raise ValueError("DSpark prefill chunk lengths must be positive")
            if start < 0 or start + length > model.runtime.max_seq_len:
                raise ValueError(
                    f"prefill chunk positions [{start}, {start + length}) "
                    f"exceed max_seq_len={model.runtime.max_seq_len}"
                )
            if offset < 0 or offset + length > min(
                batch.token_ids.shape[0], batch.input_embeddings.shape[0]
            ):
                raise ValueError("DSpark prefill chunk exceeds its token or embedding buffer")
            packed_offsets.append(group_lengths[group])
            request_ordinals.append(counts[group])
            counts[group] += 1
            group_lengths[group] += length
        request_rows = max(counts)
        if request_rows > min(layout.prefill_requests, layout.max_logit_rows):
            raise ValueError(
                f"DSpark prefill requests per TP group exceed capacity {layout.prefill_requests}"
            )

        builder = self.cache_metadata
        rope = self._require_rope_tables()
        tokens = self._prefill_kernel_tokens(max(group_lengths))
        local_tokens = tokens // layout.tp_size
        max_position = rope.max_position

        input_ids = torch.zeros((layout.ranks, local_tokens), dtype=torch.int64)
        position_ids_local = torch.zeros((layout.ranks, local_tokens), dtype=torch.int32)
        position_ids_full = torch.zeros((layout.ranks, tokens), dtype=torch.int32)
        group_input_ids = torch.zeros((layout.partitions, tokens), dtype=torch.int64)
        # Packed-prefill boundaries (pypto-lib#1095): monotonic per-rank starts
        # ending at the group's logical length; [0, 0] leaves a group idle.
        query_start_loc = torch.zeros(
            (layout.ranks, request_rows + 1), dtype=torch.int32
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
                (layout.ranks, request_rows, depth), -1, dtype=torch.int32
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
        actual_tokens_by_request: list[int] = []
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
            packed_start = packed_offsets[index]
            packed_end = packed_start + actual_tokens
            ordinal = request_ordinals[index]
            actual_tokens_by_request.append(actual_tokens)
            chunk_starts.append(chunk_start)
            ranks = tuple(
                range(group * layout.tp_size, (group + 1) * layout.tp_size)
            )
            positions = torch.arange(chunk_start, chunk_start + actual_tokens, dtype=torch.int64)
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
                rope_tables[name][rank_lo : rank_lo + layout.tp_size, packed_start:packed_end] = rows
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
            group_input_ids[group, packed_start:packed_end] = token_ids
            for rank in ranks:
                position_ids_full[rank, packed_start:packed_end] = positions_c.to(torch.int32)
            logit_row_indices[group * layout.tp_size, ordinal] = packed_end - 1

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
                query_start_loc[rank, ordinal + 1:] = packed_end
                for name, table in tables.items():
                    block_tables[name][rank, ordinal] = table
                for name, mapping in mappings.items():
                    slot_mappings[name][rank, packed_start:packed_end] = mapping.reshape(-1)

        for group, length in enumerate(group_lengths):
            for member in range(layout.tp_size):
                rank = group * layout.tp_size + member
                # Padding has no request ID and no cache publication slots.
                # Synthetic positions stay distinct from every live position.
                if length:
                    tail_start = int(position_ids_full[rank, :length].max()) + 1
                    position_ids_full[rank, length:] = torch.arange(
                        tail_start, tail_start + tokens - length, dtype=torch.int32
                    )
                lo = member * local_tokens
                input_ids[rank] = group_input_ids[group, lo:lo + local_tokens]
                position_ids_local[rank] = position_ids_full[rank, lo:lo + local_tokens]

        return DSparkPreparedPrefillInputs(
            request_ids=tuple(batch.request_ids),
            groups=groups,
            actual_tokens=tuple(actual_tokens_by_request),
            physical_tokens=tokens,
            chunk_starts=tuple(chunk_starts),
            packed_offsets=tuple(packed_offsets),
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
                (group * layout.tp_size, ordinal)
                for group, ordinal in zip(groups, request_ordinals, strict=True)
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
        for group, offset, embeddings in zip(
            inputs.groups, inputs.packed_offsets, inputs.embeddings, strict=True
        ):
            replicated = embeddings.unsqueeze(1).expand(-1, layout.hc_mult, -1)
            for rank in range(group * layout.tp_size, (group + 1) * layout.tp_size):
                x_hc[rank, offset:offset + embeddings.shape[0]].copy_(replicated)
        for name, value in values.items():
            destination = tensors[name]
            if name in _PREFILL_GROUP_DYNAMIC_NAMES:
                destination = self._packed_host_prefix(destination, inputs.physical_tokens)
            elif name in _PREFILL_LOCAL_DYNAMIC_NAMES:
                destination = self._packed_host_prefix(
                    destination, inputs.physical_tokens // layout.tp_size
                )
            elif name in _PREFILL_REQUEST_DYNAMIC_NAMES or name == "query_start_loc":
                destination = self._packed_host_prefix(destination, value.shape[1])
            copy_shared(destination, value, name=f"dspark_prefill_{name}")

        # Idle TP groups keep their zero-initialized staging (query_start_loc
        # terminal 0, -1 logit rows and cache mappings): the kernel skips their
        # attention and sampling tails natively while staying in the EP MoE
        # waves (pypto-lib#1161), so no mirror replay is staged.

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
            if self._compiled.decode_full_fused:
                plan = self.prepare_decode(model, batch, buffer_slot=buffer_slot)
                return self.run_prepared_decode(model, batch, plan)
            inputs = self.prepare_decode_inputs(model, batch, buffer_slot=buffer_slot)
            return self._execute_decode(batch, inputs)

    def prepare_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        buffer_slot: int,
    ) -> DSparkPreparedDecodePlan:
        """Build position-independent Host metadata while the prior step runs."""
        del model
        if self._compiled.decode is None:
            raise RuntimeError("DSpark kernels were not compiled for this runner")
        if not batch.allow_device_greedy_sampling:
            raise RuntimeError(
                "DSpark serving currently supports greedy decoding only "
                "(the kernels expose device greedy sampling; no temperature ABI yet)"
            )
        with profile_span("DSparkModelRunner.decode.prepare_early", cat="executor"):
            plan = self._prepare_decode_plan(batch, buffer_slot=buffer_slot)
            if self._compiled.decode_full_fused:
                inputs = self._stage_device_prepared_decode(plan)
                plan = replace(plan, dispatch_inputs=inputs)
            return plan

    def run_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> DecodeResult:
        """Late-bind mutable Host state and execute an early-prepared decode plan."""
        if not isinstance(prepared, DSparkPreparedDecodePlan):
            raise TypeError("DSpark prepared decode has an unexpected type")
        if tuple(batch.request_ids) != prepared.request_ids:
            raise ValueError("prepared DSpark decode request order changed before execution")
        if tuple(int(group) for group in batch.cache_partitions) != prepared.groups:
            raise ValueError("prepared DSpark decode cache partitions changed before execution")
        with profile_span("DSparkModelRunner.decode", cat="executor"):
            self._ensure_l3_shared_buffers(model)
            if self._compiled.decode_full_fused:
                return self.reclaim_prepared_decode(
                    self.dispatch_prepared_decode(model, batch, prepared)
                )
            with profile_span("DSparkModelRunner.decode.prepare_late", cat="executor"):
                inputs = self.prepare_decode_inputs(
                    model,
                    batch,
                    buffer_slot=prepared.buffer_slot,
                    plan=prepared,
                )
            return self._execute_decode(batch, inputs)

    def dispatch_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> object:
        """Submit a completely bound one-L2 snapshot without waiting for outputs."""
        if not self._compiled.decode_full_fused:
            raise RuntimeError("split DSpark decode reclaim requires the one-L2 path")
        if not isinstance(prepared, DSparkPreparedDecodePlan):
            raise TypeError("DSpark prepared decode has an unexpected type")
        if tuple(batch.request_ids) != prepared.request_ids:
            raise ValueError("prepared DSpark decode request order changed before execution")
        if tuple(int(group) for group in batch.cache_partitions) != prepared.groups:
            raise ValueError("prepared DSpark decode cache partitions changed before execution")
        inputs = prepared.dispatch_inputs
        if inputs is None or inputs.dispatch_args is None:
            raise RuntimeError("DSpark fused decode was not fully bound during prepare")
        self._ensure_l3_shared_buffers(model)
        for request_id in inputs.request_ids:
            if not self._drafter_state(request_id).device_state_initialized:
                raise RuntimeError(
                    f"DSpark device state was not finalized during prefill for {request_id!r}"
                )
        return self._launch_fused_decode(batch, inputs)

    def reclaim_prepared_decode(self, pending: object) -> DecodeResult:
        """Wait for one-L2 completion and materialize only compact Host outputs."""
        if not isinstance(pending, _DSparkPendingDecode):
            raise TypeError("DSpark pending decode has an unexpected type")
        return self._reclaim_fused_decode(pending)

    def _execute_decode(
        self,
        batch: DecodeBatch,
        inputs: DSparkPreparedDecodeInputs,
    ) -> DecodeResult:
        """Execute target, device acceptance, drafter, and Markov in FIFO order."""
        if self._compiled.decode_full_fused:
            return self._reclaim_fused_decode(self._launch_fused_decode(batch, inputs))
        with profile_span("DSparkModelRunner.decode.execute", cat="executor"):
            task_args = self._decode_task_args[inputs.buffer_slot]
            fused_device_state = self._compiled.decode_device_state_fused
            if (
                self.speculative
                and not fused_device_state
                and self._compiled.state_prepare is not None
            ):
                self._prepare_target_inputs_on_device(inputs.buffer_slot, task_args)
            args = task_args.build()
            if fused_device_state:
                args = (*args, *self._fused_decode_device_state_args(inputs.buffer_slot))
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
            num_draft_tokens = self._verified_draft_counts(inputs)
            if fused_device_state:
                accepted, rows_by_rank = self._accept_decode_outputs_on_device(
                    inputs, sampled, dispatch=False
                )
            elif self.speculative and self._compiled.state_accept is not None:
                accepted, rows_by_rank = self._accept_decode_outputs_on_device(
                    inputs, sampled
                )
            else:
                accepted, rows_by_rank = self._accept_decode_outputs_on_host(inputs, sampled)
            if any(rows_by_rank):
                self._run_decode_drafter(rows_by_rank, buffer_slot=inputs.buffer_slot)
            return DecodeResult(
                hidden_states=None,
                logits=None,
                accepted_token_ids=accepted,
                num_draft_tokens=num_draft_tokens,
            )

    def _verified_draft_counts(self, inputs: DSparkPreparedDecodeInputs) -> list[int] | None:
        """Count consumed drafts before acceptance or redrafting mutates Host state."""
        if not self.speculative:
            return None
        max_position = self._require_rope_tables().max_position
        counts: list[int] = []
        for request_id, speculative in zip(inputs.request_ids, inputs.speculative_flags, strict=True):
            state = self._drafter_state(request_id)
            count = len(state.pending_draft_tokens)
            # Fused prepare runs ahead of acceptance and marks every row as
            # speculative. FIFO reclaim provides the previous step's Host
            # mirror; committed_count is the current device anchor. Apply the
            # same position limit as device acceptance without a device read.
            if not speculative or state.committed_count + count >= max_position:
                count = 0
            counts.append(count)
        return counts

    def _bind_fused_decode_args(
        self,
        inputs: DSparkPreparedDecodeInputs,
    ) -> tuple[Any, ...]:
        """Bind every one-L2 argument while the prepared slot is Host-owned."""
        task_args = self._decode_task_args[inputs.buffer_slot]
        target_args = tuple(
            value
            for name, value in zip(
                task_args.names,
                task_args.build(),
                strict=True,
            )
            if name not in _DSPARK_FUSED_INTERNAL_PREPARE_NAMES
        )
        return (
            *target_args,
            *self._fused_decode_device_state_args(inputs.buffer_slot),
            *self._fused_decode_prepare_args(inputs.buffer_slot),
            *self._fused_decode_drafter_args(inputs.buffer_slot),
        )

    def _launch_fused_decode(
        self,
        batch: DecodeBatch,
        inputs: DSparkPreparedDecodeInputs,
    ) -> _DSparkPendingDecode:
        """Submit one command-lane-complete one-L2 decode snapshot."""
        args = inputs.dispatch_args
        if args is None:
            # Direct synchronous callers may bypass ``prepare_decode``; keep
            # that compatibility path outside the steady serving pipeline.
            args = self._bind_fused_decode_args(inputs)
        try:
            with profile_span(
                "DSparkModelRunner.decode.l3_dispatch",
                cat="executor",
                args={"actual_batch": len(batch.request_ids), "one_l2": True},
            ):
                dispatch = self._submit_l3(
                    self._compiled.decode,
                    *args,
                    config=self._decode_run_config,
                )
        except RuntimeError as exc:
            raise RuntimeError(
                "DSpark packed one-L2 decode dispatch failed "
                f"(actual_batch={len(batch.request_ids)})"
            ) from exc
        self._track_pending_decode_dispatch(inputs.buffer_slot, dispatch)
        return _DSparkPendingDecode(
            dispatch=dispatch,
            inputs=inputs,
            sampled_ids=self._decode_task_args[inputs.buffer_slot].tensors["sampled_ids"],
        )

    def _reclaim_fused_decode(self, pending: _DSparkPendingDecode) -> DecodeResult:
        """Read ping-ponged acceptance and diagnostic draft outputs."""
        try:
            pending.dispatch.wait()
        finally:
            self._forget_pending_decode_dispatch(
                pending.inputs.buffer_slot,
                pending.dispatch,
            )
        with profile_span("DSparkModelRunner.decode.reclaim", cat="executor"):
            num_draft_tokens = self._verified_draft_counts(pending.inputs)
            accepted, rows_by_rank = self._accept_decode_outputs_on_device(
                pending.inputs,
                pending.sampled_ids,
                dispatch=False,
            )
            self._collect_fused_decode_drafts(
                rows_by_rank,
                buffer_slot=pending.inputs.buffer_slot,
            )
        return DecodeResult(
            hidden_states=None,
            logits=None,
            accepted_token_ids=accepted,
            num_draft_tokens=num_draft_tokens,
        )

    def _fused_decode_device_state_args(self, buffer_slot: int) -> tuple[Any, ...]:
        """Append the persistent-state ABI consumed by the fused target L3."""
        state_buffers = self._dspark_state_buffers[buffer_slot]
        return (
            state_buffers.state_slot_ids,
            state_buffers.state_generations,
            self._materialize_dspark_device_state_tokens(),
            self._materialize_dspark_device_state_meta(),
            state_buffers.accepted_token_ids,
            state_buffers.accepted_counts,
        )

    def _fused_decode_drafter_args(self, buffer_slot: int) -> tuple[Any, ...]:
        """Append the drafter/Markov ABI consumed inside the one-L2 decode."""
        if self._fused_drafter_task_args and self._fused_markov_task_args:
            drafter_task_args = self._fused_drafter_task_args[buffer_slot]
            markov_task_args = self._fused_markov_task_args[buffer_slot]
            batch = self._fused_drafter_batches[buffer_slot]
            block_tables = self._fused_drafter_block_tables[buffer_slot]
        else:
            drafter_task_args = self._drafter_task_args
            markov_task_args = self._markov_task_args
            batch = self._fused_drafter_batch
            block_tables = self._drafter_block_table_staging
        if drafter_task_args is None or markov_task_args is None:
            raise RuntimeError("DSpark fused decode TaskArgs are not staged")
        if batch is None:
            raise RuntimeError("DSpark fused drafter batch was not staged")
        context_rows = batch * self._compiled.layout.decode_seq
        drafter_args = self._drafter_dispatch_args(
            batch,
            context_rows,
            task_args=drafter_task_args,
            block_table_staging=block_tables,
        )
        drafter = dict(
            zip(drafter_task_args.names, drafter_args, strict=True)
        )
        markov_args = self._markov_dispatch_args(batch, task_args=markov_task_args)
        markov = dict(zip(markov_task_args.names, markov_args, strict=True))

        drafter_order = (
            "initial_hidden", "intermediate_hidden",
            "main_proj_weight", "main_norm_weight",
            "embedding_weight", "block_tables",
        )
        args = [drafter[name] for name in drafter_order]
        args.extend(
            drafter[name]
            for name in (
                "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
                "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
                "kv_caches", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
                "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm_w",
                "gate_w", "gate_bias", "tid2eid",
                "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
                "routed_w2", "routed_w2_scale",
                "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
                "shared_w2", "shared_w2_scale",
                "hc_head_fn", "hc_head_scale", "hc_head_base", "head_hidden",
            )
        )
        args.extend(
            markov[name]
            for name in (
                "final_norm_weight", "lm_head_weight",
                "markov_w1", "markov_w2", "confidence_head_weight",
                "draft_token_ids", "confidence_probs",
            )
        )
        return tuple(args)

    def _fused_decode_prepare_args(self, buffer_slot: int) -> tuple[Any, ...]:
        """Append group descriptors and resident RoPE tables for device prepare."""
        state_buffers = self._dspark_state_buffers[buffer_slot]
        if (
            state_buffers.group_state_slot_ids is None
            or state_buffers.group_state_generations is None
            or state_buffers.group_ori_block_tables is None
            or state_buffers.group_hca_cmp_block_tables is None
            or state_buffers.group_csa_cmp_block_tables is None
            or state_buffers.group_idx_block_tables is None
            or state_buffers.group_hca_state_block_tables is None
            or state_buffers.group_csa_state_block_tables is None
            or state_buffers.group_csa_inner_state_block_tables is None
        ):
            raise RuntimeError("DSpark group device-prepare buffers are unavailable")
        rope = self._materialize_dspark_rope_tables()
        return (
            state_buffers.group_state_slot_ids,
            state_buffers.group_state_generations,
            state_buffers.group_ori_block_tables,
            state_buffers.group_hca_cmp_block_tables,
            state_buffers.group_csa_cmp_block_tables,
            state_buffers.group_idx_block_tables,
            state_buffers.group_hca_state_block_tables,
            state_buffers.group_csa_state_block_tables,
            state_buffers.group_csa_inner_state_block_tables,
            rope["swa_rope_cos_table"],
            rope["swa_rope_sin_table"],
            rope["ratio4_rope_cos_table"],
            rope["ratio4_rope_sin_table"],
            rope["ratio128_rope_cos_table"],
            rope["ratio128_rope_sin_table"],
        )

    def _collect_fused_decode_drafts(
        self,
        rows_by_rank: list[list[DSparkDrafterRequestRow]],
        *,
        buffer_slot: int,
    ) -> None:
        """Mirror the one-L2 Markov outputs into request scheduling state."""
        markov_task_args = (
            self._fused_markov_task_args[buffer_slot]
            if self._fused_markov_task_args
            else self._markov_task_args
        )
        if markov_task_args is None:
            raise RuntimeError("DSpark fused Markov outputs are not staged")
        batch = (
            self._fused_drafter_batches[buffer_slot]
            if self._fused_markov_task_args
            else self._fused_drafter_batch
        )
        if batch is None:
            raise RuntimeError("DSpark fused drafter batch was not staged")
        drafts = self._packed_host_prefix(
            markov_task_args.tensors["draft_token_ids"], batch
        )
        confidence = self._packed_host_prefix(
            markov_task_args.tensors["confidence_probs"], batch
        )
        for rank, rows in enumerate(rows_by_rank):
            for index, row in enumerate(rows):
                state = self._drafter_state(row.request_id)
                state.pending_draft_tokens = [int(value) for value in drafts[rank, index]]
                state.pending_confidence = [
                    float(value) for value in confidence[rank, index]
                ]
                if os.environ.get("PYPTO_DSPARK_DEBUG_ACCEPT") == "1" and state.verify_steps <= 3:
                    logger.info(
                        "DSpark redraft debug: path=one_l2 request=%s verify=%d rank=%d "
                        "row=%d anchor=%d accepted=%d last=%d drafts=%s confidence=%s",
                        row.request_id,
                        state.verify_steps,
                        rank,
                        index,
                        row.anchor,
                        row.valid_count,
                        row.token_source,
                        state.pending_draft_tokens,
                        [round(value, 6) for value in state.pending_confidence],
                    )
        self._active_drafter_target_hidden = None

    def _prepare_target_inputs_on_device(
        self,
        buffer_slot: int,
        task_args: TaskArgs,
    ) -> None:
        """Late-bind recurrent target inputs from persistent device state."""
        program = self._compiled.state_prepare
        if program is None or not self._dspark_state_buffers:
            raise RuntimeError("DSpark device-state prepare program is not available")
        state_buffers = self._dspark_state_buffers[buffer_slot]
        tensors = task_args.tensors
        with profile_span("DSparkModelRunner.decode.prepare_device", cat="executor"):
            self._run_l3(
                program,
                state_buffers.state_slot_ids,
                state_buffers.state_generations,
                self._materialize_dspark_device_state_tokens(),
                self._materialize_dspark_device_state_meta(),
                tensors["input_ids"],
                tensors["position_ids_local"],
                tensors["csa_kv_seq_lens"],
                tensors["hca_kv_seq_lens"],
                state_buffers.accepted_counts,
                config=self._decode_run_config,
            )

    def _accept_decode_outputs_on_device(
        self,
        inputs: DSparkPreparedDecodeInputs,
        sampled: torch.Tensor,
        *,
        dispatch: bool = True,
    ) -> tuple[list[list[int]], list[list[DSparkDrafterRequestRow]]]:
        """Run K=7 prefix acceptance and recurrent-state advance on device."""
        program = self._compiled.state_accept
        if (dispatch and program is None) or not self._dspark_state_buffers:
            raise RuntimeError("DSpark device-state acceptance program is not available")
        layout = self._compiled.layout
        state_buffers = self._dspark_state_buffers[inputs.buffer_slot]
        drafter_hidden = None
        if dispatch:
            if any(
                value is None
                for value in (
                    state_buffers.sampled_row_offsets,
                    state_buffers.hidden_row_offsets,
                    state_buffers.context_positions,
                    state_buffers.context_valid,
                    state_buffers.last_sampled,
                    state_buffers.anchor_positions,
                    state_buffers.drafter_row_offsets,
                )
            ):
                raise RuntimeError("DSpark standalone accept scratch is unavailable")
            target_hidden = self._alloc_zeroed_stacked_tensor(
                "dspark_target_hidden",
                (layout.ranks, DSPARK_DECODE_LOCAL_TOKENS, DSPARK_MAIN_HIDDEN_DIM),
                torch.bfloat16,
                scope="decode",
            )
            drafter_hidden = self._alloc_zeroed_stacked_tensor(
                "device_state_target_hidden",
                (layout.ranks, DSPARK_DRAFTER_CONTEXT_ROWS, DSPARK_MAIN_HIDDEN_DIM),
                torch.bfloat16,
                scope="drafter",
            )
            with profile_span("DSparkModelRunner.decode.accept_device", cat="executor"):
                self._run_l3(
                    program,
                    state_buffers.state_slot_ids,
                    state_buffers.state_generations,
                    state_buffers.sampled_row_offsets,
                    state_buffers.hidden_row_offsets,
                    self._materialize_dspark_device_state_tokens(),
                    self._materialize_dspark_device_state_meta(),
                    sampled,
                    target_hidden,
                    state_buffers.accepted_token_ids,
                    state_buffers.accepted_counts,
                    drafter_hidden,
                    state_buffers.context_positions,
                    state_buffers.context_valid,
                    state_buffers.last_sampled,
                    state_buffers.anchor_positions,
                    state_buffers.drafter_row_offsets,
                    config=self._drafter_run_config,
                )

        accepted: list[list[int]] = []
        rows_by_rank: list[list[DSparkDrafterRequestRow]] = [
            [] for _ in range(layout.ranks)
        ]
        for request_id, rank, local_row in zip(
            inputs.request_ids, inputs.owner_ranks, inputs.owner_rows, strict=True
        ):
            state = self._drafter_state(request_id)
            count = int(state_buffers.accepted_counts[rank, local_row].item())
            if count < 1 or count > layout.decode_seq:
                raise RuntimeError(
                    f"DSpark device state returned invalid accepted count {count} "
                    f"for request {request_id!r}"
                )
            tokens = [
                int(token)
                for token in state_buffers.accepted_token_ids[rank, local_row, :count]
            ]
            accepted.append(tokens)
            state.verify_steps += 1
            state.proposed_tokens += DSPARK_DRAFTER_QUERY_WIDTH
            state.matched_drafts += count - 1
            state.accepted_tokens += count
            state.committed_count += count
            state.current_token_id = tokens[-1]
            # The standalone accept L3 publishes its compacted drafter row and
            # anchor through Host-visible scratch.  In the one-L2 path those
            # values are deliberately invocation-local: dense owner rows map
            # directly to the dense drafter batch, and the committed count is
            # the authoritative next anchor.  Reading the legacy scratch here
            # would reintroduce a stale cross-invocation dependency.
            drafter_row = (
                int(state_buffers.drafter_row_offsets[rank, local_row].item())
                if dispatch
                else local_row
            )
            if drafter_row >= 0:
                anchor = (
                    int(state_buffers.anchor_positions[rank, local_row].item())
                    if dispatch
                    else state.committed_count - 1
                )
                rows_by_rank[rank].append(
                    DSparkDrafterRequestRow(
                        request_id=request_id,
                        group=state.group,
                        lease=state.lease,
                        anchor=anchor,
                        valid_count=count,
                        token_source=tokens[-1],
                        hidden_row=drafter_row,
                        decode_mode=True,
                    )
                )
            else:
                state.pending_draft_tokens = []
                state.pending_confidence = []
        if drafter_hidden is not None:
            self._debug_dump_accepted_drafter_hidden(drafter_hidden, rows_by_rank)
        self._active_drafter_target_hidden = drafter_hidden
        self._maybe_log_acceptance()
        return accepted, rows_by_rank

    def _accept_decode_outputs_on_host(
        self,
        inputs: DSparkPreparedDecodeInputs,
        sampled: torch.Tensor,
    ) -> tuple[list[list[int]], list[list[DSparkDrafterRequestRow]]]:
        """Accept target samples and advance authoritative speculative Host state."""
        with profile_span("DSparkModelRunner.decode.accept_host", cat="executor"):
            accepted: list[list[int]] = []
            rows_by_rank: list[list[DSparkDrafterRequestRow]] = [
                [] for _ in range(self._compiled.layout.ranks)
            ]
            for index, (rank, row) in enumerate(inputs.sampled_slots):
                request_id = inputs.request_ids[index]
                state = self._drafter_states.get(request_id) if self.speculative else None
                if state is not None and inputs.speculative_flags[index]:
                    main = [
                        int(sampled[rank, row + offset, 0].item())
                        for offset in range(self._compiled.layout.decode_seq)
                    ]
                    tokens, matched = _accept_dspark_tokens(main, state.pending_draft_tokens)
                    state.verify_steps += 1
                    state.proposed_tokens += DSPARK_DRAFTER_QUERY_WIDTH
                    state.matched_drafts += matched
                    state.accepted_tokens += len(tokens)
                    state.committed_count += len(tokens)
                    accepted.append(tokens)
                    # The committed inputs span positions p..p+m (m+1 tokens),
                    # so the drafter's next anchor -- the last committed input
                    # position and the end of its context window -- is p+m,
                    # one before the next verify's anchor.
                    anchor = inputs.anchor_positions[index]
                    rows_by_rank[rank].append(
                        DSparkDrafterRequestRow(
                            request_id=request_id,
                            group=state.group,
                            lease=state.lease,
                            anchor=anchor + len(tokens) - 1,
                            valid_count=len(tokens),
                            token_source=tokens[-1],
                            hidden_row=inputs.verify_hidden_rows[index],
                            decode_mode=True,
                        )
                    )
                else:
                    accepted.append([int(sampled[rank, row, 0].item())])
                    if state is not None:
                        state.verify_steps += 1
                        state.accepted_tokens += 1
                        state.committed_count += 1
            self._maybe_log_acceptance()
            return accepted, rows_by_rank

    def _run_decode_drafter(
        self,
        rows_by_rank: list[list[DSparkDrafterRequestRow]],
        *,
        buffer_slot: int,
    ) -> None:
        """Redraft from the just-committed rows of every speculative request."""
        layout = self._compiled.layout
        # A request whose seven query positions would cross the position
        # ceiling simply gets no next draft: its state empties and the next
        # verify falls back to the single-anchor path instead of raising.
        max_position = self._require_rope_tables().max_position
        kept: list[list[DSparkDrafterRequestRow]] = [[] for _ in rows_by_rank]
        for rank, rows in enumerate(rows_by_rank):
            for row in rows:
                if row.anchor + DSPARK_DRAFTER_QUERY_WIDTH < max_position:
                    kept[rank].append(row)
                else:
                    state = self._drafter_state(row.request_id)
                    state.pending_draft_tokens = []
                    state.pending_confidence = []
        if not any(kept):
            return
        rows_by_rank = kept
        batch = next(
            size
            for size in DSPARK_DRAFTER_BATCHES
            if max(len(rows) for rows in rows_by_rank) <= size
        )
        context_rows = batch * layout.decode_seq
        # Device acceptance already packs each committed target-hidden span
        # into the dense drafter context layout.  No target-hidden D2H bounce
        # is needed in the steady-state loop.
        device_hidden = self._active_drafter_target_hidden
        host_hidden = None
        if device_hidden is None:
            mirror = self._read_drafter_hidden(
                self._drafter_hidden_mirror, rows=DSPARK_DRAFTER_CONTEXT_ROWS
            )
            host_hidden = torch.zeros(
                (layout.ranks, context_rows, DSPARK_MAIN_HIDDEN_DIM), dtype=torch.bfloat16
            )
            for rank, rows in enumerate(rows_by_rank):
                for index, row in enumerate(rows):
                    source = mirror[rank, row.hidden_row : row.hidden_row + row.valid_count]
                    host_hidden[
                        rank,
                        index * layout.decode_seq : index * layout.decode_seq + row.valid_count,
                    ] = source
        self._prepare_drafter_inputs(
            rows_by_rank, hidden=host_hidden, context_rows=context_rows
        )
        self._run_drafter_and_markov(
            batch,
            context_rows,
            target_hidden=device_hidden,
            buffer_slot=buffer_slot,
            rows_by_rank=rows_by_rank,
        )
        drafts = self._packed_host_prefix(
            self._markov_task_args.tensors["draft_token_ids"], batch
        )
        confidence = self._packed_host_prefix(
            self._markov_task_args.tensors["confidence_probs"], batch
        )
        for rank, rows in enumerate(rows_by_rank):
            for index, row in enumerate(rows):
                state = self._drafter_state(row.request_id)
                state.pending_draft_tokens = [int(v) for v in drafts[rank, index]]
                state.pending_confidence = [float(v) for v in confidence[rank, index]]
                if os.environ.get("PYPTO_DSPARK_DEBUG_ACCEPT") == "1" and state.verify_steps <= 3:
                    logger.info(
                        "DSpark redraft debug: path=split_l3 request=%s verify=%d rank=%d "
                        "row=%d anchor=%d accepted=%d last=%d drafts=%s confidence=%s",
                        row.request_id,
                        state.verify_steps,
                        rank,
                        index,
                        row.anchor,
                        row.valid_count,
                        row.token_source,
                        state.pending_draft_tokens,
                        [round(value, 6) for value in state.pending_confidence],
                    )
        if (
            not self._compiled.draft_device_state_fused
            and self._compiled.state_commit is not None
        ):
            self._commit_dspark_drafts(
                buffer_slot=buffer_slot,
                batch=batch,
                rows_by_rank=rows_by_rank,
            )
        self._active_drafter_target_hidden = None

    def _maybe_log_acceptance(self) -> None:
        """Periodically report acceptance progress across live requests."""
        states = list(self._drafter_states.values())
        if not states:
            return
        self._acceptance_log_steps += 1
        # The first line is unconditional: a completion-time summary races
        # the worker shutdown after the last response, but every speculative
        # run emits this line at its first verify step.
        if self._acceptance_log_steps > 1 and self._acceptance_log_steps % 10:
            return
        proposed = sum(state.proposed_tokens for state in states)
        matched = sum(state.matched_drafts for state in states)
        accepted = sum(state.accepted_tokens for state in states)
        verifies = sum(state.verify_steps for state in states)
        fallbacks = sum(state.fallback_steps for state in states)
        logger.info(
            "DSpark speculation progress: requests=%d verifies=%d matched=%d "
            "proposed=%d accepted=%d mean_len=%.2f fallbacks=%d",
            len(states),
            verifies,
            matched,
            proposed,
            accepted,
            (accepted / verifies) if verifies else 0.0,
            fallbacks,
        )

    def dspark_speculation_summary(self) -> dict[str, float]:
        """Aggregate speculation counters before scheduler truncation."""
        states = list(self._drafter_states.values())
        verifies = sum(state.verify_steps for state in states)
        return {
            "requests": float(len(states)),
            "verify_steps": float(verifies),
            "proposed_drafts": float(sum(state.proposed_tokens for state in states)),
            "matched_drafts": float(sum(state.matched_drafts for state in states)),
            "accepted_tokens": float(sum(state.accepted_tokens for state in states)),
            "fallback_steps": float(sum(state.fallback_steps for state in states)),
            "mean_accepted_length": (
                sum(state.accepted_tokens for state in states) / verifies
            )
            if verifies
            else 0.0,
        }

    def _correct_dspark_seq_lens(self, batch: DecodeBatch) -> torch.Tensor:
        """Return request lengths corrected from the committed token stream.

        Async scheduling reserves the full speculative width when it queues
        the next decode command, before acceptance is known: after a verify
        at anchor 64 accepts one token, the queued command's ``seq_lens``
        already counts all seven reserved rows (73) instead of 65.  Once a
        request is seeded, the runner's committed count -- prompt plus every
        accepted token -- is the authoritative length, mirroring the MTP
        runner's ``_correct_mtp_seq_lens``.
        """
        actual_batch = len(batch.request_ids)
        corrected = batch.seq_lens[:actual_batch].detach().cpu().to(torch.int64).clone()
        if not self.speculative:
            return corrected
        for index, request_id in enumerate(batch.request_ids):
            state = self._drafter_states.get(request_id)
            if state is not None and state.prompt_len > 0:
                corrected[index] = state.committed_count + 1
        return corrected

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
            owner_counts = [0] * layout.tp_size
            for ordinal, request_index in enumerate(request_indices):
                # The group stream is rank-major: slot ``s`` lives on TP rank
                # ``s // local_batch`` at its local row ``s % local_batch``.
                # Persistent state keeps its lease-selected owner across batch
                # compaction; other paths spread ordinals round-robin. Both
                # assignments keep every rank's active local rows dense.
                if self.speculative and (
                    self._compiled.state_accept is not None
                    or self._compiled.decode_full_fused
                ):
                    state = self._drafter_state(batch.request_ids[request_index])
                    if state.group != group:
                        raise RuntimeError(
                            f"DSpark state group changed for {batch.request_ids[request_index]!r}: "
                            f"state={state.group}, scheduled={group}"
                        )
                    owner = state.lease % layout.tp_size
                    local_row = owner_counts[owner]
                    owner_counts[owner] += 1
                    if local_row >= local_batch:
                        raise ValueError(
                            f"DSpark TP owner rank {owner} exceeds local capacity {local_batch}"
                        )
                    stream_slot = owner * local_batch + local_row
                else:
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

    def _prepare_decode_plan(
        self,
        batch: DecodeBatch,
        *,
        buffer_slot: int,
    ) -> DSparkPreparedDecodePlan:
        """Build immutable Host-only metadata that does not depend on prior output."""
        if buffer_slot < 0 or buffer_slot >= len(self._decode_task_args):
            raise ValueError(
                f"DSpark decode buffer_slot must be in [0, {len(self._decode_task_args)}), "
                f"got {buffer_slot}"
            )

        layout = self._compiled.layout
        assignment = self._decode_assignment(batch)
        actual_batch = len(batch.request_ids)
        group_batch = layout.decode_local_batch * layout.tp_size
        builder = self.cache_metadata
        scratch = self._scratch_blocks()
        request_blocks = tuple(
            self._normalize_group_block_ids(
                batch.block_ids_by_group,
                actual_batch=actual_batch,
            )
        )
        group_plans: list[_DSparkDecodeGroupPlan] = []
        owner_ranks = [-1] * actual_batch
        owner_rows = [-1] * actual_batch
        state_slot_ids = torch.full(
            (layout.ranks, layout.decode_local_batch), -1, dtype=torch.int32
        )
        state_generations = torch.full_like(state_slot_ids, -1)
        group_state_slot_ids = torch.full(
            (layout.ranks, layout.decode_batch), -1, dtype=torch.int32
        )
        group_state_generations = torch.full_like(group_state_slot_ids, -1)
        group_ori_block_tables = torch.full(
            (
                layout.ranks,
                layout.decode_batch,
                DSPARK_DECODE_ORI_TABLE_BLOCKS,
            ),
            -1,
            dtype=torch.int32,
        )
        group_hca_cmp_block_tables = torch.full(
            (
                layout.ranks,
                layout.decode_batch,
                DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS,
            ),
            -1,
            dtype=torch.int32,
        )
        group_csa_cmp_block_tables = torch.full(
            (
                layout.ranks,
                layout.decode_batch,
                DSPARK_DECODE_CMP_C4_TABLE_BLOCKS,
            ),
            -1,
            dtype=torch.int32,
        )
        group_idx_block_tables = torch.full(
            (
                layout.ranks,
                layout.decode_batch,
                DSPARK_DECODE_IDX_TABLE_BLOCKS,
            ),
            -1,
            dtype=torch.int32,
        )
        group_hca_state_block_tables = torch.full(
            (
                layout.ranks,
                layout.decode_batch,
                self._decode_state_table_depth("hca_state"),
            ),
            -1,
            dtype=torch.int32,
        )
        group_csa_state_block_tables = torch.full(
            (
                layout.ranks,
                layout.decode_batch,
                self._decode_state_table_depth("csa_state"),
            ),
            -1,
            dtype=torch.int32,
        )
        group_csa_inner_state_block_tables = torch.full_like(
            group_csa_state_block_tables,
            -1,
        )
        sampled_row_offsets = torch.full_like(state_slot_ids, -1)
        hidden_row_offsets = torch.full_like(state_slot_ids, -1)
        for group in range(layout.partitions):
            request_indices: list[int | None] = [None] * group_batch
            for request_index, stream_slot in assignment.active_by_group[group]:
                request_indices[stream_slot] = request_index
                if self.speculative:
                    request_id = batch.request_ids[request_index]
                    state = self._drafter_state(request_id)
                    if state.group != group:
                        raise RuntimeError(
                            f"DSpark state group changed for {request_id!r}: "
                            f"state={state.group}, scheduled={group}"
                        )
                    rank = group * layout.tp_size + stream_slot // layout.decode_local_batch
                    local_row = stream_slot % layout.decode_local_batch
                    owner_ranks[request_index] = rank
                    owner_rows[request_index] = local_row
                    state_slot_ids[rank, local_row] = state.lease
                    state_generations[rank, local_row] = state.generation
                    group_rank_begin = group * layout.tp_size
                    group_rank_end = group_rank_begin + layout.tp_size
                    group_state_slot_ids[
                        group_rank_begin:group_rank_end, stream_slot
                    ] = state.lease
                    group_state_generations[
                        group_rank_begin:group_rank_end, stream_slot
                    ] = state.generation
                    sampled_row_offsets[rank, local_row] = local_row * layout.decode_seq
                    hidden_row_offsets[rank, local_row] = local_row * layout.decode_seq
            anchor_flags = torch.tensor(
                [request_index is not None for request_index in request_indices],
                dtype=torch.bool,
            )

            def blocks(row: int, name: str) -> tuple[int, ...]:
                request_index = request_indices[row]
                if request_index is None:
                    return (scratch[name][row],)
                return request_blocks[request_index][name]

            host_state_tables = not self._compiled.decode_full_fused
            group_plan = _DSparkDecodeGroupPlan(
                request_indices=tuple(request_indices),
                anchor_flags=anchor_flags,
                ori_tables=torch.stack(
                    [
                        builder.ring_table(
                            blocks(row, "ori"),
                            depth=DSPARK_DECODE_ORI_TABLE_BLOCKS,
                        )
                        for row in range(group_batch)
                    ]
                ),
                hca_cmp_tables=torch.stack(
                    [
                        builder.absolute_table(
                            blocks(row, "cmp_c128"),
                            depth=DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS,
                        )
                        for row in range(group_batch)
                    ]
                ),
                csa_cmp_tables=torch.stack(
                    [
                        builder.absolute_table(
                            blocks(row, "cmp_c4"),
                            depth=DSPARK_DECODE_CMP_C4_TABLE_BLOCKS,
                        )
                        for row in range(group_batch)
                    ]
                ),
                idx_tables=torch.stack(
                    [
                        builder.absolute_table(
                            blocks(row, "idx"),
                            depth=DSPARK_DECODE_IDX_TABLE_BLOCKS,
                        )
                        for row in range(group_batch)
                    ]
                ),
                hca_state_tables=torch.stack(
                    [
                        builder.ring_table(
                            blocks(row, "hca_state"),
                            depth=DSPARK_DECODE_HCA_STATE_TABLE_BLOCKS,
                        )
                        for row in range(group_batch)
                    ]
                ) if host_state_tables else None,
                csa_state_tables=torch.stack(
                    [
                        builder.ring_table(
                            blocks(row, "csa_state"),
                            depth=DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS,
                        )
                        for row in range(group_batch)
                    ]
                ) if host_state_tables else None,
                csa_inner_state_tables=torch.stack(
                    [
                        builder.ring_table(
                            blocks(row, "csa_inner_state"),
                            depth=DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS,
                        )
                        for row in range(group_batch)
                    ]
                ) if host_state_tables else None,
            )
            group_plans.append(group_plan)
            group_rank_begin = group * layout.tp_size
            group_rank_end = group_rank_begin + layout.tp_size
            group_ori_block_tables[group_rank_begin:group_rank_end] = (
                group_plan.ori_tables.unsqueeze(0)
            )
            group_hca_cmp_block_tables[group_rank_begin:group_rank_end] = (
                group_plan.hca_cmp_tables.unsqueeze(0)
            )
            group_csa_cmp_block_tables[group_rank_begin:group_rank_end] = (
                group_plan.csa_cmp_tables.unsqueeze(0)
            )
            group_idx_block_tables[group_rank_begin:group_rank_end] = (
                group_plan.idx_tables.unsqueeze(0)
            )
            group_hca_state_tables = torch.stack(
                [
                    builder.ring_table(
                        blocks(row, "hca_state"),
                        depth=self._decode_state_table_depth("hca_state"),
                    )
                    for row in range(group_batch)
                ]
            )
            group_csa_state_tables = torch.stack(
                [
                    builder.ring_table(
                        blocks(row, "csa_state"),
                        depth=self._decode_state_table_depth("csa_state"),
                    )
                    for row in range(group_batch)
                ]
            )
            group_csa_inner_state_tables = torch.stack(
                [
                    builder.ring_table(
                        blocks(row, "csa_inner_state"),
                        depth=self._decode_state_table_depth("csa_state"),
                    )
                    for row in range(group_batch)
                ]
            )
            group_hca_state_block_tables[group_rank_begin:group_rank_end] = (
                group_hca_state_tables.unsqueeze(0)
            )
            group_csa_state_block_tables[group_rank_begin:group_rank_end] = (
                group_csa_state_tables.unsqueeze(0)
            )
            group_csa_inner_state_block_tables[group_rank_begin:group_rank_end] = (
                group_csa_inner_state_tables.unsqueeze(0)
            )
        prepared = DSparkPreparedDecodePlan(
            request_ids=tuple(batch.request_ids),
            groups=assignment.groups,
            group_ordinals=assignment.ordinals,
            assignment=assignment,
            request_blocks=request_blocks,
            group_plans=tuple(group_plans),
            state_slot_ids=state_slot_ids,
            state_generations=state_generations,
            group_state_slot_ids=group_state_slot_ids,
            group_state_generations=group_state_generations,
            group_ori_block_tables=group_ori_block_tables,
            group_hca_cmp_block_tables=group_hca_cmp_block_tables,
            group_csa_cmp_block_tables=group_csa_cmp_block_tables,
            group_idx_block_tables=group_idx_block_tables,
            group_hca_state_block_tables=group_hca_state_block_tables,
            group_csa_state_block_tables=group_csa_state_block_tables,
            group_csa_inner_state_block_tables=group_csa_inner_state_block_tables,
            sampled_row_offsets=sampled_row_offsets,
            hidden_row_offsets=hidden_row_offsets,
            owner_ranks=tuple(owner_ranks),
            owner_rows=tuple(owner_rows),
            buffer_slot=buffer_slot,
        )
        if self.speculative and self._dspark_state_buffers:
            state_buffers = self._dspark_state_buffers[buffer_slot]
            copy_shared(
                state_buffers.state_slot_ids,
                prepared.state_slot_ids,
                name="dspark_state_slot_ids",
            )
            copy_shared(
                state_buffers.state_generations,
                prepared.state_generations,
                name="dspark_state_generations",
            )
            if (
                state_buffers.group_state_slot_ids is None
                or state_buffers.group_state_generations is None
                or state_buffers.group_ori_block_tables is None
                or state_buffers.group_hca_cmp_block_tables is None
                or state_buffers.group_csa_cmp_block_tables is None
                or state_buffers.group_idx_block_tables is None
                or state_buffers.group_hca_state_block_tables is None
                or state_buffers.group_csa_state_block_tables is None
                or state_buffers.group_csa_inner_state_block_tables is None
            ):
                raise RuntimeError("DSpark group device-state descriptors are unavailable")
            copy_shared(
                state_buffers.group_state_slot_ids,
                prepared.group_state_slot_ids,
                name="dspark_group_state_slot_ids",
            )
            copy_shared(
                state_buffers.group_state_generations,
                prepared.group_state_generations,
                name="dspark_group_state_generations",
            )
            copy_shared(
                state_buffers.group_ori_block_tables,
                prepared.group_ori_block_tables,
                name="dspark_group_ori_block_tables",
            )
            copy_shared(
                state_buffers.group_hca_cmp_block_tables,
                prepared.group_hca_cmp_block_tables,
                name="dspark_group_hca_cmp_block_tables",
            )
            copy_shared(
                state_buffers.group_csa_cmp_block_tables,
                prepared.group_csa_cmp_block_tables,
                name="dspark_group_csa_cmp_block_tables",
            )
            copy_shared(
                state_buffers.group_idx_block_tables,
                prepared.group_idx_block_tables,
                name="dspark_group_idx_block_tables",
            )
            copy_shared(
                state_buffers.group_hca_state_block_tables,
                prepared.group_hca_state_block_tables,
                name="dspark_group_hca_state_block_tables",
            )
            copy_shared(
                state_buffers.group_csa_state_block_tables,
                prepared.group_csa_state_block_tables,
                name="dspark_group_csa_state_block_tables",
            )
            copy_shared(
                state_buffers.group_csa_inner_state_block_tables,
                prepared.group_csa_inner_state_block_tables,
                name="dspark_group_csa_inner_state_block_tables",
            )
            if not self._compiled.decode_full_fused:
                if state_buffers.hidden_row_offsets is None:
                    raise RuntimeError("DSpark hidden-row scratch is unavailable")
                copy_shared(
                    state_buffers.hidden_row_offsets,
                    prepared.hidden_row_offsets,
                    name="dspark_hidden_row_offsets",
                )
        return prepared

    def _stage_device_prepared_decode(
        self, plan: DSparkPreparedDecodePlan
    ) -> DSparkPreparedDecodeInputs:
        """Stage only acceptance-independent inputs for a fused device-state step."""
        layout = self._compiled.layout
        task_args = self._decode_task_args[plan.buffer_slot]
        staged = task_args.tensors
        local_batch = layout.decode_local_batch
        staged["num_tokens_per_owner"].zero_()
        for group, group_plan in enumerate(plan.group_plans):
            ranks = range(group * layout.tp_size, (group + 1) * layout.tp_size)
            for rank in ranks:
                tp_rank = rank % layout.tp_size
                local_requests = slice(
                    tp_rank * local_batch,
                    (tp_rank + 1) * local_batch,
                )
                active_requests = int(
                    group_plan.anchor_flags[local_requests].sum().item()
                )
                staged["num_tokens_per_owner"][rank] = (
                    active_requests * layout.decode_seq
                )
                staged["csa_cmp_block_table"][rank].copy_(
                    group_plan.csa_cmp_tables[local_requests]
                )
                staged["csa_idx_block_table"][rank].copy_(
                    group_plan.idx_tables[local_requests]
                )
                staged["hca_cmp_block_table"][rank].copy_(
                    group_plan.hca_cmp_tables[local_requests]
                )
                # State transaction tables are acceptance-dependent and are
                # rebuilt by the device preamble from the physical rings.
        self._stage_fused_decode_drafter_inputs(
            request_ids=plan.request_ids,
            anchors=(0,) * len(plan.request_ids),
            owner_ranks=plan.owner_ranks,
            owner_rows=plan.owner_rows,
            stage_rope=False,
            buffer_slot=plan.buffer_slot,
        )
        sampled_slots = tuple(
            (rank, row * layout.decode_seq)
            for rank, row in zip(plan.owner_ranks, plan.owner_rows, strict=True)
        )
        inputs = DSparkPreparedDecodeInputs(
            request_ids=plan.request_ids,
            groups=plan.groups,
            group_ordinals=plan.group_ordinals,
            anchor_positions=(0,) * len(plan.request_ids),
            input_ids=None,
            position_ids_local=None,
            position_ids=None,
            logit_row_indices=None,
            sampled_slots=sampled_slots,
            speculative_flags=(True,) * len(plan.request_ids),
            verify_hidden_rows=tuple(
                row * layout.decode_seq for row in plan.owner_rows
            ),
            owner_ranks=plan.owner_ranks,
            owner_rows=plan.owner_rows,
            buffer_slot=plan.buffer_slot,
        )
        return replace(inputs, dispatch_args=self._bind_fused_decode_args(inputs))

    def prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        buffer_slot: int = 0,
        plan: DSparkPreparedDecodePlan | None = None,
    ) -> DSparkPreparedDecodeInputs:
        """Late-bind mutable Host state and stage one decode execution slot."""
        del model
        layout = self._compiled.layout
        if plan is None:
            plan = self._prepare_decode_plan(batch, buffer_slot=buffer_slot)
        elif plan.buffer_slot != buffer_slot:
            raise ValueError(
                f"prepared DSpark slot {plan.buffer_slot} does not match requested slot {buffer_slot}"
            )
        assignment = plan.assignment
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
            max(int(length) - 1, 0)
            for length in self._correct_dspark_seq_lens(batch).tolist()
        ]
        # Speculative rows stage pending drafts into verify rows 1..7 and
        # publish all eight.  A row falls back to the single-anchor contract
        # when its window cannot fit under the position ceiling (clamped
        # duplicate positions must never carry real publication slots) or at
        # a legitimate terminal transition without drafts.
        speculative_flags: list[bool] = []
        pending_drafts: list[list[int]] = []
        if self.speculative:
            for index in range(actual_batch):
                request_id = batch.request_ids[index]
                state = self._drafter_states.get(request_id)
                if state is None:
                    raise RuntimeError(
                        f"DSpark speculation is active but request {request_id!r} has "
                        "no drafter state (seeded before its first decode?)"
                    )
                drafts = state.pending_draft_tokens
                window_fits = anchors[index] + layout.decode_seq <= max_position
                if len(drafts) == DSPARK_DRAFTER_QUERY_WIDTH and window_fits:
                    speculative_flags.append(True)
                    pending_drafts.append(list(drafts))
                else:
                    speculative_flags.append(False)
                    pending_drafts.append([])
                    state.fallback_steps += 1
        else:
            speculative_flags = [False] * actual_batch
            pending_drafts = [[] for _ in range(actual_batch)]
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
        for group in range(layout.partitions):
            for request_index, stream_slot in assignment.active_by_group[group]:
                anchor = anchors[request_index]
                positions = torch.arange(layout.decode_seq, dtype=torch.int64) + anchor
                positions = positions.clamp(max=max_position - 1)
                group_positions[group, stream_slot] = positions
                group_tokens[group, stream_slot, 0] = int(token_rows[request_index])
                if speculative_flags[request_index]:
                    # Verify rows 1..7 carry the pending draft chain; the
                    # target computes all eight rows and the device greedy
                    # sampler emits one sample per logit row.
                    for offset, draft in enumerate(pending_drafts[request_index]):
                        group_tokens[group, stream_slot, 1 + offset] = int(draft)

        # ---- stage per-rank tensors ----
        logit_rows = staged["logit_row_indices"]
        logit_rows.fill_(-1)
        owner_token_counts = staged["num_tokens_per_owner"]
        owner_token_counts.zero_()
        sampled_slots: list[tuple[int, int]] = [(-1, -1)] * actual_batch
        verify_hidden_rows: list[int] = [-1] * actual_batch
        scratch = self._scratch_blocks()

        for group in range(layout.partitions):
            group_plan = plan.group_plans[group]
            ranks = tuple(range(group * layout.tp_size, (group + 1) * layout.tp_size))
            positions = group_positions[group]  # [decode_batch, seq]
            positions_flat = positions.reshape(-1)
            tokens_flat = group_tokens[group].reshape(-1)
            anchor_flags = group_plan.anchor_flags
            starts = positions[:, 0]
            # Position-independent tables were built on the command/prepare lane.
            # Split decode maps raw KV through compact physical ring pages.
            # Fused device prepare still consumes the full absolute-position table.
            ori_block_ids = [
                plan.request_blocks[request_index]["ori"]
                if request_index is not None
                else (scratch["ori"][row],)
                for row, request_index in enumerate(group_plan.request_indices)
            ]
            hca_cmp_tables = group_plan.hca_cmp_tables
            csa_cmp_tables = group_plan.csa_cmp_tables
            idx_tables = group_plan.idx_tables
            hca_state_tables = group_plan.hca_state_tables
            if hca_state_tables is None:
                raise RuntimeError(
                    "DSpark split decode requires Host-lowered HCA state tables"
                )
            # The CSA state rings are addressed by absolute position modulo
            # the 16-token transaction ring, so every page this window writes
            # (anchor..anchor+7) must resolve to its own absolute page id.
            # Building the trailing table at the anchor leaves the pages past
            # anchor//2 on stale ids, so a speculative write at anchor+2 lands
            # where the next step's rebuilt table never reads it back.  Build
            # at the window's end instead: window and recent-history pages all
            # keep their absolute ids, and the anchor-only milestone-1 write
            # resolves to the same slot either way.
            csa_state_tables = torch.stack(
                [
                    builder.trailing_ring_table(
                        plan.request_blocks[group_plan.request_indices[row]]["csa_state"],
                        position=int(starts[row].item()) + layout.decode_seq - 1,
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
                        plan.request_blocks[group_plan.request_indices[row]]["csa_inner_state"],
                        position=int(starts[row].item()) + layout.decode_seq - 1,
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
            # Speculative rows eagerly publish the full eight-row window:
            # every stale position a truncated acceptance leaves behind falls
            # inside the next dispatch's window and is rewritten with correct
            # tokens before any read (reads stay bounded by kv_seq_lens).
            row_flags = [
                speculative_flags[request_index]
                if request_index is not None
                else False
                for request_index in group_plan.request_indices
            ]
            commit = torch.where(
                torch.tensor(row_flags, dtype=torch.bool),
                torch.full((group_batch,), layout.decode_seq, dtype=torch.int64),
                torch.ones((group_batch,), dtype=torch.int64),
            )
            committed_rows = anchor_flags.unsqueeze(-1) & (
                torch.arange(positions.shape[-1]).unsqueeze(0) < commit.unsqueeze(-1)
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
                builder.ring_slot_mapping(
                    positions, ori_block_ids, block_size=layout.block_size
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
            swa_indices, swa_lens = builder.ring_swa_window_indices_and_lens(
                positions, ori_block_ids
            )
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
                # The RoPE tables ride the owner-token T_DYN axis since
                # pypto-lib#1182: stage the rank's own slice of the group
                # stream (the rank's query rows are its contiguous slice).
                staged["freqs_cos"][rank] = group_cos[local_tokens_slice]
                staged["freqs_sin"][rank] = group_sin[local_tokens_slice]
                staged["compressed_freqs_cos"][rank] = compressed_group_cos[
                    local_tokens_slice
                ]
                staged["compressed_freqs_sin"][rank] = compressed_group_sin[
                    local_tokens_slice
                ]
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
                # Logit rows: one anchor entry per active request on this
                # rank; speculative rows enumerate all eight window rows so
                # the device greedy sampler emits one prediction per row.
                # ``entry`` is the packed output row the sampler writes and
                # the readback consumes; ``anchor_entry`` is the hidden-row
                # base the drafter's context scatters from.
                entry = 0
                for local_index in range(local_batch):
                    stream_row = tp_rank * local_batch + local_index
                    if not bool(anchor_flags[stream_row]):
                        continue
                    request_index = group_plan.request_indices[stream_row]
                    assert request_index is not None
                    anchor_entry = local_index * layout.decode_seq
                    width = (
                        layout.decode_seq
                        if speculative_flags[request_index]
                        else 1
                    )
                    for offset in range(width):
                        logit_rows[rank, entry + offset] = anchor_entry + offset
                    sampled_slots[request_index] = (rank, entry)
                    verify_hidden_rows[request_index] = anchor_entry
                    entry += width

        if self.speculative and self._dspark_state_buffers:
            sampled_offsets = torch.full_like(plan.sampled_row_offsets, -1)
            for request_index, (rank, sampled_row) in enumerate(sampled_slots):
                if rank >= 0 and sampled_row >= 0:
                    sampled_offsets[rank, plan.owner_rows[request_index]] = sampled_row
            if not self._compiled.decode_full_fused:
                sampled_row_offsets = self._dspark_state_buffers[
                    buffer_slot
                ].sampled_row_offsets
                if sampled_row_offsets is None:
                    raise RuntimeError("DSpark sampled-row scratch is unavailable")
                copy_shared(
                    sampled_row_offsets,
                    sampled_offsets,
                    name="dspark_sampled_row_offsets",
                )
            if self._compiled.decode_full_fused:
                self._stage_fused_decode_drafter_inputs(
                    request_ids=batch.request_ids,
                    anchors=anchors,
                    owner_ranks=plan.owner_ranks,
                    owner_rows=plan.owner_rows,
                    buffer_slot=buffer_slot,
                )

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
            speculative_flags=tuple(speculative_flags),
            verify_hidden_rows=tuple(verify_hidden_rows),
            owner_ranks=plan.owner_ranks,
            owner_rows=plan.owner_rows,
            buffer_slot=buffer_slot,
        )

    def _stage_fused_decode_drafter_inputs(
        self,
        *,
        request_ids: Sequence[str],
        anchors: Sequence[int],
        owner_ranks: Sequence[int],
        owner_rows: Sequence[int],
        stage_rope: bool = True,
        buffer_slot: int = 0,
    ) -> None:
        """Stage acceptance-independent inputs for the one-L2 decode step."""
        layout = self._compiled.layout
        rows_by_rank: list[list[DSparkDrafterRequestRow]] = [
            [] for _ in range(layout.ranks)
        ]
        for request_id, anchor, rank, local_row in zip(
            request_ids, anchors, owner_ranks, owner_rows, strict=True
        ):
            state = self._drafter_state(request_id)
            if local_row != len(rows_by_rank[rank]):
                raise RuntimeError(
                    "DSpark fused decode requires dense rank-local request rows"
                )
            rows_by_rank[rank].append(
                DSparkDrafterRequestRow(
                    request_id=request_id,
                    group=state.group,
                    lease=state.lease,
                    anchor=int(anchor),
                    valid_count=layout.decode_seq,
                    token_source=0,
                    hidden_row=local_row * layout.decode_seq,
                    decode_mode=True,
                )
            )
        max_rows = max(len(rows) for rows in rows_by_rank)
        batch = next(
            (size for size in DSPARK_DRAFTER_BATCHES if max_rows <= size),
            None,
        )
        if batch is None:
            raise ValueError(
                f"DSpark one-L2 drafter supports at most {DSPARK_DRAFTER_MAX_BATCH} "
                f"live rank-local rows, got {max_rows}"
            )
        self._fused_drafter_batch = batch
        if self._fused_drafter_block_tables:
            self._fused_drafter_batches[buffer_slot] = batch
            block_tables = self._fused_drafter_block_tables[buffer_slot]
            self._drafter_block_tables(rows_by_rank, batch, staging=block_tables)
        else:
            block_tables = self._drafter_block_table_staging
            self._drafter_block_tables(rows_by_rank, batch)

        if not stage_rope:
            return

        rope_candidates = (
            self._fused_drafter_rope_candidates[buffer_slot]
            if self._fused_drafter_rope_candidates
            else self._drafter_rope_candidates
        )
        if not rope_candidates:
            raise RuntimeError(
                "DSpark fused one-L2 builds drafter RoPE candidates on device"
            )

        rope = self._require_rope_tables()
        candidate_rows = DSPARK_DRAFTER_ROPE_CANDIDATE_ROWS
        positions = torch.zeros(
            (layout.ranks, batch, candidate_rows),
            dtype=torch.int64,
        )
        for anchor, rank, local_row in zip(
            anchors, owner_ranks, owner_rows, strict=True
        ):
            positions[rank, local_row] = torch.arange(
                int(anchor), int(anchor) + candidate_rows, dtype=torch.int64
            ).clamp(max=rope.max_position - 1)
        flat_positions = positions.reshape(-1)
        self._packed_host_prefix(
            rope_candidates["rope_cos_candidates"], batch
        ).copy_(
            rope.gather(rope.swa_cos, flat_positions)
            .reshape(*positions.shape, DSPARK_ROPE_HEAD_DIM)
            .to(torch.bfloat16)
        )
        self._packed_host_prefix(
            rope_candidates["rope_sin_candidates"], batch
        ).copy_(
            rope.gather(rope.swa_sin, flat_positions)
            .reshape(*positions.shape, DSPARK_ROPE_HEAD_DIM)
            .to(torch.bfloat16)
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
                # Immutable prefix pages may be shared across requests, but
                # two logical pages of the same request must never alias.
                allocated = [block_id for block_id in blocks if block_id >= 0]
                if len(allocated) != len(set(allocated)):
                    raise ValueError("a grouped KV row must not repeat physical blocks")
                minimum = -1 if name in ("ori", "hca_state", "csa_state", "csa_inner_state") else 0
                if any(
                    block_id < minimum or block_id >= self._cache_group_num_blocks[name]
                    for block_id in blocks
                ):
                    raise ValueError(
                        f"grouped KV block IDs for {name} must be in "
                        f"[{minimum}, {self._cache_group_num_blocks[name]}); "
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

    # ------------------------------------------------------------------
    # speculative drafter: leases, block rings, and staging (milestone 2)
    # ------------------------------------------------------------------
    def _reserve_drafter_state(
        self, request_id: str, *, group: int, prompt_len: int
    ) -> _DSparkDraftRequestState:
        """Take a stable group-local lease for one newly prefilled request.

        The lease is independent of the request's current compute rank or dense
        batch row: ``_decode_assignment`` recomputes those every dispatch, so
        keying drafter storage to them would corrupt a surviving request
        whenever admission or removal reshuffles a batch.
        """
        existing = self._drafter_states.get(request_id)
        if existing is not None:
            raise RuntimeError(
                f"DSpark drafter state already exists for request {request_id!r}"
            )
        free = self._drafter_free_leases.get(group)
        if not free:
            raise RuntimeError(
                f"DSpark drafter leases exhausted for TP group {group} "
                f"({DSPARK_DRAFTER_LEASES_PER_GROUP} live requests per group)"
            )
        lease = free.pop()
        self._drafter_lease_generations[group][lease] += 1
        state = _DSparkDraftRequestState(
            group=group,
            lease=lease,
            generation=self._drafter_lease_generations[group][lease],
            prompt_len=prompt_len,
        )
        self._drafter_states[request_id] = state
        return state

    def _drafter_state(self, request_id: str) -> _DSparkDraftRequestState:
        state = self._drafter_states.get(request_id)
        if state is None:
            raise KeyError(f"DSpark drafter state is missing for request {request_id!r}")
        return state

    def _drafter_ring_rows(self, base_block: int) -> torch.Tensor:
        """Rotated ring block ids ``[DSPARK_DRAFT_LAYERS, TABLE_BLOCKS]``.

        Each logical block maps to ``base + (L + 7*layer) % RING`` so the three
        draft layers rotate over the same six-block private range and a
        128-deep window plus the seven query rows never aliases itself.
        """
        logical = torch.arange(DSPARK_DRAFTER_TABLE_BLOCKS, dtype=torch.int64)
        rows = torch.empty(
            (DSPARK_DRAFT_LAYERS, DSPARK_DRAFTER_TABLE_BLOCKS), dtype=torch.int64
        )
        for layer in range(DSPARK_DRAFT_LAYERS):
            rows[layer] = base_block + (
                logical + 7 * layer
            ) % DSPARK_DRAFTER_RING_BLOCKS
        return rows

    def _drafter_block_tables(
        self,
        rows_by_rank: list[list[DSparkDrafterRequestRow]],
        batch: int,
        *,
        staging: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Stage dense lease rings into the batch's shared block-table buffer."""
        tables = (self._drafter_block_table_staging if staging is None else staging).get(batch)
        if tables is None:
            raise RuntimeError(
                f"DSpark drafter block-table staging for batch {batch} is not allocated"
            )
        filler = self._drafter_ring_rows(DSPARK_DRAFTER_FILLER_BLOCK_BASE)
        for rank, rows in enumerate(rows_by_rank):
            for index in range(batch):
                if index < len(rows):
                    base = rows[index].lease * DSPARK_DRAFTER_RING_BLOCKS
                    ring = self._drafter_ring_rows(base)
                else:
                    ring = filler
                tables[rank, :, index] = ring.to(torch.int32)

    @staticmethod
    def _drafter_slots_for_positions(
        ring_rows: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Map absolute positions through one lease ring; ``-1`` where unmapped."""
        valid = positions >= 0
        safe = positions.clamp(min=0)
        logical = safe // DSPARK_BLOCK_SIZE
        index = (
            logical.clamp(max=DSPARK_DRAFTER_TABLE_BLOCKS - 1)
            .unsqueeze(0)
            .expand(ring_rows.shape[0], -1)
        )
        block = ring_rows.gather(1, index)
        slot = block * DSPARK_BLOCK_SIZE + safe % DSPARK_BLOCK_SIZE
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def _prepare_drafter_inputs(
        self,
        rows_by_rank: list[list[DSparkDrafterRequestRow]],
        *,
        hidden: torch.Tensor | None,
        context_rows: int,
        seed_contexts: dict[int, tuple[int, torch.Tensor]] | None = None,
    ) -> tuple[int, int]:
        """Stage one world-wide drafter (+ markov) dispatch.

        ``rows_by_rank`` holds each rank's dense real rows; ``hidden`` is the
        already-read ``[ranks, context_rows, MAIN_HIDDEN_DIM]`` backbone tap
        (zero rows for padding and fillers); every rank's real row count must
        stay within the uniform padded batch.  Context rows end at each
        request's anchor; the group assembly replicates every group row's slot
        on all four ranks of the group (the drafter's KV-replica contract).
        Returns ``(batch, context_rows)`` for the dispatch-args slicing.
        """
        if self._drafter_task_args is None or self._markov_task_args is None:
            raise RuntimeError("DSpark drafter TaskArgs are not staged")
        layout = self._compiled.layout
        ranks = layout.ranks
        tp = layout.tp_size
        real_counts = [len(rows) for rows in rows_by_rank]
        batch = next(
            (size for size in DSPARK_DRAFTER_BATCHES if max(real_counts) <= size),
            None,
        )
        if batch is None or max(real_counts) > DSPARK_DRAFTER_MAX_BATCH:
            raise ValueError(
                f"DSpark drafter batch must be one of {DSPARK_DRAFTER_BATCHES}, "
                f"got up to {max(real_counts)} real rows"
            )
        # The group-context tensors dispatch through fixed shared extents, so
        # callers stage a bucket extent and pad padding rows with position
        # zero / -1 slots themselves.
        if context_rows not in DSPARK_DRAFTER_CONTEXT_BUCKETS:
            raise ValueError(
                f"DSpark drafter context rows {context_rows} must be one of "
                f"{DSPARK_DRAFTER_CONTEXT_BUCKETS}"
            )
        context_staging = self._drafter_context_staging[context_rows]
        group_context = tp * context_rows
        query_rows = DSPARK_DRAFTER_MAX_BATCH * DSPARK_DRAFTER_QUERY_WIDTH
        group_query = tp * query_rows
        rope = self._require_rope_tables()
        max_position = rope.max_position

        task_args = self._drafter_task_args
        tensors = task_args.tensors
        # Dim-1 dynamic names dispatch as packed prefixes (contiguous-from-
        # front), so every stage and readback must go through the same packed
        # view: writing the max-sized slot directly would land at rank stride
        # and the dispatch would read another rank's rows.
        selectors = {
            name: self._packed_host_prefix(tensors[name], batch)
            for name in (
                "num_sampled",
                "last_sampled",
                "next_prefill_tokens",
                "anchor_positions",
            )
        }
        for view in selectors.values():
            view.zero_()
        if hidden is not None:
            self._packed_host_prefix(tensors["target_hidden"], context_rows).copy_(hidden)
        self._drafter_block_tables(rows_by_rank, batch)
        tensors["query_group_position_ids"].zero_()
        tensors["query_group_slot_mapping"].fill_(-1)
        context_staging["context_group_position_ids"].zero_()
        context_staging["context_group_slot_mapping"].fill_(-1)

        # Per-rank local staging, then rank-major group assembly.  ``local``
        # arrays are indexed [rank, ...]; group arrays concatenate the four
        # CP ranks of each group so every rank carries the group's rows.
        context_positions_local = torch.zeros(
            (ranks, context_rows), dtype=torch.int64
        )
        context_valid_local = torch.zeros(
            (ranks, context_rows, DSPARK_DRAFT_LAYERS), dtype=torch.bool
        )
        context_slots_local = torch.full(
            (ranks, DSPARK_DRAFT_LAYERS, context_rows), -1, dtype=torch.int64
        )
        query_positions_local = torch.zeros((ranks, query_rows), dtype=torch.int64)
        query_valid_local = torch.zeros(
            (ranks, query_rows, DSPARK_DRAFT_LAYERS), dtype=torch.bool
        )
        # Prefill seeding stages the prompt tail as the group context: every
        # rank of the group carries its rank-major band of the tail (padded
        # with -1 positions to the uniform extent), and the seed request's
        # batch row (on its group leader) selects its token through
        # ``next_prefill_tokens`` instead of ``last_sampled``.
        if seed_contexts:
            for group, (lease, tail_positions) in seed_contexts.items():
                ring = self._drafter_ring_rows(lease * DSPARK_DRAFTER_RING_BLOCKS)
                for member in range(tp):
                    rank = group * tp + member
                    start = member * context_rows
                    positions = tail_positions[start : start + context_rows]
                    rows_here = int((positions >= 0).sum())
                    if rows_here:
                        context_positions_local[rank, :rows_here] = positions[:rows_here]
                        context_valid_local[rank, :rows_here, :] = True
                        context_slots_local[rank, :, :rows_here] = (
                            self._drafter_slots_for_positions(ring, positions[:rows_here])
                        )
        for rank, rows in enumerate(rows_by_rank):
            for index, row in enumerate(rows):
                selectors["num_sampled"][rank, index] = 1 if row.decode_mode else 0
                if row.decode_mode:
                    selectors["last_sampled"][rank, index] = row.token_source
                else:
                    selectors["next_prefill_tokens"][rank, index] = row.token_source
                selectors["anchor_positions"][rank, index] = row.anchor
                base = row.lease * DSPARK_DRAFTER_RING_BLOCKS
                ring = self._drafter_ring_rows(base)
                # Context: the request's ``valid_count`` committed rows ending
                # at the anchor, at dense row offsets index*DECODE_SEQ.
                start = index * layout.decode_seq
                for offset in range(row.valid_count):
                    position = row.anchor - row.valid_count + 1 + offset
                    if position < 0 or position >= max_position:
                        raise ValueError(
                            f"DSpark drafter context position {position} outside "
                            f"[0, {max_position}) for request {row.request_id!r}"
                        )
                    context_positions_local[rank, start + offset] = position
                    context_valid_local[rank, start + offset, :] = True
                # Query: seven fresh positions after the anchor.
                for offset in range(DSPARK_DRAFTER_QUERY_WIDTH):
                    position = row.anchor + 1 + offset
                    if position >= max_position:
                        raise ValueError(
                            f"DSpark drafter query position {position} exceeds the "
                            f"rope table for request {row.request_id!r}"
                        )
                    token = index * DSPARK_DRAFTER_QUERY_WIDTH + offset
                    query_positions_local[rank, token] = position
                    query_valid_local[rank, token, :] = True
                # Slot mappings are computed per rank from the OWNER's lease
                # ring; the group assembly below replicates them so all four
                # ranks of the group write their pool replicas.
        query_slots_local = torch.full(
            (ranks, DSPARK_DRAFT_LAYERS, query_rows), -1, dtype=torch.int64
        )
        for rank, rows in enumerate(rows_by_rank):
            for index, row in enumerate(rows):
                ring = self._drafter_ring_rows(row.lease * DSPARK_DRAFTER_RING_BLOCKS)
                start = index * layout.decode_seq
                for offset in range(row.valid_count):
                    local_row = start + offset
                    if context_valid_local[rank, local_row, 0]:
                        slots = self._drafter_slots_for_positions(
                            ring,
                            context_positions_local[rank, local_row : local_row + 1],
                        )
                        context_slots_local[rank, :, local_row] = slots.reshape(-1)
                for offset in range(DSPARK_DRAFTER_QUERY_WIDTH):
                    token = index * DSPARK_DRAFTER_QUERY_WIDTH + offset
                    if query_valid_local[rank, token, 0]:
                        slots = self._drafter_slots_for_positions(
                            ring,
                            query_positions_local[rank, token : token + 1],
                        )
                        query_slots_local[rank, :, token] = slots.reshape(-1)

        for rank in range(ranks):
            group_base = rank // tp * tp
            group_slice = slice(group_base, group_base + tp)
            group_positions = (
                context_positions_local[group_slice].reshape(-1).to(torch.int64)
            )
            group_slots = (
                context_slots_local[group_slice]
                .permute(1, 0, 2)
                .reshape(DSPARK_DRAFT_LAYERS, -1)
            )
            context_staging["context_group_position_ids"][rank, :group_context] = (
                group_positions.to(torch.int32)
            )
            tensors["query_group_position_ids"][rank] = (
                query_positions_local[group_slice].reshape(-1).to(torch.int32)
            )
            # Group slots keep the layer axis first: [layers, 4 * rows].
            context_staging["context_group_slot_mapping"][rank, :, :group_context] = (
                group_slots
            )
            tensors["query_group_slot_mapping"][rank] = (
                query_slots_local[group_slice]
                .permute(1, 0, 2)
                .reshape(DSPARK_DRAFT_LAYERS, -1)
            )
            gather_positions = torch.where(
                group_slots[0] >= 0, group_positions, torch.zeros_like(group_positions)
            )
            context_staging["context_group_freqs_cos"][rank, :group_context] = (
                rope.gather(rope.swa_cos, gather_positions).to(torch.bfloat16)
            )
            context_staging["context_group_freqs_sin"][rank, :group_context] = (
                rope.gather(rope.swa_sin, gather_positions).to(torch.bfloat16)
            )
            local_query = query_positions_local[rank]
            local_query_mask = query_valid_local[rank, :, 0]
            gather_query = torch.where(
                local_query_mask, local_query, torch.zeros_like(local_query)
            )
            tensors["query_freqs_cos"][rank] = rope.gather(
                rope.swa_cos, gather_query
            ).to(torch.bfloat16)
            tensors["query_freqs_sin"][rank] = rope.gather(
                rope.swa_sin, gather_query
            ).to(torch.bfloat16)
            group_query_positions = tensors["query_group_position_ids"][rank].to(
                torch.int64
            )
            group_query_mask = query_valid_local[group_slice][:, :, 0].reshape(-1)
            gather_group_query = torch.where(
                group_query_mask, group_query_positions, torch.zeros_like(group_query_positions)
            )
            tensors["query_group_freqs_cos"][rank] = rope.gather(
                rope.swa_cos, gather_group_query
            ).to(torch.bfloat16)
            tensors["query_group_freqs_sin"][rank] = rope.gather(
                rope.swa_sin, gather_group_query
            ).to(torch.bfloat16)

        # Markov consumes the same request-state tensors plus its own logit
        # rows: one row per (request, step) over the dense padded batch.  Its
        # selectors also dispatch packed, so stage through the packed views.
        markov_tensors = self._markov_task_args.tensors
        for name in ("num_sampled", "last_sampled", "next_prefill_tokens"):
            self._packed_host_prefix(markov_tensors[name], batch).copy_(selectors[name])
        markov_tensors["logit_row_indices"].fill_(-1)
        for rank, rows in enumerate(rows_by_rank):
            real = len(rows)
            if real:
                markov_tensors["logit_row_indices"][rank, : real * DSPARK_DRAFTER_QUERY_WIDTH] = (
                    torch.arange(real * DSPARK_DRAFTER_QUERY_WIDTH, dtype=torch.int32)
                )
        return batch, context_rows

    def _capture_prefill_tails(self, batch: PrefillBatch, inputs) -> None:
        """Roll this chunk's backbone tap rows into each request's seed tail.

        The tap is read back immediately, before the next prefill dispatch can
        reuse the scratch; only rows inside the chunk's logical extent join
        the tail (synthetic padding positions never become drafter context).
        """
        layout = self._compiled.layout
        tp = layout.tp_size
        # The tap is a device scratch, not a host slot; the materializer's
        # (scope, name) cache returns the identical buffer the dispatch used.
        device = self._alloc_zeroed_stacked_tensor(
            "dspark_target_hidden",
            (layout.ranks, layout.prefill_local_tokens, DSPARK_MAIN_HIDDEN_DIM),
            torch.bfloat16,
            scope="prefill",
        )
        worker = self._shared_l3_worker()
        local_tokens = inputs.physical_tokens // tp
        row_bytes = DSPARK_MAIN_HIDDEN_DIM * 2
        group_rows = {}
        for group in set(inputs.groups):
            rows = torch.empty(
                (tp, local_tokens, DSPARK_MAIN_HIDDEN_DIM), dtype=torch.bfloat16
            )
            for member in range(tp):
                rank = group * tp + member
                worker.copy_from(
                    rows[member].data_ptr(),
                    device.shards[rank].data_ptr,
                    local_tokens * row_bytes,
                    worker_id=device.worker_ids[rank],
                )
            group_rows[group] = rows
        for index, (group, request_id) in enumerate(
            zip(inputs.groups, batch.request_ids, strict=True)
        ):
            state = self._drafter_states.get(request_id)
            if state is None:
                state = self._reserve_drafter_state(request_id, group=group, prompt_len=0)
            actual = int(inputs.actual_tokens[index])
            chunk_start = int(inputs.chunk_starts[index])
            # Rank-major logical order: each rank owns a contiguous band of
            # the packed group, which may contain several request boundaries.
            chunk_rows = self._prefill_chunk_bands(
                group_rows[group], local_tokens, actual, inputs.packed_offsets[index]
            )
            if chunk_rows is None:
                continue
            self._append_prefill_tail(state, chunk_rows, chunk_start)

    @staticmethod
    def _prefill_chunk_bands(
        rows: torch.Tensor, local_tokens: int, actual: int, packed_offset: int = 0
    ) -> torch.Tensor | None:
        """Extract one request's packed interval across rank bands."""
        if actual <= 0:
            return None
        packed = rows[:, :local_tokens].reshape(-1, rows.shape[-1])
        return packed[packed_offset:packed_offset + actual].clone()

    @staticmethod
    def _append_prefill_tail(
        state: _DSparkDraftRequestState, chunk_rows: torch.Tensor, chunk_start: int
    ) -> None:
        """Append one chunk's rows and keep a window-deep tail."""
        chunk_positions = torch.arange(
            chunk_start, chunk_start + int(chunk_rows.shape[0]), dtype=torch.int64
        )
        if state.prefill_tail_rows is None:
            state.prefill_tail_rows = chunk_rows
            state.prefill_tail_positions = chunk_positions
        else:
            state.prefill_tail_rows = torch.cat([state.prefill_tail_rows, chunk_rows], dim=0)
            state.prefill_tail_positions = torch.cat(
                [state.prefill_tail_positions, chunk_positions], dim=0
            )
        if state.prefill_tail_rows.shape[0] > DSPARK_SLIDING_WINDOW:
            state.prefill_tail_rows = (
                state.prefill_tail_rows[-DSPARK_SLIDING_WINDOW:].clone().contiguous()
            )
            state.prefill_tail_positions = (
                state.prefill_tail_positions[-DSPARK_SLIDING_WINDOW:]
                .clone()
                .contiguous()
            )

    def finalize_prefill(
        self,
        request_ids: Sequence[str],
        sampled_token_ids: Sequence[int],
        sampling_params: Sequence[SamplingParams] | None = None,
    ) -> None:
        """Seed the first draft chain for each terminal-prefill request.

        Called by the worker with exactly the completed subset, after the
        terminal chunk sampled its first generated token.  The prompt tail
        captured across chunks becomes the group context; the sampled token
        becomes ``next_prefill_tokens`` (the query row-0 token and the anchor
        of the first target verify).
        """
        if not self.speculative:
            return
        del sampling_params  # greedy-only serving; nothing to select
        if len(request_ids) != len(sampled_token_ids):
            raise ValueError("DSpark seeding requires one sampled token per request")
        # The drafter seed ABI has one context/lease per group. Pack independent
        # groups together, then seed further requests in that group in later waves.
        waves: list[list[tuple[str, int]]] = []
        group_counts: dict[int, int] = {}
        for request_id, token in zip(request_ids, sampled_token_ids, strict=True):
            group = self._drafter_state(request_id).group
            wave = group_counts.get(group, 0)
            group_counts[group] = wave + 1
            if wave == len(waves):
                waves.append([])
            waves[wave].append((request_id, int(token)))
        for wave in waves:
            self._seed_prefill_wave(
                [request_id for request_id, _ in wave], [token for _, token in wave]
            )

    def _seed_prefill_wave(
        self, request_ids: Sequence[str], sampled_token_ids: Sequence[int]
    ) -> None:
        """Seed at most one request per TP group using independent prompt tails."""
        layout = self._compiled.layout
        tp = layout.tp_size
        rows_by_rank: list[list[DSparkDrafterRequestRow]] = [[] for _ in range(layout.ranks)]
        seed_contexts: dict[int, tuple[int, torch.Tensor]] = {}
        tails: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        max_position = self._require_rope_tables().max_position
        for request_id, token in zip(request_ids, sampled_token_ids, strict=True):
            state = self._drafter_state(request_id)
            state.current_token_id = int(token)
            if state.prefill_tail_rows is None or state.prefill_tail_positions is None:
                raise RuntimeError(
                    f"DSpark seeding requires a captured prompt tail for {request_id!r}"
                )
            anchor = int(state.prefill_tail_positions[-1].item())
            expected_positions = torch.arange(
                max(0, anchor + 1 - DSPARK_SLIDING_WINDOW), anchor + 1,
                dtype=torch.int64,
            )
            if not torch.equal(state.prefill_tail_positions, expected_positions):
                raise RuntimeError(
                    f"DSpark seeding requires the complete prompt tail for {request_id!r}; "
                    "prefix-cache hits must replay the drafter sliding window"
                )
            state.prompt_len = anchor + 1
            state.committed_count = state.prompt_len
            if anchor + DSPARK_DRAFTER_QUERY_WIDTH >= max_position:
                # No room for a draft chain under the ceiling: seed nothing
                # and let the first verify fall back to the anchor-only path.
                state.pending_draft_tokens = []
                state.pending_confidence = []
                continue
            rows_by_rank[state.group * tp].append(
                DSparkDrafterRequestRow(
                    request_id=request_id,
                    group=state.group,
                    lease=state.lease,
                    anchor=anchor,
                    valid_count=0,
                    token_source=int(token),
                    hidden_row=0,
                    decode_mode=False,
                )
            )
            seed_contexts[state.group] = (state.lease, state.prefill_tail_positions)
            tails[state.group] = (state.prefill_tail_rows, state.prefill_tail_positions)
        if not seed_contexts:
            for request_id in request_ids:
                self._initialize_dspark_device_state(self._drafter_state(request_id))
            return
        context_rows = max(
            -(-int(positions.shape[0]) // tp) for _, positions in seed_contexts.values()
        )
        # Group-context buffers dispatch at fixed extents: round the seed's
        # tail up to the next shared bucket.
        context_rows = next(
            size for size in DSPARK_DRAFTER_CONTEXT_BUCKETS if context_rows <= size
        )
        hidden = torch.zeros(
            (layout.ranks, context_rows, DSPARK_MAIN_HIDDEN_DIM), dtype=torch.bfloat16
        )
        for group, (tail_rows, tail_positions) in tails.items():
            total = int(tail_positions.shape[0])
            padded = context_rows * tp
            positions = torch.full((padded,), -1, dtype=torch.int64)
            positions[:total] = tail_positions
            row_buffer = torch.zeros(
                (padded, DSPARK_MAIN_HIDDEN_DIM), dtype=torch.bfloat16
            )
            row_buffer[:total] = tail_rows
            seed_contexts[group] = (seed_contexts[group][0], positions)
            for member in range(tp):
                rank = group * tp + member
                hidden[rank] = row_buffer[
                    member * context_rows : (member + 1) * context_rows
                ]
        batch, _ = self._prepare_drafter_inputs(
            rows_by_rank,
            hidden=hidden,
            context_rows=context_rows,
            seed_contexts=seed_contexts,
        )
        self._run_drafter_and_markov(batch, context_rows)
        drafts = self._packed_host_prefix(
            self._markov_task_args.tensors["draft_token_ids"], batch
        )
        confidence = self._packed_host_prefix(
            self._markov_task_args.tensors["confidence_probs"], batch
        )
        for request_id in request_ids:
            state = self._drafter_state(request_id)
            leader = state.group * tp
            row_of = next(
                (
                    i
                    for i, row in enumerate(rows_by_rank[leader])
                    if row.request_id == request_id
                ),
                None,
            )
            if row_of is None:
                continue  # capacity-skipped above; decode falls back
            state.pending_draft_tokens = [int(v) for v in drafts[leader, row_of]]
            state.pending_confidence = [float(v) for v in confidence[leader, row_of]]
            if len(state.pending_draft_tokens) != DSPARK_DRAFTER_QUERY_WIDTH:
                raise RuntimeError(
                    f"DSpark seeding produced {len(state.pending_draft_tokens)} drafts "
                    f"for {request_id!r}"
                )
            state.proposed_tokens += DSPARK_DRAFTER_QUERY_WIDTH
            self._initialize_dspark_device_state(state)
        # Requests too close to the position ceiling carry draft_count=0 but
        # still need a valid device slot for their anchor-only fallback.
        for request_id in request_ids:
            self._initialize_dspark_device_state(self._drafter_state(request_id))
    def _initialize_dspark_device_state(self, state: _DSparkDraftRequestState) -> None:
        """Publish one complete request slot to every rank in its TP group."""
        if state.device_state_initialized:
            return
        token_row, meta_row = self._build_dspark_device_state_rows(state)
        worker = self._shared_l3_worker()
        token_state = self._materialize_dspark_device_state_tokens()
        meta_state = self._materialize_dspark_device_state_meta()
        group_base = state.group * self._compiled.layout.tp_size
        for rank in range(group_base, group_base + self._compiled.layout.tp_size):
            for device_tensor, source in ((token_state, token_row), (meta_state, meta_row)):
                row_nbytes = source.numel() * source.element_size()
                worker.copy_to(
                    device_tensor.shards[rank].data_ptr,
                    source.data_ptr(),
                    row_nbytes,
                    dst_offset=state.lease * row_nbytes,
                    worker_id=device_tensor.worker_ids[rank],
                )
        state.device_state_initialized = True

    def _build_dspark_device_state_rows(
        self,
        state: _DSparkDraftRequestState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one Host seed into the persistent device-state ABI."""
        if state.current_token_id is None or state.prompt_len <= 0:
            raise RuntimeError("DSpark device state cannot be initialized from partial state")
        token_row = torch.full(
            (_DSPARK_STATE_TOKEN_WIDTH,),
            DSPARK_NOISE_TOKEN_ID,
            dtype=torch.long,
        )
        token_row[0] = state.current_token_id
        if state.pending_draft_tokens:
            if len(state.pending_draft_tokens) != DSPARK_DRAFTER_QUERY_WIDTH:
                raise RuntimeError("DSpark device state requires either zero or seven drafts")
            token_row[1:] = torch.tensor(state.pending_draft_tokens, dtype=torch.long)
        meta_row = torch.zeros((_DSPARK_STATE_META_WIDTH,), dtype=torch.int32)
        meta_row[_DSPARK_STATE_VALID] = 1
        meta_row[_DSPARK_STATE_GENERATION] = state.generation
        # The terminal-prefill sampled token is the first target input after
        # the prompt, so its zero-based position equals prompt_len.
        meta_row[_DSPARK_STATE_ANCHOR_POSITION] = state.prompt_len
        meta_row[_DSPARK_STATE_COMMITTED_COUNT] = state.committed_count
        meta_row[_DSPARK_STATE_DRAFT_COUNT] = len(state.pending_draft_tokens)
        meta_row[_DSPARK_STATE_POSITION_LIMIT] = self._require_rope_tables().max_position
        return token_row, meta_row

    def _materialize_dspark_device_state_tokens(self) -> StackedDeviceTensor:
        """Allocate replicated current-token and K=7 draft state."""
        state = self._dspark_device_state_tokens
        if state is None:
            state = self._alloc_empty_stacked_tensor(
                (
                    self._compiled.layout.ranks,
                    DSPARK_DRAFTER_LEASES_PER_GROUP,
                    _DSPARK_STATE_TOKEN_WIDTH,
                ),
                torch.long,
            )
            self._dspark_device_state_tokens = state
        return state

    def _materialize_dspark_device_state_meta(self) -> StackedDeviceTensor:
        """Allocate replicated generation, position, and accepted-length state."""
        state = self._dspark_device_state_meta
        if state is None:
            state = self._alloc_empty_stacked_tensor(
                (
                    self._compiled.layout.ranks,
                    DSPARK_DRAFTER_LEASES_PER_GROUP,
                    _DSPARK_STATE_META_WIDTH,
                ),
                torch.int32,
            )
            self._dspark_device_state_meta = state
        return state

    def _run_drafter_and_markov(
        self,
        batch: int,
        context_rows: int,
        *,
        target_hidden: StackedDeviceTensor | None = None,
        buffer_slot: int | None = None,
        rows_by_rank: list[list[DSparkDrafterRequestRow]] | None = None,
    ) -> None:
        """Dispatch the staged drafter + markov pair under their profiles."""
        if self._compiled.drafter is None:
            raise RuntimeError("DSpark speculation requires the drafter program")
        drafter_args = self._drafter_dispatch_args(
            batch, context_rows, target_hidden=target_hidden
        )
        if self._compiled.draft_device_state_fused:
            markov_args = self._markov_dispatch_args(batch)
            # head_hidden and the three request selectors are already part of
            # the drafter ABI; the fused L3 consumes the remaining Markov ABI.
            markov_tail = tuple(
                arg
                for name, arg in zip(
                    self._markov_task_args.names, markov_args, strict=True
                )
                if name
                not in {
                    "head_hidden",
                    "num_sampled",
                    "last_sampled",
                    "next_prefill_tokens",
                }
            )
            state_buffer_slot = 0 if buffer_slot is None else buffer_slot
            state_buffers = self._dspark_state_buffers[state_buffer_slot]
            if rows_by_rank is None:
                # Prefill seeding publishes the real persistent state with a
                # direct Host copy after reading the first draft. Keep every
                # commit argument on private scratch so bootstrap cannot race
                # with an early-prepared decode slot's descriptors or state.
                state_slot_ids = self._alloc_zeroed_stacked_tensor(
                    "dspark_seed_state_slot_ids",
                    (
                        self._compiled.layout.ranks,
                        self._compiled.layout.decode_local_batch,
                    ),
                    torch.int32,
                    scope="drafter",
                )
                state_generations = self._alloc_zeroed_stacked_tensor(
                    "dspark_seed_state_generations",
                    (
                        self._compiled.layout.ranks,
                        self._compiled.layout.decode_local_batch,
                    ),
                    torch.int32,
                    scope="drafter",
                )
                state_tokens = self._alloc_zeroed_stacked_tensor(
                    "dspark_seed_state_tokens",
                    (
                        self._compiled.layout.ranks,
                        DSPARK_DRAFTER_LEASES_PER_GROUP,
                        _DSPARK_STATE_TOKEN_WIDTH,
                    ),
                    torch.long,
                    scope="drafter",
                )
                state_meta = self._alloc_zeroed_stacked_tensor(
                    "dspark_seed_state_meta",
                    (
                        self._compiled.layout.ranks,
                        DSPARK_DRAFTER_LEASES_PER_GROUP,
                        _DSPARK_STATE_META_WIDTH,
                    ),
                    torch.int32,
                    scope="drafter",
                )
            else:
                self._stage_dspark_draft_commit(state_buffers, rows_by_rank)
                state_slot_ids = state_buffers.state_slot_ids
                state_generations = state_buffers.state_generations
                state_tokens = self._materialize_dspark_device_state_tokens()
                state_meta = self._materialize_dspark_device_state_meta()
            state_args = (
                state_slot_ids,
                state_generations,
                state_tokens,
                state_meta,
            )
            self._run_l3(
                self._compiled.drafter,
                *drafter_args,
                *markov_tail,
                *state_args,
                config=self._drafter_run_config,
            )
            return
        if self._compiled.markov is None:
            raise RuntimeError("DSpark speculation requires the markov program")
        self._run_l3(self._compiled.drafter, *drafter_args, config=self._drafter_run_config)
        markov_args = self._markov_dispatch_args(batch)
        self._run_l3(self._compiled.markov, *markov_args, config=self._markov_run_config)

    def _stage_dspark_draft_commit(
        self,
        state_buffers: _DSparkDeviceStateBuffers,
        rows_by_rank: list[list[DSparkDrafterRequestRow]],
    ) -> None:
        """Stage dense drafter rows as persistent-state commit selectors."""
        commit_slots = torch.full_like(state_buffers.state_slot_ids, -1)
        commit_generations = torch.full_like(state_buffers.state_generations, -1)
        for rank, rows in enumerate(rows_by_rank):
            for index, row in enumerate(rows):
                state = self._drafter_state(row.request_id)
                commit_slots[rank, index] = state.lease
                commit_generations[rank, index] = state.generation
        copy_shared(
            state_buffers.state_slot_ids,
            commit_slots,
            name="dspark_commit_state_slot_ids",
        )
        copy_shared(
            state_buffers.state_generations,
            commit_generations,
            name="dspark_commit_state_generations",
        )

    def _commit_dspark_drafts(
        self,
        *,
        buffer_slot: int,
        batch: int,
        rows_by_rank: list[list[DSparkDrafterRequestRow]],
    ) -> None:
        """Commit Markov's K=7 output into persistent device request state."""
        program = self._compiled.state_commit
        if program is None:
            raise RuntimeError("DSpark device-state draft commit program is not available")
        state_buffers = self._dspark_state_buffers[buffer_slot]
        packed = self._packed_host_prefix(
            self._markov_task_args.tensors["draft_token_ids"], batch
        )
        self._stage_dspark_draft_commit(state_buffers, rows_by_rank)
        draft_token_ids = state_buffers.draft_token_ids
        if draft_token_ids is None:
            raise RuntimeError("DSpark standalone draft-commit scratch is unavailable")
        draft_token_ids.zero_()
        draft_token_ids[:, :batch].copy_(packed)
        with profile_span("DSparkModelRunner.decode.commit_drafts_device", cat="executor"):
            self._run_l3(
                program,
                state_buffers.state_slot_ids,
                state_buffers.state_generations,
                self._materialize_dspark_device_state_tokens(),
                self._materialize_dspark_device_state_meta(),
                draft_token_ids,
                config=self._markov_run_config,
            )

    def _drafter_dispatch_args(
        self,
        batch: int,
        context_rows: int,
        *,
        target_hidden: StackedDeviceTensor | None = None,
        task_args: TaskArgs | None = None,
        block_table_staging: dict[int, torch.Tensor] | None = None,
    ) -> tuple[Any, ...]:
        """Bind the drafter's dynamic extents over the staged slots."""
        from pypto_serving.model.deepseek_dspark.task_args import (  # noqa: PLC0415
            _DRAFTER_B_DYNAMIC_NAMES,
            _DRAFTER_T_MAIN_DYNAMIC_NAMES,
        )

        task_args = self._drafter_task_args if task_args is None else task_args
        if task_args is None:
            raise RuntimeError("DSpark drafter TaskArgs are not staged")
        block_table_staging = (
            self._drafter_block_table_staging
            if block_table_staging is None
            else block_table_staging
        )
        context_staging = self._drafter_context_staging[context_rows]
        bounded: list[Any] = []
        for name, arg in zip(task_args.names, task_args.build(), strict=True):
            if name == "target_hidden" and target_hidden is not None:
                bounded.append(self._stacked_device_prefix(target_hidden, context_rows))
            elif name == "block_tables":
                bounded.append(block_table_staging[batch])
            elif name in context_staging:
                bounded.append(context_staging[name])
            elif name in _DRAFTER_B_DYNAMIC_NAMES:
                bounded.append(
                    self._packed_host_prefix(arg, batch)
                    if isinstance(arg, torch.Tensor)
                    else self._stacked_device_prefix(arg, batch)
                )
            elif name in _DRAFTER_T_MAIN_DYNAMIC_NAMES:
                bounded.append(
                    self._packed_host_prefix(arg, context_rows)
                    if isinstance(arg, torch.Tensor)
                    else self._stacked_device_prefix(arg, context_rows)
                )
            else:
                bounded.append(arg)
        return tuple(bounded)

    def _markov_dispatch_args(
        self,
        batch: int,
        *,
        task_args: TaskArgs | None = None,
    ) -> tuple[Any, ...]:
        """Bind the markov sampler's B_DYN extent over the staged slots."""
        task_args = self._markov_task_args if task_args is None else task_args
        if task_args is None:
            raise RuntimeError("DSpark Markov TaskArgs are not staged")
        bounded: list[Any] = []
        for name, arg in zip(task_args.names, task_args.build(), strict=True):
            if name in ("num_sampled", "last_sampled", "next_prefill_tokens"):
                bounded.append(self._packed_host_prefix(arg, batch))
            elif name == "head_hidden":
                bounded.append(
                    self._stacked_device_prefix(arg, batch)
                    if not isinstance(arg, torch.Tensor)
                    else self._packed_host_prefix(arg, batch)
                )
            elif name in ("draft_token_ids", "confidence_probs"):
                bounded.append(self._packed_host_prefix(arg, batch))
            else:
                bounded.append(arg)
        return tuple(bounded)

    def _read_drafter_hidden(
        self, host_mirror: torch.Tensor, *, rows: int
    ) -> torch.Tensor:
        """D2H readback of the decode tap's first ``rows`` rows per rank."""
        device = self._alloc_zeroed_stacked_tensor(
            "dspark_target_hidden",
            (
                self._compiled.layout.ranks,
                DSPARK_DECODE_LOCAL_TOKENS,
                DSPARK_MAIN_HIDDEN_DIM,
            ),
            torch.bfloat16,
            scope="decode",
        )
        worker = self._shared_l3_worker()
        row_bytes = DSPARK_MAIN_HIDDEN_DIM * 2
        for index, shard in enumerate(device.shards):
            worker.copy_from(
                host_mirror[index].data_ptr(),
                shard.data_ptr,
                rows * row_bytes,
                worker_id=device.worker_ids[index],
            )
        return host_mirror[:, :rows]

    def _debug_dump_accepted_drafter_hidden(
        self,
        device: StackedDeviceTensor,
        rows_by_rank: list[list[DSparkDrafterRequestRow]],
    ) -> None:
        """Dump the first recurrent accept hidden rows for split/fused comparison."""
        dump_dir = os.environ.get("PYPTO_DSPARK_DEBUG_DUMP_DIR")
        if not dump_dir or self._drafter_hidden_mirror is None:
            return
        debug_rows = [
            (rank, row)
            for rank, rows in enumerate(rows_by_rank)
            for row in rows
            if self._drafter_state(row.request_id).verify_steps <= 3
        ]
        if not debug_rows:
            return
        rows = DSPARK_DRAFTER_BATCHES[0] * self._compiled.layout.decode_seq
        worker = self._shared_l3_worker()
        row_bytes = DSPARK_MAIN_HIDDEN_DIM * 2
        for rank, shard in enumerate(device.shards):
            worker.copy_from(
                self._drafter_hidden_mirror[rank].data_ptr(),
                shard.data_ptr,
                rows * row_bytes,
                worker_id=device.worker_ids[rank],
            )
        os.makedirs(dump_dir, exist_ok=True)
        path_kind = "one_l2" if self._compiled.decode_full_fused else "split_l3"
        for rank, row in debug_rows:
            hidden = self._drafter_hidden_mirror[
                rank, row.hidden_row : row.hidden_row + row.valid_count
            ].clone()
            path = os.path.join(
                dump_dir,
                f"{path_kind}-{row.request_id}-verify"
                f"{self._drafter_state(row.request_id).verify_steps}.pt",
            )
            torch.save(
                {
                    "rank": rank,
                    "hidden_row": row.hidden_row,
                    "valid_count": row.valid_count,
                    "anchor": row.anchor,
                    "token_source": row.token_source,
                    "hidden": hidden,
                },
                path,
            )

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Free each finished request's drafter lease and pending state.

        Idempotent and safe for requests this runner never saw: completion,
        abort, and preemption all funnel through here, and a re-admitted
        re-prefill starts a fresh state incarnation with a new lease.
        """
        for request_id in request_ids:
            state = self._drafter_states.pop(request_id, None)
            if state is not None:
                if state.verify_steps:
                    logger.info(
                        "DSpark speculation finished: request=%s verifies=%d "
                        "matched=%d proposed=%d accepted=%d mean_len=%.2f "
                        "fallbacks=%d",
                        request_id,
                        state.verify_steps,
                        state.matched_drafts,
                        state.proposed_tokens,
                        state.accepted_tokens,
                        state.accepted_tokens / state.verify_steps,
                        state.fallback_steps,
                    )
                free = self._drafter_free_leases.setdefault(state.group, [])
                free.append(state.lease)

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
            self._dspark_device_state_tokens = None
            self._dspark_device_state_meta = None
            self._dspark_rope_device_tables = None
            self._device_scratch.clear()
            self._decode_device_cache = None
            self._global_weights = None
            self._static_lm_head_weight = None
            self._static_lm_head_device_weight = None
            self._hc_head_buffers = None
            self._l3_shared_buffers_ready = False
            self._l3_static_tensors.clear()
            if self._prefill_task_args is not None:
                self._prefill_task_args.close()
                self._prefill_task_args = None
            for task_args in self._decode_task_args:
                task_args.close()
            self._decode_task_args = []
            # Speculative resources: the executor retains its runners, so
            # every drafter-era reference must drop here or staging buffers,
            # weights, and per-request states outlive the model.
            self._drafter_states.clear()
            for leases in self._drafter_free_leases.values():
                leases.clear()
            self._drafter_context_staging.clear()
            self._drafter_block_table_staging.clear()
            self._drafter_rope_candidates.clear()
            self._fused_drafter_block_tables.clear()
            self._fused_drafter_rope_candidates.clear()
            self._fused_drafter_batches = [None, None]
            self._dspark_state_buffers = []
            self._drafter_hidden_mirror = None
            self._active_drafter_target_hidden = None
            self._drafter_host_weights = None
            self._drafter_device_weights = None
            if self._fused_drafter_task_args:
                for task_args in self._fused_drafter_task_args:
                    task_args.close()
            elif self._drafter_task_args is not None:
                self._drafter_task_args.close()
            self._fused_drafter_task_args = []
            self._drafter_task_args = None
            if self._fused_markov_task_args:
                for task_args in self._fused_markov_task_args:
                    task_args.close()
            elif self._markov_task_args is not None:
                self._markov_task_args.close()
            self._fused_markov_task_args = []
            self._markov_task_args = None
            with self._pending_decode_dispatch_lock:
                self._pending_decode_dispatches.clear()
