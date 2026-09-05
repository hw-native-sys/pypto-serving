# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import math
import threading
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pypto_serving.model.common.runner.task_args import TaskArgs
    from pypto_serving.model.deepseek.task_args import DeepSeekPrefillTaskArgs

import numpy as np
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
from pypto_serving.model.common.runner.buffer_set import (
    copy_shared,
    share_cpu_tensor,
    shared_empty,
)
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin, PendingL3Dispatch
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.deepseek.weight_loader import (
    DeepSeekV4GlobalWeights,
    DeepSeekV4MtpWeights,
    DeepSeekV4StackedLayerWeights,
    DeepSeekV4WeightStore,
)
from pypto_serving.tools.profile import profile_span


logger = logging.getLogger(__name__)


class DeepSeekV4ServingContract(Protocol):
    """Serving-visible constraints exported by the pypto-lib kernels."""

    schema_version: str
    prefill_tile_tokens: int
    max_prefill_tokens_per_request: int
    max_prefill_requests_per_partition: int
    requires_homogeneous_prefill_decode: bool

    def padded_prefill_tokens(self, active_tokens: int) -> int:
        """Return the kernel extent for one active prefill request."""
        ...


DEEPSEEK_V4_RANKS = 8
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_VOCAB_SIZE = 129280
DEEPSEEK_V4_BLOCK_SIZE = 128
DEEPSEEK_V4_DECODE_BATCH = 4
DEEPSEEK_V4_DECODE_SEQ = 2
DEEPSEEK_V4_DECODE_TOKENS = DEEPSEEK_V4_DECODE_BATCH * DEEPSEEK_V4_DECODE_SEQ
DEEPSEEK_V4_MTP_DECODE_TOKENS = 16
DEEPSEEK_V4_PREFILL_BATCH = 4
# The main prefill wrapper accepts four rank-local requests, gives each a
# dynamic extent, and walks it in fixed 128-token tiles. The standalone MTP
# prefill wrapper still accepts exactly one tile.
DEEPSEEK_V4_PREFILL_SEQ = 128
# Async serving pipelines at most two device steps. Rolling caches must retain
# the history needed by the next token plus every token that can be in flight;
# otherwise scheduling the second prefill chunk can recycle state pages before
# the first chunk has consumed them. This is the same admission-cap principle
# used by vLLM's SlidingWindowSpec.
DEEPSEEK_V4_MAX_IN_FLIGHT_PREFILL_TOKENS = 2 * DEEPSEEK_V4_PREFILL_SEQ
DEEPSEEK_V4_MAX_SEQ_LEN = 16384
# Prefill and decode share scheduler-owned rank-local physical pools. Group
# block IDs are local to each DP rank and address worker-resident cache shards.
DEEPSEEK_V4_PREFILL_ORI_MAX_BLOCKS = 128
DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS = 128
DEEPSEEK_V4_ORI_TABLE_MAX_BLOCKS = 128
DEEPSEEK_V4_SLIDING_WINDOW = 128
DEEPSEEK_V4_CMP_MAX_BLOCKS = 128
DEEPSEEK_V4_IDX_MAX_BLOCKS = 128
DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS = 64
DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_C128_STATE_BLOCK_SIZE = 8
DEEPSEEK_V4_C4_STATE_BLOCK_SIZE = 4
DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS = DEEPSEEK_V4_CMP_MAX_BLOCKS
DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS = DEEPSEEK_V4_IDX_MAX_BLOCKS
DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS = 2048
DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS = 4096
DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS = 4096
DEEPSEEK_V4_HEAD_DIM = 512
DEEPSEEK_V4_IDX_HEAD_DIM = 128
DEEPSEEK_V4_HCA_MAIN_OUT_DIM = 512
DEEPSEEK_V4_CSA_MAIN_OUT_DIM = 1024
DEEPSEEK_V4_CSA_INNER_OUT_DIM = 256
DEEPSEEK_V4_HCA_STATE_DIM = 2 * DEEPSEEK_V4_HCA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_STATE_DIM = 2 * DEEPSEEK_V4_CSA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_INNER_STATE_DIM = 2 * DEEPSEEK_V4_CSA_INNER_OUT_DIM
# Layer-stacking counts for the packed all-layer decode_fwd kernel.
DEEPSEEK_V4_FWD_NUM_LAYERS = 43
DEEPSEEK_V4_CSA_NUM_LAYERS = 21
DEEPSEEK_V4_HCA_NUM_LAYERS = 20
DEEPSEEK_V4_LM_HEAD_TP_SIZE = 4
DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS = 8
DEEPSEEK_V4_SAMPLED_IDS_PAD = 8


def build_deepseek_v4_cache_group_specs(
    num_hidden_layers: int,
    compress_ratios: Sequence[int] | None = None,
    *,
    decode_batch: int = DEEPSEEK_V4_DECODE_TOKENS,
    enable_mtp: bool = False,
    max_seq_len: int = DEEPSEEK_V4_MAX_SEQ_LEN,
) -> tuple[KVCacheGroupSpec, ...]:
    """Describe cache namespaces independently from their runtime pool sizes.

    ``max_blocks_per_seq`` is a per-request ring/table limit. The number of
    physical blocks in each rank-local pool is intentionally left unset: the
    runner fills available NPU memory after weights and persistent scratch have
    been materialized, then the engine scales every group to the reported
    runtime capacity. Full-history compressed groups are sized from
    ``max_seq_len`` so distinct logical pages never alias the same physical
    page within a request.
    """
    decode_batch = int(decode_batch)
    if decode_batch <= 0:
        raise ValueError("decode_batch must be positive")
    max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    all_layers = tuple(range(int(num_hidden_layers)))
    ori_layers = all_layers + ((int(num_hidden_layers),) if enable_mtp else ())
    ratios = tuple(int(ratio) for ratio in (compress_ratios or ()))[:num_hidden_layers]
    csa_layers = tuple(index for index, ratio in enumerate(ratios) if ratio == 4) or all_layers
    hca_layers = tuple(index for index, ratio in enumerate(ratios) if ratio == 128) or all_layers
    full_history_blocks = math.ceil(max_seq_len / DEEPSEEK_V4_BLOCK_SIZE)
    if full_history_blocks > min(DEEPSEEK_V4_CMP_MAX_BLOCKS, DEEPSEEK_V4_IDX_MAX_BLOCKS):
        raise ValueError(
            f"DeepSeekV4 max_seq_len={max_seq_len} needs {full_history_blocks} source-token "
            f"blocks, but the compiled block tables support {DEEPSEEK_V4_CMP_MAX_BLOCKS}"
        )

    # SWA can start at an arbitrary offset inside a source-token block, so it
    # needs vLLM's extra boundary page. Compressor states are emitted on their
    # block boundary and therefore need exactly history + in-flight rows.
    ori_blocks_per_request = (
        math.ceil(
            (
                DEEPSEEK_V4_SLIDING_WINDOW
                - 1
                + DEEPSEEK_V4_MAX_IN_FLIGHT_PREFILL_TOKENS
            )
            / DEEPSEEK_V4_BLOCK_SIZE
        )
        + 1
    )
    hca_state_blocks_per_request = math.ceil(
        (
            DEEPSEEK_V4_SLIDING_WINDOW
            + DEEPSEEK_V4_MAX_IN_FLIGHT_PREFILL_TOKENS
        )
        / DEEPSEEK_V4_C128_STATE_BLOCK_SIZE
    )
    csa_state_blocks_per_request = math.ceil(
        (4 + DEEPSEEK_V4_MAX_IN_FLIGHT_PREFILL_TOKENS)
        / DEEPSEEK_V4_C4_STATE_BLOCK_SIZE
    )
    rolling_capacities = (
        ("ori", ori_blocks_per_request, DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS),
        ("hca_state", hca_state_blocks_per_request, DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS),
        ("csa_state", csa_state_blocks_per_request, DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS),
        (
            "csa_inner_state",
            csa_state_blocks_per_request,
            DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS,
        ),
    )
    for name, required, compiled_limit in rolling_capacities:
        if required > compiled_limit:
            raise ValueError(
                f"DeepSeekV4 {name} cache needs {required} blocks per request to cover "
                f"the rolling history and in-flight prefill tokens, but the compiled "
                f"block table supports {compiled_limit}"
            )

    def group(
        name: str,
        layers: tuple[int, ...],
        *,
        block_size: int,
        element_bytes: int,
        row_width: int,
        max_blocks: int,
        max_blocks_per_seq: int | None = None,
        compress_ratio: int = 1,
        extra_row_bytes: int = 0,
        sliding_window: int | None = None,
        is_eagle_group: bool = False,
    ) -> KVCacheGroupSpec:
        if max_blocks <= decode_batch:
            raise ValueError(
                f"DeepSeekV4 {name} cache needs more than {decode_batch} physical blocks"
            )
        blocks_per_request = (
            max(1, int(max_blocks_per_seq))
            if max_blocks_per_seq is not None
            else max(1, max_blocks // decode_batch)
        )
        storage_rows = block_size // compress_ratio
        return KVCacheGroupSpec(
            name=name,
            layer_indices=layers,
            spec=KVCacheSpec(
                block_size=block_size,
                # One scheduler-visible block owns a page in every layer of
                # this cache family. Keep the complete physical footprint here
                # so all heterogeneous pools share one accurate HBM budget.
                page_size_bytes=(
                    len(layers)
                    * storage_rows
                    * (row_width * element_bytes + extra_row_bytes)
                ),
                compress_ratio=compress_ratio,
            ),
            # Each request may address this many pages through its logical ring;
            # physical capacity across concurrent requests is runtime-sized.
            max_blocks_per_seq=blocks_per_request,
            num_partitions=DEEPSEEK_V4_RANKS,
            sliding_window=sliding_window,
            is_eagle_group=is_eagle_group,
        )

    return (
        group(
            "ori",
            ori_layers,
            block_size=DEEPSEEK_V4_BLOCK_SIZE,
            element_bytes=2,
            row_width=DEEPSEEK_V4_HEAD_DIM,
            max_blocks=DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS,
            max_blocks_per_seq=ori_blocks_per_request,
            sliding_window=DEEPSEEK_V4_SLIDING_WINDOW,
            is_eagle_group=enable_mtp,
        ),
        group(
            "cmp_c128",
            hca_layers,
            block_size=DEEPSEEK_V4_BLOCK_SIZE,
            element_bytes=2,
            row_width=DEEPSEEK_V4_HEAD_DIM,
            max_blocks=DEEPSEEK_V4_CMP_MAX_BLOCKS,
            compress_ratio=128,
            max_blocks_per_seq=full_history_blocks,
        ),
        group(
            "cmp_c4",
            csa_layers,
            block_size=DEEPSEEK_V4_BLOCK_SIZE,
            element_bytes=2,
            row_width=DEEPSEEK_V4_HEAD_DIM,
            max_blocks=DEEPSEEK_V4_CMP_MAX_BLOCKS,
            compress_ratio=4,
            max_blocks_per_seq=full_history_blocks,
        ),
        group(
            "idx",
            csa_layers,
            block_size=DEEPSEEK_V4_BLOCK_SIZE,
            element_bytes=1,
            row_width=DEEPSEEK_V4_IDX_HEAD_DIM,
            max_blocks=DEEPSEEK_V4_IDX_MAX_BLOCKS,
            compress_ratio=4,
            max_blocks_per_seq=full_history_blocks,
            # One float32 scale accompanies every int8 index-cache row.
            extra_row_bytes=4,
        ),
        group(
            "hca_state",
            hca_layers,
            block_size=DEEPSEEK_V4_C128_STATE_BLOCK_SIZE,
            element_bytes=4,
            row_width=DEEPSEEK_V4_HCA_STATE_DIM,
            max_blocks=DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS,
            max_blocks_per_seq=hca_state_blocks_per_request,
            sliding_window=DEEPSEEK_V4_SLIDING_WINDOW,
        ),
        group(
            "csa_state",
            csa_layers,
            block_size=DEEPSEEK_V4_C4_STATE_BLOCK_SIZE,
            element_bytes=4,
            row_width=DEEPSEEK_V4_CSA_STATE_DIM,
            max_blocks=DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS,
            max_blocks_per_seq=csa_state_blocks_per_request,
            sliding_window=4,
        ),
        group(
            "csa_inner_state",
            csa_layers,
            block_size=DEEPSEEK_V4_C4_STATE_BLOCK_SIZE,
            element_bytes=4,
            row_width=DEEPSEEK_V4_CSA_INNER_STATE_DIM,
            max_blocks=DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS,
            max_blocks_per_seq=csa_state_blocks_per_request,
            sliding_window=4,
        ),
    )


DEEPSEEK_V4_CACHE_GROUP_NAMES = (
    "ori",
    "cmp_c128",
    "cmp_c4",
    "idx",
    "hca_state",
    "csa_state",
    "csa_inner_state",
)


def deepseek_v4_cache_blocks_for_slots(
    group_specs: Sequence[KVCacheGroupSpec],
    capacity_slots: int,
) -> dict[str, int]:
    """Return scheduler-visible blocks per rank for ``capacity_slots`` requests."""
    capacity_slots = int(capacity_slots)
    if capacity_slots <= 0:
        raise ValueError("DeepSeekV4 cache capacity_slots must be positive")
    specs = {spec.name: spec for spec in group_specs}
    missing = [name for name in DEEPSEEK_V4_CACHE_GROUP_NAMES if name not in specs]
    if missing:
        raise ValueError("missing DeepSeekV4 cache groups: " + ", ".join(missing))
    return {
        name: capacity_slots * specs[name].max_blocks_per_seq
        for name in DEEPSEEK_V4_CACHE_GROUP_NAMES
    }


def deepseek_v4_physical_cache_blocks(
    group_specs: Sequence[KVCacheGroupSpec],
    capacity_slots: int,
    *,
    scratch_blocks: int,
) -> dict[str, int]:
    """Return physical blocks per rank, including isolated padding pages."""
    if scratch_blocks <= 0:
        raise ValueError("DeepSeekV4 scratch_blocks must be positive")
    return {
        name: num_blocks + int(scratch_blocks)
        for name, num_blocks in deepseek_v4_cache_blocks_for_slots(
            group_specs,
            capacity_slots,
        ).items()
    }


_MTP_DEVICE_STATE_TOKEN_WIDTH = 2
_MTP_DEVICE_STATE_META_WIDTH = 7
_MTP_STATE_VALID = 0
_MTP_STATE_GENERATION = 1
_MTP_STATE_TAIL_POSITION = 2
_MTP_STATE_COMMITTED_COUNT = 3
_MTP_STATE_TEMPERATURE_MICROS = 4
_MTP_STATE_TOP_K = 5
_MTP_STATE_SEED = 6
_MTP_TEMPERATURE_SCALE = 1_000_000
_MTP_MAX_TEMPERATURE_MICROS = 2_147_483_647


@dataclass(frozen=True)
class DeepSeekV4CacheLayout:
    """Kernel table depths and fixed execution dimensions.

    Physical cache-pool sizes are runtime state on ``DeepSeekV4ModelRunner``;
    the ``*_max_blocks`` fields here only bound per-request metadata tables and
    provide compatibility defaults for shape-only callers.
    """

    ranks: int = DEEPSEEK_V4_RANKS
    hc_mult: int = DEEPSEEK_V4_HC_MULT
    block_size: int = DEEPSEEK_V4_BLOCK_SIZE
    decode_batch: int = DEEPSEEK_V4_DECODE_BATCH
    decode_seq: int = DEEPSEEK_V4_DECODE_SEQ
    decode_tokens: int = DEEPSEEK_V4_DECODE_TOKENS
    prefill_batch: int = DEEPSEEK_V4_PREFILL_BATCH
    prefill_seq: int = DEEPSEEK_V4_PREFILL_SEQ
    prefill_ori_max_blocks: int = DEEPSEEK_V4_PREFILL_ORI_MAX_BLOCKS
    decode_ori_max_blocks: int = DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS
    ori_table_max_blocks: int = DEEPSEEK_V4_ORI_TABLE_MAX_BLOCKS
    sliding_window: int = DEEPSEEK_V4_SLIDING_WINDOW
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

    def validate_runtime(self, config: ModelConfig, runtime: RuntimeConfig, device_ids: Sequence[int]) -> None:
        """Validate serving/runtime options against kernel-fixed dimensions."""
        if len(device_ids) != self.ranks:
            raise ValueError(f"DeepSeekV4 requires exactly {self.ranks} devices, got {len(device_ids)}")
        if runtime.page_size != self.block_size:
            raise ValueError(f"DeepSeekV4 kernels require page_size={self.block_size}, got {runtime.page_size}")
        global_decode_capacity = self.ranks * self.decode_batch
        if runtime.max_batch_size > global_decode_capacity:
            raise ValueError(
                f"DeepSeekV4 decode kernels support at most {global_decode_capacity} global active rows "
                f"({self.decode_batch} per rank x {self.ranks} ranks), "
                f"got max_batch_size={runtime.max_batch_size}"
            )
        decode_state_capacity = self.prefill_csa_state_max_blocks * self.c4_state_block_size
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


def deepseek_v4_decode_layout(num_speculative_tokens: int) -> DeepSeekV4CacheLayout:
    """Select an aggressive 16-row decode tile for one MTP chunk.

    Autoregressive decode retains the established eight-row tile. MTP doubles
    the per-rank tile to 16 rows and groups them into power-of-two request-local
    sequences so target verification preserves twice the request capacity.
    Draft depths larger than seven use repeated S=8 target chunks.
    """
    num_speculative_tokens = int(num_speculative_tokens)
    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be non-negative")
    decode_tokens = (
        DEEPSEEK_V4_DECODE_TOKENS
        if num_speculative_tokens == 0
        else DEEPSEEK_V4_MTP_DECODE_TOKENS
    )
    if num_speculative_tokens == 0:
        decode_seq = 1
    elif num_speculative_tokens == 1:
        decode_seq = 2
    elif num_speculative_tokens <= 3:
        decode_seq = 4
    else:
        decode_seq = 8
    return DeepSeekV4CacheLayout(
        decode_batch=decode_tokens // decode_seq,
        decode_seq=decode_seq,
        decode_tokens=decode_tokens,
    )


@dataclass(frozen=True)
class DeepSeekV4CacheMetadataBuilder:
    """Build kernel metadata from scheduler-owned rank-local cache block IDs."""

    layout: DeepSeekV4CacheLayout = field(default_factory=DeepSeekV4CacheLayout)

    @staticmethod
    def block_table_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a padded block table from scheduler-owned physical IDs."""
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        table = DeepSeekV4CacheMetadataBuilder._prepare_table_output(
            out,
            rows=len(per_request_block_ids),
            max_blocks=max_blocks,
        )
        table_array = table.numpy()
        table_array.fill(0)
        buckets: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for row, block_ids in enumerate(per_request_block_ids):
            ids = tuple(int(block_id) for block_id in block_ids)
            if len(ids) > max_blocks:
                raise ValueError(f"row {row} has {len(ids)} blocks, maximum is {max_blocks}")
            if any(block_id < 0 for block_id in ids):
                raise ValueError("block IDs must not be negative")
            if ids:
                buckets.setdefault(len(ids), []).append((row, ids))
        for width, rows in buckets.items():
            row_indices = np.fromiter((row for row, _ in rows), dtype=np.intp, count=len(rows))
            values = np.asarray([ids for _, ids in rows], dtype=np.int32)
            table_array[row_indices, :width] = values
        return table

    @staticmethod
    def ring_block_table_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand scheduler-owned physical IDs across a logical ring table."""
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        table = DeepSeekV4CacheMetadataBuilder._prepare_table_output(
            out,
            rows=len(per_request_block_ids),
            max_blocks=max_blocks,
        )
        table_array = table.numpy()
        buckets: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for row, block_ids in enumerate(per_request_block_ids):
            ids = tuple(int(block_id) for block_id in block_ids)
            if not ids:
                raise ValueError(f"ring block-table row {row} has no allocated blocks")
            if any(block_id < 0 for block_id in ids):
                raise ValueError("block IDs must not be negative")
            buckets.setdefault(len(ids), []).append((row, ids))
        for width, rows in buckets.items():
            row_indices = np.fromiter((row for row, _ in rows), dtype=np.intp, count=len(rows))
            values = np.asarray([ids for _, ids in rows], dtype=np.int32)
            if width == 1:
                table_array[row_indices] = values
                continue
            repeated = np.tile(values, (1, math.ceil(max_blocks / width)))[:, :max_blocks]
            table_array[row_indices] = repeated
        return table

    @staticmethod
    def _prepare_table_output(
        out: torch.Tensor | None,
        *,
        rows: int,
        max_blocks: int,
    ) -> torch.Tensor:
        expected_shape = (rows, max_blocks)
        if out is None:
            return torch.empty(expected_shape, dtype=torch.int32)
        if (
            out.device.type != "cpu"
            or out.dtype != torch.int32
            or tuple(out.shape) != expected_shape
            or not out.is_contiguous()
        ):
            raise ValueError(
                "block-table output must be a contiguous CPU int32 tensor with "
                f"shape {expected_shape}, got shape={tuple(out.shape)} "
                f"dtype={out.dtype} device={out.device} contiguous={out.is_contiguous()}"
            )
        return out

    @staticmethod
    def slot_mapping_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        *,
        block_size: int,
        compress_ratio: int = 1,
    ) -> torch.Tensor:
        """Map logical positions through scheduler-owned physical blocks."""
        if block_size <= 0 or compress_ratio <= 0:
            raise ValueError("block_size and compress_ratio must be positive")
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        width = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(positions), width), -1, dtype=torch.long)
        mapping_array = mapping.numpy()
        buckets: dict[tuple[int, int], list[tuple[int, tuple[int, ...], tuple[int, ...]]]] = {}
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            ids = tuple(int(block_id) for block_id in block_ids)
            row_positions = tuple(int(position) for position in row_positions)
            if row_positions and not ids:
                raise ValueError(f"slot-mapping row {row} has no allocated blocks")
            if row_positions:
                buckets.setdefault((len(ids), len(row_positions)), []).append(
                    (row, ids, row_positions)
                )
        storage_block_size = block_size // compress_ratio
        for (block_count, position_count), rows in buckets.items():
            row_indices = np.fromiter((row for row, _, _ in rows), dtype=np.intp, count=len(rows))
            ids = np.asarray([row_ids for _, row_ids, _ in rows], dtype=np.int64)
            row_positions = np.asarray(
                [values for _, _, values in rows],
                dtype=np.int64,
            )
            source_blocks, source_offsets = np.divmod(row_positions, block_size)
            physical_blocks = np.take_along_axis(ids, source_blocks % block_count, axis=1)
            values = physical_blocks * storage_block_size + source_offsets // compress_ratio
            mapping_array[row_indices, :position_count] = values
        return mapping

    def paged_ori_block_table_from_ids(
        self,
        per_request_block_ids: Sequence[Sequence[int]],
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand scheduler-owned ori ring blocks into the absolute logical table."""
        for row, block_ids in enumerate(per_request_block_ids):
            ids = tuple(int(block_id) for block_id in block_ids)
            if not ids:
                raise ValueError(f"ori ring row {row} has no allocated blocks")
            if len(ids) > self.layout.decode_ori_max_blocks:
                raise ValueError(
                    f"ori ring row {row} has {len(ids)} blocks, maximum is "
                    f"{self.layout.decode_ori_max_blocks}"
                )
            if any(block_id < 0 for block_id in ids):
                raise ValueError("ori ring block IDs must not be negative")
        return self.ring_block_table_from_ids(
            per_request_block_ids,
            max_blocks=self.layout.ori_table_max_blocks,
            out=out,
        )

    def paged_decode_slot_mapping_from_ids(
        self,
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Map absolute decode writes through scheduler-owned ori ring blocks."""
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        width = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(positions), width), -1, dtype=torch.long)
        mapping_array = mapping.numpy()
        block_size = int(self.layout.block_size)
        buckets: dict[tuple[int, int], list[tuple[int, tuple[int, ...], tuple[int, ...]]]] = {}
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            ids = tuple(int(block_id) for block_id in block_ids)
            if not ids:
                raise ValueError(f"ori ring row {row} has no allocated blocks")
            row_positions = tuple(int(position) for position in row_positions)
            if row_positions:
                buckets.setdefault((len(ids), len(row_positions)), []).append(
                    (row, ids, row_positions)
                )
        for (block_count, position_count), rows in buckets.items():
            row_indices = np.fromiter((row for row, _, _ in rows), dtype=np.intp, count=len(rows))
            ids = np.asarray([row_ids for _, row_ids, _ in rows], dtype=np.int64)
            row_positions = np.asarray(
                [values for _, _, values in rows],
                dtype=np.int64,
            )
            logical_blocks, offsets = np.divmod(row_positions, block_size)
            physical_blocks = np.take_along_axis(ids, logical_blocks % block_count, axis=1)
            mapping_array[row_indices, :position_count] = physical_blocks * block_size + offsets
        return mapping

    def swa_window_indices_and_lens_from_ids(
        self,
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower visible SWA rows through scheduler-owned ori ring blocks."""
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        width = max((len(row) for row in positions), default=0)
        window = int(self.layout.sliding_window)
        block_size = int(self.layout.block_size)
        indices = torch.full(
            (len(positions) * width, window),
            -1,
            dtype=torch.int32,
        )
        lens = torch.zeros((len(positions) * width,), dtype=torch.int32)
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            ids = tuple(int(block_id) for block_id in block_ids)
            if not ids:
                raise ValueError(f"SWA row {row} has no allocated blocks")
            if any(block_id < 0 for block_id in ids):
                raise ValueError("block IDs must not be negative")
            for seq_index, position in enumerate(row_positions):
                position = int(position)
                if position < 0:
                    raise ValueError("decode positions must not be negative")
                token = row * width + seq_index
                start = max(0, position - window + 1)
                for offset, visible_position in enumerate(range(start, position + 1)):
                    logical_block, block_offset = divmod(visible_position, block_size)
                    physical_block = ids[logical_block % len(ids)]
                    indices[token, offset] = physical_block * block_size + block_offset
                lens[token] = position - start + 1
        return indices, lens

    @staticmethod
    def compressed_slot_mapping_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        *,
        block_size: int,
        compress_ratio: int,
    ) -> torch.Tensor:
        """Map compression-boundary positions through physical cache blocks."""
        if block_size <= 0 or compress_ratio <= 0:
            raise ValueError("block_size and compress_ratio must be positive")
        if len(per_request_block_ids) != len(positions):
            raise ValueError("block IDs and positions must have the same row count")
        width = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(positions), width), -1, dtype=torch.long)
        mapping_array = mapping.numpy()
        buckets: dict[tuple[int, int], list[tuple[int, tuple[int, ...], tuple[int, ...]]]] = {}
        for row, (block_ids, row_positions) in enumerate(
            zip(per_request_block_ids, positions, strict=True)
        ):
            ids = tuple(int(block_id) for block_id in block_ids)
            row_positions = tuple(int(position) for position in row_positions)
            if row_positions:
                buckets.setdefault((len(ids), len(row_positions)), []).append(
                    (row, ids, row_positions)
                )
        storage_block_size = block_size // compress_ratio
        for (block_count, position_count), rows in buckets.items():
            row_indices = np.fromiter((row for row, _, _ in rows), dtype=np.intp, count=len(rows))
            row_positions = np.asarray(
                [values for _, _, values in rows],
                dtype=np.int64,
            )
            valid = (row_positions + 1) % compress_ratio == 0
            if not np.any(valid):
                continue
            if block_count == 0:
                bucket_row = int(np.argwhere(valid)[0, 0])
                first_row = rows[bucket_row][0]
                raise ValueError(
                    f"compressed slot-mapping row {first_row} has no allocated blocks"
                )
            source_blocks, source_offsets = np.divmod(row_positions, block_size)
            invalid_rows = np.argwhere(valid & (source_blocks >= block_count))
            if invalid_rows.size:
                bucket_row, col = (int(value) for value in invalid_rows[0])
                original_row = rows[bucket_row][0]
                position = int(row_positions[bucket_row, col])
                source_block = int(source_blocks[bucket_row, col])
                raise ValueError(
                    f"compressed slot-mapping row {original_row} position {position} requires "
                    f"source block {source_block}, but only {block_count} blocks are allocated"
                )
            ids = np.asarray([row_ids for _, row_ids, _ in rows], dtype=np.int64)
            safe_source_blocks = np.where(valid, source_blocks, 0)
            physical_blocks = np.take_along_axis(ids, safe_source_blocks, axis=1)
            values = physical_blocks * storage_block_size + source_offsets // compress_ratio
            bucket_output = np.full((len(rows), position_count), -1, dtype=np.int64)
            bucket_output[valid] = values[valid]
            mapping_array[row_indices, :position_count] = bucket_output
        return mapping

    @staticmethod
    def state_slot_mapping_from_ids(
        per_request_block_ids: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]],
        *,
        state_block_size: int,
    ) -> torch.Tensor:
        """Map absolute positions through physical compressor-state blocks."""
        return DeepSeekV4CacheMetadataBuilder.slot_mapping_from_ids(
            per_request_block_ids,
            positions,
            block_size=state_block_size,
        )


class DeepSeekV4InputBuilder:
    """Build fixed-shape host inputs for DeepSeekV4 HC-stack kernels."""

    def __init__(self, *, layout: DeepSeekV4CacheLayout, hidden_size: int) -> None:
        self.layout = layout
        self.hidden_size = int(hidden_size)

    def prefill_x_hc(
        self,
        embeddings: Sequence[torch.Tensor],
        *,
        ranks: Sequence[int],
        local_rows: Sequence[int],
        token_rows: int,
    ) -> torch.Tensor:
        """Build distinct rank/slot-local prefill streams for one EP dispatch."""
        if token_rows <= 0:
            raise ValueError("prefill token rows must be positive")
        if (
            not embeddings
            or len(embeddings) != len(ranks)
            or len(embeddings) != len(local_rows)
        ):
            raise ValueError("prefill embeddings, ranks, and local rows must be aligned")

        padded_rows = []
        for rows in embeddings:
            if rows.ndim != 2 or rows.shape[0] <= 0 or int(rows.shape[1]) != self.hidden_size:
                raise ValueError("rank-local prefill embeddings must have shape [tokens, hidden]")
            if rows.shape[0] > token_rows:
                raise ValueError("rank-local prefill embeddings exceed the kernel token rows")
            rows = rows.to(torch.float32)
            padded = torch.zeros((token_rows, self.hidden_size), dtype=rows.dtype, device=rows.device)
            padded[: rows.shape[0]].copy_(rows)
            if rows.shape[0] < token_rows:
                pad_indices = torch.arange(token_rows - rows.shape[0], device=rows.device) % rows.shape[0]
                padded[rows.shape[0] :].copy_(rows.index_select(0, pad_indices))
            padded_rows.append(padded)

        # Each request slot has a rank-wide token extent. Inactive rank/slot
        # pairs carry harmless filler rows while their token count stays zero.
        rank_rows = padded_rows[0].reshape(1, 1, token_rows, self.hidden_size).expand(
            self.layout.ranks,
            self.layout.prefill_batch,
            -1,
            -1,
        ).clone()
        for rank, local_row, rows in zip(ranks, local_rows, padded_rows, strict=True):
            rank = int(rank)
            local_row = int(local_row)
            if not 0 <= rank < self.layout.ranks:
                raise ValueError(f"prefill rank {rank} is out of range")
            if not 0 <= local_row < self.layout.prefill_batch:
                raise ValueError(f"prefill local row {local_row} is out of range")
            rank_rows[rank, local_row].copy_(rows)
        return (
            rank_rows.unsqueeze(3)
            .expand(
                self.layout.ranks,
                self.layout.prefill_batch,
                token_rows,
                self.layout.hc_mult,
                self.hidden_size,
            )
            .contiguous()
        )


# Compiled HOST-dispatched program. Unified with qwen's _L3Callable in
# pypto_serving.model.common.compiler.l3_callable.
from pypto_serving.model.common.compiler.l3_callable import L3Callable as DeepSeekV4L3Callable


@dataclass
class DeepSeekV4DeviceCache:
    """Worker-resident rank shards shared by packed prefill and decode."""

    kv_cache: StackedDeviceTensor
    hca_cmp_kv: StackedDeviceTensor
    csa_cmp_kv: StackedDeviceTensor
    idx_kv_cache: StackedDeviceTensor
    idx_kv_scale: StackedDeviceTensor
    hca_compress_state: StackedDeviceTensor
    csa_compress_state: StackedDeviceTensor
    csa_inner_compress_state: StackedDeviceTensor


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
    kernel_contract: DeepSeekV4ServingContract
    prepacked_layer_weights: DeepSeekV4StackedLayerWeights | None = None
    runtime_model: RuntimeModel | None = None
    prefill: DeepSeekV4L3Callable | None = None
    decode: DeepSeekV4L3Callable | None = None
    mtp_prefill: DeepSeekV4L3Callable | None = None
    mtp_decode: DeepSeekV4L3Callable | None = None
    freqs_cos: torch.Tensor | None = None
    freqs_sin: torch.Tensor | None = None
    platform: str = "a2a3"
    device_id: int = 0
    device_ids: tuple[int, ...] = ()
    n_routed_experts: int = 256
    num_hash_layers: int = 3
    embedding_weight: torch.Tensor | None = None
    num_speculative_tokens: int = 0

    def l3_callables(self) -> tuple[DeepSeekV4L3Callable, ...]:
        """Return every compiled L3 program that the shared worker may run."""
        callables: list[DeepSeekV4L3Callable] = []
        if self.prefill is not None:
            callables.append(self.prefill)
        if self.decode is not None:
            callables.append(self.decode)
        if self.mtp_prefill is not None:
            callables.append(self.mtp_prefill)
        if self.mtp_decode is not None:
            callables.append(self.mtp_decode)
        return tuple(callables)


@dataclass(frozen=True)
class DeepSeekV4PreparedPrefillInputs:
    """Tile-padded dynamic host tensors for one serving prefill chunk."""

    request_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    local_rows: tuple[int, ...]
    actual_tokens: tuple[int, ...]
    kernel_tokens: int
    x_hc: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    ori_block_table: torch.Tensor
    ori_slot_mapping: torch.Tensor
    hca_cmp_block_table: torch.Tensor
    csa_cmp_block_table: torch.Tensor
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
    num_tokens_per_owner: torch.Tensor
    logit_row_indices: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4PreparedDecodeInputs:
    """Fixed-shape shared tensors derived from one decode scheduler batch.

    The tensor fields are views over one runner-owned execution slot and remain
    valid until that slot is reused.
    """

    request_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    local_rows: tuple[int, ...]
    per_rank_counts: tuple[int, ...]
    actual_batch: int
    x_hc: torch.Tensor | None
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_table: torch.Tensor
    hca_cmp_block_table: torch.Tensor
    csa_cmp_block_table: torch.Tensor
    idx_block_table: torch.Tensor
    hca_compress_state_block_table: torch.Tensor
    csa_compress_state_block_table: torch.Tensor
    csa_inner_compress_state_block_table: torch.Tensor
    block_counts: torch.Tensor
    block_ids_by_group: tuple[dict[str, tuple[int, ...]], ...]
    num_tokens_per_owner: torch.Tensor
    # Main-model LM-head routing.  Each non-negative entry is a flattened
    # decode hidden-row index whose logits must be computed; -1 disables that
    # output row.  For MTP S=2, active request k normally contributes rows
    # 2*k and 2*k+1 so the verifier can compare both main-model predictions.
    logit_row_indices: torch.Tensor
    buffer_slot: int = 0
    # Per-batch-row mapping to the request's stable device-state/tail-pool
    # slot.  Unlike local_row, this slot survives batch reorder and compaction;
    # -1 marks padding or an inactive row.
    mtp_tail_slot_ids: torch.Tensor | None = None
    # Expected allocation generation for mtp_tail_slot_ids.  A kernel may use
    # a slot only when this value matches state_meta[slot].generation, which
    # prevents a stale prepared step from touching a slot recycled to another
    # request (the classic ABA reuse case).
    mtp_state_generations: torch.Tensor | None = None
    # MTP-draft LM-head routing.  There is one next-draft prediction per active
    # request, selected from the last row of its committed S=2 window; unused
    # entries are -1.
    mtp_logit_row_indices: torch.Tensor | None = None
    # Fully bound L3 arguments for the steady fused-MTP path.  Preparing this
    # tuple on the command lane keeps Python argument binding off the device
    # lane, including the first decode prepared while terminal prefill runs.
    dispatch_args: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class _DeepSeekV4DecodeAssignment:
    """Mapping from scheduler order to rank-local kernel rows."""

    ranks: tuple[int, ...]
    local_rows: tuple[int, ...]
    per_rank_counts: tuple[int, ...]
    indices_by_rank: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _DeepSeekV4MainDecodeOutput:
    """Main-model tensors produced by one packed decode dispatch."""

    inputs: DeepSeekV4PreparedDecodeInputs
    hidden: torch.Tensor
    pre_hc_hidden: torch.Tensor | StackedDeviceTensor
    logits: torch.Tensor
    sampled_ids: torch.Tensor


@dataclass(frozen=True)
class _DeepSeekV4PendingMtpDecode:
    """Submitted device dispatch whose small host outputs await reclaim."""

    dispatch: "PendingL3Dispatch"
    inputs: DeepSeekV4PreparedDecodeInputs
    sampled_ids: torch.Tensor
    mtp_sampled_ids: torch.Tensor
    accepted_counts: torch.Tensor
    states: tuple["_DeepSeekV4MtpRequestState", ...]


@dataclass(frozen=True)
class _DeepSeekV4MtpVerification:
    """Accepted tokens and recurrent inputs produced by target verification."""

    accepted_token_ids: list[list[int]]
    tail_token_ids: torch.Tensor
    tail_pre_hc_hidden: torch.Tensor
    tail_positions: torch.Tensor
    first_logits: torch.Tensor


@dataclass
class _DeepSeekV4MtpSharedBuffers:
    """MTP weights, KV-cache placeholder, and recurrent-state init buffers.

    The prefill host inputs/outputs live on the MTP prefill TaskArgs; the decode
    per-slot reclaimed outputs and write-only outputs live on the MTP decode
    TaskArgs.  What remains here: the MTP host weights (pre-upload), the 1-page
    KV-cache placeholder, the state-init seed buffers, and the prefill host
    readback mirror for ``_read_mtp_prefill_logits``.  ``prefill_pre_hc_mirror``
    holds the D2H readback of the MTP prefill pre-HC output consumed by the
    arbitrary-depth recurrent path.
    """

    weights: dict[str, torch.Tensor]
    prefill_kv_cache: torch.Tensor
    state_init_tokens: torch.Tensor
    state_init_meta: torch.Tensor
    tail_init_hidden: torch.Tensor
    prefill_logits: torch.Tensor
    prefill_pre_hc_mirror: torch.Tensor


@dataclass(frozen=True)
class _DeepSeekV4MtpPrefillContext:
    """One pending main-model row retained until its next token is known."""

    rank: int
    prev_hidden_state: torch.Tensor
    position_id: int
    block_table: torch.Tensor
    slot_mapping: int
    prompt_len: int


@dataclass
class _DeepSeekV4MtpRequestState:
    """Speculative state owned by one serving request."""

    prefill_context: _DeepSeekV4MtpPrefillContext | None = None
    draft_token_id: int | None = None
    draft_pre_hc_hidden: torch.Tensor | None = None
    draft_position: int | None = None
    tail_token_id: int | None = None
    tail_rank: int | None = None
    tail_slot_id: int | None = None
    tail_position: int | None = None
    generation: int = 0
    device_state_initialized: bool = False
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    # Running count of committed output tokens (advances by 1 or 2 per decode
    # step, unlike tail_position which stalls on draft rejection). Used to
    # correct seq_len under async scheduling.
    committed_count: int = 0
    # Prompt length, captured at initialization for seq_len correction.
    prompt_len: int = 0


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


def accept_mtp_tokens(main_token_ids: torch.Tensor, draft_token_ids: torch.Tensor) -> list[list[int]]:
    """Accept the longest matching MTP prefix plus one target-model token."""
    main = main_token_ids.detach().cpu().to(torch.long)
    draft = draft_token_ids.detach().cpu().to(torch.long)
    if main.ndim != 2 or main.shape[1] < 2:
        raise ValueError(f"main_token_ids must have shape [batch, K+1], got {tuple(main.shape)}")
    if draft.ndim != 2 or draft.shape != (main.shape[0], main.shape[1] - 1):
        raise ValueError(
            "draft_token_ids must have shape "
            f"{(main.shape[0], main.shape[1] - 1)}, got {tuple(draft.shape)}"
        )
    accepted: list[list[int]] = []
    for row in range(main.shape[0]):
        matched = 0
        while matched < draft.shape[1] and int(draft[row, matched]) == int(main[row, matched]):
            matched += 1
        accepted.append([int(token) for token in main[row, : matched + 1]])
    return accepted


class DeepSeekV4ModelRunner(L3DispatchMixin, ModelRunner):
    """Runner boundary for DeepSeekV4 W8A8 kernels and model-specific caches."""

    def __init__(
        self,
        *,
        compiled: DeepSeekV4CompiledKernels,
    ) -> None:
        super().__init__()
        self._compiled = compiled
        self.cache_metadata = DeepSeekV4CacheMetadataBuilder(layout=compiled.layout)
        self.input_builder: DeepSeekV4InputBuilder | None = None
        self._init_l3_dispatch(stacked=True)
        self._cache_group_specs: tuple[KVCacheGroupSpec, ...] = ()
        self._cache_group_num_blocks: dict[str, int] = {}
        self._decode_device_cache: DeepSeekV4DeviceCache | None = None
        self._global_weights: DeepSeekV4GlobalWeights | None = None
        self._static_final_norm_weight: torch.Tensor | None = None
        self._static_lm_head_weight: torch.Tensor | None = None
        self._static_freqs_cos: torch.Tensor | None = None
        self._static_freqs_sin: torch.Tensor | None = None
        self._prefill_task_args: DeepSeekPrefillTaskArgs | None = None
        # Per ping-pong slot, decode buffers live on the decode TaskArgs. Fused
        # K=1 write-only outputs are device-resident; scheduler-visible inputs,
        # metadata, and sampled IDs remain shared on Host.
        self._decode_task_args: list[TaskArgs] = []
        self._decode_input_slots: list[dict[str, torch.Tensor]] = []
        self._decode_static_metadata_keys: list[tuple[object, ...] | None] = [
            None
        ] * compiled.layout.ranks
        self._decode_assignment_cache_key: tuple[tuple[str, ...], tuple[int, ...]] | None = None
        self._decode_assignment_cache: _DeepSeekV4DecodeAssignment | None = None
        self._decode_fwd_args_cache: tuple[Any, ...] | None = None
        self._mtp_decode_args_cache: tuple[Any, ...] | None = None
        self._fused_mtp_main_args: tuple[Any, ...] | None = None
        self._fused_mtp_args_cache: dict[int, tuple[Any, ...]] = {}
        self._stacked_host_weights: dict[str, torch.Tensor] | None = None
        self._stacked_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._embedding_device_weight: StackedDeviceTensor | None = None
        self._mtp_tail_pre_hc_pool: StackedDeviceTensor | None = None
        self._mtp_device_state_tokens: StackedDeviceTensor | None = None
        self._mtp_device_state_meta: StackedDeviceTensor | None = None
        self._l3_shared_buffers_ready = False
        self._mtp_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._hc_head_buffers: dict[str, torch.Tensor] | None = None
        self._mtp_buffers: _DeepSeekV4MtpSharedBuffers | None = None
        self._mtp_device_kv_cache: StackedDeviceTensor | None = None
        self._main_pre_hc_host_mirror: torch.Tensor | None = None
        self._mtp_prefill_task_args: TaskArgs | None = None
        self._mtp_decode_task_args: list[TaskArgs] = []
        self._mtp_request_states: dict[str, _DeepSeekV4MtpRequestState] = {}
        self._mtp_state_lock = threading.RLock()
        self._pending_mtp_dispatch_lock = threading.Lock()
        self._pending_mtp_dispatches: dict[int, PendingL3Dispatch] = {}
        self._mtp_free_tail_slots: list[list[int]] = [
            list(range(compiled.layout.decode_batch - 1, -1, -1))
            for _ in range(compiled.layout.ranks)
        ]
        self._mtp_slot_generations = [
            [0] * compiled.layout.decode_batch for _ in range(compiled.layout.ranks)
        ]
        self._mtp_proposed_tokens = 0
        self._mtp_accepted_tokens = 0
        if compiled.num_speculative_tokens:
            self._decode_flow = self._run_mtp_decode
            self._prefill_completion = self._capture_mtp_prefill_context
        else:
            self._decode_flow = self._run_autoregressive_decode
            self._prefill_completion = self._ignore_prefill_context

    @property
    def supports_async_decode_reclaim(self) -> bool:
        """Fused MTP owns stable device state and ping-ponged host outputs."""
        return self._compiled.num_speculative_tokens == 1

    @staticmethod
    def prepared_decode_requires_token(prepared: object) -> bool:
        """Fused MTP consumes persistent state initialized during prefill."""
        return not isinstance(prepared, DeepSeekV4PreparedDecodeInputs)

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> int:
        """Allocate heterogeneous cache pools from the post-weight HBM budget.

        The returned value is the scheduler-visible ``ori`` block count in one
        rank-local partition. Other group capacities are derived from the same
        number of maximum-sequence cache slots by ``KvCacheManager``.
        """
        self.input_builder = DeepSeekV4InputBuilder(
            layout=self._compiled.layout,
            hidden_size=config.hidden_size,
        )
        self._cache_group_specs = self._resolve_cache_group_specs(config, runtime)
        self._configure_l3_rings(runtime)

        # Shape/metadata-only unit tests construct a runner without compiled
        # callables. Preserve a deterministic fixed fallback for that path; the
        # production executor always attaches the RuntimeModel and callables.
        model = self._compiled.runtime_model
        if model is None or not self._compiled.l3_callables():
            self._cache_group_num_blocks = self._fallback_cache_group_num_blocks()
            return self._cache_group_num_blocks["ori"]

        logger.info("[init_kv_cache] preparing DeepSeekV4 worker and resident weights …")
        self._ensure_l3_shared_buffers(model)

        requested_slots = self._compute_kv_cache_capacity_slots(runtime)
        allocated_slots = self._alloc_kv_cache_with_retry(requested_slots)
        self._log_kv_cache_allocation(runtime, requested_slots, allocated_slots)
        return self._cache_group_num_blocks["ori"]

    def _resolve_cache_group_specs(
        self,
        config: ModelConfig,
        runtime: RuntimeConfig,
    ) -> tuple[KVCacheGroupSpec, ...]:
        specs = runtime.kv_cache_groups or build_deepseek_v4_cache_group_specs(
            config.num_hidden_layers,
            self._compiled.compress_ratios,
            decode_batch=self._compiled.layout.decode_batch,
            enable_mtp=self._compiled.num_speculative_tokens == 1,
            max_seq_len=runtime.max_seq_len,
        )
        names = tuple(spec.name for spec in specs)
        if names != DEEPSEEK_V4_CACHE_GROUP_NAMES:
            raise ValueError(
                "DeepSeekV4 KV cache groups must be ordered as "
                + ", ".join(DEEPSEEK_V4_CACHE_GROUP_NAMES)
                + f"; got {names}"
            )
        if any(spec.num_partitions != self._compiled.layout.ranks for spec in specs):
            raise ValueError(
                f"DeepSeekV4 KV cache groups must use {self._compiled.layout.ranks} partitions"
            )
        return tuple(specs)

    def _fallback_cache_group_num_blocks(self) -> dict[str, int]:
        """Return one scheduler slot for shape-only/non-runtime callers.

        The compiled table depths limit one request's logical addressing, not
        the physical pool size.  Decode scratch pages are added separately by
        ``_physical_cache_num_blocks`` and must therefore not be subtracted
        from those logical limits here.
        """
        return deepseek_v4_cache_blocks_for_slots(
            self._cache_group_specs,
            1,
        )

    def _compute_kv_cache_capacity_slots(self, runtime: RuntimeConfig) -> int:
        """Compute common cache slots using Qwen's utilization-budget formula."""
        ori_spec = self._cache_group_specs[0]
        if runtime.total_kv_pages is not None:
            requested_pages = int(runtime.total_kv_pages)
            if requested_pages < ori_spec.max_blocks_per_seq:
                raise ValueError(
                    "DeepSeekV4 total_kv_pages must hold at least one maximum ring: "
                    f"expected >= {ori_spec.max_blocks_per_seq}, got {requested_pages}"
                )
            return requested_pages // ori_spec.max_blocks_per_seq

        # Ranks are enumerated as logical Worker IDs on the shared L3 worker:
        # the query runs in the chip process that owns each device context.
        # Physical device IDs are metadata for the log/error text only and are
        # never used to route the query (they may be non-contiguous).
        device_ids = self._compiled.device_ids or (self._compiled.device_id,)
        worker = self._shared_l3_worker()
        utilization = float(getattr(runtime, "npu_memory_utilization", 0.90))
        budgets = []
        memory_rows = []
        for worker_id in range(self._compiled.layout.ranks):
            free_bytes, total_bytes = worker.device_memory_info(worker_id)
            device_id = device_ids[worker_id] if worker_id < len(device_ids) else worker_id
            peak_non_kv = int(total_bytes) - int(free_bytes)
            budget = int(int(total_bytes) * utilization - peak_non_kv)
            budgets.append(budget)
            memory_rows.append((int(device_id), int(free_bytes), int(total_bytes), budget))

        bytes_per_slot = sum(
            spec.max_blocks_per_seq * spec.spec.page_size_bytes
            for spec in self._cache_group_specs
        )
        scratch_bytes = sum(
            self._compiled.layout.decode_batch * spec.spec.page_size_bytes
            for spec in self._cache_group_specs
        )
        # The mtp=1 prefix-cache path accounts for its extra KV layer in the
        # EAGLE ``ori`` group.  Arbitrary-depth MTP is intentionally not adapted
        # to grouped prefix caching yet and still owns a separate one-layer pool,
        # so retain the latest serving baseline's explicit capacity charge.
        multi_mtp_page_bytes = (
            self._compiled.layout.block_size
            * DEEPSEEK_V4_HEAD_DIM
            * torch.bfloat16.itemsize
            if self._compiled.num_speculative_tokens > 1
            else 0
        )
        bytes_per_slot += ori_spec.max_blocks_per_seq * multi_mtp_page_bytes
        scratch_bytes += self._compiled.layout.decode_batch * multi_mtp_page_bytes
        kv_budget = min(budgets)
        minimum_bytes = scratch_bytes + bytes_per_slot
        if kv_budget < minimum_bytes:
            device_id, free_bytes, total_bytes, budget = min(
                memory_rows,
                key=lambda row: row[3],
            )
            raise RuntimeError(
                "DeepSeekV4 KV cache cannot fit one capacity slot within "
                f"npu_memory_utilization={utilization:.2f} on npu:{device_id}: "
                f"post-resident budget={budget} bytes, requires at least "
                f"{minimum_bytes} bytes ({scratch_bytes} scratch + "
                f"{bytes_per_slot} one slot); total={total_bytes} bytes, "
                f"free={free_bytes} bytes"
            )
        capacity_slots = (kv_budget - scratch_bytes) // bytes_per_slot
        logger.info(
            "DeepSeekV4 KV cache sizing: utilization=%.2f, limiting_budget=%.2f GB, "
            "slot=%.1f MB, scratch=%.1f MB, requested_slots=%d",
            utilization,
            kv_budget / 1e9,
            bytes_per_slot / 1e6,
            scratch_bytes / 1e6,
            capacity_slots,
        )
        for device_id, free_bytes, total_bytes, budget in memory_rows:
            logger.info(
                "  npu:%d total=%.2f GB free=%.2f GB post-resident KV budget=%.2f GB",
                device_id,
                total_bytes / 1e9,
                free_bytes / 1e9,
                budget / 1e9,
            )
        return int(capacity_slots)

    def _alloc_kv_cache_with_retry(self, requested_slots: int) -> int:
        """Allocate every cache family atomically, halving capacity on OOM."""
        capacity_slots = max(int(requested_slots), 1)
        while capacity_slots >= 1:
            self._cache_group_num_blocks = deepseek_v4_cache_blocks_for_slots(
                self._cache_group_specs,
                capacity_slots,
            )
            try:
                self._materialize_decode_device_cache()
                self._materialize_mtp_device_kv_cache()
                return capacity_slots
            except (RuntimeError, MemoryError) as exc:
                self._free_device_caches()
                if capacity_slots == 1:
                    raise RuntimeError(
                        "DeepSeekV4 KV cache allocation failed at the one-slot minimum"
                    ) from exc
                previous = capacity_slots
                capacity_slots = max(capacity_slots // 2, 1)
                logger.warning(
                    "DeepSeekV4 KV cache allocation failed (%s); retrying slots %d -> %d",
                    exc,
                    previous,
                    capacity_slots,
                )
        raise RuntimeError("DeepSeekV4 KV cache allocation failed")

    def _log_kv_cache_allocation(
        self,
        runtime: RuntimeConfig,
        requested_slots: int,
        allocated_slots: int,
    ) -> None:
        main_bytes = sum(
            self._physical_cache_num_blocks(spec.name) * spec.spec.page_size_bytes
            for spec in self._cache_group_specs
        )
        multi_mtp_bytes = 0
        if self._compiled.num_speculative_tokens > 1:
            multi_mtp_bytes = (
                self._physical_cache_num_blocks("ori")
                * self._compiled.layout.block_size
                * DEEPSEEK_V4_HEAD_DIM
                * torch.bfloat16.itemsize
            )
        logger.info(
            "[init_kv_cache] allocated DeepSeekV4 cache: slots=%d (requested=%d), "
            "per-rank=%.2f GB, ori_blocks=%d, max_seq_len=%d",
            allocated_slots,
            requested_slots,
            (main_bytes + multi_mtp_bytes) / 1e9,
            self._cache_group_num_blocks["ori"],
            runtime.max_seq_len,
        )

    def _physical_cache_num_blocks(self, group_name: str) -> int:
        try:
            return self._cache_group_num_blocks[group_name] + self._compiled.layout.decode_batch
        except KeyError as exc:
            raise RuntimeError("DeepSeekV4 KV cache capacity is not initialized") from exc

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Discard request-local MTP state for finished or preempted requests."""
        request_ids = tuple(request_ids)
        for request_id in request_ids:
            with self._mtp_state_lock:
                state = self._mtp_request_states.pop(request_id, None)
                if state is not None and state.tail_rank is not None and state.tail_slot_id is not None:
                    # Lifecycle release runs on the FIFO device lane. A future
                    # owner receives a new generation and overwrites the complete
                    # meta row before publishing its descriptor. The kernel also
                    # requires STATE_VALID and an exact generation match.
                    self._mtp_free_tail_slots[state.tail_rank].append(state.tail_slot_id)
            if state is None:
                continue
            if state.proposed_tokens:
                logger.info(
                    "DeepSeekV4 MTP acceptance for %s: accepted=%d proposed=%d rate=%.2f%%",
                    request_id,
                    state.accepted_tokens,
                    state.proposed_tokens,
                    100.0 * state.accepted_tokens / state.proposed_tokens,
                )

    def preflight(self, record: ModelRecord) -> None:
        """Stage host buffers and allocate the resident cache before worker readiness."""
        self._ensure_l3_shared_buffers(record.runtime_model)
        self._materialize_decode_device_cache()
        self._materialize_mtp_device_kv_cache()

    def load_packed_global_weights(self) -> DeepSeekV4GlobalWeights:
        """Load global tensors and shard the device LM head across its TP ranks."""
        if self._global_weights is None:
            loaded = self._compiled.weight_store.load_packed_global_weights(
                ranks=DEEPSEEK_V4_LM_HEAD_TP_SIZE
            )
            embed_weight = loaded.embed_weight.to(
                device="cpu",
                dtype=torch.bfloat16,
            ).contiguous()
            exact_weight = loaded.lm_head_weight[:, : loaded.lm_head_layout.vocab_per_rank, :].contiguous()
            self._global_weights = replace(
                loaded,
                embed_weight=embed_weight,
                lm_head_weight=exact_weight,
            )
            self._compiled.embedding_weight = embed_weight
        return self._global_weights

    def load_stacked_layer_weights(self) -> DeepSeekV4StackedLayerWeights:
        """Load and stack all hidden-layer weights for the packed decode_fwd kernel."""
        if self._compiled.prepacked_layer_weights is not None:
            return self._compiled.prepacked_layer_weights
        compress_ratios = tuple(int(layer.compress_ratio) for layer in self._compiled.layer_plan)
        return self._compiled.weight_store.load_stacked_layer_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=self._compiled.num_hash_layers,
            use_prepacked=False,
        )

    def load_mtp_weights(self) -> DeepSeekV4MtpWeights:
        """Load the single checkpoint MTP draft layer."""
        return self._compiled.weight_store.load_mtp_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
        )

    @staticmethod
    def _validate_prefill_batch_metadata(batch: PrefillBatch, request_count: int) -> None:
        """Validate the packed-token metadata consumed by DeepSeek prefill."""
        metadata = (
            ("seq_lens", batch.seq_lens),
            ("chunk_lens", batch.chunk_lens),
            ("chunk_offsets", batch.chunk_offsets),
            ("chunk_starts", batch.chunk_starts),
        )
        for name, values in metadata:
            if len(values) != request_count:
                raise ValueError(
                    f"DeepSeekV4 prefill {name} has {len(values)} entries for "
                    f"{request_count} requests"
                )

        packed_tokens = 0
        for index in range(request_count):
            seq_len = int(batch.seq_lens[index])
            chunk_len = int(batch.chunk_lens[index])
            chunk_offset = int(batch.chunk_offsets[index])
            chunk_start = int(batch.chunk_starts[index])
            if chunk_len <= 0:
                raise ValueError(
                    f"DeepSeekV4 prefill chunk_lens[{index}] must be positive, got {chunk_len}"
                )
            if seq_len < 0:
                raise ValueError(
                    f"DeepSeekV4 prefill seq_lens[{index}] must be non-negative, got {seq_len}"
                )
            if chunk_start < 0:
                raise ValueError(
                    f"DeepSeekV4 prefill chunk_starts[{index}] must be non-negative, got {chunk_start}"
                )
            expected_seq_len = chunk_start + chunk_len
            if seq_len != expected_seq_len:
                raise ValueError(
                    f"DeepSeekV4 prefill seq_lens[{index}]={seq_len} must equal "
                    f"chunk_starts[{index}]={chunk_start} + chunk_lens[{index}]={chunk_len}"
                )
            if chunk_offset != packed_tokens:
                raise ValueError(
                    f"DeepSeekV4 prefill chunk_offsets[{index}]={chunk_offset} must equal "
                    f"packed token offset {packed_tokens}"
                )
            packed_tokens += chunk_len

        if batch.token_ids.ndim != 1:
            raise ValueError(
                "DeepSeekV4 prefill token_ids must be 1-D packed, "
                f"got shape={tuple(batch.token_ids.shape)}"
            )
        token_extent = int(batch.token_ids.shape[0])
        if token_extent != packed_tokens:
            raise ValueError(
                f"DeepSeekV4 prefill token_ids contains {token_extent} packed tokens, "
                f"expected {packed_tokens} from chunk_lens"
            )
        if batch.input_embeddings is not None:
            if batch.input_embeddings.ndim != 2:
                raise ValueError(
                    "DeepSeekV4 prefill input_embeddings must have shape [tokens, hidden], "
                    f"got shape={tuple(batch.input_embeddings.shape)}"
                )
            embedding_extent = int(batch.input_embeddings.shape[0])
            if embedding_extent != packed_tokens:
                raise ValueError(
                    f"DeepSeekV4 prefill input_embeddings contains {embedding_extent} packed rows, "
                    f"expected {packed_tokens} from chunk_lens"
                )

    def prepare_prefill_inputs(self, model: RuntimeModel, batch: PrefillBatch) -> DeepSeekV4PreparedPrefillInputs:
        """Build DeepSeekV4 prefill host inputs for the current scheduler chunk."""
        layout = self._compiled.layout
        request_count = len(batch.request_ids)
        if request_count <= 0 or request_count > layout.ranks * layout.prefill_batch:
            raise ValueError(
                "DeepSeekV4 prefill exceeds the fixed global capacity "
                f"{layout.ranks * layout.prefill_batch}, got {request_count}"
            )
        self._validate_prefill_batch_metadata(batch, request_count)
        builder = self._require_input_builder()
        if len(batch.cache_partitions) != request_count:
            raise ValueError("DeepSeekV4 prefill requires one cache partition per request")
        ranks = tuple(int(rank) for rank in batch.cache_partitions)
        if min(ranks) < 0 or max(ranks) >= layout.ranks:
            raise ValueError(f"DeepSeekV4 prefill cache partitions must be in [0, {layout.ranks - 1}]")
        assignment = self._prefill_assignment(ranks)
        local_rows = assignment.local_rows
        group_rows = self._normalize_group_block_ids(
            batch.block_ids_by_group,
            actual_batch=request_count,
        )

        actual_tokens_by_request = []
        kernel_embeddings_by_request = []
        input_ids_by_request = []
        position_ids_by_request = []
        positions_by_request = []

        if batch.input_embeddings is None:
            raise ValueError("DeepSeek V4 prefill requires host input embeddings")
        kernel_tokens = self._prefill_kernel_tokens(
            max(int(tokens) for tokens in batch.chunk_lens),
            runtime=model.runtime,
        )

        # Prefill writes directly into the scheduler-owned rank-local physical
        # pools. The same worker-resident shards are passed to decode, so no
        # parent-side cache snapshot or handoff is required.
        for index in range(request_count):
            actual_tokens = batch.chunk_lens[index]
            chunk_offset = batch.chunk_offsets[index]
            chunk_start = batch.chunk_starts[index]
            positions = tuple(range(chunk_start, chunk_start + actual_tokens))
            if positions[-1] >= model.runtime.max_seq_len:
                raise ValueError(
                    f"prefill position {positions[-1]} exceeds max_seq_len={model.runtime.max_seq_len}"
                )
            chunk_end = chunk_offset + actual_tokens
            embeddings = batch.input_embeddings[chunk_offset:chunk_end].to(torch.float32).cpu()
            token_ids = batch.token_ids[chunk_offset:chunk_end].detach().cpu().to(torch.long)
            kernel_positions = self._prefill_kernel_positions(
                positions,
                kernel_tokens=kernel_tokens,
                max_seq_len=model.runtime.max_seq_len,
            )
            kernel_embeddings_by_request.append(self._padded_rows(embeddings, kernel_tokens))
            input_ids_by_request.append(self._padded_vector(token_ids, kernel_tokens, dtype=torch.long))
            position_ids_by_request.append(self._prefill_position_ids(kernel_positions, kernel_tokens))
            positions_by_request.append(positions)
            actual_tokens_by_request.append(actual_tokens)

        group_ids = {
            name: tuple(groups[name] for groups in group_rows)
            for name in DEEPSEEK_V4_CACHE_GROUP_NAMES
        }
        ori_block_tables = self.cache_metadata.ring_block_table_from_ids(
            group_ids["ori"],
            max_blocks=layout.prefill_ori_max_blocks,
        )
        hca_cmp_block_tables = self.cache_metadata.ring_block_table_from_ids(
            group_ids["cmp_c128"],
            max_blocks=layout.prefill_cmp_max_blocks,
        )
        csa_cmp_block_tables = self.cache_metadata.ring_block_table_from_ids(
            group_ids["cmp_c4"],
            max_blocks=layout.prefill_cmp_max_blocks,
        )
        idx_block_tables = self.cache_metadata.ring_block_table_from_ids(
            group_ids["idx"],
            max_blocks=layout.prefill_idx_max_blocks,
        )
        hca_state_block_tables = self.cache_metadata.ring_block_table_from_ids(
            group_ids["hca_state"],
            max_blocks=layout.prefill_hca_state_max_blocks,
        )
        csa_state_block_tables = self.cache_metadata.ring_block_table_from_ids(
            group_ids["csa_state"],
            max_blocks=layout.prefill_csa_state_max_blocks,
        )
        csa_inner_state_block_tables = self.cache_metadata.ring_block_table_from_ids(
            group_ids["csa_inner_state"],
            max_blocks=layout.prefill_csa_inner_state_max_blocks,
        )

        ori_slot_mappings = self._pad_prefill_mappings(
            self.cache_metadata.paged_decode_slot_mapping_from_ids(
                group_ids["ori"],
                positions_by_request,
            ),
            kernel_tokens,
        )
        hca_cmp_slot_mappings = self._pad_prefill_mappings(
            self.cache_metadata.compressed_slot_mapping_from_ids(
                group_ids["cmp_c128"],
                positions_by_request,
                block_size=layout.block_size,
                compress_ratio=128,
            ),
            kernel_tokens,
        )
        hca_state_slot_mappings = self._pad_prefill_mappings(
            self.cache_metadata.state_slot_mapping_from_ids(
                group_ids["hca_state"],
                positions_by_request,
                state_block_size=layout.c128_state_block_size,
            ),
            kernel_tokens,
        )
        csa_cmp_slot_mappings = self._pad_prefill_mappings(
            self.cache_metadata.compressed_slot_mapping_from_ids(
                group_ids["cmp_c4"],
                positions_by_request,
                block_size=layout.block_size,
                compress_ratio=4,
            ),
            kernel_tokens,
        )
        csa_idx_slot_mappings = self._pad_prefill_mappings(
            self.cache_metadata.compressed_slot_mapping_from_ids(
                group_ids["idx"],
                positions_by_request,
                block_size=layout.block_size,
                compress_ratio=4,
            ),
            kernel_tokens,
        )
        csa_state_slot_mappings = self._pad_prefill_mappings(
            self.cache_metadata.state_slot_mapping_from_ids(
                group_ids["csa_state"],
                positions_by_request,
                state_block_size=layout.c4_state_block_size,
            ),
            kernel_tokens,
        )
        csa_inner_state_slot_mappings = self._pad_prefill_mappings(
            self.cache_metadata.state_slot_mapping_from_ids(
                group_ids["csa_inner_state"],
                positions_by_request,
                state_block_size=layout.c4_state_block_size,
            ),
            kernel_tokens,
        )

        num_tokens_per_owner = torch.zeros(
            (layout.prefill_batch, layout.ranks),
            dtype=torch.int32,
        )
        logit_row_indices = torch.full(
            (layout.ranks, layout.prefill_batch, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS),
            -1,
            dtype=torch.int32,
        )
        for rank, local_row, actual_tokens in zip(
            ranks,
            local_rows,
            actual_tokens_by_request,
            strict=True,
        ):
            num_tokens_per_owner[local_row, rank] = actual_tokens
            logit_row_indices[rank, local_row, 0] = actual_tokens - 1

        return DeepSeekV4PreparedPrefillInputs(
            request_ids=tuple(batch.request_ids),
            ranks=ranks,
            local_rows=local_rows,
            actual_tokens=tuple(actual_tokens_by_request),
            kernel_tokens=kernel_tokens,
            x_hc=builder.prefill_x_hc(
                kernel_embeddings_by_request,
                ranks=ranks,
                local_rows=local_rows,
                token_rows=kernel_tokens,
            ),
            input_ids=self._rank_slot_scatter(input_ids_by_request, ranks, local_rows),
            position_ids=self._rank_slot_scatter(position_ids_by_request, ranks, local_rows),
            ori_block_table=self._rank_slot_scatter(ori_block_tables, ranks, local_rows),
            ori_slot_mapping=self._rank_slot_scatter_mappings(
                ori_slot_mappings,
                ranks,
                local_rows,
            ),
            hca_cmp_block_table=self._rank_slot_scatter(
                hca_cmp_block_tables,
                ranks,
                local_rows,
            ),
            csa_cmp_block_table=self._rank_slot_scatter(
                csa_cmp_block_tables,
                ranks,
                local_rows,
            ),
            idx_block_table=self._rank_slot_scatter(idx_block_tables, ranks, local_rows),
            hca_compress_state_block_table=self._rank_slot_scatter(
                hca_state_block_tables,
                ranks,
                local_rows,
            ),
            csa_compress_state_block_table=self._rank_slot_scatter(
                csa_state_block_tables,
                ranks,
                local_rows,
            ),
            csa_inner_compress_state_block_table=self._rank_slot_scatter(
                csa_inner_state_block_tables,
                ranks,
                local_rows,
            ),
            hca_cmp_slot_mapping=self._rank_slot_scatter_mappings(
                hca_cmp_slot_mappings,
                ranks,
                local_rows,
            ),
            hca_state_slot_mapping=self._rank_slot_scatter_mappings(
                hca_state_slot_mappings,
                ranks,
                local_rows,
            ),
            csa_cmp_slot_mapping=self._rank_slot_scatter_mappings(
                csa_cmp_slot_mappings,
                ranks,
                local_rows,
            ),
            csa_idx_slot_mapping=self._rank_slot_scatter_mappings(
                csa_idx_slot_mappings,
                ranks,
                local_rows,
            ),
            csa_state_slot_mapping=self._rank_slot_scatter_mappings(
                csa_state_slot_mappings,
                ranks,
                local_rows,
            ),
            csa_inner_state_slot_mapping=self._rank_slot_scatter_mappings(
                csa_inner_state_slot_mappings,
                ranks,
                local_rows,
            ),
            num_tokens_per_owner=num_tokens_per_owner,
            logit_row_indices=logit_row_indices,
        )

    def prepare_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        buffer_slot: int,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Prepare an isolated decode metadata snapshot while the device is busy.

        Token IDs and exact speculative sequence lengths are placeholders. The
        fused MTP kernel replaces active rows from stable device-state slots.
        """
        assignment = self._decode_assignment(batch)
        fused_mtp = self._compiled.num_speculative_tokens == 1
        if fused_mtp:
            if not batch.allow_device_greedy_sampling:
                raise RuntimeError("DeepSeekV4 fused MTP decode requires device sampling")
            for request_id, rank in zip(batch.request_ids, assignment.ranks, strict=True):
                self._reserve_mtp_request_state(request_id, rank)
        actual_batch = len(batch.request_ids)
        active_seq = self._compiled.layout.decode_seq if fused_mtp else 1
        if not fused_mtp and max(assignment.per_rank_counts) > 1:
            raise ValueError(
                "DeepSeekV4 non-MTP decode supports at most one request per DP rank"
            )
        # Exact positions depend on step N acceptance. Use valid filler values;
        # the fused kernel binds active rows from device state before main decode.
        positions = tuple((0,) * self._compiled.layout.decode_seq for _ in range(actual_batch))
        placeholder_rows = torch.zeros(
            (actual_batch, self._compiled.layout.decode_seq), dtype=torch.long
        )
        with profile_span("DeepSeekV4ModelRunner.decode.prepare_early", cat="executor"):
            prepared = self._prepare_decode_inputs(
                model,
                batch,
                assignment=assignment,
                active_seq=active_seq,
                positions=positions,
                token_rows=placeholder_rows,
                x_hc=None,
                buffer_slot=buffer_slot,
            )
            prepared = self._prepare_mtp_device_state_descriptors(prepared)
            if (
                fused_mtp
                and prepared.mtp_tail_slot_ids is not None
                and self._l3_shared_buffers_ready
            ):
                prepared = self._bind_prepared_mtp_dispatch(
                    prepared,
                    model.config.hidden_size,
                    model.config.vocab_size,
                )
            if fused_mtp and self._l3_shared_buffers_ready:
                if prepared.mtp_tail_slot_ids is None or prepared.dispatch_args is None:
                    raise RuntimeError("DeepSeekV4 MTP prepare did not produce a complete dispatch")
            return prepared

    def run_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> DecodeResult:
        """Attach stable state descriptors and execute an early-prepared slot."""
        if not isinstance(prepared, DeepSeekV4PreparedDecodeInputs):
            raise TypeError("DeepSeekV4 prepared decode has an unexpected type")
        if tuple(batch.request_ids) != prepared.request_ids:
            raise ValueError("prepared decode request order changed before execution")
        if tuple(int(rank) for rank in batch.cache_partitions) != prepared.ranks:
            raise ValueError("prepared decode cache partitions changed before execution")
        # Production preflight prepares every shared tensor before the L3
        # worker forks.  Keep a guarded late fallback for shape-only/unit-test
        # runners, but do not put the idempotent readiness walk on every steady
        # decode step.
        if not self._l3_shared_buffers_ready:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.ensure_ready_late",
                cat="executor",
            ):
                self._ensure_l3_shared_buffers(model)
        if self._compiled.num_speculative_tokens == 1:
            return self.reclaim_prepared_decode(
                self.dispatch_prepared_decode(model, batch, prepared)
            )
        return self._run_autoregressive_decode(model, batch, prepared=prepared)

    def dispatch_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> object:
        """Submit PyPTO device work, leaving completion and reclaim to another lane."""
        if self._compiled.num_speculative_tokens != 1:
            raise RuntimeError("split decode reclaim requires fused DeepSeekV4 MTP")
        if not isinstance(prepared, DeepSeekV4PreparedDecodeInputs):
            raise TypeError("DeepSeekV4 prepared decode has an unexpected type")
        if prepared.dispatch_args is None:
            raise RuntimeError("DeepSeekV4 fused decode was not fully bound during prepare")
        if tuple(batch.request_ids) != prepared.request_ids:
            raise ValueError("prepared decode request order changed before execution")
        if tuple(int(rank) for rank in batch.cache_partitions) != prepared.ranks:
            raise ValueError("prepared decode cache partitions changed before execution")
        states = tuple(
            self._require_mtp_request_state(request_id)
            for request_id in prepared.request_ids
        )
        for request_id, state in zip(prepared.request_ids, states, strict=True):
            if not state.device_state_initialized:
                raise RuntimeError(
                    f"DeepSeekV4 MTP device state was not finalized during prefill for {request_id!r}"
                )
        return self._launch_prepared_mtp_decode(prepared, states)

    def reclaim_prepared_decode(self, pending: object) -> DecodeResult:
        """Read ping-ponged sampled outputs after a fused dispatch completes."""
        if not isinstance(pending, _DeepSeekV4PendingMtpDecode):
            raise TypeError("DeepSeekV4 pending decode has an unexpected type")
        return self._reclaim_mtp_decode(pending)

    def prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Build inputs for the single-token autoregressive decode flow."""
        assignment = self._decode_assignment(batch)
        if self._compiled.layout.decode_seq != 1 and max(assignment.per_rank_counts) > 1:
            raise ValueError(
                "DeepSeekV4 non-MTP decode supports at most one request per DP rank; "
                "the fixed S=2 kernel can expose only one cache-safe active token per rank"
            )
        actual_batch = len(batch.request_ids)
        positions = self._autoregressive_decode_positions(batch, actual_batch)
        token_rows = self._autoregressive_decode_token_rows(batch.token_ids, actual_batch)
        with profile_span("DeepSeekV4ModelRunner.decode.prepare_metadata_sources", cat="executor"):
            return self._prepare_decode_inputs(
                model,
                batch,
                assignment=assignment,
                active_seq=1,
                positions=positions,
                token_rows=token_rows,
                x_hc=None,
            )

    def prepare_mtp_target_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        token_rows: torch.Tensor,
        positions: Sequence[Sequence[int]],
        active_width: int,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Build one fixed-width target-verification chunk."""
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if token_rows.shape != (actual_batch, layout.decode_seq):
            raise ValueError(
                "MTP target token rows must have shape "
                f"{(actual_batch, layout.decode_seq)}, got {tuple(token_rows.shape)}"
            )
        if not 1 <= int(active_width) <= layout.decode_seq:
            raise ValueError("MTP target active width must fit the fixed decode sequence")
        position_rows = tuple(tuple(int(value) for value in row) for row in positions)
        if len(position_rows) != actual_batch or any(
            len(row) != layout.decode_seq for row in position_rows
        ):
            raise ValueError("MTP target positions must align with the fixed decode sequence")
        assignment = self._decode_assignment(batch)
        if active_width < layout.decode_seq and any(
            count > 1 for count in assignment.per_rank_counts
        ):
            raise ValueError(
                "partial MTP target chunks require at most one request per rank"
            )
        token_rows = token_rows.detach().cpu().to(torch.long)
        with profile_span("DeepSeekV4ModelRunner.decode.build_metadata", cat="executor"):
            prepared = self._prepare_decode_inputs(
                model,
                batch,
                assignment=assignment,
                active_seq=active_width,
                positions=position_rows,
                token_rows=token_rows,
                x_hc=None,
            )
        return prepared

    def _prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        assignment: _DeepSeekV4DecodeAssignment,
        active_seq: int,
        positions: tuple[tuple[int, ...], ...],
        token_rows: torch.Tensor,
        x_hc: torch.Tensor | None,
        buffer_slot: int = 0,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Build mode-independent cache metadata around explicit token rows."""
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        ranks = assignment.ranks
        local_rows = assignment.local_rows
        per_rank_counts = assignment.per_rank_counts
        active_group_ids = self._normalize_group_block_ids(
            batch.block_ids_by_group,
            actual_batch,
        )
        max_position = max(max(row) for row in positions)
        if max_position >= model.runtime.max_seq_len:
            raise ValueError(f"decode position {max_position} exceeds max_seq_len={model.runtime.max_seq_len}")

        self._ensure_decode_buffers(model.config.hidden_size, model.config.vocab_size)
        if not 0 <= buffer_slot < len(self._decode_task_args):
            raise ValueError(f"decode buffer_slot must be 0 or 1, got {buffer_slot}")
        staged = self._decode_task_args[buffer_slot].tensors

        staged["num_tokens_per_owner"].zero_()
        staged["logit_row_indices"].fill_(-1)
        self._stage_decode_dynamic_inputs(
            staged,
            batch,
            assignment=assignment,
            positions=positions,
            token_rows=token_rows,
        )

        padded_group_ids = self._rank_major_decode_group_ids(
            active_group_ids,
            assignment=assignment,
        )

        flat_rows = layout.ranks * layout.decode_batch
        table_specs = (
            ("ori", "block_table", layout.ori_table_max_blocks, True),
            ("cmp_c128", "hca_cmp_block_table", layout.cmp_max_blocks, False),
            ("cmp_c4", "csa_cmp_block_table", layout.cmp_max_blocks, False),
            ("idx", "idx_block_table", layout.idx_max_blocks, False),
            (
                "hca_state",
                "hca_compress_state_block_table",
                layout.prefill_hca_state_max_blocks,
                True,
            ),
            (
                "csa_state",
                "csa_compress_state_block_table",
                layout.prefill_csa_state_max_blocks,
                True,
            ),
            (
                "csa_inner_state",
                "csa_inner_compress_state_block_table",
                layout.prefill_csa_inner_state_max_blocks,
                True,
            ),
        )
        for group_name, tensor_name, max_blocks, repeat in table_specs:
            output = staged[tensor_name].view(flat_rows, max_blocks)
            if group_name == "ori":
                self.cache_metadata.paged_ori_block_table_from_ids(
                    padded_group_ids[group_name],
                    out=output,
                )
            elif repeat:
                self.cache_metadata.ring_block_table_from_ids(
                    padded_group_ids[group_name],
                    max_blocks=max_blocks,
                    out=output,
                )
            else:
                self.cache_metadata.block_table_from_ids(
                    padded_group_ids[group_name],
                    max_blocks=max_blocks,
                    out=output,
                )

        block_counts = staged["block_counts"].view(flat_rows, -1).numpy()
        for group_index, group_name in enumerate(DEEPSEEK_V4_CACHE_GROUP_NAMES):
            block_counts[:, group_index] = np.fromiter(
                (len(ids) for ids in padded_group_ids[group_name]),
                dtype=np.int32,
                count=flat_rows,
            )

        for rank, count in enumerate(per_rank_counts):
            row_count = count * active_seq
            if row_count > layout.decode_tokens:
                raise ValueError(
                    f"rank {rank} requires {row_count} logit rows, capacity is {layout.decode_tokens}"
                )
            staged["num_tokens_per_owner"][rank] = row_count
            if row_count:
                staged["logit_row_indices"][rank, :row_count].copy_(
                    torch.arange(row_count, dtype=torch.int32)
                )

        return DeepSeekV4PreparedDecodeInputs(
            request_ids=tuple(batch.request_ids),
            ranks=ranks,
            local_rows=tuple(local_rows),
            per_rank_counts=per_rank_counts,
            actual_batch=actual_batch,
            x_hc=x_hc,
            input_ids=staged["input_ids"],
            position_ids=staged["position_ids"],
            kv_seq_lens=staged["kv_seq_lens"],
            block_table=staged["block_table"],
            hca_cmp_block_table=staged["hca_cmp_block_table"],
            csa_cmp_block_table=staged["csa_cmp_block_table"],
            idx_block_table=staged["idx_block_table"],
            hca_compress_state_block_table=staged["hca_compress_state_block_table"],
            csa_compress_state_block_table=staged["csa_compress_state_block_table"],
            csa_inner_compress_state_block_table=staged[
                "csa_inner_compress_state_block_table"
            ],
            block_counts=staged["block_counts"],
            block_ids_by_group=active_group_ids,
            num_tokens_per_owner=staged["num_tokens_per_owner"],
            logit_row_indices=staged["logit_row_indices"],
            buffer_slot=buffer_slot,
        )

    def _stage_decode_dynamic_inputs(
        self,
        staged: dict[str, torch.Tensor],
        batch: DecodeBatch,
        *,
        assignment: _DeepSeekV4DecodeAssignment,
        positions: tuple[tuple[int, ...], ...],
        token_rows: torch.Tensor,
    ) -> None:
        """Write only token/position/length fields into one execution slot."""
        layout = self._compiled.layout
        input_rows = staged["input_ids"].view(
            layout.ranks, layout.decode_batch, layout.decode_seq
        )
        position_rows = staged["position_ids"].view(
            layout.ranks, layout.decode_batch, layout.decode_seq
        )
        first_tokens = token_rows[0]
        first_positions = torch.tensor(positions[0], dtype=torch.int32)
        input_rows.copy_(
            first_tokens.view(1, 1, layout.decode_seq).expand(
                layout.ranks, layout.decode_batch, layout.decode_seq
            )
        )
        position_rows.copy_(
            first_positions.view(1, 1, layout.decode_seq).expand(
                layout.ranks, layout.decode_batch, layout.decode_seq
            )
        )
        staged["kv_seq_lens"].fill_(int(batch.seq_lens[0].item()))
        for rank, request_indices in enumerate(assignment.indices_by_rank):
            if not request_indices:
                continue
            first_request = request_indices[0]
            first_local_positions = torch.tensor(positions[first_request], dtype=torch.int32)
            input_rows[rank].copy_(
                token_rows[first_request].view(1, layout.decode_seq).expand(
                    layout.decode_batch, layout.decode_seq
                )
            )
            position_rows[rank].copy_(
                first_local_positions.view(1, layout.decode_seq).expand(
                    layout.decode_batch, layout.decode_seq
                )
            )
            staged["kv_seq_lens"][rank].fill_(int(batch.seq_lens[first_request].item()))
            for local_row, request_index in enumerate(request_indices):
                input_rows[rank, local_row].copy_(token_rows[request_index])
                position_rows[rank, local_row].copy_(
                    torch.tensor(positions[request_index], dtype=torch.int32)
                )
                staged["kv_seq_lens"][rank, local_row] = batch.seq_lens[request_index]

        self._stage_sampling_inputs(
            staged,
            batch,
            assignment=assignment,
            positions=positions,
        )

    def _stage_sampling_inputs(
        self,
        staged: dict[str, torch.Tensor],
        batch: DecodeBatch,
        *,
        assignment: _DeepSeekV4DecodeAssignment,
        positions: tuple[tuple[int, ...], ...],
    ) -> None:
        """Stage per-logit-row controls for the integrated DSV4 sampler."""
        if self._compiled.num_speculative_tokens == 1:
            return
        layout = self._compiled.layout
        params = batch.sampling_params
        if params and len(params) != len(batch.request_ids):
            raise ValueError("decode sampling params must align with request IDs")
        default_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=None)

        staged["sampling_temperatures"].zero_()
        staged["sampling_top_ks"].fill_(DEEPSEEK_V4_VOCAB_SIZE)
        staged["sampling_seeds"].zero_()
        staged["sampling_positions"].zero_()
        for rank, request_indices in enumerate(assignment.indices_by_rank):
            for local_row, request_index in enumerate(request_indices):
                request_params = params[request_index] if params else default_params
                top_k = request_params.top_k
                if top_k is None or top_k <= 0:
                    top_k = DEEPSEEK_V4_VOCAB_SIZE
                seed = request_params.seed
                if seed is None:
                    seed = zlib.crc32(batch.request_ids[request_index].encode())
                seed = int(seed) & 0x3FFFFFFF
                row_start = local_row * layout.decode_seq
                row_stop = row_start + layout.decode_seq
                staged["sampling_temperatures"][rank, row_start:row_stop].fill_(
                    float(request_params.temperature)
                )
                staged["sampling_top_ks"][rank, row_start:row_stop].fill_(
                    min(int(top_k), DEEPSEEK_V4_VOCAB_SIZE)
                )
                staged["sampling_seeds"][rank, row_start:row_stop].fill_(seed)
                staged["sampling_positions"][rank, row_start:row_stop].copy_(
                    torch.tensor(positions[request_index], dtype=torch.int32) + 1
                )

    def _stage_decode_cache_metadata(
        self,
        staged: dict[str, torch.Tensor],
        *,
        assignment: _DeepSeekV4DecodeAssignment,
        positions: tuple[tuple[int, ...], ...],
        active_group_ids: tuple[dict[str, tuple[int, ...]], ...],
    ) -> None:
        """Stage the raw metadata consumed by the current main decode ABI."""
        layout = self._compiled.layout
        if any(len(row) != layout.decode_seq for row in positions):
            raise ValueError("decode positions must match the compiled sequence width")

        for rank, request_indices in enumerate(assignment.indices_by_rank):
            local_groups = [active_group_ids[index] for index in request_indices]

            padded_group_ids: dict[str, tuple[tuple[int, ...], ...]] = {}
            for name in DEEPSEEK_V4_CACHE_GROUP_NAMES:
                if local_groups:
                    padded_group_ids[name] = self._pad_group_block_ids(
                        [groups[name] for groups in local_groups],
                        group_name=name,
                        kernel_rows=layout.decode_batch,
                    )
                else:
                    padded_group_ids[name] = self._scratch_group_block_ids(
                        group_name=name,
                        kernel_rows=layout.decode_batch,
                    )

            values: dict[str, torch.Tensor] = {
                "block_table": self.cache_metadata.paged_ori_block_table_from_ids(
                    padded_group_ids["ori"]
                ),
                "hca_cmp_block_table": self.cache_metadata.block_table_from_ids(
                    padded_group_ids["cmp_c128"],
                    max_blocks=layout.cmp_max_blocks,
                ),
                "csa_cmp_block_table": self.cache_metadata.block_table_from_ids(
                    padded_group_ids["cmp_c4"],
                    max_blocks=layout.cmp_max_blocks,
                ),
                "idx_block_table": self.cache_metadata.block_table_from_ids(
                    padded_group_ids["idx"],
                    max_blocks=layout.idx_max_blocks,
                ),
                "hca_compress_state_block_table": self.cache_metadata.ring_block_table_from_ids(
                    padded_group_ids["hca_state"],
                    max_blocks=layout.prefill_hca_state_max_blocks,
                ),
                "csa_compress_state_block_table": self.cache_metadata.ring_block_table_from_ids(
                    padded_group_ids["csa_state"],
                    max_blocks=layout.prefill_csa_state_max_blocks,
                ),
                "csa_inner_compress_state_block_table": (
                    self.cache_metadata.ring_block_table_from_ids(
                        padded_group_ids["csa_inner_state"],
                        max_blocks=layout.prefill_csa_inner_state_max_blocks,
                    )
                ),
            }

            values["block_counts"] = torch.tensor(
                [
                    [
                        len(padded_group_ids[name][row])
                        for name in DEEPSEEK_V4_CACHE_GROUP_NAMES
                    ]
                    for row in range(layout.decode_batch)
                ],
                dtype=torch.int32,
            )

            for name, value in values.items():
                copy_shared(
                    staged[name][rank],
                    value,
                    name=f"decode_{name}_rank{rank}",
                )

    @staticmethod
    def _normalize_group_block_ids(
        rows: Sequence[dict[str, list[int]]],
        actual_batch: int,
    ) -> tuple[dict[str, tuple[int, ...]], ...]:
        """Validate and normalize grouped scheduler metadata for active rows."""
        if not rows:
            raise ValueError("DeepSeekV4 requires grouped KV block IDs")
        if len(rows) != actual_batch:
            raise ValueError(
                f"grouped KV metadata has {len(rows)} rows, expected decode batch {actual_batch}"
            )
        required = DEEPSEEK_V4_CACHE_GROUP_NAMES
        normalized = []
        for row_index, row in enumerate(rows):
            missing = [name for name in required if not row.get(name)]
            if missing:
                raise ValueError(
                    f"decode row {row_index} is missing grouped KV blocks: {', '.join(missing)}"
                )
            normalized.append(
                {name: tuple(int(block_id) for block_id in row[name]) for name in required}
            )
        return tuple(normalized)

    def _rank_major_decode_group_ids(
        self,
        active_group_ids: tuple[dict[str, tuple[int, ...]], ...],
        *,
        assignment: _DeepSeekV4DecodeAssignment,
    ) -> dict[str, tuple[tuple[int, ...], ...]]:
        """Arrange active and scratch block IDs in flattened rank-major order."""
        layout = self._compiled.layout
        flat_rows = layout.ranks * layout.decode_batch
        if len(active_group_ids) != len(assignment.ranks):
            raise ValueError("decode cache metadata must align with the rank assignment")

        grouped: dict[str, tuple[tuple[int, ...], ...]] = {}
        for group_name in DEEPSEEK_V4_CACHE_GROUP_NAMES:
            scratch = self._scratch_group_block_ids(
                group_name=group_name,
                kernel_rows=layout.decode_batch,
            )
            rows = [scratch[row % layout.decode_batch] for row in range(flat_rows)]
            allocator_blocks = self._cache_group_num_blocks[group_name]
            for request_index, (rank, local_row) in enumerate(
                zip(assignment.ranks, assignment.local_rows, strict=True)
            ):
                ids = active_group_ids[request_index][group_name]
                if len(ids) != len(set(ids)):
                    raise ValueError("a grouped KV row must not repeat physical blocks")
                if any(block_id < 0 or block_id >= allocator_blocks for block_id in ids):
                    raise ValueError(
                        f"grouped KV block IDs must be in [0, {allocator_blocks}); "
                        f"[{allocator_blocks}, {allocator_blocks + layout.decode_batch}) is reserved "
                        "for kernel padding"
                    )
                rows[rank * layout.decode_batch + local_row] = ids
            grouped[group_name] = tuple(rows)
        return grouped

    def _pad_group_block_ids(
        self,
        active_rows: Sequence[Sequence[int]],
        *,
        group_name: str,
        kernel_rows: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Pad inactive rows with pages at the tail of a dynamic cache pool."""
        if not active_rows or len(active_rows) > kernel_rows:
            raise ValueError("active grouped KV rows must fit the kernel batch")
        normalized = [tuple(int(block_id) for block_id in row) for row in active_rows]
        if any(not row for row in normalized):
            raise ValueError("active grouped KV rows must not be empty")
        if any(len(row) != len(set(row)) for row in normalized):
            raise ValueError("a grouped KV row must not repeat physical blocks")
        used = [block_id for row in normalized for block_id in row]
        # Different requests may legitimately share immutable prefix-cache
        # pages. The cache manager detaches a request with copy-on-write before
        # any shared rolling page is overwritten, so only duplicates within one
        # request's own table are invalid here.
        scratch = self._scratch_group_block_ids(
            group_name=group_name,
            kernel_rows=kernel_rows,
        )
        try:
            allocator_blocks = self._cache_group_num_blocks[group_name]
        except KeyError as exc:
            raise ValueError(f"unknown DeepSeekV4 cache group: {group_name}") from exc
        if any(block_id < 0 or block_id >= allocator_blocks for block_id in used):
            raise ValueError(
                f"grouped KV block IDs must be in [0, {allocator_blocks}); "
                f"[{allocator_blocks}, {allocator_blocks + kernel_rows}) is reserved "
                "for kernel padding"
            )
        padded = list(normalized)
        # Attention and compressor cache writes are fixed-B and are not fully
        # gated by num_tokens. Give every inactive row a distinct reserved page
        # instead of mirroring a live request's metadata.
        padded.extend(scratch[: kernel_rows - len(normalized)])
        return tuple(padded)

    def _scratch_group_block_ids(
        self,
        *,
        group_name: str,
        kernel_rows: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return one isolated physical scratch page for every fixed kernel row."""
        if kernel_rows <= 0:
            raise ValueError("kernel_rows must be positive")
        try:
            first = self._cache_group_num_blocks[group_name]
        except KeyError as exc:
            raise ValueError(f"unknown DeepSeekV4 cache group: {group_name}") from exc
        if self._physical_cache_num_blocks(group_name) < first + kernel_rows:
            raise ValueError("cache pool must provide one scratch page per kernel row")
        return tuple((first + row,) for row in range(kernel_rows))

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceTensor:
        raise NotImplementedError("DeepSeekV4 uses model-specific cache pools, not generic KV tensors")

    def _free_kv_cache_tensor(self, tensor: DeviceTensor) -> None:
        return None

    def _track_pending_mtp_dispatch(
        self,
        buffer_slot: int,
        dispatch: PendingL3Dispatch,
    ) -> None:
        """Keep a fused dispatch visible until reclaim or a prefill barrier."""
        with self._pending_mtp_dispatch_lock:
            if buffer_slot in self._pending_mtp_dispatches:
                raise RuntimeError(
                    f"DeepSeekV4 MTP decode buffer slot {buffer_slot} was reused before completion"
                )
            self._pending_mtp_dispatches[buffer_slot] = dispatch

    def _forget_pending_mtp_dispatch(
        self,
        buffer_slot: int,
        dispatch: PendingL3Dispatch,
    ) -> None:
        """Drop a completed dispatch without removing a newer slot owner."""
        with self._pending_mtp_dispatch_lock:
            if self._pending_mtp_dispatches.get(buffer_slot) is dispatch:
                del self._pending_mtp_dispatches[buffer_slot]

    def _wait_for_pending_mtp_dispatches(self) -> None:
        """Fence shared prefill staging behind earlier fused decode work."""
        with self._pending_mtp_dispatch_lock:
            pending = tuple(sorted(self._pending_mtp_dispatches.items()))
        if not pending:
            return

        first_error: BaseException | None = None
        with profile_span(
            "DeepSeekV4ModelRunner.prefill.wait_pending_decode",
            cat="executor",
            args={"buffer_slots": [buffer_slot for buffer_slot, _ in pending]},
        ):
            for buffer_slot, dispatch in pending:
                try:
                    dispatch.wait()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                finally:
                    self._forget_pending_mtp_dispatch(buffer_slot, dispatch)
        if first_error is not None:
            raise first_error

    def run_prefill(self, model, batch: PrefillBatch) -> PrefillResult:
        """Run one B4 prefill dispatch across all active rank-local slots."""
        if self._compiled.prefill is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        self._wait_for_pending_mtp_dispatches()
        with profile_span("DeepSeekV4ModelRunner.prefill.prepare", cat="executor"):
            with profile_span("DeepSeekV4ModelRunner.prefill.ensure_l3_shared_buffers", cat="executor"):
                self._ensure_l3_shared_buffers(model)
            with profile_span("DeepSeekV4ModelRunner.prefill.prepare_inputs", cat="executor"):
                inputs = self.prepare_prefill_inputs(model, batch)
        with profile_span(
            "DeepSeekV4ModelRunner.prefill.prepare_fwd_args",
            cat="executor",
            args={
                "actual_tokens": max(inputs.actual_tokens),
                "active_slots": max(inputs.local_rows) + 1,
            },
        ):
            self._stage_prefill_fwd_inputs(inputs)
            self._prefill_task_args.clear_outputs()
            args = self._prefill_fwd_args(inputs.kernel_tokens)
            pre_hc_hidden_buffer = self._prefill_task_args.tensors["pre_hc_hidden_out"]
            logits_buffer = self._prefill_task_args.tensors["logits"]
        try:
            with profile_span(
                "DeepSeekV4ModelRunner.prefill.l3_dispatch",
                cat="executor",
                args={
                    "actual_tokens": max(inputs.actual_tokens),
                    "active_slots": max(inputs.local_rows) + 1,
                },
            ):
                self._run_l3(
                    self._require_prefill_callable(),
                    *args,
                )
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 packed prefill dispatch failed "
                f"(tokens={inputs.actual_tokens}, ranks={inputs.ranks})"
            ) from exc
        self._prefill_completion(inputs, pre_hc_hidden_buffer)

        logits = torch.stack(
            tuple(
                logits_buffer[rank, local_row, 0]
                for rank, local_row in zip(
                    inputs.ranks,
                    inputs.local_rows,
                    strict=True,
                )
            ),
        ).float()
        return PrefillResult(last_hidden=None, logits=logits)

    def finalize_prefill(
        self,
        request_ids: Sequence[str],
        sampled_token_ids: Sequence[int],
        sampling_params: Sequence[SamplingParams] | None = None,
    ) -> None:
        """Run terminal MTP prefill and publish persistent decode state."""
        if not self._compiled.num_speculative_tokens:
            return
        if len(request_ids) != len(sampled_token_ids):
            raise ValueError("MTP prefill requires one sampled token per request")
        if sampling_params is None:
            sampling_params = [
                SamplingParams(temperature=0.0, top_p=1.0, top_k=None)
                for _ in request_ids
            ]
        if len(request_ids) != len(sampling_params):
            raise ValueError("MTP sampling params must align with request IDs")
        with profile_span(
            "DeepSeekV4ModelRunner.prefill.mtp_initialize",
            cat="executor",
            args={"batch_size": len(request_ids)},
        ):
            for request_id, token_id, params in zip(
                request_ids,
                sampled_token_ids,
                sampling_params,
                strict=True,
            ):
                state = self._require_mtp_request_state(request_id)
                if state.draft_token_id is None:
                    self._initialize_mtp_draft(request_id, state, int(token_id), params)

    def run_decode(self, model, batch: DecodeBatch) -> DecodeResult:
        """Dispatch to the decode flow selected when the model was compiled."""
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        with profile_span("DeepSeekV4ModelRunner.decode.prepare", cat="executor"):
            self._ensure_l3_shared_buffers(model)
        return self._decode_flow(model, batch)

    def _run_autoregressive_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        prepared: DeepSeekV4PreparedDecodeInputs | None = None,
    ) -> DecodeResult:
        """Run the single-token autoregressive decode flow."""
        if prepared is None:
            inputs = self.prepare_decode_inputs(model, batch)
        else:
            assignment = _DeepSeekV4DecodeAssignment(
                ranks=prepared.ranks,
                local_rows=prepared.local_rows,
                per_rank_counts=prepared.per_rank_counts,
                indices_by_rank=self._indices_by_rank(prepared.ranks),
            )
            positions = self._autoregressive_decode_positions(batch, len(batch.request_ids))
            token_rows = self._autoregressive_decode_token_rows(
                batch.token_ids, len(batch.request_ids)
            )
            self._stage_decode_dynamic_inputs(
                self._decode_task_args[prepared.buffer_slot].tensors,
                batch,
                assignment=assignment,
                positions=positions,
                token_rows=token_rows,
            )
            self._stage_decode_cache_metadata(
                self._decode_task_args[prepared.buffer_slot].tensors,
                assignment=assignment,
                positions=positions,
                active_group_ids=prepared.block_ids_by_group,
            )
            inputs = prepared
        output = self._execute_main_decode(
            model,
            inputs,
            active_seq=1,
        )
        if batch.allow_device_greedy_sampling:
            sampled_token_ids = torch.stack(
                tuple(
                    output.sampled_ids[rank, local_row, 0]
                    for rank, local_row in zip(
                        output.inputs.ranks,
                        output.inputs.local_rows,
                        strict=True,
                    )
                )
            ).to(torch.int32)
            return DecodeResult(
                hidden_states=None,
                logits=None,
                sampled_token_ids=sampled_token_ids,
            )
        logits = torch.stack(
            tuple(
                output.logits[rank, local_row]
                for rank, local_row in zip(
                    output.inputs.ranks,
                    output.inputs.local_rows,
                    strict=True,
                )
            )
        ).float()
        return DecodeResult(hidden_states=None, logits=logits)

    def _verify_mtp_drafts(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        drafts: torch.Tensor,
    ) -> _DeepSeekV4MtpVerification:
        """Verify arbitrary draft depth with fixed-width target chunks."""
        layout = self._compiled.layout
        width = layout.decode_seq
        actual_batch = len(batch.request_ids)
        drafts = drafts.detach().cpu().to(torch.long)
        if drafts.ndim != 2 or drafts.shape[0] != actual_batch:
            raise ValueError(f"MTP drafts must have shape [batch, K], got {tuple(drafts.shape)}")

        current = (
            batch.token_ids[:actual_batch]
            .detach()
            .cpu()
            .to(torch.long)
            .reshape(actual_batch, -1)[:, 0]
        )
        input_sequences = torch.cat((current.reshape(-1, 1), drafts), dim=1)
        base_seq_lens = batch.seq_lens[:actual_batch].detach().cpu().to(torch.int32)
        accepted: list[list[int]] = [[] for _ in range(actual_batch)]
        first_logits: list[torch.Tensor | None] = [None] * actual_batch
        tail_tokens: list[int | None] = [None] * actual_batch
        tail_hidden: list[torch.Tensor | None] = [None] * actual_batch
        tail_positions: list[int | None] = [None] * actual_batch
        pending: dict[int, tuple[int, torch.Tensor, int]] = {}
        active = list(range(actual_batch))
        input_offset = 0

        while active:
            if input_offset:
                continuing = []
                for request_index in active:
                    predicted, previous_hidden, position = pending.pop(request_index)
                    accepted[request_index].append(predicted)
                    if predicted == int(drafts[request_index, input_offset - 1]):
                        continuing.append(request_index)
                    else:
                        tail_tokens[request_index] = predicted
                        tail_hidden[request_index] = previous_hidden
                        tail_positions[request_index] = position
                active = continuing
                if not active:
                    break

            real_width = min(width, input_sequences.shape[1] - input_offset)
            active_rows = input_sequences[active, input_offset : input_offset + real_width]
            if real_width < width:
                active_rows = torch.cat(
                    (active_rows, active_rows[:, -1:].expand(-1, width - real_width)),
                    dim=1,
                )
            position_rows = []
            for request_index in active:
                first_position = int(base_seq_lens[request_index]) - 1 + input_offset
                real_positions = tuple(first_position + offset for offset in range(real_width))
                position_rows.append(real_positions + (real_positions[-1],) * (width - real_width))

            continuing = []
            if real_width == width:
                waves = [list(range(len(active)))]
            else:
                active_batch = self._select_decode_batch_rows(batch, active)
                active_assignment = self._decode_assignment(active_batch)
                waves = [
                    [
                        request_indices[wave]
                        for request_indices in active_assignment.indices_by_rank
                        if wave < len(request_indices)
                    ]
                    for wave in range(max(active_assignment.per_rank_counts))
                ]

            for wave_local_indices in waves:
                wave_request_indices = [active[index] for index in wave_local_indices]
                wave_batch = self._select_decode_batch_rows(batch, wave_request_indices)
                wave_batch.seq_lens = torch.tensor(
                    [
                        int(base_seq_lens[request_index])
                        + input_offset
                        + real_width
                        - 1
                        for request_index in wave_request_indices
                    ],
                    dtype=torch.int32,
                )
                wave_position_rows = [position_rows[index] for index in wave_local_indices]
                output = self._execute_main_decode(
                    model,
                    self.prepare_mtp_target_inputs(
                        model,
                        wave_batch,
                        token_rows=active_rows[wave_local_indices],
                        positions=wave_position_rows,
                        active_width=real_width,
                    ),
                    active_seq=real_width,
                )

                for chunk_index, request_index in enumerate(wave_request_indices):
                    rank = output.inputs.ranks[chunk_index]
                    row_start = output.inputs.local_rows[chunk_index] * width
                    row_logits = output.logits[rank, row_start : row_start + real_width].float()
                    row_predictions = output.sampled_ids[
                        rank, row_start : row_start + real_width, 0
                    ].to(torch.long)
                    if first_logits[request_index] is None:
                        first_logits[request_index] = row_logits[0].clone()

                    rejected = False
                    for offset in range(real_width - 1):
                        predicted = int(row_predictions[offset])
                        accepted[request_index].append(predicted)
                        draft_index = input_offset + offset
                        if predicted != int(drafts[request_index, draft_index]):
                            tail_tokens[request_index] = predicted
                            tail_hidden[request_index] = self._copy_main_pre_hc_row(
                                output.pre_hc_hidden,
                                rank=rank,
                                row=row_start + offset,
                                hidden_size=model.config.hidden_size,
                            )
                            tail_positions[request_index] = (
                                wave_position_rows[chunk_index][offset] + 1
                            )
                            rejected = True
                            break
                    if rejected:
                        continue

                    last_offset = real_width - 1
                    predicted = int(row_predictions[last_offset])
                    previous_hidden = self._copy_main_pre_hc_row(
                        output.pre_hc_hidden,
                        rank=rank,
                        row=row_start + last_offset,
                        hidden_size=model.config.hidden_size,
                    )
                    position = wave_position_rows[chunk_index][last_offset] + 1
                    if input_offset + last_offset == drafts.shape[1]:
                        accepted[request_index].append(predicted)
                        tail_tokens[request_index] = predicted
                        tail_hidden[request_index] = previous_hidden
                        tail_positions[request_index] = position
                    else:
                        pending[request_index] = predicted, previous_hidden, position
                        continuing.append(request_index)

            active = continuing
            input_offset += real_width

        if (
            any(value is None for value in first_logits)
            or any(value is None for value in tail_tokens)
            or any(value is None for value in tail_hidden)
            or any(value is None for value in tail_positions)
        ):
            raise RuntimeError("DeepSeekV4 MTP target verification left incomplete request state")
        return _DeepSeekV4MtpVerification(
            accepted_token_ids=accepted,
            tail_token_ids=torch.tensor(tail_tokens, dtype=torch.long),
            tail_pre_hc_hidden=torch.stack(tail_hidden),
            tail_positions=torch.tensor(tail_positions, dtype=torch.int32),
            first_logits=torch.stack(first_logits).float(),
        )

    def _copy_main_pre_hc_row(
        self,
        source: torch.Tensor | StackedDeviceTensor,
        *,
        rank: int,
        row: int,
        hidden_size: int,
    ) -> torch.Tensor:
        """Copy one target-model pre-HC row needed by recurrent MTP."""
        if not isinstance(source, torch.Tensor):
            raise RuntimeError("MTP verification requires captured host pre-HC output")
        hidden = source[rank, row]
        expected_shape = (self._compiled.layout.hc_mult, int(hidden_size))
        if tuple(hidden.shape) != expected_shape:
            raise ValueError(
                f"captured pre-HC row must have shape {expected_shape}, got {tuple(hidden.shape)}"
            )
        return hidden.detach().cpu().clone()

    def _run_mtp_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        prepared: DeepSeekV4PreparedDecodeInputs | None = None,
    ) -> DecodeResult:
        """Run fused K=1 or synchronous arbitrary-depth MTP decode."""
        if self._compiled.num_speculative_tokens != 1:
            if prepared is not None:
                raise ValueError("arbitrary-depth MTP does not consume async prepared inputs")
            return self._run_chunked_mtp_decode(model, batch)
        pending = self._dispatch_mtp_decode(model, batch, prepared=prepared)
        return self._reclaim_mtp_decode(pending)

    def _dispatch_mtp_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        prepared: DeepSeekV4PreparedDecodeInputs | None = None,
    ) -> _DeepSeekV4PendingMtpDecode:
        """Prepare and run fused MTP device work for synchronous callers."""
        if self._compiled.num_speculative_tokens != 1:
            raise RuntimeError("split decode dispatch requires fused DeepSeekV4 K=1 MTP")
        if not batch.allow_device_greedy_sampling:
            raise RuntimeError("DeepSeekV4 fused MTP decode requires device sampling")
        if prepared is None:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.prepare_inputs_fallback",
                cat="executor",
            ):
                speculative_batch = self._device_state_placeholder_batch(batch)
                inputs = self.prepare_decode(model, speculative_batch, buffer_slot=0)
        else:
            inputs = prepared
        if inputs.mtp_tail_slot_ids is None:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.bind_device_state_late",
                cat="executor",
            ):
                inputs = self._prepare_mtp_device_state_descriptors(inputs, require_ready=True)
        if getattr(inputs, "dispatch_args", None) is None:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.bind_fwd_args_late",
                cat="executor",
                args={"actual_tokens": max(inputs.per_rank_counts) * self._compiled.layout.decode_seq},
            ):
                inputs = self._bind_prepared_mtp_dispatch(
                    inputs,
                    model.config.hidden_size,
                    model.config.vocab_size,
                )
        if getattr(inputs, "dispatch_args", None) is None:
            raise RuntimeError("DeepSeekV4 fused decode arguments were not bound")
        return self.dispatch_prepared_decode(model, batch, inputs)

    def _launch_prepared_mtp_decode(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        states: tuple[_DeepSeekV4MtpRequestState, ...],
    ) -> _DeepSeekV4PendingMtpDecode:
        """Submit a command-lane-complete fused MTP snapshot."""
        if inputs.dispatch_args is None:
            raise RuntimeError("DeepSeekV4 fused decode snapshot is not ready to launch")
        active_tokens = max(inputs.per_rank_counts) * self._compiled.layout.decode_seq
        try:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.l3_dispatch",
                cat="executor",
                args={"actual_tokens": active_tokens, "fused_mtp": True},
            ):
                dispatch = self._submit_l3(
                    self._require_decode_callable(),
                    *inputs.dispatch_args,
                )
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 fused main/MTP decode dispatch failed "
                f"(actual_batch={inputs.actual_batch}, ranks={inputs.ranks})"
            ) from exc
        self._track_pending_mtp_dispatch(inputs.buffer_slot, dispatch)
        ta = self._decode_task_args[inputs.buffer_slot]
        mtp_ta = self._mtp_decode_task_args[inputs.buffer_slot]
        mtp = mtp_ta.tensors
        return _DeepSeekV4PendingMtpDecode(
            dispatch=dispatch,
            inputs=inputs,
            sampled_ids=ta.tensors["sampled_ids"],
            mtp_sampled_ids=mtp["sampled_ids"],
            accepted_counts=mtp["accepted_counts"],
            states=states,
        )

    def _reclaim_mtp_decode(self, pending: _DeepSeekV4PendingMtpDecode) -> DecodeResult:
        """Convert one completed output slot into scheduler-visible tokens."""
        try:
            pending.dispatch.wait()
        finally:
            self._forget_pending_mtp_dispatch(
                pending.inputs.buffer_slot,
                pending.dispatch,
            )
        inputs = pending.inputs
        layout = self._compiled.layout
        decode_seq = layout.decode_seq
        with profile_span("DeepSeekV4ModelRunner.decode.mtp_reclaim", cat="executor"):
            accepted_counts_list = []
            accepted = []
            states = pending.states
            for state, rank, local_row in zip(
                states,
                inputs.ranks,
                inputs.local_rows,
                strict=True,
            ):
                row_start = local_row * decode_seq
                main_tokens = pending.sampled_ids[
                    rank,
                    row_start : row_start + decode_seq,
                    0,
                ].tolist()
                accepted_count = int(pending.accepted_counts[rank, local_row].item())
                if accepted_count not in (1, decode_seq):
                    raise RuntimeError(
                        "DeepSeekV4 MTP device state returned invalid accepted count "
                        f"{accepted_count} for decode_seq={decode_seq}"
                    )
                accepted_counts_list.append(accepted_count)
                accepted.append([int(token) for token in main_tokens[:accepted_count]])
            accepted_counts = tuple(accepted_counts_list)
            self._mtp_proposed_tokens += inputs.actual_batch
            self._mtp_accepted_tokens += sum(count == decode_seq for count in accepted_counts)
            for state, tokens in zip(states, accepted, strict=True):
                state.proposed_tokens += 1
                state.accepted_tokens += int(len(tokens) == decode_seq)
                state.committed_count += len(tokens)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "DeepSeekV4 MTP acceptance progress: accepted=%d proposed=%d rate=%.2f%%",
                    self._mtp_accepted_tokens,
                    self._mtp_proposed_tokens,
                    100.0 * self._mtp_accepted_tokens / self._mtp_proposed_tokens,
                )
        # Keep the next draft mirrored on Host for request statistics and
        # diagnostics. Recurrent token state remains authoritative on device.
        for state, rank, local_row in zip(
            states,
            inputs.ranks,
            inputs.local_rows,
            strict=True,
        ):
            state.draft_token_id = int(
                pending.mtp_sampled_ids[rank, local_row, 0].item()
            )
        return DecodeResult(
            hidden_states=None,
            logits=None,
            accepted_token_ids=accepted,
        )

    def _run_chunked_mtp_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> DecodeResult:
        """Generate and verify the configured number of request-local MTP drafts."""
        if not batch.allow_device_greedy_sampling:
            raise RuntimeError("DeepSeekV4 standalone MTP decode requires device sampling")
        batch = replace(batch, seq_lens=self._correct_mtp_seq_lens(batch))
        num_drafts = self._mtp_draft_count(model, batch)
        drafts = self._propose_mtp_tokens(model, batch, num_drafts=num_drafts)
        verification = self._verify_mtp_drafts(model, batch, drafts)
        accepted = verification.accepted_token_ids
        proposed = len(batch.request_ids) * drafts.shape[1]
        accepted_drafts = sum(len(tokens) - 1 for tokens in accepted)
        self._mtp_proposed_tokens += proposed
        self._mtp_accepted_tokens += accepted_drafts
        for request_id, tokens in zip(batch.request_ids, accepted, strict=True):
            state = self._require_mtp_request_state(request_id)
            state.proposed_tokens += drafts.shape[1]
            state.accepted_tokens += len(tokens) - 1
            state.committed_count += len(tokens)
        logger.info(
            "DeepSeekV4 MTP acceptance progress: accepted=%d proposed=%d rate=%.2f%%",
            self._mtp_accepted_tokens,
            self._mtp_proposed_tokens,
            (
                100.0 * self._mtp_accepted_tokens / self._mtp_proposed_tokens
                if self._mtp_proposed_tokens
                else 0.0
            ),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "DeepSeekV4 MTP step: draft=%s main=%s",
                drafts.tolist(),
                accepted,
            )
        with profile_span(
            "DeepSeekV4ModelRunner.decode.mtp_advance",
            cat="executor",
            args={"accepted_counts": tuple(len(tokens) for tokens in accepted)},
        ):
            next_drafts, next_hidden = self._run_mtp_token_step(
                model,
                batch,
                verification.tail_token_ids,
                verification.tail_pre_hc_hidden,
                verification.tail_positions,
            )
            for index, request_id in enumerate(batch.request_ids):
                state = self._require_mtp_request_state(request_id)
                state.draft_token_id = int(next_drafts[index])
                state.draft_pre_hc_hidden = next_hidden[index].clone()
                state.draft_position = int(verification.tail_positions[index]) + 1
        return DecodeResult(
            hidden_states=None,
            logits=verification.first_logits,
            accepted_token_ids=accepted,
        )

    def _mtp_draft_count(self, model: RuntimeModel, batch: DecodeBatch) -> int:
        """Cap this step's draft depth at the shortest remaining context."""
        remaining_context = min(
            model.runtime.max_seq_len - int(seq_len)
            for seq_len in batch.seq_lens[: len(batch.request_ids)]
        )
        return min(self._compiled.num_speculative_tokens, max(0, remaining_context))

    def _execute_main_decode(
        self,
        model: RuntimeModel,
        prepared: DeepSeekV4PreparedDecodeInputs,
        *,
        active_seq: int,
    ) -> _DeepSeekV4MainDecodeOutput:
        """Run the mode-independent packed main-model decode kernel."""
        with profile_span("DeepSeekV4ModelRunner.decode.prepare_inputs", cat="executor"):
            inputs = prepared
        ta = self._decode_task_args[inputs.buffer_slot]
        active_decode_tokens = max(inputs.per_rank_counts) * active_seq
        # The active hidden/pre-HC rows are pl.Out tensors and the grouped LM
        # head overwrites every logits row, including rows selected with -1.
        # Pre-clearing these reusable buffers only adds host memory bandwidth.
        num_tokens = active_decode_tokens
        with profile_span(
            "DeepSeekV4ModelRunner.decode.prepare_fwd_args",
            cat="executor",
            args={"actual_tokens": num_tokens},
        ):
            args = self._decode_fwd_args(inputs)
        try:
            with profile_span(
                "DeepSeekV4ModelRunner.decode.l3_dispatch",
                cat="executor",
                args={"actual_tokens": num_tokens},
            ):
                self._run_l3(
                    self._require_decode_callable(),
                    *args,
                )
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 packed decode dispatch failed "
                f"(actual_batch={inputs.actual_batch}, ranks={inputs.ranks})"
            ) from exc
        hidden_buffer = ta.tensors["hidden_out"]

        captured_pre_hc: torch.Tensor | StackedDeviceTensor = ta.tensors["pre_hc_hidden_out"]
        if self._compiled.mtp_decode is not None:
            # Arbitrary-depth verification needs selected target pre-HC rows on
            # the host, while recurrent MTP still consumes the device-resident
            # output. Read back complete shards from their allocation base; a
            # raw D2H copy from an offset device pointer is not supported by the
            # distributed control path.
            host_mirror = self._require_main_pre_hc_host_mirror()
            self._shared_l3_worker().copy_stacked_from(
                ta.tensors["pre_hc_hidden_out"],
                host_mirror,
            )
            captured_pre_hc = host_mirror

        return _DeepSeekV4MainDecodeOutput(
            inputs=inputs,
            hidden=hidden_buffer,
            pre_hc_hidden=captured_pre_hc,
            logits=ta.tensors["logits"],
            sampled_ids=ta.tensors["sampled_ids"],
        )

    @staticmethod
    def _ignore_prefill_context(
        inputs: DeepSeekV4PreparedPrefillInputs,
        pre_hc_hidden: torch.Tensor,
    ) -> None:
        """Ignore main-prefill intermediates in autoregressive mode."""
        return None

    def _capture_mtp_prefill_context(
        self,
        inputs: DeepSeekV4PreparedPrefillInputs,
        pre_hc_hidden: torch.Tensor,
    ) -> None:
        """Rebuild the MTP sliding window from the main-model prefill tail."""
        layout = self._compiled.layout
        for request_id, rank, local_row, actual_tokens in zip(
            inputs.request_ids,
            inputs.ranks,
            inputs.local_rows,
            inputs.actual_tokens,
            strict=True,
        ):
            n = int(actual_tokens)
            if n <= 0:
                raise ValueError("DeepSeekV4 MTP prefill chunks must not be empty")
            rank = int(rank)
            state = self._reserve_mtp_request_state(request_id, rank)
            embeddings = inputs.x_hc[rank, local_row, :n, 0].detach().cpu()
            input_ids = inputs.input_ids[rank, local_row, :n].detach().cpu().to(torch.long)
            position_ids = (
                inputs.position_ids[rank, local_row, :n].detach().cpu().to(torch.int32)
            )
            slot_mapping = (
                inputs.ori_slot_mapping[rank, local_row, :n]
                .detach()
                .cpu()
                .to(torch.long)
            )
            block_table = inputs.ori_block_table[rank, local_row].detach().cpu().clone()
            tail_tokens = min(n, int(layout.prefill_seq))
            tail_start = n - tail_tokens
            current_pre_hc = pre_hc_hidden[rank, local_row, :tail_tokens].detach().cpu()
            if current_pre_hc.shape[0] != tail_tokens:
                raise ValueError(
                    f"DeepSeekV4 main prefill returned {current_pre_hc.shape[0]} pre-HC rows "
                    f"for a {tail_tokens}-row MTP tail"
                )
            pending = state.prefill_context

            hidden_parts: list[torch.Tensor] = []
            prev_hidden_parts: list[torch.Tensor] = []
            id_parts: list[torch.Tensor] = []
            position_parts: list[torch.Tensor] = []
            slot_parts: list[torch.Tensor] = []
            if pending is not None:
                if pending.rank != rank:
                    raise RuntimeError(
                        f"DeepSeekV4 MTP request {request_id!r} moved cache partitions "
                        f"from {pending.rank} to {rank} during prefill"
                    )
                first_position = int(position_ids[0].item())
                if first_position != pending.position_id + 1:
                    raise RuntimeError(
                        f"DeepSeekV4 MTP prefill for {request_id!r} is not contiguous: "
                        f"pending={pending.position_id}, next={first_position}"
                    )
            if pending is not None and n < layout.prefill_seq:
                hidden_parts.append(embeddings[:1])
                prev_hidden_parts.append(pending.prev_hidden_state.unsqueeze(0))
                id_parts.append(input_ids[:1])
                position_parts.append(torch.tensor((pending.position_id,), dtype=torch.int32))
                slot_parts.append(torch.tensor((pending.slot_mapping,), dtype=torch.long))

            if tail_tokens > 1:
                hidden_parts.append(embeddings[tail_start + 1 : n])
                prev_hidden_parts.append(current_pre_hc[: tail_tokens - 1])
                id_parts.append(input_ids[tail_start + 1 : n])
                position_parts.append(position_ids[tail_start : n - 1])
                slot_parts.append(slot_mapping[tail_start : n - 1])

            if hidden_parts:
                self._run_mtp_prefill_rows(
                    rank=rank,
                    hidden_states=torch.cat(hidden_parts),
                    prev_hidden_states=torch.cat(prev_hidden_parts),
                    input_ids=torch.cat(id_parts),
                    position_ids=torch.cat(position_parts),
                    block_table=block_table,
                    slot_mapping=torch.cat(slot_parts),
                    produce_draft=False,
                )

            last_position = int(position_ids[n - 1].item())
            state.prefill_context = _DeepSeekV4MtpPrefillContext(
                rank=rank,
                prev_hidden_state=current_pre_hc[tail_tokens - 1].clone(),
                position_id=last_position,
                block_table=block_table,
                slot_mapping=int(slot_mapping[n - 1].item()),
                prompt_len=last_position + 1,
            )

    def _run_mtp_prefill_rows(
        self,
        *,
        rank: int,
        hidden_states: torch.Tensor,
        prev_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        produce_draft: bool,
    ) -> int | None:
        """Run one contiguous shifted-token MTP prefill segment."""
        self._require_mtp_buffers()
        ta = self._mtp_prefill_task_args
        layout = self._compiled.layout
        n = int(input_ids.numel())
        if n <= 0 or n > layout.prefill_seq:
            raise ValueError(
                f"DeepSeekV4 MTP prefill rows must be in [1, {layout.prefill_seq}], got {n}"
            )
        if (
            hidden_states.shape != (n, ta.tensors["hidden_states"].shape[-1])
            or prev_hidden_states.shape
            != (n, *ta.tensors["prev_hidden_states"].shape[-2:])
            or position_ids.numel() != n
            or slot_mapping.numel() != n
        ):
            raise ValueError("DeepSeekV4 MTP prefill row tensors are not aligned")

        def pad_rows(values: torch.Tensor) -> torch.Tensor:
            padded = torch.empty(
                (layout.prefill_seq, *values.shape[1:]),
                dtype=values.dtype,
                device="cpu",
            )
            padded[:n].copy_(values.detach().cpu())
            if n < layout.prefill_seq:
                indices = torch.arange(layout.prefill_seq - n) % n
                padded[n:].copy_(values.detach().cpu().index_select(0, indices))
            return padded

        hidden = pad_rows(hidden_states.to(torch.bfloat16))
        previous = pad_rows(prev_hidden_states.to(torch.float32))
        ids = self._padded_vector(input_ids, layout.prefill_seq, dtype=torch.long)
        positions = self._prefill_position_ids(
            position_ids.detach().cpu().tolist(),
            layout.prefill_seq,
        )
        ta.tensors["hidden_states"].copy_(
            hidden.unsqueeze(0).expand(layout.ranks, -1, -1)
        )
        ta.tensors["prev_hidden_states"].copy_(
            previous.unsqueeze(0).expand(layout.ranks, -1, -1, -1)
        )
        ta.tensors["input_ids"].copy_(ids.unsqueeze(0).expand(layout.ranks, -1))
        ta.tensors["position_ids"].copy_(
            positions.unsqueeze(0).expand(layout.ranks, -1)
        )
        ta.tensors["ori_block_table"].copy_(
            block_table.detach().cpu().unsqueeze(0).expand(layout.ranks, -1)
        )
        ori_slot_mapping = ta.tensors["ori_slot_mapping"]
        ori_slot_mapping.fill_(-1)
        ori_slot_mapping[rank, :n].copy_(slot_mapping.detach().cpu().to(torch.long))
        # These are pl.Out tensors for every active row. Reusing them avoids
        # clearing several large host buffers on every chunk.
        logit_row_indices = ta.tensors["logit_row_indices"]
        logit_row_indices.fill_(-1)
        if produce_draft:
            logit_row_indices[rank, 0] = n - 1
        with profile_span(
            "DeepSeekV4ModelRunner.mtp.prefill.l3_dispatch",
            cat="executor",
            args={"actual_tokens": n, "produce_draft": produce_draft},
        ):
            self._run_l3(
                self._require_mtp_prefill_callable(),
                *self._mtp_prefill_args(),
                self._int32_scalar(n),
            )
        if not produce_draft:
            return None
        with profile_span(
            "DeepSeekV4ModelRunner.mtp.prefill.read_logits",
            cat="executor",
            args={"bytes": DEEPSEEK_V4_VOCAB_SIZE * torch.float32.itemsize},
        ):
            logits = self._read_mtp_prefill_logits(rank)
        return int(logits.argmax().item())

    def _require_prefill_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.prefill is None:
            raise RuntimeError("DeepSeekV4 prefill kernel is not compiled")
        return self._compiled.prefill

    def _require_decode_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 decode kernel is not compiled")
        return self._compiled.decode

    def _ensure_l3_shared_buffers(self, model: RuntimeModel) -> None:
        """Allocate every CPU tensor visible to the L3 worker before it forks.

        ``DistributedWorker`` creates per-chip children on first use. Mutable CPU
        arguments must already live in shared memory at that point; immutable
        weights are registered for fork inheritance. This method prepares both
        groups before the first ``_run_l3`` call.
        """
        if self._l3_shared_buffers_ready:
            return
        with profile_span("DeepSeekV4ModelRunner.prepare.load_global_weights", cat="executor"):
            self.load_packed_global_weights()
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_rope_tables", cat="executor"):
            self._static_freqs_cos_tensor()
            self._static_freqs_sin_tensor()
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_decode_buffers", cat="executor"):
            self._ensure_decode_buffers(model.config.hidden_size, model.config.vocab_size)
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_mtp_buffers", cat="executor"):
            self._ensure_mtp_buffers(model.config.hidden_size)
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_final_norm", cat="executor"):
            self._static_final_norm_weight_tensor()
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_lm_head", cat="executor"):
            self._static_lm_head_weight_tensor()
        if self._stacked_host_weights is None:
            if self._stacked_device_weights is None:
                with profile_span(
                    "DeepSeekV4ModelRunner.prepare.load_and_pack_layer_weights",
                    cat="executor",
                ):
                    stacked_weights = self.load_stacked_layer_weights()
                with profile_span("DeepSeekV4ModelRunner.prepare.retain_layer_weights", cat="executor"):
                    self._retain_stacked_host_weights(stacked_weights)
                del stacked_weights
        with profile_span("DeepSeekV4ModelRunner.prepare.prepare_hc_head", cat="executor"):
            self._hc_head_tensors()
        with profile_span("DeepSeekV4ModelRunner.prepare.allocate_prefill_task_args", cat="executor"):
            from pypto_serving.model.deepseek.task_args import prefill_task_args  # noqa: PLC0415

            self._prefill_task_args = prefill_task_args(
                self, model.config.hidden_size, model.config.vocab_size
            )
            self._prefill_task_args.allocate_host_shared(None)
        with profile_span("DeepSeekV4ModelRunner.upload_resident_weights", cat="executor"):
            self._materialize_resident_weights()
        self._l3_shared_buffers_ready = True

    def _prefill_fwd_args(self, kernel_tokens: int) -> tuple[Any, ...]:
        """Build the single packed ``l3_prefill_fwd`` argument tuple.

        The kernel runs final RMSNorm and the device-side LM-head. Every positional
        arg is declared on ``_prefill_task_args`` (see ``deepseek/task_args.py``).
        """
        return self._prefill_task_args.build_for_tokens(kernel_tokens)

    def _decode_fwd_args(self, inputs: DeepSeekV4PreparedDecodeInputs) -> tuple[Any, ...]:
        """Build the single packed ``l3_decode_fwd`` argument tuple from the decode TaskArgs.

        Per-step metadata remains Host-shared so prepare can overlap the prior
        dispatch without issuing synchronous worker copies.
        """
        return self._decode_task_args[inputs.buffer_slot].build()

    def _device_cache_values(self) -> dict[str, StackedDeviceTensor]:
        """Return the unified worker-resident cache pools by kernel argument name."""
        cache = self._materialize_decode_device_cache()
        return {
            "kv_cache": cache.kv_cache,
            "hca_cmp_kv": cache.hca_cmp_kv,
            "csa_cmp_kv": cache.csa_cmp_kv,
            "idx_kv_cache": cache.idx_kv_cache,
            "idx_kv_scale": cache.idx_kv_scale,
            "hca_compress_state": cache.hca_compress_state,
            "csa_compress_state": cache.csa_compress_state,
            "csa_inner_compress_state": cache.csa_inner_compress_state,
        }

    def _mtp_prefill_args(self) -> tuple[Any, ...]:
        """Build the single packed ``l3_mtp_prefill`` argument tuple from the prefill TaskArgs."""
        return self._mtp_prefill_task_args.build()

    def _mtp_decode_args(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs | None = None,
    ) -> tuple[Any, ...]:
        """Build PR985's standalone ``l3_decode_mtp`` argument tuple."""
        if self._compiled.num_speculative_tokens == 1:
            raise RuntimeError("standalone MTP decode arguments are unavailable in fused K=1 mode")
        buffer_slot = inputs.buffer_slot if inputs is not None else 0
        return self._mtp_decode_task_args[buffer_slot].build()

    def _fused_mtp_decode_args(
        self,
        main_args: tuple[Any, ...],
        inputs: DeepSeekV4PreparedDecodeInputs,
        active_tokens: int,
    ) -> tuple[Any, ...]:
        """Stitch the main decode tuple with the MTP-only args + tail prepend + scalar.

        The MTP decode TaskArgs owns every ``_FUSED_MTP_DECODE_TENSOR_ORDER`` arg;
        the shared names (embed/freqs/lm_head/ori_block_table/main_pre_hc_hidden)
        are stripped because the fused kernel takes them from the main decode
        tuple.  The tail token ids / positions are fused-only prepend buffers (not
        in the MTP arg order) and stay in ``_decode_input_slots``.
        """
        from pypto_serving.model.deepseek.task_args import _FUSED_MTP_SHARED_TENSORS  # noqa: PLC0415

        mtp_ta = self._mtp_decode_task_args[inputs.buffer_slot]
        mtp_args = mtp_ta.build()
        mtp_only = tuple(
            arg
            for name, arg in zip(mtp_ta.names, mtp_args, strict=True)
            if name not in _FUSED_MTP_SHARED_TENSORS
        )
        staged = self._decode_input_slots[inputs.buffer_slot]
        return (
            *main_args,
            staged["mtp_tail_token_ids"],
            staged["mtp_tail_positions"],
            *mtp_only,
            self._int32_scalar(active_tokens),
        )

    def _bind_prepared_mtp_dispatch(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        hidden_size: int,
        vocab_size: int,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Bind all steady L3 arguments on the prepare lane."""
        main_args = self._decode_fwd_args(inputs)
        active_tokens = max(inputs.per_rank_counts) * self._compiled.layout.decode_seq
        return replace(
            inputs,
            dispatch_args=self._fused_mtp_decode_args(main_args, inputs, active_tokens),
        )

    def _require_mtp_buffers(self) -> _DeepSeekV4MtpSharedBuffers:
        if self._mtp_buffers is None:
            raise RuntimeError("DeepSeekV4 MTP shared buffers are not staged")
        return self._mtp_buffers

    def _require_mtp_prefill_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.mtp_prefill is None:
            raise RuntimeError("DeepSeekV4 MTP prefill kernel is not compiled")
        return self._compiled.mtp_prefill

    def _require_mtp_decode_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.mtp_decode is None:
            raise RuntimeError("DeepSeekV4 MTP decode kernel is not compiled")
        return self._compiled.mtp_decode

    def _embedding_rows(self, token_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        embed = self._compiled.embedding_weight
        if embed is None:
            embed = self._compiled.weight_store.load_tensor("embed.weight").contiguous().cpu()
            self._compiled.embedding_weight = embed
        return embed.index_select(0, token_ids.detach().cpu().to(torch.long).reshape(-1)).to(dtype)

    def _device_state_placeholder_batch(self, batch: DecodeBatch) -> DecodeBatch:
        """Build valid fallback rows; the fused kernel replaces every active row."""
        actual_batch = len(batch.request_ids)
        current = batch.token_ids[:actual_batch].detach().cpu().to(torch.long).reshape(-1)
        for request_id in batch.request_ids:
            state = self._require_mtp_request_state(request_id)
            if not state.device_state_initialized:
                raise RuntimeError(f"DeepSeekV4 MTP device state is missing for {request_id!r}")
        return replace(
            batch,
            token_ids=current.reshape(actual_batch, 1),
            hidden_states=None,
            prev_token_ids=current,
            prev_hidden_states=None,
            seq_lens=torch.clamp(batch.seq_lens.detach().cpu().to(torch.int32), min=2),
        )

    def _correct_mtp_seq_lens(self, batch: DecodeBatch) -> torch.Tensor:
        """Return request lengths corrected from committed speculative output."""
        actual_batch = len(batch.request_ids)
        corrected = batch.seq_lens[:actual_batch].detach().cpu().to(torch.int32).clone()
        # Async scheduling reserves the maximum accepted width before the worker
        # knows which drafts match. After the first MTP step, the runner's
        # committed count is therefore the authoritative request length.
        for index, request_id in enumerate(batch.request_ids):
            state = self._mtp_request_states.get(request_id)
            if state is not None and state.proposed_tokens > 0:
                corrected[index] = state.prompt_len + state.committed_count + 1
        return corrected

    def _require_mtp_request_state(self, request_id: str) -> _DeepSeekV4MtpRequestState:
        with self._mtp_state_lock:
            state = self._mtp_request_states.get(request_id)
        if state is None:
            raise RuntimeError(f"DeepSeekV4 MTP state is missing for request {request_id!r}")
        return state

    def _reserve_mtp_request_state(
        self,
        request_id: str,
        rank: int,
    ) -> _DeepSeekV4MtpRequestState:
        """Publish a stable slot identity before terminal prefill completes."""
        rank = int(rank)
        with self._mtp_state_lock:
            state = self._mtp_request_states.get(request_id)
            if state is None:
                state = _DeepSeekV4MtpRequestState()
                self._mtp_request_states[request_id] = state
            self._reserve_mtp_state_slot(state, rank, request_id=request_id)
            return state

    def _reserve_mtp_state_slot(
        self,
        state: _DeepSeekV4MtpRequestState,
        rank: int,
        *,
        request_id: str,
    ) -> None:
        """Assign one host-visible slot identity without initializing its device contents."""
        rank = int(rank)
        with self._mtp_state_lock:
            if state.tail_rank is not None:
                if state.tail_rank != rank:
                    raise RuntimeError(
                        f"DeepSeekV4 MTP state rank changed for {request_id!r}: "
                        f"slot rank={state.tail_rank}, decode rank={rank}"
                    )
                return
            free_slots = self._mtp_free_tail_slots[rank]
            if not free_slots:
                raise RuntimeError(f"DeepSeekV4 MTP tail slots are exhausted on rank {rank}")
            slot = free_slots.pop()
            generation = self._mtp_slot_generations[rank][slot] + 1
            if generation >= 2**31:
                generation = 1
            self._mtp_slot_generations[rank][slot] = generation
            state.tail_rank = rank
            state.tail_slot_id = slot
            state.generation = generation

    def _mtp_drafts_for_requests(self, request_ids: Sequence[str]) -> torch.Tensor:
        draft_ids = []
        for request_id in request_ids:
            draft_token_id = self._require_mtp_request_state(request_id).draft_token_id
            if draft_token_id is None:
                raise RuntimeError(f"DeepSeekV4 MTP draft is not initialized for {request_id!r}")
            draft_ids.append(draft_token_id)
        return torch.tensor(draft_ids, dtype=torch.long)

    @staticmethod
    def _select_decode_batch_rows(batch: DecodeBatch, indices: Sequence[int]) -> DecodeBatch:
        """Return a request-aligned subset without changing cache ownership."""
        rows = tuple(int(index) for index in indices)

        def select_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            index = torch.tensor(rows, dtype=torch.long, device=value.device)
            return value.index_select(0, index)

        def select_list(values):
            return [values[index] for index in rows] if values else []

        return replace(
            batch,
            request_ids=select_list(batch.request_ids),
            token_ids=select_tensor(batch.token_ids),
            hidden_states=select_tensor(batch.hidden_states),
            seq_lens=select_tensor(batch.seq_lens),
            kv_allocations=select_list(batch.kv_allocations),
            block_ids=select_list(batch.block_ids),
            block_ids_by_group=select_list(batch.block_ids_by_group),
            cache_partitions=select_list(batch.cache_partitions),
        )

    def _propose_mtp_tokens(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        num_drafts: int | None = None,
    ) -> torch.Tensor:
        """Generate the configured number of drafts by recurrently reusing the MTP layer."""
        if num_drafts is None:
            num_drafts = self._compiled.num_speculative_tokens
        num_drafts = int(num_drafts)
        if not 0 <= num_drafts <= self._compiled.num_speculative_tokens:
            raise ValueError("MTP draft count must fit the compiled speculative depth")
        if num_drafts == 0:
            return torch.empty((len(batch.request_ids), 0), dtype=torch.long)
        self._initialize_mtp_drafts(batch)
        draft_columns = [self._mtp_drafts_for_requests(batch.request_ids)]
        if num_drafts == 1:
            return draft_columns[0].reshape(-1, 1)

        recurrent_hidden = []
        positions = []
        for request_id in batch.request_ids:
            state = self._require_mtp_request_state(request_id)
            if state.draft_pre_hc_hidden is None or state.draft_position is None:
                raise RuntimeError(f"DeepSeekV4 recurrent MTP state is missing for {request_id!r}")
            recurrent_hidden.append(state.draft_pre_hc_hidden)
            positions.append(state.draft_position)
        hidden = torch.stack(recurrent_hidden)
        position_ids = torch.tensor(positions, dtype=torch.int32)

        while len(draft_columns) < num_drafts:
            next_tokens, hidden = self._run_mtp_token_step(
                model,
                batch,
                draft_columns[-1],
                hidden,
                position_ids,
            )
            draft_columns.append(next_tokens)
            position_ids = position_ids + 1
        return torch.stack(draft_columns, dim=1)

    def _run_mtp_token_step(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        token_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one recurrent MTP token per request with PR985's lowered ABI."""
        actual_batch = len(batch.request_ids)
        if token_ids.numel() != actual_batch or positions.numel() != actual_batch:
            raise ValueError("MTP token IDs and positions must align with active requests")
        layout = self._compiled.layout
        expected_hidden_shape = (
            actual_batch,
            layout.hc_mult,
            int(model.config.hidden_size),
        )
        if tuple(previous_hidden.shape) != expected_hidden_shape:
            raise ValueError(
                "MTP recurrent hidden rows must have shape "
                f"{expected_hidden_shape}, got {tuple(previous_hidden.shape)}"
            )

        self._require_mtp_buffers()
        mtp_slots = self._mtp_decode_task_args[0].tensors
        assignment = self._decode_assignment(batch)
        token_ids = token_ids.detach().cpu().to(torch.long).reshape(actual_batch)
        positions = positions.detach().cpu().to(torch.int32).reshape(actual_batch)
        previous_hidden = previous_hidden.detach().cpu().to(torch.float32)
        next_tokens: list[int | None] = [None] * actual_batch
        next_hidden: list[torch.Tensor | None] = [None] * actual_batch
        wave_count = max(assignment.per_rank_counts)

        for wave in range(wave_count):
            wave_items = [
                (rank, request_indices[wave])
                for rank, request_indices in enumerate(assignment.indices_by_rank)
                if wave < len(request_indices)
            ]
            wave_ranks = tuple(rank for rank, _request_index in wave_items)
            wave_indices = tuple(request_index for _rank, request_index in wave_items)
            self._stage_recurrent_mtp_swa_metadata(
                batch,
                request_indices=wave_indices,
                ranks=wave_ranks,
                positions=positions,
            )

            fallback_index = wave_indices[0]
            mtp_slots["input_ids"].fill_(int(token_ids[fallback_index]))
            mtp_slots["position_ids"].fill_(int(positions[fallback_index]))
            mtp_slots["prev_pre_hc_hidden"].copy_(
                previous_hidden[fallback_index]
                .view(1, 1, layout.hc_mult, model.config.hidden_size)
                .expand(
                    layout.ranks,
                    layout.decode_tokens,
                    layout.hc_mult,
                    model.config.hidden_size,
                )
            )
            mtp_slots["logit_row_indices"].fill_(-1)
            mtp_slots["sampling_temperatures"].zero_()
            mtp_slots["sampling_top_ks"].fill_(DEEPSEEK_V4_VOCAB_SIZE)
            mtp_slots["sampling_seeds"].zero_()
            mtp_slots["sampling_positions"].zero_()
            for rank, request_index in wave_items:
                mtp_slots["input_ids"][rank, 0] = int(token_ids[request_index])
                mtp_slots["position_ids"][rank, 0] = int(positions[request_index])
                mtp_slots["prev_pre_hc_hidden"][rank, 0].copy_(
                    previous_hidden[request_index]
                )
                mtp_slots["logit_row_indices"][rank, 0] = 0
                mtp_slots["sampling_positions"][rank, 0] = int(positions[request_index]) + 1
            mtp_slots["hidden_states"].copy_(
                self._embedding_rows(
                    mtp_slots["input_ids"].reshape(-1),
                    torch.bfloat16,
                ).reshape(
                    layout.ranks,
                    layout.decode_tokens,
                    model.config.hidden_size,
                )
            )

            with profile_span(
                "DeepSeekV4ModelRunner.mtp.decode.l3_dispatch",
                cat="executor",
                args={"actual_tokens": 1},
            ):
                self._run_l3(
                    self._require_mtp_decode_callable(),
                    *self._mtp_decode_args(),
                    self._int32_scalar(1),
                )

            for rank, request_index in wave_items:
                next_tokens[request_index] = int(
                    mtp_slots["sampled_ids"][rank, 0, 0].item()
                )
                next_hidden[request_index] = (
                    mtp_slots["next_pre_hc_hidden"][rank, 0].detach().cpu().clone()
                )

        if any(token is None for token in next_tokens) or any(
            hidden is None for hidden in next_hidden
        ):
            raise RuntimeError("recurrent MTP decoding left incomplete request state")
        return (
            torch.tensor(next_tokens, dtype=torch.long),
            torch.stack(next_hidden),
        )

    def _stage_recurrent_mtp_swa_metadata(
        self,
        batch: DecodeBatch,
        *,
        request_indices: Sequence[int],
        ranks: Sequence[int],
        positions: torch.Tensor,
    ) -> None:
        """Lower one recurrent wave's MTP cache writes and visible SWA rows."""
        if not request_indices or len(request_indices) != len(ranks):
            raise ValueError("recurrent MTP requests and ranks must be non-empty and aligned")
        if len(set(int(rank) for rank in ranks)) != len(ranks):
            raise ValueError("recurrent MTP waves support at most one request per rank")

        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if len(batch.block_ids_by_group) != actual_batch:
            raise ValueError(
                "grouped KV metadata has "
                f"{len(batch.block_ids_by_group)} rows, expected decode batch {actual_batch}"
            )
        requests_by_rank = {
            int(rank): int(request_index)
            for rank, request_index in zip(ranks, request_indices, strict=True)
        }
        if any(not 0 <= rank < layout.ranks for rank in requests_by_rank):
            raise ValueError("recurrent MTP rank is outside the compiled world")
        if any(not 0 <= index < actual_batch for index in requests_by_rank.values()):
            raise ValueError("recurrent MTP request index is outside the active batch")
        positions = positions.detach().cpu().to(torch.int32).reshape(-1)
        if positions.numel() != actual_batch:
            raise ValueError("recurrent MTP positions must align with the active batch")
        active_ori_ids = {}
        for request_index in requests_by_rank.values():
            ori_ids = tuple(
                int(block_id)
                for block_id in batch.block_ids_by_group[request_index].get("ori", ())
            )
            if not ori_ids:
                raise ValueError(f"decode row {request_index} is missing grouped KV blocks: ori")
            active_ori_ids[request_index] = ori_ids

        slots = self._mtp_decode_task_args[0].tensors
        fallback_position = int(positions[int(request_indices[0])].item())
        scratch_ids = self._scratch_group_block_ids(
            group_name="ori",
            kernel_rows=layout.decode_batch,
        )
        for rank in range(layout.ranks):
            request_index = requests_by_rank.get(rank)
            if request_index is None:
                padded_ids = scratch_ids
                padded_positions = [
                    (fallback_position,) * layout.decode_seq
                    for _ in range(layout.decode_batch)
                ]
            else:
                active_ids = active_ori_ids[request_index]
                padded_ids = self._pad_group_block_ids(
                    [active_ids],
                    group_name="ori",
                    kernel_rows=layout.decode_batch,
                )
                padded_positions = [
                    (int(positions[request_index].item()),) * layout.decode_seq
                ]
                padded_positions.extend(
                    (fallback_position,) * layout.decode_seq
                    for _ in range(layout.decode_batch - 1)
                )
            slot_mapping = self.cache_metadata.paged_decode_slot_mapping_from_ids(
                padded_ids,
                padded_positions,
            ).reshape(-1)
            swa_indices, swa_lens = self.cache_metadata.swa_window_indices_and_lens_from_ids(
                padded_ids,
                padded_positions,
            )
            # decode_mtp gates MoE with num_tokens but its projection and SWA
            # attention still execute the complete fixed tile. Keep token 0 of
            # an active rank on the live request and isolate every filler token
            # in a distinct reserved cache slot.
            for row, (scratch_block,) in enumerate(scratch_ids):
                live_prefix = int(request_index is not None and row == 0)
                start = row * layout.decode_seq + live_prefix
                stop = (row + 1) * layout.decode_seq
                if start == stop:
                    continue
                offsets = torch.arange(live_prefix, layout.decode_seq, dtype=torch.long)
                filler_slots = offsets + int(scratch_block) * layout.block_size
                slot_mapping[start:stop].copy_(filler_slots)
                swa_indices[start:stop].fill_(-1)
                swa_indices[start:stop, 0].copy_(filler_slots.to(torch.int32))
                swa_lens[start:stop].fill_(1)
            copy_shared(
                slots["swa_slot_mapping"][rank],
                slot_mapping,
                name=f"mtp_recurrent_swa_slot_mapping_rank{rank}",
            )
            copy_shared(
                slots["swa_indices"][rank],
                swa_indices,
                name=f"mtp_recurrent_swa_indices_rank{rank}",
            )
            copy_shared(
                slots["swa_lens"][rank],
                swa_lens,
                name=f"mtp_recurrent_swa_lens_rank{rank}",
            )

    def _initialize_mtp_drafts(self, batch: DecodeBatch) -> None:
        """Initialize every request's first draft without sharing mutable state."""
        for batch_index, request_id in enumerate(batch.request_ids):
            state = self._require_mtp_request_state(request_id)
            if state.draft_token_id is None:
                first_token_id = int(
                    batch.token_ids[batch_index].detach().cpu().to(torch.long).reshape(-1)[0]
                )
                self._initialize_mtp_draft(request_id, state, first_token_id)

    def _initialize_mtp_draft(
        self,
        request_id: str,
        state: _DeepSeekV4MtpRequestState,
        first_token_id: int,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        context = state.prefill_context
        if context is None:
            raise RuntimeError(f"DeepSeekV4 MTP prefill context is missing for {request_id!r}")
        owner_rank = context.rank
        first_token = torch.tensor([first_token_id], dtype=torch.long)
        first_hidden = self._embedding_rows(first_token, torch.bfloat16)

        draft_token_id = self._run_mtp_prefill_rows(
            rank=owner_rank,
            hidden_states=first_hidden,
            prev_hidden_states=context.prev_hidden_state.unsqueeze(0),
            input_ids=first_token,
            position_ids=torch.tensor((context.position_id,), dtype=torch.int32),
            block_table=context.block_table,
            slot_mapping=torch.tensor((context.slot_mapping,), dtype=torch.long),
            produce_draft=True,
        )
        if draft_token_id is None:
            raise RuntimeError("DeepSeekV4 MTP draft prefill did not produce a token")
        state.draft_token_id = draft_token_id
        state.draft_pre_hc_hidden = self._read_mtp_prefill_pre_hc(owner_rank, row=0)
        state.draft_position = context.position_id + 1
        state.tail_token_id = int(first_token[0].item())
        self._write_mtp_tail_hidden(
            state,
            owner_rank,
            context.prev_hidden_state,
        )
        state.tail_position = context.position_id
        state.prompt_len = context.prompt_len
        self._initialize_mtp_device_state(state, request_id, sampling_params)
        state.prefill_context = None

    def _write_mtp_tail_hidden(
        self,
        state: _DeepSeekV4MtpRequestState,
        rank: int,
        hidden: torch.Tensor,
    ) -> None:
        """Initialize the preallocated persistent tail slot contents."""
        self._reserve_mtp_state_slot(state, rank, request_id="<prefill>")
        if state.tail_rank != rank or state.tail_slot_id is None:
            raise RuntimeError("DeepSeekV4 MTP tail slot reservation is inconsistent")
        slot = state.tail_slot_id
        buffers = self._require_mtp_buffers()
        staged = buffers.tail_init_hidden[rank, slot]
        staged.copy_(hidden.to(dtype=torch.float32, device="cpu"))
        pool = self._materialize_mtp_tail_pre_hc_pool(int(staged.shape[-1]))
        shard = pool.shards[rank]
        row_nbytes = staged.numel() * staged.element_size()
        self._shared_l3_worker().copy_to(
            shard.data_ptr,
            staged.data_ptr(),
            row_nbytes,
            dst_offset=slot * row_nbytes,
            worker_id=rank,
        )

    def _initialize_mtp_device_state(
        self,
        state: _DeepSeekV4MtpRequestState,
        request_id: str,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        """Seed one stable device slot after the first MTP draft is available."""
        if state.device_state_initialized:
            return
        if (
            state.tail_rank is None
            or state.tail_slot_id is None
            or state.tail_token_id is None
            or state.draft_token_id is None
            or state.tail_position is None
        ):
            raise RuntimeError("DeepSeekV4 MTP device state cannot be initialized from partial state")
        buffers = self._require_mtp_buffers()
        rank = state.tail_rank
        slot = state.tail_slot_id
        token_row = buffers.state_init_tokens[rank, slot]
        token_row[0] = state.tail_token_id
        token_row[1] = state.draft_token_id
        meta_row = buffers.state_init_meta[rank, slot]
        meta_row.zero_()
        # Publish a completely initialized payload.  STATE_VALID is an
        # initialization guard; slot reuse safety comes from generation.
        meta_row[_MTP_STATE_VALID] = 1
        meta_row[_MTP_STATE_GENERATION] = state.generation
        meta_row[_MTP_STATE_TAIL_POSITION] = state.tail_position
        meta_row[_MTP_STATE_COMMITTED_COUNT] = state.committed_count
        params = sampling_params or SamplingParams(temperature=0.0, top_p=1.0, top_k=None)
        top_k = params.top_k
        if top_k is None or top_k <= 0:
            top_k = DEEPSEEK_V4_VOCAB_SIZE
        meta_row[_MTP_STATE_TEMPERATURE_MICROS] = min(
            round(max(float(params.temperature), 0.0) * _MTP_TEMPERATURE_SCALE),
            _MTP_MAX_TEMPERATURE_MICROS,
        )
        meta_row[_MTP_STATE_TOP_K] = min(int(top_k), DEEPSEEK_V4_VOCAB_SIZE)
        seed = params.seed
        if seed is None:
            seed = zlib.crc32(request_id.encode())
        meta_row[_MTP_STATE_SEED] = int(seed) & 0x3FFFFFFF
        worker = self._shared_l3_worker()
        for device_tensor, source in (
            (self._materialize_mtp_device_state_tokens(), token_row),
            (self._materialize_mtp_device_state_meta(), meta_row),
        ):
            shard = device_tensor.shards[rank]
            row_nbytes = source.numel() * source.element_size()
            worker.copy_to(
                shard.data_ptr,
                source.data_ptr(),
                row_nbytes,
                dst_offset=slot * row_nbytes,
                worker_id=rank,
            )
        state.device_state_initialized = True

    def _prepare_mtp_device_state_descriptors(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        *,
        require_ready: bool = False,
    ) -> DeepSeekV4PreparedDecodeInputs:
        """Bind batch rows to stable state slots in the input ping-pong slot.

        Stable slot ids and generations do not depend on step-N acceptance, so
        steady-state N+1 preparation can write them while N executes.  The
        first MTP decode still uses the guarded late path because it creates the
        device state itself.

        ``tail_slot_ids[rank, local_row]`` maps the transient scheduler row to
        the persistent request slot.  ``state_generations`` carries the slot's
        allocation tag, while ``logit_row_indices`` selects the final row of
        each S-token committed window for the next-draft LM head.
        """
        if self._compiled.num_speculative_tokens != 1:
            return inputs
        staged = self._mtp_decode_task_args[inputs.buffer_slot].tensors
        tail_slot_ids = staged["tail_slot_ids"]
        state_generations = staged["state_generations"]
        logit_row_indices = staged["logit_row_indices"]
        tail_slot_ids.fill_(-1)
        state_generations.zero_()
        logit_row_indices.fill_(-1)
        for request_id, rank, local_row in zip(
            inputs.request_ids,
            inputs.ranks,
            inputs.local_rows,
            strict=True,
        ):
            with self._mtp_state_lock:
                state = self._mtp_request_states.get(request_id)
                if state is None:
                    if require_ready:
                        raise RuntimeError(
                            f"DeepSeekV4 MTP state is missing for request {request_id!r}"
                        )
                    return inputs
                if state.tail_slot_id is None:
                    if require_ready:
                        raise RuntimeError(
                            f"DeepSeekV4 MTP tail slot is not reserved for {request_id!r}"
                        )
                    return inputs
                if state.tail_rank != rank:
                    raise RuntimeError(
                        f"DeepSeekV4 MTP state rank changed for {request_id!r}: "
                        f"slot rank={state.tail_rank}, decode rank={rank}"
                    )
                tail_slot_ids[rank, local_row] = state.tail_slot_id
                state_generations[rank, local_row] = state.generation
            # The MTP model emits one next draft per request.  It must consume
            # the hidden state at the end of that request's committed window,
            # not every main-model verifier row.
            logit_row_indices[rank, local_row] = (
                local_row * self._compiled.layout.decode_seq
                + self._compiled.layout.decode_seq
                - 1
            )
        return replace(
            inputs,
            mtp_tail_slot_ids=tail_slot_ids,
            mtp_state_generations=state_generations,
            mtp_logit_row_indices=logit_row_indices,
        )

    def _require_stacked_weights(self) -> DeepSeekV4StackedLayerWeights:
        tensors = self._stacked_device_weights or self._stacked_host_weights
        if tensors is None:
            raise RuntimeError("DeepSeekV4 stacked decode weights are not available")
        return DeepSeekV4StackedLayerWeights(tensors=tensors)

    def _ensure_decode_buffers(self, hidden_size: int, vocab_size: int = DEEPSEEK_V4_VOCAB_SIZE) -> None:
        """Allocate the decode TaskArgs (host-shared slots) and MTP reclaim slots.

        Runs before the L3 worker forks so the fork-inherited host shared memory
        carries every buffer the chip workers read.  The device-resident pre-HC
        output slot is allocated lazily at first dispatch (post-fork) by
        ``_decode_fwd_args``.
        """
        if self._decode_task_args:
            return
        self._ensure_shared_host_allocation_before_worker("decode inputs")
        layout = self._compiled.layout
        ranks = layout.ranks
        batch = layout.decode_batch

        from pypto_serving.model.deepseek.task_args import (  # noqa: PLC0415
            decode_task_args,
            mtp_decode_task_args,
        )

        # Two execution snapshots let the command thread prepare step N+1 while
        # the device thread consumes step N.  Worker slot ownership is released
        # only after output reclaim has finished reading the slot.  The
        # ``_DECODE_FWD_TENSOR_ORDER`` buffers (metadata inputs, sampled_ids,
        # hidden/logits outputs) are owned by the decode TaskArgs; the
        # device-resident pre-HC output is a device slot on the same TaskArgs.
        self._decode_task_args = []
        for slot in (0, 1):
            task_args = decode_task_args(
                self,
                int(hidden_size),
                int(vocab_size),
            )
            task_args.allocate_host_shared(None)
            self._decode_task_args.append(task_args)

        # MTP decode TaskArgs (one per ping-pong slot) own the
        # ``_MTP_DECODE_TENSOR_ORDER`` reclaimed + write-only outputs.  The fused
        # prepend buffers (tail token ids / positions) are not part of the kernel
        # arg order, so they stay in ``_decode_input_slots``.
        legacy_mtp = (
            self._compiled.mtp_prefill is not None
            and self._compiled.mtp_decode is not None
        )
        if self._compiled.num_speculative_tokens or legacy_mtp:
            self._mtp_decode_task_args = []
            for slot in (0, 1):
                mtp_ta = mtp_decode_task_args(
                    self,
                    int(hidden_size),
                    buffer_slot=slot,
                )
                mtp_ta.allocate_host_shared(None)
                self._mtp_decode_task_args.append(mtp_ta)

            def allocate_mtp_prepend_slot(slot: int) -> dict[str, torch.Tensor]:
                prefix = f"decode_slot{slot}"
                return {
                    "mtp_tail_token_ids": shared_empty(
                        (ranks, batch), torch.long, name=f"{prefix}_mtp_tail_token_ids"
                    ),
                    "mtp_tail_positions": shared_empty(
                        (ranks, batch), torch.int32, name=f"{prefix}_mtp_tail_positions"
                    ),
                }

            self._decode_input_slots = [allocate_mtp_prepend_slot(0), allocate_mtp_prepend_slot(1)]
        else:
            self._decode_input_slots = []

        # Host mirror for the device-resident main decode pre-HC output.  The
        # arbitrary-depth target-verification path reads committed pre-HC rows on
        # the host (``copy_stacked_from`` cannot target an offset device pointer),
        # so one ping-pong-independent shared buffer holds the D2H readback.  It is
        # not a kernel argument and therefore lives outside the decode TaskArgs.
        self._main_pre_hc_host_mirror = shared_empty(
            (ranks, layout.decode_tokens, layout.hc_mult, int(hidden_size)),
            torch.float32,
            name="decode_main_pre_hc_host_mirror",
        )

    def _require_main_pre_hc_host_mirror(self) -> torch.Tensor:
        """Return the host mirror used for the arbitrary-depth pre-HC readback."""
        if self._main_pre_hc_host_mirror is None:
            raise RuntimeError("DeepSeekV4 main decode pre-HC host mirror is not staged")
        return self._main_pre_hc_host_mirror

    def _ensure_mtp_buffers(self, hidden_size: int) -> _DeepSeekV4MtpSharedBuffers | None:
        """Load immutable MTP weights and allocate mutable shared buffers before worker fork."""
        legacy_mtp = (
            self._compiled.mtp_prefill is not None
            and self._compiled.mtp_decode is not None
        )
        if not self._compiled.num_speculative_tokens and not legacy_mtp:
            return None
        if self._mtp_buffers is not None:
            return self._mtp_buffers
        self._ensure_shared_host_allocation_before_worker("mtp buffers")
        layout = self._compiled.layout
        ranks = layout.ranks
        hidden = int(hidden_size)
        self._ensure_decode_buffers(hidden)
        loaded = self.load_mtp_weights()
        weights = dict(loaded.tensors)
        # The real MTP pool is allocated directly on each worker after runtime
        # HBM sizing. Keep only a one-page shared placeholder here so legacy
        # buffer consumers retain the unified prefill/decode object contract.
        mtp_kv_cache = shared_empty(
            (ranks, 1, layout.block_size, 1, DEEPSEEK_V4_HEAD_DIM),
            torch.bfloat16,
            name="mtp_kv_cache_placeholder",
        )
        self._mtp_buffers = _DeepSeekV4MtpSharedBuffers(
            weights=weights,
            prefill_kv_cache=mtp_kv_cache,
            state_init_tokens=shared_empty(
                (ranks, layout.decode_batch, _MTP_DEVICE_STATE_TOKEN_WIDTH),
                torch.long,
                name="mtp_state_init_tokens",
            ),
            state_init_meta=shared_empty(
                (ranks, layout.decode_batch, _MTP_DEVICE_STATE_META_WIDTH),
                torch.int32,
                name="mtp_state_init_meta",
            ),
            tail_init_hidden=shared_empty(
                (ranks, layout.decode_batch, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_tail_init_hidden",
            ),
            prefill_logits=shared_empty(
                (ranks, DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS, DEEPSEEK_V4_VOCAB_SIZE),
                torch.float32,
                name="mtp_prefill_logits",
            ),
            prefill_pre_hc_mirror=shared_empty(
                (ranks, layout.prefill_seq, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_prefill_pre_hc_mirror",
            ),
        )
        # MTP prefill host inputs + device outputs live on the prefill TaskArgs.
        from pypto_serving.model.deepseek.task_args import mtp_prefill_task_args  # noqa: PLC0415

        self._mtp_prefill_task_args = mtp_prefill_task_args(self, hidden)
        self._mtp_prefill_task_args.allocate_host_shared(None)
        self._mtp_buffers.prefill_kv_cache.zero_()
        return self._mtp_buffers

    def _stage_prefill_fwd_inputs(self, inputs: DeepSeekV4PreparedPrefillInputs) -> None:
        """Copy one prefill chunk's mutable metadata into shared host buffers.

        Request metadata is packed by rank and local B4 slot. The RoPE tables
        and cache pools remain worker-resident and are not staged here.
        """
        ta = self._prefill_task_args
        if ta is None:
            raise RuntimeError("DeepSeekV4 prefill TaskArgs are not staged")
        values = {
            "x_hc": inputs.x_hc,
            "ori_block_table": inputs.ori_block_table,
            "ori_slot_mapping": inputs.ori_slot_mapping,
            "hca_cmp_block_table": inputs.hca_cmp_block_table,
            "csa_cmp_block_table": inputs.csa_cmp_block_table,
            "idx_block_table": inputs.idx_block_table,
            "position_ids": inputs.position_ids,
            "hca_cmp_slot_mapping": inputs.hca_cmp_slot_mapping,
            "hca_state_slot_mapping": inputs.hca_state_slot_mapping,
            "csa_cmp_slot_mapping": inputs.csa_cmp_slot_mapping,
            "csa_idx_slot_mapping": inputs.csa_idx_slot_mapping,
            "csa_state_slot_mapping": inputs.csa_state_slot_mapping,
            "csa_inner_state_slot_mapping": inputs.csa_inner_state_slot_mapping,
            "input_ids": inputs.input_ids,
            "hca_compress_state_block_table": inputs.hca_compress_state_block_table,
            "csa_compress_state_block_table": inputs.csa_compress_state_block_table,
            "csa_inner_compress_state_block_table": inputs.csa_inner_compress_state_block_table,
            "num_tokens_per_owner": inputs.num_tokens_per_owner,
            "logit_row_indices": inputs.logit_row_indices,
        }
        ta.stage_for_tokens(values, inputs.kernel_tokens)

    def _retain_stacked_host_weights(
        self,
        weights: DeepSeekV4StackedLayerWeights,
    ) -> DeepSeekV4StackedLayerWeights:
        """Retain immutable layer-stacked weights for fork inheritance and resident upload."""
        host_weights = self._stacked_host_weights
        if host_weights is None:
            self._ensure_shared_host_allocation_before_worker("stacked layer weights")
            host_weights = dict(weights.tensors)
            self._stacked_host_weights = host_weights

        missing = sorted(set(weights.tensors) - set(host_weights))
        if missing:
            raise KeyError(f"DeepSeekV4 stacked Host weights are missing: {', '.join(missing)}")

        return DeepSeekV4StackedLayerWeights(tensors=host_weights)

    def _hc_head_tensors(self) -> dict[str, torch.Tensor]:
        """Return rank-replicated hc_head weights for the decode_fwd output collapse."""
        buffers = self._hc_head_buffers
        if buffers is not None:
            return buffers
        self._ensure_shared_host_allocation_before_worker("hc_head weights")
        global_weights = self.load_packed_global_weights()
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

    def _static_final_norm_weight_tensor(self) -> torch.Tensor:
        """Return the worker-resident per-rank final RMSNorm weight ``[ranks, D]``.

        Uses the model's final RMSNorm weight, rank-replicated and cast to bf16.
        """
        if self._static_final_norm_weight is None:
            global_weights = self.load_packed_global_weights()
            self._ensure_shared_host_allocation_before_worker("final_norm_w")
            final_norm_w = global_weights.final_norm_weight.to(torch.bfloat16).contiguous().cpu()
            self._static_final_norm_weight = self._static_device_tensor(self._rank_stack(final_norm_w))
        return self._static_final_norm_weight

    def _static_lm_head_weight_tensor(self) -> torch.Tensor:
        """Return the worker-visible LM-head weight, one vocab shard per DP rank.

        The kernel groups the DP world into ``ranks // tp`` independent TP groups,
        so every card consumes shard ``rank % tp``. Resident arguments are handed
        out per rank, so the shard has to be replicated here rather than indexed
        inside the kernel.
        """
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

    @staticmethod
    def _int32_scalar(value: int) -> int:
        return int(value)

    def _ensure_shared_host_allocation_before_worker(self, name: str) -> None:
        if self._l3_worker is not None:
            raise RuntimeError(
                f"DeepSeekV4 shared host buffer '{name}' must be allocated before the L3 worker starts"
            )

    @staticmethod
    def _upload_weight_group(
        worker: Any,
        host_weights: dict[str, torch.Tensor],
    ) -> dict[str, StackedDeviceTensor]:
        """Upload a rank-stacked weight group, rolling back partial allocation."""
        device_weights: dict[str, StackedDeviceTensor] = {}
        try:
            for name, tensor in host_weights.items():
                device_weights[name] = worker.alloc_stacked_tensor(tensor)
        except Exception:
            for tensor in device_weights.values():
                worker.free_stacked_tensor(tensor)
            raise
        return device_weights

    def _materialize_resident_weights(self) -> None:
        """Upload inherited weights once and release their parent-process Host references."""
        worker = self._shared_l3_worker()
        if self._global_weights is not None:
            hidden_size = int(self.load_packed_global_weights().embed_weight.shape[1])
            self._materialize_embedding_device_weight()
            # Allocate TaskArgs device-resident slots once on the init lane.
            # This must NOT happen lazily on the prepare lane: prepare runs
            # concurrently with in-flight dispatches in the depth-2 pipeline,
            # and a worker.alloc_tensor racing worker.run corrupts device state.
            for task_args in self._decode_task_args:
                task_args.allocate_device(worker, None)
            if self._compiled.num_speculative_tokens:
                for task_args in self._mtp_decode_task_args:
                    task_args.allocate_device(worker, None)
                self._materialize_mtp_tail_pre_hc_pool(hidden_size)
                self._mtp_prefill_task_args.allocate_device(worker, None)
                self._materialize_mtp_device_state_tokens()
                self._materialize_mtp_device_state_meta()
        if self._stacked_device_weights is None:
            host_weights = self._stacked_host_weights
            if not host_weights:
                raise RuntimeError("DeepSeekV4 stacked Host weights are not retained")
            parent_host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in host_weights.values())
            with profile_span("DeepSeekV4ModelRunner.upload_resident_main_weights", cat="executor"):
                self._stacked_device_weights = self._upload_weight_group(worker, host_weights)
            self._compiled.prepacked_layer_weights = None
            self._stacked_host_weights = None
            logger.info(
                "DeepSeekV4 resident main weights uploaded; released_parent_host_bytes=%d",
                parent_host_bytes,
            )

        buffers = self._mtp_buffers
        if buffers is not None and self._mtp_device_weights is None:
            if not buffers.weights:
                raise RuntimeError("DeepSeekV4 MTP Host weights are not staged")
            parent_host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in buffers.weights.values())
            with profile_span("DeepSeekV4ModelRunner.upload_resident_mtp_weights", cat="executor"):
                self._mtp_device_weights = self._upload_weight_group(worker, buffers.weights)
            buffers.weights.clear()
            logger.info(
                "DeepSeekV4 resident MTP weights uploaded; released_parent_host_bytes=%d",
                parent_host_bytes,
            )
        worker.release_inherited_host_tensor_refs()

    def _shared_l3_worker(self) -> Any:
        worker = self._l3_worker
        if worker is None:
            compiled_callables = self._compiled.l3_callables()
            if not compiled_callables:
                raise RuntimeError("DeepSeekV4 L3 callables are not compiled")
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            compiled = [callable_spec.compiled for callable_spec in compiled_callables]
            with profile_span(
                "DeepSeekV4ModelRunner.create_persistent_l3_worker",
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
                    # Materialize the full ring arena before KV sizing reads free HBM.
                    worker_kwargs["config"] = run_config
                worker = DistributedWorker(compiled, **worker_kwargs)
            self._l3_worker = worker
        return worker

    def _inherited_host_weights(self) -> list[torch.Tensor]:
        """Return the resident main and MTP weights that must be visible at worker fork.

        Passing this list to ``inherited_host_tensors`` is a guarantee, not a hint: pypto
        names these ranges in place instead of staging a full host copy of them, on our word
        that every one is backed by a mapping visible across processes and that the mapping
        outlives the worker. It cannot check that — the prepacked sidecar is ``MAP_SHARED``
        but reaches torch through ``mmap`` + ``from_numpy``, so ``torch.is_shared()`` reports
        ``False`` for all of it, and pypto warns once at prepare time that it is trusting us.

        So anything added here must be backed by shared memory or a shared file mapping, and
        written before the fork and never after. A ``MAP_PRIVATE`` tensor in this list does
        not fail loudly: copy-on-write leaves the child reading its pre-fork snapshot and the
        upload carries stale bytes.
        """
        tensors = list(self._stacked_host_weights.values()) if self._stacked_host_weights else []
        global_weights = getattr(self, "_global_weights", None)
        if global_weights is not None:
            tensors.append(global_weights.embed_weight)
        if self._mtp_buffers is not None:
            tensors.extend(self._mtp_buffers.weights.values())
        return tensors

    def _alloc_empty_stacked_tensor(
        self,
        full_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> StackedDeviceTensor:
        """Allocate an uninitialized shard directly on every chip worker."""
        worker = self._shared_l3_worker()
        worker_ids = tuple(range(self._compiled.layout.ranks))
        shards: list[DeviceTensor] = []
        try:
            for worker_id in worker_ids:
                shards.append(
                    worker.alloc_tensor(
                        full_shape[1:],
                        dtype,
                        worker_id=worker_id,
                    )
                )
        except Exception:
            for shard, worker_id in zip(shards, worker_ids, strict=False):
                worker.free_tensor(shard, worker_id=worker_id)
            raise
        return StackedDeviceTensor(shards, full_shape, worker_ids)

    def _materialize_embedding_device_weight(self) -> StackedDeviceTensor:
        """Upload one full embedding table to every decode rank."""
        stacked = self._embedding_device_weight
        if stacked is not None:
            return stacked
        source = self.load_packed_global_weights().embed_weight
        if source.device.type != "cpu" or source.dtype != torch.bfloat16 or not source.is_contiguous():
            raise ValueError("DeepSeekV4 embedding weight must be contiguous BF16 CPU storage before worker fork")
        worker = self._shared_l3_worker()
        worker_ids = tuple(range(self._compiled.layout.ranks))
        shards: list[DeviceTensor] = []
        try:
            for worker_id in worker_ids:
                shards.append(worker.alloc_tensor(source.shape, source.dtype, init=source, worker_id=worker_id))
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

    def _materialize_main_pre_hc_device(self, hidden_size: int) -> StackedDeviceTensor:
        """Return the main decode pre-HC output (a decode-TaskArgs device slot).

        The device slot is allocated once at init in ``_materialize_resident_weights``.
        The MTP decode TaskArgs reads it here as the strip-shared
        ``main_pre_hc_hidden`` lazy source (its value is irrelevant on the fused
        path -- the kernel takes it from the main decode tuple).  Must not touch
        the worker -- this runs on the prepare lane, concurrently with dispatches.
        """
        return self._decode_task_args[0].tensors["pre_hc_hidden_out"]

    def _materialize_mtp_tail_pre_hc_pool(self, hidden_size: int) -> StackedDeviceTensor:
        """Allocate persistent request tail hidden slots on every rank."""
        stacked = self._mtp_tail_pre_hc_pool
        if stacked is None:
            layout = self._compiled.layout
            stacked = self._alloc_empty_stacked_tensor(
                (layout.ranks, layout.decode_batch, layout.hc_mult, int(hidden_size)),
                torch.float32,
            )
            self._mtp_tail_pre_hc_pool = stacked
        return stacked

    def _read_mtp_prefill_logits(self, owner_rank: int) -> torch.Tensor:
        """Read only the valid owner logits row needed for Host argmax."""
        buffers = self._require_mtp_buffers()
        dev_logits = self._mtp_prefill_task_args.tensors["logits"]
        host_row = buffers.prefill_logits[owner_rank, 0]
        shard = dev_logits.shards[owner_rank]
        worker_id = dev_logits.worker_ids[owner_rank]
        self._shared_l3_worker().copy_from(
            host_row.data_ptr(),
            shard.data_ptr,
            host_row.numel() * host_row.element_size(),
            worker_id=worker_id,
        )
        return host_row

    def _read_mtp_prefill_pre_hc(self, owner_rank: int, *, row: int) -> torch.Tensor:
        """Read one recurrent hidden row needed by arbitrary-depth MTP."""
        buffers = self._require_mtp_buffers()
        device_pre_hc = self._mtp_prefill_task_args.tensors["pre_hc_hidden_out"]
        host_row = buffers.prefill_pre_hc_mirror[owner_rank, row]
        shard = device_pre_hc.shards[owner_rank]
        worker_id = device_pre_hc.worker_ids[owner_rank]
        row_nbytes = host_row.numel() * host_row.element_size()
        self._shared_l3_worker().copy_from(
            host_row.data_ptr(),
            shard.data_ptr,
            row_nbytes,
            src_offset=row * row_nbytes,
            worker_id=worker_id,
        )
        return host_row.detach().cpu().clone()

    def _materialize_mtp_device_state_tokens(self) -> StackedDeviceTensor:
        """Allocate persistent tail/draft token slots on every rank."""
        stacked = self._mtp_device_state_tokens
        if stacked is None:
            layout = self._compiled.layout
            stacked = self._alloc_empty_stacked_tensor(
                (layout.ranks, layout.decode_batch, _MTP_DEVICE_STATE_TOKEN_WIDTH),
                torch.long,
            )
            self._mtp_device_state_tokens = stacked
        return stacked

    def _materialize_mtp_device_state_meta(self) -> StackedDeviceTensor:
        """Allocate persistent generation, position, and commit metadata."""
        stacked = self._mtp_device_state_meta
        if stacked is None:
            layout = self._compiled.layout
            stacked = self._alloc_empty_stacked_tensor(
                (layout.ranks, layout.decode_batch, _MTP_DEVICE_STATE_META_WIDTH),
                torch.int32,
            )
            self._mtp_device_state_meta = stacked
        return stacked

    def _materialize_decode_device_cache(self) -> DeepSeekV4DeviceCache:
        """Allocate dynamically sized cache shards directly on each NPU."""
        cache = self._decode_device_cache
        if cache is not None:
            return cache
        worker = self._shared_l3_worker()
        allocated: list[StackedDeviceTensor] = []
        layout = self._compiled.layout

        def resident(shape: tuple[int, ...], dtype: torch.dtype) -> StackedDeviceTensor:
            stacked = self._alloc_empty_stacked_tensor(shape, dtype)
            allocated.append(stacked)
            return stacked

        try:
            cache = DeepSeekV4DeviceCache(
                kv_cache=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_FWD_NUM_LAYERS * self._physical_cache_num_blocks("ori"),
                        layout.block_size,
                        1,
                        DEEPSEEK_V4_HEAD_DIM,
                    ),
                    torch.bfloat16,
                ),
                hca_cmp_kv=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_HCA_NUM_LAYERS
                        * self._physical_cache_num_blocks("cmp_c128"),
                        layout.block_size // 128,
                        1,
                        DEEPSEEK_V4_HEAD_DIM,
                    ),
                    torch.bfloat16,
                ),
                csa_cmp_kv=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS
                        * self._physical_cache_num_blocks("cmp_c4"),
                        layout.block_size // 4,
                        1,
                        DEEPSEEK_V4_HEAD_DIM,
                    ),
                    torch.bfloat16,
                ),
                idx_kv_cache=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS
                        * self._physical_cache_num_blocks("idx"),
                        layout.block_size // 4,
                        1,
                        DEEPSEEK_V4_IDX_HEAD_DIM,
                    ),
                    torch.int8,
                ),
                idx_kv_scale=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS
                        * self._physical_cache_num_blocks("idx"),
                        layout.block_size // 4,
                        1,
                        1,
                    ),
                    torch.float32,
                ),
                hca_compress_state=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_HCA_NUM_LAYERS * self._physical_cache_num_blocks("hca_state"),
                        layout.c128_state_block_size,
                        DEEPSEEK_V4_HCA_STATE_DIM,
                    ),
                    torch.float32,
                ),
                csa_compress_state=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS * self._physical_cache_num_blocks("csa_state"),
                        layout.c4_state_block_size,
                        DEEPSEEK_V4_CSA_STATE_DIM,
                    ),
                    torch.float32,
                ),
                csa_inner_compress_state=resident(
                    (
                        layout.ranks,
                        DEEPSEEK_V4_CSA_NUM_LAYERS
                        * self._physical_cache_num_blocks("csa_inner_state"),
                        layout.c4_state_block_size,
                        DEEPSEEK_V4_CSA_INNER_STATE_DIM,
                    ),
                    torch.float32,
                ),
            )
        except Exception:
            for tensor in allocated:
                worker.free_stacked_tensor(tensor)
            raise
        self._decode_device_cache = cache
        return cache

    def _materialize_mtp_device_kv_cache(self) -> StackedDeviceTensor | None:
        """Materialize the optional MTP cache once for both MTP kernels."""
        cache = self._mtp_device_kv_cache
        if cache is not None:
            return cache
        buffers = self._mtp_buffers
        if buffers is None:
            return None
        layout = self._compiled.layout
        cache = self._alloc_empty_stacked_tensor(
            (
                layout.ranks,
                self._physical_cache_num_blocks("ori"),
                layout.block_size,
                1,
                DEEPSEEK_V4_HEAD_DIM,
            ),
            torch.bfloat16,
        )
        self._mtp_device_kv_cache = cache
        return cache

    def _free_device_caches(self) -> None:
        worker = self._l3_worker
        if worker is None:
            self._decode_device_cache = None
            self._mtp_device_kv_cache = None
            return
        cache = self._decode_device_cache
        if cache is not None:
            for tensor in (
                cache.kv_cache,
                cache.hca_cmp_kv,
                cache.csa_cmp_kv,
                cache.idx_kv_cache,
                cache.idx_kv_scale,
                cache.hca_compress_state,
                cache.csa_compress_state,
                cache.csa_inner_compress_state,
            ):
                worker.free_stacked_tensor(tensor)
        if self._mtp_device_kv_cache is not None:
            worker.free_stacked_tensor(self._mtp_device_kv_cache)
        self._decode_device_cache = None
        self._mtp_device_kv_cache = None

    @staticmethod
    def _static_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        return share_cpu_tensor(tensor)

    def close(self) -> None:
        worker = self._l3_worker
        try:
            if worker is not None:
                worker.close()
        finally:
            self._l3_worker = None
            self._cache_group_num_blocks.clear()
            self._stacked_host_weights = None
            self._stacked_device_weights = None
            self._embedding_device_weight = None
            self._mtp_tail_pre_hc_pool = None
            self._mtp_device_state_tokens = None
            self._mtp_device_state_meta = None
            self._l3_shared_buffers_ready = False
            self._mtp_device_weights = None
            if self._mtp_prefill_task_args is not None:
                self._mtp_prefill_task_args.close()
                self._mtp_prefill_task_args = None
            self._mtp_buffers = None
            self._global_weights = None
            self._decode_device_cache = None
            self._mtp_device_kv_cache = None
            self._main_pre_hc_host_mirror = None
            self._mtp_request_states.clear()
            with self._pending_mtp_dispatch_lock:
                self._pending_mtp_dispatches.clear()
            self._l3_static_tensors.clear()
            for task_args in self._decode_task_args:
                task_args.close()
            self._decode_task_args = []
            for task_args in self._mtp_decode_task_args:
                task_args.close()
            self._mtp_decode_task_args = []
            self._decode_input_slots = []
            if self._prefill_task_args is not None:
                self._prefill_task_args.close()
                self._prefill_task_args = None

    def _require_input_builder(self) -> DeepSeekV4InputBuilder:
        if self.input_builder is None:
            raise RuntimeError("DeepSeekV4 input builder is not initialized")
        return self.input_builder

    def _rank_stack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0).expand(self._compiled.layout.ranks, *tensor.shape).contiguous()

    def _rank_slot_scatter(
        self,
        tensors: Sequence[torch.Tensor],
        ranks: Sequence[int],
        local_rows: Sequence[int],
    ) -> torch.Tensor:
        """Place distinct tensors in fixed rank-local prefill slots."""
        if (
            len(tensors) == 0
            or len(tensors) != len(ranks)
            or len(tensors) != len(local_rows)
        ):
            raise ValueError("rank/slot-scattered tensors and coordinates must be aligned")
        reference = tensors[0]
        if any(tensor.shape != reference.shape or tensor.dtype != reference.dtype for tensor in tensors):
            raise ValueError("rank/slot-scattered tensors must have identical shapes and dtypes")
        result = reference.reshape(1, 1, *reference.shape).expand(
            self._compiled.layout.ranks,
            self._compiled.layout.prefill_batch,
            *reference.shape,
        ).clone()
        for rank, local_row, tensor in zip(ranks, local_rows, tensors, strict=True):
            result[int(rank), int(local_row)].copy_(tensor)
        return result.contiguous()

    def _rank_slot_scatter_mappings(
        self,
        tensors: Sequence[torch.Tensor],
        ranks: Sequence[int],
        local_rows: Sequence[int],
    ) -> torch.Tensor:
        """Scatter cache-write mappings and disable every inactive slot."""
        if (
            len(tensors) == 0
            or len(tensors) != len(ranks)
            or len(tensors) != len(local_rows)
        ):
            raise ValueError("rank/slot-scattered mappings and coordinates must be aligned")
        reference = tensors[0]
        if any(tensor.shape != reference.shape or tensor.dtype != reference.dtype for tensor in tensors):
            raise ValueError("rank/slot-scattered mappings must have identical shapes and dtypes")
        result = torch.full(
            (
                self._compiled.layout.ranks,
                self._compiled.layout.prefill_batch,
                *reference.shape,
            ),
            -1,
            dtype=reference.dtype,
        )
        for rank, local_row, tensor in zip(ranks, local_rows, tensors, strict=True):
            result[int(rank), int(local_row)].copy_(tensor)
        return result.contiguous()

    def _prefill_assignment(self, ranks: Sequence[int]) -> _DeepSeekV4DecodeAssignment:
        """Assign scheduler requests to the fixed rank-local B4 prefill rows."""
        layout = self._compiled.layout
        indices_by_rank: list[list[int]] = [[] for _ in range(layout.ranks)]
        local_rows = [0] * len(ranks)
        for request_index, rank in enumerate(ranks):
            local_row = len(indices_by_rank[int(rank)])
            if local_row >= layout.prefill_batch:
                raise ValueError(
                    f"DeepSeekV4 rank {rank} prefill batch exceeds local capacity "
                    f"{layout.prefill_batch}"
                )
            local_rows[request_index] = local_row
            indices_by_rank[int(rank)].append(request_index)
        return _DeepSeekV4DecodeAssignment(
            ranks=tuple(int(rank) for rank in ranks),
            local_rows=tuple(local_rows),
            per_rank_counts=tuple(len(indices) for indices in indices_by_rank),
            indices_by_rank=tuple(tuple(indices) for indices in indices_by_rank),
        )

    def _prefill_active_token_limit(self, runtime: RuntimeConfig | None) -> int:
        """Return the configured active-token limit for one main-prefill call."""
        limits = [int(self._compiled.kernel_contract.max_prefill_tokens_per_request)]
        if runtime is not None:
            limits.append(int(runtime.max_num_batched_tokens))
            limits.append(int(runtime.max_seq_len))
            if runtime.max_prefill_tokens_per_request is not None:
                limits.append(int(runtime.max_prefill_tokens_per_request))
        if any(limit <= 0 for limit in limits):
            raise ValueError("DeepSeekV4 prefill token limits must be positive")
        return min(limits)

    def _prefill_buffer_tokens(self) -> int:
        """Size mutable shared storage before the persistent worker forks."""
        runtime_model = self._compiled.runtime_model
        if runtime_model is None:
            return self._compiled.layout.prefill_seq
        return self._prefill_kernel_tokens(
            self._prefill_active_token_limit(runtime_model.runtime),
            runtime=runtime_model.runtime,
        )

    def _prefill_kernel_tokens(
        self,
        actual_tokens: int,
        *,
        runtime: RuntimeConfig | None = None,
    ) -> int:
        if actual_tokens <= 0:
            raise ValueError("actual_tokens must be positive")
        active_limit = self._prefill_active_token_limit(runtime)
        if actual_tokens > active_limit:
            mode = "MTP" if self._compiled.num_speculative_tokens else "main"
            raise ValueError(
                f"DeepSeekV4 {mode} prefill received {actual_tokens} active tokens; "
                f"the configured single-request limit is {active_limit}"
            )
        return int(self._compiled.kernel_contract.padded_prefill_tokens(actual_tokens))

    @staticmethod
    def _prefill_kernel_positions(
        positions: Sequence[int],
        *,
        kernel_tokens: int,
        max_seq_len: int,
    ) -> list[int]:
        if len(positions) <= 0:
            raise ValueError("positions must not be empty")
        if kernel_tokens < len(positions):
            raise ValueError("kernel_tokens must cover all active positions")
        active_positions = [int(position) for position in positions]
        if active_positions[0] < 0 or active_positions[-1] >= max_seq_len:
            raise ValueError(
                f"prefill active positions [{active_positions[0]}, {active_positions[-1]}] "
                f"exceed max_seq_len={max_seq_len}"
            )
        kernel_positions = list(active_positions)
        next_position = active_positions[-1] + 1
        while len(kernel_positions) < kernel_tokens:
            kernel_positions.append(min(next_position, max_seq_len - 1))
            next_position += 1
        return kernel_positions

    @staticmethod
    def _padded_rows(values: torch.Tensor, length: int) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(f"values must be rank-2, got shape={tuple(values.shape)}")
        if values.shape[0] <= 0:
            raise ValueError("values must not be empty")
        if values.shape[0] > length:
            raise ValueError(f"values rows {values.shape[0]} exceed padded length {length}")
        out = torch.empty((length, values.shape[1]), dtype=values.dtype, device=values.device)
        out[: values.shape[0]].copy_(values)
        if values.shape[0] < length:
            pad_rows = torch.arange(values.shape[0], length, device=values.device) % values.shape[0]
            out[values.shape[0] :].copy_(values.index_select(0, pad_rows))
        return out

    @staticmethod
    def _padded_vector(values: torch.Tensor, length: int, *, dtype: torch.dtype) -> torch.Tensor:
        if values.numel() <= 0:
            raise ValueError("values must not be empty")
        if values.numel() > length:
            raise ValueError(f"values length {values.numel()} exceeds padded length {length}")
        out = torch.empty((length,), dtype=dtype)
        out[: values.numel()] = values.to(dtype=dtype)
        if values.numel() < length:
            pad_rows = torch.arange(values.numel(), length) % values.numel()
            out[values.numel() :] = values.to(dtype=dtype).index_select(0, pad_rows)
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
    def _pad_prefill_mappings(mappings: torch.Tensor, length: int) -> torch.Tensor:
        if mappings.ndim != 2:
            raise ValueError(f"prefill mappings must be rank-2, got shape={tuple(mappings.shape)}")
        if mappings.shape[1] > length:
            raise ValueError(
                f"prefill mapping width {mappings.shape[1]} exceeds padded length {length}"
            )
        out = torch.full((mappings.shape[0], length), -1, dtype=mappings.dtype)
        out[:, : mappings.shape[1]].copy_(mappings)
        return out

    def _decode_assignment(self, batch: DecodeBatch) -> _DeepSeekV4DecodeAssignment:
        """Assign scheduler rows to the fixed rank-local decode tile."""
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if actual_batch <= 0:
            raise ValueError("DeepSeekV4 decode batch must not be empty")
        if len(batch.cache_partitions) != actual_batch:
            raise ValueError("DeepSeekV4 decode requires one cache partition per request")
        ranks = tuple(int(rank) for rank in batch.cache_partitions)
        if min(ranks) < 0 or max(ranks) >= layout.ranks:
            raise ValueError(f"DeepSeekV4 decode cache partitions must be in [0, {layout.ranks - 1}]")
        indices_by_rank: list[list[int]] = [[] for _ in range(layout.ranks)]
        local_rows = [0] * actual_batch
        for request_index, rank in enumerate(ranks):
            local_row = len(indices_by_rank[rank])
            if local_row >= layout.decode_batch:
                raise ValueError(
                    f"DeepSeekV4 rank {rank} decode batch exceeds local capacity {layout.decode_batch}"
                )
            local_rows[request_index] = local_row
            indices_by_rank[rank].append(request_index)
        return _DeepSeekV4DecodeAssignment(
            ranks=ranks,
            local_rows=tuple(local_rows),
            per_rank_counts=tuple(len(indices) for indices in indices_by_rank),
            indices_by_rank=tuple(tuple(indices) for indices in indices_by_rank),
        )

    def _indices_by_rank(self, ranks: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        """Reconstruct scheduler-row ownership for an immutable prepared slot."""
        indices: list[list[int]] = [[] for _ in range(self._compiled.layout.ranks)]
        for request_index, rank in enumerate(ranks):
            indices[int(rank)].append(request_index)
        return tuple(tuple(rows) for rows in indices)

    def _autoregressive_decode_positions(
        self,
        batch: DecodeBatch,
        actual_batch: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the single current position for each autoregressive request."""
        decode_seq = self._compiled.layout.decode_seq
        positions = []
        for row in range(actual_batch):
            seq_len = int(batch.seq_lens[row].item())
            if seq_len < 1:
                raise ValueError("decode seq_lens must be positive")
            positions.append((seq_len - 1,) * decode_seq)
        return tuple(positions)

    def _autoregressive_decode_token_rows(
        self,
        token_ids: torch.Tensor,
        actual_batch: int,
    ) -> torch.Tensor:
        """Expand one current token per request to the fixed sequence width."""
        layout = self._compiled.layout
        token_ids = token_ids.detach().cpu().to(torch.long)
        if token_ids.ndim == 1:
            active = token_ids[:actual_batch].reshape(actual_batch, 1)
        else:
            active = token_ids[:actual_batch, :1]
        return active.expand(actual_batch, layout.decode_seq).clone()
