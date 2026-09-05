# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark per-dispatch-class :class:`TaskArgs` builders.

The ``_PREFILL_TENSOR_ORDER`` / ``_DECODE_TENSOR_ORDER`` tuples below ARE the
positional contracts of ``l3_prefill_fwd`` (100 args) and ``l3_decode_fwd``
(107 args) -- register them in exactly this order.  Every argument declares its
kind at registration: host-shared slots (per-step metadata), static weights
(upload-once), worker-resident cache pools and scratch (runner materializers),
and the stacked layer-weight banks.

Both prefill and decode dispatch their kernel-validated physical extents.
Prefill binds the TP-aligned packed P/L prefixes over maximum-sized backing
allocations. Decode always passes the full 16-request local tile because
sub-tile KV tails are not a supported device contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from pypto_serving.model.common.runner.buffer_set import Placement, Slot
from pypto_serving.model.common.runner.task_args import TaskArgs
from pypto_serving.model.deepseek_dspark.npu_runner import (
    DSPARK_DECODE_BATCH,
    DSPARK_DECODE_CMP_C4_TABLE_BLOCKS,
    DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS,
    DSPARK_DECODE_HCA_STATE_TABLE_BLOCKS,
    DSPARK_DECODE_IDX_TABLE_BLOCKS,
    DSPARK_DECODE_LOCAL_BATCH,
    DSPARK_DECODE_LOCAL_TOKENS,
    DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS,
    DSPARK_DECODE_TOKENS,
    DSPARK_HC_MULT,
    DSPARK_HIDDEN_SIZE,
    DSPARK_MAX_LOGIT_ROWS,
    DSPARK_MOE_TOKENS,
    DSPARK_PREFILL_CSA_CMP_TABLE_BLOCKS,
    DSPARK_PREFILL_CSA_INNER_STATE_TABLE_BLOCKS,
    DSPARK_PREFILL_CSA_STATE_TABLE_BLOCKS,
    DSPARK_PREFILL_HCA_CMP_TABLE_BLOCKS,
    DSPARK_PREFILL_HCA_STATE_TABLE_BLOCKS,
    DSPARK_PREFILL_IDX_TABLE_BLOCKS,
    DSPARK_PREFILL_MAX_REQUESTS,
    DSPARK_PREFILL_ORI_TABLE_BLOCKS,
    DSPARK_ROPE_HEAD_DIM,
    DSPARK_SAMPLED_IDS_PAD,
    DSPARK_SLIDING_WINDOW,
    DSPARK_VOCAB_SIZE,
)
from pypto_serving.model.deepseek_dspark.weight_spec import (  # noqa: PLC0415 -- packing dims
    DSPARK_O_GROUPS,
    DSPARK_O_LORA,
)

if TYPE_CHECKING:
    from pypto_serving.model.deepseek_dspark.npu_runner import DSparkModelRunner

__all__ = ["decode_task_args", "prefill_task_args"]

# ---- shared source name sets ----
# Stacked layer-weight bank names shared by prefill and decode.
_PREFILL_STATIC_WEIGHT_NAMES = ("hc_head_fn", "hc_head_scale", "hc_head_base",
                                "final_norm_w", "lm_head_weight")
_CACHE_POOL_NAMES = (
    "kv_cache",
    "hca_cmp_kv",
    "csa_cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
    "hca_compress_state",
    "csa_compress_state",
    "csa_inner_compress_state",
)


def _prefill_slots(layout) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Host-shared prefill slot name -> (dtype, full shape)."""
    ranks = layout.ranks
    tokens = layout.prefill_tokens
    local_tokens = layout.prefill_local_tokens
    group_rope = (tokens, DSPARK_ROPE_HEAD_DIM)
    slot_specs = {
        "x_hc": (torch.float32, (ranks, tokens, DSPARK_HC_MULT, layout.hidden_size)),
        # Packed-prefill boundaries (pypto-lib#1095): one entry per request plus
        # the terminal logical length.
        "query_start_loc": (
            torch.int32, (ranks, DSPARK_PREFILL_MAX_REQUESTS + 1),
        ),
        "hca_compress_state_block_table": (
            torch.int32,
            (ranks, DSPARK_PREFILL_MAX_REQUESTS, DSPARK_PREFILL_HCA_STATE_TABLE_BLOCKS),
        ),
        "csa_compress_state_block_table": (
            torch.int32,
            (ranks, DSPARK_PREFILL_MAX_REQUESTS, DSPARK_PREFILL_CSA_STATE_TABLE_BLOCKS),
        ),
        "csa_inner_compress_state_block_table": (
            torch.int32,
            (
                ranks,
                DSPARK_PREFILL_MAX_REQUESTS,
                DSPARK_PREFILL_CSA_INNER_STATE_TABLE_BLOCKS,
            ),
        ),
        "swa_freqs_cos": (torch.bfloat16, (ranks, *group_rope)),
        "swa_freqs_sin": (torch.bfloat16, (ranks, *group_rope)),
        "compressed_freqs_cos": (torch.bfloat16, (ranks, *group_rope)),
        "compressed_freqs_sin": (torch.bfloat16, (ranks, *group_rope)),
        "hca_cmp_freqs_cos": (torch.bfloat16, (ranks, *group_rope)),
        "hca_cmp_freqs_sin": (torch.bfloat16, (ranks, *group_rope)),
        "csa_cmp_freqs_cos": (torch.bfloat16, (ranks, *group_rope)),
        "csa_cmp_freqs_sin": (torch.bfloat16, (ranks, *group_rope)),
        "ori_block_table": (
            torch.int32, (ranks, DSPARK_PREFILL_MAX_REQUESTS, DSPARK_PREFILL_ORI_TABLE_BLOCKS),
        ),
        "hca_cmp_block_table": (
            torch.int32,
            (ranks, DSPARK_PREFILL_MAX_REQUESTS, DSPARK_PREFILL_HCA_CMP_TABLE_BLOCKS),
        ),
        "csa_cmp_block_table": (
            torch.int32,
            (ranks, DSPARK_PREFILL_MAX_REQUESTS, DSPARK_PREFILL_CSA_CMP_TABLE_BLOCKS),
        ),
        "idx_block_table": (
            torch.int32, (ranks, DSPARK_PREFILL_MAX_REQUESTS, DSPARK_PREFILL_IDX_TABLE_BLOCKS),
        ),
        "ori_slot_mapping_full": (torch.int64, (ranks, tokens)),
        "position_ids_local": (torch.int32, (ranks, local_tokens)),
        "position_ids_full": (torch.int32, (ranks, tokens)),
        "input_ids": (torch.int64, (ranks, local_tokens)),
        "hca_cmp_slot_mapping_full": (torch.int64, (ranks, tokens)),
        "hca_state_slot_mapping_full": (torch.int64, (ranks, tokens)),
        "csa_cmp_slot_mapping_full": (torch.int64, (ranks, tokens)),
        "csa_idx_slot_mapping_full": (torch.int64, (ranks, tokens)),
        "csa_state_slot_mapping_full": (torch.int64, (ranks, tokens)),
        "csa_inner_state_slot_mapping_full": (torch.int64, (ranks, tokens)),
        "logit_row_indices": (torch.int32, (ranks, DSPARK_MAX_LOGIT_ROWS)),
        "sampled_ids": (
            torch.int32, (ranks, DSPARK_MAX_LOGIT_ROWS, DSPARK_SAMPLED_IDS_PAD),
        ),
    }
    return slot_specs


def _prefill_scratch_sources(runner: DSparkModelRunner) -> dict[str, Any]:
    """Zero-initialized device-resident prefill staging buffers."""
    layout = runner._compiled.layout
    ranks = layout.ranks
    hidden = layout.hidden_size
    tokens = layout.prefill_tokens
    local_tokens = layout.prefill_local_tokens
    shapes = {
        "o_proj_wo_a_full": (
            (ranks, DSPARK_O_GROUPS, DSPARK_O_LORA, 4096), torch.bfloat16,
        ),
        "o_proj_wo_b_full": (
            (ranks, hidden, DSPARK_O_GROUPS * DSPARK_O_LORA), torch.int8,
        ),
        "attn_stage": ((ranks, tokens, DSPARK_HC_MULT, hidden), torch.float32),
        "x_mixed": ((ranks, tokens, hidden), torch.bfloat16),
        "post_ffn": ((ranks, tokens, DSPARK_HC_MULT), torch.float32),
        "comb_ffn": ((ranks, tokens, DSPARK_HC_MULT * DSPARK_HC_MULT), torch.float32),
        "ffn_out": ((ranks, local_tokens, hidden), torch.bfloat16),
        "hidden_workspace": ((ranks, tokens, hidden), torch.bfloat16),
        "x_out": ((ranks, tokens, hidden), torch.bfloat16),
        "logits": (
            (ranks, DSPARK_MAX_LOGIT_ROWS, DSPARK_VOCAB_SIZE), torch.float32,
        ),
    }
    return {
        name: (
            lambda n=name, s=shape: runner._alloc_zeroed_stacked_tensor(
                n, s[0], s[1], scope="prefill"
            )
        )
        for name, shape in shapes.items()
    }


def prefill_task_args(runner: DSparkModelRunner) -> TaskArgs:
    """Build the ``TaskArgs`` for the packed ``l3_prefill_fwd`` dispatch."""
    layout = runner._compiled.layout
    slot_specs = _prefill_slots(layout)
    static_weights = set(_PREFILL_STATIC_WEIGHT_NAMES)
    cache_pools = set(_CACHE_POOL_NAMES)
    scratch = _prefill_scratch_sources(runner)

    ta = TaskArgs(stacked=True)
    for name in _PREFILL_TENSOR_ORDER:
        if name in slot_specs:
            dtype, shape = slot_specs[name]
            ta.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, s=shape: s))
        elif name == "hc_attn_fn" or name == "hc_ffn_fn":
            ta.add_arg(name, lambda n=name: runner._require_stacked_weights(prefill=True)[n])
        elif name in static_weights:
            ta.add_arg(name, lambda n=name: runner._static_weight(n))
        elif name in cache_pools:
            ta.add_arg(name, lambda n=name: runner._device_cache_values()[n])
        elif name in scratch:
            ta.add_arg(name, scratch[name])
        else:
            ta.add_arg(name, lambda n=name: runner._require_stacked_weights()[n])
    return ta


def _decode_slots(layout) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Host-shared decode slot name -> (dtype, full shape)."""
    ranks = layout.ranks
    local_batch = DSPARK_DECODE_LOCAL_BATCH
    group_batch = DSPARK_DECODE_BATCH
    local_tokens = DSPARK_DECODE_LOCAL_TOKENS
    group_tokens = DSPARK_DECODE_TOKENS
    rope = (DSPARK_ROPE_HEAD_DIM,)
    window = (DSPARK_SLIDING_WINDOW,)
    slot_specs = {
        "freqs_cos_local": (torch.bfloat16, (ranks, local_tokens, *rope)),
        "freqs_sin_local": (torch.bfloat16, (ranks, local_tokens, *rope)),
        "freqs_cos": (torch.bfloat16, (ranks, group_tokens, *rope)),
        "freqs_sin": (torch.bfloat16, (ranks, group_tokens, *rope)),
        "compressed_freqs_cos_local": (
            torch.bfloat16, (ranks, local_tokens, *rope),
        ),
        "compressed_freqs_sin_local": (
            torch.bfloat16, (ranks, local_tokens, *rope),
        ),
        "compressed_freqs_cos": (torch.bfloat16, (ranks, group_tokens, *rope)),
        "compressed_freqs_sin": (torch.bfloat16, (ranks, group_tokens, *rope)),
        "swa_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "swa_indices": (torch.int32, (ranks, local_tokens, *window)),
        "swa_lens": (torch.int32, (ranks, local_tokens)),
        "position_ids_local": (torch.int32, (ranks, local_tokens)),
        "position_ids": (torch.int32, (ranks, group_tokens)),
        "csa_cmp_freqs_cos": (torch.bfloat16, (ranks, group_tokens, *rope)),
        "csa_cmp_freqs_sin": (torch.bfloat16, (ranks, group_tokens, *rope)),
        "csa_compress_state_block_table": (
            torch.int32, (ranks, group_batch, DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS),
        ),
        "csa_inner_compress_state_block_table": (
            torch.int32, (ranks, group_batch, DSPARK_DECODE_CSA_STATE_TABLE_BLOCKS),
        ),
        "csa_cmp_block_table": (
            torch.int32, (ranks, local_batch, DSPARK_DECODE_CMP_C4_TABLE_BLOCKS),
        ),
        "csa_idx_block_table": (
            torch.int32, (ranks, local_batch, DSPARK_DECODE_IDX_TABLE_BLOCKS),
        ),
        "csa_ori_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "csa_window_swa_indices": (torch.int32, (ranks, local_tokens, *window)),
        "csa_window_swa_lens": (torch.int32, (ranks, local_tokens)),
        "csa_cmp_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "csa_idx_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "csa_state_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "csa_inner_state_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "csa_kv_seq_lens": (torch.int32, (ranks, local_batch)),
        "hca_cmp_freqs_cos": (
            torch.float32, (ranks, group_batch, DSPARK_ROPE_HEAD_DIM // 2),
        ),
        "hca_cmp_freqs_sin": (
            torch.float32, (ranks, group_batch, DSPARK_ROPE_HEAD_DIM // 2),
        ),
        "hca_compress_state_block_table": (
            torch.int32, (ranks, group_batch, DSPARK_DECODE_HCA_STATE_TABLE_BLOCKS),
        ),
        "hca_cmp_block_table": (
            torch.int32, (ranks, local_batch, DSPARK_DECODE_HCA_CMP_TABLE_BLOCKS),
        ),
        "hca_ori_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "hca_window_swa_indices": (torch.int32, (ranks, local_tokens, *window)),
        "hca_window_swa_lens": (torch.int32, (ranks, local_tokens)),
        "hca_cmp_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "hca_state_slot_mapping": (torch.int64, (ranks, group_tokens)),
        "hca_kv_seq_lens": (torch.int32, (ranks, local_batch)),
        "input_ids": (torch.int64, (ranks, local_tokens)),
        "num_tokens_per_owner": (torch.int32, (ranks,)),
        "logit_row_indices": (torch.int32, (ranks, DSPARK_MAX_LOGIT_ROWS)),
        "sampled_ids": (
            torch.int32, (ranks, DSPARK_MAX_LOGIT_ROWS, DSPARK_SAMPLED_IDS_PAD),
        ),
    }
    return slot_specs


# Pool names the decode ABI spells differently from prefill's.
_DECODE_CACHE_POOL_NAMES = ("raw_kv_pool", "csa_idx_kv_cache", "csa_idx_kv_scale")

def decode_task_args(runner: DSparkModelRunner) -> TaskArgs:
    """Build the ``TaskArgs`` for the packed ``l3_decode_fwd`` dispatch."""
    layout = runner._compiled.layout
    slot_specs = _decode_slots(layout)
    static_weights = set(_PREFILL_STATIC_WEIGHT_NAMES)
    ranks = layout.ranks
    scratch = {
        "hidden_workspace": (
            (ranks, DSPARK_DECODE_LOCAL_TOKENS, DSPARK_HIDDEN_SIZE), torch.bfloat16,
        ),
        "x_ping": (
            (ranks, DSPARK_DECODE_LOCAL_TOKENS, DSPARK_HC_MULT, DSPARK_HIDDEN_SIZE),
            torch.float32,
        ),
        "x_pong": (
            (ranks, DSPARK_DECODE_LOCAL_TOKENS, DSPARK_HC_MULT, DSPARK_HIDDEN_SIZE),
            torch.float32,
        ),
        "x_attn_active": (
            (ranks, DSPARK_DECODE_LOCAL_TOKENS, DSPARK_HC_MULT, DSPARK_HIDDEN_SIZE),
            torch.float32,
        ),
        "x_moe_next": (
            (ranks, DSPARK_MOE_TOKENS, DSPARK_HC_MULT, DSPARK_HIDDEN_SIZE),
            torch.float32,
        ),
        "pre_hc_hidden_out": (
            (ranks, DSPARK_DECODE_LOCAL_TOKENS, DSPARK_HC_MULT, DSPARK_HIDDEN_SIZE),
            torch.float32,
        ),
        "x_out": (
            (ranks, DSPARK_DECODE_LOCAL_TOKENS, DSPARK_HIDDEN_SIZE), torch.bfloat16,
        ),
        "logits": (
            (ranks, DSPARK_MAX_LOGIT_ROWS, DSPARK_VOCAB_SIZE), torch.float32,
        ),
    }

    ta = TaskArgs(stacked=True)
    for name in _DECODE_TENSOR_ORDER:
        if name in slot_specs:
            dtype, shape = slot_specs[name]
            ta.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, s=shape: s))
        elif name == "embed_weight":
            ta.add_arg(name, lambda: runner._materialize_embedding_device_weight())
        elif name in static_weights:
            ta.add_arg(name, lambda n=name: runner._static_weight(n))
        elif name in _CACHE_POOL_NAMES:
            # Prefill's pool names the decode ABI shares verbatim.
            ta.add_arg(name, lambda n=name: runner._device_cache_values()[n])
        elif name in _DECODE_CACHE_POOL_NAMES:
            # The decode ABI renames the raw pool and the indexer caches.
            ta.add_arg(name, lambda n=name: runner._device_cache_values()[n])
        elif name in scratch:
            ta.add_arg(
                name,
                lambda n=name, s=scratch[name]: runner._alloc_zeroed_stacked_tensor(
                    n, s[0], s[1], scope="decode"
                ),
            )
        else:
            ta.add_arg(name, lambda n=name: runner._require_stacked_weights()[n])
    return ta


# Argument order for the packed all-43-layer ``l3_prefill_fwd`` kernel
# (pypto-lib prefill_fwd.py): layer-stacked weights and cache pools in core
# parameter order, the four rope profiles, the paged full-context block
# tables and slot mappings, the o-projection regather scratch, the layer-major
# staging buffers, the output collapse and device LM head, and the
# greedy-sampled ids.
_PREFILL_TENSOR_ORDER = (
    "x_hc",
    "query_start_loc",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hca_cmp_kv", "csa_cmp_kv",
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state", "idx_kv_cache", "idx_kv_scale",
    "hca_compress_state_block_table", "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "swa_freqs_cos", "swa_freqs_sin",
    "compressed_freqs_cos", "compressed_freqs_sin",
    "hca_cmp_freqs_cos", "hca_cmp_freqs_sin",
    "csa_cmp_freqs_cos", "csa_cmp_freqs_sin",
    "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
    "ori_slot_mapping_full", "position_ids_local", "position_ids_full", "input_ids",
    "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
    "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full",
    "csa_state_slot_mapping_full", "csa_inner_state_slot_mapping_full",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
    "o_proj_wo_a_full", "o_proj_wo_b_full",
    "attn_stage", "x_mixed", "post_ffn", "comb_ffn", "ffn_out",
    "hc_head_fn", "hc_head_scale", "hc_head_base",
    "final_norm_w", "lm_head_weight", "logit_row_indices",
    "hidden_workspace", "x_out", "logits", "sampled_ids",
)

# Argument order for the packed ``l3_decode_fwd`` kernel (pypto-lib
# decode_fwd.py): the weight bank and replicated pools, the split
# local/group-row RoPE and position tables, the three attention families'
# pre-lowered metadata, the router and experts, the recurrent HC staging
# buffers, and the greedy-sampled ids.
_DECODE_TENSOR_ORDER = (
    "embed_weight",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "raw_kv_pool",
    "freqs_cos_local", "freqs_sin_local", "freqs_cos", "freqs_sin",
    "compressed_freqs_cos_local", "compressed_freqs_sin_local",
    "compressed_freqs_cos", "compressed_freqs_sin",
    "swa_slot_mapping", "swa_indices", "swa_lens",
    "position_ids_local", "position_ids",
    "csa_cmp_freqs_cos", "csa_cmp_freqs_sin",
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "csa_compress_state", "csa_compress_state_block_table",
    "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj", "csa_hadamard_idx",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state", "csa_inner_compress_state_block_table",
    "csa_cmp_kv", "csa_cmp_block_table",
    "csa_idx_kv_cache", "csa_idx_kv_scale", "csa_idx_block_table",
    "csa_ori_slot_mapping", "csa_window_swa_indices", "csa_window_swa_lens",
    "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
    "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
    "csa_kv_seq_lens",
    "hca_cmp_freqs_cos", "hca_cmp_freqs_sin",
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hca_compress_state", "hca_compress_state_block_table",
    "hca_cmp_kv", "hca_cmp_block_table",
    "hca_ori_slot_mapping", "hca_window_swa_indices", "hca_window_swa_lens",
    "hca_cmp_slot_mapping", "hca_state_slot_mapping",
    "hca_kv_seq_lens",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids", "num_tokens_per_owner",
    "hc_head_fn", "hc_head_scale", "hc_head_base",
    "final_norm_w", "lm_head_weight", "logit_row_indices",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
    "hidden_workspace", "x_ping", "x_pong",
    "x_attn_active", "x_moe_next",
    "pre_hc_hidden_out", "x_out", "logits", "sampled_ids",
)
