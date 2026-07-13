# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HOST-level wrappers for Qwen3-14B prefill/decode kernels."""

from __future__ import annotations

import pypto.language as pl


prefill_fwd = None
decode_fwd = None
greedy_sample_fwd = None

# Shape contract for the three HOST wrappers below.
#
# The wrappers are compiled straight from their signatures (``compile()`` with no
# sample tensors — pypto #2014), so every tensor parameter must carry a full
# ``pl.Tensor[[...], dtype]`` annotation. ``from __future__ import annotations``
# makes those annotations lazy strings, resolved from this module's globals only
# when ``.compile()`` runs — so the names below only need to be populated before
# the first compile. ``Qwen314BPyptoExecutor._compile_model`` injects the real
# values (model constants + the *same* ``pl.dynamic`` instances the kernels use,
# so host↔kernel dynamic dims unify) right after it wires up the kernels.
#
# Static extents (model constants; runtime-derived ROPE_SEQ / DEC_BLOCK_TABLE_FLAT)
# only need to match the shapes the runtime dispatches. Dims the kernels bind
# dynamic (paged KV rows, prefill token/batch/layer dims, and decode's
# lm_head-driven USER_BATCH_DYN) are re-marked dynamic from the dep graph even
# when the annotation names a plain int, so the compiled artifact is identical to
# the equivalent ``compile(sample_tensors)`` call.
HIDDEN = None
KV_HIDDEN = None
HEAD_DIM = None
INTERMEDIATE = None
VOCAB = None
NUM_LAYERS = None
BATCH = None
SAMPLED_IDS_PAD = None
ROPE_SEQ = None            # runtime max_seq_len (rope table rows)
DEC_BLOCK_TABLE_FLAT = None  # kernel_batch * max_blocks_per_seq (decode block table)

USER_BATCH_DYN = None
PREFILL_TOKENS_DYN = None
KV_CACHE_ROWS_DYN = None
BLOCK_TABLE_FLAT_DYN = None
LAYER_DYN = None
LAYER_HIDDEN_ROWS_DYN = None
LAYER_INTER_ROWS_DYN = None


@pl.jit.host
def qwen3_prefill_host(
    hidden_states: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[ROPE_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ, HEAD_DIM], pl.FP32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor:
    return prefill_fwd(
        hidden_states,
        seq_lens,
        chunk_lens,
        chunk_offsets,
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        block_table,
        slot_mapping,
        k_cache,
        v_cache,
        wo,
        post_rms_weight,
        w_gate,
        w_up,
        w_down,
        final_norm_weight,
        lm_head_weight,
        out,
    )


@pl.jit.host
def qwen3_decode_host(
    input_rms_weight: pl.Tensor[[NUM_LAYERS, HIDDEN], pl.FP32],
    wq: pl.Tensor[[NUM_LAYERS * HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[NUM_LAYERS * HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[NUM_LAYERS * HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[BATCH], pl.INT32],
    block_table: pl.Tensor[[DEC_BLOCK_TABLE_FLAT], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[NUM_LAYERS * HIDDEN, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[NUM_LAYERS * HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[NUM_LAYERS * HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[NUM_LAYERS * INTERMEDIATE, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[NUM_LAYERS, HIDDEN], pl.FP32],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH, VOCAB], pl.FP32]],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    sampled_ids_in: pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32]],
    next_hidden: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    # Bind the first return to the ``out`` param name so signature-mode
    # specialization can fill the return tuple's element type from the param
    # annotation (a renamed local like ``logits`` leaves it a bare pl.Tensor,
    # which the parser rejects inside a tuple[...] return).
    out, sampled_ids, next_hidden = decode_fwd(
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        w_gate,
        w_up,
        w_down,
        post_rms_weight,
        final_norm_weight,
        lm_head_weight,
        out,
        embed_weight,
        sampled_ids_in,
        sampled_ids,
        next_hidden,
    )
    return out, sampled_ids, next_hidden


@pl.jit.host
def qwen3_greedy_sample_host(
    logits: pl.Tensor[[BATCH, VOCAB], pl.FP32],
    sampled_ids: pl.Out[pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32]],
) -> pl.Tensor:
    return greedy_sample_fwd(
        logits,
        sampled_ids,
    )
