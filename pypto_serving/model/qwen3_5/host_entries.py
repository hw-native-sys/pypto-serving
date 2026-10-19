# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 entrypoints around the accepted single-chip Qwen3.8 text programs."""

from __future__ import annotations

from typing import Any

import pypto.language as pl

from pypto_serving.model.qwen3_5.weight_spec import (
    ATTN_LAYERS, DELTA_LAYERS, HIDDEN, KV_WIDTH, MAX_SEQUENCE, MLP_WIDTH,
    NUM_LAYERS, QKV, Q_WIDTH, VALUE_WIDTH, VOCAB,
)

ATTN_HEAD_DIM = 256
DELTA_HEAD_DIM = 128
VALUE_HEADS = 48
TOKENS_PAD = 16
ROTARY_DIM = 64

_prefill_fwd: Any = None
_decode_fwd: Any = None


def bind_entries(*, prefill_fwd: Any, decode_fwd: Any) -> None:
    """Bind the exact pinned pypto-lib entries before compiling the HOST wrappers."""
    global _prefill_fwd, _decode_fwd
    _prefill_fwd = prefill_fwd
    _decode_fwd = decode_fwd


@pl.jit.host
def prefill_host(
    token_ids: pl.Tensor[[MAX_SEQUENCE], pl.INT64],
    token_count: pl.Scalar[pl.INDEX],
    embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    input_norms: pl.Tensor[[NUM_LAYERS, 1, HIDDEN], pl.BF16],
    post_norms: pl.Tensor[[NUM_LAYERS, 1, HIDDEN], pl.BF16],
    mlp_gates: pl.Tensor[[NUM_LAYERS, MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_ups: pl.Tensor[[NUM_LAYERS, MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_downs: pl.Tensor[[NUM_LAYERS, HIDDEN, MLP_WIDTH], pl.BF16],
    delta_qkvs: pl.Tensor[[DELTA_LAYERS, QKV, HIDDEN], pl.BF16],
    delta_convs: pl.Tensor[[DELTA_LAYERS, 4, QKV], pl.BF16],
    delta_as: pl.Tensor[[DELTA_LAYERS, DELTA_HEAD_DIM, HIDDEN], pl.BF16],
    delta_bs: pl.Tensor[[DELTA_LAYERS, DELTA_HEAD_DIM, HIDDEN], pl.BF16],
    delta_zs: pl.Tensor[[DELTA_LAYERS, VALUE_WIDTH, HIDDEN], pl.BF16],
    delta_a_logs: pl.Tensor[[DELTA_LAYERS, 1, DELTA_HEAD_DIM], pl.BF16],
    delta_dt_biases: pl.Tensor[[DELTA_LAYERS, 1, DELTA_HEAD_DIM], pl.BF16],
    delta_norms: pl.Tensor[[DELTA_LAYERS, 1, DELTA_HEAD_DIM], pl.BF16],
    delta_outs: pl.Tensor[[DELTA_LAYERS, HIDDEN, VALUE_WIDTH], pl.BF16],
    attn_qs: pl.Tensor[[ATTN_LAYERS, 2 * Q_WIDTH, HIDDEN], pl.BF16],
    attn_ks: pl.Tensor[[ATTN_LAYERS, KV_WIDTH, HIDDEN], pl.BF16],
    attn_vs: pl.Tensor[[ATTN_LAYERS, KV_WIDTH, HIDDEN], pl.BF16],
    attn_q_norms: pl.Tensor[[ATTN_LAYERS, 1, ATTN_HEAD_DIM], pl.BF16],
    attn_k_norms: pl.Tensor[[ATTN_LAYERS, 1, ATTN_HEAD_DIM], pl.BF16],
    attn_outs: pl.Tensor[[ATTN_LAYERS, HIDDEN, Q_WIDTH], pl.BF16],
    final_norm: pl.Tensor[[1, HIDDEN], pl.BF16],
    lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    cos_table: pl.Tensor[[MAX_SEQUENCE, ROTARY_DIM], pl.BF16],
    sin_table: pl.Tensor[[MAX_SEQUENCE, ROTARY_DIM], pl.BF16],
    conv_states: pl.InOut[pl.Tensor[[DELTA_LAYERS, TOKENS_PAD, QKV], pl.BF16]],
    recurrent_states: pl.InOut[pl.Tensor[[DELTA_LAYERS, VALUE_HEADS, DELTA_HEAD_DIM, DELTA_HEAD_DIM], pl.FP32]],
    key_states: pl.InOut[pl.Tensor[[ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH], pl.BF16]],
    value_states: pl.InOut[pl.Tensor[[ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH], pl.BF16]],
    logits: pl.Out[pl.Tensor[[1, VOCAB], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
    return _prefill_fwd(
        token_ids, token_count, embedding, input_norms, post_norms, mlp_gates, mlp_ups,
        mlp_downs, delta_qkvs, delta_convs, delta_as, delta_bs, delta_zs,
        delta_a_logs, delta_dt_biases, delta_norms, delta_outs, attn_qs,
        attn_ks, attn_vs, attn_q_norms, attn_k_norms, attn_outs, final_norm,
        lm_head, cos_table, sin_table, conv_states, recurrent_states,
        key_states, value_states, logits,
    )


@pl.jit.host
def decode_host(
    token_ids: pl.Tensor[[MAX_SEQUENCE], pl.INT64],
    position: pl.Scalar[pl.INDEX],
    embedding: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    input_norms: pl.Tensor[[NUM_LAYERS, 1, HIDDEN], pl.BF16],
    post_norms: pl.Tensor[[NUM_LAYERS, 1, HIDDEN], pl.BF16],
    mlp_gates: pl.Tensor[[NUM_LAYERS, MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_ups: pl.Tensor[[NUM_LAYERS, MLP_WIDTH, HIDDEN], pl.BF16],
    mlp_downs: pl.Tensor[[NUM_LAYERS, HIDDEN, MLP_WIDTH], pl.BF16],
    delta_qkvs: pl.Tensor[[DELTA_LAYERS, QKV, HIDDEN], pl.BF16],
    delta_convs: pl.Tensor[[DELTA_LAYERS, 4, QKV], pl.BF16],
    delta_as: pl.Tensor[[DELTA_LAYERS, DELTA_HEAD_DIM, HIDDEN], pl.BF16],
    delta_bs: pl.Tensor[[DELTA_LAYERS, DELTA_HEAD_DIM, HIDDEN], pl.BF16],
    delta_zs: pl.Tensor[[DELTA_LAYERS, VALUE_WIDTH, HIDDEN], pl.BF16],
    delta_a_logs: pl.Tensor[[DELTA_LAYERS, 1, DELTA_HEAD_DIM], pl.BF16],
    delta_dt_biases: pl.Tensor[[DELTA_LAYERS, 1, DELTA_HEAD_DIM], pl.BF16],
    delta_norms: pl.Tensor[[DELTA_LAYERS, 1, DELTA_HEAD_DIM], pl.BF16],
    delta_outs: pl.Tensor[[DELTA_LAYERS, HIDDEN, VALUE_WIDTH], pl.BF16],
    attn_qs: pl.Tensor[[ATTN_LAYERS, 2 * Q_WIDTH, HIDDEN], pl.BF16],
    attn_ks: pl.Tensor[[ATTN_LAYERS, KV_WIDTH, HIDDEN], pl.BF16],
    attn_vs: pl.Tensor[[ATTN_LAYERS, KV_WIDTH, HIDDEN], pl.BF16],
    attn_q_norms: pl.Tensor[[ATTN_LAYERS, 1, ATTN_HEAD_DIM], pl.BF16],
    attn_k_norms: pl.Tensor[[ATTN_LAYERS, 1, ATTN_HEAD_DIM], pl.BF16],
    attn_outs: pl.Tensor[[ATTN_LAYERS, HIDDEN, Q_WIDTH], pl.BF16],
    final_norm: pl.Tensor[[1, HIDDEN], pl.BF16],
    lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    cos_table: pl.Tensor[[MAX_SEQUENCE, ROTARY_DIM], pl.BF16],
    sin_table: pl.Tensor[[MAX_SEQUENCE, ROTARY_DIM], pl.BF16],
    conv_states: pl.InOut[pl.Tensor[[DELTA_LAYERS, TOKENS_PAD, QKV], pl.BF16]],
    recurrent_states: pl.InOut[pl.Tensor[[DELTA_LAYERS, VALUE_HEADS, DELTA_HEAD_DIM, DELTA_HEAD_DIM], pl.FP32]],
    key_states: pl.InOut[pl.Tensor[[ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH], pl.BF16]],
    value_states: pl.InOut[pl.Tensor[[ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH], pl.BF16]],
    logits: pl.Out[pl.Tensor[[1, VOCAB], pl.FP32]],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
    return _decode_fwd(
        token_ids, position, embedding, input_norms, post_norms, mlp_gates, mlp_ups,
        mlp_downs, delta_qkvs, delta_convs, delta_as, delta_bs, delta_zs,
        delta_a_logs, delta_dt_biases, delta_norms, delta_outs, attn_qs,
        attn_ks, attn_vs, attn_q_norms, attn_k_norms, attn_outs, final_norm,
        lm_head, cos_table, sin_table, conv_states, recurrent_states,
        key_states, value_states, logits,
    )
