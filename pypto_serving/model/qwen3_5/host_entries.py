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

_prefill_fwd: Any = None
_decode_fwd: Any = None


def bind_entries(*, prefill_fwd: Any, decode_fwd: Any) -> None:
    """Bind the exact pinned pypto-lib entries before compiling the HOST wrappers."""
    global _prefill_fwd, _decode_fwd
    _prefill_fwd = prefill_fwd
    _decode_fwd = decode_fwd


@pl.jit.host
def prefill_host(
    token_ids: pl.Tensor,
    token_count: pl.Scalar[pl.INDEX],
    embedding: pl.Tensor,
    input_norms: pl.Tensor,
    post_norms: pl.Tensor,
    mlp_gates: pl.Tensor,
    mlp_ups: pl.Tensor,
    mlp_downs: pl.Tensor,
    delta_qkvs: pl.Tensor,
    delta_convs: pl.Tensor,
    delta_as: pl.Tensor,
    delta_bs: pl.Tensor,
    delta_zs: pl.Tensor,
    delta_a_logs: pl.Tensor,
    delta_dt_biases: pl.Tensor,
    delta_norms: pl.Tensor,
    delta_outs: pl.Tensor,
    attn_qs: pl.Tensor,
    attn_ks: pl.Tensor,
    attn_vs: pl.Tensor,
    attn_q_norms: pl.Tensor,
    attn_k_norms: pl.Tensor,
    attn_outs: pl.Tensor,
    final_norm: pl.Tensor,
    lm_head: pl.Tensor,
    cos_table: pl.Tensor,
    sin_table: pl.Tensor,
    conv_states: pl.InOut[pl.Tensor],
    recurrent_states: pl.InOut[pl.Tensor],
    key_states: pl.InOut[pl.Tensor],
    value_states: pl.InOut[pl.Tensor],
    logits: pl.Out[pl.Tensor],
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
    token_ids: pl.Tensor,
    position: pl.Scalar[pl.INDEX],
    embedding: pl.Tensor,
    input_norms: pl.Tensor,
    post_norms: pl.Tensor,
    mlp_gates: pl.Tensor,
    mlp_ups: pl.Tensor,
    mlp_downs: pl.Tensor,
    delta_qkvs: pl.Tensor,
    delta_convs: pl.Tensor,
    delta_as: pl.Tensor,
    delta_bs: pl.Tensor,
    delta_zs: pl.Tensor,
    delta_a_logs: pl.Tensor,
    delta_dt_biases: pl.Tensor,
    delta_norms: pl.Tensor,
    delta_outs: pl.Tensor,
    attn_qs: pl.Tensor,
    attn_ks: pl.Tensor,
    attn_vs: pl.Tensor,
    attn_q_norms: pl.Tensor,
    attn_k_norms: pl.Tensor,
    attn_outs: pl.Tensor,
    final_norm: pl.Tensor,
    lm_head: pl.Tensor,
    cos_table: pl.Tensor,
    sin_table: pl.Tensor,
    conv_states: pl.InOut[pl.Tensor],
    recurrent_states: pl.InOut[pl.Tensor],
    key_states: pl.InOut[pl.Tensor],
    value_states: pl.InOut[pl.Tensor],
    logits: pl.Out[pl.Tensor],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
    return _decode_fwd(
        token_ids, position, embedding, input_norms, post_norms, mlp_gates, mlp_ups,
        mlp_downs, delta_qkvs, delta_convs, delta_as, delta_bs, delta_zs,
        delta_a_logs, delta_dt_biases, delta_norms, delta_outs, attn_qs,
        attn_ks, attn_vs, attn_q_norms, attn_k_norms, attn_outs, final_norm,
        lm_head, cos_table, sin_table, conv_states, recurrent_states,
        key_states, value_states, logits,
    )
