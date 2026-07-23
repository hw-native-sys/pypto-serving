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


@pl.jit.host
def qwen3_prefill_host(
    input_ids: pl.Tensor,
    seq_lens: pl.Tensor,
    chunk_lens: pl.Tensor,
    chunk_offsets: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    final_norm_weight: pl.Tensor,
    lm_head_weight: pl.Tensor,
    embed_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
) -> pl.Tensor:
    return prefill_fwd(
        input_ids,
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
        embed_weight,
        out,
    )


@pl.jit.host
def qwen3_decode_host(
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    final_norm_weight: pl.Tensor,
    lm_head_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
    embed_weight: pl.Tensor,
    sampled_ids_in: pl.Tensor,
    sampled_ids: pl.Out[pl.Tensor],
    next_hidden: pl.Out[pl.Tensor],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    logits, sampled_ids, next_hidden = decode_fwd(
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
    return logits, sampled_ids, next_hidden


@pl.jit.host
def qwen3_greedy_sample_host(
    logits: pl.Tensor,
    sampled_ids: pl.Out[pl.Tensor],
) -> pl.Tensor:
    return greedy_sample_fwd(
        logits,
        sampled_ids,
    )


def create_qwen3_a8w8_dispatch(prefill_hidden_a8w8, decode_fwd):
    """Create A8W8 HOST wrappers with format-local kernel dependencies."""

    @pl.jit.host
    def qwen3_a8w8_prefill_host(
        hidden_states: pl.Tensor,
        seq_lens: pl.Tensor,
        chunk_lens: pl.Tensor,
        chunk_offsets: pl.Tensor,
        input_rms_weight: pl.Tensor,
        wq: pl.Tensor,
        wk: pl.Tensor,
        wv: pl.Tensor,
        wq_scale: pl.Tensor,
        wk_scale: pl.Tensor,
        wv_scale: pl.Tensor,
        q_norm_weight: pl.Tensor,
        k_norm_weight: pl.Tensor,
        rope_cos: pl.Tensor,
        rope_sin: pl.Tensor,
        block_table: pl.Tensor,
        slot_mapping: pl.Tensor,
        k_cache: pl.Tensor,
        v_cache: pl.Tensor,
        wo: pl.Tensor,
        wo_scale: pl.Tensor,
        post_rms_weight: pl.Tensor,
        w_gate: pl.Tensor,
        w_up: pl.Tensor,
        w_gate_scale: pl.Tensor,
        w_up_scale: pl.Tensor,
        w_down: pl.Tensor,
        final_norm_weight: pl.Tensor,
        lm_head_weight: pl.Tensor,
        out: pl.Out[pl.Tensor],
        hidden_out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        return prefill_hidden_a8w8(
            hidden_states,
            seq_lens,
            chunk_lens,
            chunk_offsets,
            input_rms_weight,
            wq,
            wk,
            wv,
            wq_scale,
            wk_scale,
            wv_scale,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            block_table,
            slot_mapping,
            k_cache,
            v_cache,
            wo,
            wo_scale,
            post_rms_weight,
            w_gate,
            w_up,
            w_gate_scale,
            w_up_scale,
            w_down,
            final_norm_weight,
            lm_head_weight,
            out,
            hidden_out,
        )

    @pl.jit.host
    def qwen3_a8w8_decode_host(
        hidden_states: pl.Tensor,
        input_rms_weight: pl.Tensor,
        wq: pl.Tensor,
        wk: pl.Tensor,
        wv: pl.Tensor,
        wq_scale: pl.Tensor,
        wk_scale: pl.Tensor,
        wv_scale: pl.Tensor,
        q_norm_weight: pl.Tensor,
        k_norm_weight: pl.Tensor,
        seq_lens: pl.Tensor,
        active_batch: pl.Tensor,
        block_table: pl.Tensor,
        slot_mapping: pl.Tensor,
        rope_cos: pl.Tensor,
        rope_sin: pl.Tensor,
        k_cache: pl.Tensor,
        v_cache: pl.Tensor,
        wo: pl.Tensor,
        wo_scale: pl.Tensor,
        w_gate: pl.Tensor,
        w_up: pl.Tensor,
        w_gate_scale: pl.Tensor,
        w_up_scale: pl.Tensor,
        w_down: pl.Tensor,
        post_rms_weight: pl.Tensor,
        final_norm_weight: pl.Tensor,
        lm_head_weight: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ) -> pl.Tensor:
        return decode_fwd(
            hidden_states,
            input_rms_weight,
            wq,
            wk,
            wv,
            wq_scale,
            wk_scale,
            wv_scale,
            q_norm_weight,
            k_norm_weight,
            seq_lens,
            active_batch,
            block_table,
            slot_mapping,
            rope_cos,
            rope_sin,
            k_cache,
            v_cache,
            wo,
            wo_scale,
            w_gate,
            w_up,
            w_gate_scale,
            w_up_scale,
            w_down,
            post_rms_weight,
            final_norm_weight,
            lm_head_weight,
            out,
        )

    return qwen3_a8w8_prefill_host, qwen3_a8w8_decode_host
