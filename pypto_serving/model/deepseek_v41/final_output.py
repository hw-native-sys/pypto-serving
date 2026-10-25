# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Diagnostic final-output composition for lib revisions without l3_final_output."""

import importlib
import sys

import pypto.language as pl
import pypto.language.distributed as pld

from models.deepseek_v4_1_flash.config import FLASH, EP_SIZE, LOCAL_T_DYN, TP_SIZE
from models.deepseek_v4_1_flash.hc_head import hc_head
from models.deepseek_v4_1_flash.rmsnorm import rms_norm_tiled


def make_final_output_program(tp_size: int, dp_size: int):
    if (tp_size, dp_size, tp_size * dp_size) != (TP_SIZE, EP_SIZE // TP_SIZE, EP_SIZE):
        raise ValueError("final output topology differs from the backbone")

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--tp", str(tp_size), "--dp", str(dp_size)]
        lm = importlib.import_module("models.deepseek_v4_1_flash.lm_head")
    finally:
        sys.argv = old_argv
    if (lm.TP_SIZE, lm.DP_SIZE, lm.WORLD_SIZE) != (tp_size, dp_size, EP_SIZE):
        raise ValueError("LM head was imported for a different TP/DP topology")
    lm_head_with_sampling_test = lm.lm_head_with_sampling_test

    rows = lm.MAX_LOGIT_ROWS
    vocab = lm.VOCAB
    vocab_per_tp = lm.VOCAB_PER_TP
    sampled_pad = lm.SAMPLED_IDS_PAD
    group_logit_rows = lm.GROUP_LOGIT_ROWS
    hidden_size = FLASH.hidden_size
    hc_mult = FLASH.hc_mult

    @pl.jit
    def final_norm_rank(
        x_hc: pl.Tensor[[LOCAL_T_DYN, hc_mult, hidden_size], pl.FP32],
        pre_mix: pl.Tensor[[LOCAL_T_DYN, hc_mult], pl.FP32],
        norm_weight: pl.Tensor[[hidden_size], pl.BF16],
        normed: pl.Tensor[[LOCAL_T_DYN, hidden_size], pl.BF16],
    ):
        hidden = pl.create_tensor([pl.tensor.dim(x_hc, 0), hidden_size], dtype=pl.BF16)
        hc_head(x_hc, pre_mix, hidden)
        norm_tokens = pl.tensor.dim(hidden, 0)
        for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
            rms_norm_tiled(hidden, norm_weight, normed, norm_block, norm_tokens)

    @pl.jit.host
    def l3_final_output(
        x_hc: pl.Tensor[[EP_SIZE, LOCAL_T_DYN, hc_mult, hidden_size], pl.FP32],
        pre_mix: pl.Tensor[[EP_SIZE, LOCAL_T_DYN, hc_mult], pl.FP32],
        norm_weight: pl.Tensor[[EP_SIZE, hidden_size], pl.BF16],
        head_weight: pl.Tensor[[EP_SIZE, vocab_per_tp, hidden_size], pl.BF16],
        logit_row_indices: pl.Tensor[[EP_SIZE, rows], pl.INT32],
        normed: pl.Out[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, hidden_size], pl.BF16]],
        logits: pl.Out[pl.Tensor[[EP_SIZE, rows, vocab], pl.FP32]],
        sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, rows, sampled_pad], pl.INT32]],
        done_epoch: pl.Scalar[pl.INT32],
    ):
        hidden_window_buf = pld.alloc_window_buffer(group_logit_rows * hidden_size * 2)
        logits_window_buf = pld.alloc_window_buffer(rows * vocab * 4)
        hidden_done_buf = pld.alloc_window_buffer(tp_size * 4)
        logits_done_buf = pld.alloc_window_buffer(tp_size * 4)
        for rank in pl.range(pld.world_size()):
            final_norm_rank(x_hc[rank], pre_mix[rank], norm_weight[rank], normed[rank], device=rank)
        for rank in pl.range(pld.world_size()):
            hidden_window = pld.window(hidden_window_buf, [group_logit_rows, hidden_size], dtype=pl.BF16)
            hidden_done = pld.window(hidden_done_buf, [tp_size, 1], dtype=pl.INT32)
            logits_window = pld.window(logits_window_buf, [rows, vocab], dtype=pl.FP32)
            logits_done = pld.window(logits_done_buf, [tp_size, 1], dtype=pl.INT32)
            lm_head_with_sampling_test(
                normed[rank], head_weight[rank], logit_row_indices[rank], logits[rank], sampled_ids[rank],
                hidden_window, hidden_done, logits_window, logits_done,
                rank // tp_size * tp_size, rank % tp_size, done_epoch, device=rank,
            )

    return l3_final_output
