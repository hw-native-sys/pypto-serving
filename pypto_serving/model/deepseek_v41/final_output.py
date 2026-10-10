# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final-state output composition using existing V4.1 lib operators."""

import ctypes
import importlib
import sys

from .composite import LayerState
from .swa_segment import load_segment_modules


def make_final_output_program(lib_root, topology):
    """Collapse the last delayed HC state, normalize, project and sample."""
    import pypto.language as pl
    import pypto.language.distributed as pld

    load_segment_modules(lib_root, topology)
    package = "models.deepseek_v4_1_flash"
    config = importlib.import_module(package + ".config")
    hc_head = importlib.import_module(package + ".hc_head").hc_head
    rms_norm = importlib.import_module(package + ".rmsnorm").rms_norm

    # The LM-head module parses DP separately from the backbone's EP axis.
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--tp", str(topology.tp), "--dp", str(topology.dp)]
        lm = importlib.import_module(package + ".lm_head")
    finally:
        sys.argv = old_argv
    if (lm.TP_SIZE, lm.DP_SIZE, lm.WORLD_SIZE) != (topology.tp, topology.dp, topology.world):
        raise ValueError("LM head was imported for a different TP/DP topology")

    world = topology.world
    local = config.LOCAL_T_DYN
    hc = config.HC_MULT
    d = config.D
    rows = lm.MAX_LOGIT_ROWS
    vocab = lm.VOCAB
    vocab_per_tp = lm.VOCAB_PER_TP
    sampled_pad = lm.SAMPLED_IDS_PAD
    tp = topology.tp
    group_logit_rows = lm.GROUP_LOGIT_ROWS
    head_entry = lm.lm_head_with_sampling_test

    @pl.jit
    def final_norm_rank(
        x_hc: pl.Tensor[[local, hc, d], pl.FP32],
        pre_mix: pl.Tensor[[local, hc], pl.FP32],
        norm_weight: pl.Tensor[[d], pl.BF16],
        normed: pl.Tensor[[local, d], pl.BF16],
    ):
        hidden = pl.create_tensor([pl.tensor.dim(x_hc, 0), d], dtype=pl.BF16)
        hc_head(x_hc, pre_mix, hidden)
        rms_norm(hidden, norm_weight, normed)

    @pl.jit.host
    def l3_final_output(
        x_hc: pl.Tensor[[world, local, hc, d], pl.FP32],
        pre_mix: pl.Tensor[[world, local, hc], pl.FP32],
        norm_weight: pl.Tensor[[world, d], pl.BF16],
        head_weight: pl.Tensor[[world, vocab_per_tp, d], pl.BF16],
        logit_row_indices: pl.Tensor[[world, rows], pl.INT32],
        normed: pl.Out[pl.Tensor[[world, local, d], pl.BF16]],
        logits: pl.Out[pl.Tensor[[world, rows, vocab], pl.FP32]],
        sampled_ids: pl.Out[pl.Tensor[[world, rows, sampled_pad], pl.INT32]],
        done_epoch: pl.Scalar[pl.INT32],
    ):
        hidden_window_buf = pld.alloc_window_buffer(group_logit_rows * d * 2)
        logits_window_buf = pld.alloc_window_buffer(rows * vocab * 4)
        hidden_done_buf = pld.alloc_window_buffer(tp * 4)
        logits_done_buf = pld.alloc_window_buffer(tp * 4)
        for rank in pl.range(pld.world_size()):
            final_norm_rank(x_hc[rank], pre_mix[rank], norm_weight[rank], normed[rank], device=rank)
        for rank in pl.range(pld.world_size()):
            hidden_window = pld.window(hidden_window_buf, [group_logit_rows, d], dtype=pl.BF16)
            hidden_done = pld.window(hidden_done_buf, [tp, 1], dtype=pl.INT32)
            logits_window = pld.window(logits_window_buf, [rows, vocab], dtype=pl.FP32)
            logits_done = pld.window(logits_done_buf, [tp, 1], dtype=pl.INT32)
            head_entry(
                normed[rank], head_weight[rank], logit_row_indices[rank], logits[rank], sampled_ids[rank],
                hidden_window, hidden_done, logits_window, logits_done,
                rank // tp * tp, rank % tp, done_epoch,
                device=rank,
            )

    return l3_final_output


def compile_final_output(compiler, lib_root, topology):
    import pypto.language as pl

    entry = make_final_output_program(lib_root, topology)
    return compiler.compile("v41_final_output", entry, done_epoch=pl.RUNTIME), tuple(entry.param_names)


class FinalOutput:
    """Dispatch final logits while retaining one completion epoch per worker."""

    def __init__(self, worker, program, param_names, run_config):
        self.worker = worker
        self.program = program
        self.param_names = tuple(param_names)
        self.run_config = run_config
        self.epoch = 0
        self.failed = False
        required = {"x_hc", "pre_mix", "norm_weight", "head_weight", "logit_row_indices",
                    "normed", "logits", "sampled_ids", "done_epoch"}
        if set(self.param_names) != required or len(self.param_names) != len(required):
            raise ValueError("final output ABI differs from the serving composition")

    def run(self, state: LayerState, arguments):
        if self.failed:
            raise RuntimeError("output dispatch failed; close the worker before retrying")
        if state.layout != "tp_local_token":
            raise ValueError("final output requires TP-local residual and pre_mix")
        reserved = {"x_hc", "pre_mix", "done_epoch"}
        if reserved & arguments.keys():
            raise ValueError("final state and epoch must come from this dispatch")
        missing = set(self.param_names) - reserved - arguments.keys()
        if missing:
            raise ValueError("missing output ABI arguments: " + ", ".join(sorted(missing)))
        if self.epoch >= (2**31 - 1) // 16:
            raise OverflowError("output completion epoch exhausted; recreate worker")
        next_epoch = self.epoch + 1
        bound = dict(arguments, x_hc=state.residual, pre_mix=state.pre_mix,
                     done_epoch=ctypes.c_int32(next_epoch))
        try:
            self.worker.run(self.program.compiled, *(bound[name] for name in self.param_names),
                            config=self.run_config)
        except BaseException:
            self.failed = True
            raise
        self.epoch = next_epoch
        return arguments["logits"], arguments["sampled_ids"]
