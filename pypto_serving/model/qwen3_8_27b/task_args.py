# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Positional Qwen3.8 kernel arguments and host-shared dispatch slots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from pypto_serving.model.common.runner.buffer_set import Placement, Slot
from pypto_serving.model.common.runner.task_args import TaskArgs
from pypto_serving.model.qwen3_8_27b.npu_runner import Qwen38Layout, QWEN38_VOCAB_SIZE


# Order is the annotated argument order in pypto-lib, not a presentation list.
PREFILL_ARGS = (
    "input_ids", "embed_weight", "residual", "norm_in_w", "norm_post_w",
    "w_gate", "w_gate_scale", "w_up", "w_up_scale", "w_down", "w_down_scale",
    "w_qkv", "w_qkv_scale", "w_z", "w_z_scale", "w_ab", "conv_w", "a_log",
    "dt_bias", "norm_w", "w_out", "w_q", "w_q_scale", "w_k", "w_k_scale",
    "w_v", "w_v_scale", "w_o", "w_o_scale", "q_norm_w", "k_norm_w",
    "tril", "mask_strict", "eye", "m_diag", "m_low", "rope_cos", "rope_sin",
    "mask_bias", "row_base", "row_owner", "row_ends", "state_rows",
    "page_bases", "visible_pages", "block_table", "conv_state",
    "final_state", "k_pages", "v_pages", "final_norm_w", "lm_w", "logit_ids",
    "residual_out", "mlp_out", "logits", "valid_rows",
)

DECODE_ARGS = (
    'residual', 'delta', 'norm_in_w', 'norm_post_w',
    'w_gate', 'w_gate_scale', 'w_up', 'w_up_scale', 'w_down', 'w_down_scale',
    'w_qkv', 'w_qkv_scale', 'w_z', 'w_z_scale', 'w_ab', 'conv_w', 'a_log', 'dt_bias',
    'norm_w', 'w_out', 'decode_w_qkv', 'decode_w_qkv_scale', 'w_o', 'w_o_scale',
    'q_norm_w', 'k_norm_w', 'rope_cos', 'rope_sin', 'state_rows', 'block_tables',
    'seq_lens', 'tail_mask', 'conv_state', 'final_state', 'k_pages', 'v_pages',
    'residual_out', 'mlp_out', 'group_start',
)

EMBEDDING_ARGS = ("input_ids", "embed_weight", "out")
FINAL_HEAD_ARGS = ("residual", "delta", "norm_w", "lm_w", "logits", "valid_rows")

ArgSource = Callable[[str], Any]
SlotSpecs = Mapping[str, tuple[torch.dtype, tuple[int, ...]]]


def _task_args(order: tuple[str, ...], slots: SlotSpecs, source: ArgSource) -> TaskArgs:
    """Register each slot or lazy resident source at its kernel position."""
    args = TaskArgs(stacked=False)
    for name in order:
        if name in slots:
            dtype, shape = slots[name]
            args.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, s=shape: s))
        else:
            args.add_arg(name, lambda n=name: source(n))
    return args


def prefill_task_args(layout: Qwen38Layout, source: ArgSource, rows: int = 1) -> TaskArgs:
    """Packed text prefill, with device-resident states and K/V pools."""
    tokens = layout.prefill_tokens
    logit_rows = ((rows + 15) // 16) * 16
    slots = {
        "input_ids": (torch.int32, (tokens,)),
        "rope_cos": (torch.float32, (tokens, 64)),
        "rope_sin": (torch.float32, (tokens, 64)),
        "row_base": (torch.int32, (tokens,)),
        "row_owner": (torch.int32, (tokens,)),
        "row_ends": (torch.int32, (rows,)),
        "state_rows": (torch.int32, (48 * rows,)),
        "page_bases": (torch.int32, (rows,)),
        "visible_pages": (torch.int32, (rows,)),
        "block_table": (torch.int32, (rows, layout.max_blocks_per_seq)),
        "logit_ids": (torch.int32, (logit_rows,)),
        "logits": (torch.float32, (logit_rows, QWEN38_VOCAB_SIZE)),
    }
    return _task_args(PREFILL_ARGS, slots, source)


def decode_task_args(layout: Qwen38Layout, source: ArgSource) -> TaskArgs:
    """One decoder stack with separate metadata for every layer and padded row."""
    batch = layout.decode_batch_pad
    slots = {
        "rope_cos": (torch.float32, (batch, 64)),
        "rope_sin": (torch.float32, (batch, 64)),
        "state_rows": (torch.int32, (48 * batch,)),
        "block_tables": (torch.int32, (16 * batch, layout.max_blocks_per_seq)),
        "seq_lens": (torch.int32, (batch,)),
    }
    return _task_args(DECODE_ARGS, slots, source)


def embedding_task_args(layout: Qwen38Layout, source: ArgSource) -> TaskArgs:
    return _task_args(EMBEDDING_ARGS,
                      {"input_ids": (torch.int32, (layout.decode_batch_pad,))}, source)


def final_head_task_args(layout: Qwen38Layout, source: ArgSource) -> TaskArgs:
    return _task_args(FINAL_HEAD_ARGS,
                      {"logits": (torch.float32, (layout.decode_batch_pad, QWEN38_VOCAB_SIZE))},
                      source)
