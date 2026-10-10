# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B per-dispatch-class :class:`TaskArgs` builders.

Each builder registers, **in kernel-positional order**, every argument one L3
callable takes -- host-shared I/O slots (allocated/staged here), static weights
(uploaded once via ``StaticDeviceTensor``), and the worker-resident KV-cache
handles (lazy sources resolved at ``build()`` time). There is no separate
kernel-order tuple: the order is the ``add_slot`` / ``add_arg`` registration
sequence, and each arg declares its kind at registration. This file is the
single source of truth for each kernel's positional contract.

Qwen is single-rank, so every ``TaskArgs`` uses ``stacked=False``.

Two per-call wrinkles are handled by the runner *after* ``build()`` rather than
in the builder:

* ``qwen3_prefill_host`` takes ``input_ids`` / ``slot_mapping`` as
  length-``total_tokens`` *slices* over the full kernel-batch slots. The slots
  own the full buffers; the runner splices the per-call slices into the built
  tuple.
* ``qwen3_decode_host`` receives runtime-batch prefix views over its max-capacity
  slots. It writes ``logits`` / ``next_hidden`` to device-resident scratch under
  device sampling instead of the host slots; the runner splices and narrows the
  device scratch when device sampling is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pypto_serving.model.common.runner.buffer_set import Placement, Slot
from pypto_serving.model.common.runner.task_args import TaskArgs

if TYPE_CHECKING:
    from pypto_serving.model.qwen.npu_runner import Qwen314BModelRunner

__all__ = [
    "decode_task_args",
    "prefill_task_args",
    "topk_select_task_args",
]


# Static-weight arg names -- each resolves to an already-built
# ``StaticDeviceTensor`` marker on ``runner._require_static_args()`` (uploaded
# once and cached by the resolver).
_STATIC_WEIGHT_ACCESSORS = {
    "rope_cos": lambda s: s.rope_cos,
    "rope_sin": lambda s: s.rope_sin,
    "final_norm_weight": lambda s: s.final_norm_weight,
    "padded_lm_head_weight": lambda s: s.padded_lm_head_weight,
    "padded_embed_weight": lambda s: s.padded_embed_weight,
}


def _static_weight(runner: Qwen314BModelRunner, name: str):
    """Eager ``StaticDeviceTensor`` marker for one static weight.

    The markers are built once in ``__init__`` and are stable, so they are
    captured at builder time (no lazy indirection needed).
    """
    return _STATIC_WEIGHT_ACCESSORS[name](runner._require_static_args())


def _decode_weight(runner: Qwen314BModelRunner, name: str):
    """Eager ``StaticDeviceTensor`` marker for one stacked decode weight."""
    return runner._require_static_args().decode_weights[name]


def _kv_key_pages(runner: Qwen314BModelRunner):
    """Lazy source: the active paged KV-cache key pages (resolved at build)."""
    return runner._active_kv_cache.key_pages


def _kv_value_pages(runner: Qwen314BModelRunner):
    """Lazy source: the active paged KV-cache value pages (resolved at build)."""
    return runner._active_kv_cache.value_pages


def _prefill_slot_specs(layout) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Host-shared slot name -> (dtype, full shape) for the prefill dispatch.

    ``input_ids`` and ``slot_mapping`` own the full kernel-batch buffers; the
    runner dispatches per-call slices over them.
    """
    batch = layout.kernel_batch
    seq = layout.max_seq_len
    return {
        "input_ids": (torch.int32, (batch * seq,)),
        "seq_lens": (torch.int32, (batch,)),
        "chunk_lens": (torch.int32, (batch,)),
        "chunk_offsets": (torch.int32, (batch,)),
        "block_table": (torch.int32, (batch * layout.max_blocks_per_seq,)),
        "slot_mapping": (torch.int32, (batch * seq,)),
        "logits": (torch.float32, (batch, layout.padded_vocab)),
    }


# ``qwen3_prefill_host`` positional order (the contract this builder encodes).
_PREFILL_ORDER = (
    "input_ids", "seq_lens", "chunk_lens", "chunk_offsets",
    "decode_input_rms_weight", "decode_wq", "decode_wk", "decode_wv",
    "decode_q_norm_weight", "decode_k_norm_weight",
    "rope_cos", "rope_sin",
    "block_table", "slot_mapping",
    "k_cache", "v_cache",
    "decode_wo", "decode_w_gate", "decode_w_up", "decode_w_down", "decode_post_rms_weight",
    "final_norm_weight", "padded_lm_head_weight", "padded_embed_weight",
    "logits",
)


def prefill_task_args(runner: Qwen314BModelRunner) -> TaskArgs:
    """Build the ``TaskArgs`` for the single ``qwen3_prefill_host`` dispatch.

    Host-shared I/O buffers are slots (allocated here); static weights are eager
    ``StaticDeviceTensor`` markers; the KV-cache handles are lazy sources
    resolved at ``build()``. The caller splices per-call ``input_ids`` /
    ``slot_mapping`` slices into the built tuple.
    """
    slot_specs = _prefill_slot_specs(runner._compiled.layout)
    static_weights = set(_STATIC_WEIGHT_ACCESSORS)

    ta = TaskArgs(stacked=False)
    for name in _PREFILL_ORDER:
        if name in slot_specs:
            dtype, shape = slot_specs[name]
            ta.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, s=shape: s))
        elif name == "k_cache":
            ta.add_arg(name, lambda: _kv_key_pages(runner))
        elif name == "v_cache":
            ta.add_arg(name, lambda: _kv_value_pages(runner))
        elif name in static_weights:
            ta.add_arg(name, _static_weight(runner, name))
        else:  # stacked decode weight
            ta.add_arg(name, _decode_weight(runner, name))
    return ta


def _decode_slot_specs(layout) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Host-shared slot name -> (dtype, full shape) for the decode dispatch.

    ``logits`` and ``next_hidden`` are the host mirrors the runner swaps device
    scratch in for under device sampling.
    """
    batch = layout.kernel_batch
    return {
        "seq_lens": (torch.int32, (batch,)),
        "block_table": (torch.int32, (batch * layout.max_blocks_per_seq,)),
        "slot_mapping": (torch.int32, (batch,)),
        "logits": (torch.float32, (batch, layout.padded_vocab)),
        "token_ids": (torch.int32, (batch, layout.sampled_ids_width)),
        "sampled_ids": (torch.int32, (batch, layout.sampled_ids_width)),
        "next_hidden": (torch.bfloat16, (batch, layout.hidden_size)),
    }


# ``qwen3_decode_host`` positional order (the contract this builder encodes).
_DECODE_ORDER = (
    "decode_input_rms_weight", "decode_wq", "decode_wk", "decode_wv",
    "decode_q_norm_weight", "decode_k_norm_weight",
    "seq_lens", "block_table", "slot_mapping",
    "rope_cos", "rope_sin",
    "k_cache", "v_cache",
    "decode_wo", "decode_w_gate", "decode_w_up", "decode_w_down", "decode_post_rms_weight",
    "final_norm_weight", "padded_lm_head_weight",
    "logits",
    "padded_embed_weight",
    "token_ids", "sampled_ids", "next_hidden",
)


def decode_task_args(runner: Qwen314BModelRunner) -> TaskArgs:
    """Build the ``TaskArgs`` for the single ``qwen3_decode_host`` dispatch.

    Host-shared I/O buffers are max-capacity slots; static weights are eager
    markers; the KV-cache handles are lazy sources. The caller narrows all
    batch-shaped slots to the runtime batch and splices device-resident
    ``logits`` / ``next_hidden`` scratch when device sampling is active.
    """
    slot_specs = _decode_slot_specs(runner._compiled.layout)
    static_weights = set(_STATIC_WEIGHT_ACCESSORS)

    ta = TaskArgs(stacked=False)
    for name in _DECODE_ORDER:
        if name in slot_specs:
            dtype, shape = slot_specs[name]
            ta.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, s=shape: s))
        elif name == "k_cache":
            ta.add_arg(name, lambda: _kv_key_pages(runner))
        elif name == "v_cache":
            ta.add_arg(name, lambda: _kv_value_pages(runner))
        elif name in static_weights:
            ta.add_arg(name, _static_weight(runner, name))
        else:  # stacked decode weight
            ta.add_arg(name, _decode_weight(runner, name))
    return ta


def _topk_slot_specs(layout) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Host-shared slot name -> (dtype, full shape) for the topk-select buffers.

    The prefill-side and decode-side topk outputs are distinct buffers; the
    ``sampling_control`` buffer is shared. ``topk_select_task_args`` owns all
    five for allocation; the runner assembles the 4-arg dispatch tuple inline
    (the ``logits`` input is external, owned by the prefill/decode TaskArgs).
    """
    batch = layout.kernel_batch
    width = layout.topk_width
    return {
        "sampling_control": (torch.int32, (2,)),
        "prefill_topk_values": (torch.float32, (batch, width)),
        "prefill_topk_indices": (torch.int32, (batch, width)),
        "decode_topk_values": (torch.float32, (batch, width)),
        "decode_topk_indices": (torch.int32, (batch, width)),
    }


def topk_select_task_args(runner: Qwen314BModelRunner) -> TaskArgs:
    """Build the ``TaskArgs`` that owns the shared topk-select buffers.

    The ``topk_select`` callable takes ``(logits, control, values, indices)``
    where ``logits`` is an external input and ``values`` / ``indices`` differ
    between the prefill-side and decode-side calls, so the runner reads the
    slot tensors directly (``.tensors[...]``) and assembles the dispatch tuple
    inline rather than calling ``build()``.
    """
    slot_specs = _topk_slot_specs(runner._compiled.layout)

    ta = TaskArgs(stacked=False)
    for name, (dtype, shape) in slot_specs.items():
        ta.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, s=shape: s))
    return ta
