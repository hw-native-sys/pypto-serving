# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B per-dispatch-class :class:`TaskArgs` builders.

Each builder owns only the host-shared buffers used by one dispatch class.
The pypto-lib Contract owns the kernel's positional ABI; the runner passes
these buffers, static weights, and KV-cache handles through the corresponding
Contract ``runtime_args_builder``.

Qwen is single-rank, so every ``TaskArgs`` uses ``stacked=False``.

Prefill dispatches use per-call slices of the full ``input_ids`` and
``slot_mapping`` buffers. Decode dispatches may replace host ``logits`` and
``next_hidden`` buffers with device-resident scratch for device sampling.
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


def _task_args_for_slots(
    slot_specs: dict[str, tuple[torch.dtype, tuple[int, ...]]],
) -> TaskArgs:
    """Allocate a single-rank TaskArgs containing only host-shared slots."""
    task_args = TaskArgs(stacked=False)
    for name, (dtype, shape) in slot_specs.items():
        task_args.add_slot(Slot(name, Placement.HOST_SHARED, dtype, lambda _, value=shape: value))
    return task_args


def prefill_task_args(runner: Qwen314BModelRunner) -> TaskArgs:
    """Build the host-shared buffers used by the prefill Contract stage."""
    return _task_args_for_slots(_prefill_slot_specs(runner._compiled.layout))


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


def decode_task_args(runner: Qwen314BModelRunner) -> TaskArgs:
    """Build the host-shared buffers used by the decode Contract stage."""
    return _task_args_for_slots(_decode_slot_specs(runner._compiled.layout))


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
        "sampling_control": (torch.int32, (layout.sampling_control_fields,)),
        "prefill_topk_values": (torch.float32, (batch, width)),
        "prefill_topk_indices": (torch.int32, (batch, width)),
        "decode_topk_values": (torch.float32, (batch, width)),
        "decode_topk_indices": (torch.int32, (batch, width)),
    }


def topk_select_task_args(runner: Qwen314BModelRunner) -> TaskArgs:
    """Build the ``TaskArgs`` that owns the shared topk-select buffers.

    The ``topk_select`` callable takes ``(logits, control, values, indices)``
    where ``logits`` is an external input and ``values`` / ``indices`` differ
    between the prefill-side and decode-side calls. The runner passes the
    selected slot tensors to the Contract's ``runtime_args_builder``.
    """
    return _task_args_for_slots(_topk_slot_specs(runner._compiled.layout))
