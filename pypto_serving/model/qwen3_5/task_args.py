# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared input/output slots and device-state shapes for Qwen3.8 dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pypto_serving.model.common.runner.buffer_set import shared_empty
from .weight_spec import ATTN_LAYERS, DELTA_LAYERS, KV_WIDTH, MAX_SEQUENCE, QKV, VOCAB

STATE_SPECS = (
    ((DELTA_LAYERS, 16, QKV), torch.bfloat16),
    ((DELTA_LAYERS, 48, 128, 128), torch.float32),
    ((ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH), torch.bfloat16),
    ((ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH), torch.bfloat16),
)


@dataclass
class Qwen35HostSlot:
    """One pre-fork shared slot retained until its pending handle completes."""

    token_ids: torch.Tensor
    logits: torch.Tensor
    busy: bool = False


def allocate_host_slots(count: int = 2) -> tuple[Qwen35HostSlot, ...]:
    if count < 1:
        raise ValueError("at least one Qwen3.8 dispatch slot is required")
    return tuple(
        Qwen35HostSlot(
            token_ids=shared_empty((MAX_SEQUENCE,), torch.int64, name="qwen35_token_ids"),
            logits=shared_empty((1, VOCAB), torch.float32, name="qwen35_logits"),
        )
        for _ in range(count)
    )


def allocate_zero_states() -> tuple[torch.Tensor, ...]:
    """Allocate shared zero templates before DistributedWorker forks."""
    return tuple(torch.zeros(shape, dtype=dtype).share_memory_() for shape, dtype in STATE_SPECS)
