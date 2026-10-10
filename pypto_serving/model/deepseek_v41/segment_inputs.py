# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host input packing for the lib's TP-local-token SWA/MoE boundary.

V4 prepares host embedding rows before device dispatch. V4.1 input_pack.pack_x_hc
(lib fbe92bfc) establishes four FP32 copies and golden.identity_pre_mix selects
lane zero. These are fresh per-token inputs at the model entrance, never a
replacement for residual/pre_mix between layers. Device upload is caller-owned.
"""
from dataclasses import dataclass

import torch

from .request_state import ForwardStep
from .swa_segment import SegmentTopology


@dataclass(frozen=True)
class SegmentInputs:
    residual: torch.Tensor
    pre_mix: torch.Tensor
    # Source indices into the original packed request order; -1 means padding.
    row_indices: torch.Tensor
    # (world rank, local row) in the original request order, for output selection.
    request_last_rows: tuple[tuple[int, int], ...]
    group_counts: tuple[int, ...]


def prepare_segment_inputs(
    embeddings: torch.Tensor,
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    max_prepare_bytes: int = 256 << 20,
) -> SegmentInputs:
    """Pack fresh embedding inputs without depending on request batch order.

    Each DP group contains contiguous local-capacity TP slabs, including empty
    slabs. Padding is zero in both residual and pre_mix. row_indices supplies the
    same mapping for positions and other token metadata. This does not lower
    cache pages or initialize a production all-mode model adapter.
    """
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0:
        raise ValueError("max_prepare_bytes must be a positive integer")
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("segment input preparation requires a ForwardStep and SegmentTopology")
    if (not isinstance(embeddings, torch.Tensor) or embeddings.layout != torch.strided
            or embeddings.device.type != "cpu" or embeddings.dtype != torch.bfloat16
            or embeddings.ndim != 2 or embeddings.shape[1] <= 0):
        raise ValueError("embeddings must be packed CPU BF16 [active_tokens, hidden_size]")
    counts = [0] * topology.dp
    source_ranges, last_rows = [], []
    offset = 0
    request_ids = set()
    for request in step.requests:
        partition, count = request.partition, len(request.token_ids)
        if type(partition) is not int or not 0 <= partition < topology.dp or count <= 0:
            raise ValueError("requests require valid DP partitions and nonempty token slices")
        if request.request_id in request_ids:
            raise ValueError("duplicate request in forward step")
        request_ids.add(request.request_id)
        start = counts[partition]
        if start + count > topology.capacity:
            raise ValueError("DP token count exceeds segment capacity; split the step first")
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one fresh token per request")
        source_ranges.append((partition, start, offset, count))
        owner, row = divmod(start + count - 1, topology.local_capacity)
        last_rows.append((partition * topology.tp + owner, row))
        counts[partition] += count
        offset += count
    if step.phase not in ("prefill", "decode") or embeddings.shape[0] != offset:
        raise ValueError("embedding rows must match a prefill/decode forward step")
    hidden = embeddings.shape[1]
    # Include output storage and the largest temporary expanded FP32 row block.
    allocated_rows = topology.world * topology.local_capacity
    estimate = allocated_rows * (4 * hidden * 4 + 4 * 4 + 8) + offset * hidden * 4
    if estimate > max_prepare_bytes:
        raise ValueError(f"segment preparation requires estimated {estimate} bytes; budget={max_prepare_bytes}")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite")
    residual = torch.zeros(topology.dp, topology.capacity, 4, hidden, dtype=torch.float32)
    pre_mix = torch.zeros(topology.dp, topology.capacity, 4, dtype=torch.float32)
    row_indices = torch.full((topology.dp, topology.capacity), -1, dtype=torch.int64)
    for partition, start, source, count in source_ranges:
        residual[partition, start:start + count].copy_(embeddings[source:source + count, None])
        pre_mix[partition, start:start + count, 0] = 1
        row_indices[partition, start:start + count] = torch.arange(source, source + count)
    return SegmentInputs(
        residual.reshape(topology.world, topology.local_capacity, 4, hidden),
        pre_mix.reshape(topology.world, topology.local_capacity, 4),
        row_indices.reshape(topology.world, topology.local_capacity),
        tuple(last_rows), tuple(counts),
    )
