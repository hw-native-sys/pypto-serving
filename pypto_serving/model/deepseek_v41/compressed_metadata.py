# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-owned C1A/C2A page and compressor metadata for lib SP composites."""
from dataclasses import dataclass

import torch

from .request_state import ForwardStep
from .swa_segment import SegmentTopology


@dataclass(frozen=True)
class CompressedMetadata:
    position_ids: torch.Tensor
    request_ids: torch.Tensor
    query_start_loc: torch.Tensor
    compressed_lens: torch.Tensor
    compressed_slots: torch.Tensor
    compressed_rope_positions: torch.Tensor
    index_block_table: torch.Tensor
    state_block_table: torch.Tensor
    group_counts: tuple[int, ...]


def prepare_compressed_metadata(
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    ratio: int,
    compressed_group: str,
    cache_pages: int,
    max_requests: int,
    state_blocks: int,
    max_prepare_bytes: int = 16 << 20,
) -> CompressedMetadata:
    """Lower full-history pages with joint KV/index physical page numbering.

    The resource allocator must use the same page IDs for compressed KV and
    index-key pools. Each pool stores 128 compressed rows per page. This does
    not select candidates/Top-K or materialize compressed attention indices.
    Those remain device outputs from the appropriate lib producer composite.

    Request IDs are local packed batch rows; state slots are stable ledger IDs.
    Ratio-2 publishes only completed pairs, rotating them at the pair's first
    position. An odd-length chunk leaves its tail in the request's state block.
    The formulas match lib metadata.paged_slots/compressor_metadata at fbe92bfc.
    """
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("compressed metadata requires a ForwardStep and SegmentTopology")
    if type(ratio) is not int or ratio not in (1, 2):
        raise ValueError("compression ratio must be 1 or 2")
    if (type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128
            or type(max_requests) is not int or max_requests <= 0
            or type(state_blocks) is not int or not 0 < state_blocks <= 2**31 - 1
            or type(max_prepare_bytes) is not int or max_prepare_bytes <= 0):
        raise ValueError("cache, request, state and allocation capacities must be positive and fit the ABI")
    if not isinstance(compressed_group, str) or not compressed_group:
        raise ValueError("compressed_group must identify a jointly numbered KV/index pool")
    if step.phase not in ("prefill", "decode"):
        raise ValueError("unsupported forward phase")
    groups = [[] for _ in range(topology.dp)]
    counts, owners, slots, names = [0] * topology.dp, set(), set(), set()
    columns = 1
    for request in step.requests:
        group, count = request.partition, len(request.token_ids)
        if (type(group) is not int or not 0 <= group < topology.dp or count <= 0
                or type(request.start) is not int or not 0 <= request.start < request.end <= 2**31 - 1):
            raise ValueError("invalid compressed request partition or extent")
        if request.request_id in names:
            raise ValueError("duplicate request in forward step")
        names.add(request.request_id)
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one token per request")
        if counts[group] + count > topology.capacity or len(groups[group]) >= max_requests:
            raise ValueError("compressed requests exceed token or request capacity")
        if type(request.state_slot) is not int or not 0 <= request.state_slot < state_blocks:
            raise ValueError("request state slot is outside the compressor pool")
        key = (group, request.state_slot)
        if ratio == 2 and key in slots:
            raise ValueError("requests must own distinct compressor state slots")
        slots.add(key)
        pages = request.pages.get(compressed_group)
        needed = (request.end // ratio + 127) // 128
        if not isinstance(pages, (tuple, list)) or len(pages) < needed:
            raise ValueError("compressed pages must cover all completed history")
        for page in pages:
            if type(page) is not int or not 0 <= page < cache_pages:
                raise ValueError("compressed page is outside the joint KV/index pool")
            if (group, page) in owners:
                raise ValueError("compressed pages must be private within each DP group")
            owners.add((group, page))
        columns = max(columns, len(pages))
        groups[group].append((request, counts[group], tuple(pages)))
        counts[group] += count
    # Include DP scratch plus replicated TP outputs and conservative small work arrays.
    estimate = topology.dp * (topology.capacity * 40 + max_requests * (columns + 4) * 4) * (topology.tp + 1)
    if estimate > max_prepare_bytes:
        raise ValueError("compressed metadata exceeds its allocation budget")
    shape = (topology.dp, topology.capacity)
    positions = torch.zeros(shape, dtype=torch.int32)
    ids = torch.full(shape, -1, dtype=torch.int32)
    lens = torch.zeros(shape, dtype=torch.int32)
    writes = torch.full(shape, -1, dtype=torch.int64)
    rope = torch.full(shape, -1, dtype=torch.int32)
    starts = torch.zeros(topology.dp, max_requests + 1, dtype=torch.int32)
    tables = torch.full((topology.dp, max_requests, columns), -1, dtype=torch.int32)
    states = torch.full((topology.dp, max_requests, 1), -1, dtype=torch.int32)
    for group, requests in enumerate(groups):
        for batch_row, (request, offset, pages) in enumerate(requests):
            count = len(request.token_ids)
            section = slice(offset, offset + count)
            p = torch.arange(request.start, request.end, dtype=torch.int32)
            positions[group, section] = p
            ids[group, section] = batch_row
            lens[group, section] = (p + 1) // ratio
            if ratio == 2:
                states[group, batch_row, 0] = request.state_slot
            if pages:
                tables[group, batch_row, :len(pages)] = torch.tensor(pages, dtype=torch.int32)
            starts[group, batch_row + 1:] = offset + count
            for local, position in enumerate(range(request.start, request.end)):
                if (position + 1) % ratio == 0:
                    logical = position // ratio
                    writes[group, offset + local] = pages[logical // 128] * 128 + logical % 128
                    rope[group, offset + local] = position + 1 - ratio
    def replicate(tensor):
        return tensor.repeat_interleave(topology.tp, dim=0)
    return CompressedMetadata(*(replicate(t) for t in (positions, ids, starts, lens, writes, rope, tables, states)),
                              tuple(counts))
