# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lower private full-history pages to the existing SWA composite metadata ABI."""
from dataclasses import dataclass

import torch

from .request_state import ForwardStep
from .swa_segment import SegmentTopology


@dataclass(frozen=True)
class SwaWindowMetadata:
    positions: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    group_counts: tuple[int, ...]


def gather_swa_rope_rows(
    metadata: SwaWindowMetadata,
    tables: tuple[torch.Tensor, torch.Tensor],
    *,
    max_prepare_bytes: int = 16 << 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather lib-built FP32 SWA tables in the same order as cache metadata.

    Like V4, the executor builds tables with the selected lib's host helper.
    This boundary only selects rows; it does not implement model-side RoPE.
    Padding receives identity rotation and does not index the table.
    """
    if not isinstance(metadata, SwaWindowMetadata) or len(tables) != 2:
        raise ValueError("SWA RoPE requires metadata and a cosine/sine table pair")
    positions, slots = metadata.positions, metadata.window_slots
    if (positions.device.type != "cpu" or positions.dtype != torch.int64 or positions.ndim != 2
            or slots.device.type != "cpu" or slots.dtype != torch.int64 or slots.shape != positions.shape):
        raise ValueError("SWA positions and slots must be matching CPU INT64 matrices")
    cos, sin = tables
    if any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.dtype != torch.float32
           or t.layout != torch.strided or t.ndim != 2 for t in tables):
        raise ValueError("SWA RoPE tables must be CPU FP32 matrices")
    if cos.shape != sin.shape or min(cos.shape) <= 0:
        raise ValueError("SWA cosine/sine tables must have matching nonempty shapes")
    estimate = positions.numel() * (cos.shape[1] * 4 * 4 + 32)
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0 or estimate > max_prepare_bytes:
        raise ValueError("SWA RoPE preparation exceeds its allocation budget")
    active = slots >= 0
    selected = positions[active]
    if bool(((selected < 0) | (selected >= cos.shape[0])).any()):
        raise ValueError("active SWA position is outside the RoPE table")
    selected_cos, selected_sin = cos[selected], sin[selected]
    if not bool(torch.isfinite(selected_cos).all() and torch.isfinite(selected_sin).all()):
        raise ValueError("selected SWA RoPE rows must be finite")
    shape = (*positions.shape, cos.shape[1])
    rows_cos, rows_sin = torch.ones(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)
    rows_cos[active], rows_sin[active] = selected_cos, selected_sin
    return rows_cos, rows_sin


def prepare_swa_window_metadata(
    step: ForwardStep,
    topology: SegmentTopology,
    *,
    cache_pages: int,
    window_group: str = "window",
    max_prepare_bytes: int = 16 << 20,
) -> SwaWindowMetadata:
    """Map each active row to a unique cache write and at most 128 causal reads.

    Page IDs address a partition-private cache of [pages, 128, 1, 512] payloads.
    Full-history tables preserve old rows while the whole new chunk is published;
    a 128-slot modulo ring would overwrite keys needed by early chunk queries.
    TP peers receive the same group-global metadata. No cache contents, RoPE,
    request ownership or completion state are changed by this host preparation.
    """
    if not isinstance(step, ForwardStep) or not isinstance(topology, SegmentTopology):
        raise ValueError("SWA metadata requires a ForwardStep and SegmentTopology")
    if type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128:
        raise ValueError("cache_pages must fit the INT32 window-index ABI")
    if type(max_prepare_bytes) is not int or max_prepare_bytes <= 0:
        raise ValueError("max_prepare_bytes must be a positive integer")
    if step.phase not in ("prefill", "decode"):
        raise ValueError("unsupported forward phase")
    counts, ranges, owners, requests = [0] * topology.dp, [], set(), set()
    for request in step.requests:
        group, count = request.partition, len(request.token_ids)
        if (type(group) is not int or not 0 <= group < topology.dp or count <= 0
                or type(request.start) is not int or request.start < 0):
            raise ValueError("request partition, start and token count must be valid")
        if request.request_id in requests:
            raise ValueError("duplicate request in forward step")
        requests.add(request.request_id)
        if step.phase == "decode" and count != 1:
            raise ValueError("decode requires one token per request")
        if counts[group] + count > topology.capacity:
            raise ValueError("DP token count exceeds SWA capacity")
        pages = request.pages.get(window_group)
        if not isinstance(pages, (tuple, list)) or len(pages) < (request.end + 127) // 128:
            raise ValueError("SWA requires a full-history physical page table through the chunk end")
        for page in pages:
            if type(page) is not int or not 0 <= page < cache_pages:
                raise ValueError("SWA physical page is outside its partition cache pool")
            key = (group, page)
            if key in owners:
                raise ValueError("SWA pages must be private; shared or repeated physical pages are unsupported")
            owners.add(key)
        ranges.append((group, counts[group], request.start, count, tuple(pages)))
        counts[group] += count
    rows = topology.dp * topology.capacity
    estimate = rows * (128 * 4 + 2 * 8) * (topology.tp + 1)
    if estimate > max_prepare_bytes:
        raise ValueError(f"SWA metadata requires estimated {estimate} bytes; budget={max_prepare_bytes}")
    positions = torch.zeros(topology.dp, topology.capacity, dtype=torch.int64)
    slots = torch.full_like(positions, -1)
    indices = torch.full((topology.dp, topology.capacity, 128), -1, dtype=torch.int32)
    for group, row, start, count, pages in ranges:
        for local in range(count):
            position = start + local
            positions[group, row + local] = position
            slots[group, row + local] = pages[position // 128] * 128 + position % 128
            visible = range(max(0, position - 127), position + 1)
            addresses = [pages[p // 128] * 128 + p % 128 for p in visible]
            indices[group, row + local, :len(addresses)] = torch.tensor(addresses, dtype=torch.int32)
    return SwaWindowMetadata(
        positions.repeat_interleave(topology.tp, dim=0),
        slots.repeat_interleave(topology.tp, dim=0),
        indices.repeat_interleave(topology.tp, dim=0), tuple(counts),
    )
