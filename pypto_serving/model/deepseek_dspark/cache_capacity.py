# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-rank DSpark pool sizing; request ring periods remain kernel-owned."""

import math


def cache_pool_bytes(groups, counts, *, filler_pages):
    """Budget filler and each allocation's ACL/HBM huge-page backing.

    The allocator's 512-byte logical alignment is not its physical HBM
    footprint. Round each allocation separately and allow one further huge
    page for allocator/runtime bookkeeping between the budget and postcheck.
    This allowance stays inside, rather than increasing, the memory target.
    """
    alignment = 2 << 20
    total = 0
    for group in groups:
        size = (counts[group.name] + filler_pages) * group.spec.page_size_bytes
        # Index vectors and their FP32 scales are two allocations of one group.
        sizes = (size // 132 * 128, size // 132 * 4) if group.name == "idx" else (size,)
        for allocation in sizes:
            total += ((allocation + 32 + alignment - 1) // alignment + 1) * alignment
    return total


def plan_cache_pages(groups, budget, *, active_requests, step_tokens, filler_pages):
    """Reserve live rings, one in-flight replacement and bounded checkpoints.

    Full-history pools fit one maximum-context request, not every concurrent
    request at maximum length. Remaining bytes buy common source-token coverage.
    """
    if active_requests <= 0 or step_tokens <= 0:
        raise ValueError("cache planning requires positive request and step extents")
    counts = {}
    history = [group for group in groups if group.sliding_window is None]
    if not history:
        raise ValueError("DSpark requires full-history cache groups")
    for group in groups:
        page = group.spec.token_capacity
        if group.sliding_window is None:
            counts[group.name] = max(group.max_blocks_per_seq, active_requests * math.ceil(step_tokens / page))
        else:
            # Maximum new logical pages for one more scheduled step, including
            # unaligned starts. No request ring is extended or advanced here.
            working = (step_tokens + page - 1) // page
            checkpoint = group.sliding_window // page
            counts[group.name] = active_requests * (group.max_blocks_per_seq + working + checkpoint)
    minimum = cache_pool_bytes(groups, counts, filler_pages=filler_pages)
    if minimum > budget:
        raise RuntimeError(f"DSpark cache budget {budget} cannot fit live/async/checkpoint minimum {minimum}")
    alignment = math.lcm(*(group.spec.token_capacity for group in history))
    increment_bytes = sum(alignment // group.spec.token_capacity * group.spec.page_size_bytes
                          for group in history)
    # Extra logical pages can first consume padding already paid for by the
    # minimum. Bound by the total budget, not by its rounded remaining bytes.
    low, high = 0, budget // increment_bytes
    base = dict(counts)
    while low < high:
        extra = (low + high + 1) // 2
        for group in history:
            counts[group.name] = base[group.name] + extra * alignment // group.spec.token_capacity
        if cache_pool_bytes(groups, counts, filler_pages=filler_pages) <= budget:
            low = extra
        else:
            high = extra - 1
    for group in history:
        counts[group.name] = base[group.name] + low * alignment // group.spec.token_capacity
    return counts
