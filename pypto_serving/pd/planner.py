# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure chunk planning from Serving snapshots, never from a private Prefix index."""

import math

import msgspec

from .protocol import Copy, Plan


class Geometry(msgspec.Struct, frozen=True):
    name: str
    layers: tuple[int, ...]
    blocks: int
    rows: int
    row_bytes: int
    dtype: str

    @property
    def page_bytes(self):
        return self.rows * self.row_bytes

    @property
    def extent(self):
        return len(self.layers) * self.blocks * self.page_bytes

    @property
    def layout(self):
        return self.name, self.layers, self.rows, self.row_bytes, self.dtype


def rank_mapping(source_topology, destination_topology, source_partition, destination_partition):
    """Select a complete source replica for every required destination owner."""
    for topology, partition in ((source_topology, source_partition),
                                (destination_topology, destination_partition)):
        if type(partition) is not int or not 0 <= partition < len(topology.groups):
            raise ValueError("invalid partition/rank topology")
    source = source_topology.groups[source_partition]
    return tuple((source[rank % len(source)], rank)
                 for rank in destination_topology.groups[destination_partition])


def rolling_bounds(group, p_hit, d_hit, end, alignment):
    """Keep the Decode tail and the latest queryable checkpoint, when resident."""
    page = group.spec.token_capacity
    boundary = max(0, end - 1) // alignment * alignment
    lower = max(d_hit, p_hit - group.sliding_window, max(0, boundary - group.sliding_window))
    upper = max(d_hit, max(0, end - group.sliding_window) // page * page)
    return lower, upper


def valid_from(groups, components, p_hit, d_hit, end, source, destination):
    final = {c.group for c in components if c.final}
    alignment = math.lcm(*(g.spec.token_capacity for g in groups))
    floors = {}
    for group in groups:
        if group.sliding_window is None or group.name in final:
            continue
        page = group.spec.token_capacity
        lower, upper = rolling_bounds(group, p_hit, d_hit, end, alignment)
        logical_end = math.ceil(end / page)
        floor = max(lower, max(0, logical_end - min(len(source[group.name]),
                    len(destination[group.name]))) * page)
        if floor > upper:
            raise ValueError("rolling snapshot cannot cover the Decode tail")
        floors[group.name] = floor
    return floors


def page_indices(group, component, start, end, final, source_table, p_hit, d_hit, publication_floor):
    capacity = group.spec.token_capacity
    logical_end = (end + capacity - 1) // capacity
    if component.final:
        if not final:
            return ()
        indices = range(max(0, logical_end - len(source_table)), logical_end)
    else:
        if group.sliding_window is not None:
            if not final:
                return ()  # D consumes rolling KV only after final commit.
            floor = publication_floor[group.name]
        else:
            floor = max(start, d_hit)
        indices = range(floor // capacity, end // capacity)
    rows = {i: group.spec.storage_block_size for i in indices}
    remainder = end % capacity
    if final and remainder and (component.final or end > d_hit):
        rows[end // capacity] = remainder // group.spec.compress_ratio
    return tuple((i, n) for i, n in sorted(rows.items()) if n > 0)


def physical(group, table, logical):
    if not table:
        raise ValueError("empty cache table")
    slot = logical if group.sliding_window is None else logical % len(table)
    if slot >= len(table):
        raise ValueError("logical cache block outside snapshot")
    return table[slot]


def page_ranges(pages):
    """Canonical half-open intervals for a set of physical page IDs."""
    ranges = []
    for page in sorted(set(pages)):
        if ranges and ranges[-1][1] == page:
            ranges[-1] = (ranges[-1][0], page + 1)
        else:
            ranges.append((page, page + 1))
    return tuple(ranges)


def merge_ranges(ranges, *, reject_overlap=False):
    """Normalize coverage, rejecting repeated writes before merging adjacency."""
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            if reject_overlap and start < result[-1][1]:
                raise ValueError("write aliases shared Prefix or another write")
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


def _mapped_runs(pages):
    """Merge only page pairs contiguous in both source and destination order."""
    runs = []
    for source, destination in sorted(pages):
        if runs and (runs[-1][0] + runs[-1][2], runs[-1][1] + runs[-1][2]) == (source, destination):
            a, b, count = runs[-1]
            runs[-1] = (a, b, count + 1)
        else:
            runs.append((source, destination, 1))
    return tuple(runs)


def plan_chunk(binding, chunk, placement, contract, groups, source, destination, p_hit, mapping, continuation_digest):
    specs = {g.name: g for g in groups}
    src = {g.name: g for g in source}
    dst = {g.name: g for g in destination}
    start = 0 if chunk.chunk == 0 else chunk.start
    if chunk.request != binding.request or placement.request != binding.request:
        raise ValueError("chunk/placement belongs to another request")
    if not 0 <= start < chunk.end or placement.hit < 0 or not 0 <= p_hit <= chunk.end:
        raise ValueError("invalid chunk or Prefix range")
    if set(chunk.blocks) != set(specs) or set(placement.blocks) != set(specs):
        raise ValueError("cache snapshot differs from model groups")
    for tables in (chunk.blocks, placement.blocks):
        if any(len(table) != len(set(table)) or any(type(b) is not int or b < 0 for b in table)
               for table in tables.values()):
            raise ValueError("invalid physical block table")
    floors = (valid_from(groups, contract.components, p_hit, placement.hit, chunk.end,
                         chunk.blocks, placement.blocks) if chunk.final else {})
    copies, units, mapped = [], [], {}
    for sr, dr in mapping:
        for component in contract.components:
            if component.final and not chunk.final:
                continue
            a, b = src[component.name], dst[component.name]
            if a.layout != b.layout or min(a.blocks, b.blocks) <= 0:
                raise ValueError("incompatible source/destination geometry")
            units.append((dr, component.name))
            key = component.group, component.final
            if key not in mapped:
                if component.group in specs:
                    group = specs[component.group]
                    pages = ((physical(group, chunk.blocks[group.name], logical),
                              physical(group, placement.blocks[group.name], logical))
                             for logical, _ in page_indices(group, component, start, chunk.end, chunk.final,
                                 chunk.blocks[group.name], p_hit, placement.hit, floors))
                else:
                    pages = ((chunk.state_slots[component.group], placement.state_slots[component.group]),)
                mapped[key] = _mapped_runs(pages)
            runs = mapped[key]
            if any(min(sb, db) < 0 or sb + count > a.blocks or db + count > b.blocks
                   for sb, db, count in runs):
                raise ValueError("physical block outside registered region")
            for layer in range(len(a.layers)):
                for sb, db, count in runs:
                    # Final partial pages are exclusively reserved and copied whole.
                    copy = Copy(component.name, sr, dr, (layer * a.blocks + sb) * a.page_bytes,
                                (layer * b.blocks + db) * b.page_bytes, count * a.page_bytes)
                    if copies and copies[-1].component == copy.component:
                        previous = copies[-1]
                        if (previous.source_rank == sr and previous.destination_rank == dr
                                and previous.source_offset + previous.length == copy.source_offset
                                and previous.destination_offset + previous.length == copy.destination_offset):
                            copies[-1] = msgspec.structs.replace(previous, length=previous.length + copy.length)
                            continue
                    copies.append(copy)
    return Plan(binding, continuation_digest, chunk.chunk, start, chunk.end, chunk.final, mapping,
                tuple(copies), tuple(units), chunk.first_token, floors)


class SourcePins:
    def __init__(self, cache):
        self.cache = cache
        self.active = {}

    def retain(self, chunk):
        if chunk.request in self.active:
            raise RuntimeError("one transfer per request may be in flight")
        self.active[chunk.request] = self.cache.retain_group_block_snapshot(chunk.blocks, chunk.partition)

    def release(self, request):
        self.cache.release_group_block_snapshot(self.active.pop(request))
