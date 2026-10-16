# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure chunk planning from Serving snapshots, never from a private Prefix index."""

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


def rank_mapping(ranks, tp, source_partition, destination_partition):
    if ranks <= 0 or tp <= 0 or ranks % tp or any(
        type(p) is not int or not 0 <= p < ranks // tp for p in (source_partition, destination_partition)
    ):
        raise ValueError("invalid partition/rank topology")
    return tuple((source_partition * tp + slot, destination_partition * tp + slot) for slot in range(tp))


def valid_from(groups, components, p_hit, d_hit):
    final = {c.group for c in components if c.final}
    return {g.name: max(d_hit, p_hit - g.sliding_window) for g in groups
            if g.sliding_window is not None and g.name not in final}


def page_indices(group, component, start, end, final, source_table, p_hit, d_hit):
    capacity = group.spec.token_capacity
    logical_end = (end + capacity - 1) // capacity
    if component.final:
        if not final:
            return ()
        indices = range(max(0, logical_end - len(source_table)), logical_end)
    else:
        floor = max(start, d_hit, p_hit - group.sliding_window if group.sliding_window else 0)
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
    copies, units = [], []
    for sr, dr in mapping:
        for component in contract.components:
            if component.final and not chunk.final:
                continue
            group, a, b = specs[component.group], src[component.name], dst[component.name]
            if a.layout != b.layout or min(a.blocks, b.blocks) <= 0:
                raise ValueError("incompatible source/destination geometry")
            units.append((dr, component.name))
            for logical, _ in page_indices(group, component, start, chunk.end, chunk.final,
                                           chunk.blocks[group.name], p_hit, placement.hit):
                sb = physical(group, chunk.blocks[group.name], logical)
                db = physical(group, placement.blocks[group.name], logical)
                if sb >= a.blocks or db >= b.blocks:
                    raise ValueError("physical block outside registered region")
                for layer in range(len(a.layers)):
                    # Final partial pages are exclusively reserved and copied whole.
                    copy = Copy(component.name, sr, dr, (layer * a.blocks + sb) * a.page_bytes,
                                (layer * b.blocks + db) * b.page_bytes, a.page_bytes)
                    if copies and copies[-1].component == copy.component:
                        previous = copies[-1]
                        if (previous.source_rank == sr and previous.destination_rank == dr
                                and previous.source_offset + previous.length == copy.source_offset
                                and previous.destination_offset + previous.length == copy.destination_offset):
                            copies[-1] = msgspec.structs.replace(previous, length=previous.length + copy.length)
                            continue
                    copies.append(copy)
    return Plan(binding, continuation_digest, chunk.chunk, start, chunk.end, chunk.final, mapping,
                tuple(copies), tuple(units), chunk.first_token)


class SourcePins:
    def __init__(self, cache):
        self.cache = cache
        self.active = {}

    def retain(self, chunk):
        if chunk.request in self.active:
            raise RuntimeError("one transfer per request may be in flight")
        tables = {k: list(v) for k, v in chunk.blocks.items()}
        self.cache.retain_group_block_snapshot(tables, chunk.partition)
        self.active[chunk.request] = (tables, chunk.partition)

    def release(self, request):
        tables, partition = self.active.pop(request)
        self.cache.release_group_block_snapshot(tables, partition)
