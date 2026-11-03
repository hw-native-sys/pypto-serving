# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure chunk planning from Serving snapshots, never from a private Prefix index."""
"""Page-by-page reference algorithms for plan equivalence and safety tests."""

import msgspec

from pypto_serving.pd.planner import page_indices, physical, rolling_bounds, valid_from
from pypto_serving.pd.protocol import Copy, Plan


def page_plan(binding, chunk, placement, contract, groups, source, destination, p_hit, mapping, continuation_digest):
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
    copies, units = [], []
    for sr, dr in mapping:
        for component in contract.components:
            if component.final and not chunk.final:
                continue
            a, b = src[component.name], dst[component.name]
            if a.layout != b.layout or min(a.blocks, b.blocks) <= 0:
                raise ValueError("incompatible source/destination geometry")
            units.append((dr, component.name))
            if component.group in specs:
                group = specs[component.group]
                pages = tuple((physical(group, chunk.blocks[group.name], logical),
                               physical(group, placement.blocks[group.name], logical))
                              for logical, _ in page_indices(group, component, start, chunk.end, chunk.final,
                                  chunk.blocks[group.name], p_hit, placement.hit, floors))
            else:
                pages = ((chunk.state_slots[component.group], placement.state_slots[component.group]),)
            if any(min(sb, db) < 0 or sb >= a.blocks or db >= b.blocks for sb, db in pages):
                raise ValueError("physical block outside registered region")
            for layer in range(len(a.layers)):
                for sb, db in sorted(pages):
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
                tuple(copies), tuple(units), chunk.first_token, floors)


def page_validate(self, entry, plan, geometry, p_hit, source_topology):
    if (entry.reservation is None or entry.pending is not None or plan.binding != entry.binding
            or plan.continuation_digest != entry.normalized_digest
            or plan.chunk != entry.chunk or plan.start != entry.end
            or not plan.start < plan.end <= len(entry.continuation.tokens)
            or plan.final != (plan.end == len(entry.continuation.tokens))
            or (plan.final and plan.first_token is None)):
        raise ValueError("invalid chunk sequence or final boundary")
    topology = self.node.topology
    self.adapter.validate_mapping(source_topology, topology, entry.reservation, plan.mapping)
    if entry.mapping and entry.mapping != plan.mapping:
        raise ValueError("rank mapping changed during a request")
    expected = {(rank, c.name) for rank in topology.groups[entry.reservation.partition]
                for c in self.node.contract.components if plan.final or not c.final}
    if set(plan.units) != expected or len(plan.units) != len(expected):
        raise ValueError("missing required rank/region completion units")
    final_groups = {c.group for c in self.node.contract.components if c.final}
    rolling = {g.name: g for g in self.cache.group_specs
               if g.sliding_window is not None and g.name not in final_groups} if plan.final else {}
    if set(plan.valid_from) != set(rolling):
        raise ValueError("invalid rolling publication groups")
    for name, group in rolling.items():
        floor = plan.valid_from[name]
        lower, upper = rolling_bounds(group, p_hit, entry.reservation.prefix_hit_tokens,
                                      plan.end, self.cache.group_prefix_cache_alignment)
        page = group.spec.token_capacity
        table = entry.reservation.block_ids_by_group[name]
        lower = max(lower, max(0, (plan.end + page - 1) // page - len(table)) * page)
        if type(floor) is not int or floor % page or not lower <= floor <= upper:
            raise ValueError("invalid rolling publication floor")
    geo = {g.name: g for g in self.geometry}
    sources = {g.name: g for g in geometry}
    component_group = {c.name: c.group for c in self.node.contract.components}
    occupied = set()
    writable_sets = {}
    for copy in plan.copies:
        if ((copy.destination_rank, copy.component) not in expected
                or not 0 <= copy.source_rank < source_topology.ranks):
            raise ValueError("copy outside rank/component mapping")
        if (copy.source_rank, copy.destination_rank) not in plan.mapping:
            raise ValueError("copy outside request rank mapping")
        g, src = geo[copy.component], sources[copy.component]
        if (g.layout != src.layout or min(copy.source_offset, copy.destination_offset) < 0
                or copy.length <= 0 or copy.length % g.page_bytes
                or copy.destination_offset % g.page_bytes or copy.source_offset % src.page_bytes
                or copy.destination_offset + copy.length > g.extent
                or copy.source_offset + copy.length > src.extent):
            raise ValueError("copy outside compatible physical layout")
        group_name = component_group[copy.component]
        if group_name not in writable_sets:
            writable = (entry.reservation.writable_block_ids_by_group[group_name]
                        if group_name in entry.reservation.writable_block_ids_by_group
                        else (entry.state_slots[group_name],))
            writable_sets[group_name] = frozenset(writable)
        for page in range(copy.destination_offset // g.page_bytes,
                          (copy.destination_offset + copy.length) // g.page_bytes):
            key = copy.destination_rank, copy.component, page
            if page % g.blocks not in writable_sets[group_name] or key in occupied:
                raise ValueError("write aliases shared Prefix or another write")
            occupied.add(key)
    required = set()
    groups = {group.name: group for group in self.cache.group_specs}
    for _, rank in plan.mapping:
        for component in self.node.contract.components:
            if component.final and not plan.final:
                continue
            g = geo[component.name]
            if component.group in groups:
                group = groups[component.group]
                table = entry.reservation.block_ids_by_group[component.group]
                blocks = (physical(group, table, logical) for logical, _ in page_indices(
                    group, component, plan.start, plan.end, plan.final,
                    table, p_hit, entry.reservation.prefix_hit_tokens, plan.valid_from))
            else:
                blocks = (entry.state_slots[component.group],)
            for block in blocks:
                required.update((rank, component.name, layer * g.blocks + block)
                                for layer in range(len(g.layers)))
    if occupied != required:
        raise ValueError("copy plan does not cover the required destination pages")
