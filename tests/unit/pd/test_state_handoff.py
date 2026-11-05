# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sparse logical-content checks for unequal rings, without SDK/device memory."""

import math
from dataclasses import replace

import msgspec
import pytest

from pypto_serving.pd.adapter import CACHE_COMPONENTS, DeepSeekDSpark
from pypto_serving.pd.planner import plan_chunk
from pypto_serving.pd.protocol import digest
from tests.unit.pd.test_plan_ranges import plan_case


def assert_logical_contents(case, plan):
    """Execute page copies as logical markers, independently of the planner.

    One marker identifies component, layer and absolute logical page. Reading
    an expired source slot or using the other peer's modulo changes the marker.
    No device-sized tensor is allocated for this oracle.
    """
    groups = {g.name: g for g in case.groups}
    components = {c.name: c for c in CACHE_COMPONENTS}
    source = {g.name: g for g in case.geometry}
    destination = {g.name: g for g in case.service.geometry}
    inverse, expected = {}, set()
    boundary = max(0, plan.end - 1) // 128 * 128
    for name, group in groups.items():
        page = group.spec.token_capacity
        stop = (plan.end + page - 1) // page
        table = case.chunk.blocks[name]
        begin = max(0, stop - len(table)) if group.sliding_window else 0
        inverse[name] = {table[q % len(table) if group.sliding_window else q]: q
                         for q in range(begin, stop)}
    for _, rank in plan.mapping:
        for name, component in components.items():
            group = groups[component.group]
            page = group.spec.token_capacity
            if component.final:
                if not plan.final:
                    continue
                begin = max(0, boundary - group.sliding_window) // page
            elif group.sliding_window:
                if not plan.final:
                    continue
                resident = min(len(case.chunk.blocks[group.name]), len(case.placement.blocks[group.name]))
                begin = max(case.placement.hit, case.p_hit - group.sliding_window,
                    max(0, boundary - group.sliding_window),
                    max(0, math.ceil(plan.end / page) - resident) * page) // page
            else:
                begin = max(plan.start, case.placement.hit) // page
            stop = plan.end // page
            pages = set(range(begin, stop))
            if (plan.final and plan.end % page >= group.spec.compress_ratio
                    and (component.final or plan.end > case.placement.hit)):
                pages.add(stop)
            expected.update((rank, name, layer, q) for layer in range(len(source[name].layers)) for q in pages)
        if plan.final:
            expected.add((rank, "bootstrap_hidden", 0, "hidden"))
    actual = set()
    for copy in plan.copies:
        a, b = source[copy.component], destination[copy.component]
        for offset in range(0, copy.length, a.page_bytes):
            sl, sp = divmod((copy.source_offset + offset) // a.page_bytes, a.blocks)
            dl, dp = divmod((copy.destination_offset + offset) // b.page_bytes, b.blocks)
            assert sl == dl
            if copy.component == "bootstrap_hidden":
                assert sp == case.chunk.state_slots["bootstrap"]
                assert dp == case.placement.state_slots["bootstrap"]
                logical = "hidden"
            else:
                group = groups[components[copy.component].group]
                logical = inverse[group.name][sp]
                table = case.placement.blocks[group.name]
                slot = logical % len(table) if group.sliding_window else logical
                assert dp == table[slot]
            key = copy.destination_rank, copy.component, dl, logical
            assert key not in actual
            actual.add(key)
    assert actual == expected
    return actual


@pytest.mark.parametrize("destination_threshold", [8192, 2048])
@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("layout", ["contiguous", "wrapped", "fragmented"])
@pytest.mark.parametrize("length", [1, 31, 32, 33, 127, 128, 129, 2047, 2048, 2049,
                                   8191, 8192, 8193, 8205, 8319, 8320, 16897])
def test_terminal_state_copies_preserve_absolute_logical_content(destination_threshold, d_tp, layout, length):
    case = plan_case(8192, destination_threshold=destination_threshold, length=length, d_tp=d_tp,
                     layout=layout, partition=16 // d_tp - 1)
    plan = plan_chunk(*case.args)
    case.service.validate_plan(case.entry, plan, case.geometry, case.p_hit, case.source.topology)
    assert_logical_contents(case, plan)


@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("layout", ["contiguous", "wrapped", "fragmented"])
@pytest.mark.parametrize("length", [31, 32, 33, 127, 128, 129, 511, 512, 513,
                                   8192, 8205, 8319, 8320, 1048445])
@pytest.mark.parametrize("source_state_tokens", [1024, 8192])
def test_compact_decode_state_rings_keep_logical_handoff(d_tp, layout, length, source_state_tokens):
    case = plan_case(8192, destination_threshold=2048, destination_state_tokens=144,
                     length=length, d_tp=d_tp, layout=layout, partition=16 // d_tp - 1,
                     source_state_tokens=source_state_tokens)
    plan = plan_chunk(*case.args)
    case.service.validate_plan(case.entry, plan, case.geometry, case.p_hit, case.source.topology)
    assert_logical_contents(case, plan)
    groups = {g.name: g for g in case.service.core.kv_cache_manager.group_specs}
    assert groups["hca_state"].max_blocks_per_seq == 35
    assert groups["csa_state"].max_blocks_per_seq == groups["csa_inner_state"].max_blocks_per_seq == 77


@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("p_hit,d_hit", [(0, 0), (4096, 0), (0, 4096), (4096, 4096),
                                       (8192, 4096), (4096, 8192), (8320, 8320)])
@pytest.mark.parametrize("source_state_tokens", [1024, 8192])
def test_compact_decode_state_is_not_omitted_on_prefix_hits(d_tp, p_hit, d_hit, source_state_tokens):
    case = plan_case(8192, destination_threshold=2048, destination_state_tokens=144,
                     length=8320, p_hit=p_hit, d_hit=d_hit, d_tp=d_tp, layout="wrapped",
                     source_state_tokens=source_state_tokens)
    plan = plan_chunk(*case.args)
    case.service.validate_plan(case.entry, plan, case.geometry, p_hit, case.source.topology)
    markers = assert_logical_contents(case, plan)
    assert {"hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"} <= {
        name for _, name, _, _ in markers}


@pytest.mark.parametrize("destination_threshold", [8192, 2048])
@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("p_hit,d_hit", [(0, 0), (4096, 0), (0, 4096), (4096, 4096),
                                       (8192, 4096), (4096, 8192), (8320, 8320)])
def test_prefix_hits_do_not_remove_request_local_final_state(destination_threshold, d_tp, p_hit, d_hit):
    case = plan_case(8192, destination_threshold=destination_threshold, length=8320,
                     d_tp=d_tp, p_hit=p_hit, d_hit=d_hit, layout="wrapped", layers=43)
    plan = plan_chunk(*case.args)
    case.service.validate_plan(case.entry, plan, case.geometry, p_hit, case.source.topology)
    markers = assert_logical_contents(case, plan)
    assert {name for _, name, _, _ in markers} >= {
        "hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"}


@pytest.mark.parametrize("destination_threshold", [8192, 2048])
def test_multi_chunk_transfer_keeps_state_final_and_mapping_request_local(destination_threshold):
    for number, (start, end) in enumerate(((0, 8192), (8192, 16384), (16384, 16897))):
        case = plan_case(8192, destination_threshold=destination_threshold, length=16897,
                         start=start, end=end, number=number, layout="wrapped", d_hit=4096)
        plan = plan_chunk(*case.args)
        case.service.validate_plan(case.entry, plan, case.geometry, 0, case.source.topology)
        markers = assert_logical_contents(case, plan)
        assert any(name.endswith("state") for _, name, _, _ in markers) == plan.final


@pytest.mark.parametrize("length", [8192, 8205, 8319, 8320, 16897])
def test_state_workset_does_not_depend_on_ring_working_capacity(length):
    large = plan_case(8192, length=length)
    small = plan_case(8192, destination_threshold=2048, length=length)
    adapter = DeepSeekDSpark()
    assert adapter.state_ranges(large.groups, length) == adapter.state_ranges(
        small.service.cache.group_specs, length)
    for case in (large, small):
        markers = assert_logical_contents(case, plan_chunk(*case.args))
        boundary = max(0, length - 1) // 128 * 128
        for group in case.groups:
            if group.name not in ("hca_state", "csa_state", "csa_inner_state"):
                continue
            # Both the complete checkpoint tail and the terminal state exist.
            capacity = group.spec.token_capacity
            required = range(max(0, boundary - group.sliding_window) // capacity,
                             math.ceil(length / capacity))
            actual = {q for _, name, layer, q in markers if name == group.name and layer == 0}
            assert actual == set(required)


def test_contract_pairs_unequal_rings_but_rejects_the_whole_ring_state_policy():
    case = plan_case(8192, destination_threshold=2048, length=16897)
    assert case.source.compatible(case.service.node)
    assert case.source.contract == case.service.node.contract
    layouts = {g.name: g.layout for g in case.geometry}
    old_layout = digest(tuple(layouts[c.name] for c in case.source.contract.components))
    old = msgspec.structs.replace(case.source, contract=msgspec.structs.replace(
        case.source.contract, layout=old_layout))
    assert not old.compatible(case.service.node)


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("name", ["hca_state", "csa_state", "csa_inner_state"])
def test_insufficient_state_ring_rejects_aliases_before_a_plan_can_be_sent(side, name):
    case = plan_case(8192, destination_threshold=2048)
    args = list(case.args)
    if side == "source":
        tables = {**case.chunk.blocks, name: case.chunk.blocks[name][:1]}
        args[1] = replace(case.chunk, blocks=tables)
    else:
        tables = {**case.placement.blocks, name: case.placement.blocks[name][:1]}
        args[2] = msgspec.structs.replace(case.placement, blocks=tables)
    with pytest.raises(ValueError, match="required logical pages"):
        plan_chunk(*args)


@pytest.mark.parametrize("name", ["hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"])
def test_missing_final_region_is_not_hidden_by_an_independent_destination_ring(name):
    case = plan_case(8192, destination_threshold=2048)
    plan = plan_chunk(*case.args)
    bad = msgspec.structs.replace(plan, copies=tuple(c for c in plan.copies if c.component != name))
    with pytest.raises(ValueError):
        case.service.validate_plan(case.entry, bad, case.geometry, 0, case.source.topology)


def test_sparse_content_oracle_detects_an_in_bounds_but_wrong_source_page():
    case = plan_case(8192, destination_threshold=2048, length=16897)
    plan = plan_chunk(*case.args)
    copies = list(plan.copies)
    i = next(i for i, c in enumerate(copies) if c.component == "hca_state")
    original = copies[i]
    page = next(g.page_bytes for g in case.geometry if g.name == "hca_state")
    assert original.length > page
    copies[i:i + 1] = [msgspec.structs.replace(original, source_offset=original.source_offset + page, length=page),
        msgspec.structs.replace(original, source_offset=original.source_offset + page,
            destination_offset=original.destination_offset + page, length=original.length - page)]
    with pytest.raises(AssertionError):
        assert_logical_contents(case, msgspec.structs.replace(plan, copies=tuple(copies)))
