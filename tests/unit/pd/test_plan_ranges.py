# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Interval plans preserve the wire representation and page-level safety rules."""

import asyncio
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.pd.adapter import CACHE_COMPONENTS, DeepSeekDSpark
from pypto_serving.pd.integration import Chunk
from pypto_serving.pd.planner import Geometry, plan_chunk, rank_mapping
from pypto_serving.pd.protocol import Binding, Continuation, Node, Placement, Topology, digest
from pypto_serving.pd.service import Service
from tests.unit.pd.plan_reference import page_plan, page_validate
from tests.unit.serving.test_grouped_prefix_cache import GEOMETRY

ROW_BYTES = dict(ori=1024, hca_cmp=1024, csa_cmp=1024, idx_k=128, idx_scale=4,
                 hca_state=4096, csa_state=8192, csa_inner_state=2048)


def plan_case(threshold=8192, *, length=8205, end=None, start=0, number=0,
              p_hit=0, d_hit=0, d_tp=1, layout="contiguous", seed=0, layers=2, partition=0,
              pool_requests=1, request_index=0, destination_threshold=None):
    """Actual page/window widths, without tensors or allocator-sized metadata pools."""
    end = length if end is None else end
    rng = random.Random(seed)
    layer_ids = {name: tuple(range(layers)) for name, *_ in GEOMETRY}
    if layers == 43:
        for name in ("cmp_c128", "hca_state"):
            layer_ids[name] = tuple(range(3, 42, 2))
        for name in ("cmp_c4", "idx", "csa_state", "csa_inner_state"):
            layer_ids[name] = tuple(range(2, 43, 2))
    destination_threshold = threshold if destination_threshold is None else destination_threshold
    source, destination, groups, d_groups, a_blocks, b_blocks, writable = {}, {}, [], [], {}, {}, {}
    for name, page, ratio, _, window in GEOMETRY:
        ring = math.ceil((window - 1 + threshold) / page) + 1 if window else math.ceil((length + 512) / page)
        groups.append(KVCacheGroupSpec(name, layer_ids[name], KVCacheSpec(page, page // ratio, ratio),
            ring, sliding_window=window))
        d_ring = (math.ceil((window - 1 + destination_threshold) / page) + 1 if window else ring)
        d_groups.append(replace(groups[-1], max_blocks_per_seq=d_ring))
        na, nb = math.ceil(end / page), math.ceil(length / page)
        if window is not None:
            na, nb = min(ring, na), min(d_ring, nb)
        a, b = list(range(na)), list(range(nb))
        if layout == "fragmented":
            a, b = [2 * i for i in a], [3 * i + 7 for i in b]
        elif layout == "random":
            rng.shuffle(a)
            rng.shuffle(b)
        elif layout == "wrapped":
            shift_a, shift_b = (math.ceil(end / page) + 1) % na, (math.ceil(length / page) + 3) % nb
            a, b = a[shift_a:] + a[:shift_a], b[shift_b:] + b[:shift_b]
        elif layout == "contiguous":
            b = [i + 7 for i in b]
        else:
            assert layout == "packed"
        a_stride = max(a) + (1 if layout == "packed" else 11)
        b_stride = max(b) + (1 if layout == "packed" else 23)
        a, b = [p + request_index * a_stride for p in a], [p + request_index * b_stride for p in b]
        source[name], destination[name] = tuple(a), tuple(b)
        a_blocks[name], b_blocks[name] = pool_requests * a_stride, pool_requests * b_stride
        shared = set(b[:d_hit // page]) if window is None else set()
        writable[name] = tuple(p for p in b if p not in shared)
    specs = {g.name: g for g in groups}
    def geometry(blocks):
        return tuple(Geometry(c.name, specs[c.group].layer_indices, blocks[c.group],
            specs[c.group].spec.storage_block_size, ROW_BYTES[c.name], "uint8") for c in CACHE_COMPONENTS) + (
                Geometry("bootstrap_hidden", (0,), 4, 128, 24576, "uint8"),)
    pg, dg = geometry(a_blocks), geometry(b_blocks)
    adapter = DeepSeekDSpark()
    pn = Node("p", "prefill", "p-boot", adapter.contract("model", pg), "p-registry", Topology.for_tp(16, 4))
    dn = Node("d", "decode", "d-boot", adapter.contract("model", dg), "d-registry", Topology.for_tp(16, d_tp))
    binding = Binding.create(pn, dn, {})
    chunk = Chunk(binding.request, number, start, end, end == length, 7 if end == length else None,
                  source, 1, {"bootstrap": 0})
    placement = Placement(binding.request, d_hit, partition, destination, {"bootstrap": 1})
    reserved = SimpleNamespace(partition=partition, prefix_hit_tokens=d_hit, block_ids_by_group=destination,
                               writable_block_ids_by_group=writable)
    cache = SimpleNamespace(group_specs=tuple(d_groups), group_prefix_cache_alignment=128)
    core = SimpleNamespace(kv_cache_manager=cache, is_ready=lambda: True)
    service = Service(SimpleNamespace(prefill=(0,), decode=(0,), capacity=16),
                      SimpleNamespace(node_id="d"), "decode", adapter, core, None)
    service.node, service.geometry = dn, dg
    entry = SimpleNamespace(binding=binding, reservation=reserved, pending=None, mapping=(),
        normalized_digest="continuation", chunk=number, end=0 if number == 0 else start,
        continuation=Continuation(tuple(range(length)), {"max_new_tokens": 512}, None, None),
        state_slots={"bootstrap": 1})
    args = (binding, chunk, placement, pn.contract, tuple(groups), pg, dg, p_hit,
            rank_mapping(pn.topology, dn.topology, 1, partition), entry.normalized_digest,
            adapter.state_ranges(groups, end) if chunk.final else {})
    return SimpleNamespace(args=args, service=service, entry=entry, source=pn, p_hit=p_hit,
                           geometry=pg, chunk=chunk, placement=placement, groups=tuple(groups))


def compare(case, plan, *, valid=True):
    args = (case.service, case.entry, plan, case.geometry, case.p_hit, case.source.topology)
    if valid:
        page_validate(*args)
        Service.validate_plan(*args)
    else:
        for validate in (page_validate, Service.validate_plan):
            with pytest.raises(ValueError):
                validate(*args)


@pytest.mark.parametrize("threshold", [128, 8192])
@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("layout", ["packed", "wrapped", "fragmented"])
@pytest.mark.parametrize("length,p_hit,d_hit", [
    (1, 0, 0), (31, 0, 0), (32, 0, 0), (33, 0, 0), (127, 0, 0), (128, 0, 0), (129, 0, 0),
    (8192, 0, 0), (8205, 0, 0), (8320, 0, 0), (8205, 4096, 0), (8205, 0, 4096),
    (8320, 4096, 4096), (8320, 8192, 4096), (8320, 4096, 8192), (8192, 8192, 8192),
])
def test_wire_and_digest_match_page_reference(threshold, d_tp, layout, length, p_hit, d_hit):
    case = plan_case(threshold, length=length, p_hit=p_hit, d_hit=d_hit, layout=layout, d_tp=d_tp,
                     partition=16 // d_tp - 1)
    old, new = page_plan(*case.args), plan_chunk(*case.args)
    assert old == new
    assert msgspec.msgpack.encode(old) == msgspec.msgpack.encode(new)
    assert msgspec.json.encode(old) == msgspec.json.encode(new)
    assert digest(old) == digest(new)
    compare(case, new)
    if d_hit == length:
        assert {c.component for c in new.copies} == {"hca_state", "csa_state", "csa_inner_state", "bootstrap_hidden"}


@pytest.mark.parametrize("threshold", [128, 8192])
@pytest.mark.parametrize("d_tp", [1, 4])
def test_multiple_chunks_preserve_prefix_and_final_state(threshold, d_tp):
    for number, (start, end) in enumerate(((0, 128), (128, 256), (256, 8192), (8192, 8320))):
        case = plan_case(threshold, length=8320, start=start, end=end, number=number,
                         d_hit=4096, d_tp=d_tp, layout="wrapped")
        plan = plan_chunk(*case.args)
        assert plan == page_plan(*case.args)
        compare(case, plan)
        assert any(c.component == "bootstrap_hidden" for c in plan.copies) == plan.final


@pytest.mark.parametrize("threshold,d_tp", [(128, 4), (8192, 1)])
def test_complete_model_layer_geometry_matches_reference(threshold, d_tp):
    case = plan_case(threshold, d_tp=d_tp, layers=43, layout="wrapped")
    plan = plan_chunk(*case.args)
    assert plan == page_plan(*case.args)
    compare(case, plan)


@pytest.mark.parametrize("p_prefix,d_prefix", [(False, False), (False, True), (True, False), (True, True)])
def test_node_prefix_flags_keep_independent_hit_semantics(p_prefix, d_prefix):
    case = plan_case(128, p_hit=4096 if p_prefix else 0, d_hit=8192 if d_prefix else 0)
    case.source = msgspec.structs.replace(case.source, prefix=p_prefix)
    case.service.node = msgspec.structs.replace(case.service.node, prefix=d_prefix)
    plan = plan_chunk(*case.args)
    assert plan == page_plan(*case.args)
    compare(case, plan)


@pytest.mark.parametrize("threshold", [128, 8192])
def test_repeated_planning_uses_new_cow_mapping_for_both_index_regions(threshold):
    case = plan_case(threshold)
    old = plan_chunk(*case.args)
    source = dict(case.chunk.blocks)
    destination = dict(case.placement.blocks)
    source_page = next(g.blocks - 1 for g in case.geometry if g.name == "idx_k")
    destination_page = next(g.blocks - 1 for g in case.service.geometry if g.name == "idx_k")
    source["idx"] = (source_page,) + source["idx"][1:]
    destination["idx"] = (destination_page,) + destination["idx"][1:]
    args = list(case.args)
    args[1] = replace(case.chunk, blocks=source)
    args[2] = msgspec.structs.replace(case.placement, blocks=destination)
    case.entry.reservation.block_ids_by_group = destination
    case.entry.reservation.writable_block_ids_by_group["idx"] = destination["idx"]
    plan = plan_chunk(*args)
    assert plan != old and plan == page_plan(*args)
    compare(case, plan)
    compare(case, old, valid=False)


@pytest.mark.parametrize("threshold", [128, 8192])
@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("seed", range(8))
def test_random_mappings_and_copy_splits_keep_equivalent_coverage(threshold, d_tp, seed):
    case = plan_case(threshold, d_tp=d_tp, layout="random", seed=seed, d_hit=(seed % 3) * 128)
    plan = plan_chunk(*case.args)
    assert plan == page_plan(*case.args)
    rng, copies = random.Random(seed), []
    geometries = {g.name: g for g in case.service.geometry}
    for copy in plan.copies:
        page = geometries[copy.component].page_bytes
        count = copy.length // page
        cuts = [0] + sorted(rng.sample(range(1, count), min(3, count - 1))) + [count]
        copies.extend(msgspec.structs.replace(copy, source_offset=copy.source_offset + a * page,
            destination_offset=copy.destination_offset + a * page, length=(b - a) * page)
            for a, b in zip(cuts, cuts[1:]))
    rng.shuffle(copies)
    compare(case, msgspec.structs.replace(plan, copies=tuple(copies)))


@pytest.mark.parametrize("threshold", [128, 8192])
@pytest.mark.parametrize("d_tp", [1, 4])
@pytest.mark.parametrize("change", [
    "shared", "empty", "source_bounds", "bounds", "source_alignment", "alignment", "zero", "partial_page",
    "rank", "source_rank", "mapping", "identity", "continuation", "chunk", "start", "end", "final", "token",
    "floor", "unit_missing", "unit_duplicate", "bootstrap", "duplicate", "contained", "overlap", "missing",
    "hole", "extra", "layout", "region", "index_unit", "index_data",
])
def test_invalid_interval_plans_match_page_reference(threshold, d_tp, change):
    case = plan_case(threshold, d_tp=d_tp, d_hit=4096)
    plan = plan_chunk(*case.args)
    copies = list(plan.copies)
    index = next(i for i, c in enumerate(copies) if c.component == "hca_cmp")
    copy = copies[index]
    geometry = next(g for g in case.service.geometry if g.name == copy.component)
    source_geometry = next(g for g in case.geometry if g.name == copy.component)
    page = geometry.page_bytes
    if change == "shared":
        copies[index] = msgspec.structs.replace(copy, destination_offset=case.placement.blocks["cmp_c128"][0] * page)
    elif change == "empty":
        case.entry.reservation.writable_block_ids_by_group["cmp_c128"] = ()
    elif change == "source_bounds":
        copies[index] = msgspec.structs.replace(copy, source_offset=source_geometry.extent)
    elif change == "bounds":
        copies[index] = msgspec.structs.replace(copy, destination_offset=geometry.extent)
    elif change in ("source_alignment", "alignment"):
        field = "source_offset" if change == "source_alignment" else "destination_offset"
        copies[index] = msgspec.structs.replace(copy, **{field: getattr(copy, field) + 1})
    elif change in ("zero", "partial_page"):
        copies[index] = msgspec.structs.replace(copy, length=0 if change == "zero" else page + 1)
    elif change in ("rank", "source_rank"):
        field = "source_rank" if change == "source_rank" else "destination_rank"
        copies[index] = msgspec.structs.replace(copy, **{field: 16})
    elif change == "mapping":
        plan = msgspec.structs.replace(plan, mapping=((0, 0),))
    elif change == "identity":
        plan = msgspec.structs.replace(plan, binding=msgspec.structs.replace(plan.binding, request="other"))
    elif change == "continuation":
        plan = msgspec.structs.replace(plan, continuation_digest="other")
    elif change in ("chunk", "start", "end"):
        plan = msgspec.structs.replace(plan, **{change: getattr(plan, change) + 1})
    elif change == "final":
        plan = msgspec.structs.replace(plan, final=False)
    elif change == "token":
        plan = msgspec.structs.replace(plan, first_token=None)
    elif change == "floor":
        plan = msgspec.structs.replace(plan, valid_from={"ori": 1})
    elif change.startswith("unit"):
        plan = msgspec.structs.replace(plan, units=plan.units[:-1] if change == "unit_missing" else plan.units + plan.units[:1])
    elif change == "index_unit":
        plan = msgspec.structs.replace(plan, units=tuple(unit for unit in plan.units if unit[1] != "idx_scale"))
    elif change == "index_data":
        copies = [c for c in copies if c.component != "idx_scale"]
    elif change == "layout":
        case.geometry = tuple(msgspec.structs.replace(g, dtype="other") if g.name == copy.component else g
                              for g in case.geometry)
    elif change == "region":
        copies[index] = msgspec.structs.replace(copy, component="other")
    elif change == "bootstrap":
        i = next(i for i, c in enumerate(copies) if c.component == "bootstrap_hidden")
        copies[i] = msgspec.structs.replace(copies[i], destination_offset=0)
    elif change == "duplicate":
        copies.append(copy)
    elif change in ("contained", "overlap"):
        copies.append(msgspec.structs.replace(copy, length=page,
            source_offset=copy.source_offset + (page if change == "overlap" else 0),
            destination_offset=copy.destination_offset + (copy.length - page if change == "overlap" else page)))
    elif change == "missing":
        copies.pop(index)
    elif change == "hole":
        assert copy.length >= 3 * page
        copies[index:index + 1] = [msgspec.structs.replace(copy, length=page),
            msgspec.structs.replace(copy, source_offset=copy.source_offset + 2 * page,
                destination_offset=copy.destination_offset + 2 * page, length=copy.length - 2 * page)]
    else:
        assert change == "extra"
        case.entry.reservation.writable_block_ids_by_group["cmp_c128"] += (geometry.blocks - 1,)
        copies.append(msgspec.structs.replace(copy, source_offset=source_geometry.extent - page,
                                             destination_offset=geometry.extent - page, length=page))
    compare(case, msgspec.structs.replace(plan, copies=tuple(copies)), valid=False)


@pytest.mark.parametrize("change", ["overlap", "hole", "index_hole"])
def test_cross_layer_copies_keep_page_level_overlap_and_hole_checks(change):
    case = plan_case(128, length=256, layout="packed", layers=3)
    plan = plan_chunk(*case.args)
    copies = list(plan.copies)
    component = "idx_scale" if change == "index_hole" else "hca_state"
    i = next(i for i, c in enumerate(copies) if c.component == component)
    copy = copies[i]
    g = next(g for g in case.service.geometry if g.name == component)
    page, boundary = g.page_bytes, g.blocks * g.page_bytes
    assert copy.length == len(g.layers) * boundary
    if change == "overlap":
        copies.append(msgspec.structs.replace(copy, source_offset=boundary - page,
            destination_offset=boundary - page, length=2 * page))
    else:
        copies[i:i + 1] = [msgspec.structs.replace(copy, length=boundary - page),
            msgspec.structs.replace(copy, source_offset=boundary + page,
                destination_offset=boundary + page, length=copy.length - boundary - page)]
    compare(case, msgspec.structs.replace(plan, copies=tuple(copies)), valid=False)


def test_control_records_check_and_digest_without_authorizing_invalid_plan():
    async def run():
        case = plan_case(128)
        plan = plan_chunk(*case.args)
        events, authorized = [], []
        case.entry.cancelled = False
        case.entry.prefix = SimpleNamespace(alignment=128)
        case.service.requests[plan.binding.request] = case.entry
        case.service.cache.authorize_group_cache_write = authorized.append
        case.service.observe = lambda event, entry, **fields: events.append((event, fields))
        payload = dict(binding=msgspec.to_builtins(plan.binding), plan=msgspec.to_builtins(plan), p_hit=0)
        bad = msgspec.structs.replace(plan, copies=plan.copies + plan.copies[:1])
        with pytest.raises(ValueError):
            await case.service.control(case.source, case.geometry, "plan",
                dict(payload, plan=msgspec.to_builtins(bad)))
        assert not authorized and not events and case.entry.pending is None
        result = await case.service.control(case.source, case.geometry, "plan", payload)
        assert result == digest(plan) and authorized == [plan.binding.request]
        event, fields = events[-1]
        assert event == "plan.authorized"
        assert fields["validation_ms"] >= fields["plan_check_ms"] + fields["plan_digest_ms"] >= 0
        assert fields["validation_thread_cpu_ms"] >= fields["plan_check_thread_cpu_ms"] + fields["plan_digest_thread_cpu_ms"] >= 0
    asyncio.run(run())
