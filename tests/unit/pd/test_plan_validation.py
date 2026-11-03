# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Plan permission lookup remains request-local and preserves all page checks."""

import math
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest

from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
from pypto_serving.pd.adapter import CACHE_COMPONENTS, DeepSeekDSpark
from pypto_serving.pd.integration import Chunk
from pypto_serving.pd.planner import Geometry, plan_chunk, rank_mapping
from pypto_serving.pd.protocol import Binding, Continuation, Node, Placement, Topology
from pypto_serving.pd.service import Service
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from tests.unit.pd.test_service import warm
from tests.unit.serving.test_grouped_prefix_cache import GEOMETRY


def case(threshold, *, hit=0, final=True):
    length = 8205
    tokens = (1000000,) + tuple(range(1, length))
    groups = tuple(KVCacheGroupSpec(name, (0, 1), KVCacheSpec(page, page // ratio, ratio),
        blocks, num_blocks=4 * blocks, sliding_window=window)
        for name, page, ratio, blocks, window in GEOMETRY
        for blocks in [math.ceil((length + 512) / page) if window is None
                       else math.ceil((window - 1 + threshold) / page) + 1])
    p, d = KvCacheManager(enable_prefix_cache=True), KvCacheManager(enable_prefix_cache=True)
    for cache in (p, d):
        cache.init_groups(groups, max_batch_size=2)
    warm(d, tokens, hit)
    specs = {g.name: g for g in groups}
    geometry = tuple(Geometry(c.name, (0, 1), d.group_num_blocks(c.group) + 4,
        specs[c.group].spec.storage_block_size, 64, "uint8") for c in CACHE_COMPONENTS) + (
            Geometry("bootstrap_hidden", (0, 1), 8, 128, 64, "uint8"),)
    adapter = DeepSeekDSpark()
    contract = adapter.contract("model", geometry)
    pn = Node("p", "prefill", "p-boot", contract, "p-registry", Topology.for_tp(16, 4))
    dn = Node("d", "decode", "d-boot", contract, "d-registry", Topology.for_tp(16, 1))
    binding = Binding.create(pn, dn, {})
    prefix = adapter.prefix_spec(tokens, d)
    reservation = d.reserve_group_cache(binding.request, binding.request, length + 512,
        prompt_token_count=length, prefix_block_hashes={k: [bytes.fromhex(h) for h in hashes]
        for k, hashes in prefix.hashes.items()}, shareable_group_names=tuple(prefix.hashes))
    assert reservation.prefix_hit_tokens == hit
    destination = Service(SimpleNamespace(prefill=(0,), decode=(0,), capacity=2),
        SimpleNamespace(node_id="d"), "decode", adapter, SimpleNamespace(kv_cache_manager=d), None)
    destination.node, destination.geometry = dn, geometry
    entry = SimpleNamespace(binding=binding, reservation=reservation, pending=None, mapping=(),
        continuation=Continuation(tokens, {"max_new_tokens": 512}, None, None), chunk=0, end=0,
        normalized_digest="continuation", state_slots={"bootstrap": 1})
    end = length if final else 128
    tables = p.ensure_group_blocks("source", end, partition=0)
    chunk = Chunk(binding.request, 0, 0, end, final, 7 if final else None,
                  {k: tuple(v) for k, v in tables.items()}, 0, {"bootstrap": 0})
    placement = Placement(binding.request, hit, reservation.partition,
                          reservation.block_ids_by_group, entry.state_slots)
    plan = plan_chunk(binding, chunk, placement, contract, groups, geometry, geometry, 0,
        rank_mapping(pn.topology, dn.topology, 0, reservation.partition), entry.normalized_digest)
    return destination, entry, plan, geometry, 0, pn.topology


class Pages(tuple):
    def __new__(cls, pages):
        result = super().__new__(cls, pages)
        result.iterations = 0
        return result

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()

    def __contains__(self, page):
        pytest.fail("permission lookup scanned the writable tuple")


@pytest.mark.parametrize("threshold", [128, 8192])
@pytest.mark.parametrize("final", [False, True])
def test_permissions_built_once_per_used_group_per_validation(threshold, final):
    args = case(threshold, final=final)
    destination, entry, plan, *_ = args
    pages = {name: Pages(ids) for name, ids in entry.reservation.writable_block_ids_by_group.items()}
    entry.reservation = replace(entry.reservation, writable_block_ids_by_group=pages)
    groups = {c.name: c.group for c in destination.node.contract.components}
    used = {groups[copy.component] for copy in plan.copies}
    destination.validate_plan(*args[1:])
    assert all(values.iterations == int(name in used) for name, values in pages.items())
    # A second validation rebuilds its own lookup; no request-global permission cache.
    destination.validate_plan(*args[1:])
    assert all(values.iterations == 2 * int(name in used) for name, values in pages.items())


@pytest.mark.parametrize("threshold", [128, 8192])
@pytest.mark.parametrize("change", ["shared", "empty", "bounds", "source_bounds", "unaligned", "duplicate", "missing",
                                    "rank", "source_rank", "bootstrap", "identity", "completion"])
def test_invalid_plans_still_rejected(threshold, change):
    args = list(case(threshold, hit=4096))
    destination, entry, plan, geometry, *_ = args
    destination.validate_plan(*args[1:])
    copies = list(plan.copies)
    index = next(i for i, copy in enumerate(copies) if copy.component == "hca_cmp")
    copy = copies[index]
    g = next(g for g in geometry if g.name == copy.component)
    if change == "shared":
        shared = entry.reservation.shared_block_ids_by_group["cmp_c128"][0]
        copies[index] = msgspec.structs.replace(copy, destination_offset=shared * g.page_bytes)
    elif change == "empty":
        permissions = dict(entry.reservation.writable_block_ids_by_group)
        permissions["cmp_c128"] = ()
        entry.reservation = replace(entry.reservation, writable_block_ids_by_group=permissions)
    elif change == "bounds":
        copies[index] = msgspec.structs.replace(copy, destination_offset=g.extent)
    elif change == "source_bounds":
        copies[index] = msgspec.structs.replace(copy, source_offset=g.extent)
    elif change == "unaligned":
        copies[index] = msgspec.structs.replace(copy, source_offset=copy.source_offset + 1)
    elif change == "duplicate":
        copies.append(copy)
    elif change == "missing":
        copies.pop(index)
    elif change == "rank":
        copies[index] = msgspec.structs.replace(copy, destination_rank=1)
    elif change == "source_rank":
        copies[index] = msgspec.structs.replace(copy, source_rank=16)
    elif change == "bootstrap":
        index = next(i for i, copy in enumerate(copies) if copy.component == "bootstrap_hidden")
        g = next(g for g in geometry if g.name == "bootstrap_hidden")
        copies[index] = msgspec.structs.replace(copies[index], destination_offset=0, length=g.page_bytes)
    elif change == "identity":
        plan = msgspec.structs.replace(plan, continuation_digest="another-request")
    elif change == "completion":
        plan = msgspec.structs.replace(plan, units=plan.units[:-1])
    args[2] = msgspec.structs.replace(plan, copies=tuple(copies))
    with pytest.raises(ValueError):
        destination.validate_plan(*args[1:])
