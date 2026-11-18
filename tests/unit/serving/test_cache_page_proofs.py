# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise real grouped allocator epochs and bounded checkpoint publication."""

from dataclasses import replace

import pytest

from pypto_serving.serving.memory.protocol import CachePublicationProof, GroupPageRange
from pypto_serving.model.deepseek_lowlat.capacity import LowlatCapacityPlan
from pypto_serving.serving.memory.kv_cache import KvCacheManager


def setup_cache():
    specs = LowlatCapacityPlan(16384, 512, "rolling", True).cache_specs()
    cache = KvCacheManager(enable_prefix_cache=True)
    cache.init_groups(specs, max_batch_size=1)
    return cache


def checkpoint(cache, request="cold", boundary=1408):
    tokens = list(range(1536))
    hashes = cache.compute_group_block_hashes(tokens)
    pages = cache.ensure_group_blocks(request, boundary)
    identities = cache.group_page_identities(pages, 0)
    ranges = tuple(
        GroupPageRange(
            s.name,
            max(0, boundary // s.spec.token_capacity - s.sliding_window // s.spec.token_capacity)
            if s.sliding_window
            else 0,
            boundary // s.spec.token_capacity,
        )
        for s in cache.group_specs
    )
    proof = CachePublicationProof(request, 1, 1, boundary, boundary, ranges, identities)
    return hashes, pages, proof


def publish(cache, hashes, pages, proof):
    cache.publish_group_page_proof(
        proof,
        request_id="cold",
        generation=1,
        mapping_version=1,
        snapshot=pages,
        partition=0,
        block_hashes=hashes,
    )


def test_joint_proof_only_publishes_materialized_state_suffix():
    cache = setup_cache()
    hashes, pages, proof = checkpoint(cache)
    publish(cache, hashes, pages, proof)
    for spec in cache.group_specs:
        pool = cache._group_pools[spec.name]
        actual = {i for i, h in enumerate(hashes[spec.name]) if (0, h) in pool.hash_to_block}
        item = next(r for r in proof.complete_logical_pages if r.group == spec.name)
        assert actual == set(range(item.start, item.end))
    cache.release_all_group_requests("cold")
    _, hit, _ = cache.acquire_group_prefix_blocks("hot", hashes, max_cache_hit_tokens=1408)
    assert hit == 1408
    cache.release_all_group_requests("hot")


@pytest.mark.parametrize(
    "corruption", ["missing-group", "overclaim-state", "stale-epoch", "wrong-generation", "wrong-mapping"]
)
def test_invalid_proofs_publish_nothing(corruption):
    cache = setup_cache()
    hashes, pages, proof = checkpoint(cache)
    if corruption == "missing-group":
        proof = replace(proof, complete_logical_pages=proof.complete_logical_pages[:-1])
    elif corruption == "overclaim-state":
        proof = replace(
            proof,
            complete_logical_pages=tuple(
                replace(r, start=0) if r.group == "hca_state" else r for r in proof.complete_logical_pages
            ),
        )
    elif corruption == "stale-epoch":
        proof = replace(
            proof,
            page_identities=(
                replace(proof.page_identities[0], allocation_epoch=99),
                *proof.page_identities[1:],
            ),
        )
    elif corruption == "wrong-generation":
        proof = replace(proof, generation=2)
    else:
        proof = replace(proof, mapping_version=2)
    with pytest.raises(ValueError):
        publish(cache, hashes, pages, proof)
    assert all(not p.hash_to_block for p in cache._group_pools.values())


def test_same_page_id_reassignment_invalidates_old_identity():
    cache = setup_cache()
    pages = cache.ensure_group_blocks("a", 512)
    before = cache.group_page_identities(pages, 0)
    cache.release_all_group_requests("a")
    seen = {}
    for i in range(8):
        current = cache.ensure_group_blocks(str(i), 512)
        for page in cache.group_page_identities(current, 0):
            seen[(page.group, page.page_id)] = page.allocation_epoch
        cache.release_all_group_requests(str(i))
    assert any(seen.get((p.group, p.page_id), 0) > p.allocation_epoch for p in before)


@pytest.mark.parametrize("boundary", [0, True, 127, 1409])
def test_invalid_copy_boundary_does_not_allocate_or_leak_references(boundary):
    cache = setup_cache()
    hashes, pages, proof = checkpoint(cache)
    before = [
        (p.spec.name, tuple((b.block_id, b.ref_cnt, b.allocation_epoch) for b in p.blocks))
        for p in cache._group_pools.values()
    ]
    with pytest.raises(ValueError, match="aligned"):
        cache.reserve_prefix_checkpoint("cold", generation=1, boundary=boundary, snapshot=pages)
    after = [
        (p.spec.name, tuple((b.block_id, b.ref_cnt, b.allocation_epoch) for b in p.blocks))
        for p in cache._group_pools.values()
    ]
    assert after == before
    assert not getattr(cache, "_prefix_checkpoint_leases", {})
