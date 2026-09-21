# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from pypto_serving.model.deepseek_dspark.pd_adapter import DSV4_DSPARK_K7_ADAPTER
from pypto_serving.serving.pd.protocol import HandoffKey

from .helpers import make_cache_manager, make_registry


KEY = HandoffKey("request", "handoff", 1, 1, 1)


def _tables(manager, request_id: str, token_count: int, partition: int = 0):
    return {
        name: tuple(ids)
        for name, ids in manager.ensure_group_blocks(
            request_id,
            token_count,
            partition=partition,
        ).items()
    }


def test_chunk_planner_defers_partial_pages_and_keeps_zero_units() -> None:
    manager = make_cache_manager()
    registry = make_registry(manager)
    source = _tables(manager, "source", 33)
    destination = _tables(manager, "destination", 64)
    planner = DSV4_DSPARK_K7_ADAPTER.make_planner(registry, manager.group_specs)
    rank_ids = (0, 1, 2, 3)
    source_by_rank = {rank: source for rank in rank_ids}
    destination_by_rank = {rank: destination for rank in rank_ids}

    first = planner.plan_chunk(
        KEY,
        chunk_id=0,
        start_token=0,
        end_token=32,
        final=False,
        rank_ids=rank_ids,
        source_blocks_by_rank=source_by_rank,
        destination_blocks_by_rank=destination_by_rank,
    )
    first_copies = first.ranks[0].copies
    assert {copy.component_id for copy in first_copies} == {"ori"}
    assert {unit.component_id for unit in first.ranks[0].expected_units} == {
        "ori", "hca_cmp", "csa_cmp", "idx_k", "idx_scale"
    }
    assert any(unit.nbytes == 0 for unit in first.ranks[0].expected_units)

    final = planner.plan_chunk(
        KEY,
        chunk_id=1,
        start_token=32,
        end_token=33,
        final=True,
        rank_ids=rank_ids,
        source_blocks_by_rank=source_by_rank,
        destination_blocks_by_rank=destination_by_rank,
    )
    final_components = {copy.component_id for copy in final.ranks[0].copies}
    assert "ori" in final_components
    assert {"hca_state", "csa_state", "csa_inner_state"} <= final_components
    partial_ori = [copy for copy in final.ranks[0].copies if copy.component_id == "ori"]
    assert partial_ori and {copy.valid_tokens for copy in partial_ori} == {1}


def test_index_regions_are_atomic_and_ring_destination_wraps() -> None:
    manager = make_cache_manager()
    registry = make_registry(manager)
    ori_spec = next(spec for spec in manager.group_specs if spec.name == "ori")
    wrap_start = ori_spec.max_blocks_per_seq * ori_spec.spec.token_capacity
    wrap_end = wrap_start + ori_spec.spec.token_capacity
    source = _tables(manager, "source", wrap_end)
    destination = _tables(manager, "destination", wrap_end)
    planner = DSV4_DSPARK_K7_ADAPTER.make_planner(registry, manager.group_specs)
    ring_plan = planner.plan_chunk(
        KEY,
        chunk_id=ori_spec.max_blocks_per_seq,
        start_token=wrap_start,
        end_token=wrap_end,
        final=False,
        rank_ids=(0,),
        source_blocks_by_rank={0: source},
        destination_blocks_by_rank={0: destination},
    )
    ori = [
        copy
        for copy in ring_plan.ranks[0].copies
        if copy.component_id == "ori"
    ]
    assert ori
    assert {copy.destination_block for copy in ori} == {destination["ori"][0]}

    index_plan = planner.plan_chunk(
        KEY,
        chunk_id=0,
        start_token=0,
        end_token=128,
        final=False,
        rank_ids=(0,),
        source_blocks_by_rank={0: source},
        destination_blocks_by_rank={0: destination},
    )
    copies = index_plan.ranks[0].copies
    index_k = {
        (copy.layer, copy.source_block, copy.destination_block, copy.valid_tokens)
        for copy in copies
        if copy.component_id == "idx_k"
    }
    index_scale = {
        (copy.layer, copy.source_block, copy.destination_block, copy.valid_tokens)
        for copy in copies
        if copy.component_id == "idx_scale"
    }
    assert index_k
    assert index_k == index_scale


@pytest.mark.parametrize(
    ("p_hit", "d_hit", "expected_ori_start", "expected_full_start"),
    (
        (128, 512, 512, 512),
        (256, 256, 256, 256),
        (512, 128, 384, 128),
    ),
)
def test_pc2_plans_independent_hits_without_reading_stale_rolling_slots(
    p_hit: int,
    d_hit: int,
    expected_ori_start: int,
    expected_full_start: int,
) -> None:
    manager = make_cache_manager()
    registry = make_registry(manager)
    prompt_tokens = 640
    source = _tables(manager, "pc2-source", prompt_tokens)
    destination = _tables(manager, "pc2-destination", prompt_tokens)
    planner = DSV4_DSPARK_K7_ADAPTER.make_planner(registry, manager.group_specs)

    plan = planner.plan_chunk(
        KEY,
        chunk_id=0,
        start_token=0,
        end_token=prompt_tokens,
        final=True,
        rank_ids=(0,),
        source_blocks_by_rank={0: source},
        destination_blocks_by_rank={0: destination},
        destination_prefix_hit_tokens=d_hit,
        source_prefix_hit_tokens=p_hit,
    )

    ori_slots = {block_id: index for index, block_id in enumerate(destination["ori"])}
    hca_slots = {
        block_id: index for index, block_id in enumerate(destination["cmp_c128"])
    }
    ori_indices = {
        ori_slots[copy.destination_block]
        for copy in plan.ranks[0].copies
        if copy.component_id == "ori" and copy.layer == 0
    }
    hca_layer = next(
        copy.layer
        for copy in plan.ranks[0].copies
        if copy.component_id == "hca_cmp"
    )
    hca_indices = {
        hca_slots[copy.destination_block]
        for copy in plan.ranks[0].copies
        if copy.component_id == "hca_cmp" and copy.layer == hca_layer
    }

    assert plan.source_prefix_hit_tokens == p_hit
    assert ori_indices == set(range(expected_ori_start // 32, prompt_tokens // 32))
    assert hca_indices == set(
        range(expected_full_start // 128, prompt_tokens // 128)
    )


def test_pc2_rejects_unaligned_p_hit() -> None:
    manager = make_cache_manager()
    registry = make_registry(manager)
    tables = _tables(manager, "unaligned", 256)
    planner = DSV4_DSPARK_K7_ADAPTER.make_planner(registry, manager.group_specs)

    with pytest.raises(ValueError, match="common cache alignment"):
        planner.plan_chunk(
            KEY,
            chunk_id=0,
            start_token=0,
            end_token=256,
            final=True,
            rank_ids=(0,),
            source_blocks_by_rank={0: tables},
            destination_blocks_by_rank={0: tables},
            source_prefix_hit_tokens=127,
        )
