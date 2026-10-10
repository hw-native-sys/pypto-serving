# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Producer buffer handoff contracts, independent of device execution."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from pypto_serving.model.deepseek_v41.execution_plan import LayerPlan, plan_layers
from pypto_serving.model.deepseek_v41.prefill_segment import (
    PREFILL_ARGUMENTS, PrefillSegment, bind_prefill_producers,
)
from pypto_serving.model.deepseek_v41.swa_segment import MOE_ARGS, SegmentTopology


def plans_and_buffers():
    path = Path(__file__).resolve().parents[4] / "tests/fixtures/deepseek_v41/config.json"
    plans = plan_layers(json.loads(path.read_text()))
    buffers = {p.layer_id: {name: object() for name in PREFILL_ARGUMENTS[p.mode]} for p in plans}
    # Model loader deliberately omits weights owned by another layer.
    for p in plans:
        if p.mode in ("c1a_reindex", "c1a_reuse"):
            for name in ("compressor_wkv", "compressor_norm_weight", "index_wk", "index_norm_weight"):
                del buffers[p.layer_id][name]
        if p.mode == "c1a_reuse":
            for name in ("index_wq_b", "index_wq_b_scale", "index_weights_proj"):
                del buffers[p.layer_id][name]
    return plans, buffers


def test_actual_40_layer_plan_shares_producers_without_copying_or_mutating_inputs():
    plans, original = plans_and_buffers()
    bound = bind_prefill_producers(plans, original)
    for p in plans:
        current = bound[p.layer_id]
        assert current["window_cache"] is original[p.layer_id]["window_cache"]
        if p.mode == "swa":
            continue
        assert current["compressed_cache"] is original[p.kv_source]["compressed_cache"]
        if p.mode.endswith("reuse"):
            assert current["compressed_indices"] is original[p.index_source]["topk_indices"]
        if p.mode.startswith("c1a"):
            assert current["topk_indices"] is original[p.layer_id]["topk_indices"]
            assert current["topk_indices"] is not current["compressed_indices"]
            if not p.mode.endswith("reuse"):
                assert current["compressed_indices"] is original[p.layer_id]["compressed_indices"]
            assert current["candidate_mask"] is original[20]["candidate_mask"]
            assert current["index_cache"] is original[20]["index_cache"]
            assert current["compressor_wkv"] is original[20]["compressor_wkv"]
    assert bound[24]["topk_indices"] is original[24]["topk_indices"]
    assert bound[25]["compressed_indices"] is original[24]["topk_indices"]
    assert "compressor_wkv" not in original[24]
    assert original[21]["compressed_indices"] is not bound[21]["compressed_indices"]


@pytest.mark.parametrize("layer", [20, 24])
def test_unused_c1a_input_cannot_alias_topk_write_argument(layer):
    plans, buffers = plans_and_buffers()
    buffers[layer]["compressed_indices"] = buffers[layer]["topk_indices"]
    with pytest.raises(ValueError, match="separate allocations"):
        bind_prefill_producers(plans, buffers)


def test_unused_reuse_output_cannot_alias_its_producer_selection():
    plans, buffers = plans_and_buffers()
    buffers[25]["topk_indices"] = buffers[24]["topk_indices"]
    with pytest.raises(ValueError, match="separate allocations"):
        bind_prefill_producers(plans, buffers)


@pytest.mark.parametrize("plans", [
    (LayerPlan(3, "c2a_reuse", 2, 2, None),),
    (LayerPlan(20, "c1a_full", 20, 20, 20), LayerPlan(22, "c1a_reuse", 20, 20, 20)),
    (LayerPlan(20, "c1a_full", 20, 20, 20), LayerPlan(21, "c1a_reuse", 20, 21, 20)),
    (LayerPlan(2, "c2a_full", 2, 2, None), LayerPlan(3, "c1a_reuse", 2, 2, 2)),
    (LayerPlan(0, "swa", 0, None, None),),
])
def test_incomplete_or_invalid_chain_cannot_consume_stale_selection(plans):
    _, buffers = plans_and_buffers()
    with pytest.raises(ValueError):
        bind_prefill_producers(plans, buffers)


def test_selection_from_different_compressed_pool_is_rejected():
    plans, buffers = plans_and_buffers()
    changed = list(plans)
    changed[9] = LayerPlan(9, "c2a_reuse", 8, 2, None)
    with pytest.raises(ValueError, match="same KV"):
        bind_prefill_producers(changed, buffers)


def test_run_chain_hands_state_forward_and_resolves_late_dependencies_before_dispatch():
    plans, buffers = plans_and_buffers()
    plans = plans[2:4]
    segment = PrefillSegment.__new__(PrefillSegment)
    segment.topology = SegmentTopology(tp=2, dp=2)
    segment.attention_programs = {p.mode: object() for p in plans}
    segment._check_weights = Mock()
    first, second, final = object(), object(), object()
    segment.run_layer = Mock(side_effect=[second, final])
    moe = {p.layer_id: dict.fromkeys(MOE_ARGS, object()) for p in plans}
    late_weight = moe[3].pop("routed_w3")
    with pytest.raises(KeyError, match="routed_w3"):
        segment.run_chain(first, plans, buffers, moe, group_counts=[17, 0])
    segment.run_layer.assert_not_called()
    moe[3]["routed_w3"] = late_weight
    assert segment.run_chain(first, plans, buffers, moe, group_counts=[17, 0]) is final
    calls = segment.run_layer.call_args_list
    assert calls[0].args[0] is first and calls[1].args[0] is second
    assert calls[1].args[1]["compressed_indices"] is buffers[2]["topk_indices"]
    assert all(c.kwargs["group_counts"] == [17, 0] for c in calls)


def test_runtime_failure_prevents_later_consumer_dispatch():
    plans, buffers = plans_and_buffers()
    plans = plans[2:4]
    segment = PrefillSegment.__new__(PrefillSegment)
    segment.topology = SegmentTopology(tp=2, dp=2)
    segment.attention_programs = {p.mode: object() for p in plans}
    segment._check_weights = Mock()
    segment.run_layer = Mock(side_effect=RuntimeError("producer failed"))
    moe = {p.layer_id: dict.fromkeys(MOE_ARGS, object()) for p in plans}
    with pytest.raises(RuntimeError, match="producer failed"):
        segment.run_chain(object(), plans, buffers, moe, group_counts=[17, 0])
    assert segment.run_layer.call_count == 1
