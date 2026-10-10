# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Causal cache addressing across chunks, physical pages and DP partitions."""
import pytest
import torch

from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
from pypto_serving.model.deepseek_v41.segment_inputs import prepare_segment_inputs
from pypto_serving.model.deepseek_v41.swa_metadata import gather_swa_rope_rows, prepare_swa_window_metadata
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


TOPOLOGY = SegmentTopology(tp=2, dp=2, local_capacity=2)


def request(name="a", partition=0, start=126, count=4, pages=(5, 2)):
    return RequestSlice(name, partition, 0, start, tuple(range(count)), start + count, {"window": pages})


def test_page_crossing_preserves_early_chunk_history_and_causality():
    step = ForwardStep("prefill", (request(),), 1)
    metadata = prepare_swa_window_metadata(step, TOPOLOGY, cache_pages=8)
    assert metadata.group_counts == (4, 0)
    assert metadata.positions[0].tolist() == [126, 127, 128, 129]
    assert metadata.window_slots[0].tolist() == [766, 767, 256, 257]
    assert metadata.window_indices[0, 0, :127].tolist() == list(range(640, 767))
    assert metadata.window_indices[0, 0, 127] == -1
    assert metadata.window_indices[0, 2].tolist() == list(range(641, 768)) + [256]
    assert metadata.window_indices[0, 3].tolist() == list(range(642, 768)) + [256, 257]
    assert torch.equal(metadata.window_indices[0], metadata.window_indices[1])
    assert (metadata.window_slots[2:] == -1).all()
    assert (metadata.window_indices[2:] == -1).all()
    continuation = ForwardStep("decode", (request(start=130, count=1),), 2)
    decoded = prepare_swa_window_metadata(continuation, TOPOLOGY, cache_pages=8)
    assert decoded.window_indices[0, 0].tolist() == list(range(643, 768)) + [256, 257, 258]
    assert decoded.window_slots[0, 0] == 258


def test_interleaved_dp_rows_match_hc_input_packing():
    requests = (request("b", 1, 0, 2, (3,)), request("a", 0, 2, 1, (3,)),
                request("c", 1, 5, 1, (4,)))
    step = ForwardStep("prefill", requests, 1)
    inputs = prepare_segment_inputs(torch.ones(4, 8).bfloat16(), step, TOPOLOGY)
    metadata = prepare_swa_window_metadata(step, TOPOLOGY, cache_pages=8)
    assert metadata.group_counts == inputs.group_counts == (1, 3)
    source_positions = torch.tensor(step.positions)
    for rank, local in (inputs.row_indices >= 0).nonzero().tolist():
        group_row = rank % TOPOLOGY.tp * TOPOLOGY.local_capacity + local
        assert metadata.positions[rank, group_row] == source_positions[inputs.row_indices[rank, local]]
    assert metadata.window_slots[2].tolist() == [384, 385, 517, -1]


@pytest.mark.parametrize("requests,match", [
    ((request(pages=(5,)),), "full-history"),
    ((request(pages=(5, 8)),), "outside"),
    ((request(pages=(5, 5)),), "private"),
    ((request("a", 0, 0, 1, (3,)), request("b", 0, 0, 1, (3,))), "private"),
    ((request(start=-1),), "start"),
    ((request(count=5),), "capacity"),
])
def test_invalid_pages_and_extents_fail_before_dispatch(requests, match):
    with pytest.raises(ValueError, match=match):
        prepare_swa_window_metadata(ForwardStep("prefill", requests, 1), TOPOLOGY, cache_pages=8)


def test_small_budget_is_rejected():
    with pytest.raises(ValueError, match="budget"):
        prepare_swa_window_metadata(ForwardStep("prefill", (request(),), 1), TOPOLOGY,
                                    cache_pages=8, max_prepare_bytes=1)


def test_rope_rows_use_absolute_positions_and_identity_padding():
    step = ForwardStep("prefill", (request("b", 1, 126, 2), request("a", 0, 3, 1, (2,))), 1)
    metadata = prepare_swa_window_metadata(step, TOPOLOGY, cache_pages=8)
    angles = torch.arange(130, dtype=torch.float32)[:, None] * torch.tensor([[1., .1]])
    cos, sin = gather_swa_rope_rows(metadata, (angles.cos(), angles.sin()))
    for rank, positions in enumerate(([3], [3], [126, 127], [126, 127])):
        torch.testing.assert_close(cos[rank, :len(positions)], angles[positions].cos(), rtol=0, atol=0)
        torch.testing.assert_close(sin[rank, :len(positions)], angles[positions].sin(), rtol=0, atol=0)
        assert (cos[rank, len(positions):] == 1).all()
        assert (sin[rank, len(positions):] == 0).all()


@pytest.mark.parametrize("case,match", [("short", "outside"), ("nan", "finite"),
                                       ("dtype", "FP32"), ("budget", "budget")])
def test_rope_rejects_invalid_active_rows(case, match):
    metadata = prepare_swa_window_metadata(ForwardStep("prefill", (request(),), 1), TOPOLOGY, cache_pages=8)
    cos, sin = torch.ones(130, 2), torch.zeros(130, 2)
    if case == "short":
        cos, sin = cos[:129], sin[:129]
    if case == "nan":
        sin[129, 0] = float("nan")
    if case == "dtype":
        cos = cos.bfloat16()
    with pytest.raises(ValueError, match=match):
        gather_swa_rope_rows(metadata, (cos, sin), max_prepare_bytes=1 if case == "budget" else 16 << 20)
