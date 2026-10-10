# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real tensor checks for fresh HC state and request-to-rank mapping."""
import pytest
import torch

from pypto_serving.model.deepseek_v41.request_state import ForwardStep, RequestSlice
from pypto_serving.model.deepseek_v41.segment_inputs import prepare_segment_inputs
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


def request(name, partition, count):
    return RequestSlice(name, partition, 0, 3, tuple(range(count)), count + 3, {})


def test_interleaved_requests_keep_tp_slabs_and_output_order():
    topology = SegmentTopology(tp=2, dp=2, local_capacity=2)
    step = ForwardStep("prefill", (request("b", 1, 3), request("a", 0, 1), request("c", 1, 1)), 1)
    embeddings = torch.arange(5 * 8).reshape(5, 8).bfloat16()
    result = prepare_segment_inputs(embeddings, step, topology)
    assert result.group_counts == (1, 4)
    assert result.row_indices.tolist() == [[3, -1], [-1, -1], [0, 1], [2, 4]]
    assert result.request_last_rows == ((3, 0), (0, 0), (3, 1))
    for rank, row in (result.row_indices >= 0).nonzero().tolist():
        source = result.row_indices[rank, row]
        expected = embeddings[source].float()
        assert torch.equal(result.residual[rank, row], expected.expand(4, -1))
        assert result.pre_mix[rank, row].tolist() == [1, 0, 0, 0]
    padding = result.row_indices < 0
    assert torch.count_nonzero(result.residual[padding]) == 0
    assert torch.count_nonzero(result.pre_mix[padding]) == 0
    embeddings.zero_()
    assert result.residual.abs().sum() > 0  # Own storage, safe until upload completes.


def test_decode_starts_fresh_token_hc_and_empty_partition_stays_zero():
    topology = SegmentTopology(tp=2, dp=2, local_capacity=2)
    step = ForwardStep("decode", (request("a", 1, 1), request("b", 1, 1)), 8)
    inputs = prepare_segment_inputs(torch.ones(2, 8).bfloat16(), step, topology)
    assert inputs.group_counts == (0, 2)
    assert inputs.request_last_rows == ((2, 0), (2, 1))
    assert torch.count_nonzero(inputs.residual[:2]) == 0
    assert torch.equal(inputs.residual[2], torch.ones(2, 4, 8))


@pytest.mark.parametrize("requests,phase,rows,match", [
    ((request("a", 0, 5),), "prefill", 5, "capacity"),
    ((request("a", 2, 1),), "prefill", 1, "partitions"),
    ((request("a", 0, 2),), "decode", 2, "one fresh token"),
    ((request("a", 0, 1), request("a", 1, 1)), "prefill", 2, "duplicate"),
    ((request("a", 0, 1),), "prefill", 2, "embedding rows"),
])
def test_invalid_mapping_is_rejected(requests, phase, rows, match):
    with pytest.raises(ValueError, match=match):
        prepare_segment_inputs(torch.zeros(rows, 8).bfloat16(), ForwardStep(phase, requests, 1),
                               SegmentTopology(tp=2, dp=2, local_capacity=2))


def test_allocation_budget_rejects_before_output_allocation(monkeypatch):
    embeddings = torch.ones(1, 8).bfloat16()
    def forbidden(*args, **kwargs):
        raise AssertionError("must not allocate output before budget check")
    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises(ValueError, match="budget"):
        prepare_segment_inputs(embeddings, ForwardStep("prefill", (request("a", 0, 1),), 1),
                               SegmentTopology(tp=2, dp=2, local_capacity=2), max_prepare_bytes=1)
