# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host Engram staging preserves TP copies, row slots and zero padding."""
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41.npu_runner import HostEngramResources, SegmentTopology


@pytest.mark.parametrize("row_stride", [None, 6])
def test_stage_uses_active_capacity_and_clears_reused_slots(row_stride):
    calls = []

    def lookup(token_ids, capacity, start_pos, *, request_id):
        calls.append((len(token_ids), capacity, request_id))
        return torch.full((2, capacity, 8), token_ids[0] if token_ids else 0, dtype=torch.bfloat16)

    resources = object.__new__(HostEngramResources)
    resources.lookup = torch.zeros((4, 2, 160, 8), dtype=torch.bfloat16)
    resources.host = SimpleNamespace(lookup=lookup)
    topology = SegmentTopology(tp=2, dp=2, local_capacity=80)
    target = torch.full_like(resources.lookup, -1)
    requests = [SimpleNamespace(token_ids=[3, 3, 3], partition=1, start=0, request_id='a'),
                SimpleNamespace(token_ids=[7], partition=1, start=3, request_id='b')]
    step = SimpleNamespace(requests=requests)
    resources.stage_lookup(target, step, topology, row_stride=row_stride)
    expected = torch.zeros_like(target)
    expected[2:, :, :3] = 3
    offset = 3 if row_stride is None else row_stride
    expected[2:, :, offset:offset+1] = 7
    assert torch.equal(target, expected)
    assert calls == [(3, 3, 'a'), (1, 1, 'b')]
    step.requests = [SimpleNamespace(token_ids=[], partition=0, start=0, request_id='empty')]
    resources.stage_lookup(target, step, topology, row_stride=row_stride)
    assert not target.any()
    assert calls[-1] == (0, 1, 'empty')
    step.requests = [SimpleNamespace(token_ids=[1]*161, partition=0, start=0, request_id='large')]
    with pytest.raises(ValueError, match='capacity'):
        resources.stage_lookup(target, step, topology, row_stride=row_stride)
