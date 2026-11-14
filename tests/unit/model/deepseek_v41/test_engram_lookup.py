# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Engram dequantization preserves FP8 and E8M0 byte semantics."""
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41.engram import HostEngram


@pytest.mark.parametrize("count", [0, 1, 6, 64])
def test_lookup_matches_expanded_scale_reference(count):
    weight_bytes = torch.arange(256*64, dtype=torch.int64).to(torch.uint8).reshape(256, 64)
    scale_bytes = torch.arange(256*2, dtype=torch.int64).to(torch.uint8).reshape(256, 2)
    weights = weight_bytes.view(torch.float8_e4m3fn)
    scales = scale_bytes.view(torch.float8_e8m0fnu)
    hashes = torch.arange(count*4, dtype=torch.int64).reshape(count, 4)
    host = object.__new__(HostEngram)
    host.layout = SimpleNamespace(layer_ids=(0, 1), head_dim=64, lookup_width=256)
    host.hash_ids = lambda *args, **kwargs: torch.stack((hashes, hashes.flip(-1)))

    def tensor_rows(name, indices):
        return (weight_bytes if name.endswith('weight') else scale_bytes)[indices].view(
            torch.float8_e4m3fn if name.endswith('weight') else torch.float8_e8m0fnu)

    host.checkpoint = SimpleNamespace(tensor_rows=tensor_rows)
    expected = torch.zeros((2, count+3, 256), dtype=torch.bfloat16)
    for slot, indices in enumerate((hashes, hashes.flip(-1))):
        values = weights.view(torch.uint8)[indices].view(weights.dtype).float()
        factors = torch.exp2(scales.view(torch.uint8)[indices].float()-127.0).repeat_interleave(32, -1)
        expected[slot, :count] = (values*factors).flatten(1).to(torch.bfloat16)
    actual = host.lookup([0]*count, count+3)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
