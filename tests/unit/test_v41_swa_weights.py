# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Byte-level checks for multi-expert scale and LUT assembly."""
import torch

from pypto_serving.model.deepseek_v41.swa_weights import merge_expert_scales, mxfp4_pair_lut
from pypto_serving.model.deepseek_v41.weight_packing import pack_mx_scale


def test_multi_expert_scale_layout():
    logical = [((torch.arange(4 * 32).reshape(4, 32) + e * 17) % 254).to(torch.uint8)
               for e in range(3)]
    individual = [pack_mx_scale(s) for s in logical]
    actual = merge_expert_scales(individual)
    # Independent physical-address formula: output block, expert/group-pair,
    # output lane, adjacent group parity. This catches a simple concatenation.
    for expert in range(3):
        for group in range(4):
            for col in range(32):
                offset = (((col // 16) * 6 + expert * 2 + group // 2) * 16 + col % 16) * 2 + group % 2
                assert actual.flatten()[offset] == logical[expert][group, col]
    assert not torch.equal(actual, torch.cat(individual))


def test_pair_lut_matches_fp4_values():
    table = mxfp4_pair_lut().view(torch.uint8).view(torch.float8_e4m3fn).float()
    values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    expected = torch.stack([values[torch.arange(256) % 16], values[torch.arange(256) // 16]], dim=1)
    for lane in range(2):
        torch.testing.assert_close(table[lane].reshape(256, 2), expected, rtol=0, atol=0)
