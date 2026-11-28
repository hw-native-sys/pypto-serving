# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 packed-FP4 expert tile reordering stays bit-exact to the reference loop."""

import pytest
import torch

from pypto_serving.model.deepseek_v41.weight_spec import pack_fp4_tiles


def _reference_pack(payload, tile=256):
    """Per-tile reference: the nested-loop implementation this module replaced."""
    source = payload.contiguous().view(torch.uint8)
    n, half_k = source.shape
    k = half_k * 2
    output = torch.empty((n // tile, k // tile, tile, tile // 2), dtype=torch.uint8)
    row = tile // 2
    for nb in range(n // tile):
        for kb in range(k // tile):
            block = source[nb * tile:(nb + 1) * tile, kb * row:(kb + 1) * row]
            codes = torch.empty((tile, tile), dtype=torch.uint8)
            codes[:, 0::2] = block & 15
            codes[:, 1::2] = block >> 4
            transposed = codes.T
            output[nb, kb] = transposed[:, 0::2] | (transposed[:, 1::2] << 4)
    return output.reshape(k * n // tile, tile // 2)


@pytest.mark.parametrize("shape", [(256, 128), (512, 256), (768, 384), (2304, 2560)])
def test_pack_fp4_tiles_matches_reference(shape):
    torch.manual_seed(shape[0] * 131 + shape[1])
    payload = torch.randint(0, 256, shape, dtype=torch.uint8)
    got = pack_fp4_tiles(payload)
    assert torch.equal(got, _reference_pack(payload))
    assert got.shape == (shape[0] * (shape[1] // 128), 128)


@pytest.mark.parametrize("shape", [(250, 128), (256, 120)])
def test_pack_fp4_tiles_rejects_partial_tiles(shape):
    payload = torch.zeros(shape, dtype=torch.uint8)
    with pytest.raises(ValueError, match="complete 256x256 tiles"):
        pack_fp4_tiles(payload)
