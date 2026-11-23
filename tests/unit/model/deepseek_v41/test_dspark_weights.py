# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for V4.1 DSpark checkpoint weight layouts."""

from types import SimpleNamespace

import torch

from pypto_serving.model.deepseek_v41.dspark_weights import V41DSparkWeightContract


def test_main_projection_payload_is_packed_in_nz_order():
    hidden = 64
    output = torch.arange(hidden)[:, None]
    inputs = torch.arange(3 * hidden)[None, :]
    payload = ((7 * output + 19 * inputs) % 126).to(torch.uint8)
    checkpoint = {
        "mtp.0.main_proj.weight": payload.view(torch.float8_e4m3fn),
        "mtp.0.main_proj.scale": torch.full((2, 6), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        "mtp.0.main_norm.weight": torch.ones(hidden),
        "mtp.2.norm.weight": torch.ones(hidden),
        "mtp.2.markov_head.embed.weight": torch.zeros(8, 4),
        "mtp.2.markov_head.head.weight": torch.zeros(8, 4),
        "mtp.2.confidence_head.proj.weight": torch.zeros(1, hidden + 4),
    }
    loader = SimpleNamespace(
        text={"hidden_size": hidden, "n_routed_experts": 1, "vocab_size": 8},
        store=SimpleNamespace(load_many=lambda names: {name: checkpoint[name] for name in names}),
    )

    weights = V41DSparkWeightContract(loader).load_heads().tensors

    packed = weights["main_proj_weight"]
    # Each consecutive 32-output block stores all input rows in order.
    expected = torch.cat([block.T.contiguous().flatten() for block in payload.split(32)])
    assert packed.shape == (3 * hidden, hidden)
    assert packed.dtype == torch.float8_e4m3fn
    assert packed.is_contiguous()
    assert torch.equal(packed.view(torch.uint8).flatten(), expected)
    scale = weights["main_proj_scale"]
    assert scale.dtype == torch.float8_e8m0fnu
    assert torch.equal(scale.view(torch.uint8), torch.full((6, hidden), 127, dtype=torch.uint8))
