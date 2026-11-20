# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

import torch

from pypto_serving.model.common.weights.nz import unpack_nz
from pypto_serving.model.deepseek_v41.dspark_resources import _DRAFT_MODEL_KEYS
from pypto_serving.model.deepseek_v41.dspark_weights import V41DSparkWeightContract


def test_dspark_main_projection_matches_lib_mxfp8_abi():
    hidden = 64
    payload = torch.arange(3 * hidden * hidden).remainder(127).to(torch.uint8).reshape(hidden, 3 * hidden)
    codes = torch.arange(120, 132, dtype=torch.uint8).reshape(2, 6)
    checkpoint = {
        "mtp.0.main_proj.weight": payload.view(torch.float8_e4m3fn),
        "mtp.0.main_proj.scale": codes.view(torch.float8_e8m0fnu),
        "mtp.0.main_norm.weight": torch.empty(hidden),
        "mtp.2.norm.weight": torch.empty(hidden),
        "mtp.2.markov_head.embed.weight": torch.empty(2, 1),
        "mtp.2.markov_head.head.weight": torch.empty(2, 1),
        "mtp.2.confidence_head.proj.weight": torch.empty(1, hidden + 1),
    }
    store = SimpleNamespace(
        load_many=lambda names: {name: checkpoint[name] for name in names},
        require=lambda names: None,
    )
    heads = V41DSparkWeightContract(SimpleNamespace(store=store, text={"n_routed_experts": 1})).load_heads()
    weight, scale = (heads.tensors[name] for name in ("main_proj_weight", "main_proj_scale"))
    assert {"main_proj_weight", "main_proj_scale"} <= _DRAFT_MODEL_KEYS
    assert weight.dtype == torch.float8_e4m3fn and weight.shape == (3 * hidden, hidden)
    assert scale.dtype == torch.float8_e8m0fnu and scale.shape == (6, hidden)
    assert torch.equal(unpack_nz(weight).view(torch.uint8), payload.T)
    unpacked_scale = scale.view(torch.uint8).reshape(4, 3, 16, 2).permute(1, 3, 0, 2).reshape(6, hidden)
    assert torch.equal(unpacked_scale, codes.T.repeat_interleave(32, dim=1))
