# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the six AIV fields against independently packed checkpoint experts."""

import torch

from pypto_serving.model.common.moe_offload import ROUTED_FIELDS
from pypto_serving.model.deepseek_v41.expert_offload import routed_expert_backing
from pypto_serving.model.deepseek_v41.weight_loader import (
    DecodeWeightBank, DecodeWeightLayout, merge_expert_scales, stack_bytes,
)
from pypto_serving.model.deepseek_v41.weight_spec import pack_fp4_tiles, pack_mx_scale


def test_rank_local_mapped_weights_preserve_each_experts_fp4_and_mx_bytes():
    layers, ranks, experts = 2, 2, 3
    bank = DecodeWeightBank(DecodeWeightLayout(layers, 1, 1, 1, world=ranks))
    generator = torch.Generator().manual_seed(17)
    packed = {}
    for name in ROUTED_FIELDS:
        shape = (8, 256) if name.endswith("_scale") else (256, 128)
        pack = pack_mx_scale if name.endswith("_scale") else pack_fp4_tiles
        rows = []
        for layer in range(layers):
            local = []
            for rank in range(ranks):
                all_experts = []
                for expert in range(experts):
                    value = pack(torch.randint(256, shape, dtype=torch.uint8, generator=generator))
                    if name.endswith("_scale"):
                        value = value.view(torch.float8_e8m0fnu)
                    all_experts.append(value)
                    packed[layer, rank, expert, name] = value
                local.append(merge_expert_scales(all_experts) if name.endswith("_scale")
                             else stack_bytes(all_experts))
            rows.append(stack_bytes(local))
        bank.tensors[name] = stack_bytes(rows)

    expert_bytes = sum(packed[0, 0, 0, name].nbytes for name in ROUTED_FIELDS)
    backing = routed_expert_backing(bank, experts * expert_bytes)
    assert backing.capacity == experts
    for rank, fields in enumerate(backing.by_rank):
        image = bytearray(layers * experts * expert_bytes)
        backing.initialize(rank, memoryview(image))
        desc = backing.descriptors([0] * ranks)[rank]
        for layer in range(layers):
            for expert in range(experts):
                for i, name in enumerate(fields):
                    base, stride, size, _ = desc[layer, i].tolist()
                    start = base + expert * stride
                    actual = torch.frombuffer(image, dtype=torch.uint8, count=size, offset=start)
                    assert torch.equal(actual, packed[layer, rank, expert, name].view(torch.uint8).flatten())
