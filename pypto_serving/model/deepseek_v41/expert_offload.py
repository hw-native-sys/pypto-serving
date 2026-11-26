# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mapped FP4 expert slot geometry and per-rank MX weight bindings."""

import torch

from pypto_serving.model.common.moe_offload import ExpertBacking, ROUTED_FIELDS, expert_slot_capacity


def v41_expert_slots(budget, text, *, world, layers):
    local_experts = text["n_routed_experts"] // world
    matrix_elements = text["hidden_size"] * text["moe_intermediate_size"]
    expert_bytes = 3 * (matrix_elements // 2 + matrix_elements // 32)
    expert_slots = expert_slot_capacity(
        budget, expert_bytes, layers, local_experts)
    if expert_slots < local_experts:
        raise ValueError(f"V4.1 grouped prefill needs {local_experts} expert slots per rank "
                         f"({local_experts * expert_bytes} bytes)")
    return expert_slots


def routed_expert_backing(bank, budget_bytes):
    """Expose the checkpoint's packed FP4 panels and one independent MX_B_NN per expert."""
    experts = bank.tensors["routed_w1"].shape[2]
    by_rank = []
    for rank in range(bank.layout.world):
        fields = {}
        for name in ROUTED_FIELDS:
            source = bank.tensors[name][:, rank]
            if name.endswith("_scale"):
                layers, all_groups, columns = source.shape
                groups = all_groups // experts
                # Resident scales interleave experts by output block. Each cache slot
                # needs that same packed MX layout with an expert axis on the outside.
                fields[name] = (source.view(torch.uint8)
                                .reshape(layers, columns // 16, experts, groups * 16)
                                .transpose(1, 2).contiguous()
                                .reshape(layers, experts, groups, columns).view(source.dtype))
            else:
                fields[name] = source
        by_rank.append(fields)
    return ExpertBacking(by_rank, budget_bytes)


def mapped_weight_arguments(pool):
    from pypto.runtime import StackedDeviceTensor

    bindings = pool.arguments()
    for name in ROUTED_FIELDS:
        if name.endswith("_scale"):
            tensor = bindings[name]
            world, slots, groups, width = tensor.full_shape
            shape = (slots * groups, width)
            shards = [pool.worker.view_tensor(shard, shape, worker_id=rank)
                      for rank, shard in enumerate(tensor.shards)]
            bindings[name] = StackedDeviceTensor(shards, (world, *shape), tensor.worker_ids)
    return bindings
