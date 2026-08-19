# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeekV4 W8A8 as data: which checkpoint tensor feeds which kernel weight.

This is the declarative half of the hand-written pack table in ``weight_loader.py``. It
covers the layer weights that are present for every layer, whatever its attention kind;
the optional compressor/indexer branches, the router and the routed experts stay in code
for now because each needs a rule kind this schema does not express yet.

**Order is contract.** The sequence below is the order the packed mapping is built in, the
slab allocator lays out whole-model tensors in that order, and the prepacked sidecar stores
the resulting name-to-offset map. Reordering these entries would invalidate every sidecar
already written, so new entries go at the end of the group they belong to.
"""

import torch

from pypto_serving.model.common.weights.shard import Replicate
from pypto_serving.model.common.weights.spec import LayerWeightRule

# `wo_a` arrives flattened and is split into this many output groups.
DEEPSEEK_V4_O_GROUPS = 8

_MISMATCH_ERROR = "packed DeepSeekV4 destination {name} shape/dtype mismatch: expected={expected}, got={got}"

DEEPSEEK_V4_CORE_LAYER_RULES: tuple[LayerWeightRule, ...] = (
    LayerWeightRule("hc_attn_fn", "hc_attn_fn", torch.float32),
    LayerWeightRule("hc_attn_scale", "hc_attn_scale", torch.float32),
    LayerWeightRule("hc_attn_base", "hc_attn_base", torch.float32),
    LayerWeightRule("attn_norm_w", "attn_norm.weight", torch.bfloat16),
    LayerWeightRule("wq_a", "attn.wq_a.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("wq_b", "attn.wq_b.weight", torch.int8, transpose=True),
    LayerWeightRule("wq_b_scale", "attn.wq_b.scale", torch.float32),
    LayerWeightRule("wkv", "attn.wkv.weight", torch.bfloat16, transpose=True),
    LayerWeightRule("gamma_cq", "attn.q_norm.weight", torch.bfloat16),
    LayerWeightRule("gamma_ckv", "attn.kv_norm.weight", torch.bfloat16),
    LayerWeightRule("attn_sink", "attn.attn_sink", torch.float32),
    LayerWeightRule("wo_a", "attn.wo_a.weight", torch.bfloat16, reshape_groups=DEEPSEEK_V4_O_GROUPS),
    LayerWeightRule("wo_b", "attn.wo_b.weight", torch.int8),
    LayerWeightRule("wo_b_scale", "attn.wo_b.scale", torch.float32),
    LayerWeightRule("hc_ffn_fn", "hc_ffn_fn", torch.float32),
    LayerWeightRule("hc_ffn_scale", "hc_ffn_scale", torch.float32),
    LayerWeightRule("hc_ffn_base", "hc_ffn_base", torch.float32),
    LayerWeightRule("norm_w", "ffn_norm.weight", torch.bfloat16),
    LayerWeightRule("gate_w", "ffn.gate.weight", torch.float32),
    LayerWeightRule("shared_w1", "ffn.shared_experts.w1.weight", torch.int8),
    LayerWeightRule("shared_w1_scale", "ffn.shared_experts.w1.scale", torch.float32),
    LayerWeightRule("shared_w3", "ffn.shared_experts.w3.weight", torch.int8),
    LayerWeightRule("shared_w3_scale", "ffn.shared_experts.w3.scale", torch.float32),
    LayerWeightRule("shared_w2", "ffn.shared_experts.w2.weight", torch.int8),
    LayerWeightRule("shared_w2_scale", "ffn.shared_experts.w2.scale", torch.float32),
)


def deepseek_v4_replicate(ranks: int) -> Replicate:
    """The rank policy for DeepSeekV4, carrying the diagnostics its users recognise."""
    return Replicate(ranks=ranks, mismatch_error=_MISMATCH_ERROR)
