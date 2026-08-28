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

from collections.abc import Sequence

import torch

from pypto_serving.model.common.weights.pipeline import StagingPolicy
from pypto_serving.model.common.weights.shard import ExpertParallel, Replicate, TensorParallel
from pypto_serving.model.common.weights.spec import (
    DefaultedWeightRule,
    ExpertWeightRule,
    GlobalWeightRule,
    LayerRule,
    LayerWeightRule,
    OptionalWeightRule,
    SyntheticWeightRule,
)
from pypto_serving.model.common.weights.stacker import StackGroup

# `wo_a` arrives flattened and is split into this many output groups.
DEEPSEEK_V4_O_GROUPS = 8

# Attention kinds, and the fixed dimensions their compressor/indexer weights have. These are
# model constants rather than config knobs: the packer validates the active branch against
# them and zero-fills the inactive branch at the same sizes, so every layer presents one
# kernel signature.
DEEPSEEK_V4_CSA_RATIO = 4
DEEPSEEK_V4_HCA_RATIO = 128
_HIDDEN = 4096
_HEAD_DIM = 512
_HCA_OUT = 512
_CSA_OUT = 1024
_CSA_INNER_OUT = 256
_Q_LORA = 1024
_ATTENTION_OUT = 64 * 512
_HADAMARD_DIM = 128
_VOCAB_SIZE = 129280
_TOPK = 6

_MISMATCH_ERROR = "packed DeepSeekV4 destination {name} shape/dtype mismatch: expected={expected}, got={got}"
# The wording the loader's users already recognise, kept identical across the rewire.
DEEPSEEK_V4_SOURCE_MISSING_ERROR = "missing raw DeepSeekV4 layer tensor: {name}"
DEEPSEEK_V4_EXPERT_MISSING_ERROR = "missing raw DeepSeekV4 expert tensor: {name}"

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


def deepseek_v4_o_projection_tp_policies(
    ranks: int,
    tp_size: int,
) -> dict[str, TensorParallel]:
    """Return the TP4 O-projection layout used by the DSpark kernels.

    ``wo_a`` is reshaped to ``[o_groups, o_lora, group_input]`` before the
    policy runs, so it shards on group axis 0. ``wo_b`` retains checkpoint
    shape ``[hidden, attention_out]`` and shards its output columns on axis 1.
    """
    return {
        "wo_a": TensorParallel(
            ranks=ranks,
            tp_size=tp_size,
            axis=0,
            mismatch_error=_MISMATCH_ERROR,
        ),
        "wo_b": TensorParallel(
            ranks=ranks,
            tp_size=tp_size,
            axis=1,
            mismatch_error=_MISMATCH_ERROR,
        ),
    }


# Compressor and indexer weights: present for one attention kind, zeros for the others. Order
# is the hand-written table's order, including where the synthetic Hadamard index sits.
DEEPSEEK_V4_OPTIONAL_LAYER_RULES: tuple[LayerRule, ...] = (
    OptionalWeightRule(
        "hca_cmp_wkv", "attn.compressor.wkv.weight", torch.bfloat16, (_HCA_OUT, _HIDDEN), (DEEPSEEK_V4_HCA_RATIO,)
    ),
    OptionalWeightRule(
        "hca_cmp_wgate",
        "attn.compressor.wgate.weight",
        torch.bfloat16,
        (_HCA_OUT, _HIDDEN),
        (DEEPSEEK_V4_HCA_RATIO,),
    ),
    OptionalWeightRule(
        "hca_cmp_ape",
        "attn.compressor.ape",
        torch.float32,
        (DEEPSEEK_V4_HCA_RATIO, _HCA_OUT),
        (DEEPSEEK_V4_HCA_RATIO,),
    ),
    OptionalWeightRule(
        "hca_cmp_norm_w", "attn.compressor.norm.weight", torch.bfloat16, (_HEAD_DIM,), (DEEPSEEK_V4_HCA_RATIO,)
    ),
    OptionalWeightRule(
        "csa_cmp_wkv", "attn.compressor.wkv.weight", torch.bfloat16, (_CSA_OUT, _HIDDEN), (DEEPSEEK_V4_CSA_RATIO,)
    ),
    OptionalWeightRule(
        "csa_cmp_wgate",
        "attn.compressor.wgate.weight",
        torch.bfloat16,
        (_CSA_OUT, _HIDDEN),
        (DEEPSEEK_V4_CSA_RATIO,),
    ),
    OptionalWeightRule(
        "csa_cmp_ape",
        "attn.compressor.ape",
        torch.float32,
        (DEEPSEEK_V4_CSA_RATIO, _CSA_OUT),
        (DEEPSEEK_V4_CSA_RATIO,),
    ),
    OptionalWeightRule(
        "csa_cmp_norm_w", "attn.compressor.norm.weight", torch.bfloat16, (_HEAD_DIM,), (DEEPSEEK_V4_CSA_RATIO,)
    ),
    OptionalWeightRule(
        "csa_idx_wq_b",
        "attn.indexer.wq_b.weight",
        torch.int8,
        (_Q_LORA, _ATTENTION_OUT // 4),
        (DEEPSEEK_V4_CSA_RATIO,),
        transpose=True,
    ),
    OptionalWeightRule(
        "csa_idx_wq_b_scale",
        "attn.indexer.wq_b.scale",
        torch.float32,
        (_ATTENTION_OUT // 4,),
        (DEEPSEEK_V4_CSA_RATIO,),
    ),
    OptionalWeightRule(
        "csa_weights_proj",
        "attn.indexer.weights_proj.weight",
        torch.bfloat16,
        (_HIDDEN, 64),
        (DEEPSEEK_V4_CSA_RATIO,),
        transpose=True,
    ),
    SyntheticWeightRule("csa_hadamard_idx", torch.bfloat16, "hadamard_idx"),
    OptionalWeightRule(
        "csa_inner_wkv",
        "attn.indexer.compressor.wkv.weight",
        torch.bfloat16,
        (_CSA_INNER_OUT, _HIDDEN),
        (DEEPSEEK_V4_CSA_RATIO,),
    ),
    OptionalWeightRule(
        "csa_inner_wgate",
        "attn.indexer.compressor.wgate.weight",
        torch.bfloat16,
        (_CSA_INNER_OUT, _HIDDEN),
        (DEEPSEEK_V4_CSA_RATIO,),
    ),
    OptionalWeightRule(
        "csa_inner_ape",
        "attn.indexer.compressor.ape",
        torch.float32,
        (DEEPSEEK_V4_CSA_RATIO, _CSA_INNER_OUT),
        (DEEPSEEK_V4_CSA_RATIO,),
    ),
    OptionalWeightRule(
        "csa_inner_norm_w",
        "attn.indexer.compressor.norm.weight",
        torch.bfloat16,
        (_HADAMARD_DIM,),
        (DEEPSEEK_V4_CSA_RATIO,),
    ),
)

# Router weights: the checkpoint carries one or the other depending on the layer, and the
# unused mode is a zero placeholder of the shape its kernel signature expects.
# `required_when` and beyond are passed by keyword on purpose: these rules grew two optional
# fields after `default_shape`, and a positional argument silently landing in the wrong one is
# the kind of mistake that produces a plausible tensor of the wrong shape.
DEEPSEEK_V4_ROUTER_LAYER_RULES: tuple[LayerRule, ...] = (
    DefaultedWeightRule(
        "gate_bias",
        "ffn.gate.bias",
        torch.float32,
        ("n_routed_experts",),
        required_when="include_gate_bias",
    ),
    DefaultedWeightRule(
        "tid2eid",
        "ffn.gate.tid2eid",
        torch.int32,
        (_VOCAB_SIZE, _TOPK),
        required_when="include_tid2eid",
    ),
)

# Routed experts, sharded across ranks rather than replicated.
DEEPSEEK_V4_EXPERT_LAYER_RULES: tuple[LayerRule, ...] = (
    ExpertWeightRule("routed_w1", "w1.weight", torch.int8),
    ExpertWeightRule("routed_w1_scale", "w1.scale", torch.float32),
    ExpertWeightRule("routed_w3", "w3.weight", torch.int8),
    ExpertWeightRule("routed_w3_scale", "w3.scale", torch.float32),
    ExpertWeightRule("routed_w2", "w2.weight", torch.int8),
    ExpertWeightRule("routed_w2_scale", "w2.scale", torch.float32),
)

# The full 49-name layer contract, in the order the hand-written packer builds it.
DEEPSEEK_V4_LAYER_RULES: tuple[LayerRule, ...] = (
    *DEEPSEEK_V4_CORE_LAYER_RULES,
    *DEEPSEEK_V4_OPTIONAL_LAYER_RULES,
    *DEEPSEEK_V4_ROUTER_LAYER_RULES,
    *DEEPSEEK_V4_EXPERT_LAYER_RULES,
)

# DSpark draft layers are sliding-window decoder blocks. They do not consume
# target-model CSA/HCA compressor or indexer weights, so retaining those
# optional placeholders would allocate large tensors that no DSpark kernel
# argument can observe.
DEEPSEEK_V4_DSPARK_LAYER_RULES: tuple[LayerRule, ...] = (
    *DEEPSEEK_V4_CORE_LAYER_RULES,
    *DEEPSEEK_V4_ROUTER_LAYER_RULES,
    *DEEPSEEK_V4_EXPERT_LAYER_RULES,
)


def deepseek_v4_expert_parallel(ranks: int, n_routed_experts: int) -> ExpertParallel:
    """The expert placement policy, taking rank ownership from the loader's own helper."""
    from pypto_serving.model.deepseek.weight_loader import (  # noqa: PLC0415 -- cycle at import time
        deepseek_v4_local_expert_ids,
    )

    return ExpertParallel(
        ranks=ranks,
        n_experts=n_routed_experts,
        local_ids=deepseek_v4_local_expert_ids,
        mismatch_error=_MISMATCH_ERROR,
    )


def deepseek_v4_factories() -> dict[str, object]:
    """Synthetic-weight factories, keyed the way the rules refer to them."""
    from pypto_serving.model.deepseek.weight_loader import (  # noqa: PLC0415 -- cycle at import time
        deepseek_v4_hadamard_idx,
    )

    return {
        "hadamard_idx": deepseek_v4_hadamard_idx,
        # One row; the rank policy replicates it to [ranks, hidden].
        "hidden_ones": lambda: torch.ones((_HIDDEN,), dtype=torch.float32),
    }


DEEPSEEK_V4_RANK_ERROR = "packed DeepSeekV4 weight {name} must have rank >= 2, got {ndim}"
DEEPSEEK_V4_STACK_MISMATCH_ERROR = (
    "packed DeepSeekV4 weight {name} shape/dtype mismatch: source={source}, destination={destination}"
)


def deepseek_v4_stack_groups(compress_ratios: Sequence[int]) -> tuple[StackGroup, ...]:
    """Describe the three whole-model slab groups for a given per-layer attention layout.

    Every layer contributes to the FWD group; a layer also contributes to CSA or HCA depending
    on its compress ratio. Membership is expressed as the ordered list of layer ids in each
    group, so a group's slab holds its layers contiguously in first-appearance order — which is
    what the fused kernels index, and what the prepacked sidecar's offsets were written from.

    The FWD group declares ``members=None``: it is everything the other two do not claim, taken
    in the packer's own order, so a new weight joins it without being named here.
    """
    from pypto_serving.model.deepseek.weight_loader import (  # noqa: PLC0415 -- cycle at import time
        DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES,
        DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES,
    )

    ratios = [int(ratio) for ratio in compress_ratios]
    return (
        StackGroup(id="fwd", members=None, layer_ids=tuple(range(len(ratios)))),
        StackGroup(
            id="csa",
            members=tuple(DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES),
            layer_ids=tuple(i for i, ratio in enumerate(ratios) if ratio == DEEPSEEK_V4_CSA_RATIO),
        ),
        StackGroup(
            id="hca",
            members=tuple(DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES),
            layer_ids=tuple(i for i, ratio in enumerate(ratios) if ratio == DEEPSEEK_V4_HCA_RATIO),
        ),
    )


# DeepSeekV4 stages serially: packing one layer allocates ~8 GB of intermediates (256 routed
# experts, each stacked and rank-replicated), so overlapping layers multiplies the peak and
# contends on memory bandwidth rather than hiding latency.
DEEPSEEK_V4_STAGING_POLICY = StagingPolicy(workers=1)

# The whole-model weights. `head.weight` has no fallback here: unlike Qwen, a DeepSeekV4
# checkpoint always ships an untied LM head, and the packing it needs (contiguous TP vocab
# shards) is not expressible as a padded cast, so it stays in `pack_deepseek_v4_lm_head_weight`.
DEEPSEEK_V4_GLOBAL_RULES: tuple[GlobalWeightRule, ...] = (
    GlobalWeightRule("embed_weight", "embed.weight", torch.bfloat16),
    GlobalWeightRule("final_norm_weight", "norm.weight", torch.bfloat16),
    GlobalWeightRule("hc_head_fn", "hc_head_fn", torch.float32),
    GlobalWeightRule("hc_head_scale", "hc_head_scale", torch.float32),
    GlobalWeightRule("hc_head_base", "hc_head_base", torch.float32),
)


# The MTP draft layer is a full layer under the `mtp.0` prefix — which `LayerContext` already
# expresses, so the same 49 rules cover it — plus these twelve. Order is the order the packed
# mapping is built in, as everywhere else.
#
# The two `_smooth` tensors are synthesized rather than read: they are all-ones, and the
# quantized projections they scale have no per-channel smoothing in this checkpoint. The
# factory returns one row and the rank policy replicates it, which is what produces the
# `[ranks, hidden]` the kernel wants — a factory returning the rank-shaped tensor directly
# would get a second rank axis bolted on.
DEEPSEEK_V4_MTP_EXTRA_RULES: tuple[LayerRule, ...] = (
    LayerWeightRule("enorm_w", "enorm.weight", torch.float32),
    LayerWeightRule("hnorm_w", "hnorm.weight", torch.float32),
    LayerWeightRule("e_proj_w", "e_proj.weight", torch.int8),
    LayerWeightRule("e_proj_w_scale", "e_proj.scale", torch.float32),
    SyntheticWeightRule("e_proj_smooth", torch.float32, "hidden_ones"),
    LayerWeightRule("h_proj_w", "h_proj.weight", torch.int8),
    LayerWeightRule("h_proj_w_scale", "h_proj.scale", torch.float32),
    SyntheticWeightRule("h_proj_smooth", torch.float32, "hidden_ones"),
    LayerWeightRule("mtp_hc_head_fn", "hc_head_fn", torch.float32),
    LayerWeightRule("mtp_hc_head_scale", "hc_head_scale", torch.float32),
    LayerWeightRule("mtp_hc_head_base", "hc_head_base", torch.float32),
    LayerWeightRule("mtp_norm_w", "norm.weight", torch.bfloat16),
)

DEEPSEEK_V4_MTP_PREFIX = "mtp.0"
