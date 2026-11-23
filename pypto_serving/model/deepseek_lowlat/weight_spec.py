# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ND target-bank transforms; TP shards every expert's intermediate axis."""

from dataclasses import dataclass

import torch

from pypto_serving.model.deepseek.weight_spec import (
    DEEPSEEK_V4_CORE_LAYER_RULES,
    DEEPSEEK_V4_OPTIONAL_LAYER_RULES,
)


@dataclass(frozen=True)
class WeightRule:
    name: str
    source: str
    dtype: torch.dtype
    transpose: bool = False
    groups: int | None = None


LAYER_RULES = tuple(
    WeightRule(r.name, r.source, r.dtype, r.transpose, r.reshape_groups)
    for r in DEEPSEEK_V4_CORE_LAYER_RULES
    if not r.name.startswith("shared_")
)
COMPRESSOR_RULES = tuple(
    WeightRule(r.name, r.source, r.dtype, getattr(r, "transpose", False))
    for r in DEEPSEEK_V4_OPTIONAL_LAYER_RULES
    if r.name != "csa_hadamard_idx"
)
EXPERT_NAMES = tuple(f"routed_w{matrix}{suffix}" for matrix in (1, 3, 2) for suffix in ("", "_scale"))
GLOBAL_SOURCES = {
    "embed_weight": "embed.weight",
    "final_norm_w": "norm.weight",
    "lm_head_weight": "head.weight",
    "hc_head_fn": "hc_head_fn",
    "hc_head_scale": "hc_head_scale",
    "hc_head_base": "hc_head_base",
}
WEIGHT_NAMES = (frozenset(r.name for r in (*LAYER_RULES, *COMPRESSOR_RULES)) - {"wo_a", "wo_b"}) | {
    *EXPERT_NAMES,
    *GLOBAL_SOURCES,
    "csa_hadamard_idx",
    "gate_bias",
    "tid2eid",
    "wo_a_shard",
    "wo_b_shard",
}


def layer_ids(name):
    if name.startswith("hca_"):
        return tuple(range(3, 43, 2))
    if name.startswith("csa_"):
        return tuple(range(2, 43, 2))
    return tuple(range(43))


def transform_weight(rule, tensor):
    if rule.dtype == torch.int8 and tensor.dtype != torch.int8:
        raise ValueError(f"{rule.source} must contain native INT8 values")
    value = tensor.to(rule.dtype)
    if rule.name.endswith("_scale"):
        value = value.reshape(-1)
    if rule.transpose:
        if value.ndim != 2:
            raise ValueError(f"{rule.source} must be a matrix")
        value = value.T
    if rule.groups is not None:
        if value.shape[0] % rule.groups:
            raise ValueError(f"{rule.source} has invalid group geometry")
        value = value.reshape(rule.groups, value.shape[0] // rule.groups, *value.shape[1:])
    return value.contiguous()


def shard_expert(value, *, matrix, scale, rank, tp=8):
    """Keep w2's output scales replicated while partitioning its input columns."""
    if not 0 <= rank < tp or matrix not in (1, 2, 3):
        raise ValueError("invalid expert TP shard")
    if scale:
        value = value.reshape(-1)
        if matrix == 2:
            return value.contiguous()
        axis = 0
    else:
        if value.dtype != torch.int8 or value.ndim != 2:
            raise ValueError("expert weights must be native INT8 matrices")
        axis = 1 if matrix == 2 else 0
    if value.shape[axis] % tp:
        raise ValueError("expert intermediate dimension must divide TP")
    width = value.shape[axis] // tp
    return value.narrow(axis, rank * width, width).contiguous()


def checkpoint_coverage(weight_map):
    required = set(GLOBAL_SOURCES.values())
    for rule in (*LAYER_RULES, *COMPRESSOR_RULES):
        required.update(f"layers.{layer}.{rule.source}" for layer in layer_ids(rule.name))
    for layer in range(43):
        required.add(f"layers.{layer}.ffn.gate." + ("tid2eid" if layer < 3 else "bias"))
        for expert in range(257):
            prefix = f"layers.{layer}.ffn." + (f"experts.{expert}" if expert < 256 else "shared_experts")
            required.update(
                f"{prefix}.w{matrix}.{suffix}" for matrix in (1, 2, 3) for suffix in ("weight", "scale")
            )
    missing = sorted(required - weight_map.keys())
    # The checkpoint stores the three next-token prediction layers under
    # layers.43..45; M0 target-only execution intentionally does not load them.
    ignored = sorted(
        name for name in weight_map if name.startswith(("layers.43.", "layers.44.", "layers.45.", "mtp."))
    )
    unknown = sorted(weight_map.keys() - required - set(ignored))
    if missing or unknown:
        raise ValueError(
            f"lowlat checkpoint coverage mismatch: missing={missing[:8]}, unconsumed={unknown[:8]}"
        )
    return {
        "required": sorted(required),
        "ignored_k0_drafter": ignored,
        "missing": missing,
        "unconsumed": unknown,
    }
