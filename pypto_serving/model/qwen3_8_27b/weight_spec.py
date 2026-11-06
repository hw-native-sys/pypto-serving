# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eco-Tech Qwen3.8-27B checkpoint names and type-local stacked weight layout.

The PyPTO kernels take row-major output-by-input matrices. Quantized weight
scales are checkpoint ``[out, 1]`` tensors and become ``[1, out]`` rows before
stacking; the runner flattens those slabs at dispatch. Convolution taps and the
paired A/B projection need two small model-specific factories. All remaining
weights use the shared declarative packer and stacker.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from pypto_serving.model.common.weights.packer import pack_layer
from pypto_serving.model.common.weights.pipeline import StagingPolicy
from pypto_serving.model.common.weights.shard import NoShard
from pypto_serving.model.common.weights.spec import (
    GlobalWeightRule,
    LayerContext,
    LayerWeightRule,
    SyntheticWeightRule,
)
from pypto_serving.model.common.weights.stacker import StackGroup, stack_layers
from pypto_serving.model.common.weights.store import LazySafetensorsStore
from pypto_serving.model.qwen3_8_27b.cache import (
    QWEN38_ATTENTION_LAYERS,
    QWEN38_GDN_LAYERS,
    QWEN38_NUM_LAYERS,
)


class Qwen38WeightStore(LazySafetensorsStore):
    """Read one layer's named tensors without materializing its neighbours."""

    missing_name_error = "Missing Qwen3.8 weight tensor in index: {name}"
    missing_names_error = "Qwen3.8 checkpoint is missing required tensors: {names}"
    missing_shard_error = "Missing safetensors shard for Qwen3.8 weight load: {path}"


QWEN38_COMMON_RULES = (
    LayerWeightRule("norm_in_w", "input_layernorm.weight", torch.bfloat16, flatten_to_row=True),
    LayerWeightRule("norm_post_w", "post_attention_layernorm.weight", torch.bfloat16,
                    flatten_to_row=True),
    LayerWeightRule("w_gate", "mlp.gate_proj.weight", torch.int8),
    LayerWeightRule("w_gate_scale", "mlp.gate_proj.weight_scale", torch.float32, transpose=True),
    LayerWeightRule("w_up", "mlp.up_proj.weight", torch.int8),
    LayerWeightRule("w_up_scale", "mlp.up_proj.weight_scale", torch.float32, transpose=True),
    LayerWeightRule("w_down", "mlp.down_proj.weight", torch.int8),
    LayerWeightRule("w_down_scale", "mlp.down_proj.weight_scale", torch.float32, transpose=True),
)

QWEN38_GDN_RULES = (
    LayerWeightRule("w_qkv", "linear_attn.in_proj_qkv.weight", torch.int8),
    LayerWeightRule("w_qkv_scale", "linear_attn.in_proj_qkv.weight_scale", torch.float32,
                    transpose=True),
    LayerWeightRule("w_z", "linear_attn.in_proj_z.weight", torch.int8),
    LayerWeightRule("w_z_scale", "linear_attn.in_proj_z.weight_scale", torch.float32,
                    transpose=True),
    SyntheticWeightRule("w_ab", torch.bfloat16, "paired_ab"),
    SyntheticWeightRule("conv_w", torch.float32, "conv_taps"),
    LayerWeightRule("a_log", "linear_attn.A_log", torch.bfloat16, flatten_to_row=True),
    LayerWeightRule("dt_bias", "linear_attn.dt_bias", torch.bfloat16, flatten_to_row=True),
    LayerWeightRule("norm_w", "linear_attn.norm.weight", torch.bfloat16,
                    flatten_to_row=True),
    LayerWeightRule("w_out", "linear_attn.out_proj.weight", torch.bfloat16),
)

QWEN38_ATTENTION_RULES = (
    # Decode's projection fuses Q/G, K and V; prefill reads their original
    # individual slabs. Both layouts are kept so neither kernel repacks at run time.
    SyntheticWeightRule("decode_w_qkv", torch.int8, "decode_qkv"),
    SyntheticWeightRule("decode_w_qkv_scale", torch.float32, "decode_qkv_scale"),
    LayerWeightRule("w_q", "self_attn.q_proj.weight", torch.int8),
    LayerWeightRule("w_q_scale", "self_attn.q_proj.weight_scale", torch.float32, transpose=True),
    LayerWeightRule("w_k", "self_attn.k_proj.weight", torch.int8),
    LayerWeightRule("w_k_scale", "self_attn.k_proj.weight_scale", torch.float32, transpose=True),
    LayerWeightRule("w_v", "self_attn.v_proj.weight", torch.int8),
    LayerWeightRule("w_v_scale", "self_attn.v_proj.weight_scale", torch.float32, transpose=True),
    LayerWeightRule("w_o", "self_attn.o_proj.weight", torch.int8),
    LayerWeightRule("w_o_scale", "self_attn.o_proj.weight_scale", torch.float32, transpose=True),
    LayerWeightRule("q_norm_w", "self_attn.q_norm.weight", torch.bfloat16,
                    flatten_to_row=True),
    LayerWeightRule("k_norm_w", "self_attn.k_norm.weight", torch.bfloat16,
                    flatten_to_row=True),
)

QWEN38_GLOBAL_RULES = (
    GlobalWeightRule("embed_weight", "model.language_model.embed_tokens.weight", torch.bfloat16),
    GlobalWeightRule("final_norm_w", "model.language_model.norm.weight", torch.bfloat16),
    GlobalWeightRule("lm_w", "lm_head.weight", torch.bfloat16),
)

_SYNTHETIC_SOURCES = {
    "w_ab": ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"),
    "conv_w": ("linear_attn.conv1d.weight",),
    "decode_w_qkv": ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                     "self_attn.v_proj.weight"),
    "decode_w_qkv_scale": ("self_attn.q_proj.weight_scale", "self_attn.k_proj.weight_scale",
                           "self_attn.v_proj.weight_scale"),
}


def qwen38_layer_prefix(layer_id: int) -> str:
    """Return the checkpoint's text-layer prefix for a global layer index."""
    if not 0 <= layer_id < QWEN38_NUM_LAYERS:
        raise IndexError(f"Qwen3.8 layer {layer_id} outside [0, {QWEN38_NUM_LAYERS})")
    return f"model.language_model.layers.{layer_id}"


def qwen38_layer_rules(layer_id: int):
    """Return the common and type-specific rules for one layer."""
    qwen38_layer_prefix(layer_id)
    mixer = QWEN38_ATTENTION_RULES if layer_id in QWEN38_ATTENTION_LAYERS else QWEN38_GDN_RULES
    return QWEN38_COMMON_RULES + mixer


def qwen38_layer_weight_names(layer_id: int) -> tuple[str, ...]:
    """Return every real checkpoint tensor consumed by this text layer."""
    prefix = qwen38_layer_prefix(layer_id)
    suffixes = []
    for rule in qwen38_layer_rules(layer_id):
        if isinstance(rule, SyntheticWeightRule):
            suffixes.extend(_SYNTHETIC_SOURCES[rule.name])
        else:
            suffixes.append(rule.source)
    return tuple(dict.fromkeys(f"{prefix}.{suffix}" for suffix in suffixes))


def qwen38_stack_groups() -> tuple[StackGroup, ...]:
    """Pack mixer slabs by type-local layer index, shared slabs by global index."""
    return (
        StackGroup("gdn", tuple(rule.name for rule in QWEN38_GDN_RULES), QWEN38_GDN_LAYERS),
        StackGroup("attention", tuple(rule.name for rule in QWEN38_ATTENTION_RULES),
                   QWEN38_ATTENTION_LAYERS),
        StackGroup("shared", None, tuple(range(QWEN38_NUM_LAYERS))),
    )


def qwen38_staging_policy(workers: int) -> StagingPolicy:
    """Overlap small independent layer reads, with one torch thread per worker."""
    return StagingPolicy(workers=max(1, min(int(workers), QWEN38_NUM_LAYERS)),
                         pin_torch_threads=True)


def pack_qwen38_layer(
    layer_id: int,
    raw: Mapping[str, torch.Tensor],
    *,
    destinations: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Pack one layer directly into its type-local and shared slab slices."""
    prefix = qwen38_layer_prefix(layer_id)

    def paired_ab() -> torch.Tensor:
        return torch.cat((raw[f"{prefix}.linear_attn.in_proj_a.weight"],
                          raw[f"{prefix}.linear_attn.in_proj_b.weight"]), dim=0)

    def conv_taps() -> torch.Tensor:
        taps = raw[f"{prefix}.linear_attn.conv1d.weight"]
        return taps.reshape(taps.shape[0], -1).transpose(0, 1).contiguous()

    def decode_qkv() -> torch.Tensor:
        return torch.cat(tuple(raw[f"{prefix}.self_attn.{projection}_proj.weight"]
                               for projection in ("q", "k", "v")), dim=0)

    def decode_qkv_scale() -> torch.Tensor:
        return torch.cat(tuple(raw[f"{prefix}.self_attn.{projection}_proj.weight_scale"]
                               for projection in ("q", "k", "v")), dim=0).transpose(0, 1)

    return pack_layer(
        qwen38_layer_rules(layer_id), raw,
        LayerContext(layer_id=layer_id, prefix=prefix, ranks=1),
        policy=NoShard(), destinations=destinations,
        factories={"paired_ab": paired_ab, "conv_taps": conv_taps,
                   "decode_qkv": decode_qkv, "decode_qkv_scale": decode_qkv_scale},
        missing_source_error=Qwen38WeightStore.missing_name_error,
    )


def stage_qwen38_weights(
    store: Qwen38WeightStore,
    *,
    workers: int = 1,
    allocate: Callable[[tuple[int, ...], torch.dtype], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Stage 64 layers into the three type-local/global slab groups."""
    def read(layer_id: int) -> dict[str, torch.Tensor]:
        return store.load_many(qwen38_layer_weight_names(layer_id))

    # The template is copied into layer 0, so its shared norm/MLP values must
    # come from layer 0. Layer 3 supplies only the attention-specific shapes.
    template = {**pack_qwen38_layer(3, read(3)), **pack_qwen38_layer(0, read(0))}

    def pack_into(layer_id: int, destinations: Mapping[str, torch.Tensor]) -> None:
        pack_qwen38_layer(layer_id, read(layer_id), destinations=destinations)

    return stack_layers(
        qwen38_stack_groups(), template,
        layer_ids=range(QWEN38_NUM_LAYERS), pack_into=pack_into,
        template_layer_id=0, stack_axis=0,
        allocate=allocate, policy=qwen38_staging_policy(workers),
    )
