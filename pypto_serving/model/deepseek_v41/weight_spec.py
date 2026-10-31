# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 weight spec."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


# weight_packing
def pack_mx_scale(codes):
    """Logical [K/32,N] E8M0 bytes to MX_B_NN physical order."""
    groups, width = codes.shape
    if groups % 2 or width % 16:
        raise ValueError("MX_B_NN requires K divisible by 64 and N divisible by 16")
    return codes.reshape(groups // 2, 2, width // 16, 16).permute(2, 0, 3, 1).contiguous().reshape(
        groups, width,
    )


def pack_fp4_tiles(payload, tile=256):
    """Reorder packed [N,K/2] to lib [K*N/256,128], without float expansion."""
    source = payload.contiguous().view(torch.uint8)
    n, half_k = source.shape
    k = half_k * 2
    if k % tile or n % tile:
        raise ValueError("routed FP4 weights require complete 256x256 tiles")
    output = torch.empty((n // tile, k // tile, tile, tile // 2), dtype=torch.uint8)
    for nb in range(n // tile):
        for kb in range(k // tile):
            block = source[nb*tile:(nb+1)*tile, kb*(tile//2):(kb+1)*(tile//2)]
            codes = torch.empty((tile, tile), dtype=torch.uint8)
            codes[:, 0::2] = block & 15
            codes[:, 1::2] = block >> 4
            transposed = codes.T
            output[nb, kb] = transposed[:, 0::2] | (transposed[:, 1::2] << 4)
    return output.reshape(k * n // tile, tile // 2)


def fp8_input_major(payload, scale):
    """Checkpoint FP8 block32 [N,K] to [K,N] plus E8M0 MX_B_NN backing."""
    n, k = payload.shape
    if k % 64 or n % 32:
        raise ValueError("FP8 native projection requires K divisible by 64 and N by 32")
    codes = scale.contiguous().view(torch.uint8)
    if tuple(codes.shape) != (n // 32, k // 32):
        raise ValueError("FP8 block scales do not match the projection")
    return payload.T.contiguous(), pack_mx_scale(codes.T.repeat_interleave(32, dim=1)).view(
        torch.float8_e8m0fnu,
    )


def pack_grouped_wo_a(payload, scale, groups):
    """Keep checkpoint Wo-A in the grouped FP8/scale layout used by lib."""
    n, k = payload.shape
    if n % groups or n % 32 or k % 64 or (n // groups) % 32:
        raise ValueError("wo_a requires complete output groups and 32x32 quantization blocks")
    codes = scale.contiguous().view(torch.uint8)
    if tuple(codes.shape) != (n // 32, k // 32):
        raise ValueError("wo_a scales do not match the projection")
    group_width = n // groups
    weight = payload.reshape(groups, group_width, k).transpose(-1, -2).contiguous()
    logical_scale = codes.repeat_interleave(32, dim=0).T.contiguous()
    packed_scale = torch.stack([
        pack_mx_scale(logical_scale[:, group * group_width:(group + 1) * group_width])
        for group in range(groups)
    ])
    return weight, packed_scale.view(torch.float8_e8m0fnu)


# weight_spec
@dataclass(frozen=True)
class TensorSpec:
    """One source tensor and its required reference-layout transformation.

    ``shape`` is the physical checkpoint shape, so FP4 weights have K/2 bytes.
    Each quantized projection keeps its scale as a runtime parameter.
    """

    shape: tuple[int, ...]
    dtype: str
    runtime_name: str
    conversion: str


def backbone_weight_specs(config: Mapping[str, Any]) -> dict[str, TensorSpec]:
    """Build required source specs from a Hugging Face V4.1 config object.

    This accepts the outer config containing ``text_config`` and
    ``quantization_config``. It fails on unsupported quantization instead of
    treating an arbitrary I8 tensor as E2M1. Layer ownership comes from source IDs.
    The independent config parser should validate full model semantics first.
    """
    text = config["text_config"]
    quant = config["quantization_config"]
    if not isinstance(text, Mapping) or not isinstance(quant, Mapping):
        raise ValueError("text_config and quantization_config must be objects")
    if (
        quant.get("quant_method") != "fp8"
        or tuple(quant.get("weight_block_size", ())) != (32, 32)
        or quant.get("scale_fmt") != "ue8m0"
        or quant.get("expert_dtype") != "fp4"
    ):
        raise ValueError("V4.1 weight specs require FP8 32x32, UE8M0 and FP4 block-32 experts")

    def positive(name: str) -> int:
        value = text[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"text_config.{name} must be a positive integer")
        return value

    dim, inter = positive("hidden_size"), positive("moe_intermediate_size")
    layers, heads = positive("num_hidden_layers"), positive("num_attention_heads")
    experts, shared = positive("n_routed_experts"), positive("n_shared_experts")
    head_dim, q_rank = positive("head_dim"), positive("q_lora_rank")
    o_rank, groups = positive("o_lora_rank"), positive("o_groups")
    hc, vocab = positive("hc_mult"), positive("vocab_size")
    index_heads, index_dim = positive("index_n_heads"), positive("index_head_dim")
    ratios = text["compress_ratios"]
    kv_sources, index_sources = set(text["kv_source_layer_ids"]), set(text["index_source_layer_ids"])
    if len(ratios) < layers:
        raise ValueError("compression metadata must cover backbone layers")
    if any(type(i) is not int or not 0 <= i < layers for i in (*kv_sources, *index_sources)):
        raise ValueError("weight source IDs must identify backbone layers")
    if not kv_sources <= index_sources or any(ratios[i] < 1 for i in kv_sources | index_sources):
        raise ValueError("KV sources must also own an indexer and use a positive compression ratio")
    if heads % groups or shared != 1:
        raise ValueError("V4.1 requires divisible attention groups and one shared expert")

    specs: dict[str, TensorSpec] = {}

    def add(name: str, shape: tuple[int, ...], dtype: str, conversion: str = "identity", target=None):
        if any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError(f"{name}: invalid tensor shape {shape}")
        if name in specs:
            raise ValueError(f"duplicate source spec: {name}")
        specs[name] = TensorSpec(shape, dtype, target or name, conversion)

    def dense(name: str, out_dim: int, in_dim: int, shard: str = "replicate"):
        target = name + ".weight"
        add(target, (out_dim, in_dim), "F8_E4M3", f"preserve_fp8_32x32_ue8m0;{shard}")
        add(
            name + ".scale",
            ((out_dim + 31) // 32, (in_dim + 31) // 32),
            "F8_E8M0",
            f"preserve_ue8m0;{shard}",
        )

    def expert(name: str, routed: bool):
        for proj, out_dim, in_dim in (("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)):
            prefix = f"{name}.{proj}"
            if routed:
                if in_dim % 32:
                    raise ValueError(f"{prefix}: FP4 input dimension must be divisible by 32")
                add(
                    prefix + ".weight",
                    (out_dim, in_dim // 2),
                    "I8",
                    "reinterpret_packed_e2m1_low_nibble_first;ep_select_expert",
                )
                add(
                    prefix + ".scale",
                    (out_dim, in_dim // 32),
                    "F8_E8M0",
                    "preserve_ue8m0_per_row_block32;ep_select_expert",
                )
            else:
                dense(prefix, out_dim, in_dim)

    add("embed.weight", (vocab, dim), "BF16", "tp_shard_axis0")
    add("head.weight", (vocab, dim), "BF16", "tp_shard_axis0")
    add("norm.weight", (dim,), "BF16")
    for layer in range(layers):
        base = f"layers.{layer}"
        attn = base + ".attn"
        add(attn + ".attn_sink", (heads,), "F32", "tp_shard_axis0")
        dense(attn + ".wq_a", q_rank, dim)
        dense(attn + ".wq_b", heads * head_dim, q_rank, "tp_shard_axis0")
        dense(attn + ".wkv", head_dim, dim)
        dense(
            attn + ".wo_a",
            groups * o_rank,
            heads * head_dim // groups,
            "tp_shard_axis0;view_grouped_output_projection",
        )
        dense(attn + ".wo_b", dim, groups * o_rank, "tp_shard_axis1")
        for name, size in (("q_norm", q_rank), ("kv_norm", head_dim)):
            add(f"{attn}.{name}.weight", (size,), "BF16")
        for sublayer in ("attn", "ffn"):
            add(f"{base}.{sublayer}_norm.weight", (dim,), "BF16")
            mix = (2 + hc) * hc
            add(f"{base}.hc_{sublayer}_fn", (mix, hc * dim), "F32")
            add(f"{base}.hc_{sublayer}_base", (mix,), "F32")
            add(f"{base}.hc_{sublayer}_scale", (3,), "F32")
        add(base + ".ffn.gate.weight", (experts, dim), "BF16", "cast_to_fp32_for_routing")
        add(base + ".ffn.gate.bias", (experts,), "F32")
        for expert_id in range(experts):
            expert(f"{base}.ffn.experts.{expert_id}", routed=True)
        expert(base + ".ffn.shared_experts", routed=False)
        if layer in kv_sources:
            prefix = attn + ".compressor"
            promotion = "cast_bf16_to_fp32" if ratios[layer] > 1 else "identity"
            add(prefix + ".wkv.weight", (head_dim, dim), "BF16", promotion)
            add(prefix + ".norm.weight", (head_dim,), "BF16")
            if ratios[layer] > 1:
                add(prefix + ".wgate.weight", (head_dim, dim), "BF16", promotion)
        if layer in index_sources:
            prefix = attn + ".indexer"
            # C1A/C2A composites at lib fbe92bfc compute complete index scores
            # on every TP rank (INDEX_H is global). Main attention heads are
            # sharded, but cutting index heads would omit part of each score.
            dense(prefix + ".wq_b", index_heads * index_dim, q_rank)
            add(prefix + ".weights_proj.weight", (index_heads, dim), "BF16", "replicate")
            if layer in kv_sources:
                add(prefix + ".wk.weight", (index_dim, head_dim), "BF16")
                add(prefix + ".k_norm.weight", (index_dim,), "BF16")
    # Engram projection weights are shared across EP ranks. Large lookup
    # tables are deliberately excluded; they remain row-addressed host data.
    engram_layers = tuple(text.get("engram_layer_ids", ()))
    if engram_layers:
        engram_heads = positive("engram_n_heads")
        engram_head_dim = positive("engram_head_dim")
        engram_k = (positive("engram_max_ngram_size") - 1) * engram_heads * engram_head_dim
        engram_out = (hc + 1) * dim
        for layer in engram_layers:
            prefix = f"layers.{layer}.engram"
            dense(prefix + ".wkv", engram_out, engram_k)
            add(prefix + ".q_weight", (hc, dim), "BF16")
            add(prefix + ".k_weight", (hc, dim), "BF16")
    return specs
