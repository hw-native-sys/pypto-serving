# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Checkpoint-to-CPU ABI bundles for the bounded SWA/MoE segment.

Uses the existing selective loader, as V4 separates loading from execution.
No lib fixtures, device execution or FP4 float expansion belongs here.
"""
import torch

from .weight_loader import V41WeightLoader
from .weight_packing import pack_mx_scale


def stack_bytes(values):
    """Stack even float8 tensors on CPU, preserving every payload bit."""
    dtype = values[0].dtype
    if any(v.dtype != dtype or v.shape != values[0].shape for v in values):
        raise ValueError("rank weights must have matching shape and dtype")
    return torch.stack([v.contiguous().view(torch.uint8) for v in values]).view(dtype)


def merge_expert_scales(scales):
    """Repack per-expert MX_B_NN into the lib's flattened expert/K-group ABI.

    Concatenating the already packed buffers would put expert before output
    block, whereas the kernel expects output block before expert/K-group.
    Only scale bytes are reordered; routed weight payloads stay packed FP4.
    """
    groups, width = scales[0].shape
    if groups % 2 or width % 16 or any(s.shape != scales[0].shape for s in scales):
        raise ValueError("expert scales must share a valid MX_B_NN shape")
    logical = [s.contiguous().view(torch.uint8).reshape(width // 16, groups // 2, 16, 2)
               .permute(1, 3, 0, 2).reshape(groups, width) for s in scales]
    return pack_mx_scale(torch.cat(logical)).view(scales[0].dtype)


def mxfp4_pair_lut():
    """Decode two E2M1 nibbles into two E4M3FN bytes in the lib LUT ABI."""
    codes = torch.tensor([0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
                          0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC], dtype=torch.int64)
    packed = torch.arange(256)
    pair = codes[packed & 15] | (codes[packed >> 4] << 8)
    pair = torch.where(pair < 32768, pair, pair - 65536)
    return pair.to(torch.int16).reshape(1, 256).repeat(2, 1)


def load_swa_layer_weights(model_dir, layer_id, topology, *, max_bundle_bytes=32 << 30):
    """Return stacked Attention and MoE weight maps for one SWA layer.

    The caller owns the results and uploads them once. max_bundle_bytes is a
    conservative bound on stacking buffers for this call, not process memory
    or the accumulated weights of other layers. Individual payload reads retain
    V41WeightLoader's separate pre-read budget.
    """
    return _load_layer_weights(model_dir, layer_id, topology, max_bundle_bytes, swa_only=True)


def load_prefill_layer_weights(model_dir, layer_id, topology, *, max_bundle_bytes=32 << 30):
    """Load this layer's real Attention/MoE weights for the prefill SP entries.

    Full modes own compressor and index-key weights; Reindex owns only index
    query/gating weights; Reuse owns neither. Producer-owned caches and unused
    C1A ABI buffers must be supplied by the resource adapter, not fabricated by
    the loader. Index heads are replicated, matching lib's global INDEX_H.
    """
    return _load_layer_weights(model_dir, layer_id, topology, max_bundle_bytes, swa_only=False)


def load_prefill_attention_weights(model_dir, layer_id, topology, *, max_bundle_bytes=4 << 30):
    """Read only Attention weights for isolated tracing or bounded preparation.

    Uses the identical packing/placement as the complete layer bundle. In
    particular, an Attention probe must not read all routed expert payloads.
    """
    attention, _ = _load_layer_weights(model_dir, layer_id, topology, max_bundle_bytes,
                                       swa_only=False, include_moe=False)
    return attention


def _load_layer_weights(model_dir, layer_id, topology, max_bundle_bytes, *, swa_only, include_moe=True):
    if type(max_bundle_bytes) is not int or max_bundle_bytes <= 0:
        raise ValueError("max_bundle_bytes must be positive")
    attention, moe = [], []
    held_bytes = 0
    for rank in range(topology.world):
        loader = V41WeightLoader(model_dir, tp_size=topology.tp, tp_rank=rank % topology.tp,
                                 ep_size=topology.world, ep_rank=rank, max_load_bytes=512 << 20)
        if type(layer_id) is not int or not 0 <= layer_id < loader.config.num_hidden_layers:
            raise ValueError("selected layer must be a backbone layer")
        ratio = loader.text["compress_ratios"][layer_id]
        if swa_only and ratio != 0:
            raise ValueError("selected layer must be a SWA backbone layer")
        prefix = f"layers.{layer_id}."
        a, m = {}, {}

        def load(name):
            nonlocal held_bytes
            bundle = loader.load(prefix + name)
            held_bytes += sum(t.numel() * t.element_size() for t in (bundle.weight, bundle.scale)
                              if t is not None)
            if 3 * held_bytes > max_bundle_bytes:
                raise ValueError("stacked layer weight budget exceeded")
            return bundle

        for target, source in {
            "hc_attn_fn": "hc_attn_fn", "hc_attn_scale": "hc_attn_scale",
            "hc_attn_base": "hc_attn_base", "attn_norm_weight": "attn_norm.weight",
            "q_norm_weight": "attn.q_norm.weight", "kv_norm_weight": "attn.kv_norm.weight",
            "attn_sink": "attn.attn_sink",
        }.items():
            a[target] = load(source).weight
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            bundle = load(f"attn.{name}.weight")
            a[name] = bundle.weight
            if bundle.scale is not None:
                a[name + "_scale"] = bundle.scale
        if layer_id in loader.text["kv_source_layer_ids"]:
            for target, source in {
                "compressor_wkv": "attn.compressor.wkv.weight",
                "compressor_norm_weight": "attn.compressor.norm.weight",
                "index_wk": "attn.indexer.wk.weight",
                "index_norm_weight": "attn.indexer.k_norm.weight",
            }.items():
                a[target] = load(source).weight
            if ratio == 2:
                a["compressor_wgate"] = load("attn.compressor.wgate.weight").weight
        if layer_id in loader.text["index_source_layer_ids"]:
            bundle = load("attn.indexer.wq_b.weight")
            a["index_wq_b"], a["index_wq_b_scale"] = bundle.weight, bundle.scale
            a["index_weights_proj"] = load("attn.indexer.weights_proj.weight").weight
        attention.append(a)
        if not include_moe:
            continue
        for target, source in {
            "hc_ffn_fn": "hc_ffn_fn", "hc_ffn_scale": "hc_ffn_scale", "hc_ffn_base": "hc_ffn_base",
            "norm_weight": "ffn_norm.weight", "gate_weight": "ffn.gate.weight",
            "correction_bias": "ffn.gate.bias",
        }.items():
            m[target] = load(source).weight
        for name in ("w1", "w2", "w3"):
            bundle = load(f"ffn.shared_experts.{name}.weight")
            m["shared_" + name], m["shared_" + name + "_scale"] = bundle.weight, bundle.scale
            count = loader.text["n_routed_experts"] // topology.world
            bundles = [load(f"ffn.experts.{expert}.{name}.weight")
                       for expert in range(rank * count, (rank + 1) * count)]
            m["routed_" + name] = stack_bytes([b.weight for b in bundles])
            m["routed_" + name + "_scale"] = merge_expert_scales([b.scale for b in bundles])
            del bundles
        m["mxfp4_pair_lut"] = mxfp4_pair_lut()
        moe.append(m)
    return ({name: stack_bytes([r[name] for r in attention]) for name in attention[0]},
            {name: stack_bytes([r[name] for r in moe]) for name in moe[0]} if moe else {})
