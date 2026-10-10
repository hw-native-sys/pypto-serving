# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded checkpoint slices for lib's 40-layer decode weight ABI.

Each yielded tensor has an EP-leading axis and occupies one layer or producer
slot in ``decode_fwd.l3_decode_fwd``. The caller places slices into resident
buffers; this iterator never retains the whole checkpoint on the host.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .execution_plan import plan_layers
from .swa_weights import load_prefill_layer_weights


_ATTENTION = {
    "hc_attn_fn": "hc_attn_fn", "hc_attn_scale": "hc_attn_scale",
    "hc_attn_base": "hc_attn_base", "attn_norm_weight": "attn_norm_weight",
    "wq_a": "wq_a", "wq_a_scale": "wq_a_scale", "q_norm_weight": "q_norm_weight",
    "wq_b": "wq_b", "wq_b_scale": "wq_b_scale", "wkv": "wkv",
    "wkv_scale": "wkv_scale", "kv_norm_weight": "kv_norm_weight",
    "attn_sink": "attn_sink", "wo_a": "wo_a", "wo_b": "wo_b",
    "wo_b_scale": "wo_b_scale",
}
_MOE = {
    "hc_ffn_fn": "hc_ffn_fn", "hc_ffn_scale": "hc_ffn_scale",
    "hc_ffn_base": "hc_ffn_base", "ffn_norm_weight": "norm_weight",
    "gate_weight": "gate_weight", "correction_bias": "correction_bias",
    "routed_w1": "routed_w1", "routed_w1_scale": "routed_w1_scale",
    "routed_w2": "routed_w2", "routed_w2_scale": "routed_w2_scale",
    "routed_w3": "routed_w3", "routed_w3_scale": "routed_w3_scale",
    "shared_w1": "shared_w1", "shared_w1_scale": "shared_w1_scale",
    "shared_w2": "shared_w2", "shared_w2_scale": "shared_w2_scale",
    "shared_w3": "shared_w3", "shared_w3_scale": "shared_w3_scale",
}


@dataclass(frozen=True)
class DecodeWeightSlice:
    """One EP-stacked payload and its destination slot in the decode ABI."""

    name: str
    slot: int
    value: torch.Tensor


def bind_decode_layer_weights(layer, attention, moe, *, kv_sources, index_sources, c2a_sources):
    """Map one loaded layer to the lib's layer/source axes without copying data."""
    if not attention or not moe or "mxfp4_pair_lut" not in moe:
        raise ValueError("decode requires complete Attention and packed-FP4 MoE weights")
    result = [DecodeWeightSlice(target, layer.layer_id, attention[source])
              for target, source in _ATTENTION.items()]
    result += [DecodeWeightSlice(target, layer.layer_id, moe[source])
               for target, source in _MOE.items()]
    if layer.layer_id == 0:
        result.append(DecodeWeightSlice("mxfp4_pair_lut", 0, moe["mxfp4_pair_lut"]))
    if layer.layer_id in kv_sources:
        kv_slot = kv_sources.index(layer.layer_id)
        result.append(DecodeWeightSlice("compressor_norm_weight", kv_slot,
                                        attention["compressor_norm_weight"]))
        if layer.mode == "c2a_full":
            slot = c2a_sources.index(layer.layer_id)
            result.extend((
                DecodeWeightSlice("c2a_compressor_wkv", slot, attention["compressor_wkv"]),
                DecodeWeightSlice("c2a_compressor_wgate", slot, attention["compressor_wgate"]),
            ))
        elif layer.mode == "c1a_full":
            result.append(DecodeWeightSlice("c1a_compressor_wkv", 0, attention["compressor_wkv"]))
        else:
            raise ValueError("decode KV source must be a C1A/C2A Full layer")
        # The decode ABI reserves eight slots, but only the four KV producers
        # have index-key and key-norm checkpoint weights.
        index_slot = index_sources.index(layer.layer_id)
        result.extend((
            DecodeWeightSlice("index_wk", index_slot, attention["index_wk"]),
            DecodeWeightSlice("index_norm_weight", index_slot, attention["index_norm_weight"]),
        ))
    if layer.layer_id in index_sources:
        slot = index_sources.index(layer.layer_id)
        result.extend((
            DecodeWeightSlice("index_wq_b", slot, attention["index_wq_b"]),
            DecodeWeightSlice("index_wq_b_scale", slot, attention["index_wq_b_scale"]),
            DecodeWeightSlice("index_weights_proj", slot, attention["index_weights_proj"]),
        ))
    return tuple(result)


def iter_decode_weight_slices(model_dir, topology, *, max_bundle_bytes=32 << 30):
    """Yield one layer at a time in checkpoint order for bounded staging."""
    raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    layers = plan_layers(raw)
    if len(layers) != 40 or (topology.tp, topology.dp, topology.world) != (4, 2, 8):
        raise ValueError("lib decode_fwd requires the 40-layer TP4/DP2/EP8 schedule")
    text = raw["text_config"]
    kv_sources = tuple(text["kv_source_layer_ids"])
    index_sources = tuple(text["index_source_layer_ids"])
    c2a_sources = tuple(layer.layer_id for layer in layers if layer.mode == "c2a_full")
    if len(kv_sources) != 4 or len(index_sources) != 8 or len(c2a_sources) != 3:
        raise ValueError("lib decode_fwd source axes disagree with the checkpoint schedule")
    for layer in layers:
        attention, moe = load_prefill_layer_weights(
            model_dir, layer.layer_id, topology, max_bundle_bytes=max_bundle_bytes,
        )
        yield from bind_decode_layer_weights(
            layer, attention, moe, kv_sources=kv_sources, index_sources=index_sources,
            c2a_sources=c2a_sources,
        )
    # decode_fwd reserves one key/norm slot per index source, although only KV
    # producers have these checkpoint tensors. The trailing four slots are not
    # read by the current graph; initialize them deterministically nonetheless.
    head_dim = text["head_dim"]
    index_dim = text["index_head_dim"]
    for slot in range(len(kv_sources), len(index_sources)):
        yield DecodeWeightSlice("index_wk", slot,
                                torch.zeros((topology.world, head_dim, index_dim), dtype=torch.bfloat16))
        yield DecodeWeightSlice("index_norm_weight", slot,
                                torch.zeros((topology.world, index_dim), dtype=torch.bfloat16))
