# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check real checkpoint weight slices against the current decode ABI geometry.

CPU only. This checks selected layer types without retaining all 40 layers or
uploading any tensors; it is not a device or numerical model validation.
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from pypto_serving.model.deepseek_v41.decode_weights import bind_decode_layer_weights
from pypto_serving.model.deepseek_v41.execution_plan import plan_layers
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology
from pypto_serving.model.deepseek_v41.swa_weights import load_prefill_layer_weights


def expected_geometry(text, topology):
    world, tp = topology.world, topology.tp
    d, inter = text["hidden_size"], text["moe_intermediate_size"]
    q, head = text["q_lora_rank"], text["head_dim"]
    heads, groups = text["num_attention_heads"], text["o_groups"]
    index_h, index_d = text["index_n_heads"], text["index_head_dim"]
    local_experts = text["n_routed_experts"] // world
    local_heads, local_groups = heads // tp, groups // tp
    local_o = local_groups * text["o_lora_rank"]
    mix = (2 + text["hc_mult"]) * text["hc_mult"]
    return {
        "hc_attn_fn": ((world, mix, text["hc_mult"] * d), torch.float32),
        "hc_attn_scale": ((world, 3), torch.float32),
        "hc_attn_base": ((world, mix), torch.float32),
        "attn_norm_weight": ((world, d), torch.bfloat16),
        "wq_a": ((world, d, q), torch.float8_e4m3fn),
        "wq_a_scale": ((world, d // 32, q), torch.float8_e8m0fnu),
        "q_norm_weight": ((world, q), torch.bfloat16),
        "wq_b": ((world, q, local_heads * head), torch.float8_e4m3fn),
        "wq_b_scale": ((world, q // 32, local_heads * head), torch.float8_e8m0fnu),
        "wkv": ((world, d, head), torch.float8_e4m3fn),
        "wkv_scale": ((world, d // 32, head), torch.float8_e8m0fnu),
        "kv_norm_weight": ((world, head), torch.bfloat16),
        "attn_sink": ((world, local_heads), torch.float32),
        "wo_a": ((world, local_groups, text["o_lora_rank"],
                  heads * head // groups), torch.bfloat16),
        "wo_b": ((world, local_o, d), torch.float8_e4m3fn),
        "wo_b_scale": ((world, local_o // 32, d), torch.float8_e8m0fnu),
        "hc_ffn_fn": ((world, mix, text["hc_mult"] * d), torch.float32),
        "hc_ffn_scale": ((world, 3), torch.float32),
        "hc_ffn_base": ((world, mix), torch.float32),
        "ffn_norm_weight": ((world, d), torch.bfloat16),
        "gate_weight": ((world, text["n_routed_experts"], d), torch.float32),
        "correction_bias": ((world, text["n_routed_experts"]), torch.float32),
        "routed_w1": ((world, local_experts, inter * d // 256, 128), torch.uint8),
        "routed_w2": ((world, local_experts, inter * d // 256, 128), torch.uint8),
        "routed_w3": ((world, local_experts, inter * d // 256, 128), torch.uint8),
        "routed_w1_scale": ((world, local_experts * d // 32, inter), torch.float8_e8m0fnu),
        "routed_w2_scale": ((world, local_experts * inter // 32, d), torch.float8_e8m0fnu),
        "routed_w3_scale": ((world, local_experts * d // 32, inter), torch.float8_e8m0fnu),
        "shared_w1": ((world, d, inter), torch.float8_e4m3fn),
        "shared_w1_scale": ((world, d // 32, inter), torch.float8_e8m0fnu),
        "shared_w2": ((world, inter, d), torch.float8_e4m3fn),
        "shared_w2_scale": ((world, inter // 32, d), torch.float8_e8m0fnu),
        "shared_w3": ((world, d, inter), torch.float8_e4m3fn),
        "shared_w3_scale": ((world, d // 32, inter), torch.float8_e8m0fnu),
        "mxfp4_pair_lut": ((world, 2, 256), torch.int16),
        "c2a_compressor_wkv": ((world, d, head), torch.float32),
        "c2a_compressor_wgate": ((world, d, head), torch.float32),
        "c1a_compressor_wkv": ((world, d, head), torch.bfloat16),
        "compressor_norm_weight": ((world, head), torch.bfloat16),
        "index_wk": ((world, head, index_d), torch.bfloat16),
        "index_norm_weight": ((world, index_d), torch.bfloat16),
        "index_wq_b": ((world, q, index_h * index_d), torch.float8_e4m3fn),
        "index_wq_b_scale": ((world, q // 32, index_h * index_d), torch.float8_e8m0fnu),
        "index_weights_proj": ((world, d, index_h), torch.bfloat16),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=(0, 2, 20, 24, 39))
    args = parser.parse_args()
    raw = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    layers = plan_layers(raw)
    topology = SegmentTopology(tp=4, dp=2)
    geometry = expected_geometry(raw["text_config"], topology)
    kv_sources = tuple(raw["text_config"]["kv_source_layer_ids"])
    index_sources = tuple(raw["text_config"]["index_source_layer_ids"])
    c2a_sources = tuple(layer.layer_id for layer in layers if layer.mode == "c2a_full")
    if len(layers) != 40 or (len(kv_sources), len(index_sources), len(c2a_sources)) != (4, 8, 3):
        raise ValueError("checkpoint schedule differs from lib decode_fwd")
    for layer_id in args.layers:
        layer = layers[layer_id]
        attention, moe = load_prefill_layer_weights(args.model_dir, layer_id, topology)
        bound = bind_decode_layer_weights(layer, attention, moe, kv_sources=kv_sources,
                                          index_sources=index_sources, c2a_sources=c2a_sources)
        for part in bound:
            tensor = part.value
            if not tensor.is_contiguous() or tensor.shape[0] != topology.world:
                raise ValueError(f"layer {layer_id} {part.name}: invalid EP placement")
            shape, dtype = geometry[part.name]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"layer {layer_id} {part.name}: got {tensor.shape}/{tensor.dtype}; "
                                 f"expected {shape}/{dtype}")
        print(f"layer={layer_id} mode={layer.mode} checked_slices={len(bound)}")
        del attention, moe, bound
        gc.collect()


if __name__ == "__main__":
    main()
