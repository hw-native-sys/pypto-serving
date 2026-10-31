# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41.dspark_weights import (
    DSPARK_EXPERT_SOURCE_TO_ABI,
    DSPARK_HEAD_SOURCE_TO_ABI,
    DSPARK_LAYER_SOURCE_TO_ABI,
    V41DSparkWeightContract,
    dspark_v41_layer_weight_names,
    dspark_v41_required_weight_names,
)


def _contract(names, *, hidden=4, markov_rank=2):
    class Store:
        def __init__(self):
            self.required = None

        def require(self, values):
            self.required = tuple(values)
            missing = set(values) - set(names)
            if missing:
                raise KeyError(sorted(missing)[0])

        def load_many(self, values):
            result = {}
            for name in values:
                if name.endswith("main_proj.weight"):
                    result[name] = torch.zeros((hidden, 3 * hidden), dtype=torch.bfloat16)
                elif name.endswith("main_norm.weight") or name.endswith("norm.weight"):
                    result[name] = torch.ones(hidden, dtype=torch.bfloat16)
                elif "markov_w" in name:
                    result[name] = torch.zeros((8, markov_rank), dtype=torch.bfloat16)
                elif "confidence_head" in name:
                    result[name] = torch.zeros((1, hidden + markov_rank), dtype=torch.float32)
                else:
                    result[name] = torch.zeros(1)
            return result

    loader = SimpleNamespace(store=Store(), text={"n_routed_experts": 2})
    return V41DSparkWeightContract(loader), loader


def test_required_contract_covers_three_layers_and_markov_heads():
    names = dspark_v41_required_weight_names(2)
    assert set(DSPARK_HEAD_SOURCE_TO_ABI).issubset(names)
    assert "mtp.0.attn.wq_a.weight" in names
    assert "mtp.1.ffn.experts.1.w3.scale" in names
    assert "mtp.2.ffn.shared_experts.w2.weight" in names
    assert len(names) == len(set(names))


def test_layer_names_are_bounded_to_one_mtp_layer():
    names = dspark_v41_layer_weight_names(1, 2)
    assert names
    assert all(name.startswith("mtp.1.") for name in names)
    assert not any(name.endswith("main_proj.weight") for name in names)
    with pytest.raises(ValueError, match="one of 0, 1 or 2"):
        dspark_v41_layer_weight_names(3, 2)


def test_head_mapping_transposes_main_projection_and_keeps_markov_shape():
    names = dspark_v41_required_weight_names(2)
    contract, _ = _contract(names)
    heads = contract.load_heads().tensors
    assert heads["main_proj_weight"].shape == (12, 4)
    assert heads["markov_w1"].shape == (8, 2)
    assert heads["confidence_weight"].shape == (1, 6)


def test_validate_index_rejects_missing_deferred_tensor():
    names = dspark_v41_required_weight_names(2)
    contract, _ = _contract(names[:-1])
    with pytest.raises(KeyError):
        contract.validate_index()


def test_v41_drafter_weight_contract_matches_lib_abi():
    """Keep the serving source map aligned with V4.1, not the older V4 ABI."""
    lib_file = Path(__file__).parents[5] / "pypto-lib" / "models" / "deepseek_v4_1_flash" / "dspark_drafter.py"
    if not lib_file.is_file():
        pytest.skip("sibling pypto-lib checkout is not available")
    tree = ast.parse(lib_file.read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "l3_dspark_drafter"
    )
    args = {arg.arg for arg in function.args.args}
    expected = {
        "main_proj_weight", "main_norm_weight", "embedding_weight",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
        "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale",
        "wkv", "wkv_scale", "kv_norm_weight", "attn_sink", "wo_a",
        "wo_a_scale", "wo_b", "wo_b_scale", "hc_ffn_fn", "hc_ffn_scale",
        "hc_ffn_base", "ffn_norm_weight", "gate_weight", "correction_bias",
        "routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale",
        "routed_w3", "routed_w3_scale", "shared_w1", "shared_w1_scale",
        "shared_w2", "shared_w2_scale", "shared_w3", "shared_w3_scale",
    }
    assert expected <= args
    assert {"hc_head_fn", "hc_head_scale", "hc_head_base", "tid2eid"}.isdisjoint(args)
    assert {"main_proj_weight", "main_norm_weight"} <= set(DSPARK_HEAD_SOURCE_TO_ABI.values())
    source_suffixes = {
        "attn_norm.weight", "ffn_norm.weight", "attn.wq_a.weight", "attn.wq_a.scale",
        "attn.wq_b.weight", "attn.wq_b.scale", "attn.wkv.weight", "attn.wkv.scale",
        "attn.q_norm.weight", "attn.kv_norm.weight", "attn.attn_sink", "attn.wo_a.weight",
        "attn.wo_a.scale", "attn.wo_b.weight", "attn.wo_b.scale", "hc_attn_fn",
        "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
        "ffn.gate.weight", "ffn.gate.bias", "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w1.scale", "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w2.scale", "ffn.shared_experts.w3.weight",
        "ffn.shared_experts.w3.scale",
    }
    required = set(dspark_v41_required_weight_names(2))
    assert {f"mtp.0.{suffix}" for suffix in source_suffixes} <= required

    markov_tree = ast.parse(
        (lib_file.parent / "dspark_markov.py").read_text(encoding="utf-8")
    )
    markov = next(
        node for node in ast.walk(markov_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "l3_markov_sample_from_hidden"
    )
    assert {
        "final_norm_weight", "lm_head_weight", "markov_w1", "markov_w2",
        "confidence_weight", "draft_ids", "confidences", "logits_epoch",
    } <= {arg.arg for arg in markov.args.args}


def _layer_contract(*, bad_suffix=None):
    text = {
        "n_routed_experts": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "head_dim": 8,
        "q_lora_rank": 16,
        "o_lora_rank": 4,
        "o_groups": 2,
        "moe_intermediate_size": 96,
        "hc_mult": 1,
    }
    names = dspark_v41_layer_weight_names(0, text["n_routed_experts"])

    def shape(suffix):
        hidden, heads, head_dim = text["hidden_size"], text["num_attention_heads"], text["head_dim"]
        q_lora, o_lora, groups = text["q_lora_rank"], text["o_lora_rank"], text["o_groups"]
        inter, hc = text["moe_intermediate_size"], text["hc_mult"]
        q_width, o_width = heads * head_dim, heads * head_dim // groups

        def ceil32(value):
            return (value + 31) // 32

        return {
            "attn_norm.weight": (hidden,), "ffn_norm.weight": (hidden,),
            "attn.wq_a.weight": (q_lora, hidden), "attn.wq_a.scale": (ceil32(q_lora), ceil32(hidden)),
            "attn.wq_b.weight": (q_width, q_lora), "attn.wq_b.scale": (ceil32(q_width), ceil32(q_lora)),
            "attn.wkv.weight": (head_dim, hidden), "attn.wkv.scale": (ceil32(head_dim), ceil32(hidden)),
            "attn.q_norm.weight": (q_lora,), "attn.kv_norm.weight": (head_dim,),
            "attn.attn_sink": (heads,), "attn.wo_a.weight": (groups * o_lora, o_width),
            "attn.wo_a.scale": (ceil32(groups * o_lora), ceil32(o_width)),
            "attn.wo_b.weight": (hidden, groups * o_lora),
            "attn.wo_b.scale": (ceil32(hidden), ceil32(groups * o_lora)),
            "hc_attn_fn": ((2 + hc) * hc, hc * hidden), "hc_attn_scale": (3,),
            "hc_attn_base": ((2 + hc) * hc,), "hc_ffn_fn": ((2 + hc) * hc, hc * hidden),
            "hc_ffn_scale": (3,), "hc_ffn_base": ((2 + hc) * hc,),
            "ffn.gate.weight": (2, hidden), "ffn.gate.bias": (2,),
            "ffn.shared_experts.w1.weight": (inter, hidden),
            "ffn.shared_experts.w1.scale": (ceil32(inter), ceil32(hidden)),
            "ffn.shared_experts.w2.weight": (hidden, inter),
            "ffn.shared_experts.w2.scale": (ceil32(hidden), ceil32(inter)),
            "ffn.shared_experts.w3.weight": (inter, hidden),
            "ffn.shared_experts.w3.scale": (ceil32(inter), ceil32(hidden)),
        }[suffix]

    class Store:
        def require(self, values):
            missing = set(values) - set(names)
            if missing:
                raise KeyError(sorted(missing)[0])

        def load_many(self, values):
            result = {}
            for name in values:
                suffix = name.removeprefix("mtp.0.")
                if suffix.startswith("ffn.experts."):
                    suffix = suffix.split(".", 3)[-1]
                    expert_shape = {
                        "w1.weight": (96, 32), "w1.scale": (96, 2),
                        "w2.weight": (64, 48), "w2.scale": (64, 3),
                        "w3.weight": (96, 32), "w3.scale": (96, 2),
                    }[suffix]
                    result[name] = torch.zeros(expert_shape)
                else:
                    actual = (1, 1) if suffix == bad_suffix else shape(suffix)
                    result[name] = torch.zeros(actual)
            return result

    loader = SimpleNamespace(store=Store(), text=text)
    return V41DSparkWeightContract(loader)


def test_load_layer_abi_maps_source_names_and_preserves_expert_indices():
    layer = _layer_contract().load_layer_abi(0)
    assert set(layer.tensors) == set(DSPARK_LAYER_SOURCE_TO_ABI.values())
    assert layer.tensors["wq_a"].shape == (16, 64)
    assert set(layer.routed_experts) == set(DSPARK_EXPERT_SOURCE_TO_ABI.values())
    assert layer.routed_experts["routed_w2"][1].shape == (64, 48)


def test_load_layer_abi_rejects_source_layout_mismatch():
    with pytest.raises(ValueError, match="attn.wq_a.weight"):
        _layer_contract(bad_suffix="attn.wq_a.weight").load_layer_abi(0)


def _packed_layer_contract():
    """Build a tiny but tile-aligned FP8/FP4 source contract."""
    text = {
        "n_routed_experts": 2,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "head_dim": 32,
        "q_lora_rank": 64,
        "o_lora_rank": 32,
        "o_groups": 2,
        "moe_intermediate_size": 256,
        "hc_mult": 1,
    }
    names = set(dspark_v41_layer_weight_names(0, text["n_routed_experts"]))
    names.update(DSPARK_HEAD_SOURCE_TO_ABI)

    def shape(suffix):
        hidden, heads, head_dim = text["hidden_size"], text["num_attention_heads"], text["head_dim"]
        q_lora, o_lora, groups = text["q_lora_rank"], text["o_lora_rank"], text["o_groups"]
        inter, hc = text["moe_intermediate_size"], text["hc_mult"]
        q_width, o_width = heads * head_dim, heads * head_dim // groups

        def ceil32(value):
            return (value + 31) // 32

        return {
            "attn_norm.weight": (hidden,), "ffn_norm.weight": (hidden,),
            "attn.wq_a.weight": (q_lora, hidden), "attn.wq_a.scale": (ceil32(q_lora), ceil32(hidden)),
            "attn.wq_b.weight": (q_width, q_lora), "attn.wq_b.scale": (ceil32(q_width), ceil32(q_lora)),
            "attn.wkv.weight": (head_dim, hidden), "attn.wkv.scale": (ceil32(head_dim), ceil32(hidden)),
            "attn.q_norm.weight": (q_lora,), "attn.kv_norm.weight": (head_dim,),
            "attn.attn_sink": (heads,), "attn.wo_a.weight": (groups * o_lora, o_width),
            "attn.wo_a.scale": (ceil32(groups * o_lora), ceil32(o_width)),
            "attn.wo_b.weight": (hidden, groups * o_lora),
            "attn.wo_b.scale": (ceil32(hidden), ceil32(groups * o_lora)),
            "hc_attn_fn": ((2 + hc) * hc, hc * hidden), "hc_attn_scale": (3,),
            "hc_attn_base": ((2 + hc) * hc,), "hc_ffn_fn": ((2 + hc) * hc, hc * hidden),
            "hc_ffn_scale": (3,), "hc_ffn_base": ((2 + hc) * hc,),
            "ffn.gate.weight": (2, hidden), "ffn.gate.bias": (2,),
            "ffn.shared_experts.w1.weight": (inter, hidden),
            "ffn.shared_experts.w1.scale": (ceil32(inter), ceil32(hidden)),
            "ffn.shared_experts.w2.weight": (hidden, inter),
            "ffn.shared_experts.w2.scale": (ceil32(hidden), ceil32(inter)),
            "ffn.shared_experts.w3.weight": (inter, hidden),
            "ffn.shared_experts.w3.scale": (ceil32(inter), ceil32(hidden)),
        }[suffix]

    class Store:
        def require(self, values):
            missing = set(values) - names
            if missing:
                raise KeyError(sorted(missing)[0])

        def load_many(self, values):
            result = {}
            for name in values:
                suffix = name.removeprefix("mtp.0.")
                if suffix.startswith("ffn.experts."):
                    suffix = suffix.split(".", 3)[-1]
                    expert_shape = {
                        "w1.weight": (256, 128), "w1.scale": (256, 8),
                        "w2.weight": (256, 128), "w2.scale": (256, 8),
                        "w3.weight": (256, 128), "w3.scale": (256, 8),
                    }[suffix]
                    dtype = torch.uint8 if suffix.endswith("weight") else torch.float8_e8m0fnu
                elif name.endswith("main_proj.weight"):
                    expert_shape, dtype = (256, 768), torch.bfloat16
                elif suffix in ("attn.q_norm.weight", "attn.kv_norm.weight"):
                    expert_shape, dtype = shape(suffix), torch.bfloat16
                elif name.endswith("main_norm.weight") or name.endswith("norm.weight"):
                    expert_shape, dtype = (256,), torch.bfloat16
                elif "markov_w" in name:
                    expert_shape, dtype = (8, 2), torch.bfloat16
                elif "confidence_head" in name:
                    expert_shape, dtype = (1, 258), torch.float32
                else:
                    expert_shape = shape(suffix)
                    if suffix.endswith(".scale"):
                        dtype = torch.float8_e8m0fnu
                    elif suffix.endswith(".weight") and suffix.startswith(
                        ("attn.wq_", "attn.wkv", "attn.wo_", "ffn.shared")
                    ):
                        dtype = torch.float8_e4m3fn
                    elif suffix.startswith(("hc_", "ffn.gate.bias", "attn.attn_sink")):
                        dtype = torch.float32
                    else:
                        dtype = torch.bfloat16
                result[name] = torch.zeros(expert_shape, dtype=dtype)
            return result

    loader = SimpleNamespace(
        store=Store(), text=text, ep_size=2, ep_rank=1, tp_size=1,
    )
    return V41DSparkWeightContract(loader)


def test_load_layer_packed_uses_only_one_ep_rank_and_lib_layouts():
    packed = _packed_layer_contract().load_layer_packed(0).tensors
    assert packed["wq_a"].shape == (256, 64)
    assert packed["wq_a_scale"].shape == (8, 64)
    assert packed["wo_a"].shape == (2, 128, 32)
    assert packed["wo_b"].shape == (64, 256)
    assert packed["routed_w1"].shape == (256, 128)
    assert packed["routed_w1_scale"].shape == (8, 256)
    assert packed["gate_weight"].dtype is torch.float32
