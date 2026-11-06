# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8 family routing and checkpoint-to-kernel contracts."""

import ast
import json
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model import model_loader
from pypto_serving.model.model_family import detect_model_family
from pypto_serving.model.qwen3_8_27b.cache import build_qwen38_cache_group_specs
from pypto_serving.model.qwen3_8_27b import npu_executor as executor_module
from pypto_serving.model.qwen3_8_27b.npu_runner import Qwen38Layout
from pypto_serving.model.qwen3_8_27b.task_args import (
    DECODE_ARGS,
    EMBEDDING_ARGS,
    FINAL_HEAD_ARGS,
    PREFILL_ARGS,
    decode_task_args,
    prefill_task_args,
)
from pypto_serving.model.qwen3_8_27b.weight_spec import (
    QWEN38_COMMON_RULES,
    QWEN38_GLOBAL_RULES,
    pack_qwen38_layer,
    qwen38_layer_weight_names,
    stage_qwen38_weights,
)


def _config():
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "max_position_embeddings": 262144,
        },
    }


def test_hybrid_config_routes_away_from_qwen14b():
    assert detect_model_family(_config()) == "qwen3_8_27b"
    assert detect_model_family({"model_type": "qwen3", "hidden_size": 5120}) == "qwen"
    assert detect_model_family({"model_type": "deepseek_v4"}) == "deepseek_v4"


def test_loader_claims_quantized_index_and_keeps_weights_lazy(tmp_path, monkeypatch):
    config = _config()
    config["text_config"]["rope_parameters"] = {"rope_type": "default", "rope_theta": 1_000_000.0}
    (tmp_path / "config.json").write_text(json.dumps(config))
    names = {rule.source for rule in QWEN38_GLOBAL_RULES}
    for layer in range(64):
        names.update(qwen38_layer_weight_names(layer))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in names}})
    )
    tokenizer = SimpleNamespace(bos_token_id=None, eos_token_id=None, pad_token_id=None)
    monkeypatch.setattr(model_loader, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    loaded = model_loader.ModelLoader().load("qwen38", str(tmp_path))
    assert loaded.config.head_dim == 256
    assert loaded.config.num_hidden_layers == 64
    assert loaded.config.rope_theta == 1_000_000.0
    assert loaded.runtime_model.extra["family"] == "qwen3_8_27b"
    assert loaded.runtime_model.embed_tokens.numel() == 0
    assert len(loaded.layer_specs) == 64

    config["text_config"]["linear_num_value_heads"] = 32
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="linear_num_value_heads"):
        model_loader.ModelLoader().load("qwen38", str(tmp_path))


@pytest.mark.parametrize(("name", "value", "message"), [
    ("rms_norm_eps", 1e-5, "rms_norm_eps"),
    ("partial_rotary_factor", 0.5, "partial_rotary_factor"),
    ("attention_bias", True, "attention_bias"),
    ("hidden_act", "gelu", "hidden_act"),
    ("layer_types", ["full_attention"] * 64, "layer_types"),
    ("rope_parameters", {"rope_type": "linear"}, "default text RoPE"),
    ("rope_parameters", {"partial_rotary_factor": 0.5}, "partial_rotary_factor"),
    ("rope_parameters", {"rope_theta": 0}, "rope_theta"),
])
def test_loader_rejects_options_not_implemented_by_fixed_kernels(tmp_path, name, value, message):
    config = _config()
    config["text_config"][name] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text('{"weight_map":{}}')
    with pytest.raises(ValueError, match=message):
        model_loader.ModelLoader().load("qwen38", str(tmp_path))


def _tiny_tensor(name: str, layer: int) -> torch.Tensor:
    suffix = name.split(f"layers.{layer}.", 1)[-1]
    value = float(layer + 1)
    if suffix.endswith("conv1d.weight"):
        return torch.full((2, 1, 3), value, dtype=torch.bfloat16)
    if suffix.endswith("in_proj_a.weight") or suffix.endswith("in_proj_b.weight"):
        return torch.full((1, 2), value, dtype=torch.bfloat16)
    if suffix.endswith("weight_scale"):
        return torch.full((2, 1), value)
    if suffix.endswith("layernorm.weight") or suffix.endswith("norm.weight"):
        return torch.full((2,), value, dtype=torch.bfloat16)
    if suffix.endswith("A_log") or suffix.endswith("dt_bias"):
        return torch.full((2,), value, dtype=torch.bfloat16)
    return torch.full((2, 2), int(layer + 1), dtype=torch.int8)


@pytest.mark.parametrize("workers", [1, 4])
def test_type_local_stacks_and_decode_projection_fusion(workers):
    class TinyStore:
        def load_many(self, names):
            return {name: _tiny_tensor(name, int(name.split("layers.")[1].split(".")[0]))
                    for name in names}

    store = TinyStore()
    packed = pack_qwen38_layer(3, store.load_many(qwen38_layer_weight_names(3)))
    assert packed["decode_w_qkv"].shape == (6, 2)
    assert packed["decode_w_qkv_scale"].shape == (1, 6)
    assert torch.equal(packed["decode_w_qkv"][:2], packed["w_q"])

    slabs = stage_qwen38_weights(store, workers=workers)
    assert slabs["w_qkv"].shape == (48 * 2, 2)
    assert slabs["w_q"].shape == (16 * 2, 2)
    assert slabs["w_gate"].shape == (64 * 2, 2)
    assert slabs["conv_w"].shape == (48 * 3, 2)
    assert slabs["w_ab"].shape == (48 * 2, 2)
    assert slabs["w_qkv"][0, 0].item() == 1
    assert slabs["w_qkv"][-1, 0].item() == 63
    assert slabs["w_q"][0, 0].item() == 4
    assert slabs["w_q"][-1, 0].item() == 64
    assert slabs["w_gate"][-1, 0].item() == 64
    for rule in QWEN38_COMMON_RULES:
        by_layer = slabs[rule.name].reshape(64, -1)
        expected = torch.arange(1, 65, dtype=by_layer.dtype)[:, None].expand_as(by_layer)
        assert torch.equal(by_layer, expected), rule.name


def test_cache_row_geometry_and_task_arg_order():
    layout = Qwen38Layout(max_seq_len=8192, max_batch_size=16,
                          prefill_tokens=2048, num_kv_pages=16 * 64)
    assert layout.capacity_slots == 16
    assert layout.state_row(1, 0) == 48
    assert layout.state_row(1, 4) == 51
    assert layout.attention_layer_base(7) == layout.kv_layer_stride
    assert layout.conv_pool_shape == ((16 + 1) * 48 + 16, 3, 10240)
    assert layout.recurrent_pool_shape == ((16 + 1) * 48 + 16, 48, 128, 128)
    assert layout.decode_scratch_kv_page(0) == 16 * 64
    assert layout.decode_scratch_state_row(0) == (16 + 1) * 48
    assert prefill_task_args(layout, lambda name: name).names == PREFILL_ARGS
    assert decode_task_args(layout, lambda name: name).names == DECODE_ARGS


def test_executor_builds_all_kernel_entries_without_loading_full_checkpoint(monkeypatch):
    class TinyStore:
        def __init__(self, **_kwargs):
            pass

        def load_many(self, names):
            result = {}
            for name in names:
                if "layers." in name:
                    layer = int(name.split("layers.")[1].split(".")[0])
                    result[name] = _tiny_tensor(name, layer)
                elif name.endswith("norm.weight"):
                    result[name] = torch.ones(2, dtype=torch.bfloat16)
                else:
                    result[name] = torch.ones(4, 2, dtype=torch.bfloat16)
            return result

    class Compiler:
        def __init__(self):
            self.names = []

        def compile(self, name, kernel, **_kwargs):
            assert kernel is not None
            self.names.append(name)
            return name

    monkeypatch.setattr(executor_module, "Qwen38WeightStore", TinyStore)
    compiler = Compiler()
    executor = SimpleNamespace(_compiler=compiler, _use_compile_cache=False,
                               _validate_model=executor_module.Qwen38PyptoExecutor._validate_model)
    runtime = model_loader.RuntimeConfig(
        page_size=128, max_seq_len=2048, max_batch_size=16,
        max_num_batched_tokens=2048, max_prefill_tokens_per_request=2048,
        kv_cache_groups=build_qwen38_cache_group_specs(2048),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=5120, vocab_size=248320,
                               num_hidden_layers=64, head_dim=256),
        runtime=runtime,
        extra={"family": "qwen3_8_27b", "model_dir": "unused", "weight_map": {}},
    )
    compiled = executor_module.Qwen38PyptoExecutor._compile_model(executor, model)
    assert set(compiled.prefill_by_rows) == set(range(1, 9))
    assert compiler.names[:8] == [
        f"qwen38_prefill_text_packed_t2048_r{rows}_p16_k34816_s832" for rows in range(1, 9)
    ]
    assert compiler.names[8:] == [
        "qwen38_decode_fwd_b16_g16_p16_k557056_s832",
        "qwen38_decode_fwd_b16_g16_p16_k557056_s832_r1_h2",
        "qwen38_decode_fwd_b16_g16_p16_k557056_s832_r2_h2",
        "qwen38_decode_embedding", "qwen38_decode_final_head",
    ]
    assert set(compiled.decode_fwd_by_rows) == {1, 2, 16}
    assert compiled.layout.kv_pool_shape == (16 * (16 * 16 + 16) * 128, 1024)
    assert compiled.weights["decode_w_qkv"].shape == (16 * 6, 2)
    assert set(compiled.globals) == {"embed_weight", "final_norm_w", "lm_w"}


def test_rule_names_and_task_signatures_match_pypto_lib():
    kernel_dir = executor_module.find_qwen38_kernel_dir()
    root = kernel_dir.parents[1]

    import sys

    sys.path.insert(0, str(root))
    try:
        from models.qwen3_8_27b.contract import layer_weight_specs

        for layer in range(64):
            expected = {spec.name for spec in layer_weight_specs(layer) if not spec.is_zero_offset}
            assert set(qwen38_layer_weight_names(layer)) == expected
    finally:
        sys.path.pop(0)

    sources = {
        "prefill_text_fwd.py": {"qwen3_prefill_text_fwd": PREFILL_ARGS, "l3_prefill_fwd": PREFILL_ARGS},
        "decode_fwd.py": {"qwen3_decode_fwd": DECODE_ARGS, "l3_decode_fwd": DECODE_ARGS},
        "embedding.py": {"_qwen3_embedding": EMBEDDING_ARGS, "l3_embedding": EMBEDDING_ARGS},
        "rms_lm_head.py": {"_qwen3_rms_lm_head": FINAL_HEAD_ARGS, "l3_rms_lm_head": FINAL_HEAD_ARGS},
    }
    for filename, expected_functions in sources.items():
        tree = ast.parse((kernel_dir / filename).read_text())
        found = {
            node.name: tuple(arg.arg for arg in node.args.args)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in expected_functions
        }
        assert found == expected_functions


def test_group_specs_are_the_runtime_contract():
    kv, state = build_qwen38_cache_group_specs(8192)
    assert (kv.name, state.name) == ("attn_kv", "gdn_state")


def test_kernel_capabilities_match_dispatch_geometry():
    contract = executor_module.load_qwen38_serving_contract()
    layout = Qwen38Layout(4096, 2, 2048, 64)
    kv, _ = build_qwen38_cache_group_specs(layout.max_seq_len)
    assert contract.kv_page_tokens == kv.spec.block_size
    assert contract.decode_batch_tile == layout.decode_batch_pad
    assert contract.prefill_tile_tokens == layout.prefill_tokens
    assert contract.prefill_request_alignment * Qwen38Layout(4096, 16, 2048, 512).max_prefill_rows == 2048


def test_kernel_imports_restore_other_models_on_success_and_failure(tmp_path, monkeypatch):
    import sys

    other_config = SimpleNamespace(marker="other-model")
    monkeypatch.setitem(sys.modules, "config", other_config)
    monkeypatch.delitem(sys.modules, "cleanup_test_entry", raising=False)
    (tmp_path / "config.py").write_text("marker = 'qwen'\n")
    (tmp_path / "cleanup_test_entry.py").write_text("import config\nmarker = config.marker\n")
    original_path = list(sys.path)
    for fail in (False, True):
        try:
            with executor_module._qwen38_import_context(tmp_path):
                entry = executor_module.load_qwen38_kernel_module("cleanup_test_entry", tmp_path)
                assert entry.marker == "qwen"
                if fail:
                    raise RuntimeError("compile failed")
        except RuntimeError as exc:
            assert str(exc) == "compile failed"
        assert sys.modules["config"] is other_config
        assert "cleanup_test_entry" not in sys.modules
        assert sys.path == original_path
