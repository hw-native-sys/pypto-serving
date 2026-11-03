# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for V4.1 weight staging limits and the pinned kernel contract."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41 import weight_loader
from pypto_serving.model.common.weights.nz import unpack_nz
from pypto_serving.model.deepseek_v41.weight_loader import DecodeWeightBank, V41WeightLoader, _ByteBudget
from pypto_serving.model.deepseek_v41.weight_spec import backbone_weight_specs


def test_decode_weight_bank_matches_pinned_kernel_abi():
    lib = Path(__file__).resolve().parents[4] / "pypto-lib"
    source = lib / "models/deepseek_v4_1_flash/decode_fwd.py"
    module = ast.parse(source.read_text())
    entry = next(node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == "l3_decode_fwd")
    bank_option = next(node for node in ast.walk(module)
                       if isinstance(node, ast.Call) and any(
                           isinstance(argument, ast.Constant) and argument.value == "--weight-bank-size"
                           for argument in node.args))
    bank_default = next(keyword.value for keyword in bank_option.keywords if keyword.arg == "default")
    assert ast.unparse(bank_default) == "C.FLASH.num_hidden_layers"
    DecodeWeightBank.validate_abi([argument.arg for argument in entry.args.args])
    for argument in entry.args.args:
        if argument.arg in ("c1a_compressor_wkv", "c2a_compressor_wkv", "c2a_compressor_wgate"):
            assert ast.unparse(argument.annotation.slice.elts[1]) == "pl.BF16"
    arguments = {argument.arg: argument for argument in entry.args.args}
    mismatches = []
    for fields, axis, separate in (
        (weight_loader._FLAT_LAYER, "WEIGHT_BANK_SIZE", False),
        (weight_loader._AXIS_LAYER, "WEIGHT_BANK_SIZE", True),
        (weight_loader._FLAT_KV, "KV_SOURCE_COUNT", False),
        (weight_loader._FLAT_INDEX, "INDEX_SOURCE_COUNT", False),
        (weight_loader._AXIS_INDEX, "INDEX_SOURCE_COUNT", True),
        (weight_loader._AXIS_C2A, "C2A_SOURCE_COUNT", True),
    ):
        for name in sorted(fields):
            leading = arguments[name].annotation.slice.elts[0].elts[1]
            actual_separate = isinstance(leading, ast.Name) and leading.id == axis
            if actual_separate != separate or not any(
                isinstance(node, ast.Name) and node.id == axis for node in ast.walk(leading)
            ):
                mismatches.append((name, ast.unparse(leading), axis, separate))
    assert not mismatches, mismatches


def _tiny_weight_loader(ratio):
    text = {
        "hidden_size": 64, "moe_intermediate_size": 64, "num_hidden_layers": 1,
        "num_attention_heads": 4, "n_routed_experts": 1, "n_shared_experts": 1,
        "head_dim": 32, "q_lora_rank": 64, "o_lora_rank": 4, "o_groups": 1,
        "hc_mult": 2, "vocab_size": 32, "index_n_heads": 32, "index_head_dim": 32,
        "compress_ratios": [ratio], "kv_source_layer_ids": [0], "index_source_layer_ids": [0],
    }
    loader = V41WeightLoader.__new__(V41WeightLoader)
    loader.specs = backbone_weight_specs({
        "text_config": text, "quantization_config": {
            "quant_method": "fp8", "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0", "expert_dtype": "fp4",
        },
    })
    loader.tp_size, loader.tp_rank = 1, 0
    loader.ep_size, loader.ep_rank = 1, 0
    loader.max_load_bytes = 4 << 20
    return loader


@pytest.mark.parametrize("ratio,member", [
    (1, "compressor.wkv"), (2, "compressor.wkv"), (2, "compressor.wgate"),
    (2, "indexer.wk"), (2, "indexer.weights_proj"),
])
def test_dense_attention_payload_keeps_checkpoint_bf16_in_nz(ratio, member):
    loader = _tiny_weight_loader(ratio)
    source = torch.arange(32 * 64, dtype=torch.bfloat16).reshape(32, 64)
    loader._read = lambda *args, **kwargs: source.clone()
    bundle = loader.load(f"layers.0.attn.{member}.weight")
    assert bundle.weight.dtype == torch.bfloat16
    assert bundle.weight.is_contiguous()
    assert not torch.equal(bundle.weight, source.T)
    assert torch.equal(unpack_nz(bundle.weight), source.T)


def test_gate_payload_keeps_checkpoint_bf16():
    loader = _tiny_weight_loader(1)
    source = torch.arange(64, dtype=torch.bfloat16).reshape(1, 64)
    loader._read = lambda *args, **kwargs: source.clone()
    bundle = loader.load("layers.0.ffn.gate.weight")
    assert bundle.weight.dtype == torch.bfloat16
    assert bundle.weight.is_contiguous()
    assert torch.equal(bundle.weight.view(torch.uint8), source.view(torch.uint8))
    assert bundle.scale is None


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_gate_weight_dtype_matches_pinned_kernel(phase):
    lib = Path(__file__).resolve().parents[4] / "pypto-lib"
    source = lib / f"models/deepseek_v4_1_flash/{phase}_fwd.py"
    module = ast.parse(source.read_text())
    entry = next(node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == f"l3_{phase}_fwd")
    gate = next(argument for argument in entry.args.args if argument.arg == "gate_weight")
    assert ast.unparse(gate.annotation.slice.elts[1]) == "pl.BF16"


def test_prepacked_bank_rejects_fp32_compressor_format(tmp_path, monkeypatch):
    import safetensors

    sidecar = tmp_path / "weights.safetensors"
    sidecar.touch()
    reader = SimpleNamespace(metadata=lambda: {"format": "pypto-deepseek-v41-decode-slot-major-v1"})
    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: nullcontext(reader))
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda *args: 1.0)
    layout = weight_loader.DecodeWeightLayout(1, 1, 1, 1, world=1)
    assert weight_loader.load_prepacked_bank(tmp_path, layout, (), path=sidecar) is None


def _write_tiny_v2_sidecar(tmp_path, monkeypatch, *, include_legacy_lut, corrupt_lut=None,
                           gate_dtype=torch.bfloat16):
    from safetensors.torch import save_file

    # Tiny source-axis slices exercise file compatibility without model-sized allocations.
    layout = weight_loader.DecodeWeightLayout(1, 1, 1, 1, world=1)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "source.bin").write_bytes(b"checkpoint")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"weight": "source.bin"}}', encoding="utf-8",
    )
    fields = {
        name: torch.arange(4, dtype=torch.float32).reshape(
            (1, 2, 2) if name in weight_loader._SINGLE else (1, 1, 2, 2),
        ).clone()
        for name in weight_loader._ALL
    }
    fields["gate_weight"] = fields["gate_weight"].to(gate_dtype)
    if include_legacy_lut:
        lut = weight_loader._legacy_mxfp4_pair_lut().unsqueeze(0)
        if corrupt_lut == "shape":
            lut = lut[:, :, :-1].contiguous()
        elif corrupt_lut == "dtype":
            lut = lut.to(torch.int32)
        elif corrupt_lut == "values":
            lut[0, 0, 1] += 1
        fields["mxfp4_pair_lut"] = lut
    sidecar = tmp_path / "weights.safetensors"
    save_file(fields, sidecar, metadata={
        "format": weight_loader._FORMAT,
        "source_fingerprint": weight_loader.source_fingerprint(tmp_path, layout),
    })
    monkeypatch.setenv("PYPTO_V41_PREPACKED_WEIGHTS", str(sidecar))
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda *args: 0.0)
    return layout, fields


@pytest.mark.parametrize("include_legacy_lut", [False, True])
def test_cold_v2_sidecar_loads_only_active_fields(tmp_path, monkeypatch, include_legacy_lut):
    layout, fields = _write_tiny_v2_sidecar(
        tmp_path, monkeypatch, include_legacy_lut=include_legacy_lut,
    )
    bank = weight_loader.load_prepacked_bank(tmp_path, layout, tuple(weight_loader._ALL))
    assert bank.prepacked
    assert set(bank.tensors) == weight_loader._ALL
    assert "mxfp4_pair_lut" not in bank.tensors
    for name, tensor in bank.tensors.items():
        assert torch.equal(tensor, fields[name])


@pytest.mark.parametrize("corrupt_lut", ["shape", "dtype", "values"])
def test_v2_sidecar_rejects_invalid_legacy_lut(tmp_path, monkeypatch, corrupt_lut):
    layout, _ = _write_tiny_v2_sidecar(
        tmp_path, monkeypatch, include_legacy_lut=True, corrupt_lut=corrupt_lut,
    )
    with pytest.raises(ValueError, match="legacy MXFP4 LUT"):
        weight_loader.load_prepacked_bank(tmp_path, layout, tuple(weight_loader._ALL))


@pytest.mark.parametrize("include_legacy_lut", [False, True])
def test_v2_sidecar_rejects_fp32_router_weights(tmp_path, monkeypatch, include_legacy_lut):
    layout, _ = _write_tiny_v2_sidecar(
        tmp_path, monkeypatch, include_legacy_lut=include_legacy_lut, gate_dtype=torch.float32,
    )
    assert weight_loader.load_prepacked_bank(tmp_path, layout, tuple(weight_loader._ALL)) is None


def test_routed_projection_budget_bounds_in_flight_bytes():
    budget = _ByteBudget(10)
    active = 0
    peak = 0
    lock = threading.Lock()

    def task():
        nonlocal active, peak
        budget.acquire(6)
        try:
            with lock:
                active += 6
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 6
        finally:
            budget.release(6)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: task(), range(2)))
    assert peak == 6

    with pytest.raises(ValueError, match="requires 11 bytes"):
        budget.acquire(11)
