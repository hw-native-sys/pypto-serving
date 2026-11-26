# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 directory loader topology and host Engram table residency."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pypto_serving.config.types import RuntimeConfig
from pypto_serving.model.deepseek_v41 import npu_executor
from pypto_serving.model.deepseek_v41.weight_loader import V41WeightLoader
from pypto_serving.model.deepseek_v41.weight_spec import backbone_weight_specs
from pypto_serving.model.model_loader import ModelLoadRequest


@pytest.fixture
def v41_request(tmp_path, monkeypatch):
    raw = {
        "model_type": "deepseek_v41",
        "text_config": {
            "model_type": "deepseek_v41_text", "num_hidden_layers": 40,
            "hidden_size": 64, "moe_intermediate_size": 32, "n_routed_experts": 8,
            "n_shared_experts": 1, "vocab_size": 128, "num_attention_heads": 2,
            "head_dim": 32, "hc_mult": 4, "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "q_lora_rank": 32,
            "o_lora_rank": 32, "o_groups": 1, "index_n_heads": 2, "index_head_dim": 32,
            "compress_ratios": [0] * 2 + [2] * 18 + [1] * 20,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "candidate_source_layer_id": 20,
        },
        "quantization_config": {
            "quant_method": "fp8", "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0", "expert_dtype": "fp4",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    weight_map = dict.fromkeys(backbone_weight_specs(raw), "unused.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    monkeypatch.delenv("PYPTO_V41_SHARED_SHARD_DIR", raising=False)
    monkeypatch.setattr(npu_executor, "load_tokenizer", lambda *args, **kwargs: object())
    return ModelLoadRequest("v41", str(tmp_path), RuntimeConfig(max_batch_size=10, max_seq_len=1024, page_size=128))


@pytest.mark.parametrize(("configured_dp", "dp", "slots"), [
    ("1", 1, 10), ("2", 2, 5), ("4", 4, 3), ("8", 8, 2), (None, 8, 2),
])
def test_directory_loader_builds_cache_groups_for_direct_dp(v41_request, monkeypatch, configured_dp, dp, slots):
    if configured_dp is None:
        monkeypatch.delenv("PYPTO_DSV41_DP_SIZE", raising=False)
    else:
        monkeypatch.setenv("PYPTO_DSV41_DP_SIZE", configured_dp)

    model = npu_executor.DeepSeekV41DirectoryLoader().load(v41_request)

    groups = model.runtime_model.runtime.kv_cache_groups
    assert tuple(group.name for group in groups) == ("window", "c2a", "c1a")
    assert all(group.num_partitions == dp and group.max_requests_per_partition == slots for group in groups)


@pytest.mark.parametrize("configured_dp", ["3", "invalid"])
def test_directory_loader_rejects_invalid_direct_dp(v41_request, monkeypatch, configured_dp):
    monkeypatch.setenv("PYPTO_DSV41_DP_SIZE", configured_dp)

    with pytest.raises(ValueError, match="PYPTO_DSV41_DP_SIZE must be 1, 2, 4 or 8"):
        npu_executor.DeepSeekV41DirectoryLoader().load(v41_request)


@pytest.fixture
def engram_loader(v41_request):
    model_dir = Path(v41_request.model_dir)
    index = model_dir / "model.safetensors.index.json"
    config = json.loads(index.read_text())
    for layer, rows in ((1, 32), (14, 64)):
        tensors = {}
        shard = f"engram-{layer}.safetensors"
        for field, width, dtype in (("weight", 256, torch.float8_e4m3fn), ("scale", 8, torch.float8_e8m0fnu)):
            name = f"layers.{layer}.engram.embed.{field}"
            values = torch.arange(rows, dtype=torch.uint8)[:, None] + torch.arange(width, dtype=torch.uint8)
            tensors[name] = values.view(dtype)
            config["weight_map"][name] = shard
        save_file(tensors, model_dir / shard)
    index.write_text(json.dumps(config))
    return V41WeightLoader(model_dir)


def _locked_kib():
    return int(Path("/proc/self/status").read_text().split("VmLck:")[1].split()[0])


def test_resident_engram_preserves_lookup_and_releases_locks(engram_loader):
    loader = engram_loader
    names = [f"layers.{layer}.engram.embed.{field}" for layer in (1, 14) for field in ("weight", "scale")]
    rows = torch.tensor([[2, 7, 2], [0, 17, 4]])
    expected = [loader.tensor_rows(name, rows).view(torch.uint8) for name in names]
    before = _locked_kib()
    with loader.lock_engram_tables((1, 14)):
        assert _locked_kib() - before >= (32 + 64) * 264 // 1024
        for name, reference in zip(names, expected):
            assert torch.equal(loader.tensor_rows(name, rows).view(torch.uint8), reference)
    assert _locked_kib() == before
    assert not loader._row_tensors


def test_resident_engram_releases_partial_locks(engram_loader):
    before = _locked_kib()
    (engram_loader.model_dir / "engram-14.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        engram_loader.lock_engram_tables((1, 14))
    assert _locked_kib() == before
    assert not engram_loader._row_tensors
