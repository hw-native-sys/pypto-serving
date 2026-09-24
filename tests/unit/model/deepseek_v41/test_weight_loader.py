# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real safetensors reads, ownership and independent byte-layout checks."""

import json
from pathlib import Path
from contextlib import contextmanager

import pytest
import torch
from safetensors.torch import save_file
from safetensors import safe_open

from pypto_serving.model.deepseek_v41.weight_spec import backbone_weight_specs
from pypto_serving.model.deepseek_v41.weight_loader import V41WeightLoader
from pypto_serving.model.deepseek_v41.weight_packing import pack_fp4_tiles, pack_mx_scale


@pytest.fixture
def checkpoint(tmp_path):
    raw = json.loads((Path(__file__).resolve().parents[4] /
                      "tests/fixtures/deepseek_v41/config.json").read_text(encoding="utf-8"))
    raw["text_config"].update(hidden_size=256, vocab_size=256, num_hidden_layers=1,
        num_attention_heads=8, head_dim=64, q_lora_rank=256, o_lora_rank=128, o_groups=4,
        n_routed_experts=2, moe_intermediate_size=256, index_n_heads=8, index_head_dim=32,
        kv_source_layer_ids=[0], index_source_layer_ids=[0], compress_ratios=[2])
    raw.update(bos_token_id=0, eos_token_id=1, pad_token_id=2)
    specs = backbone_weight_specs(raw)
    tensors = {}
    types = {"BF16": torch.bfloat16, "F32": torch.float32, "I8": torch.int8,
             "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu}
    for name, spec in specs.items():
        size = 1
        for dim in spec.shape:
            size *= dim
        if spec.dtype == "F8_E8M0":
            value = (torch.arange(size) % 5 + 125).to(torch.uint8).view(types[spec.dtype])
        elif spec.dtype == "I8":
            value = (torch.arange(size) % 256).to(torch.uint8).view(torch.int8)
        else:
            value = ((torch.arange(size) % 13 - 6).float() / 4).to(types[spec.dtype])
        tensors[name] = value.reshape(spec.shape)
    save_file(tensors, str(tmp_path / "text.safetensors"))
    index = {name: "text.safetensors" for name in tensors}
    index.update({"layers.0.engram.embed.weight": "not-opened.safetensors",
                  "vision.weight": "not-opened.safetensors", "mtp.weight": "not-opened.safetensors"})
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}), encoding="utf-8")
    return tmp_path, raw, tensors


def logical_scales(packed):
    codes = packed.view(torch.uint8)
    groups, n = codes.shape
    # Independent physical address decoding; do not use an inverse of the packer.
    result = torch.empty_like(codes)
    flat = codes.flatten()
    for kg in range(groups):
        for col in range(n):
            index = (col//16)*(groups//2)*32 + (kg//2)*32 + (col%16)*2 + kg%2
            result[kg, col] = flat[index]
    return result


def test_index_and_ownership_do_not_read_payload(checkpoint):
    path, _, _ = checkpoint
    @contextmanager
    def forbidden(*args):
        raise AssertionError("constructor must not open any shard")
        yield
    loader = V41WeightLoader(path, ep_size=2, ep_rank=1, safe_open_fn=forbidden)
    assert all("engram" not in n and "experts.0." not in n for n in loader.names(0))
    assert any("experts.1." in n for n in loader.names(0))
    with pytest.raises(KeyError, match="supported text"):
        loader.load("layers.0.engram.embed.weight")
    with pytest.raises(ValueError, match="not owned"):
        loader.load("layers.0.ffn.experts.0.w1.weight")


def test_dense_and_fp8_tp_slices(checkpoint):
    path, _, tensors = checkpoint
    chunks = []
    for rank in range(2):
        loader = V41WeightLoader(path, tp_size=2, tp_rank=rank)
        embedding = loader.load("embed.weight")
        assert torch.equal(embedding.weight, tensors["embed.weight"][rank*128:(rank+1)*128])
        bundle = loader.load("layers.0.attn.wq_b.weight")
        chunks.append(bundle.weight)
        expected = tensors["layers.0.attn.wq_b.weight"][rank*256:(rank+1)*256].T
        assert torch.equal(bundle.weight.view(torch.uint8), expected.contiguous().view(torch.uint8))
        raw_scale = tensors["layers.0.attn.wq_b.scale"].view(torch.uint8)[rank*8:(rank+1)*8]
        assert torch.equal(logical_scales(bundle.scale), raw_scale.T.repeat_interleave(32, 1))
    assert torch.equal(torch.cat(chunks, dim=1).T.contiguous().view(torch.uint8),
                       tensors["layers.0.attn.wq_b.weight"].view(torch.uint8))


def test_input_axis_shard_keeps_scales_aligned(checkpoint):
    path, _, tensors = checkpoint
    result = V41WeightLoader(path, tp_size=2, tp_rank=1).load("layers.0.attn.wo_b.weight")
    expected = tensors["layers.0.attn.wo_b.weight"][:, 256:]
    assert torch.equal(result.weight.view(torch.uint8), expected.T.contiguous().view(torch.uint8))
    scales = tensors["layers.0.attn.wo_b.scale"].view(torch.uint8)[:, 8:]
    assert torch.equal(logical_scales(result.scale), scales.T.repeat_interleave(32, 1))


def test_wo_a_group_dequantization(checkpoint):
    path, _, tensors = checkpoint
    result = V41WeightLoader(path, tp_size=2, tp_rank=1).load("layers.0.attn.wo_a.weight")
    values = tensors["layers.0.attn.wo_a.weight"][256:].float()
    scales = torch.exp2(tensors["layers.0.attn.wo_a.scale"].view(torch.uint8)[8:].float() - 127)
    expected = (values * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)).bfloat16()
    assert result.weight.shape == (2, 128, 128)
    assert torch.equal(result.weight.flatten(0, 1), expected)
    assert result.scale is None


def test_dense_promotions_and_transpose(checkpoint):
    path, _, tensors = checkpoint
    loader = V41WeightLoader(path)
    for name in ("layers.0.hc_attn_fn", "layers.0.attn_norm.weight"):
        assert torch.equal(loader.load(name).weight, tensors[name])
    name = "layers.0.attn.compressor.wkv.weight"
    assert torch.equal(loader.load(name).weight, tensors[name].float().T)
    assert loader.load("layers.0.ffn.gate.weight").weight.dtype == torch.float32


def test_fp4_payload_is_reordered_without_loss(checkpoint):
    path, _, tensors = checkpoint
    name = "layers.0.ffn.experts.1.w1.weight"
    result = V41WeightLoader(path, ep_size=2, ep_rank=1).load(name)
    packed = result.weight.reshape(256, 128)
    raw = tensors[name].view(torch.uint8)
    # Address definition: adjacent output channels share one byte, K rows remain ordered.
    for k in (0, 1, 127, 128, 255):
        for n in (0, 1, 126, 127, 254, 255):
            expected = (int(raw[n, k//2]) >> (4*(k%2))) & 15
            actual = (int(packed[k, n//2]) >> (4*(n%2))) & 15
            assert actual == expected
    assert result.weight.dtype == torch.uint8
    expected_scale = tensors[name.replace(".weight", ".scale")].view(torch.uint8).T
    assert torch.equal(logical_scales(result.scale), expected_scale)


def test_fp4_multiple_tiles():
    n, k = 512, 768
    raw = (torch.arange(n*(k//2)) % 251).to(torch.uint8).reshape(n, k//2)
    result = pack_fp4_tiles(raw).reshape(n//256, k//256, 256, 128)
    for row in (0, 255, 256, 511):
        for col in (0, 255, 256, 511, 512, 767):
            expected = (int(raw[row, col//2]) >> (4*(col%2))) & 15
            byte = int(result[row//256, col//256, col%256, (row%256)//2])
            assert (byte >> (4*(row%2))) & 15 == expected


def test_budget_rejects_before_open(checkpoint):
    path, _, _ = checkpoint
    @contextmanager
    def forbidden(*args):
        raise AssertionError("budget must be checked before payload I/O")
        yield
    loader = V41WeightLoader(path, max_load_bytes=1, safe_open_fn=forbidden)
    with pytest.raises(ValueError, match="budget"):
        loader.load("layers.0.attn.wq_b.weight")


def test_slice_reader_never_calls_get_tensor(checkpoint):
    path, _, tensors = checkpoint
    reads = []
    @contextmanager
    def opener(path, device):
        with safe_open(str(path), framework="pt", device=device) as reader:
            class Reader:
                def get_tensor(self, name):
                    raise AssertionError("must use bounded slicing")
                def get_slice(self, name):
                    source = reader.get_slice(name)
                    class Slice:
                        def get_shape(self): return source.get_shape()
                        def get_dtype(self): return source.get_dtype()
                        def __getitem__(self, ranges):
                            reads.append((name, ranges))
                            return source[ranges]
                    return Slice()
            yield Reader()
    result = V41WeightLoader(path, tp_size=2, tp_rank=1, safe_open_fn=opener).load("embed.weight")
    assert reads == [("embed.weight", (slice(128, 256), slice(0, 256)))]
    result.weight.zero_()
    assert tensors["embed.weight"].count_nonzero() > 0


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_header_mismatch(checkpoint, change):
    path, _, _ = checkpoint
    value = torch.zeros(12) if change == "shape" else torch.zeros(256, 256, dtype=torch.float32)
    save_file({"embed.weight": value}, str(path / "text.safetensors"))
    with pytest.raises(ValueError, match="shape/dtype"):
        V41WeightLoader(path).load("embed.weight")


@pytest.mark.parametrize("options", [{"tp_size": 3}, {"ep_size": 3}, {"tp_rank": -1},
                                      {"ep_rank": True}, {"max_load_bytes": 0}])
def test_bad_topology(checkpoint, options):
    with pytest.raises(ValueError):
        V41WeightLoader(checkpoint[0], **options)


def test_missing_index_weight(checkpoint):
    path, _, _ = checkpoint
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["norm.weight"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(KeyError, match="norm.weight"):
        V41WeightLoader(path)


def test_scale_nan(checkpoint):
    path, _, tensors = checkpoint
    tensors["layers.0.attn.wq_a.scale"].view(torch.uint8)[0, 0] = 255
    save_file(tensors, str(path / "text.safetensors"))
    with pytest.raises(ValueError, match="non-finite E8M0"):
        V41WeightLoader(path).load("layers.0.attn.wq_a.weight")


def test_pack_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        pack_fp4_tiles(torch.zeros(32, 32, dtype=torch.uint8))
    with pytest.raises(ValueError):
        pack_mx_scale(torch.zeros(3, 16, dtype=torch.uint8))


@pytest.mark.parametrize("name", ["embed.weight", "head.weight"])
def test_vocabulary_row_chunks(checkpoint, name):
    path, _, tensors = checkpoint
    loader = V41WeightLoader(path, tp_size=2, tp_rank=1, max_load_bytes=(1 << 20) + 40000)
    with pytest.raises(ValueError, match="budget"):
        loader.load(name)
    chunk = loader.load_rows(name, 3, 7)
    assert torch.equal(chunk.weight, tensors[name][131:135])
    assert chunk.weight.dtype == torch.bfloat16
    for start, stop in ((-1, 2), (0, 129), (3, 3), (True, 3)):
        with pytest.raises(ValueError):
            loader.load_rows(name, start, stop)


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/absolute.safetensors"])
def test_index_rejects_path_escape(checkpoint, filename):
    path, _, _ = checkpoint
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["norm.weight"] = filename
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="inside model"):
        V41WeightLoader(path)
