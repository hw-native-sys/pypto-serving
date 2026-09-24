# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real checkpoint row reads preserve token order and honor preparation limits."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pypto_serving.model.deepseek_v41.input_preparation import lookup_token_embeddings
from pypto_serving.model.deepseek_v41.weight_loader import V41WeightLoader
from pypto_serving.model.deepseek_v41.weight_spec import backbone_weight_specs


@pytest.fixture
def embedding_checkpoint(tmp_path):
    raw = json.loads((Path(__file__).resolve().parents[4] /
                      "tests/fixtures/deepseek_v41/config.json").read_text(encoding="utf-8"))
    raw["text_config"].update(hidden_size=32, vocab_size=256, num_hidden_layers=1,
        num_attention_heads=8, head_dim=64, q_lora_rank=256, o_lora_rank=128, o_groups=4,
        n_routed_experts=2, moe_intermediate_size=256, index_n_heads=8, index_head_dim=32,
        kv_source_layer_ids=[0], index_source_layer_ids=[0], compress_ratios=[2])
    raw.update(bos_token_id=0, eos_token_id=1, pad_token_id=2)
    table = (torch.arange(256).reshape(-1, 1) + torch.arange(32).reshape(1, -1) / 32).bfloat16()
    save_file({"embed.weight": table}, str(tmp_path / "embedding.safetensors"))
    specs = backbone_weight_specs(raw)
    index = {name: "embedding.safetensors" if name == "embed.weight" else "not-opened.safetensors"
             for name in specs}
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index}), encoding="utf-8"
    )
    loaders = [V41WeightLoader(tmp_path, tp_size=2, tp_rank=rank) for rank in range(2)]
    return loaders, table


def test_lookup_restores_dp_order_and_deduplicates_reads(embedding_checkpoint, monkeypatch):
    loaders, table = embedding_checkpoint
    calls = []
    for loader in loaders:
        original = loader.load_rows
        def read(name, start, stop, original=original, rank=loader.tp_rank):
            calls.append((rank, name, start, stop))
            return original(name, start, stop)
        monkeypatch.setattr(loader, "load_rows", read)
    ids = torch.tensor([[130, 0, 129, 127, 128], [0, 131, 2, 1, 128]], dtype=torch.int32)
    result = lookup_token_embeddings(loaders[::-1], ids, max_rows_per_read=2)
    assert torch.equal(result, table[ids.long()])
    assert result.dtype == torch.bfloat16 and result.device.type == "cpu"
    assert calls == [(0, "embed.weight", 0, 2), (0, "embed.weight", 2, 3),
                     (0, "embed.weight", 127, 128), (1, "embed.weight", 0, 2),
                     (1, "embed.weight", 2, 4)]


def test_noncontiguous_token_order_and_owned_rows(embedding_checkpoint):
    loaders, table = embedding_checkpoint
    ids = torch.tensor([[3, 255], [128, 3]], dtype=torch.int64).T
    result = lookup_token_embeddings(loaders, ids)
    assert torch.equal(result, table[ids])
    result[0, 0].zero_()
    assert torch.equal(result[1, 1], table[3])
    assert torch.equal(lookup_token_embeddings(loaders, torch.tensor([3]))[0], table[3])


@pytest.mark.parametrize("shape", [(0,), (2, 0)])
def test_empty_tokens_require_no_payload(embedding_checkpoint, monkeypatch, shape):
    loaders, _ = embedding_checkpoint
    def forbidden(*args):
        raise AssertionError("empty lookup must not open a checkpoint")
    for loader in loaders:
        monkeypatch.setattr(loader, "load_rows", forbidden)
    result = lookup_token_embeddings(loaders, torch.empty(shape, dtype=torch.int64), max_prepare_bytes=1)
    assert result.shape == (*shape, 32)
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("ids", [torch.tensor([-1]), torch.tensor([256]), torch.tensor([1.0]),
                                  torch.tensor([True]), torch.tensor(1)])
def test_invalid_token_ids_fail_before_read(embedding_checkpoint, monkeypatch, ids):
    loaders, _ = embedding_checkpoint
    def forbidden(*args):
        raise AssertionError("invalid IDs must fail before checkpoint I/O")
    for loader in loaders:
        monkeypatch.setattr(loader, "load_rows", forbidden)
    with pytest.raises(ValueError, match="token_ids"):
        lookup_token_embeddings(loaders, ids)


def test_incomplete_or_mixed_shards_rejected(embedding_checkpoint, tmp_path):
    loaders, _ = embedding_checkpoint
    for values in ([], loaders[:1], [loaders[0], loaders[0]], [*loaders, *loaders]):
        with pytest.raises(ValueError, match="TP shard"):
            lookup_token_embeddings(values, torch.tensor([0]))
    other = tmp_path / "different_checkpoint"
    other.mkdir()
    for filename in ("config.json", "model.safetensors.index.json"):
        (other / filename).write_bytes((tmp_path / filename).read_bytes())
    mixed = V41WeightLoader(other, tp_size=2, tp_rank=1)
    with pytest.raises(ValueError, match="same checkpoint"):
        lookup_token_embeddings([loaders[0], mixed], torch.tensor([0]))


def test_preparation_budget_rejected_before_payload(embedding_checkpoint, monkeypatch):
    loaders, _ = embedding_checkpoint
    def forbidden(*args):
        raise AssertionError("preparation budget must be checked before I/O")
    for loader in loaders:
        monkeypatch.setattr(loader, "load_rows", forbidden)
    with pytest.raises(ValueError, match="preparation.*budget"):
        lookup_token_embeddings(loaders, torch.tensor([0, 1, 128, 129]), max_prepare_bytes=1)


def test_loader_budget_is_preserved(embedding_checkpoint):
    loaders, _ = embedding_checkpoint
    loaders[0].max_load_bytes = 1
    with pytest.raises(ValueError, match="weight load.*budget"):
        lookup_token_embeddings(loaders, torch.tensor([0]))


@pytest.mark.parametrize("options", [{"max_prepare_bytes": 0}, {"max_prepare_bytes": True},
                                      {"max_rows_per_read": 0}, {"max_rows_per_read": 1.5}])
def test_invalid_limits(embedding_checkpoint, options):
    with pytest.raises(ValueError, match="positive integer"):
        lookup_token_embeddings(embedding_checkpoint[0], torch.tensor([0]), **options)
