# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preparation must preserve layer ownership without allocating a device backend."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pypto_serving.model.deepseek_v41.execution_plan import RankPlacement, V41ExecutionPlan, plan_layers
from pypto_serving.model.deepseek_v41.weight_spec import backbone_weight_specs


@pytest.fixture
def raw():
    return json.loads((Path(__file__).resolve().parents[4] /
        "tests/fixtures/deepseek_v41/config.json").read_text(encoding="utf-8"))


def test_all_backbone_modes_and_producers(raw):
    layers = plan_layers(raw)
    assert len(layers) == 40
    assert [p.mode for p in layers] == (
        ["swa"] * 2 + (["c2a_full"] + ["c2a_reuse"] * 5) * 3 +
        ["c1a_full"] + ["c1a_reuse"] * 3 + (["c1a_reindex"] + ["c1a_reuse"] * 3) * 4
    )
    assert [(p.kv_source, p.index_source) for p in layers[18:26]] == [
        (14, 14), (14, 14), (20, 20), (20, 20), (20, 20), (20, 20), (20, 24), (20, 24)]
    assert all(p.candidate_source == 20 for p in layers[20:])
    assert all(p.candidate_source is None for p in layers[:20])


@pytest.mark.parametrize("rank", range(8))
def test_rank_coordinates(rank):
    p = RankPlacement(rank)
    assert (p.tp_rank, p.dp_rank, p.tp_group_start) == (rank % 4, rank // 4, rank // 4 * 4)


@pytest.mark.parametrize("kwargs", [{"rank": -1}, {"rank": 8}, {"rank": True},
    {"rank": 0, "tp_size": 0}, {"rank": 0, "ep_size": 4}])
def test_invalid_topology(kwargs):
    with pytest.raises(ValueError):
        RankPlacement(**kwargs)


@pytest.mark.parametrize("field,value", [
    ("kv_source_layer_ids", [8, 14, 20]),
    ("index_source_layer_ids", [2, 8, 14, 24, 28, 32, 36]),
    ("kv_source_layer_ids", [2, 2, 8, 14, 20]),
    ("kv_source_layer_ids", [0, 2, 8, 14, 20]),
    ("index_source_layer_ids", [2, 3, 8, 14, 20, 24, 28, 32, 36]),
    ("candidate_source_layer_id", 24),
    ("candidate_source_layer_id", True),
])
def test_bad_source_metadata_rejected(raw, field, value):
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        plan_layers(raw)


def test_metadata_only_then_bounded_real_read(raw, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(raw))
    names = backbone_weight_specs(raw)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "part.safetensors" for name in names}}))
    # No shard exists. Construction and name inspection must not read payloads.
    plan = V41ExecutionPlan(tmp_path, RankPlacement(7))
    assert plan.layer(24).kv_source == 20
    assert plan.layer(24).index_source == 24
    reuse = plan.weight_names(21)
    assert not any("compressor" in n or "indexer" in n or n.endswith(".scale") for n in reuse)
    assert "layers.21.ffn.experts.336.w1.weight" in reuse
    assert "layers.21.ffn.experts.383.w3.weight" in reuse
    assert "layers.21.ffn.experts.335.w1.weight" not in reuse
    reindex = plan.weight_names(24)
    assert "layers.24.attn.indexer.wq_b.weight" in reindex
    assert not any("compressor" in n or "indexer.wk." in n for n in reindex)
    with pytest.raises(ValueError, match="not owned"):
        plan.load_weight(21, "layers.20.attn.compressor.wkv.weight")
    with pytest.raises(ValueError):
        plan.layer(-1)
    name = "layers.21.hc_attn_scale"
    expected = torch.tensor([1., 2., 3.])
    save_file({name: expected}, tmp_path / "part.safetensors")
    bundle = plan.load_weight(21, name)
    assert torch.equal(bundle.weight, expected)
    assert (bundle.tp_rank, bundle.ep_rank) == (3, 7)
