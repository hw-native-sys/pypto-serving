# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode weight-slot mapping against the published 40-layer schedule."""

from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from pypto_serving.model.deepseek_v41 import decode_weights
from pypto_serving.model.deepseek_v41.execution_plan import plan_layers
from pypto_serving.model.deepseek_v41.swa_segment import SegmentTopology


CONFIG = Path(__file__).resolve().parents[4] / "tests/fixtures/deepseek_v41/config.json"


def _weights(layer_id):
    value = torch.full((8, 1), layer_id, dtype=torch.int32)
    attention = {name: value for name in decode_weights._ATTENTION.values()}
    attention.update({name: value for name in (
        "compressor_norm_weight", "compressor_wkv", "compressor_wgate",
        "index_wk", "index_norm_weight", "index_wq_b", "index_wq_b_scale",
        "index_weights_proj",
    )})
    moe = {name: value for name in decode_weights._MOE.values()}
    moe["mxfp4_pair_lut"] = torch.zeros((8, 2, 256), dtype=torch.int16)
    moe["routed_w1"] = torch.zeros((8, 1, 1, 128), dtype=torch.uint8)
    return attention, moe


def test_decode_weight_slices_follow_layer_and_producer_order(monkeypatch, tmp_path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    calls = []

    def load(_model_dir, layer_id, _topology, *, max_bundle_bytes):
        calls.append((layer_id, max_bundle_bytes))
        return _weights(layer_id)

    monkeypatch.setattr(decode_weights, "load_prefill_layer_weights", load)
    chunks = list(decode_weights.iter_decode_weight_slices(
        tmp_path, SegmentTopology(tp=4, dp=2), max_bundle_bytes=1234,
    ))
    by_name = {}
    for chunk in chunks:
        by_name.setdefault(chunk.name, []).append(chunk)
    assert calls == [(layer, 1234) for layer in range(40)]
    assert set(by_name) == set(decode_weights._ATTENTION) | set(decode_weights._MOE) | {
        "mxfp4_pair_lut", "compressor_norm_weight", "c2a_compressor_wkv",
        "c2a_compressor_wgate", "c1a_compressor_wkv", "index_wk",
        "index_norm_weight", "index_wq_b", "index_wq_b_scale", "index_weights_proj",
    }
    assert [c.slot for c in by_name["routed_w1"]] == list(range(40))
    assert all(c.value.dtype == torch.uint8 for c in by_name["routed_w1"])
    assert [c.slot for c in by_name["compressor_norm_weight"]] == list(range(4))
    assert [c.slot for c in by_name["index_wq_b"]] == list(range(8))
    assert [c.slot for c in by_name["c2a_compressor_wkv"]] == list(range(3))
    assert len(by_name["c1a_compressor_wkv"]) == 1
    assert [int(c.value[0, 0]) for c in by_name["index_wk"][:4]] == [2, 8, 14, 20]
    assert [c.slot for c in by_name["index_wk"]] == list(range(8))
    assert all(not torch.count_nonzero(c.value) for c in by_name["index_wk"][4:])
    assert all(not torch.count_nonzero(c.value) for c in by_name["index_norm_weight"][4:])
    assert Counter(c.slot for c in by_name["mxfp4_pair_lut"]) == {0: 1}


def test_decode_weight_slices_reject_wrong_topology_before_payload_reads(monkeypatch, tmp_path):
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(decode_weights, "load_prefill_layer_weights",
                        lambda *_args, **_kwargs: pytest.fail("must not read weights"))
    with pytest.raises(ValueError, match="TP4/DP2/EP8"):
        list(decode_weights.iter_decode_weight_slices(tmp_path, SegmentTopology(tp=2, dp=2)))


def test_decode_weight_slices_reject_incomplete_bundle():
    layer = plan_layers(json.loads(CONFIG.read_text(encoding="utf-8")))[0]
    with pytest.raises(ValueError, match="complete Attention"):
        decode_weights.bind_decode_layer_weights(
            layer, {}, {}, kv_sources=(2, 8, 14, 20),
            index_sources=(2, 8, 14, 20, 24, 28, 32, 36), c2a_sources=(2, 8, 14),
        )
