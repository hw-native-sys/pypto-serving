# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill capacity must not inherit decode's fixed row extent."""

from types import SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41 import npu_runner as runner


@pytest.mark.parametrize("dp", [4, 8])
def test_prefill_capacity_is_independent_of_decode(dp):
    prefill = runner.SegmentTopology.for_prefill(dp=dp, max_tokens=4096)
    decode = runner.SegmentTopology.for_decode(dp=dp)
    assert prefill.capacity == 4096
    assert decode.capacity == 80
    assert prefill.world == decode.world == dp


@pytest.mark.parametrize("tokens", [0, -1, 4097, True])
def test_prefill_capacity_rejects_invalid_extents(tokens):
    with pytest.raises(ValueError, match="prefill capacity"):
        runner.SegmentTopology.for_prefill(dp=4, max_tokens=tokens)


def test_engram_lookup_accepts_decode_with_smaller_row_capacity():
    host = object.__new__(runner.HostEngramResources)
    host.lookup = torch.zeros(4, 2, 128, 3)
    host.host = SimpleNamespace(lookup=lambda *args, **kwargs: torch.ones(2, 80, 3))
    request = SimpleNamespace(partition=2, token_ids=(7,), start=100, request_id="req")
    step = SimpleNamespace(requests=(request,))
    target = torch.zeros(4, 2, 80, 3)
    host.stage_lookup(target, step, runner.SegmentTopology.for_decode(dp=4))
    assert target[2, :, 0].eq(1).all()
    assert target.sum() == 6


@pytest.mark.parametrize("dp,budget,expected", [(4, 4096, 4096), (8, 8192, 4096), (4, 80, 80)])
def test_cli_schedules_complete_prefill_independently_of_decode(monkeypatch, tmp_path, dp, budget, expected):
    import pypto_serving.cli.main as cli
    from pypto_serving.config.types import KVCacheGroupSpec, KVCacheSpec
    from pypto_serving.model.deepseek_v41.encoding import V41TextConfig
    from pypto_serving.model.deepseek_v41 import weight_loader

    monkeypatch.setattr(V41TextConfig, "from_dict", lambda raw: SimpleNamespace(max_position_embeddings=8192))
    monkeypatch.setattr(weight_loader, "plan_layers", lambda raw: ())
    group = KVCacheGroupSpec("window", (0,), KVCacheSpec(128, 128), 64, num_partitions=dp)
    bindings = SimpleNamespace(
        segment_topology=runner.SegmentTopology.for_decode(dp=dp),
        cache_groups=(group,), require=lambda *args: None,
    )
    monkeypatch.setattr(runner, "load_composite_bindings", lambda *args, **kwargs: bindings)
    args = cli.build_parser().parse_args([
        "--model", str(tmp_path), "--platform", "a5", "--block-size", "128",
        "--dp", str(dp), "--ep", str(dp), "--tp", "1", "--max-model-len", "8192",
        "--max-num-seqs", "16", "--max-num-batched-tokens", str(budget),
        "--long-prefill-token-threshold", "16",
    ])
    config = cli._build_v41_engine_config(args, str(tmp_path), {}, tuple(range(dp)))
    assert config.max_num_scheduled_tokens == expected
    assert config.runtime_config.max_prefill_tokens_per_request == expected
    assert not config.enable_chunk_prefill
    assert config.long_prefill_token_threshold == 0


def test_prefill_compilation_keeps_decode_moe_extent(monkeypatch):
    loaded = []
    monkeypatch.setattr(runner, "load_segment_modules", lambda root, topology: loaded.append(topology))
    entry = SimpleNamespace(param_names=("embedding",))
    module = SimpleNamespace(CAPACITY=4096, RANKS=4, l3_prefill_fwd=entry)
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: module)
    compiler = SimpleNamespace(compile=lambda name, function: "program")
    program, names = runner.compile_prefill_fwd(
        compiler, "unused", runner.SegmentTopology.for_prefill(dp=4, max_tokens=4096))
    assert loaded == [runner.SegmentTopology.for_decode(dp=4)]
    assert program == "program"
    assert names == ("embedding",)


def test_large_prefill_redistributes_heap_without_growing_residency():
    short = runner.prefill_ring_heap(80)
    long = runner.prefill_ring_heap(4096)
    assert sum(long) == sum(short) == 10 << 30
    assert long[1] > short[1]
    assert min(long) >= 2 << 30
