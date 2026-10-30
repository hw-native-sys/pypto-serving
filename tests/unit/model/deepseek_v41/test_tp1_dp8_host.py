# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for the V4.1 TP1/DP8/EP8 serving placement."""

from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from pypto_serving.model.deepseek_v41 import npu_runner
from pypto_serving.model.deepseek_v41.npu_runner import (
    CacheBufferSpec,
    DecodeCachePlan,
    DecodeCachePools,
    MissingCompositeInterface,
    PrefillMetadataResources,
    SegmentTopology,
    load_final_output_host_resources,
    make_cache_page_reset_buffers,
    pad_single_decode,
    plan_decode_cache_pools,
    prepare_compressed_metadata,
    prepare_segment_inputs,
    selected_decode_topology,
    validate_page_tables,
)
from pypto_serving.model.deepseek_v41.task_args import (
    ForwardStep,
    RequestLedger,
    RequestSlice,
    build_cache_groups,
)
from pypto_serving.model.deepseek_v41.weight_loader import LayerPlan, RankPlacement
from pypto_serving.model.deepseek_v41.weight_loader import DecodeWeightLayout, source_fingerprint


def _layers():
    kv_sources = (2, 8, 14, 20)
    index_sources = (2, 4, 6, 8, 10, 12, 14, 20)
    return tuple(
        LayerPlan(
            layer_id=layer,
            mode=("swa" if layer < 2 else
                  "c1a_full" if layer == 20 else
                  "c2a_full" if layer in kv_sources else
                  "c2a_reuse" if layer < 20 else "c1a_reuse"),
            kv_source=max((source for source in kv_sources if source <= layer), default=None),
            index_source=max((source for source in index_sources if source <= layer), default=None),
            candidate_source=20 if layer >= 20 else None,
        )
        for layer in range(40)
    )


def _pages():
    return {"window": (0,), "c2a": (0,), "c1a": (0,)}


def test_tp1_dp8_partition_and_page_ownership():
    placement = [RankPlacement(rank) for rank in range(8)]
    assert [rank.dp_rank for rank in placement] == list(range(8))
    assert {rank.tp_rank for rank in placement} == {0}
    topology = SegmentTopology.for_decode()
    assert (topology.tp, topology.dp, topology.world) == (1, 8, 8)
    assert topology.local_capacity == topology.capacity == 80
    assert topology.counts((1, 0, 2, 0, 0, 0, 0, 0)) == (
        (1, 0, 2, 0, 0, 0, 0, 0), (1, 0, 2, 0, 0, 0, 0, 0),
    )
    groups = build_cache_groups(_layers(), 8192)
    assert {group.num_partitions for group in groups} == {8}
    assert validate_page_tables((_pages(), _pages()), (0, 7), (1, 1), groups) == (
        _pages(), _pages(),
    )
    with pytest.raises(ValueError, match="DP cache partition"):
        validate_page_tables((_pages(),), (8,), (1,), groups)


def test_decode_capacity_requires_the_current_lib_contract(tmp_path):
    config = tmp_path / "models" / "deepseek_v4_1_flash" / "config.py"
    config.parent.mkdir(parents=True)
    config.write_text(
        "MAX_BATCH_PER_DP = 12\nDSPARK_SPEC_TOKENS = 5\n"
        "DECODE_ROWS_PER_REQUEST = DSPARK_SPEC_TOKENS + 1\n", encoding="utf-8",
    )
    assert selected_decode_topology(tmp_path).capacity == 80
    config.write_text(
        "MAX_BATCH_PER_DP = 32\nDSPARK_SPEC_TOKENS = 5\n"
        "DECODE_ROWS_PER_REQUEST = DSPARK_SPEC_TOKENS + 1\n", encoding="utf-8",
    )
    with pytest.raises(MissingCompositeInterface, match="12-request, six-row"):
        selected_decode_topology(tmp_path)


def test_tp1_segment_accepts_80_rows_per_dp_and_rejects_81():
    topology = SegmentTopology.for_decode()
    request = RequestSlice("A", 7, 0, 0, tuple(range(80)), 80,
                           MappingProxyType(_pages()))
    step = ForwardStep("prefill", (request,), 1)
    packed = prepare_segment_inputs(torch.ones((80, 4), dtype=torch.bfloat16), step, topology)
    assert packed.group_counts == (0, 0, 0, 0, 0, 0, 0, 80)
    assert packed.request_last_rows == ((7, 79),)
    assert packed.residual.shape == (8, 80, 4, 4)
    oversized = RequestSlice("A", 7, 0, 0, tuple(range(81)), 81,
                             MappingProxyType(_pages()))
    with pytest.raises(ValueError, match="segment capacity"):
        prepare_segment_inputs(torch.ones((81, 4), dtype=torch.bfloat16),
                               ForwardStep("prefill", (oversized,), 1), topology)


def test_full_prefill_metadata_preserves_two_stage_history_and_page_families():
    topology = SegmentTopology(tp=1, dp=8, local_capacity=16)
    cosine = torch.ones((256, 32), dtype=torch.float32)
    sine = torch.zeros_like(cosine)
    host = PrefillMetadataResources(
        topology, max_seq_len=256, max_requests=2, window_pages=3,
        c2a_pages=3, c1a_pages=3, rope_tables=(cosine, sine),
        compressed_rope_tables=(cosine, sine), full_hidden_size=4,
    )
    requests = (
        RequestSlice("A", 0, 0, 0, (11, 12), 2, MappingProxyType(_pages())),
        RequestSlice("B", 7, 1, 1, (21,), 2, MappingProxyType(_pages())),
    )
    step = ForwardStep("prefill", requests, 1)
    embeddings = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    assert host.stage_full(step, embeddings) == (2, 0, 0, 0, 0, 0, 0, 1)
    args = host.full_arguments()
    assert all(tensor.is_shared() for tensor in args.values())
    assert args["counts"].tolist() == [[2, 2]] + [[0, 0]] * 6 + [[1, 1]]
    assert args["embedding"][0, :2].equal(embeddings[:2])
    assert args["embedding"][7, 0].equal(embeddings[2])
    assert args["position_ids"][7, :, 0].tolist() == [1, 1]
    assert args["replay_rows"][7, :2].tolist() == [0, -1]
    assert args["freqs_cos"].shape == (8, 2, 256, 32)
    assert args["freqs_sin"].shape == (8, 2, 256, 32)
    assert args["token_to_req_indices"][7, :, 0].tolist() == [0, 0]
    assert args["logit_rows"][0, 0] == 1
    assert args["logit_rows"][7, 0] == 0
    assert args["c2a_index_block_table"][0, 0, 0] == 0
    assert args["c1a_index_block_table"][7, 0, 0] == 0


def test_decode_cache_plan_uses_the_lib_tp1_ep8_constants():
    layers = _layers()
    groups = build_cache_groups(layers, 8192)
    abi = SimpleNamespace(
        TP_SIZE=1, EP_SIZE=8, N_LAYERS=40, KV_SOURCE_COUNT=4,
        C2A_SOURCE_COUNT=3, INDEX_SOURCE_COUNT=8, BLOCK_SIZE=128,
        HEAD_DIM=512, INDEX_DIM=128, STATE_CAPACITY=2, STATE_WIDTH=1024,
        WINDOW_CACHE_GROUP=32, COMPRESSED_CACHE_GROUP=16, INDEX_CACHE_GROUP=32,
    )
    plan = plan_decode_cache_pools(
        layers, groups, max_seq_len=8192,
        primary_num_blocks=groups[0].max_blocks_per_seq,
        request_slots=1, abi=abi,
    )
    assert plan.buffers[0].shape[0] == 8
    assert plan.physical_request_slots == 2
    dtypes = {spec.name: spec.dtype for spec in plan.buffers}
    assert dtypes["window_cache_scale_pool"] == torch.bfloat16
    assert dtypes["compressed_cache_scale_pool"] == torch.bfloat16
    assert dtypes["index_cache_scale_pool"] == torch.float8_e8m0fnu


def test_cache_scale_reset_pages_use_the_lib_unit_scale_encoding():
    buffers = tuple(CacheBufferSpec(name, (8, 1, 2, 2), dtype) for name, dtype in (
        ("window_cache_scale_pool", torch.bfloat16),
        ("compressed_cache_scale_pool", torch.bfloat16),
        ("index_cache_scale_pool", torch.float8_e8m0fnu),
    ))
    plan = DecodeCachePlan(1, None, {}, {}, {}, buffers, 1, 2)
    resets = make_cache_page_reset_buffers(plan)
    assert all(bool((resets[name] == 1).all()) for name in (
        "window_cache_scale_pool", "compressed_cache_scale_pool"))
    assert bool((resets["index_cache_scale_pool"].view(torch.uint8) == 127).all())


def test_request_ledger_keeps_eight_partitions_independent():
    ledger = RequestLedger(max_requests=1, max_seq_len=8192 + 128)
    pages = _pages()
    first = ledger.begin_prefill((
        ("A", 0, 0, (11,), 2, pages),
        ("B", 7, 0, (21,), 1, pages),
    ))
    ledger.commit(first)
    continuation = ledger.begin_prefill((("A", 0, 1, (12,), 2, pages),))
    ledger.commit(continuation)
    decode = ledger.begin_decode((
        ("A", 0, 2, 13, pages),
        ("B", 7, 1, 22, pages),
    ))
    assert tuple(request.partition for request in decode.requests) == (0, 7)
    ledger.commit(decode)
    assert (ledger.owners["A"].length, ledger.owners["B"].length) == (3, 2)


def test_request_ledger_retains_eight_k_position_across_chunks():
    ledger = RequestLedger(max_requests=1, max_seq_len=8192 + 128)
    pages = _pages()
    for start in (0, 4096):
        step = ledger.begin_prefill(((
            "A", 7, start, tuple(range(start, start + 4096)), 8192, pages,
        ),))
        ledger.commit(step)
    assert ledger.owners["A"].length == 8192
    decode = ledger.begin_decode((("A", 7, 8192, 42, pages),))
    assert decode.requests[0].start == 8192


def test_c2a_and_c1a_metadata_stay_separate_in_last_dp_partition():
    topology = SegmentTopology(tp=1, dp=8, local_capacity=16)
    request = RequestSlice("A", 7, 0, 1, (42,), 2, MappingProxyType(_pages()))
    step = ForwardStep("prefill", (request,), 1)
    c2a = prepare_compressed_metadata(
        step, topology, ratio=2, compressed_group="c2a",
        cache_pages=2, max_requests=1, state_blocks=1,
    )
    c1a = prepare_compressed_metadata(
        step, topology, ratio=1, compressed_group="c1a",
        cache_pages=2, max_requests=1, state_blocks=1,
    )
    assert c2a.group_counts == c1a.group_counts == (0, 0, 0, 0, 0, 0, 0, 1)
    assert int(c2a.compressed_lens[7, 0]) == 1
    assert int(c1a.compressed_lens[7, 0]) == 2
    assert int(c2a.compressed_slots[7, 0]) == 0
    assert int(c1a.compressed_slots[7, 0]) == 1
    assert bool((c2a.compressed_slots[:7] == -1).all())
    assert bool((c1a.compressed_slots[:7] == -1).all())


def test_single_decode_fills_each_idle_partition_without_changing_real_step():
    pages = MappingProxyType(_pages())
    request = RequestSlice("A", 3, 0, 8192, (42,), 8192, pages)
    step = ForwardStep("decode", (request,), 1)
    plan = DecodeCachePlan(
        primary_num_blocks=1, family_groups=None,
        group_blocks={name: 1 for name in pages},
        physical_group_blocks={name: 2 for name in pages},
        group_sources={name: {} for name in pages},
        buffers=(CacheBufferSpec("window_cache_pool", (8, 1), torch.uint8),),
        request_slots=1, physical_request_slots=2,
    )
    embeddings = torch.ones((1, 4), dtype=torch.bfloat16)
    padded, values, idle = pad_single_decode(step, embeddings, plan)
    assert idle == (0, 1, 2, 4, 5, 6, 7)
    assert tuple(request.partition for request in padded.requests) == (3, *idle)
    assert padded.requests[0].pages == _pages()
    assert all(request.pages == {name: (1,) for name in pages}
               for request in padded.requests[1:])
    assert values.shape == (8, 4)
    assert step.requests == (request,)


def test_decode_padding_keeps_multiple_active_partitions_and_request_order():
    pages = MappingProxyType(_pages())
    requests = (
        RequestSlice("A", 7, 0, 8192, (42,), 8192, pages),
        RequestSlice("B", 2, 0, 8192, (43,), 8192, pages),
    )
    step = ForwardStep("decode", requests, 1)
    plan = DecodeCachePlan(
        primary_num_blocks=1, family_groups=None,
        group_blocks={name: 1 for name in pages},
        physical_group_blocks={name: 2 for name in pages},
        group_sources={name: {} for name in pages},
        buffers=(CacheBufferSpec("window_cache_pool", (8, 1), torch.uint8),),
        request_slots=1, physical_request_slots=2,
    )
    embeddings = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    padded, values, idle = pad_single_decode(step, embeddings, plan)
    assert idle == (0, 1, 3, 4, 5, 6)
    assert padded.requests[:2] == requests
    assert tuple(request.partition for request in padded.requests) == (7, 2, *idle)
    assert torch.equal(values[:2], embeddings)
    assert all(torch.equal(value, embeddings[0]) for value in values[2:])


def test_request_cache_reset_touches_only_its_tp1_rank():
    names = (
        "window_cache_pool", "window_cache_scale_pool",
        "compressed_cache_pool", "compressed_cache_scale_pool",
        "index_cache_pool", "index_cache_scale_pool",
    )
    plan = DecodeCachePlan(
        primary_num_blocks=1, family_groups=None,
        group_blocks={name: 1 for name in _pages()},
        physical_group_blocks={name: 2 for name in _pages()},
        group_sources={
            "window": {"window": (0, 1)},
            "c2a": {"compressed": (0,), "index": (0,)},
            "c1a": {},
        },
        buffers=tuple(CacheBufferSpec(name, (8, 1, 2), torch.uint8) for name in names)
        + (CacheBufferSpec("state_cache_pool", (8, 3, 2), torch.float32),),
        request_slots=1, physical_request_slots=2,
    )
    calls = []
    pool = object.__new__(DecodeCachePools)
    pool.plan = plan
    pool.state_reset = torch.zeros((8, 2), dtype=torch.float32)
    pool.page_resets = {name: torch.zeros((8, 2), dtype=torch.uint8) for name in names}
    pool.source_counts = {"state_cache_pool": 3}
    pool.worker = SimpleNamespace(copy_to=lambda *args, **kwargs:
                                  calls.append((args[0], kwargs["worker_id"], kwargs["dst_offset"])))
    pool.view = lambda name, source: SimpleNamespace(
        shards=[SimpleNamespace(data_ptr=10000 + rank + source * 100,
                                byte_offset=source * 100) for rank in range(8)],
    )
    pool.reset_request(SimpleNamespace(partition=7, slot=0, pages=_pages()))
    assert calls and {base for base, _, _ in calls} == {10007}
    assert {rank for _, rank, _ in calls} == {7}
    assert any(offset == 100 for _, _, offset in calls)
    assert any(offset == 200 for _, _, offset in calls)


def test_final_head_weight_is_present_on_every_dp_rank(monkeypatch):
    class TinyLoader:
        def __init__(self, *args, **kwargs):
            self.config = SimpleNamespace(vocab_size=16, hidden_size=4)

        def load(self, name):
            assert name == "norm.weight"
            return SimpleNamespace(weight=torch.ones(4, dtype=torch.bfloat16))

        def load_rows(self, name, start, stop):
            assert name == "head.weight"
            rows = torch.arange(start * 4, stop * 4, dtype=torch.float32)
            return SimpleNamespace(weight=rows.reshape(stop - start, 4).to(torch.bfloat16))

    monkeypatch.setattr(npu_runner, "V41WeightLoader", TinyLoader)
    host = load_final_output_host_resources("unused", SegmentTopology.for_decode())
    assert host.head_weight.shape == (8, 16, 4)
    assert all(torch.equal(host.head_weight[rank], host.head_weight[0]) for rank in range(1, 8))


def test_sidecar_fingerprint_separates_tp_sharding(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"w":"model-00001.safetensors"}}', encoding="utf-8")
    (tmp_path / "model-00001.safetensors").write_bytes(b"weights")
    tp1 = DecodeWeightLayout(40, 4, 8, 3, 8, 1)
    tp4 = DecodeWeightLayout(40, 4, 8, 3, 8, 4)
    assert source_fingerprint(tmp_path, tp1) != source_fingerprint(tmp_path, tp4)
