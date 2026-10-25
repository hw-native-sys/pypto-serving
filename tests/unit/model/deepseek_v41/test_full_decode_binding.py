# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for the complete V4.1 decode dispatch."""

import ast
import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pypto.runtime import DeviceTensor, StackedDeviceTensor
from pypto_serving.model.deepseek_v41 import npu_runner
from pypto_serving.model.deepseek_v41.npu_runner import (
    DecodeForwardSegment, DecodeFullCachePools, LayerState, SegmentTopology,
    V41ResidentResources,
    compile_prefill_fwd,
)
from pypto_serving.model.deepseek_v41.weight_loader import (
    PrepackedPrefillBank, ResidentFullDecodeWeights, prepack_prefill_bank,
)
from pypto_serving.model.deepseek_v41.weight_loader import _ALL


def test_selected_lib_full_decode_abi_is_fully_bound():
    root = Path(os.environ.get("PYPTO_LIB_ROOT") or Path(__file__).resolve().parents[4] / "pypto-lib")
    source = root / "models/deepseek_v4_1_flash/decode_fwd.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    entry = next(node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == "l3_decode_fwd")
    parameters = {argument.arg for argument in entry.args.args}
    metadata = set(DecodeForwardSegment._METADATA) | {
        "window_slots", "window_indices",
        "c2a_compressed_slots", "c1a_compressed_slots", "moe_num_tokens",
    }
    caches = {
        "window_cache_pool", "window_cache_scale_pool", "compressed_cache_pool",
        "compressed_cache_scale_pool", "index_cache_pool", "index_cache_scale_pool",
        "state_cache_pool",
    }
    assert parameters == _ALL | metadata | caches | {
        "x_hc", "pre_mix", "attention_num_tokens", "final_norm_weight",
        "lm_head_weight", "logit_row_indices", "logits", "sampled_ids"}


def test_weight_views_alias_one_full_rank_allocation():
    class Worker:
        def __init__(self):
            self.next_ptr = 1000
            self.copies = []
            self.freed = []
            self.views = []

        def alloc_tensor(self, shape, dtype, *, worker_id):
            self.next_ptr += 1000
            return DeviceTensor(self.next_ptr, shape, dtype)

        def copy_to(self, base, source, nbytes, *, dst_offset, worker_id):
            self.copies.append((base, dst_offset, nbytes, worker_id))

        def view_tensor(self, parent, shape, *, byte_offset, worker_id):
            assert parent.shape[0] == 4
            assert worker_id in (0, 1)
            self.views.append((worker_id, byte_offset))
            return DeviceTensor(parent.data_ptr, shape, parent.dtype)

        def free_tensor(self, tensor, *, worker_id):
            self.freed.append((tensor.data_ptr, worker_id))

        def free_stacked_tensor(self, tensor):
            for worker_id, shard in enumerate(tensor.shards):
                self.free_tensor(shard, worker_id=worker_id)

    host = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3).share_memory_()
    bank = SimpleNamespace(
        tensors={"hc_attn_fn": host},
        layout=SimpleNamespace(world=2, slots=lambda name: 2),
        finish=lambda names: None,
    )
    worker = Worker()
    resident = ResidentFullDecodeWeights(worker, bank, ("hc_attn_fn",))
    full = resident.full("hc_attn_fn")
    second = resident.view("hc_attn_fn", 1)
    assert full.shape == (2, 4, 3)
    assert second.shape == (2, 2, 3)
    assert second.shards[0].data_ptr == full.shards[0].data_ptr
    assert second.shards[1].data_ptr == full.shards[1].data_ptr
    assert worker.views == [(0, 2 * 3 * 4), (1, 2 * 3 * 4)]
    assert len(worker.copies) == 4
    resident.close()
    assert len(worker.freed) == 2


def test_full_prefill_cache_views_borrow_decode_pools_without_copy():
    class Worker:
        def view_tensor(self, parent, shape, *, byte_offset, worker_id):
            assert byte_offset == 0
            return DeviceTensor(parent.data_ptr, shape, parent.dtype)

    pool = DecodeFullCachePools.__new__(DecodeFullCachePools)
    pool.worker = Worker()
    pool.plan = SimpleNamespace(
        family_groups=SimpleNamespace(window="window", c2a="c2a", c1a="c1a"),
        physical_group_blocks={"window": 2, "c2a": 2, "c1a": 2},
        physical_request_slots=2,
    )
    sources = {
        "window_cache_pool": 40, "window_cache_scale_pool": 40,
        "compressed_cache_pool": 4, "compressed_cache_scale_pool": 4,
        "index_cache_pool": 8, "index_cache_scale_pool": 8,
        "state_cache_pool": 3,
    }
    pool.source_counts = sources
    pool._full = {}
    for offset, (name, count) in enumerate(sources.items()):
        shape = (count * 2, 128, 1, 4)
        shards = tuple(DeviceTensor(10000 + offset * 100 + rank, shape, torch.uint8)
                       for rank in range(2))
        pool._full[name] = StackedDeviceTensor(shards, (2, *shape), (0, 1))
    views = pool.prefill_full_views()
    assert views["window_cache"].shape == (2, 40, 2, 128, 1, 4)
    assert views["index_cache"].shape == (2, 4, 2, 128, 1, 4)
    assert views["state_cache"].shape == (2, 3, 2, 128, 1, 4)
    assert all(views[target].shards[rank].data_ptr == pool.full(source).shards[rank].data_ptr
               for target, source in (
                   ("window_cache", "window_cache_pool"),
                   ("compressed_cache", "compressed_cache_pool"),
                   ("index_cache", "index_cache_pool"),
                   ("state_cache", "state_cache_pool"),
               ) for rank in range(2))


def test_full_prefill_import_uses_serving_capacity_and_complete_abi():
    root = os.environ.get("PYPTO_LIB_ROOT")
    if not root:
        pytest.skip("set PYPTO_LIB_ROOT to test the selected V4.1 lib checkout")

    class Compiler:
        def compile(self, name, entry):
            assert name == "v41_prefill_fwd"
            return SimpleNamespace(compiled=entry)

    _, names = compile_prefill_fwd(Compiler(), root, SegmentTopology.for_decode())
    inputs = {
        "embedding", "embedding_ids", "initial_pre", "identity_rows", "counts",
        "replay_rows", "position_ids", "freqs_cos", "freqs_sin", "window_slots",
        "window_indices", "c2a_compressed_slots", "c1a_compressed_slots",
        "token_to_req_indices", "query_start_loc", "c2a_index_block_table",
        "c1a_index_block_table", "state_block_table", "logit_rows",
    }
    caches = {
        "window_cache", "window_scales", "compressed_cache", "compressed_scales",
        "index_cache", "index_scales", "state_cache",
    }
    outputs = {
        "final_norm_weight", "lm_head_weight", "x_hc_out", "pre_mix_out",
        "logits", "sampled_ids",
    }
    assert set(names) == _ALL | inputs | caches | outputs
    assert not {"bank8", "bank4", "scales8", "offsets"} & set(names)


def test_full_forward_reuses_one_weight_bank_between_prefill_and_decode(monkeypatch):
    events = []

    class Prefill:
        def __init__(self, *args):
            events.append("prefill_open")

        def close(self):
            events.append("prefill_close")

    class Weights:
        def __init__(self, *args):
            events.append("decode_open")
            self.layout_names = ()

        def close(self):
            events.append("decode_close")

    class Decode:
        def __init__(self, *args):
            events.append("decode_bind")

    monkeypatch.setattr(npu_runner, "PrefillForwardSegment", Prefill)
    monkeypatch.setattr(npu_runner, "ResidentFullDecodeWeights", Weights)
    monkeypatch.setattr(npu_runner, "DecodeForwardSegment", Decode)
    resources = V41ResidentResources.__new__(V41ResidentResources)
    resources.worker = SimpleNamespace(
        free_stacked_tensor=lambda tensor: events.append("lut_close"),
        close=lambda: events.append("worker_close"),
    )
    resources._weight_bank = object()
    resources._weight_param_names = ()
    resources._prefill_binding = (object(), (), object(), object())
    resources._decode_binding = (object(), (), object(), True)
    resources.weights = Weights()
    resources.prefill_host = object()
    resources.decode_host = SimpleNamespace(topology=object())
    resources.cache = resources.final = object()
    resources._prefill_fwd = resources._decode_fwd = None
    resources._owned = []
    resources._step = resources._inputs = resources._group_counts = None
    resources.closed = False

    resources._activate_prefill()
    resources._activate_prefill()
    resources._activate_decode()
    resources._activate_decode()
    resources._activate_prefill()
    resources.close()

    assert events == ["decode_open", "prefill_open", "decode_bind",
                      "prefill_close", "decode_close", "worker_close"]


def test_prefill_bank_sidecar_replays_exact_bytes_and_rejects_stale_source(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "shard.bin"}}), encoding="utf-8")
    (model / "shard.bin").write_bytes(b"source")

    @dataclass
    class Layout:
        world: int = 2

    class Manifest:
        ranks = 2
        tensor_layouts = {"bank8": ((4,), torch.uint8), "bf16": ((2,), torch.bfloat16)}
        bytes_per_rank = 8

        @staticmethod
        def upload(runtime, rank, handles):
            for name, values in (("bank8", (rank, 1, 2, 3)),
                                 ("bf16", (4, 5, 6, rank))):
                source = torch.tensor(values, dtype=torch.uint8)
                runtime.copy_to(handles[name].data_ptr, source.data_ptr(), 4,
                                dst_offset=0, worker_id=rank)

    sidecar = tmp_path / "prefill"
    prepack_prefill_bank(Manifest(), model, Layout(), "lib-rev", sidecar)
    packed = PrepackedPrefillBank(sidecar, model, Layout(), "lib-rev", Manifest())
    recorded = {}

    class Worker:
        @staticmethod
        def copy_to(dst_ptr, src_ptr, count, *, dst_offset, worker_id):
            recorded[(worker_id, dst_ptr + dst_offset)] = ctypes.string_at(src_ptr, count)

    for rank in range(2):
        packed.upload(Worker(), rank, {
            "bank8": SimpleNamespace(data_ptr=100), "bf16": SimpleNamespace(data_ptr=200),
        })
        assert recorded[(rank, 100)] == bytes((rank, 1, 2, 3))
        assert recorded[(rank, 200)] == bytes((4, 5, 6, rank))
    with pytest.raises(ValueError, match="stale"):
        PrepackedPrefillBank(sidecar, model, Layout(), "different-lib", Manifest())


def test_full_decode_binds_distinct_c2a_and_c1a_metadata():
    fields = {source: object() for source in DecodeForwardSegment._METADATA.values()}
    for name in (
        "window_slots", "window_indices",
        "c2a_compressed_slots", "c1a_compressed_slots", "moe_counts",
    ):
        fields[name] = object()
    metadata = SimpleNamespace(fields=fields)
    weight = object()
    cache_tensor = object()
    weights = SimpleNamespace(layout_names=("hc_attn_fn",), full=lambda name: weight)
    cache = SimpleNamespace(_full={"window_cache_pool": cache_tensor},
                            full=lambda name: cache_tensor)
    names = (
        "x_hc", "pre_mix", "attention_num_tokens", "hc_attn_fn", "window_cache_pool",
        "moe_num_tokens", *DecodeForwardSegment._METADATA,
        "window_slots", "window_indices",
        "c2a_compressed_slots", "c1a_compressed_slots",
        "final_norm_weight", "lm_head_weight", "logit_row_indices", "logits", "sampled_ids",
    )
    final = SimpleNamespace(norm_weight=object(), head_weight=object(), host=SimpleNamespace(
        logit_row_indices=object(), logits=object(), sampled_ids=object()))
    calls = []
    worker = SimpleNamespace(_persistent=True, _reset_persistent_windows=False,
                             run=lambda *args, **kwargs: calls.append(args))
    program = SimpleNamespace(compiled=object())
    segment = DecodeForwardSegment(worker, program, names, object(),
                                   SegmentTopology.for_decode(tp=1, dp=8),
                                   False, weights, cache, metadata, final)
    state = LayerState(object(), object(), "tp_local_token")
    assert segment.run(state, group_counts=(1,) * 8) is state
    bound = dict(zip(names, calls[0][1:]))
    assert bound["c2a_compressed_slots"] is not bound["c1a_compressed_slots"]
    assert bound["c2a_index_block_table"] is not bound["c1a_index_block_table"]
    assert bound["x_hc"] is state.residual
    assert bound["pre_mix"] is state.pre_mix
    assert bound["hc_attn_fn"] is weight
    assert worker._reset_persistent_windows is False

    calls.clear()
    tensor_segment = DecodeForwardSegment(
        worker, program, names, object(), SegmentTopology.for_decode(tp=1, dp=8),
        True, weights, cache, metadata, final)
    assert tensor_segment.run(state, group_counts=(1,) * 8) is state
    tensor_bound = dict(zip(names, calls[0][1:]))
    assert tensor_bound["attention_num_tokens"] is fields["moe_counts"]
