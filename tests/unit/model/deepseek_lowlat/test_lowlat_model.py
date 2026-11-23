# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lowlat model ABI, bounded storage, transfers and native execution."""

import ctypes
import hashlib
import json
import sys
import threading
from contextlib import nullcontext
from dataclasses import replace
from math import prod
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor, distributed_runner

from pypto_serving.config.types import DecodeBatch, PrefillBatch
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.deepseek_lowlat import npu_runner
from pypto_serving.model.deepseek_lowlat.bindings import (
    TensorArena,
    frame_shape,
    ordered_arguments,
    parameter_rows,
)
from pypto_serving.model.deepseek_lowlat.cache import TABLE_GROUPS, TENSOR_GROUPS, LowlatCachePool
from pypto_serving.model.deepseek_lowlat.capacity import LowlatCapacityPlan
from pypto_serving.model.deepseek_lowlat.contract import build_lowlat_cache_group_specs
from pypto_serving.model.deepseek_lowlat.decoding import (
    DecodeReservation,
    decode_decode_extension,
    decode_reservation_payload,
    encode_reservation,
)
from pypto_serving.model.deepseek_lowlat.history import NativeHistorySeed
from pypto_serving.model.deepseek_lowlat.initialization import PageInitTemplates
from pypto_serving.model.deepseek_lowlat.kernel_loader import LowlatPrograms, isolated_profile
from pypto_serving.model.deepseek_lowlat.native_frames import NativeFrames, allocate_native_hosts
from pypto_serving.model.deepseek_lowlat.native_inputs import NativeInputs
from pypto_serving.model.deepseek_lowlat.native_pipeline_runner import NativePipelineRunner
from pypto_serving.model.deepseek_lowlat.npu_runner import LowlatModelRunner
from pypto_serving.model.deepseek_lowlat.prefill_transfers import PrefillTransferSlots
from pypto_serving.model.deepseek_lowlat.profiling import operation
from pypto_serving.model.deepseek_lowlat.speculation import (
    NativeState,
    validate_native_capacity,
    validate_packets,
)
from pypto_serving.model.deepseek_lowlat.state import LowlatRequestState
from pypto_serving.model.deepseek_lowlat.transfers import (
    DECODE_FIELDS,
    DYNAMIC_FIELDS,
    TABLE_FIELDS,
    DecodeTransferSlots,
)
from pypto_serving.model.deepseek_lowlat.weight_spec import WeightRule, shard_expert, transform_weight
from pypto_serving.model.deepseek_lowlat.worker_options import native_worker_options
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.model_extension import RequestExecutionContext
from pypto_serving.tools.profile import configure_profiler, create_profile_config, merge_profile


# Kernel ABI and weight adaptation.

def row(name, shape, dtype="int32"):
    return {"name": name, "shape": shape, "dtype": dtype, "direction": "In", "layout": "ND"}


def stacked(shape=(8, 2), dtype=torch.int32, worker_ids=tuple(range(8))):
    shards = [DeviceTensor(0x10000 * (rank + 1), shape[1:], dtype) for rank in range(8)]
    return StackedDeviceTensor(shards, shape, worker_ids)


def test_generated_order_and_ssa_normalization():
    metadata = {"params": [row("b__ssa_v0", [8, 2]), row("a__ssa_v0", None)]}
    metadata["params"][1]["layout"] = None
    value = stacked()
    result = ordered_arguments(metadata, {"a": 3, "b": value})
    assert result == (value, 3)


def test_dynamic_request_axes_do_not_scale_with_prompt_tokens():
    assert frame_shape("csa_compress_state_block_table", (8, -1, 524288), 8192) == (8, 1, 524288)
    assert frame_shape("state_checkpoint_ends", (8, -1, -1, 6), 8192) == (8, 1, 16, 6)
    assert frame_shape("dspark_target_hidden", (8, -1, 12288), 128) == (8, 16, 12288)
    assert frame_shape("swa_freqs_cos", (8, -1, 64), 128) == (8, 128, 64)
    with pytest.raises(ValueError, match="unknown dynamic"):
        frame_shape("new_unclassified_param", (8, -1), 128)


@pytest.mark.parametrize("value", [True, 0.0, 2**31, -(2**31) - 1])
def test_scalar_rejects_bool_float_and_overflow(value):
    with pytest.raises(ValueError, match="scalar INT32"):
        ordered_arguments({"params": [row("terminal", None)]}, {"terminal": value})


def test_rank_order_and_shape_are_checked():
    metadata = {"params": [row("ids", [8, 2])]}
    with pytest.raises(ValueError, match="ABI mismatch"):
        ordered_arguments(metadata, {"ids": stacked(worker_ids=tuple(reversed(range(8))))})
    with pytest.raises(ValueError, match="ABI mismatch"):
        ordered_arguments(metadata, {"ids": stacked((8, 3))})


def test_only_declared_shared_host_control_can_cross_directly():
    metadata = {"params": [row("num_tokens_per_owner", [8])]}
    with pytest.raises(ValueError, match="pre-fork"):
        ordered_arguments(metadata, {"num_tokens_per_owner": torch.ones(8, dtype=torch.int32)})
    host = torch.ones(8, dtype=torch.int32).share_memory_()
    assert ordered_arguments(metadata, {"num_tokens_per_owner": host})[0] is host
    with pytest.raises(TypeError, match="Worker-owned"):
        ordered_arguments({"params": [row("ids", [8, 2])]}, {"ids": torch.ones((8, 2), dtype=torch.int32)})


def test_ambiguous_or_missing_generated_parameters_fail():
    with pytest.raises(ValueError, match="duplicate"):
        parameter_rows({"params": [row("x__ssa_v0", [8, 2]), row("x__ssa_v1", [8, 2])]})
    with pytest.raises(ValueError, match="missing"):
        ordered_arguments({"params": [row("x", [8, 2])]}, {})


@pytest.mark.parametrize("matrix", [1, 2])
def test_tp8_reconstructs_all_intermediate_values(matrix):
    shape = (6, 16) if matrix == 2 else (16, 6)
    original = torch.arange(96, dtype=torch.int8).reshape(shape)
    shards = [shard_expert(original, matrix=matrix, scale=False, rank=r) for r in range(8)]
    restored = torch.cat(shards, dim=1 if matrix == 2 else 0)
    assert torch.equal(restored, original)
    assert all(s.is_contiguous() for s in shards)


def test_w2_scales_replicate_output_channels_while_w1_scales_partition():
    output_scales = torch.arange(6, dtype=torch.float32)
    hidden_scales = torch.arange(16, dtype=torch.float32)
    for rank in range(8):
        assert torch.equal(shard_expert(output_scales, matrix=2, scale=True, rank=rank), output_scales)
    restored = torch.cat([shard_expert(hidden_scales, matrix=1, scale=True, rank=r) for r in range(8)])
    assert torch.equal(restored, hidden_scales)


def test_nd_projection_transpose_is_not_nz_repacking():
    source = torch.arange(24, dtype=torch.int8).reshape(4, 6)
    rule = WeightRule("wq_b", "attn.wq_b.weight", torch.int8, transpose=True)
    actual = transform_weight(rule, source)
    assert actual.shape == (6, 4)
    assert actual[2, 1] == source[1, 2]
    assert actual.is_contiguous()


def test_float_experts_cannot_be_silently_requantized():
    with pytest.raises(ValueError, match="INT8"):
        shard_expert(torch.zeros((16, 6)), matrix=1, scale=False, rank=0)


def test_compact_projection_sources_reconstruct_without_full_device_banks():
    from types import SimpleNamespace

    from pypto_serving.model.deepseek_lowlat.weight_loader import LowlatWeightStore
    from pypto_serving.model.deepseek_lowlat.weight_spec import WEIGHT_NAMES

    assert {"wo_a_shard", "wo_b_shard"} <= WEIGHT_NAMES
    assert not {"wo_a", "wo_b"} & WEIGHT_NAMES
    store = LowlatWeightStore.__new__(LowlatWeightStore)
    store.rows = {
        "wo_a_shard": {"shape": [8, 43, 2, 3]},
        "wo_b_shard": {"shape": [8, 43 * 4, 2]},
    }
    assert store._layer_shape("wo_a") == (8, 2, 3)
    assert store._layer_shape("wo_b") == (4, 16)
    class Worker:
        def view_tensor(self, owner, shape, worker_id):
            return DeviceTensor(owner.data_ptr, shape, owner.dtype)
    from pypto.runtime import DeviceTensor, StackedDeviceTensor
    def owner(shape, dtype):
        return StackedDeviceTensor([DeviceTensor(0x10000*(r+1), shape[1:], dtype)
                                    for r in range(8)], shape, tuple(range(8)))
    store.worker = Worker()
    store.owners = {"wo_a_shard": owner((8, 43, 2, 3), torch.bfloat16),
                    "wo_b_shard": owner((8, 172, 2), torch.int8)}
    # Existing prefill view adapters retain their regular required weight rows.
    for name in ("wq_a", "wq_b", "csa_weights_proj"):
        store.owners[name] = owner((8, 2, 3), torch.bfloat16)
    store.programs = SimpleNamespace(prefill_metadata={"params": [
        {"name": name, "shape": shape, "dtype": "bf16", "direction": "In"}
        for name, shape in (("wo_a", [8, 43, 2, 3]), ("wo_b", [8, 43, 4, 2]),
                            ("wq_a", [8, 2, 3]), ("wq_b", [8, 2, 3]),
                            ("csa_weights_proj", [8, 2, 3]))]})
    aliases = store.prefill_bindings()
    assert aliases["wo_a"].shape == (8, 43, 2, 3)
    assert aliases["wo_b"].shape == (8, 43, 4, 2)
    assert set(store.owners) == {"wo_a_shard", "wo_b_shard", "wq_a", "wq_b", "csa_weights_proj"}


def test_public_depth_forwarding_preserves_original_factory_validation_and_restores_scope(monkeypatch):
    import simpler.worker

    fake = Mock(return_value=object())
    monkeypatch.setattr(simpler.worker, "Worker", fake)
    original = distributed_runner._construct_worker
    with native_worker_options() as receipt:
        with pytest.raises(ValueError, match="positive finite"):
            distributed_runner._construct_worker(SimpleNamespace(device_ids=(7, 6)), "a2a3", "aicpu", 8,
                                                 startup_timeout_s=-1)
        distributed_runner._construct_worker(SimpleNamespace(device_ids=(7, 6)), "a2a3", "aicpu", 8,
                                             enable_sdma=True, startup_timeout_s=7200)
        options = fake.call_args.kwargs
        assert options == dict(level=3, device_ids=(7, 6), num_sub_workers=8, platform="a2a3",
                               runtime="aicpu", enable_sdma=True, startup_timeout_s=7200,
                               pipeline_depth=2, launch_depth=1)
        assert receipt["sdk_files_modified"] is False
    assert distributed_runner._construct_worker is original
    with pytest.raises(RuntimeError, match="verification"):
        with native_worker_options():
            raise RuntimeError("verification")
    assert distributed_runner._construct_worker is original


def test_unknown_factory_signature_fails_closed_before_constructing_worker(monkeypatch):
    fake = Mock()
    monkeypatch.setattr(distributed_runner, "_construct_worker", fake)
    with pytest.raises(ValueError, match="signature changed"):
        with native_worker_options():
            pytest.fail("unexpected factory accepted")
    fake.assert_not_called()


# Capacity and isolated lib imports.

@pytest.mark.parametrize("context", [16384, 16385, 1048576])
def test_logical_compile_and_physical_capacities_are_separate(context):
    plan = LowlatCapacityPlan(context, 512, "rolling")
    assert context <= plan.compiled_context < context + 128
    groups = {g.name: g for g in plan.groups}
    assert groups["ori"].active_ring_blocks == 6
    assert groups["hca_state"].active_ring_blocks == 18
    assert groups["cmp_c4"].physical_pool_blocks == (context + 127) // 128
    assert groups["csa_state"].logical_table_blocks == (context + 3) // 4
    assert plan.rope_bytes_per_rank == plan.compiled_context * 512


def test_million_token_memory_ledger_and_prefix_reserve():
    base = LowlatCapacityPlan(1048576, 512, "rolling")
    assert base.cache_bytes_per_rank == 6580633600
    prefix = LowlatCapacityPlan(1048576, 512, "rolling", True)
    assert prefix.cache_bytes_per_rank == base.cache_bytes_per_rank + 49053696 + 17842176
    for group in prefix.groups:
        spec = next(s for s in prefix.cache_specs() if s.name == group.name)
        assert spec.max_blocks_per_seq == group.active_ring_blocks
        assert spec.num_blocks == group.physical_pool_blocks


@pytest.mark.parametrize(
    "context,chunk", [(True, 512), (0, 512), (1048577, 512), (16384, False), (16384, 8193)]
)
def test_invalid_sizes_fail_on_cpu(context, chunk):
    with pytest.raises(ValueError):
        LowlatCapacityPlan(context, chunk, "rolling")


def test_extended_full_layout_is_rejected_before_allocating():
    with pytest.raises(ValueError, match="rolling"):
        LowlatCapacityPlan(1048576)


def test_native_logical_context_includes_full_lookahead():
    prompt, output = 1046522, 2048
    validate_native_capacity(prompt_tokens=prompt, max_output_tokens=output, physical_capacity=1048576)
    with pytest.raises(ValueError):
        validate_native_capacity(prompt_tokens=1048576, max_output_tokens=output, physical_capacity=1048576)


def test_actual_lib_import_binds_capacity_and_restores_globals():
    from pypto_serving.model.deepseek_lowlat.kernel_loader import validate_shared_rope

    for capacity in (1048576, 16384):
        with isolated_profile(context_capacity=capacity):
            from models.deepseek_v4_flash_lowlat import decode_shapes
            from models.deepseek_v4_flash_lowlat.config import FLASH
            from models.deepseek_v4_flash_lowlat.drafter.config import FLASH as draft

            assert FLASH.max_position_embeddings == capacity
            assert decode_shapes.KV_ORI_TABLE_MAX_BLOCKS == capacity // 128
            validate_shared_rope(FLASH, draft, capacity)


PREFIX = "models.deepseek_v4_flash_lowlat"


def fake_parent(monkeypatch):
    parent = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", parent)
    for name in list(sys.modules):
        if name == PREFIX or name.startswith(PREFIX + "."):
            monkeypatch.delitem(sys.modules, name)
    return parent


def test_failed_import_restores_existing_profile_and_process_globals(monkeypatch):
    parent = fake_parent(monkeypatch)
    old = ModuleType(PREFIX)
    old_config = ModuleType(PREFIX + ".config")
    parent.deepseek_v4_flash_lowlat = old
    monkeypatch.setitem(sys.modules, PREFIX, old)
    monkeypatch.setitem(sys.modules, PREFIX + ".config", old_config)
    before_argv = sys.argv
    before_path = list(sys.path)
    transient = ModuleType(PREFIX + ".new_helper")
    with pytest.raises(RuntimeError, match="import failed"):
        with isolated_profile() as captured:
            assert PREFIX not in sys.modules
            assert sys.argv[sys.argv.index("--tp") + 1] == "8"
            sys.modules[transient.__name__] = transient
            raise RuntimeError("import failed")
    assert captured[transient.__name__] is transient
    assert transient.__name__ not in sys.modules
    assert sys.modules[PREFIX] is parent.deepseek_v4_flash_lowlat is old
    assert sys.modules[PREFIX + ".config"] is old_config
    assert sys.argv is before_argv
    assert sys.path == before_path


def test_program_context_retains_lazy_helpers_without_publishing_them(monkeypatch):
    parent = fake_parent(monkeypatch)
    private = ModuleType(PREFIX)
    helper = ModuleType(PREFIX + ".lazy_helper")
    programs = LowlatPrograms(None, None, {}, {}, {PREFIX: private}, Path("."), None, None)
    with programs.context():
        assert parent.deepseek_v4_flash_lowlat is private
        sys.modules[helper.__name__] = helper
    assert not hasattr(parent, "deepseek_v4_flash_lowlat")
    assert PREFIX not in sys.modules and helper.__name__ not in sys.modules
    with programs.context():
        assert sys.modules[helper.__name__] is helper
        assert sys.modules[PREFIX] is private
    assert programs.modules[helper.__name__] is helper


class BudgetObserved(Exception):
    pass


def test_admission_skips_borrowed_weights_but_retains_real_projection_workspace(monkeypatch):
    def row(name, shape, dtype, direction):
        return dict(name=name,shape=shape,dtype=dtype,direction=direction)
    single = {"params": [row("state", [8, 1], "int32", "In")]}
    prefill = {"params": [row("wo_a", [8, 43, 1024, 4096], "bfloat16", "In"),
        row("wo_b", [8, 176128, 1024], "int8", "In"),
        row("o_proj_wo_a_full", [8, 16, 4], "float32", "Out")]}
    config = build_pypto_run_config(platform="a2a3",device_ids=tuple(range(8)))
    program = SimpleNamespace(compiled=SimpleNamespace(prepare=Mock(return_value=object())))
    programs = SimpleNamespace(single=program,prefill=program,single_metadata=single,prefill_metadata=prefill,
                               run_config=config,model=object(),context=nullcontext)
    pool = SimpleNamespace(prepare_host_templates=lambda size:[],bytes_per_rank=lambda:0)
    monkeypatch.setattr(npu_runner,"LowlatCachePool",lambda *args:pool)
    monkeypatch.setattr(npu_runner,"DecodeTransferSlots",lambda *args:SimpleNamespace(inherited=[]))
    monkeypatch.setattr(npu_runner,"LowlatWeightStore",lambda *args:SimpleNamespace(planned_bytes_per_rank=lambda:500))
    instance=npu_runner.LowlatModelRunner(programs,None)
    instance._check_budget=Mock(side_effect=BudgetObserved)
    runtime=SimpleNamespace(total_kv_pages=None,max_prefill_tokens_per_request=64,max_num_batched_tokens=64,
        kv_cache_groups=(),ring_heap=(268435456,268435456,1073741824,1073741824),
        ring_task_window=131072,ring_dep_pool=131072)
    with pytest.raises(BudgetObserved):
        instance.init_kv_cache("m",None,runtime)
    # 500 resident bytes +4 decode state +256 real scratch.
    # The541065216byte view extent is already paid by resident weights.
    instance._check_budget.assert_called_once_with(runtime,760)


# Cache storage and initialization.

def pool():
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    metadata = {
        "params": [
            {"name": name, "shape": [8, 1, 512], "dtype": "int32", "direction": "In"} for name in TABLE_GROUPS
        ]
    }
    cache = LowlatCachePool(SimpleNamespace(single_metadata=metadata), None, specs)
    cache._initialize_new_pages = lambda groups: None
    return cache


def test_capacity_ledger_matches_selected_lib_before_allocating_host_templates(monkeypatch):
    from dataclasses import replace
    from unittest.mock import Mock

    from pypto_serving.model.deepseek_lowlat import initialization
    from pypto_serving.model.deepseek_lowlat.kernel_loader import isolated_profile

    cache = pool()
    cache.programs.context = isolated_profile
    templates = Mock(return_value=SimpleNamespace(inherited=[]))
    monkeypatch.setattr(initialization, "PageInitTemplates", templates)
    assert cache.prepare_host_templates(512) == []
    templates.assert_called_once()
    templates.reset_mock()
    group = cache.specs["ori"]
    cache.specs["ori"] = replace(group, spec=replace(group.spec, page_size_bytes=group.spec.page_size_bytes+1))
    with pytest.raises(ValueError, match="selected lib cache layouts"):
        cache.prepare_host_templates(512)
    templates.assert_not_called()


def rolling_cache(chunk=512):
    specs = build_lowlat_cache_group_specs(max_seq_len=16384, max_prefill_tokens=chunk, layout="rolling")
    metadata = {"params": [{"name": name, "shape": [8, 1, 16384//next(s.spec.token_capacity for s in specs
                           if s.name == group)], "dtype": "int32", "direction": "In"}
                          for name, group in TABLE_GROUPS.items()]}
    cache = LowlatCachePool(SimpleNamespace(single_metadata=metadata), None, specs)
    cache.begin_generation(1)
    initialized = []
    cache._initialize_pages = lambda worker, name, root, pages: initialized.append((name, tuple(pages[0])))
    cache.owners = {name: object() for name in cache.initialized}
    cache.arena = SimpleNamespace(worker=object())
    manager = KvCacheManager(enable_prefix_cache=False)
    manager.init_groups(specs, max_batch_size=1)
    return cache, manager, initialized


def test_scheduler_and_renderer_reuse_physical_slots_without_reinitializing_live_pages():
    cache, manager, initialized = rolling_cache()
    assert cache.counts == dict(ori=6, cmp_c128=128, cmp_c4=128, idx=128,
                               hca_state=18, csa_state=4, csa_inner_state=4)
    assert cache.specs["csa_state"].sliding_window == 8
    identities = None
    for end in range(512, 16385, 512):
        groups = manager.ensure_group_blocks("r", end)
        tables, counts = cache.tables_for(groups, input_end=end)
        for index, (name, group) in enumerate(TABLE_GROUPS.items()):
            pages = groups[group]
            assert len(pages) == min((end+cache.specs[group].spec.token_capacity-1)
                                    //cache.specs[group].spec.token_capacity, cache.counts[group])
            assert counts[0, 0, index] == len(pages)
            if cache.specs[group].sliding_window is not None:
                expected = torch.tensor(pages)[torch.arange(tables[name].shape[-1]) % len(pages)]
                assert torch.equal(tables[name][0, 0], expected)
                assert tables[name].ge(0).all()
        if end >= 1024:
            current = {g: tuple(groups[g]) for g in ("ori", "hca_state", "csa_state", "csa_inner_state")}
            if identities is not None:
                assert current == identities
            identities = current
    for tensor in cache.initialized:
        pages = [page for name, row in initialized if name == tensor for page in row]
        assert len(pages) == len(set(pages))
    with pytest.raises(ValueError, match="rewind"):
        cache.tables_for(groups, input_end=512)
    manager.release_all_group_requests("r")
    cache.release()
    cache.begin_generation(2)
    previous = len(initialized)
    groups = manager.ensure_group_blocks("next", 512)
    cache.tables_for(groups, input_end=512)
    assert len(initialized) > previous


@pytest.mark.parametrize("chunk", [128, 512, 8192])
def test_raw_and_carry_windows_do_not_alias_the_current_publication_write_set(chunk):
    cache, manager, _ = rolling_cache(chunk)
    groups = manager.ensure_group_blocks("r", 16384)
    tables, _ = cache.tables_for(groups, input_end=16384)
    # Private prefill has DECODE_SEQ8. K5's six rows fit the same transaction bound.
    windows = {"ori": (127, chunk), "hca_state": (128, 8),
               "csa_state": (8, 8), "csa_inner_state": (8, 8)}
    for table_name, group in TABLE_GROUPS.items():
        if group not in windows:
            continue
        history, writes = windows[group]
        page_rows = cache.specs[group].spec.token_capacity
        table = tables[table_name][0, 0]
        for anchor in range(128, 1024):
            positions = range(max(0, anchor-history+1), anchor+writes+1)
            physical = [int(table[position//page_rows])*page_rows+position%page_rows for position in positions]
            assert len(physical) == len(set(physical)), (group, chunk, anchor)


def test_actual_permuted_physical_page_ids_drive_the_entire_rolling_table():
    cache, manager, _ = rolling_cache()
    groups = manager.ensure_group_blocks("r", 16384)
    groups = {name: list(reversed(ids)) for name, ids in groups.items()}
    tables, counts = cache.tables_for(groups, input_end=16384)
    for index, (name, group) in enumerate(TABLE_GROUPS.items()):
        assert counts[0, 0, index] == len(groups[group])
        if cache.specs[group].sliding_window is not None:
            expected = torch.tensor(groups[group])[torch.arange(tables[name].shape[-1]) % len(groups[group])]
            assert torch.equal(tables[name][0, 0], expected)


@pytest.fixture(scope="module")
def library():
    with isolated_profile():
        from models.deepseek_v4_flash_lowlat import prefill_cache
        return prefill_cache


class Copies:
    def __init__(self):
        self.calls = []
        self.sources = []

    def copy_to(self, destination, source, size, *, dst_offset, worker_id):
        digest = hashlib.sha256(ctypes.string_at(source, size)).hexdigest()
        self.calls.append((destination, size, dst_offset, worker_id, digest))
        self.sources.append((source, size))


def root_for(name, layouts, specs):
    layers, rows, _, tail, dtype = layouts[name]
    blocks = next(s.num_blocks for s in specs if s.name == TENSOR_GROUPS[name])
    shape = (8, layers*blocks, rows, *tail)
    size = prod(shape[1:])*torch.empty((), dtype=dtype).element_size()
    shards = [SimpleNamespace(data_ptr=0x100000000+rank*0x10000000, byte_offset=0, nbytes=size)
              for rank in range(8)]
    return SimpleNamespace(shape=shape, dtype=dtype, worker_ids=tuple(range(8)), shards=shards)


@pytest.mark.parametrize("name", list(TENSOR_GROUPS))
def test_templates_match_original_payload_offsets_and_coalesced_calls(library, name):
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    template = PageInitTemplates(library.CACHE_LAYOUTS, TENSOR_GROUPS, specs, 512)
    root = root_for(name, library.CACHE_LAYOUTS, specs)
    pages = [[0, 2, 3] if rank%2==0 else [1, 2] for rank in range(8)]
    expected, actual = Copies(), Copies()
    library.initialize_cache_pages(expected, name, root, pages)
    template.initialize(actual, name, root, pages)
    assert actual.calls == expected.calls
    host = template.hosts[name]
    first, last = host.data_ptr(), host.data_ptr()+host.numel()*host.element_size()
    assert host.is_shared()
    assert all(first <= pointer and pointer+size <= last for pointer, size in actual.sources)


def test_invalid_rank_page_or_owner_fails_before_any_copy(library):
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    template = PageInitTemplates(library.CACHE_LAYOUTS, TENSOR_GROUPS, specs, 512)
    name = "hca_compress_state"
    root = root_for(name, library.CACHE_LAYOUTS, specs)
    worker = Copies()
    pages = [[0] for _ in range(8)]
    pages[-1] = [True]
    with pytest.raises(ValueError, match="unique in-range"):
        template.initialize(worker, name, root, pages)
    assert not worker.calls
    with pytest.raises(ValueError, match="owner/rank"):
        template.initialize(worker, name, root, [[]])
    assert not worker.calls


def test_failed_family_cannot_acknowledge_group_ownership_and_release_invalidates(library):
    specs = build_lowlat_cache_group_specs(max_seq_len=512)
    metadata = {"params": [dict(name=name, shape=[8, 1, 512], dtype="int32", direction="In")
                           for name in TABLE_GROUPS]}
    cache = LowlatCachePool(SimpleNamespace(single_metadata=metadata), SimpleNamespace(worker=None), specs)
    cache.owners = {name: object() for name in TENSOR_GROUPS}
    cache.begin_generation(1)
    completed = []

    def initialize(worker, name, root, pages):
        if name == "idx_kv_scale":
            raise RuntimeError("scale copy failed")
        completed.append(name)

    cache._initialize_pages = initialize
    groups = {name: [0] for name in cache.specs}
    with pytest.raises(RuntimeError, match="scale copy failed"):
        cache._initialize_new_pages(groups)
    assert not any(cache.owned_pages.values())
    assert all(pages == {0} for pages in cache.initialized["idx_kv_cache"])
    assert not any(cache.initialized["idx_kv_scale"])
    cache.release()
    assert cache.generation is None and not any(any(ranks) for ranks in cache.initialized.values())
    cache.begin_generation(2)
    assert cache.generation == 2


# Registered transfers and prefill completion.

class CopyRecorder:
    def __init__(self):
        self.calls = []
        self.fail_after = None

    def copy_to(self, destination, source, size, *, worker_id):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("copy failed")
        self.calls.append((destination, source, size, worker_id))


def setup_slots():
    metadata = {"params": []}
    bindings = {}
    for index, name in enumerate(DECODE_FIELDS):
        shape = (8, 1, 7) if name == "block_counts" else (8, 1, 4) if name in TABLE_FIELDS else (8, 1)
        dtype = torch.int64 if name == "input_ids" else torch.int32
        metadata["params"].append(dict(name=name, shape=list(shape), dtype="int64" if dtype == torch.int64
                                        else "int32", direction="In", layout="ND"))
        shards = [DeviceTensor((index + 1)*0x10000+rank*0x1000, shape[1:], dtype) for rank in range(8)]
        bindings[name] = StackedDeviceTensor(shards, shape, tuple(range(8)))
    slots = DecodeTransferSlots(metadata)
    worker = CopyRecorder()
    program = object()
    slots.bind(worker, TensorArena(worker), bindings, bindings, metadata, program)
    worker.calls.clear()
    tables = {name: torch.full_like(slots.hosts[name], -1) for name in TABLE_FIELDS if name != "block_counts"}
    counts = torch.ones_like(slots.hosts["block_counts"])
    for value in tables.values():
        value[:, :, 0] = 0
    return slots, worker, program, tables, counts


def stage(slots, program, tables, counts, generation=1, end=2):
    slots.acquire(generation, program)
    return slots.stage(tables, counts, token=42, end=end)


def test_decode_transfers_reuse_registered_storage_and_upload_only_changed_fields():
    slots, worker, program, tables, counts = setup_slots()
    addresses = {name: value.data_ptr() for name, value in slots.hosts.items()}
    assert stage(slots, program, tables, counts) is slots.arguments
    assert len(worker.calls) == 88
    assert all(value.is_shared() for value in slots.inherited)
    ranges = [(h.data_ptr(), h.data_ptr()+h.numel()*h.element_size()) for h in slots.inherited]
    assert all(any(first <= ptr and ptr+size <= end for first, end in ranges)
               for _, ptr, size, _ in worker.calls)

    slots.complete()
    worker.calls.clear()
    stage(slots, program, tables, counts, end=3)
    assert slots.last_uploads == DYNAMIC_FIELDS and len(worker.calls) == 24
    assert slots.hosts["position_ids"].flatten().tolist() == [2]*8
    assert slots.hosts["kv_seq_lens"].flatten().tolist() == [3]*8

    slots.complete()
    tables["block_table"][:, :, 0] = 3
    tables["block_table"][:, :, 1:] = -1
    counts[:, :, 0] = 0
    worker.calls.clear()
    stage(slots, program, tables, counts)
    assert set(slots.last_uploads) == {*DYNAMIC_FIELDS, "block_table", "block_counts"}
    assert len(worker.calls) == 40
    assert torch.equal(slots.hosts["block_table"], tables["block_table"])

    slots.complete()
    worker.calls.clear()
    stage(slots, program, tables, counts, generation=2)
    assert len(worker.calls) == 88 and slots.generation == 2
    assert addresses == {name: value.data_ptr() for name, value in slots.hosts.items()}


def test_busy_slot_and_program_change_cannot_be_reused():
    slots, worker, program, tables, counts = setup_slots()
    stage(slots, program, tables, counts)
    with pytest.raises(RuntimeError, match="busy"):
        slots.acquire(1, program)
    with pytest.raises(RuntimeError, match="leased"):
        slots.invalidate_request()
    slots.complete()
    with pytest.raises(RuntimeError, match="another program"):
        slots.acquire(1, object())


def test_partial_copy_failure_never_acknowledges_field_or_releases_lease():
    slots, worker, program, tables, counts = setup_slots()
    worker.fail_after = 3
    with pytest.raises(RuntimeError, match="copy failed"):
        stage(slots, program, tables, counts)
    assert slots.poisoned and slots.busy
    assert not slots.uploaded_tables
    with pytest.raises(RuntimeError, match="poisoned"):
        slots.complete()
    with pytest.raises(RuntimeError, match="poisoned"):
        slots.acquire(2, program)


def test_changed_owner_is_revalidated_and_receives_subsequent_uploads():
    slots, worker, program, tables, counts = setup_slots()
    previous = slots.arguments
    bindings = dict(zip(slots.names, previous, strict=True))
    metadata = {"params": [dict(name=name, shape=list(value.shape), direction="In", layout="ND",
                               dtype="int64" if value.dtype == torch.int64 else "int32")
                           for name, value in bindings.items()]}
    replacement = StackedDeviceTensor(
        [DeviceTensor(0xA00000+rank*0x1000, (1,), torch.int64) for rank in range(8)],
        (8, 1), tuple(range(8)),
    )
    bindings["input_ids"] = replacement
    slots.refresh(bindings, metadata)
    assert slots.arguments is not previous
    assert slots.destination["input_ids"] is replacement
    stage(slots, program, tables, counts)
    assert all(any(destination == shard.data_ptr for destination, _, _, _ in worker.calls)
               for shard in replacement.shards)


def prefill_slots():
    rows = [("ori_block_table", [8, 1, 16], "int32"), ("position_ids_full", [8, -1], "int32"),
            ("query_start_loc", [8, 2], "int32"),
            ("input_ids", [8, -1], "int64"), ("swa_freqs_cos", [8, -1, 64], "bf16")]
    programs = SimpleNamespace(context=isolated_profile, prefill_metadata={"params": [
        dict(name=n,shape=s,dtype=d,direction="In",layout="ND") for n,s,d in rows]})
    return PrefillTransferSlots(programs, 16)


class PrefillWorker:
    def __init__(self, inherited):
        self.inherited=inherited
        self.storage={}
        self.copies=[]
        self.failure=False
        self.fail_after=None

    def alloc_tensor(self, shape, dtype, *, worker_id):
        host=torch.empty(shape,dtype=dtype)
        self.storage[host.data_ptr()]=host
        return DeviceTensor(host.data_ptr(),shape,dtype)

    def view_tensor(self, source, shape, *, byte_offset=0, worker_id):
        assert byte_offset+prod(shape)*source.dtype.itemsize <= source.nbytes
        return DeviceTensor(source.data_ptr+byte_offset,shape,source.dtype,
                            byte_offset=source.byte_offset+byte_offset)

    def copy_to(self, dst, src, size, *, worker_id):
        assert dst in self.storage
        assert any(root.data_ptr() <= src and src+size <= root.data_ptr()+root.numel()*root.element_size()
                   for root in self.inherited)
        if self.failure or (self.fail_after is not None and len(self.copies) >= self.fail_after):
            raise RuntimeError("copy failed")
        ctypes.memmove(dst,src,size)
        self.copies.append((dst,size,worker_id))

    def free_stacked_tensor(self, owner):
        for shard in owner.shards:
            assert shard.byte_offset == 0
            del self.storage[shard.data_ptr]


def prefill_inputs(length):
    return dict(ori_block_table=torch.arange(8*16,dtype=torch.int32).reshape(8,1,16),
                query_start_loc=torch.arange(16,dtype=torch.int32).reshape(8,2),
                position_ids_full=torch.arange(8*length,dtype=torch.int32).reshape(8,length),
                input_ids=torch.arange(8*length,dtype=torch.int64).reshape(8,length)+100,
                swa_freqs_cos=torch.arange(8*length*64,dtype=torch.float32).reshape(8,length,64).bfloat16())


def check_device(views, sources):
    for name, owner in views.items():
        assert tuple(owner.shape)==tuple(sources[name].shape)
        for rank, shard in enumerate(owner.shards):
            expected=sources[name][rank].contiguous().view(torch.uint8).numpy().tobytes()
            assert ctypes.string_at(shard.data_ptr,len(expected))==expected


def test_typed_roots_preserve_rank_values_and_reuse_stable_tables_across_ragged_chunks():
    transfers=prefill_slots();worker=PrefillWorker(transfers.inherited);arena=TensorArena(worker)
    assert all(root.is_shared() and root.is_contiguous() for root in transfers.inherited)
    transfers.bind(arena)
    assert sum(t.numel()*t.element_size() for t in worker.storage.values())==8*transfers.bytes_per_rank
    first=prefill_inputs(16);views=transfers.stage(first,revision=(1,0));check_device(views,first)
    assert len(worker.copies)==32
    assert any(field.shards[0].byte_offset for field in views.values())
    second=prefill_inputs(8);views=transfers.stage(second,revision=(1,0));check_device(views,second)
    assert len(worker.copies)==56
    second['ori_block_table'].add_(300)
    views=transfers.stage(second,revision=(1,1));check_device(views,second)
    assert len(worker.copies)==88
    arena.close()
    assert not worker.storage


@pytest.mark.parametrize("fail_after", [0, 1])
def test_prefill_copy_failure_requires_recovery_before_reusing_metadata(fail_after):
    transfers=prefill_slots();worker=PrefillWorker(transfers.inherited);arena=TensorArena(worker);transfers.bind(arena)
    worker.fail_after=fail_after
    with pytest.raises(RuntimeError,match='copy failed'): transfers.stage(prefill_inputs(8),revision=1)
    assert transfers.poisoned and not transfers.busy and transfers.table_revision is None
    worker.fail_after=None
    with pytest.raises(RuntimeError,match='poisoned'): transfers.stage(prefill_inputs(8),revision=2)
    transfers.recover_after_stage_failure()
    check_device(transfers.stage(prefill_inputs(8), revision=2), prefill_inputs(8))
    arena.close()


@pytest.fixture
def native_inputs(monkeypatch):
    monkeypatch.setattr(NativeInputs, "MAX_ANCHORS", 64)
    banks = (
        ("context_group_position_ids", (512,), "int32"),
        ("context_group_slot_mapping", (3, 512), "int64"),
        ("query_group_position_ids", (80,), "int32"),
        ("query_group_slot_mapping", (3, 80), "int64"),
        ("context_group_freqs_cos", (512, 64), "bfloat16"),
        ("context_group_freqs_sin", (512, 64), "bfloat16"),
        ("query_freqs_cos", (20, 64), "bfloat16"),
        ("query_freqs_sin", (20, 64), "bfloat16"),
        ("query_group_freqs_cos", (80, 64), "bfloat16"),
        ("query_group_freqs_sin", (80, 64), "bfloat16"),
    )
    return NativeInputs({"params": [
        row(f"d_{name}_bank", [4, -1, *shape], dtype) for name, shape, dtype in banks
    ]})


@pytest.mark.parametrize("prompt,output", [(7, 7), (16314, 64)])
def test_compact_banks_keep_absolute_context_and_query_positions(native_inputs, prompt, output):
    cos = torch.arange(16384, dtype=torch.bfloat16).unsqueeze(1).expand(-1, 64).contiguous()
    sin = -cos
    count = native_inputs.prepare_banks(prompt_tokens=prompt, max_output_tokens=output,
                                  rope_cos=cos, rope_sin=sin)
    assert count == output-1
    for index in (0, count-1):
        anchor = prompt-1+index
        expected = list(range(max(0, anchor-127), anchor+1))
        for rank in range(4):
            position = native_inputs.banks["d_context_group_position_ids_bank"][rank, index].reshape(4, 128)
            assert position[0, :len(expected)].tolist() == expected
            assert position[0, len(expected):].eq(-1).all()
            assert position[1:, 0].eq(0).all() and position[1:, 1:].eq(-1).all()
            query = native_inputs.banks["d_query_group_position_ids_bank"][rank, index].reshape(4, 20)
            assert query[0, :5].tolist() == list(range(anchor+1, anchor+6))
            assert query[1:, :5].tolist() == [list(range(1, 6))]*3
            slots = native_inputs.banks["d_context_group_slot_mapping_bank"][rank, index].reshape(3, 4, 128)
            for layer in range(3):
                for row, absolute in enumerate(expected):
                    page = int(native_inputs.tables[0, layer, 0, absolute//32])
                    assert int(slots[layer, 0, row]) == page*32+absolute%32
            actual_cos = native_inputs.banks["d_context_group_freqs_cos_bank"][rank, index, :len(expected)]
            assert torch.equal(actual_cos, cos[expected])
            actual_sin = native_inputs.banks["d_query_group_freqs_sin_bank"][rank, index, :5]
            assert torch.equal(actual_sin, sin[anchor+1:anchor+6])


def test_native_bank_uploads_use_prefork_backings_and_request_extent(native_inputs):
    calls = []
    class Worker:
        def view_tensor(self, owner, shape, worker_id):
            return DeviceTensor(owner.data_ptr, shape, owner.dtype)
        def copy_to(self, destination, source, nbytes, worker_id):
            calls.append((source, nbytes))
    worker = Worker()
    destinations = {}
    for index, (name, host) in enumerate(native_inputs.banks.items()):
        shards = [DeviceTensor(0x1000000+index*0x100000+r*0x10000, host.shape[1:], host.dtype)
                  for r in range(4)]
        destinations[name] = StackedDeviceTensor(shards, host.shape, tuple(range(4)))
    views = native_inputs.upload_banks(SimpleNamespace(worker=worker), destinations, 3)
    assert len(calls) == 40
    spans = [(h.data_ptr(), h.data_ptr()+h.numel()*h.element_size()) for h in native_inputs.banks.values()]
    assert all(any(lo <= p and p+n <= hi for lo, hi in spans) for p, n in calls)
    assert all(value.shape[1] == 3 and value.worker_ids == tuple(range(4)) for value in views.values())


def prefill_runner(monkeypatch, phase=None):
    programs = SimpleNamespace(single=SimpleNamespace(compiled=object()), single_metadata={},
                               prefill=SimpleNamespace(compiled=object()), prefill_metadata={},
                               context=nullcontext)
    instance = LowlatModelRunner(programs, None)
    instance.state = LowlatRequestState("r", 1, 8)
    instance.generation, instance.chunk_capacity = 1, 8
    instance.run_config = object()
    instance.rope_cos, instance.rope_sin = None, None
    instance.cache = SimpleNamespace(tables_for=Mock(return_value=({}, object())), release=Mock(), owners={})
    instance.worker = Mock()
    handle = Mock()
    instance.worker.submit.return_value = handle
    instance.transfers = Mock(poisoned=False)
    instance.transfers.poison.side_effect = lambda: setattr(instance.transfers, "poisoned", True)
    instance.prefill_bindings = {name: SimpleNamespace(shape=(8, 1)) for name in
        ("sampled_ids", "attn_stage", "x_out", "dspark_target_hidden", "input_ids")}
    instance.weights = SimpleNamespace(lookup_embeddings=Mock(), prefill_bindings=Mock(return_value={}))
    instance.prefill_input = Mock()
    instance.prefill_input.fill.return_value = torch.zeros((8, 8, 4, 4096), dtype=torch.float32)
    instance.arena = Mock()
    instance._stage = Mock(return_value={"input_ids": object()})
    instance._read_sample = Mock(return_value=42)
    helper = SimpleNamespace(prepare_prefill_metadata=lambda *a, **k: {
        "input_ids_full": torch.zeros((8, 8), dtype=torch.int64),
        "local_request_ids": torch.zeros((8, 1), dtype=torch.int32)})
    def submit(worker, *args, **kwargs):
        if phase != "no_submit":
            accepted = worker.submit(programs.prefill.compiled)
            if phase != "no_wait":
                accepted.result()
        return {"sampled_ids": object()}
    adapter = SimpleNamespace(prepare_prefill_bindings=lambda *a: {}, submit_prefill=submit)
    monkeypatch.setattr(npu_runner.importlib, "import_module", lambda name: helper if name.endswith("prefill_metadata") else adapter)
    monkeypatch.setattr(npu_runner, "borrow_view", lambda *a: object())
    if phase == "cache_init":
        instance.cache.tables_for.side_effect = RuntimeError(phase)
    elif phase == "copy":
        instance._stage.side_effect = RuntimeError(phase)
    elif phase == "submit":
        instance.worker.submit.side_effect = RuntimeError(phase)
    elif phase == "result":
        handle.result.side_effect = RuntimeError(phase)
    elif phase == "capture":
        instance._read_sample.side_effect = RuntimeError(phase)
    batch = PrefillBatch(["r"], torch.arange(8), None, [8], [8], [0], [0],
                         prompt_lens=[8], block_ids_by_group=[{}])
    return instance, batch, handle


def test_prefill_publishes_and_captures_before_committing_cursor_and_releasing_handle(monkeypatch):
    instance, batch, handle = prefill_runner(monkeypatch)
    def complete():
        assert instance.state.pending_handles == [handle]
        assert instance.state.prefilled_tokens == 0
    handle.result.side_effect = complete
    def capture(owner):
        assert instance.state.pending_handles == [handle]
        assert instance.state.prefilled_tokens == 0
        return 42
    instance._read_sample.side_effect = capture
    result = instance.run_prefill(None, batch)
    assert result.sampled_token_ids.tolist() == [42]
    assert instance.state.prefilled_tokens == instance.state.kv_committed_tokens == 8
    assert instance.state.pending_handles == []
    instance.transfers.poison.assert_not_called()
    instance.prefill_input.begin_copy.assert_called_once()
    instance.prefill_input.end_copy.assert_called_once()
    staged = instance._stage.call_args.args[1]
    assert staged["input_ids"].shape == (8, 8)
    assert "input_ids_full" not in staged
    instance.arena.copy.assert_not_called()
    instance.weights.prefill_bindings.assert_not_called()


@pytest.mark.parametrize("phase", ["cache_init", "copy", "submit", "result", "capture", "no_submit", "no_wait"])
def test_prefill_failure_poison_retains_accepted_handle_and_never_publishes_cursor(monkeypatch, phase):
    instance, batch, handle = prefill_runner(monkeypatch, phase)
    with pytest.raises(RuntimeError):
        instance.run_prefill(None, batch)
    assert instance.state.prefilled_tokens == instance.state.kv_committed_tokens == 0
    instance.transfers.poison.assert_called_once()
    assert instance.transfers.poisoned
    expected = [handle] if phase in ("result", "capture", "no_wait") else []
    assert instance.state.pending_handles == expected
    if expected:
        def drain():
            assert instance.state.pending_handles == [handle]
            instance.cache.release.assert_not_called()
        handle.result.side_effect = drain
    instance.release_finished_requests(["r"])
    instance.cache.release.assert_called_once()
    with pytest.raises(RuntimeError):
        instance._request(["next"], 1)


def test_stage_recovery_keeps_both_transfers_poisoned_if_decode_recovery_fails(monkeypatch):
    instance, batch, _ = prefill_runner(monkeypatch, "copy")
    instance.prefill_transfers = Mock(busy=False, poisoned=True)
    instance.prefill_transfers.recover_after_stage_failure.side_effect = (
        lambda: setattr(instance.prefill_transfers, "poisoned", False)
    )
    instance.prefill_transfers.poison.side_effect = lambda: setattr(
        instance.prefill_transfers, "poisoned", True
    )
    instance.transfers.busy = False
    instance.transfers.recover_after_prefill_failure.side_effect = RuntimeError("decode recovery failed")

    with pytest.raises(RuntimeError, match="decode recovery failed"):
        instance.run_prefill(None, batch)

    instance.prefill_transfers.recover_after_stage_failure.assert_called_once()
    instance.prefill_transfers.poison.assert_called_once()
    assert instance.prefill_transfers.poisoned
    instance.transfers.poison.assert_called()
    assert instance.transfers.poisoned
    assert not instance.state.pending_handles


def test_stage_recovery_skips_poison_only_after_both_transfers_reset(monkeypatch):
    instance, batch, _ = prefill_runner(monkeypatch, "copy")
    instance.prefill_transfers = Mock(busy=False, poisoned=True)
    instance.prefill_transfers.recover_after_stage_failure.side_effect = (
        lambda: setattr(instance.prefill_transfers, "poisoned", False)
    )
    instance.transfers.busy = False
    instance.transfers.recover_after_prefill_failure.side_effect = (
        lambda: setattr(instance.transfers, "poisoned", False)
    )

    with pytest.raises(RuntimeError, match="copy"):
        instance.run_prefill(None, batch)

    instance.prefill_transfers.recover_after_stage_failure.assert_called_once()
    instance.transfers.recover_after_prefill_failure.assert_called_once()
    instance.transfers.poison.assert_not_called()
    assert not instance.prefill_transfers.poisoned
    assert not instance.transfers.poisoned


@pytest.mark.parametrize("prompt,chunk", [(7, 7), (513, 512)])
def test_cp_tap_seed_preserves_absolute_rows_across_chunks_and_padding(prompt, chunk):
    seed = NativeHistorySeed()
    seed.begin(1)
    copies = []
    for start in range(0, prompt, chunk):
        count = min(chunk, prompt-start)
        padded = (count+7)//8*8
        positions = torch.arange(start, start+padded, dtype=torch.int64)
        positions[count:] = -1
        positions = positions.reshape(8, -1)
        backing = []
        shards = []
        bases = {}
        for rank, pos in enumerate(positions):
            # A borrowed source retains a nonzero offset from its allocation base.
            host = torch.zeros((16+pos.numel()*12288,), dtype=torch.bfloat16)
            values = pos.to(torch.bfloat16).unsqueeze(1).expand(-1, 12288).contiguous()
            host[16:].copy_(values.reshape(-1))
            backing.append(host)
            bases[rank] = host.data_ptr()
            shards.append(DeviceTensor(host.data_ptr()+32, tuple(values.shape), torch.bfloat16,
                                       byte_offset=32))
        owner = StackedDeviceTensor(shards, (8, positions.shape[1], 12288), tuple(range(8)))
        class Worker:
            def copy_from(self, destination, source, nbytes, *, src_offset, worker_id):
                assert source == bases[worker_id]
                assert seed.host.data_ptr() <= destination
                assert destination+nbytes <= seed.host.data_ptr()+seed.host.numel()*2
                copies.append((worker_id, src_offset, nbytes))
                ctypes.memmove(destination, source+src_offset, nbytes)
        seed.capture(Worker(), owner, positions, prompt_length=prompt, generation=1)
    result = seed.ready(prompt_length=prompt, generation=1)
    for position in range(max(0, prompt-128), prompt):
        expected = torch.full((12288,), position, dtype=torch.bfloat16)
        assert torch.equal(result[position%128], expected)
    assert not result[128:].any()
    assert copies and max(n for _, _, n in copies) <= 128*12288*2
    seed.release()
    seed.begin(2)
    with pytest.raises(RuntimeError, match="cover"):
        seed.ready(prompt_length=prompt, generation=2)
    with pytest.raises(RuntimeError, match="generation"):
        seed.capture(None, owner, positions, prompt_length=prompt, generation=1)


def test_real_chunk_adapter_retains_host_local_ids_without_dispatching_them(monkeypatch):
    from unittest.mock import Mock

    from pypto.runtime import DeviceTensor, StackedDeviceTensor
    with isolated_profile():
        from models.deepseek_v4_flash_lowlat import prefill_fwd as adapter

        def device(shape, offset):
            return StackedDeviceTensor([DeviceTensor(offset+rank*0x10000000, shape[1:], torch.float32)
                for rank in range(8)],shape,tuple(range(8)))
        bindings=dict(x_hc=device((8,8,4,4096),0x100000),attn_stage=device((8,8,4,4096),0x200000),
            input_ids_full=torch.zeros((8,8),dtype=torch.int64),position_ids_full=torch.zeros((8,8)),
            position_ids_local=torch.zeros((8,1)),local_request_ids=torch.zeros((8,1),dtype=torch.int32),
            target_hidden=device((8,1,12288),0x300000))
        for name in ('ori','hca_cmp','hca_state','csa_cmp','csa_idx','csa_state','csa_inner_state'):
            bindings[name+'_slot_mapping_full']=torch.zeros((8,8),dtype=torch.int32)
        submitted=Mock(return_value=object());monkeypatch.setattr(adapter,'_submit_chunk',submitted)
        result=adapter.submit_prefill(object(),{'chunk':object()},{'chunk':{}},bindings,
                                      group_tokens=8,terminal=True)
        assert result is submitted.return_value
        assert bindings['local_request_ids'].device.type=='cpu'
        submitted.assert_called_once()
        assert "local_request_ids" not in adapter.load_prefill_programs()["chunk"].param_names
        del bindings['local_request_ids']
        with pytest.raises(KeyError,match='local_request_ids'):
            adapter.submit_prefill(object(),{'chunk':object()},{'chunk':{}},bindings,
                                   group_tokens=8,terminal=True)
        submitted.assert_called_once()


# Native publication, frame ownership and asynchronous execution.

def packet(state, *, accepted=0):
    advance = accepted+1
    actual = min(advance, state.remaining)
    ids = tuple(range(100, 100+actual))
    bonus = 100+accepted
    control = [state.epoch, 1, state.anchor, accepted, advance, actual, state.anchor+advance,
               bonus, ids[-1], state.remaining-actual, 1, state.anchor+1, advance,
               int(state.remaining == actual), 0, 0]
    return control, list(ids)+[-1]*(6-actual)


def published(state, *, cycles=1, accepted=0):
    control = torch.zeros((8, 2, 16), dtype=torch.int64)
    emitted = torch.full((8, 2, 6), -1, dtype=torch.int64)
    current = state
    for i in range(cycles):
        c, e = packet(current, accepted=accepted)
        control[0, i] = torch.tensor(c)
        emitted[0, i] = torch.tensor(e)
        current = NativeState(c[0]+1, c[6], c[9], c[8])
    return control, emitted


@pytest.mark.parametrize("remaining,accepted", [(1, 5), (6, 0), (7, 2), (12, 5)])
def test_native_actual_emission_and_physical_advance_are_separate(remaining, accepted):
    state = NativeState(9, 511, remaining, 42)
    control, emitted = published(state, accepted=accepted)
    result = validate_packets(control, emitted, state=state, cycles=1)
    assert len(result.token_ids) == min(remaining, accepted+1)
    assert result.state.anchor == state.anchor+accepted+1
    assert result.state.remaining == remaining-len(result.token_ids)
    assert result.accepted == (accepted,)
    # Unused cycle1 has no publication contract; do not inspect allocator residue.
    control[:, 1].fill_(-777)
    emitted[:, 1].fill_(-777)
    assert validate_packets(control, emitted, state=state, cycles=1) == result


@pytest.mark.parametrize("remaining", [7, 12])
def test_two_real_cycles_close_the_predecessor_state(remaining):
    state = NativeState(1, 127, remaining, 5)
    control, emitted = published(state, cycles=2, accepted=5)
    result = validate_packets(control, emitted, state=state, cycles=2)
    assert len(result.token_ids) == min(remaining, 12)
    assert result.advances == (6, 6)
    assert result.state.epoch == 3


@pytest.mark.parametrize("field", [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15])
def test_corrupt_commit_never_returns_a_state(field):
    state = NativeState(1, 127, 64, 9)
    control, emitted = published(state, accepted=2)
    control[0, 0, field] += 1
    with pytest.raises(ValueError):
        validate_packets(control, emitted, state=state, cycles=1)


def test_nonowner_clear_and_unused_output_contracts():
    state = NativeState(1, 0, 64, 9)
    control, emitted = published(state)
    emitted[0, 0, 5] = 7
    with pytest.raises(ValueError, match="emitted"):
        validate_packets(control, emitted, state=state, cycles=1)
    control, emitted = published(state)
    control[7, 0, 0] = 1
    with pytest.raises(ValueError, match="nonowner"):
        validate_packets(control, emitted, state=state, cycles=1)
    control, emitted = published(state)
    with pytest.raises(ValueError, match="budget"):
        validate_packets(control, emitted, state=NativeState(1, 0, 6, 9), cycles=2)


def test_native_capacity_admission_reserves_final_lookahead():
    validate_native_capacity(prompt_tokens=16383, max_output_tokens=1, physical_capacity=16384)
    validate_native_capacity(prompt_tokens=16314, max_output_tokens=64, physical_capacity=16384)
    with pytest.raises(ValueError, match="lookahead"):
        validate_native_capacity(prompt_tokens=16315, max_output_tokens=64, physical_capacity=16384)


def adapter():
    hosts = allocate_native_hosts(2)
    rows = [{"name": f"input{i}", "shape": [8, 1], "dtype": "int32", "direction": "In"}
            for i in range(150)]
    rows += [{"name": "scratch", "shape": [8, 1], "dtype": "int32", "direction": "Out"},
             {"name": "s_control", "shape": [8, 2, 16], "dtype": "int64", "direction": "Out"},
             {"name": "s_emitted", "shape": [8, 2, 6], "dtype": "int64", "direction": "Out"}]
    def tensor(base):
        return StackedDeviceTensor([DeviceTensor(base+r*256, (1,), torch.int32,
                  buffer=SimpleNamespace(base=base+r*256, size=4, tensor=lambda *a: None)) for r in range(8)],
                  (8, 1), tuple(range(8)))
    common = {r["name"]: tensor(0x10000+i*0x1000) for i, r in enumerate(rows[:150])}
    outputs = [{"scratch": tensor(0x1000000+i*0x1000)} for i in range(2)]
    class Worker:
        def __init__(self):
            self._inherited_host_spans = [((v.data_ptr(), v.data_ptr()+v.numel()*v.element_size()), v)
                                          for h in hosts for v in h.values()]
            self._states = {p: {"param_infos": rows} for p in ("one", "two")}
            self.handle = SimpleNamespace(result=lambda: None)
        def _validate_prepared_dispatch_arg(self, info, value):
            pass
        def submit(self, *args, **kwargs):
            return self.handle
    worker = Worker()
    instance = NativeFrames(worker, {1: "one", 2: "two"}, {1: {"params": rows}, 2: {"params": rows}},
                            common, outputs, hosts, None)
    return instance, hosts, outputs


def test_frame_cannot_be_captured_or_reused_before_consumption():
    instance, hosts, _ = adapter()
    state = NativeState(1, 7, 6, 42)
    instance.submit(0, generation=1, cycles=1)
    with pytest.raises(RuntimeError, match="publication"):
        instance.capture(0, generation=1, state=state)
    with pytest.raises(RuntimeError, match="unavailable"):
        instance.submit(0, generation=1, cycles=1)
    control, emitted = published(state, accepted=5)
    hosts[0]["s_control"].copy_(control)
    hosts[0]["s_emitted"].copy_(emitted)
    instance.complete(0)
    result = instance.capture(0, generation=1, state=state)
    with pytest.raises(RuntimeError, match="owned"):
        instance.release(0)
    instance.consume(0)
    instance.release(0)
    hosts[0]["s_emitted"].fill_(-777)
    assert result.token_ids == tuple(range(100, 106))
    assert instance.idle()
    instance.submit(0, generation=2, cycles=1)
    instance.complete(0)
    with pytest.raises(RuntimeError, match="generation"):
        instance.capture(0, generation=1, state=state)


def test_failed_parent_keeps_frame_and_device_outputs_owned():
    instance, _, outputs = adapter()
    def fail():
        raise RuntimeError("device failure")
    instance.worker.handle.result = fail
    instance.submit(0, generation=1, cycles=1)
    with pytest.raises(RuntimeError, match="device failure"):
        instance.complete(0)
    assert instance.poisoned and instance.frames[0].leased
    assert instance.frames[0].handle is instance.worker.handle
    assert outputs[0]["scratch"] in instance.frames[0].arguments
    with pytest.raises(RuntimeError):
        instance.release(0)
    with pytest.raises(RuntimeError):
        instance.submit(1, generation=1, cycles=1)


def runner(remaining=24):
    instance = NativePipelineRunner.__new__(NativePipelineRunner)
    instance._closed = False
    instance.state = LowlatRequestState("r", 2, 128, 128, 128, 9, "decode")
    instance.request_contexts = {"r": RequestExecutionContext("r", 128, remaining+1, execution_generation=7)}
    instance.cache = SimpleNamespace(specs={"ori": object()}, _page_ids={"ori": (4, 2)}, release=Mock())
    instance.transfers = Mock(poisoned=False)
    instance._native_lock = threading.RLock()
    instance._accepted, instance._seen_reservations = [], set()
    frames, hosts, _ = adapter()
    frames.worker.submit = Mock(side_effect=lambda *a, **k: Mock())
    instance.native = SimpleNamespace(frames=frames, state=NativeState(1, 127, remaining, 9), release=Mock())
    instance.native.commit = lambda packet: setattr(instance.native, "state", packet.state)
    reservation = DecodeReservation(1, "r", 7, 2 if remaining >= 7 else 1,
        min(12 if remaining >= 7 else 6, remaining), 128+remaining+1+6, remaining, 0)
    batch = DecodeBatch(["r"], torch.tensor([0]), None, torch.tensor([999]),
        block_ids_by_group=[{"ori": [4, 2]}],
        model_extensions=(encode_reservation(reservation),))
    return instance, batch, hosts


def publish(hosts, index, state, cycles=2):
    control, emitted = published(state, cycles=cycles, accepted=5)
    hosts[index]["s_control"].copy_(control)
    hosts[index]["s_emitted"].copy_(emitted)
    return validate_packets(control, emitted, state=state, cycles=cycles)


def accepted_pair(instance, batch):
    first = instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    second_reservation = replace(decode_reservation_payload(batch.model_extensions[0]), reservation_id=2, remaining_lower_bound=12, frame_id=1)
    second_batch = replace(batch, model_extensions=(encode_reservation(second_reservation),))
    second = instance.dispatch_prepared_decode(None, second_batch, instance.prepare_decode(None, second_batch))
    return first, second


def test_two_accepted_parents_capture_independent_frames_and_return_real_retirement_receipts():
    instance, batch, hosts = runner()
    first, second = accepted_pair(instance, batch)
    before = instance.native.state
    packet = publish(hosts, 0, before)
    publish(hosts, 1, packet.state)
    output_a = instance.reclaim_prepared_decode(first)
    assert decode_decode_extension(output_a.model_extension).retired_reservation_ids == (1,) and len(output_a.accepted_token_ids[0]) == 12
    assert output_a.num_draft_tokens is None
    assert instance._accepted == [second]
    hosts[0]["s_emitted"].fill_(-777)
    output_b = instance.reclaim_prepared_decode(second)
    assert decode_decode_extension(output_b.model_extension).retired_reservation_ids == (2,) and len(output_b.accepted_token_ids[0]) == 12
    assert output_a.accepted_token_ids[0][0] == 100
    assert instance.native.state.remaining == 0 and instance.state.kv_committed_tokens == 152
    assert instance.state.pending_handles == [] and instance.native.frames.idle()
    instance.transfers.stage_tables.assert_not_called()


def test_second_parent_can_be_accepted_while_first_result_waits_without_holding_admission_lock():
    instance, batch, hosts = runner()
    first = instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    entered, release = threading.Event(), threading.Event()
    def wait():
        entered.set()
        assert release.wait(5)
    first.handle.result.side_effect = wait
    packet = publish(hosts, 0, instance.native.state)
    errors = []
    def reclaim():
        try:
            instance.reclaim_prepared_decode(first)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=reclaim)
    thread.start()
    assert entered.wait(2)
    successor = replace(decode_reservation_payload(batch.model_extensions[0]), reservation_id=2, remaining_lower_bound=12, frame_id=1)
    second_batch = replace(batch, model_extensions=(encode_reservation(successor),))
    second = instance.dispatch_prepared_decode(None, second_batch, instance.prepare_decode(None, second_batch))
    publish(hosts, 1, packet.state)
    release.set()
    thread.join(5)
    assert not thread.is_alive() and not errors
    instance.reclaim_prepared_decode(second)


def test_end_request_captures_and_discards_both_accepted_parents_before_releasing_cache():
    instance, batch, hosts = runner()
    accepted_pair(instance, batch)
    packet = publish(hosts, 0, instance.native.state)
    publish(hosts, 1, packet.state)
    instance.native.release.side_effect = lambda: instance.native.frames.idle() or pytest.fail("live frame")
    instance.release_finished_requests(["r"])
    assert instance.state is None and instance._accepted == []
    instance.cache.release.assert_called_once()
    assert instance.native.frames.idle()


@pytest.mark.parametrize("change", ["generation", "pages", "duplicate", "third", "same_frame"])
def test_stale_or_unowned_reservation_never_submits(change):
    instance, batch, _ = runner()
    plan = instance.prepare_decode(None, batch)
    if change == "generation":
        plan = replace(plan, cache_generation=99)
    elif change == "pages":
        plan = replace(plan, pages=(("ori", (0, 1)),))
    else:
        first = instance.dispatch_prepared_decode(None, batch, plan)
        if change == "third":
            successor = replace(plan.reservation, reservation_id=2, frame_id=1, remaining_lower_bound=12)
            instance.dispatch_prepared_decode(None, batch, replace(plan, reservation=successor))
        if change == "same_frame":
            plan = replace(plan, reservation=replace(plan.reservation, reservation_id=2))
        assert first in instance.state.pending_handles
    with pytest.raises(RuntimeError):
        instance.dispatch_prepared_decode(None, batch, plan)


def test_corrupt_successor_keeps_its_owner_and_does_not_rewind_confirmed_predecessor():
    instance, batch, hosts = runner()
    first, second = accepted_pair(instance, batch)
    packet = publish(hosts, 0, instance.native.state)
    publish(hosts, 1, packet.state)
    instance.reclaim_prepared_decode(first)
    before = instance.native.state
    hosts[1]["s_control"][0, 0, 0] += 1
    with pytest.raises(ValueError):
        instance.reclaim_prepared_decode(second)
    assert instance.native.state == before and instance.native.frames.poisoned
    assert instance._accepted == [second] and second in instance.state.pending_handles
    instance.transfers.poison.assert_called_once()


def test_predecessor_can_retire_while_successor_submission_is_blocked():
    instance, batch, hosts = runner()
    first = instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    packet = publish(hosts, 0, instance.native.state)
    publish(hosts, 1, packet.state)
    entered, release = threading.Event(), threading.Event()
    def blocked_submit(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return Mock()
    instance.native.frames.worker.submit.side_effect = blocked_submit
    successor = replace(decode_reservation_payload(batch.model_extensions[0]), reservation_id=2, frame_id=1, remaining_lower_bound=12)
    second_batch = replace(batch, model_extensions=(encode_reservation(successor),))
    errors, accepted = [], []
    def dispatch():
        try:
            accepted.append(instance.dispatch_prepared_decode(None, second_batch,
                instance.prepare_decode(None, second_batch)))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=dispatch)
    thread.start()
    assert entered.wait(2)
    retired = threading.Event()
    def reclaim():
        try:
            instance.reclaim_prepared_decode(first)
        except BaseException as error:
            errors.append(error)
        finally:
            retired.set()
    output = threading.Thread(target=reclaim)
    output.start()
    try:
        assert retired.wait(2), "predecessor capture waited for successor submit"
        assert not errors and not accepted
        queued = instance._accepted[0]
        assert not queued.submission.done()
        assert instance.state.pending_handles == [queued]
        instance.cache.release.assert_not_called()
    finally:
        release.set()
        thread.join(5)
        output.join(5)
    assert not errors and not thread.is_alive() and not output.is_alive()
    instance.reclaim_prepared_decode(accepted[0])
    assert instance.native.frames.idle() and not instance.state.pending_handles


def test_failed_reserved_submission_keeps_cache_owned_until_worker_teardown():
    instance, batch, _ = runner()
    instance.native.frames.worker.submit.side_effect = RuntimeError("submit refused")
    with pytest.raises(RuntimeError, match="submit refused"):
        instance.dispatch_prepared_decode(None, batch, instance.prepare_decode(None, batch))
    pending = instance._accepted[0]
    assert pending in instance.state.pending_handles and pending.submission.done()
    assert instance.native.frames.poisoned
    with pytest.raises(RuntimeError, match="submit refused"):
        instance.release_finished_requests(["r"])
    instance.cache.release.assert_not_called()


@pytest.fixture
def recording(tmp_path):
    configure_profiler(create_profile_config(enabled=True, output=tmp_path, levels="verbose"))
    yield tmp_path
    configure_profiler(create_profile_config(enabled=False))


def test_real_pipeline_spans_preserve_two_handles_and_capture_before_frame_reuse(recording):
    instance, batch, hosts = runner()
    ids = iter((101, 102))
    handles = []
    def submit(*args, **kwargs):
        identity = next(ids)
        native = SimpleNamespace(_run_id=identity)
        handle = SimpleNamespace(_native_handle=native, _dispatch_id=identity,
                                 result=Mock(side_effect=lambda: setattr(native, "_run_id", None)))
        handles.append(handle)
        return handle
    instance.native.frames.worker.submit = Mock(side_effect=submit)
    first, second = accepted_pair(instance, batch)
    packet = publish(hosts, 0, instance.native.state)
    publish(hosts, 1, packet.state)
    a = instance.reclaim_prepared_decode(first)
    b = instance.reclaim_prepared_decode(second)
    assert len(a.accepted_token_ids[0]) == len(b.accepted_token_ids[0]) == 12
    assert instance.native.frames.idle()
    assert all(h.result.call_count == 1 for h in handles)
    merge_profile()
    spans = [e for e in json.loads((recording/"trace.json").read_text())["traceEvents"] if e.get("ph") == "X"]
    submitted = [e for e in spans if e["name"] == "lowlat.native.submit"]
    assert [e["args"]["runtime_run_id"] for e in submitted] == [101, 102]
    waits = [e for e in spans if e["name"] == "lowlat.native.wait"]
    assert [e["args"]["runtime_run_id"] for e in waits] == [101, 102]
    captures = [e for e in spans if e["name"] == "lowlat.native.capture"]
    assert [e["args"]["emitted"] for e in captures] == [12, 12]
    assert all(e["args"]["advances"] == [6, 6] for e in captures)
    retired = [e for e in spans if e["name"] == "lowlat.native.retire"]
    assert [e["args"]["reservation_id"] for e in retired] == [1, 2]
    assert all(e["args"]["success"] is True for e in spans)


def test_disabled_instrumentation_does_not_inspect_owner_or_arguments(monkeypatch):
    monkeypatch.setattr("pypto_serving.model.deepseek_lowlat.profiling.is_enabled", lambda *a: False)
    class Opaque:
        def __getattribute__(self, key):
            raise AssertionError("unexpected profiling inspection")
    token = object()
    @operation("test")
    def f(owner, value):
        return value
    assert f(Opaque(), token) is token
