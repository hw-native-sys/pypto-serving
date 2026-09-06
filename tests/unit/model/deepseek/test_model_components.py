# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

import pypto_serving.cli.main as cli
from pypto_serving.config.types import (
    PREFILL_CHUNK_SIZE_CHOICES,
    DecodeBatch,
    PrefillBatch,
    RuntimeConfig,
    SamplingParams,
)
from pypto_serving.model import model_loader
from pypto_serving.model import tokenizer as tokenizer_module
from pypto_serving.model.deepseek import npu_executor, weight_loader
from pypto_serving.model.deepseek import task_args as task_args_module
from pypto_serving.model.deepseek.npu_runner import (
    DEEPSEEK_V4_LM_HEAD_TP_SIZE,
    DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS,
    DeepSeekV4CacheLayout,
    DeepSeekV4CacheMetadataBuilder,
    DeepSeekV4CompiledKernels,
    DeepSeekV4L3Callable,
    DeepSeekV4ModelRunner,
    DeepSeekV4PreparedDecodeInputs,
    accept_mtp_tokens,
    build_deepseek_v4_cache_group_specs,
    build_deepseek_v4_layer_plan,
    deepseek_v4_cache_blocks_for_slots,
    deepseek_v4_decode_layout,
)
from pypto_serving.model.common.runner.buffer_set import StaticDeviceTensor
from pypto_serving.model.deepseek.task_args import (
    _DECODE_FWD_TENSOR_ORDER,
    _FUSED_MTP_DECODE_TENSOR_ORDER,
    _MTP_DECODE_TENSOR_ORDER,
    _MTP_PREFILL_TENSOR_ORDER,
    _PREFILL_FWD_TENSOR_ORDER,
    DeepSeekPrefillTaskArgs,
    decode_task_args,
    mtp_decode_task_args,
    mtp_prefill_task_args,
    prefill_task_args,
)
from pypto_serving.model.deepseek.weight_loader import (
    DEEPSEEK_V4_PACKED_FORMAT,
    DeepSeekV4StackedLayerWeights,
    DeepSeekV4WeightStore,
    deepseek_v4_layer_core_weight_names,
    deepseek_v4_packed_weights_path,
    deepseek_v4_hadamard_idx,
    deepseek_v4_local_expert_ids,
    deepseek_v4_routed_expert_weight_names,
    deepseek_v4_startup_weight_names,
    pack_deepseek_v4_layer_weights,
)
from pypto_serving.model.model_loader import ModelLoader
from pypto_serving.tools import prepack_deepseek_v4


def _deepseek_serving_contract(
    *,
    prefill_tile_tokens: int = 128,
    max_prefill_tokens_per_request: int | None = None,
):
    if max_prefill_tokens_per_request is None:
        max_prefill_tokens_per_request = 64 * prefill_tile_tokens

    def padded_prefill_tokens(active_tokens: int) -> int:
        if active_tokens <= 0 or active_tokens > max_prefill_tokens_per_request:
            raise ValueError("invalid active prefill extent")
        return (
            (active_tokens + prefill_tile_tokens - 1) // prefill_tile_tokens
        ) * prefill_tile_tokens

    return SimpleNamespace(
        schema_version="1",
        prefill_tile_tokens=prefill_tile_tokens,
        max_prefill_tokens_per_request=max_prefill_tokens_per_request,
        max_prefill_requests_per_partition=1,
        requires_homogeneous_prefill_decode=True,
        padded_prefill_tokens=padded_prefill_tokens,
    )


class _CountingPagedOriMetadata:
    def __init__(self, delegate):
        self._delegate = delegate
        self.table_calls = 0
        self.swa_calls = 0

    def paged_ori_block_table_from_ids(self, rows):
        self.table_calls += 1
        return self._delegate.paged_ori_block_table_from_ids(rows)

    def swa_window_indices_and_lens_from_ids(self, rows, positions):
        self.swa_calls += 1
        return self._delegate.swa_window_indices_and_lens_from_ids(rows, positions)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def test_deepseek_kernel_dir_uses_v4_flash_variant(tmp_path):
    kernel_dir = tmp_path / "models" / "deepseek_v4_flash_mtp"
    kernel_dir.mkdir(parents=True)

    assert npu_executor._find_pypto_lib_deepseek_v4_dir(str(tmp_path)) == kernel_dir
    assert npu_executor._is_deepseek_v4_module_file(kernel_dir / "decode_fwd.py", kernel_dir)


def test_deepseek_kernel_import_respecializes_sample_module(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "models" / "deepseek_v4_flash_mtp"
    kernel_dir.mkdir(parents=True)
    sample_file = kernel_dir / "sample.py"
    sample_file.write_text("SAMPLE_ROWS = 16\n", encoding="utf-8")
    stale_sample = ModuleType("sample")
    stale_sample.__file__ = str(sample_file)
    stale_sample.SAMPLE_ROWS = 8
    monkeypatch.setitem(sys.modules, "sample", stale_sample)

    with npu_executor._deepseek_v4_import_context(
        kernel_dir,
        pypto_lib_root=tmp_path,
        ep=8,
    ):
        reloaded_sample = importlib.import_module("sample")
        assert reloaded_sample is not stale_sample
        assert reloaded_sample.SAMPLE_ROWS == 16

    assert sys.modules["sample"] is stale_sample


def _pypto_lib_l3_arg_names(module_name: str, function_name: str) -> tuple[str, ...]:
    kernel_file = (
        Path(__file__).resolve().parents[4]
        / "pypto-lib"
        / "models"
        / "deepseek_v4_flash_mtp"
        / f"{module_name}.py"
    )
    module = ast.parse(kernel_file.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return tuple(arg.arg for arg in function.args.args)


def test_deepseek_decode_task_arg_orders_match_pypto_lib_abis():
    assert _pypto_lib_l3_arg_names("decode_fwd", "l3_decode_fwd") == _DECODE_FWD_TENSOR_ORDER
    assert _pypto_lib_l3_arg_names("decode_mtp", "l3_decode_mtp") == (
        *_MTP_DECODE_TENSOR_ORDER,
        "num_tokens",
    )

    fused_mtp_only = tuple(
        f"mtp_{name}"
        for name in _FUSED_MTP_DECODE_TENSOR_ORDER
        if name not in task_args_module._FUSED_MTP_SHARED_TENSORS
    )
    assert _pypto_lib_l3_arg_names("decode_fwd_mtp", "l3_decode_fwd_mtp") == (
        *_DECODE_FWD_TENSOR_ORDER,
        "mtp_tail_token_ids",
        "mtp_tail_positions",
        *fused_mtp_only,
        "mtp_num_tokens",
    )


def test_deepseek_decode_metadata_allows_shared_prefix_pages_across_requests():
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._compiled = SimpleNamespace(layout=SimpleNamespace(decode_batch=4))
    runner._cache_group_num_blocks = {"ori": 8}

    padded = runner._pad_group_block_ids(
        ((0, 1), (0, 2)),
        group_name="ori",
        kernel_rows=4,
    )

    assert padded == ((0, 1), (0, 2), (8,), (9,))
    with pytest.raises(ValueError, match="row must not repeat"):
        runner._pad_group_block_ids(
            ((0, 0),),
            group_name="ori",
            kernel_rows=4,
        )


def test_deepseek_rope_profiles_keep_swa_and_compressed_layers_distinct():
    class Utils:
        @staticmethod
        def build_rope_tables(_config, compress_ratio, *, dtype):
            return (
                torch.full((2, 2), float(compress_ratio), dtype=dtype),
                torch.full((2, 2), float(compress_ratio + 1), dtype=dtype),
            )

    executor = npu_executor.DeepSeekV4PyptoExecutor(
        platform="a2a3sim",
        device_ids=tuple(range(8)),
        compile_kernels=False,
    )
    freqs_cos, freqs_sin = executor._build_rope_tables(
        Utils(),
        SimpleNamespace(FLASH=object()),
    )

    assert freqs_cos.shape == (2, 2, 2)
    assert freqs_sin.shape == (2, 2, 2)
    assert freqs_cos.dtype == torch.bfloat16
    assert freqs_sin.dtype == torch.bfloat16
    assert torch.equal(freqs_cos[0], torch.zeros((2, 2), dtype=torch.bfloat16))
    assert torch.equal(freqs_sin[0], torch.ones((2, 2), dtype=torch.bfloat16))
    assert torch.equal(freqs_cos[1], torch.full((2, 2), 4.0, dtype=torch.bfloat16))
    assert torch.equal(freqs_sin[1], torch.full((2, 2), 5.0, dtype=torch.bfloat16))


def test_accept_mtp_tokens_commits_second_main_token_only_on_draft_match():
    accepted = accept_mtp_tokens(
        torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
        torch.tensor([[11], [99]], dtype=torch.long),
    )

    assert accepted == [[11, 12], [21]]


def test_accept_mtp_tokens_commits_matching_prefix_and_one_target_token():
    accepted = accept_mtp_tokens(
        torch.tensor(
            [
                [11, 12, 13, 14],
                [21, 22, 23, 24],
                [31, 32, 33, 34],
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [11, 12, 13],
                [21, 99, 23],
                [99, 32, 33],
            ],
            dtype=torch.long,
        ),
    )

    assert accepted == [[11, 12, 13, 14], [21, 22], [31]]


def test_deepseek_mtp_proposer_reuses_recurrent_hidden_for_configured_depth(monkeypatch):
    runner, _model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = 3
    runner._mtp_request_states["req-a"] = SimpleNamespace(
        draft_token_id=10,
        draft_pre_hc_hidden=torch.zeros((4, 1)),
        draft_position=129,
    )
    calls = []

    monkeypatch.setattr(runner, "_initialize_mtp_drafts", lambda batch: None)

    def fake_step(model, batch, token_ids, previous_hidden, positions):
        calls.append((token_ids.tolist(), positions.tolist(), previous_hidden.clone()))
        return token_ids + 1, previous_hidden + 1

    monkeypatch.setattr(runner, "_run_mtp_token_step", fake_step)
    drafts = runner._propose_mtp_tokens(
        _model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[9]], dtype=torch.long),
            hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
            seq_lens=torch.tensor([129], dtype=torch.int32),
        ),
    )

    assert drafts.tolist() == [[10, 11, 12]]
    assert [positions for _, positions, _ in calls] == [[129], [130]]
    torch.testing.assert_close(calls[1][2], torch.ones((1, 4, 1)))


def test_deepseek_mtp_token_step_runs_one_request_per_rank_wave(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    layout = deepseek_v4_decode_layout(3)
    runner._compiled.layout = layout
    mtp_slots = {
        "hidden_states": torch.empty(
            (layout.ranks, layout.decode_tokens, 4),
            dtype=torch.bfloat16,
        ),
        "prev_pre_hc_hidden": torch.empty(
            (layout.ranks, layout.decode_tokens, 4, 4),
            dtype=torch.float32,
        ),
        "input_ids": torch.empty((layout.ranks, layout.decode_tokens), dtype=torch.long),
        "position_ids": torch.empty(
            (layout.ranks, layout.decode_tokens),
            dtype=torch.int32,
        ),
        "swa_slot_mapping": torch.empty(
            (layout.ranks, layout.decode_tokens),
            dtype=torch.long,
        ),
        "swa_indices": torch.empty(
            (layout.ranks, layout.decode_tokens, layout.sliding_window),
            dtype=torch.int32,
        ),
        "swa_lens": torch.empty(
            (layout.ranks, layout.decode_tokens),
            dtype=torch.int32,
        ),
        "next_pre_hc_hidden": torch.empty(
            (layout.ranks, layout.decode_tokens, 4, 4),
            dtype=torch.float32,
        ),
        "sampled_ids": torch.empty(
            (layout.ranks, layout.decode_tokens, 8),
            dtype=torch.int32,
        ),
        "logit_row_indices": torch.empty(
            (layout.ranks, layout.decode_tokens),
            dtype=torch.int32,
        ),
        "sampling_temperatures": torch.empty(
            (layout.ranks, layout.decode_tokens), dtype=torch.float32
        ),
        "sampling_top_ks": torch.empty(
            (layout.ranks, layout.decode_tokens), dtype=torch.int32
        ),
        "sampling_seeds": torch.empty(
            (layout.ranks, layout.decode_tokens), dtype=torch.int32
        ),
        "sampling_positions": torch.empty(
            (layout.ranks, layout.decode_tokens), dtype=torch.int32
        ),
    }
    runner._mtp_buffers = SimpleNamespace()
    runner._mtp_decode_task_args = [SimpleNamespace(tensors=mtp_slots)]
    indices_by_rank = ((0, 1),) + ((),) * (layout.ranks - 1)
    assignment = SimpleNamespace(
        ranks=(0, 0),
        local_rows=(0, 1),
        per_rank_counts=(2,) + (0,) * (layout.ranks - 1),
        indices_by_rank=indices_by_rank,
    )
    monkeypatch.setattr(runner, "_decode_assignment", lambda batch: assignment)
    counting_metadata = _CountingPagedOriMetadata(runner.cache_metadata)
    runner.cache_metadata = counting_metadata
    monkeypatch.setattr(
        runner,
        "prepare_mtp_target_inputs",
        lambda *_args, **_kwargs: pytest.fail("recurrent MTP must not build full decode metadata"),
    )
    monkeypatch.setattr(runner, "_mtp_decode_args", lambda: ())
    monkeypatch.setattr(runner, "_require_mtp_decode_callable", lambda: object())
    monkeypatch.setattr(
        runner,
        "_write_mtp_tail_hidden",
        lambda *_args: pytest.fail("standalone MTP must receive previous hidden directly"),
    )
    active_tokens = []

    def fake_run_l3(callable_spec, *args):
        active_tokens.append(int(args[-1]))
        call_index = len(active_tokens)
        mtp_slots["sampled_ids"][0, 0, 0] = 10 + 10 * (call_index - 1)
        mtp_slots["next_pre_hc_hidden"][0, 0].fill_(call_index)

    monkeypatch.setattr(runner, "_run_l3", fake_run_l3)
    batch = DecodeBatch(
        request_ids=["req-a", "req-b"],
        token_ids=torch.tensor([[9], [19]], dtype=torch.long),
        hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
        seq_lens=torch.tensor([129, 130], dtype=torch.int32),
        block_ids_by_group=_grouped_cache_rows(2),
    )
    token_ids = torch.tensor([10, 20], dtype=torch.long)
    previous_hidden = torch.stack((torch.full((4, 4), 3.0), torch.full((4, 4), 4.0)))
    positions = torch.tensor([129, 130], dtype=torch.int32)
    next_tokens, next_hidden = runner._run_mtp_token_step(
        model,
        batch,
        token_ids,
        previous_hidden,
        positions,
    )

    assert active_tokens == [1, 1]
    assert counting_metadata.swa_calls == 2 * layout.ranks
    assert next_tokens.tolist() == [10, 20]
    torch.testing.assert_close(next_hidden[:, 0, 0], torch.tensor([1.0, 2.0]))
    assert mtp_slots["hidden_states"].dtype == torch.bfloat16
    assert mtp_slots["swa_lens"][0, 0].item() == layout.sliding_window


def test_deepseek_recurrent_mtp_rebuilds_position_dependent_swa_metadata():
    runner, _model = _runner_for_prepared_inputs()
    layout = deepseek_v4_decode_layout(3)
    runner._compiled.layout = layout
    runner._mtp_decode_task_args = [
        SimpleNamespace(
            tensors={
                "swa_slot_mapping": torch.empty(
                    (layout.ranks, layout.decode_tokens), dtype=torch.long
                ),
                "swa_indices": torch.empty(
                    (layout.ranks, layout.decode_tokens, layout.sliding_window),
                    dtype=torch.int32,
                ),
                "swa_lens": torch.empty(
                    (layout.ranks, layout.decode_tokens), dtype=torch.int32
                ),
            }
        )
    ]
    counting_metadata = _CountingPagedOriMetadata(runner.cache_metadata)
    runner.cache_metadata = counting_metadata
    batch = DecodeBatch(
        request_ids=["req-a"],
        token_ids=torch.tensor([[9]], dtype=torch.long),
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        seq_lens=torch.tensor([129], dtype=torch.int32),
        block_ids_by_group=_grouped_cache_rows(1),
    )

    runner._stage_recurrent_mtp_swa_metadata(
        batch,
        request_indices=(0,),
        ranks=(0,),
        positions=torch.tensor([129], dtype=torch.int32),
    )
    first_mapping = runner._mtp_decode_task_args[0].tensors["swa_slot_mapping"].clone()
    runner._stage_recurrent_mtp_swa_metadata(
        batch,
        request_indices=(0,),
        ranks=(0,),
        positions=torch.tensor([130], dtype=torch.int32),
    )

    assert counting_metadata.swa_calls == 2 * layout.ranks
    slots = runner._mtp_decode_task_args[0].tensors
    assert slots["swa_slot_mapping"][0, 0] != first_mapping[0, 0]

    scratch_slot_start = runner._cache_group_num_blocks["ori"] * layout.block_size
    active_fillers = slots["swa_slot_mapping"][0, 1:]
    inactive_fillers = slots["swa_slot_mapping"][1]
    assert active_fillers.ge(scratch_slot_start).all()
    assert inactive_fillers.ge(scratch_slot_start).all()
    assert active_fillers.unique().numel() == active_fillers.numel()
    assert inactive_fillers.unique().numel() == inactive_fillers.numel()
    assert slots["swa_lens"][0, 1:].eq(1).all()
    assert slots["swa_lens"][1].eq(1).all()
    torch.testing.assert_close(
        slots["swa_indices"][0, 1:, 0].to(torch.long),
        active_fillers,
    )
    assert slots["swa_indices"][0, 1:, 1:].eq(-1).all()


def test_deepseek_mtp_target_verification_chunks_arbitrary_depth(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.layout = deepseek_v4_decode_layout(9)
    runner._compiled.num_speculative_tokens = 9
    prepared_chunks = []

    def fake_prepare(model, batch, *, token_rows, positions, active_width):
        chunk = SimpleNamespace(
            request_ids=tuple(batch.request_ids),
            token_rows=token_rows,
            positions=positions,
            active_width=active_width,
        )
        prepared_chunks.append(chunk)
        return chunk

    def fake_execute(model, prepared, *, active_seq):
        ranks = tuple(range(len(prepared.request_ids)))
        logits = torch.zeros((8, active_seq, 128), dtype=torch.float32)
        sampled_ids = torch.zeros((8, active_seq, 8), dtype=torch.int32)
        pre_hc = torch.zeros((8, active_seq, 4, 1), dtype=torch.float32)
        for row, token_row in enumerate(prepared.token_rows):
            predictions = token_row[:active_seq] + 1
            logits[row, torch.arange(active_seq), predictions] = 1
            sampled_ids[row, :, 0] = predictions
            pre_hc[row, :, 0, 0] = torch.arange(active_seq)
        return SimpleNamespace(
            inputs=SimpleNamespace(ranks=ranks, local_rows=(0,) * len(ranks)),
            logits=logits,
            sampled_ids=sampled_ids,
            pre_hc_hidden=pre_hc,
        )

    monkeypatch.setattr(runner, "prepare_mtp_target_inputs", fake_prepare)
    monkeypatch.setattr(runner, "_execute_main_decode", fake_execute)
    monkeypatch.setattr(
        runner,
        "_copy_main_pre_hc_row",
        lambda source, *, rank, row, hidden_size: source[rank, row].clone(),
    )
    verification = runner._verify_mtp_drafts(
        model,
        DecodeBatch(
            request_ids=["req-a", "req-b"],
            token_ids=torch.tensor([[9], [19]], dtype=torch.long),
            hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
            seq_lens=torch.tensor([100, 100], dtype=torch.int32),
            cache_partitions=[0, 1],
        ),
        torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18],
                [20, 21, 99, 23, 24, 25, 26, 27, 28],
            ],
            dtype=torch.long,
        ),
    )

    assert verification.accepted_token_ids == [
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22],
    ]
    assert verification.tail_token_ids.tolist() == [19, 22]
    assert verification.tail_positions.tolist() == [109, 102]
    assert [chunk.active_width for chunk in prepared_chunks] == [8, 2]
    assert prepared_chunks[1].request_ids == ("req-a",)


def test_deepseek_mtp_partial_target_chunk_waves_requests_on_same_rank(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.layout = deepseek_v4_decode_layout(2)
    runner._compiled.num_speculative_tokens = 2
    prepared_request_ids = []

    def fake_assignment(batch):
        count = len(batch.request_ids)
        return SimpleNamespace(
            ranks=(0,) * count,
            local_rows=tuple(range(count)),
            per_rank_counts=(count,) + (0,) * 7,
            indices_by_rank=(tuple(range(count)),) + ((),) * 7,
        )

    def fake_prepare(model, batch, *, token_rows, positions, active_width):
        prepared_request_ids.append(tuple(batch.request_ids))
        return SimpleNamespace(
            request_ids=tuple(batch.request_ids),
            token_rows=token_rows,
        )

    def fake_execute(model, prepared, *, active_seq):
        predictions = prepared.token_rows + 1
        logits = torch.zeros((8, 4, 128), dtype=torch.float32)
        sampled_ids = torch.zeros((8, 4, 8), dtype=torch.int32)
        logits[0, torch.arange(active_seq), predictions[0, :active_seq]] = 1
        sampled_ids[0, :active_seq, 0] = predictions[0, :active_seq]
        return SimpleNamespace(
            inputs=SimpleNamespace(ranks=(0,), local_rows=(0,)),
            logits=logits,
            sampled_ids=sampled_ids,
            pre_hc_hidden=torch.zeros((8, 4, 4, 1), dtype=torch.float32),
        )

    monkeypatch.setattr(runner, "_decode_assignment", fake_assignment)
    monkeypatch.setattr(runner, "prepare_mtp_target_inputs", fake_prepare)
    monkeypatch.setattr(runner, "_execute_main_decode", fake_execute)
    monkeypatch.setattr(
        runner,
        "_copy_main_pre_hc_row",
        lambda source, *, rank, row, hidden_size: source[rank, row].clone(),
    )

    verification = runner._verify_mtp_drafts(
        model,
        DecodeBatch(
            request_ids=["req-a", "req-b"],
            token_ids=torch.tensor([[9], [19]], dtype=torch.long),
            hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
            seq_lens=torch.tensor([100, 100], dtype=torch.int32),
        ),
        torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
    )

    assert prepared_request_ids == [("req-a",), ("req-b",)]
    assert verification.accepted_token_ids == [[10, 11, 12], [20, 21, 22]]


@pytest.mark.parametrize(
    ("num_speculative_tokens", "decode_seq", "decode_batch", "decode_tokens"),
    [(0, 1, 8, 8), (1, 2, 8, 16), (2, 4, 4, 16), (3, 4, 4, 16), (4, 8, 2, 16), (32, 8, 2, 16)],
)
def test_deepseek_mtp_depth_selects_expanded_decode_layout(
    num_speculative_tokens,
    decode_seq,
    decode_batch,
    decode_tokens,
):
    layout = deepseek_v4_decode_layout(num_speculative_tokens)

    assert layout.decode_seq == decode_seq
    assert layout.decode_batch == decode_batch
    assert layout.decode_tokens == decode_tokens


def test_deepseek_mtp_draft_depth_is_capped_by_remaining_context():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = 9
    model.runtime = replace(model.runtime, max_seq_len=130)
    batch = DecodeBatch(
        request_ids=["req-a", "req-b"],
        token_ids=torch.tensor([[9], [19]], dtype=torch.long),
        hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
        seq_lens=torch.tensor([128, 125], dtype=torch.int32),
    )

    assert runner._mtp_draft_count(model, batch) == 2


def test_deepseek_mtp_corrects_async_lengths_from_committed_tokens():
    runner, _model = _runner_for_prepared_inputs()
    runner._mtp_request_states = {
        "continued": SimpleNamespace(
            proposed_tokens=3,
            prompt_len=100,
            committed_count=5,
        ),
        "first-step": SimpleNamespace(
            proposed_tokens=0,
            prompt_len=40,
            committed_count=0,
        ),
    }
    batch = DecodeBatch(
        request_ids=["continued", "first-step"],
        token_ids=torch.tensor([[9], [19]], dtype=torch.long),
        hidden_states=None,
        seq_lens=torch.tensor([999, 41], dtype=torch.int32),
    )

    assert runner._correct_mtp_seq_lens(batch).tolist() == [106, 41]


def test_deepseek_mtp_exhausted_context_verifies_single_target_column(monkeypatch):
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.layout = deepseek_v4_decode_layout(9)
    runner._compiled.num_speculative_tokens = 9
    model.runtime = replace(model.runtime, max_seq_len=130)
    batch = DecodeBatch(
        request_ids=["req-a"],
        token_ids=torch.tensor([[9]], dtype=torch.long),
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        seq_lens=torch.tensor([130], dtype=torch.int32),
        cache_partitions=[0],
    )
    prepared_chunks = []

    def fake_prepare(model, batch, *, token_rows, positions, active_width):
        chunk = SimpleNamespace(
            request_ids=tuple(batch.request_ids),
            token_rows=token_rows,
            positions=positions,
            active_width=active_width,
        )
        prepared_chunks.append(chunk)
        return chunk

    def fake_execute(model, prepared, *, active_seq):
        logits = torch.zeros((8, active_seq, 128), dtype=torch.float32)
        logits[0, 0, 10] = 1
        sampled_ids = torch.zeros((8, active_seq, 8), dtype=torch.int32)
        sampled_ids[0, 0, 0] = 10
        return SimpleNamespace(
            inputs=SimpleNamespace(ranks=(0,), local_rows=(0,)),
            logits=logits,
            sampled_ids=sampled_ids,
            pre_hc_hidden=torch.zeros((8, active_seq, 4, 1), dtype=torch.float32),
        )

    monkeypatch.setattr(runner, "prepare_mtp_target_inputs", fake_prepare)
    monkeypatch.setattr(runner, "_execute_main_decode", fake_execute)
    monkeypatch.setattr(
        runner,
        "_copy_main_pre_hc_row",
        lambda source, *, rank, row, hidden_size: source[rank, row].clone(),
    )

    num_drafts = runner._mtp_draft_count(model, batch)
    drafts = runner._propose_mtp_tokens(model, batch, num_drafts=num_drafts)
    verification = runner._verify_mtp_drafts(model, batch, drafts)

    assert num_drafts == 0
    assert drafts.shape == (1, 0)
    assert verification.accepted_token_ids == [[10]]
    assert verification.tail_token_ids.tolist() == [10]
    assert verification.tail_positions.tolist() == [130]
    assert len(prepared_chunks) == 1
    assert prepared_chunks[0].active_width == 1
    assert prepared_chunks[0].token_rows.tolist() == [[9] * 8]


def test_cli_selects_deepseek_executor_and_configures_mtp_depth(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2,3,4,5,6,7",
            "--dp",
            "8",
            "--ep",
            "8",
            "--tp",
            "1",
            "--block-size",
            "128",
            "--max-model-len",
            "260",
            "--dtype",
            "int8",
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":4}',
            "--max-num-seqs",
            "16",
            "--use-compile-cache",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.executor_cls == "PyptoDeepSeekV4Executor"
    assert config.device_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert config.parallel_config.replica_device_groups == ((0, 1, 2, 3, 4, 5, 6, 7),)
    assert config.runtime_config.page_size == 128
    assert config.runtime_config.weight_dtype == "int8"
    assert config.enable_prefix_cache is False
    assert config.executor_kwargs["num_speculative_tokens"] == 4
    assert config.runtime_config.num_speculative_tokens == 4
    assert config.runtime_config.max_prefill_tokens_per_request == 8192
    assert config.runtime_config.prefill_chunk_size_choices == PREFILL_CHUNK_SIZE_CHOICES
    assert config.runtime_config.supports_chunked_prefill_with_speculation is True
    assert config.runtime_config.requires_homogeneous_prefill_decode is True
    assert config.max_num_running_reqs == 16
    assert config.long_prefill_token_threshold == 2048
    assert config.executor_kwargs["use_compile_cache"] is True


@pytest.mark.parametrize(
    ("num_speculative_tokens", "expected"),
    [(0, True), (1, True), (3, False)],
)
def test_deepseek_async_decode_prepare_excludes_arbitrary_mtp_depth(
    num_speculative_tokens,
    expected,
):
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._num_speculative_tokens = num_speculative_tokens

    assert executor.supports_async_decode_prepare is expected


def test_cli_keeps_deepseek_autoregressive_decode_when_mtp_is_disabled(tmp_path):
    model_dir = _write_deepseek_model_dir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--model",
            str(model_dir),
            "--devices",
            "0,1,2,3,4,5,6,7",
            "--dp",
            "8",
            "--ep",
            "8",
        ]
    )

    config = cli.build_serving_engine_config(args)

    assert config.executor_kwargs["num_speculative_tokens"] == 0
    contract = npu_executor.load_deepseek_v4_serving_contract()
    assert (
        config.runtime_config.max_prefill_tokens_per_request
        == contract.max_prefill_tokens_per_request
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"method": "draft_model", "num_speculative_tokens": 3}, "method='mtp'"),
        ({"method": "mtp"}, "requires num_speculative_tokens"),
        ({"method": "mtp", "num_speculative_tokens": 0}, "must be positive"),
    ],
)
def test_cli_rejects_invalid_deepseek_speculative_config(config, message):
    args = SimpleNamespace(
        speculative_config=config,
        num_speculative_tokens=None,
    )

    with pytest.raises(ValueError, match=message):
        cli._resolve_num_speculative_tokens(args)


def test_cli_rejects_speculative_config_with_num_speculative_tokens_alias():
    args = SimpleNamespace(
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        num_speculative_tokens=2,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        cli._resolve_num_speculative_tokens(args)


def test_tokenizer_falls_back_when_deepseek_config_fails_strict_validation(tmp_path, monkeypatch):
    class StrictDataclassFieldValidationError(Exception):
        pass

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise StrictDataclassFieldValidationError("attention_dropout expected float")

    sentinel = object()
    fake_transformers = type(
        "FakeTransformers",
        (),
        {"AutoTokenizer": AutoTokenizer, "PreTrainedTokenizerFast": object},
    )
    fake_hub_errors = ModuleType("huggingface_hub.errors")
    fake_hub_errors.StrictDataclassFieldValidationError = StrictDataclassFieldValidationError
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_hub_errors)
    monkeypatch.setattr(tokenizer_module, "_load_fast_tokenizer_from_file", lambda *args: sentinel)
    (tmp_path / "tokenizer.json").touch()

    adapter = tokenizer_module.TransformersTokenizerAdapter.from_pretrained(str(tmp_path))

    assert adapter.tokenizer is sentinel


@pytest.mark.parametrize("extra_draft_ratios", [(), (0, 0)])
def test_deepseek_compile_attaches_lazy_weight_store_without_opening_shards(
    tmp_path,
    monkeypatch,
    extra_draft_ratios,
):
    model_dir = _write_deepseek_model_dir(tmp_path)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["compress_ratios"].extend(extra_draft_ratios)
    config_path.write_text(json.dumps(config))
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    monkeypatch.setattr(
        model_loader,
        "load_tokenizer",
        lambda *args, **kwargs: _Tokenizer(),
    )
    opened: list[Path] = []

    def _fail_open(path: Path, device: str):
        opened.append(path)
        raise AssertionError(f"unexpected safetensors open on {device}: {path}")

    monkeypatch.setattr(weight_loader, "_default_safe_open", _fail_open)
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(
            page_size=128,
            max_batch_size=4,
            max_seq_len=256,
            weight_dtype="int8",
        ),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(platform="a2a3sim", device_ids=tuple(range(8)))

    compiled = executor._compile_model(loaded.runtime_model)

    assert opened == []
    assert isinstance(compiled.weight_store, DeepSeekV4WeightStore)
    assert compiled.weight_store.filename_for("head.weight") == "model-00001-of-00001.safetensors"
    assert compiled.weight_store.device == "cpu"
    assert compiled.layer_plan[0].attention_kind == "swa"
    assert compiled.layer_plan[2].attention_kind == "csa"
    assert compiled.layer_plan[2].include_tid2eid is True
    assert compiled.layer_plan[3].attention_kind == "hca"
    assert compiled.layer_plan[3].include_gate_bias is True


@pytest.mark.parametrize("use_compile_cache", [False, True])
def test_deepseek_compiler_only_sets_cache_dir_when_enabled(tmp_path, monkeypatch, use_compile_cache):
    """Disabled caching keeps PyPTO's fresh per-kernel build directories."""
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    captured: dict[str, object] = {}

    class _Compiler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    monkeypatch.setattr(npu_executor, "KernelCompiler", _Compiler)

    executor = npu_executor.DeepSeekV4PyptoExecutor(
        platform="a2a3sim",
        device_ids=tuple(range(8)),
        pypto_build_dir=str(tmp_path / "build"),
        use_compile_cache=use_compile_cache,
    )

    expected = executor._pypto_build_dir if use_compile_cache else None
    assert captured["cache_dir"] == expected
    assert getattr(captured["run_config"], "save_kernels_dir") == expected


@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_deepseek_compile_selects_mtp_programs(
    tmp_path,
    monkeypatch,
    num_speculative_tokens,
):
    model_dir = _write_deepseek_model_dir(tmp_path)
    kernel_dir = _write_deepseek_kernel_dir(tmp_path, lm_head_tp_size=8)
    monkeypatch.setattr(model_loader, "load_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(npu_executor, "_find_pypto_lib_deepseek_v4_dir", lambda *args, **kwargs: kernel_dir)
    monkeypatch.setattr(DeepSeekV4WeightStore, "validate_mtp_startup_contract", lambda *args, **kwargs: None)
    loaded = ModelLoader().load(
        model_id="dsv4",
        model_dir=str(model_dir),
        runtime_config=RuntimeConfig(
            page_size=128,
            max_batch_size=4,
            max_seq_len=256,
            weight_dtype="int8",
            num_speculative_tokens=num_speculative_tokens,
        ),
    )

    prefill_fwd = SimpleNamespace(l3_prefill_fwd=object())
    decode_fwd = SimpleNamespace(l3_decode_fwd=object())
    decode_fwd_mtp = SimpleNamespace(l3_decode_fwd_mtp=object())
    decode_mtp = SimpleNamespace(l3_decode_mtp=object())
    prefill_mtp = SimpleNamespace(l3_mtp_prefill_fwd=object())
    compile_calls: list[tuple[str, object, frozenset[str] | None]] = []

    def _fake_compile(self, name, jit_fn, *, layout, runtime_scalar_names=None):
        assert layout == deepseek_v4_decode_layout(num_speculative_tokens)
        compile_calls.append((name, jit_fn, runtime_scalar_names))
        return DeepSeekV4L3Callable(compiled=object(), name=name)

    monkeypatch.setattr(
        npu_executor.DeepSeekV4PyptoExecutor,
        "_load_kernel_modules",
        lambda self, layout: {
            "config": object(),
            "prefill_fwd": prefill_fwd,
            "decode_fwd": decode_fwd,
            "decode_fwd_mtp": decode_fwd_mtp,
            "decode_mtp": decode_mtp,
            "prefill_mtp": prefill_mtp,
            "utils": object(),
        },
    )
    monkeypatch.setattr(npu_executor.DeepSeekV4PyptoExecutor, "_compile_l3_callable", _fake_compile)
    monkeypatch.setattr(
        npu_executor.DeepSeekV4PyptoExecutor,
        "_build_rope_tables",
        lambda self, utils_module, config_module: (
            torch.empty((2, 1, 1)),
            torch.empty((2, 1, 1)),
        ),
    )
    executor = npu_executor.DeepSeekV4PyptoExecutor(
        platform="a2a3sim",
        device_ids=tuple(range(8)),
        compile_kernels=True,
        num_speculative_tokens=num_speculative_tokens,
    )

    compiled = executor._compile_model(loaded.runtime_model)

    expected_calls = [
        ("deepseek_v4_prefill", prefill_fwd.l3_prefill_fwd, None),
    ]
    if num_speculative_tokens == 1:
        expected_calls.append(
            (
                "deepseek_v4_decode_mtp_fused",
                decode_fwd_mtp.l3_decode_fwd_mtp,
                frozenset({"mtp_num_tokens"}),
            )
        )
    else:
        expected_calls.append(("deepseek_v4_decode", decode_fwd.l3_decode_fwd, None))
    expected_calls.extend(
        [
            ("deepseek_v4_mtp_prefill", prefill_mtp.l3_mtp_prefill_fwd, frozenset({"num_tokens"})),
        ]
    )
    if num_speculative_tokens > 1:
        expected_calls.append(
            ("deepseek_v4_mtp_decode", decode_mtp.l3_decode_mtp, frozenset({"num_tokens"}))
        )
    assert compile_calls == expected_calls
    assert compiled.prefill is not None
    assert compiled.decode is not None
    assert compiled.mtp_prefill is not None
    assert (compiled.mtp_decode is None) == (num_speculative_tokens == 1)


def test_deepseek_l3_compile_passes_runtime_scalars_unspecialized():
    """Runtime scalars are passed as pl.RUNTIME (unspecialized) to the compiler."""
    from pypto.language import RUNTIME

    captured: dict[str, object] = {}

    class _FakeCompiler:
        def compile(self, name, jit_fn, *, use_cache=False, **compile_kwargs):
            captured["name"] = name
            captured["compile_kwargs"] = compile_kwargs
            return "compiled"

    def _kernel(x, num_tokens):
        pass

    class _JitFunction:
        def __init__(self) -> None:
            self._func = _kernel

    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._compiler = _FakeCompiler()
    executor._use_compile_cache = False

    compiled = executor._compile_l3_callable(
        "deepseek_v4_mtp_prefill",
        _JitFunction(),
        layout=DeepSeekV4CacheLayout(),
        runtime_scalar_names=frozenset({"num_tokens"}),
    )

    assert compiled == "compiled"
    assert captured["name"] == "deepseek_v4_mtp_prefill"
    assert captured["compile_kwargs"] == {"num_tokens": RUNTIME}


def test_deepseek_compile_l3_callable_threads_use_cache_to_compiler():
    """_compile_l3_callable forwards use_compile_cache to the shared compiler."""
    captured: dict[str, object] = {}

    class _FakeCompiler:
        def compile(self, name, jit_fn, *, use_cache=False, **compile_kwargs):
            captured["name"] = name
            captured["use_cache"] = use_cache
            return "compiled"

    def _kernel(x: tuple[int, int]):
        pass

    class _JitFunction:
        _func = _kernel

    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._compiler = _FakeCompiler()
    executor._use_compile_cache = True

    compiled = executor._compile_l3_callable(
        "deepseek_v4_decode", _JitFunction(), layout=DeepSeekV4CacheLayout()
    )

    assert compiled == "compiled"
    assert captured["name"] == "deepseek_v4_decode"
    assert captured["use_cache"] is True


def test_deepseek_weight_store_reads_real_safetensors_by_name(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "embed.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "head.weight": torch.ones(2, 2),
        },
        str(tmp_path / "global.safetensors"),
    )
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={
            "embed.weight": "global.safetensors",
            "head.weight": "global.safetensors",
        },
    )

    loaded = store.load_tensor("embed.weight")

    assert loaded.tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_deepseek_weight_store_maps_valid_prepacked_sidecar(tmp_path, monkeypatch, caplog):
    from safetensors.torch import save_file

    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"source-checkpoint")
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"source.weight": shard_path.name},
    )
    params = {
        "ranks": 2,
        "n_routed_experts": 4,
        "compress_ratios": (4,),
        "num_hash_layers": 1,
    }
    fingerprint = store.packed_stacked_layer_weights_fingerprint(**params)
    expected = {
        name: torch.arange(2, dtype=torch.float32).reshape(2, 1)
        for name in weight_loader._DEEPSEEK_V4_PACKED_WEIGHT_NAMES
    }
    save_file(
        expected,
        str(deepseek_v4_packed_weights_path(tmp_path, ranks=2)),
        metadata={
            "format": DEEPSEEK_V4_PACKED_FORMAT,
            "source_fingerprint": fingerprint,
        },
    )
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda fd, path: 1.0)
    monkeypatch.setattr(
        store,
        "load_packed_layer_weights",
        lambda *args, **kwargs: pytest.fail("valid sidecar must skip checkpoint packing"),
    )

    packed = store.load_stacked_layer_weights(**params)

    assert packed.tensors.keys() == expected.keys()
    assert all(torch.equal(packed.tensors[name], tensor) for name, tensor in expected.items())

    shard_path.write_bytes(b"changed-source-checkpoint")
    caplog.set_level("WARNING")
    assert store.load_prepacked_stacked_layer_weights(**params) is None
    assert "Ignoring stale DeepSeekV4 packed weights sidecar" in caplog.text


def test_deepseek_weight_store_ignores_prepacked_sidecar_with_wrong_tensor_names(
    tmp_path,
    monkeypatch,
    caplog,
):
    from safetensors.torch import save_file

    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"source-checkpoint")
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"source.weight": shard_path.name},
    )
    params = {
        "ranks": 2,
        "n_routed_experts": 4,
        "compress_ratios": (4,),
        "num_hash_layers": 1,
    }
    save_file(
        {"unexpected": torch.zeros((2, 1))},
        str(deepseek_v4_packed_weights_path(tmp_path, ranks=2)),
        metadata={
            "format": DEEPSEEK_V4_PACKED_FORMAT,
            "source_fingerprint": store.packed_stacked_layer_weights_fingerprint(**params),
        },
    )
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda fd, path: 1.0)
    caplog.set_level("WARNING")

    assert store.load_prepacked_stacked_layer_weights(**params) is None
    assert "invalid tensor names" in caplog.text
    assert "unexpected" in caplog.text


def test_deepseek_weight_store_skips_cold_prepacked_sidecar(tmp_path, monkeypatch, caplog):
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"source-checkpoint")
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"source.weight": shard_path.name},
    )
    deepseek_v4_packed_weights_path(tmp_path, ranks=2).write_bytes(b"not-opened")
    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda fd, path: 0.5)
    caplog.set_level("INFO")

    packed = store.load_prepacked_stacked_layer_weights(
        ranks=2,
        n_routed_experts=4,
        compress_ratios=(4,),
        num_hash_layers=1,
    )

    assert packed is None
    assert "Skipping cold DeepSeekV4 packed weights sidecar" in caplog.text


def test_deepseek_page_cache_probe_unusable_descriptor_falls_back(tmp_path, caplog):
    # The loader owns the descriptor now, so the probe's failure mode is a
    # descriptor it cannot map rather than a path it cannot open.
    packed_path = tmp_path / "packed.safetensors"
    packed_path.write_bytes(b"")
    fd = os.open(packed_path, os.O_RDONLY)
    try:
        caplog.set_level("WARNING")
        os.close(fd)
        assert weight_loader._sample_file_page_cache_residency(fd, packed_path) is None
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    assert "Could not inspect page-cache residency" in caplog.text


def test_deepseek_prepack_fingerprints_before_packing_and_preserves_shard_mode(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "compress_ratios": [4],
                "n_routed_experts": 4,
                "num_hash_layers": 1,
            }
        )
    )
    shard_path = model_dir / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"checkpoint")
    shard_path.chmod(0o640)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"source.weight": shard_path.name}})
    )
    events: list[str] = []

    class FakeStore:
        def __init__(self, *, model_dir, weight_map):
            pass

        def packed_stacked_layer_weights_fingerprint(self, **kwargs):
            events.append("fingerprint")
            return "source-fingerprint"

        def load_stacked_layer_weights(self, **kwargs):
            assert kwargs["use_prepacked"] is False
            events.append("pack")
            return DeepSeekV4StackedLayerWeights(tensors={"weight": torch.zeros((1, 1))})

    monkeypatch.setattr(prepack_deepseek_v4, "DeepSeekV4WeightStore", FakeStore)
    output = model_dir / "packed.safetensors"

    prepack_deepseek_v4.build_sidecar(
        model_dir,
        ranks=1,
        output=output,
        force=False,
    )

    assert events == ["fingerprint", "pack"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o640


def test_deepseek_executor_lazily_loads_and_caches_embeddings(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"embed.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4)},
        str(tmp_path / "embed.safetensors"),
    )
    open_count = 0
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"embed.weight": "embed.safetensors"},
    )
    original_open = store._safe_open_fn

    def _counting_open(path: Path, device: str):
        nonlocal open_count
        open_count += 1
        return original_open(path, device)

    store._safe_open_fn = _counting_open
    executor = npu_executor.DeepSeekV4PyptoExecutor.__new__(npu_executor.DeepSeekV4PyptoExecutor)
    executor._compiled = {
        "dsv4": DeepSeekV4CompiledKernels(
            layout=DeepSeekV4CacheLayout(),
            model_dir=str(tmp_path),
            weight_map=store.weight_map,
            weight_store=store,
            compress_ratios=tuple([0] * 44),
            layer_plan=build_deepseek_v4_layer_plan(
                compress_ratios=tuple([0] * 44),
                num_hidden_layers=43,
                num_hash_layers=3,
            ),
            kernel_dir=str(tmp_path),
            kernel_contract=_deepseek_serving_contract(),
        )
    }
    executor._embedding_cache = {}
    model = _runtime_model_for_embeddings()

    first = executor.lookup_embeddings(model, torch.tensor([1, 3], dtype=torch.long))
    second = executor.lookup_embeddings(model, torch.tensor([[2, 4]], dtype=torch.long))
    runner = DeepSeekV4ModelRunner(compiled=executor._compiled["dsv4"])
    runner_rows = runner._embedding_rows(torch.tensor([0, 5]), torch.float32)

    assert first.tolist() == [[4.0, 5.0, 6.0, 7.0], [12.0, 13.0, 14.0, 15.0]]
    assert second.shape == (1, 2, 4)
    assert second[0, 1].tolist() == [16.0, 17.0, 18.0, 19.0]
    assert runner_rows.tolist() == [[0.0, 1.0, 2.0, 3.0], [20.0, 21.0, 22.0, 23.0]]
    assert open_count == 1


def test_deepseek_weight_store_loads_rank_local_experts(tmp_path):
    core_names = deepseek_v4_layer_core_weight_names(0, include_tid2eid=True)
    local_experts = deepseek_v4_local_expert_ids(rank=1, ranks=4, n_routed_experts=8)
    expert_names = deepseek_v4_routed_expert_weight_names(0, local_experts)
    weight_map = {name: "layer.safetensors" for name in (*core_names, *expert_names)}
    (tmp_path / "layer.safetensors").touch()
    reads: list[str] = []

    class _Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name: str) -> torch.Tensor:
            reads.append(name)
            return torch.tensor([len(reads)])

    store = DeepSeekV4WeightStore(
        model_dir=tmp_path, weight_map=weight_map, safe_open_fn=lambda path, device: _Reader()
    )

    loaded = store.load_rank_layer_weights(
        0,
        rank=1,
        ranks=4,
        n_routed_experts=8,
        include_tid2eid=True,
    )

    assert local_experts == (2, 3)
    assert set(loaded) == set(weight_map)
    assert all(".experts.2." in name or ".experts.3." in name for name in expert_names)
    assert not any(".experts.0." in name or ".experts.1." in name for name in loaded)


def test_deepseek_weight_store_packs_lm_head_into_8_tp_shards(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "embed.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4),
            "norm.weight": torch.arange(4, dtype=torch.float32),
            "head.weight": torch.arange(64, dtype=torch.float32).reshape(16, 4) + 100,
            "hc_head_fn": torch.zeros((4, 16), dtype=torch.float32),
            "hc_head_scale": torch.ones((1,), dtype=torch.float32),
            "hc_head_base": torch.zeros((4,), dtype=torch.float32),
        },
        str(tmp_path / "global.safetensors"),
    )
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={
            "embed.weight": "global.safetensors",
            "norm.weight": "global.safetensors",
            "head.weight": "global.safetensors",
            "hc_head_fn": "global.safetensors",
            "hc_head_scale": "global.safetensors",
            "hc_head_base": "global.safetensors",
        },
    )

    global_weights = store.load_packed_global_weights(ranks=8)

    assert global_weights.lm_head_layout.vocab_per_rank == 2
    assert global_weights.lm_head_layout.padded_vocab_per_rank == 512
    assert global_weights.lm_head_weight.shape == (8, 512, 4)
    assert global_weights.lm_head_weight[0, :2].tolist() == [
        [100.0, 101.0, 102.0, 103.0],
        [104.0, 105.0, 106.0, 107.0],
    ]
    assert global_weights.lm_head_weight[1, :2].tolist() == [
        [108.0, 109.0, 110.0, 111.0],
        [112.0, 113.0, 114.0, 115.0],
    ]
    assert torch.count_nonzero(global_weights.lm_head_weight[:, 2:]) == 0


def test_deepseek_layer_packer_transposes_and_stacks_rank_local_experts():
    raw = _synthetic_layer_raw(layer_id=0, n_experts=4)

    packed = pack_deepseek_v4_layer_weights(
        0,
        raw,
        ranks=2,
        n_routed_experts=4,
        compress_ratio=4,
        include_tid2eid=False,
        include_gate_bias=True,
    )

    assert packed.tensors["wq_a"].shape == (2, 4, 2)
    assert packed.tensors["wq_a"][0].tolist() == raw["layers.0.attn.wq_a.weight"].t().tolist()
    assert packed.tensors["wo_a"].shape == (2, 8, 2, 4)
    assert packed.tensors["csa_cmp_wkv"].shape == (2, 2, 4)
    assert packed.tensors["csa_cmp_wkv"][0].tolist() == raw["layers.0.attn.compressor.wkv.weight"].tolist()
    assert packed.tensors["csa_inner_wkv"].shape == (2, 2, 4)
    assert (
        packed.tensors["csa_inner_wkv"][0].tolist()
        == raw["layers.0.attn.indexer.compressor.wkv.weight"].tolist()
    )
    assert packed.tensors["hca_cmp_wkv"].shape == (2, 512, 4096)
    assert torch.count_nonzero(packed.tensors["hca_cmp_wkv"]) == 0
    assert packed.tensors["gate_bias"].shape == (2, 4)
    assert packed.tensors["tid2eid"].shape == (2, 129280, 6)
    assert packed.tensors["routed_w1"].shape == (2, 2, 2, 4)
    assert packed.tensors["routed_w1"][0, 0].tolist() == raw["layers.0.ffn.experts.0.w1.weight"].tolist()
    assert packed.tensors["routed_w1"][1, 0].tolist() == raw["layers.0.ffn.experts.2.w1.weight"].tolist()
    assert torch.equal(packed.tensors["csa_hadamard_idx"][0], deepseek_v4_hadamard_idx())

    destination_storage = {
        name: torch.empty(
            (int(tensor.shape[0]), int(tensor.shape[1]) * 2, *tensor.shape[2:]),
            dtype=tensor.dtype,
        )
        for name, tensor in packed.tensors.items()
    }
    destinations = {
        name: storage[:, int(packed.tensors[name].shape[1]) :]
        for name, storage in destination_storage.items()
    }
    assert not all(destination.is_contiguous() for destination in destinations.values())
    direct = pack_deepseek_v4_layer_weights(
        0,
        raw,
        ranks=2,
        n_routed_experts=4,
        compress_ratio=4,
        include_tid2eid=False,
        include_gate_bias=True,
        destinations=destinations,
    )

    assert direct.tensors.keys() == packed.tensors.keys()
    for name, expected in packed.tensors.items():
        assert direct.tensors[name] is destinations[name]
        assert torch.equal(direct.tensors[name], expected), name


def test_deepseek_stacked_weight_loader_packs_subsequent_layers_into_final_slices(monkeypatch):
    def packed_layer(layer_id: int) -> weight_loader.DeepSeekV4PackedLayerWeights:
        tensors = {"fwd": torch.full((2, 2), layer_id, dtype=torch.int8)}
        tensors.update(
            {
                name: torch.full((2, 1), layer_id * 20 + index, dtype=torch.float32)
                for index, name in enumerate(weight_loader.DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES)
            }
        )
        tensors.update(
            {
                name: torch.full((2, 1), layer_id * 20 + 12 + index, dtype=torch.float32)
                for index, name in enumerate(weight_loader.DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES)
            }
        )
        return weight_loader.DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=tensors)

    layers = [packed_layer(layer_id) for layer_id in range(3)]
    direct_flags: list[bool] = []
    store = DeepSeekV4WeightStore(model_dir=".", weight_map={})

    def fake_load(layer_id: int, **kwargs):
        destinations = kwargs.get("destinations")
        direct_flags.append(destinations is not None)
        packed = layers[layer_id]
        if destinations is None:
            return packed
        for name, destination in destinations.items():
            destination.copy_(packed.tensors[name])
        return weight_loader.DeepSeekV4PackedLayerWeights(layer_id=layer_id, tensors=destinations)

    monkeypatch.setattr(store, "load_packed_layer_weights", fake_load)
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("torch.cat must not be used"))

    stacked = store.load_stacked_layer_weights(
        ranks=2,
        n_routed_experts=4,
        compress_ratios=(0, 4, 128),
        num_hash_layers=1,
    )

    assert direct_flags == [False, True, True]
    assert stacked.tensors["fwd"].tolist() == [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]]
    for name in weight_loader.DEEPSEEK_V4_CSA_STACKED_WEIGHT_NAMES:
        assert torch.equal(stacked.tensors[name], layers[1].tensors[name])
    for name in weight_loader.DEEPSEEK_V4_HCA_STACKED_WEIGHT_NAMES:
        assert torch.equal(stacked.tensors[name], layers[2].tensors[name])
    assert all(tensor.is_contiguous() for tensor in stacked.tensors.values())


def test_deepseek_worker_registers_main_and_mtp_weights_for_inheritance(monkeypatch):
    main_weight = torch.zeros((1, 2), dtype=torch.float32)
    mtp_weight = torch.ones((1, 2), dtype=torch.float32)
    compiled_program = object()
    captured = {}

    class FakeDistributedWorker:
        def __init__(
            self,
            compiled,
            *,
            config,
            persistent,
            reset_persistent_windows,
            inherited_host_tensors,
        ):
            captured["compiled"] = compiled
            captured["config"] = config
            captured["persistent"] = persistent
            captured["reset_persistent_windows"] = reset_persistent_windows
            captured["inherited"] = inherited_host_tensors

    monkeypatch.setattr("pypto.runtime.DistributedWorker", FakeDistributedWorker)
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._l3_worker = None
    runner._l3_run_config = object()
    runner._stacked_host_weights = {"main": main_weight}
    runner._mtp_buffers = type("MtpBuffers", (), {"weights": {"mtp": mtp_weight}})()
    runner._compiled = type(
        "Compiled",
        (),
        {
            "l3_callables": lambda _self: (
                DeepSeekV4L3Callable(
                    compiled_program,
                    "decode",
                ),
            )
        },
    )()

    worker = runner._shared_l3_worker()

    assert isinstance(worker, FakeDistributedWorker)
    assert captured["compiled"] == [compiled_program]
    assert captured["config"] is runner._l3_run_config
    assert captured["persistent"] is True
    assert captured["reset_persistent_windows"] is False
    assert captured["inherited"] == [main_weight, mtp_weight]


def test_deepseek_resident_upload_releases_inherited_host_references():
    main_weight = torch.zeros((1, 2), dtype=torch.float32)
    mtp_weight = torch.ones((1, 2), dtype=torch.float32)

    class FakeWorker:
        def __init__(self):
            self.released = False

        def alloc_stacked_tensor(self, tensor):
            return tensor

        def free_stacked_tensor(self, _tensor):
            pass

        def release_inherited_host_tensor_refs(self):
            self.released = True

    worker = FakeWorker()
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._stacked_host_weights = {"main": main_weight}
    runner._stacked_device_weights = None
    runner._global_weights = None
    runner._mtp_buffers = type("MtpBuffers", (), {"weights": {"mtp": mtp_weight}})()
    runner._mtp_device_weights = None
    runner._compiled = SimpleNamespace(
        prepacked_layer_weights=DeepSeekV4StackedLayerWeights(tensors={"main": main_weight})
    )
    runner._shared_l3_worker = lambda: worker

    runner._materialize_resident_weights()

    assert worker.released
    assert runner._compiled.prepacked_layer_weights is None
    assert runner._stacked_host_weights is None
    assert not runner._mtp_buffers.weights


def test_deepseek_cache_metadata_maps_scheduler_block_ids():
    metadata = DeepSeekV4CacheMetadataBuilder(layout=DeepSeekV4CacheLayout())

    table = metadata.block_table_from_ids([[64, 65]], max_blocks=4)
    assert table.tolist() == [[64, 65, 0, 0]]

    hca_state_mapping = metadata.state_slot_mapping_from_ids(
        [[64, 65]],
        [[0, 7, 8]],
        state_block_size=8,
    )
    state_base = 64 * 8
    assert hca_state_mapping.tolist() == [[state_base, state_base + 7, state_base + 8]]

    cmp_blocks = list(range(64, 128))
    long_cmp_mapping = metadata.compressed_slot_mapping_from_ids(
        [cmp_blocks],
        [[3, 7, 4095, 4099, 8191]],
        block_size=128,
        compress_ratio=4,
    )
    assert long_cmp_mapping.tolist() == [
        [64 * 32, 64 * 32 + 1, 95 * 32 + 31, 96 * 32, 127 * 32 + 31]
    ]

    with pytest.raises(ValueError, match="position 4099 requires source block 32"):
        metadata.compressed_slot_mapping_from_ids(
            [cmp_blocks[:32]],
            [[4099]],
            block_size=128,
            compress_ratio=4,
        )

    swa_indices, swa_lens = metadata.swa_window_indices_and_lens_from_ids(
        [[64, 65]],
        [[127, 128]],
    )
    assert swa_lens.tolist() == [128, 128]
    assert swa_indices[0, -1].item() == 64 * 128 + 127
    assert swa_indices[1, -1].item() == 65 * 128


_DEEPSEEK_TEST_COMPRESS_RATIOS = (0, 0, *([4] * 21), *([128] * 20))


class _MemoryInfoFakeL3Worker:
    """L3 worker stand-in exposing device_memory_info keyed by logical worker id."""

    def __init__(self, memory_by_worker):
        self._memory = memory_by_worker
        self.queried_worker_ids = []

    def device_memory_info(self, worker_id=0):
        self.queried_worker_ids.append(worker_id)
        return self._memory[worker_id]


def _deepseek_cache_group_specs(max_seq_len, compress_ratios=_DEEPSEEK_TEST_COMPRESS_RATIOS):
    return build_deepseek_v4_cache_group_specs(43, compress_ratios, decode_batch=8, max_seq_len=max_seq_len)


def test_deepseek_cache_group_specs_leave_physical_capacity_for_runtime_sizing():
    specs = _deepseek_cache_group_specs(8192)
    by_name = {spec.name: spec for spec in specs}

    assert all(spec.num_blocks is None for spec in specs)
    assert by_name["ori"].spec.page_size_bytes == 43 * 128 * 512 * 2
    assert by_name["cmp_c128"].max_blocks_per_seq == 64
    assert by_name["cmp_c4"].max_blocks_per_seq == 64
    assert by_name["idx"].max_blocks_per_seq == 64
    assert by_name["cmp_c128"].spec.page_size_bytes == 20 * (128 // 128) * 512 * 2
    assert by_name["cmp_c4"].spec.page_size_bytes == 21 * (128 // 4) * 512 * 2
    assert by_name["idx"].spec.page_size_bytes == 21 * (128 // 4) * (128 + 4)
    assert by_name["hca_state"].spec.page_size_bytes == 20 * 8 * 1024 * 4
    assert deepseek_v4_cache_blocks_for_slots(specs, 3) == {
        "ori": 12,
        "cmp_c128": 192,
        "cmp_c4": 192,
        "idx": 192,
        "hca_state": 144,
        "csa_state": 195,
        "csa_inner_state": 195,
    }

    tail_by_name = {spec.name: spec for spec in _deepseek_cache_group_specs(8320)}
    assert {
        tail_by_name[name].max_blocks_per_seq
        for name in ("cmp_c128", "cmp_c4", "idx")
    } == {65}

    with pytest.raises(ValueError, match="needs 129 source-token blocks"):
        _deepseek_cache_group_specs(16385)


def test_deepseek_cache_sizing_uses_limiting_rank_post_weight_budget():
    layout = DeepSeekV4CacheLayout(decode_batch=8, decode_seq=1, decode_tokens=8, ranks=2)
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
            device_id=2,
            device_ids=(2, 5),
        )
    )
    runner._cache_group_specs = _deepseek_cache_group_specs(4096, runner._compiled.compress_ratios)

    # Logical worker 0 -> physical npu:2, logical worker 1 -> physical npu:5;
    # rank 1 is the limiting budget either way.
    worker = _MemoryInfoFakeL3Worker({
        0: (5_000_000_000, 10_000_000_000),
        1: (4_000_000_000, 10_000_000_000),
    })
    runner._l3_worker = worker
    runtime = RuntimeConfig(npu_memory_utilization=0.8)

    bytes_per_slot = sum(
        spec.max_blocks_per_seq * spec.spec.page_size_bytes for spec in runner._cache_group_specs
    )
    scratch_bytes = sum(layout.decode_batch * spec.spec.page_size_bytes for spec in runner._cache_group_specs)
    expected = max((2_000_000_000 - scratch_bytes) // bytes_per_slot, 1)

    assert runner._compute_kv_cache_capacity_slots(runtime) == expected
    # Ranks map to logical worker IDs, not the physical device ids (2, 5).
    assert worker.queried_worker_ids == [0, 1]


def test_deepseek_cache_sizing_uses_prewarmed_worker_memory():
    layout = DeepSeekV4CacheLayout(decode_batch=8, decode_seq=1, decode_tokens=8, ranks=2)
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(
                prefill_tile_tokens=layout.prefill_seq
            ),
            device_id=0,
            device_ids=(0, 1),
        )
    )
    runner._cache_group_specs = _deepseek_cache_group_specs(4096, runner._compiled.compress_ratios)

    total = 65_800_000_000
    free = 9_600_000_000  # post-weights with the configured ring arena resident
    worker = _MemoryInfoFakeL3Worker({
        0: (free, total),
        1: (free, total),
    })
    runner._l3_worker = worker
    runtime = RuntimeConfig(
        npu_memory_utilization=0.90,
        ring_dep_pool=[16384, 16384, 16384, 16384],
        ring_task_window=[16384, 16384, 16384, 16384],
        ring_heap=[1073741824, 1073741824, 1073741824, 1073741824],
    )

    bytes_per_slot = sum(
        spec.max_blocks_per_seq * spec.spec.page_size_bytes for spec in runner._cache_group_specs
    )
    scratch_bytes = sum(layout.decode_batch * spec.spec.page_size_bytes for spec in runner._cache_group_specs)
    post_resident_budget = int(total * 0.90) - (total - free)
    expected = (post_resident_budget - scratch_bytes) // bytes_per_slot
    assert expected >= 1

    assert runner._compute_kv_cache_capacity_slots(runtime) == expected


def test_deepseek_cache_allocation_halves_all_groups_together_on_oom(monkeypatch):
    layout = DeepSeekV4CacheLayout(decode_batch=8, decode_seq=1, decode_tokens=8)
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
        )
    )
    runner._cache_group_specs = _deepseek_cache_group_specs(4096, runner._compiled.compress_ratios)
    attempts = []
    ori_blocks_per_slot = runner._cache_group_specs[0].max_blocks_per_seq

    def allocate_main_cache():
        slots = runner._cache_group_num_blocks["ori"] // ori_blocks_per_slot
        attempts.append(slots)
        if slots > 2:
            raise MemoryError("synthetic OOM")
        return object()

    monkeypatch.setattr(runner, "_materialize_decode_device_cache", allocate_main_cache)
    monkeypatch.setattr(runner, "_materialize_mtp_device_kv_cache", lambda: None)

    assert runner._alloc_kv_cache_with_retry(8) == 2
    assert attempts == [8, 4, 2]
    assert runner._cache_group_num_blocks == deepseek_v4_cache_blocks_for_slots(
        runner._cache_group_specs,
        2,
    )


def test_deepseek_device_cache_allocates_runtime_sized_rank_shards():
    layout = DeepSeekV4CacheLayout(
        ranks=2,
        block_size=128,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=tuple([0] * 43),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
        )
    )
    runner._cache_group_num_blocks = {
        name: 2
        for name in (
            "ori",
            "cmp_c128",
            "cmp_c4",
            "idx",
            "hca_state",
            "csa_state",
            "csa_inner_state",
        )
    }

    class FakeWorker:
        def __init__(self):
            self.allocations = []
            self.frees = []

        def alloc_tensor(self, shape, dtype, *, worker_id=0):
            tensor = DeviceTensor(
                0x1000 + len(self.allocations) * 0x100,
                tuple(shape),
                dtype,
            )
            self.allocations.append((worker_id, tensor))
            return tensor

        def free_tensor(self, tensor, *, worker_id=0):
            self.frees.append((worker_id, tensor))

        def free_stacked_tensor(self, stacked):
            for tensor, worker_id in zip(
                stacked.shards,
                stacked.worker_ids,
                strict=True,
            ):
                self.free_tensor(tensor, worker_id=worker_id)

    worker = FakeWorker()
    runner._l3_worker = worker

    cache = runner._materialize_decode_device_cache()

    assert cache.kv_cache.full_shape == (2, 43 * 3, 128, 1, 512)
    assert cache.hca_cmp_kv.full_shape == (2, 20 * 3, 1, 1, 512)
    assert cache.csa_cmp_kv.full_shape == (2, 21 * 3, 32, 1, 512)
    assert cache.idx_kv_cache.full_shape == (2, 21 * 3, 32, 1, 128)
    assert cache.hca_compress_state.full_shape == (2, 20 * 3, 8, 1024)
    assert len(worker.allocations) == 16
    assert {worker_id for worker_id, _tensor in worker.allocations} == {0, 1}

    runner._free_device_caches()
    assert len(worker.frees) == 16


def _grouped_cache_rows(count: int) -> list[dict[str, list[int]]]:
    base_ids = {
        "ori": 0,
        "cmp_c128": 1,
        "cmp_c4": 1,
        "idx": 2,
        "hca_state": 3,
        "csa_state": 4,
        "csa_inner_state": 5,
    }
    return [
        {name: [base_id + request_index] for name, base_id in base_ids.items()}
        for request_index in range(count)
    ]


def _prefill_batch(
    chunk_lens,
    *,
    chunk_starts=None,
    cache_partitions=None,
    block_ids_by_group=None,
    token_base=0,
):
    count = len(chunk_lens)
    chunk_starts = chunk_starts or [0] * count
    total_tokens = sum(chunk_lens)
    return PrefillBatch(
        request_ids=["req-a"] if count == 1 else [f"req-{index}" for index in range(count)],
        token_ids=token_base + torch.arange(total_tokens, dtype=torch.long),
        input_embeddings=torch.arange(total_tokens * 4, dtype=torch.float32)
        .reshape(total_tokens, 4)
        .to(torch.bfloat16),
        seq_lens=[start + length for start, length in zip(chunk_starts, chunk_lens, strict=True)],
        chunk_lens=chunk_lens,
        chunk_offsets=[sum(chunk_lens[:index]) for index in range(count)],
        chunk_starts=chunk_starts,
        block_ids_by_group=block_ids_by_group or _grouped_cache_rows(count),
        cache_partitions=cache_partitions or list(range(count)),
    )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"chunk_lens": []}, r"prefill chunk_lens has 0 entries for 2 requests"),
        ({"chunk_lens": [0, 2]}, r"chunk_lens\[0\] must be positive"),
        ({"seq_lens": [-1, 2]}, r"seq_lens\[0\] must be non-negative"),
        ({"seq_lens": [1, 2], "chunk_starts": [-1, 0]}, r"chunk_starts\[0\] must be non-negative"),
        ({"seq_lens": [3, 2]}, r"seq_lens\[0\]=3 must equal chunk_starts\[0\]=0"),
        ({"chunk_offsets": [0, 1]}, r"chunk_offsets\[1\]=1.*packed token offset 2"),
        ({"token_ids": torch.arange(4).reshape(2, 2)}, r"token_ids must be 1-D packed"),
        ({"token_ids": torch.arange(3)}, r"token_ids contains 3 packed tokens, expected 4"),
        ({"input_embeddings": torch.zeros(4)}, r"input_embeddings must have shape \[tokens, hidden\]"),
        ({"input_embeddings": torch.zeros(3, 4)}, r"input_embeddings contains 3 packed rows, expected 4"),
    ],
)
def test_deepseek_prepare_prefill_inputs_rejects_inconsistent_packed_metadata(updates, error):
    runner, model = _runner_for_prepared_inputs()
    batch = _prefill_batch([2, 2])

    with pytest.raises(ValueError, match=error):
        runner.prepare_prefill_inputs(model, replace(batch, **updates))


def test_deepseek_prepare_prefill_inputs_maps_chunk_metadata():
    runner, model = _runner_for_prepared_inputs()
    model = replace(model, runtime=replace(model.runtime, max_seq_len=129))
    layout = runner._compiled.layout

    prepared = runner.prepare_prefill_inputs(
        model,
        _prefill_batch([3], chunk_starts=[126], token_base=10),
    )

    assert prepared.request_ids == ("req-a",)
    assert prepared.ranks == (0,)
    assert prepared.actual_tokens == (3,)
    assert prepared.kernel_tokens == 128
    assert prepared.x_hc.shape == (8, 128, 4, 4)
    assert prepared.x_hc.dtype == torch.float32
    assert prepared.ori_block_table.shape == (8, 128)
    assert prepared.ori_block_table[0, :4].tolist() == [0, 0, 0, 0]
    assert prepared.hca_cmp_block_table.shape == (layout.ranks, layout.prefill_cmp_max_blocks)
    assert prepared.csa_cmp_block_table.shape == (layout.ranks, layout.prefill_cmp_max_blocks)
    assert prepared.idx_block_table.shape == (layout.ranks, layout.prefill_idx_max_blocks)
    assert prepared.position_ids.shape == (8, 128)
    assert prepared.position_ids[0, :4].tolist() == [126, 127, 128, 128]
    assert torch.all(prepared.position_ids[0] < model.runtime.max_seq_len)
    assert prepared.input_ids[0, :4].tolist() == [10, 11, 12, 10]
    assert prepared.ori_slot_mapping.shape == (8, 128)
    assert prepared.ori_slot_mapping[0, :4].tolist() == [126, 127, 0, -1]
    assert prepared.hca_cmp_slot_mapping.shape == (8, 128)
    assert prepared.hca_cmp_slot_mapping[0, :3].tolist() == [-1, 1, -1]
    assert prepared.hca_cmp_slot_mapping[0, 3].item() == -1
    assert prepared.csa_cmp_slot_mapping.shape == (8, 128)
    assert prepared.csa_cmp_slot_mapping[0, :3].tolist() == [-1, 63, -1]
    assert prepared.csa_cmp_slot_mapping[0, 3].item() == -1
    assert prepared.csa_idx_slot_mapping.shape == (8, 128)
    assert prepared.csa_idx_slot_mapping[0, :3].tolist() == [-1, 95, -1]
    assert prepared.csa_idx_slot_mapping[0, 3].item() == -1
    assert prepared.hca_state_slot_mapping.shape == (8, 128)
    assert prepared.hca_state_slot_mapping[0, :4].tolist() == [
        30,
        31,
        24,
        -1,
    ]
    assert prepared.csa_state_slot_mapping.shape == (8, 128)
    assert prepared.csa_state_slot_mapping[0, :4].tolist() == [
        18,
        19,
        16,
        -1,
    ]
    assert prepared.csa_inner_state_slot_mapping.shape == (8, 128)
    assert prepared.csa_inner_state_slot_mapping[0, :4].tolist() == [
        22,
        23,
        20,
        -1,
    ]
    assert prepared.num_tokens_per_owner.tolist() == [3, 0, 0, 0, 0, 0, 0, 0]
    assert prepared.logit_row_indices[0].tolist() == [2, -1, -1, -1, -1, -1, -1, -1]


@pytest.mark.parametrize(
    ("actual_tokens", "kernel_tokens", "num_speculative_tokens"),
    [(129, 256, 0), (8191, 8192, 1), (8192, 8192, 1)],
)
def test_deepseek_prepare_prefill_inputs_uses_dynamic_main_extent(
    actual_tokens,
    kernel_tokens,
    num_speculative_tokens,
):
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = num_speculative_tokens
    model = replace(
        model,
        runtime=replace(
            model.runtime,
            max_seq_len=8193,
            max_num_batched_tokens=8192,
            max_prefill_tokens_per_request=(
                runner._compiled.kernel_contract.max_prefill_tokens_per_request
            ),
        ),
    )
    grouped_rows = _grouped_cache_rows(1)
    compressed_blocks = max(1, (actual_tokens + 127) // 128)
    grouped_rows[0]["cmp_c128"] = list(range(100, 100 + compressed_blocks))
    grouped_rows[0]["cmp_c4"] = list(range(200, 200 + compressed_blocks))
    grouped_rows[0]["idx"] = list(range(300, 300 + compressed_blocks))

    prepared = runner.prepare_prefill_inputs(
        model,
        _prefill_batch([actual_tokens], block_ids_by_group=grouped_rows),
    )

    assert prepared.actual_tokens == (actual_tokens,)
    assert prepared.kernel_tokens == kernel_tokens
    assert prepared.x_hc.shape == (8, kernel_tokens, 4, 4)
    row_tensors = (
        "input_ids",
        "position_ids",
        "ori_slot_mapping",
        "hca_cmp_slot_mapping",
        "hca_state_slot_mapping",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    )
    assert all(getattr(prepared, name).shape == (8, kernel_tokens) for name in row_tensors)
    assert prepared.logit_row_indices[0, 0].item() == actual_tokens - 1
    if actual_tokens < kernel_tokens:
        assert torch.all(prepared.ori_slot_mapping[0, actual_tokens:] == -1)
    if actual_tokens == 8192:
        assert prepared.hca_cmp_block_table[0, :64].tolist() == list(range(100, 164))
        assert prepared.csa_cmp_block_table[0, :64].tolist() == list(range(200, 264))
        assert prepared.csa_cmp_slot_mapping[0, 3].item() == 200 * 32
        assert prepared.csa_cmp_slot_mapping[0, 4099].item() == 232 * 32
        assert prepared.csa_cmp_slot_mapping[0, 8191].item() == 263 * 32 + 31
    if actual_tokens == 129:
        runner._compiled.runtime_model = replace(
            model, runtime=replace(model.runtime, max_num_batched_tokens=actual_tokens)
        )
        assert runner._prefill_buffer_tokens() == kernel_tokens


def test_deepseek_prepare_prefill_inputs_pads_mixed_ranks_to_one_dynamic_extent():
    runner, model = _runner_for_prepared_inputs()
    model = replace(
        model,
        runtime=replace(
            model.runtime,
            max_seq_len=1024,
            max_num_batched_tokens=512,
            max_prefill_tokens_per_request=(
                runner._compiled.kernel_contract.max_prefill_tokens_per_request
            ),
        ),
    )
    chunk_lens = [129, 257]
    grouped_rows = _grouped_cache_rows(2)
    for row, block_count in zip(grouped_rows, (2, 3), strict=True):
        for group_name in ("cmp_c128", "cmp_c4", "idx"):
            first_block = row[group_name][0] * 10
            row[group_name] = list(range(first_block, first_block + block_count))
    prepared = runner.prepare_prefill_inputs(
        model,
        _prefill_batch(
            chunk_lens,
            cache_partitions=[1, 6],
            block_ids_by_group=grouped_rows,
        ),
    )

    assert prepared.actual_tokens == (129, 257)
    assert prepared.kernel_tokens == 384
    assert prepared.x_hc.shape == (8, 384, 4, 4)
    assert prepared.num_tokens_per_owner.tolist() == [0, 129, 0, 0, 0, 0, 257, 0]
    assert prepared.logit_row_indices[1, 0].item() == 128
    assert prepared.logit_row_indices[6, 0].item() == 256
    assert torch.all(prepared.ori_slot_mapping[1, 129:] == -1)
    assert torch.all(prepared.ori_slot_mapping[6, 257:] == -1)

def test_deepseek_prepare_decode_inputs_accepts_device_embedding_batch():
    runner, model = _runner_for_prepared_inputs()

    prepared = runner.prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=None,
            seq_lens=torch.tensor([128], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )
    assert prepared.x_hc is None
    assert prepared.input_ids[0, :2].tolist() == [5, 5]
    assert prepared.block_table is not None
    assert prepared.block_counts is not None


def test_deepseek_prepare_mtp_target_inputs_limits_partial_chunk_rows():
    runner, model = _runner_for_prepared_inputs()

    prepared = runner.prepare_mtp_target_inputs(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            seq_lens=torch.tensor([129], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
        token_rows=torch.tensor([[5, 5]], dtype=torch.long),
        positions=((128, 128),),
        active_width=1,
    )

    assert prepared.input_ids[0, :2].tolist() == [5, 5]
    assert prepared.position_ids[0, :2].tolist() == [128, 128]
    assert prepared.num_tokens_per_owner.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert prepared.logit_row_indices[0].tolist() == [0, -1, -1, -1, -1, -1, -1, -1]


def test_deepseek_prepare_decode_inputs_rebuilds_slot_metadata():
    runner, model = _runner_for_prepared_inputs()
    original_metadata = runner.cache_metadata

    class CountingMetadata:
        def __init__(self):
            self.ring_table_calls = 0

        def ring_block_table_from_ids(self, *args, **kwargs):
            self.ring_table_calls += 1
            return original_metadata.ring_block_table_from_ids(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(original_metadata, name)

    counting_metadata = CountingMetadata()
    runner.cache_metadata = counting_metadata

    def prepare(seq_len, grouped_rows):
        return runner.prepare_decode_inputs(
            model,
            DecodeBatch(
                request_ids=["req-a"],
                token_ids=torch.tensor([[5]], dtype=torch.long),
                hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
                seq_lens=torch.tensor([seq_len], dtype=torch.int32),
                block_ids_by_group=grouped_rows,
                cache_partitions=[0],
            ),
        )

    first = prepare(128, _grouped_cache_rows(1))
    first_ring_table_calls = counting_metadata.ring_table_calls
    assert first_ring_table_calls == 3
    assert first.block_table.is_shared()

    second = prepare(129, _grouped_cache_rows(1))
    assert counting_metadata.ring_table_calls == 2 * first_ring_table_calls
    assert second.block_table.data_ptr() == first.block_table.data_ptr()
    assert second.position_ids[0, 0].item() == 128

    changed_rows = _grouped_cache_rows(1)
    changed_rows[0]["ori"] = [1]
    prepare(130, changed_rows)
    assert counting_metadata.ring_table_calls == 3 * first_ring_table_calls


def test_deepseek_early_decode_prepare_uses_isolated_ping_pong_slots():
    runner, model = _runner_for_prepared_inputs()
    batch = DecodeBatch(
        request_ids=["req-a"],
        token_ids=torch.tensor([[99]], dtype=torch.long),
        hidden_states=None,
        seq_lens=torch.tensor([128], dtype=torch.int32),
        block_ids_by_group=_grouped_cache_rows(1),
        cache_partitions=[0],
    )

    first = runner.prepare_decode(model, batch, buffer_slot=0)
    second = runner.prepare_decode(model, batch, buffer_slot=1)

    assert first.buffer_slot == 0
    assert second.buffer_slot == 1
    assert first.input_ids.data_ptr() != second.input_ids.data_ptr()
    assert first.block_table.data_ptr() != second.block_table.data_ptr()
    # Early preparation must not consume the not-yet-known prior-step token.
    assert first.input_ids.eq(0).all()
    assert second.input_ids.eq(0).all()
    second.input_ids.fill_(7)
    assert first.input_ids.eq(0).all()


@pytest.mark.parametrize("buffer_slot", [-1, 2])
def test_deepseek_early_decode_prepare_rejects_invalid_buffer_slot(buffer_slot):
    runner, model = _runner_for_prepared_inputs()
    batch = DecodeBatch(
        request_ids=["req-a"],
        token_ids=torch.tensor([[99]], dtype=torch.long),
        hidden_states=None,
        seq_lens=torch.tensor([128], dtype=torch.int32),
        block_ids_by_group=_grouped_cache_rows(1),
        cache_partitions=[0],
    )

    with pytest.raises(ValueError, match="decode buffer_slot must be 0 or 1"):
        runner.prepare_decode(model, batch, buffer_slot=buffer_slot)


def test_deepseek_early_decode_prepare_binds_stable_device_state_per_slot():
    runner, model = _runner_for_prepared_inputs(num_speculative_tokens=1)
    runner._mtp_request_states["req-a"] = SimpleNamespace(
        tail_rank=0,
        tail_slot_id=3,
        generation=7,
        device_state_initialized=True,
    )
    batch = DecodeBatch(
        request_ids=["req-a"],
        token_ids=torch.tensor([[99]], dtype=torch.long),
        hidden_states=None,
        seq_lens=torch.tensor([128], dtype=torch.int32),
        block_ids_by_group=_grouped_cache_rows(1),
        cache_partitions=[0],
        allow_device_greedy_sampling=True,
    )

    first = runner.prepare_decode(model, batch, buffer_slot=0)
    second = runner.prepare_decode(model, batch, buffer_slot=1)

    assert first.mtp_tail_slot_ids[0, 0].item() == 3
    assert first.mtp_state_generations[0, 0].item() == 7
    assert first.mtp_logit_row_indices[0, 0].item() == 1
    assert first.mtp_tail_slot_ids.data_ptr() != second.mtp_tail_slot_ids.data_ptr()
    assert (
        runner._decode_task_args[0].tensors["sampled_ids"].data_ptr()
        != runner._decode_task_args[1].tensors["sampled_ids"].data_ptr()
    )
    second.mtp_tail_slot_ids[0, 0] = 5
    assert first.mtp_tail_slot_ids[0, 0].item() == 3


def test_deepseek_first_decode_prepare_reserves_and_fully_binds_state():
    runner, model = _runner_for_prepared_inputs(num_speculative_tokens=1)
    runner._l3_shared_buffers_ready = True
    runner._bind_prepared_mtp_dispatch = lambda inputs, _hidden_size, _vocab_size: replace(
        inputs,
        dispatch_args=("fused",),
    )
    batch = DecodeBatch(
        request_ids=["req-a"],
        token_ids=torch.tensor([[99]], dtype=torch.long),
        hidden_states=None,
        seq_lens=torch.tensor([128], dtype=torch.int32),
        block_ids_by_group=_grouped_cache_rows(1),
        cache_partitions=[0],
        allow_device_greedy_sampling=True,
    )

    prepared = runner.prepare_decode(model, batch, buffer_slot=0)

    state = runner._mtp_request_states["req-a"]
    assert state.tail_rank == 0
    assert state.tail_slot_id == 0
    assert state.generation == 1
    assert not state.device_state_initialized
    assert prepared.mtp_tail_slot_ids[0, 0].item() == 0
    assert prepared.mtp_state_generations[0, 0].item() == 1
    assert prepared.dispatch_args == ("fused",)

    state.device_state_initialized = True
    state.draft_token_id = 5
    runner._require_decode_callable = lambda: SimpleNamespace(name="decode_mtp_fused")
    runner._submit_l3 = lambda _callable, *args: (
        SimpleNamespace(wait=lambda: None)
        if args == ("fused",)
        else pytest.fail("prepared dispatch arguments changed")
    )
    pending = runner.dispatch_prepared_decode(model, batch, prepared)
    assert pending.states == (state,)


def test_deepseek_mtp_device_state_initializes_sampling_metadata():
    runner = DeepSeekV4ModelRunner.__new__(DeepSeekV4ModelRunner)
    runner._mtp_buffers = SimpleNamespace(
        state_init_tokens=torch.zeros((1, 1, 2), dtype=torch.long),
        state_init_meta=torch.zeros((1, 1, 7), dtype=torch.int32),
    )
    device_tokens = SimpleNamespace(shards=(SimpleNamespace(data_ptr=0x1000),))
    device_meta = SimpleNamespace(shards=(SimpleNamespace(data_ptr=0x2000),))
    runner._materialize_mtp_device_state_tokens = lambda: device_tokens
    runner._materialize_mtp_device_state_meta = lambda: device_meta
    copied_nbytes = []
    runner._shared_l3_worker = lambda: SimpleNamespace(
        copy_to=lambda _dst, _src, nbytes, *, dst_offset, worker_id: copied_nbytes.append(
            (nbytes, dst_offset, worker_id)
        )
    )
    state = SimpleNamespace(
        device_state_initialized=False,
        tail_rank=0,
        tail_slot_id=0,
        tail_token_id=11,
        draft_token_id=12,
        tail_position=128,
        generation=7,
        committed_count=5,
    )

    runner._initialize_mtp_device_state(
        state,
        "req-a",
        SamplingParams(temperature=0.25, top_p=1.0, top_k=64, seed=123),
    )

    assert runner._mtp_buffers.state_init_tokens[0, 0].tolist() == [11, 12]
    assert runner._mtp_buffers.state_init_meta[0, 0].tolist() == [
        1,
        7,
        128,
        5,
        250000,
        64,
        123,
    ]
    assert copied_nbytes == [(16, 0, 0), (28, 0, 0)]
    assert state.device_state_initialized is True


def test_deepseek_prefill_waits_for_both_pending_mtp_decode_slots():
    runner, _model = _runner_for_prepared_inputs()
    waits = []

    class FakeDispatch:
        def __init__(self, buffer_slot):
            self.buffer_slot = buffer_slot

        def wait(self):
            waits.append(self.buffer_slot)

    first = FakeDispatch(0)
    second = FakeDispatch(1)
    runner._track_pending_mtp_dispatch(0, first)
    runner._track_pending_mtp_dispatch(1, second)

    assert waits == []
    assert runner._pending_mtp_dispatches == {0: first, 1: second}

    runner._wait_for_pending_mtp_dispatches()

    assert waits == [0, 1]
    assert runner._pending_mtp_dispatches == {}


def test_deepseek_prefill_context_preserves_prepare_reserved_state(monkeypatch):
    runner, _model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = 1
    state = runner._reserve_mtp_request_state("req-a", 0)
    slot = state.tail_slot_id
    generation = state.generation
    layout = runner._compiled.layout
    tokens = 1
    hidden = 4
    inputs = SimpleNamespace(
        request_ids=("req-a",),
        ranks=(0,),
        actual_tokens=(tokens,),
        x_hc=torch.zeros(layout.ranks, tokens, 1, hidden),
        input_ids=torch.zeros(layout.ranks, tokens, dtype=torch.long),
        position_ids=torch.zeros(layout.ranks, tokens, dtype=torch.int32),
        ori_block_table=torch.zeros(layout.ranks, 1, dtype=torch.int32),
        ori_slot_mapping=torch.zeros(layout.ranks, tokens, dtype=torch.int32),
    )
    monkeypatch.setattr(runner, "_run_mtp_prefill_rows", lambda **_kwargs: None)

    runner._capture_mtp_prefill_context(
        inputs,
        torch.zeros(layout.ranks, tokens, 1, hidden),
    )

    assert runner._mtp_request_states["req-a"] is state
    assert state.tail_slot_id == slot
    assert state.generation == generation
    assert state.prefill_context is not None


def _mtp_prefill_capture_harness(
    *,
    ranks: int = 1,
    prefill_seq: int = 128,
    prefill_ori_max_blocks: int = 4,
    decode_batch: int = 1,
):
    layout = replace(
        DeepSeekV4CacheLayout(),
        ranks=ranks,
        prefill_seq=prefill_seq,
        prefill_ori_max_blocks=prefill_ori_max_blocks,
        decode_batch=decode_batch,
        decode_seq=2,
        decode_tokens=2,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
        )
    )
    calls = []

    def capture_rows(**kwargs):
        calls.append(
            {
                name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
        )
        return None

    runner._run_mtp_prefill_rows = capture_rows
    return runner, layout, calls


def _mtp_capture_inputs(
    layout: DeepSeekV4CacheLayout,
    owners,
    *,
    kernel_tokens: int | None = None,
):
    # owner: (request_id, rank, actual_tokens, start, token_base, hidden_base)
    extent = kernel_tokens or max(owner[2] for owner in owners)
    inputs = SimpleNamespace(
        request_ids=tuple(owner[0] for owner in owners),
        ranks=tuple(owner[1] for owner in owners),
        actual_tokens=tuple(owner[2] for owner in owners),
        x_hc=torch.zeros((layout.ranks, extent, 4, 1), dtype=torch.float32),
        input_ids=torch.zeros((layout.ranks, extent), dtype=torch.long),
        position_ids=torch.zeros((layout.ranks, extent), dtype=torch.int32),
        ori_block_table=torch.zeros((layout.ranks, layout.prefill_ori_max_blocks), dtype=torch.int32),
        ori_slot_mapping=torch.full((layout.ranks, extent), -1, dtype=torch.long),
    )
    pre_hc = torch.zeros((layout.ranks, layout.prefill_seq, 4, 1), dtype=torch.float32)
    for _request_id, rank, actual_tokens, start, token_base, hidden_base in owners:
        token_values = torch.arange(actual_tokens)
        positions = start + token_values
        inputs.x_hc[rank, :actual_tokens, 0, 0] = token_base + token_values
        inputs.input_ids[rank, :actual_tokens] = token_base + token_values
        inputs.position_ids[rank, :actual_tokens] = positions
        inputs.ori_slot_mapping[rank, :actual_tokens] = positions
        tail_tokens = min(actual_tokens, layout.prefill_seq)
        pre_hc[rank, :tail_tokens, 0, 0] = hidden_base + torch.arange(tail_tokens)
    return inputs, pre_hc


def _assert_tensor_range(tensor: torch.Tensor, start: int, stop: int) -> None:
    assert torch.equal(tensor, torch.arange(start, stop, dtype=tensor.dtype))


def test_deepseek_mtp_prefill_advances_one_row_behind_chunked_main_prefill():
    runner, layout, calls = _mtp_prefill_capture_harness(
        ranks=2,
        prefill_seq=4,
    )
    first, first_pre_hc = _mtp_capture_inputs(layout, [("req-a", 0, 4, 0, 10, 100)])
    runner._capture_mtp_prefill_context(first, first_pre_hc)
    assert len(calls) == 1
    assert calls[0]["rank"] == 0
    assert calls[0]["input_ids"].tolist() == [11, 12, 13]
    assert calls[0]["position_ids"].tolist() == [0, 1, 2]
    assert calls[0]["prev_hidden_states"][:, 0, 0].tolist() == [100.0, 101.0, 102.0]
    assert runner._mtp_request_states["req-a"].prefill_context.position_id == 3

    second, second_pre_hc = _mtp_capture_inputs(layout, [("req-a", 0, 2, 4, 20, 200)])
    runner._capture_mtp_prefill_context(second, second_pre_hc)
    assert len(calls) == 2
    assert calls[1]["input_ids"].tolist() == [20, 21]
    assert calls[1]["position_ids"].tolist() == [3, 4]
    assert calls[1]["prev_hidden_states"][:, 0, 0].tolist() == [103.0, 200.0]
    assert runner._mtp_request_states["req-a"].prefill_context.position_id == 5


@pytest.mark.parametrize(
    ("actual_tokens", "has_pending"),
    [(127, False), (128, True), (129, False)],
)
def test_deepseek_mtp_prefill_capture_uses_fixed_tail_window_at_boundaries(
    actual_tokens,
    has_pending,
):
    runner, layout, calls = _mtp_prefill_capture_harness()
    start = int(has_pending)
    tail_start = max(actual_tokens - layout.prefill_seq, 0)
    expected_rows = min(actual_tokens - 1, layout.prefill_seq - 1)
    old_context = None
    if has_pending:
        pending, pending_pre_hc = _mtp_capture_inputs(layout, [("req-boundary", 0, 1, 0, 0, 9_999)])
        runner._capture_mtp_prefill_context(pending, pending_pre_hc)
        old_context = runner._mtp_request_states["req-boundary"].prefill_context

    inputs, pre_hc = _mtp_capture_inputs(
        layout, [("req-boundary", 0, actual_tokens, start, 200, 1_000 + tail_start)]
    )
    runner._capture_mtp_prefill_context(inputs, pre_hc)

    assert len(calls) == 1
    call = calls[0]
    for tensor, range_start, range_stop in (
        (call["input_ids"], 201 + tail_start, 200 + actual_tokens),
        (call["position_ids"], start + tail_start, start + actual_tokens - 1),
        (call["prev_hidden_states"][:, 0, 0], 1_000 + tail_start, 999 + actual_tokens),
    ):
        assert tensor.numel() == expected_rows
        _assert_tensor_range(tensor, range_start, range_stop)
    context = runner._mtp_request_states["req-boundary"].prefill_context
    assert context.position_id == start + actual_tokens - 1
    assert context.slot_mapping == start + actual_tokens - 1
    assert context.prev_hidden_state[0, 0].item() == 999 + actual_tokens
    if has_pending:
        assert 9_999 not in call["prev_hidden_states"][:, 0, 0]
        assert context is not old_context


def test_deepseek_mtp_prefill_rebuilds_fixed_tail_window_for_mixed_long_chunks():
    runner, layout, calls = _mtp_prefill_capture_harness(
        ranks=2,
        prefill_seq=128,
        prefill_ori_max_blocks=64,
        decode_batch=2,
    )
    kernel_tokens = 8_192
    short_tokens = 65
    inputs, pre_hc = _mtp_capture_inputs(
        layout,
        [
            ("req-long", 0, kernel_tokens, 0, 20_000, 30_000),
            ("req-short", 1, short_tokens, 0, 50_000, 60_000),
        ],
        kernel_tokens=kernel_tokens,
    )
    runner._capture_mtp_prefill_context(inputs, pre_hc)

    assert [call["rank"] for call in calls] == [0, 1]
    _assert_tensor_range(calls[0]["input_ids"], 28_065, 28_192)
    _assert_tensor_range(calls[0]["position_ids"], 8_064, 8_191)
    _assert_tensor_range(calls[0]["prev_hidden_states"][:, 0, 0], 30_000, 30_127)
    _assert_tensor_range(calls[1]["input_ids"], 50_001, 50_065)
    _assert_tensor_range(calls[1]["position_ids"], 0, 64)
    _assert_tensor_range(calls[1]["prev_hidden_states"][:, 0, 0], 60_000, 60_064)

    long_context = runner._mtp_request_states["req-long"].prefill_context
    short_context = runner._mtp_request_states["req-short"].prefill_context
    assert long_context.position_id == 8191
    assert long_context.prev_hidden_state[0, 0].item() == 30_127
    assert short_context.position_id == 64
    assert short_context.prev_hidden_state[0, 0].item() == 60_064


def test_deepseek_run_decode_dispatches_active_token_count():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.decode = DeepSeekV4L3Callable(compiled=object(), name="decode")
    captured: dict[str, object] = {}

    def fake_decode_fwd_args(inputs):
        captured["num_tokens_per_owner"] = inputs.num_tokens_per_owner
        # The device-resident pre-HC slot is normally allocated before the L3
        # worker fork (TaskArgs.allocate_device at init) -- the _decode_fwd_args
        # hot path only assembles the tuple and must not touch the worker. This
        # test never runs allocate_device, so inject a stand-in for
        # _execute_main_decode to read.
        task_args = runner._decode_task_args[inputs.buffer_slot]
        task_args.tensors.setdefault(
            "pre_hc_hidden_out",
            torch.empty(
                runner._compiled.layout.ranks,
                runner._compiled.layout.decode_tokens,
                runner._compiled.layout.hc_mult,
                model.config.hidden_size,
                dtype=torch.float32,
            ),
        )
        return ()

    runner._ensure_l3_shared_buffers = lambda _model: None
    runner._decode_fwd_args = fake_decode_fwd_args
    runner._run_l3 = lambda _callable, *args: None

    result = runner.run_decode(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[5]], dtype=torch.long),
            hidden_states=torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
            seq_lens=torch.tensor([128], dtype=torch.int32),
            block_ids_by_group=_grouped_cache_rows(1),
            cache_partitions=[0],
        ),
    )

    assert captured["num_tokens_per_owner"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert result.logits.shape == (1, model.config.vocab_size)


def test_deepseek_main_decode_copies_pre_hc_to_bound_host_buffer():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.decode = DeepSeekV4L3Callable(compiled=object(), name="decode")
    runner._compiled.mtp_decode = DeepSeekV4L3Callable(compiled=object(), name="mtp_decode")
    layout = runner._compiled.layout
    host_pre_hc = torch.empty(
        layout.ranks,
        layout.decode_tokens,
        layout.hc_mult,
        model.config.hidden_size,
        dtype=torch.float32,
    )
    device_pre_hc = object()
    hidden_out = torch.empty(
        layout.ranks, layout.decode_tokens, model.config.hidden_size, dtype=torch.bfloat16
    )
    logits = torch.empty(layout.ranks, layout.decode_tokens, model.config.vocab_size, dtype=torch.float32)
    sampled_ids = torch.empty(layout.ranks, layout.decode_tokens, 8, dtype=torch.int32)
    runner._decode_task_args = [
        SimpleNamespace(
            tensors={
                "pre_hc_hidden_out": device_pre_hc,
                "hidden_out": hidden_out,
                "logits": logits,
                "sampled_ids": sampled_ids,
            }
        )
    ]
    runner._main_pre_hc_host_mirror = host_pre_hc
    copied: list[tuple[object, torch.Tensor]] = []
    runner._require_decode_callable = lambda: object()
    runner._decode_fwd_args = lambda *_args: ()
    runner._run_l3 = lambda *_args: None
    runner._shared_l3_worker = lambda: SimpleNamespace(
        copy_stacked_from=lambda source, destination: copied.append((source, destination))
    )
    prepared = SimpleNamespace(
        per_rank_counts=(1,) + (0,) * (layout.ranks - 1),
        actual_batch=1,
        ranks=(0,),
        buffer_slot=0,
    )

    output = runner._execute_main_decode(model, prepared, active_seq=1)

    assert copied == [(device_pre_hc, host_pre_hc)]
    assert output.pre_hc_hidden is host_pre_hc


def test_deepseek_prepared_mtp_decode_skips_redundant_dynamic_input_staging():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = 1
    runner._compiled.layout = deepseek_v4_decode_layout(1)
    runner._compiled.decode = DeepSeekV4L3Callable(compiled=object(), name="decode_mtp_fused")
    runner._decode_flow = runner._run_mtp_decode
    layout = runner._compiled.layout
    main_sampled_ids = torch.zeros(
        layout.ranks,
        layout.decode_tokens,
        8,
        dtype=torch.int32,
    )
    # The MTP decode TaskArgs owns the per-slot reclaimed outputs; stand in a
    # fake whose .tensors carries the reclaimed slot tensors.
    mtp_reclaim = SimpleNamespace(
        tensors={
            "accepted_counts": torch.ones(
                layout.ranks,
                layout.decode_batch,
                dtype=torch.int32,
            ),
            "input_ids": torch.zeros(
                layout.ranks,
                layout.decode_tokens,
                dtype=torch.long,
            ),
            "position_ids": torch.zeros(
                layout.ranks,
                layout.decode_tokens,
                dtype=torch.int32,
            ),
            "sampled_ids": torch.zeros(
                layout.ranks,
                layout.decode_tokens,
                8,
                dtype=torch.int32,
            ),
        }
    )
    state = SimpleNamespace(
        draft_token_id=4,
        tail_token_id=3,
        tail_slot_id=0,
        tail_position=126,
        proposed_tokens=0,
        accepted_tokens=0,
        committed_count=0,
        generation=1,
        device_state_initialized=True,
    )
    runner._mtp_request_states["req-a"] = state
    placeholder = torch.empty(0)
    staged = DeepSeekV4PreparedDecodeInputs(
        request_ids=("req-a",),
        ranks=(0,),
        local_rows=(0,),
        actual_batch=1,
        per_rank_counts=(1,) + (0,) * (layout.ranks - 1),
        x_hc=None,
        input_ids=placeholder,
        position_ids=placeholder,
        kv_seq_lens=placeholder,
        block_table=placeholder,
        hca_cmp_block_table=placeholder,
        csa_cmp_block_table=placeholder,
        idx_block_table=placeholder,
        hca_compress_state_block_table=placeholder,
        csa_compress_state_block_table=placeholder,
        csa_inner_compress_state_block_table=placeholder,
        block_counts=placeholder,
        block_ids_by_group=(),
        num_tokens_per_owner=placeholder,
        logit_row_indices=placeholder,
        mtp_tail_slot_ids=torch.zeros(layout.ranks, layout.decode_batch, dtype=torch.int32),
        buffer_slot=0,
        dispatch_args=None,
    )
    dispatches = []

    runner._ensure_l3_shared_buffers = lambda _model: None
    runner._stage_decode_dynamic_inputs = lambda *_args, **_kwargs: pytest.fail(
        "prepared MTP decode must bind recurrent inputs from device state"
    )
    runner._require_mtp_buffers = lambda: SimpleNamespace()
    runner._fused_mtp_decode_args = lambda _main_args, _inputs, _active_tokens: ("fused",)
    runner._bind_prepared_mtp_dispatch = lambda inputs, _hidden_size, _vocab_size: replace(
        inputs,
        dispatch_args=("fused",),
    )
    runner._decode_task_args = [SimpleNamespace(tensors={"sampled_ids": main_sampled_ids})]
    runner._mtp_decode_task_args = [mtp_reclaim]
    runner._decode_input_slots = [{}]  # fused compose is mocked; prepend buffers unused here

    def fake_submit_l3(callable_spec, *args):
        dispatches.append((callable_spec.name, args))

        def complete():
            main_sampled_ids[0, 0, 0] = 5
            main_sampled_ids[0, 1, 0] = 9
            mtp_reclaim.tensors["accepted_counts"][0, 0] = 2
            mtp_reclaim.tensors["input_ids"][0, 1] = 9
            mtp_reclaim.tensors["position_ids"][0, 1] = 128
            mtp_reclaim.tensors["sampled_ids"][0, 0, 0] = 7

        return SimpleNamespace(wait=complete)

    runner._submit_l3 = fake_submit_l3

    result = runner._run_mtp_decode(
        model,
        DecodeBatch(
            request_ids=["req-a"],
            token_ids=torch.tensor([[3]], dtype=torch.long),
            hidden_states=None,
            seq_lens=torch.tensor([128], dtype=torch.int32),
            cache_partitions=[0],
            allow_device_greedy_sampling=True,
        ),
        prepared=staged,
    )

    assert dispatches == [("decode_mtp_fused", ("fused",))]
    assert result.accepted_token_ids == [[5, 9]]
    # The verifier count wins even when the host draft mirror is stale.
    assert state.draft_token_id == 7
    assert state.tail_token_id == 3
    assert state.tail_position == 126
    assert state.proposed_tokens == 1
    assert state.accepted_tokens == 1
    assert state.committed_count == 2


def test_deepseek_sync_mtp_decode_uses_general_prepare_fallback():
    runner, model = _runner_for_prepared_inputs()
    runner._compiled.num_speculative_tokens = 1
    batch = SimpleNamespace(allow_device_greedy_sampling=True)
    placeholder_batch = object()
    prepared = SimpleNamespace(
        mtp_tail_slot_ids=torch.zeros(1, dtype=torch.int32),
        dispatch_args=("bound",),
    )
    pending = object()
    calls = []

    runner._device_state_placeholder_batch = lambda value: (
        calls.append(("placeholder", value)) or placeholder_batch
    )
    runner.prepare_decode = lambda runtime_model, value, *, buffer_slot: (
        calls.append(("prepare", runtime_model, value, buffer_slot)) or prepared
    )
    runner.dispatch_prepared_decode = lambda runtime_model, value, inputs: (
        calls.append(("dispatch", runtime_model, value, inputs)) or pending
    )

    result = runner._dispatch_mtp_decode(model, batch)

    assert result is pending
    assert calls == [
        ("placeholder", batch),
        ("prepare", model, placeholder_batch, 0),
        ("dispatch", model, batch, prepared),
    ]


def _tiny_prefill_layout(*, ranks: int = 1, prefill_seq: int = 1):
    return replace(
        DeepSeekV4CacheLayout(),
        ranks=ranks,
        hc_mult=1,
        prefill_seq=prefill_seq,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
        prefill_ori_max_blocks=1,
    )


def test_deepseek_prefill_staging_keeps_worker_resident_cache_tensors_out():
    runner = _TaskArgsStubRunner(hidden=1, vocab=1, layout=_tiny_prefill_layout())
    prefill = prefill_task_args(runner, hidden=1, vocab=1)
    prefill.allocate_host_shared(None)

    assert not hasattr(runner, "_decode_work_cache")
    for name in (
        "kv_cache",
        "cmp_kv",
        "idx_kv_cache",
        "idx_kv_scale",
        "hca_compress_state",
        "csa_compress_state",
        "csa_inner_compress_state",
    ):
        assert name not in prefill.tensors


def test_deepseek_dynamic_prefill_views_reuse_pre_fork_shared_storage():
    layout = _tiny_prefill_layout(ranks=2, prefill_seq=4)
    runner = _TaskArgsStubRunner(
        hidden=1,
        vocab=1,
        layout=layout,
        prefill_buffer_tokens=12,
    )
    task_args = prefill_task_args(runner, hidden=1, vocab=1)
    task_args.allocate_host_shared(None)
    assert task_args.tensors["x_hc"].shape == (2, 12, 1, 1)

    task_args.tensors["hidden_out"].fill_(7)
    task_args.tensors["pre_hc_hidden_out"].fill_(7)
    task_args.tensors["logits"].fill_(7)
    task_args.clear_outputs()
    assert task_args.tensors["hidden_out"].eq(7).all()
    assert task_args.tensors["pre_hc_hidden_out"].eq(0).all()
    assert task_args.tensors["logits"].eq(0).all()

    kernel_tokens = 8
    dynamic_inputs = task_args_module._PREFILL_DYNAMIC_INPUT_NAMES
    values = {}
    for name in dynamic_inputs:
        target = task_args.tensors[name]
        shape = (layout.ranks, kernel_tokens, *target.shape[2:])
        values[name] = torch.arange(target[:, :kernel_tokens].numel(), dtype=target.dtype).reshape(shape)
    values["ori_block_table"] = torch.tensor([[11], [22]], dtype=torch.int32)

    task_args.stage_for_tokens(values, kernel_tokens)

    for name in dynamic_inputs:
        staged = task_args.token_view(name, kernel_tokens)
        assert staged.is_shared() and staged.is_contiguous()
        assert staged.data_ptr() == task_args.tensors[name].data_ptr()
        assert torch.equal(staged, values[name])
    assert torch.equal(task_args.tensors["ori_block_table"], values["ori_block_table"])

    shared_scalar = torch.zeros(1).share_memory_()
    task_args.build = lambda: tuple(task_args.tensors.get(name, shared_scalar) for name in task_args.names)
    dispatch_values = dict(zip(task_args.names, task_args.build_for_tokens(kernel_tokens), strict=True))

    for name in dynamic_inputs | {"hidden_out"}:
        active = dispatch_values[name]
        assert active.shape[1] == kernel_tokens
        assert active.is_shared() and active.is_contiguous()
        assert active.data_ptr() == task_args.tensors[name].data_ptr()
    assert dispatch_values["x_hc"].shape == (2, 8, 1, 1)
    assert dispatch_values["hidden_out"].shape == (2, 8, 1)
    assert dispatch_values["pre_hc_hidden_out"].shape == (2, 4, 1, 1)
    assert dispatch_values["logits"].shape == (2, 8, 1)


def test_deepseek_mtp_prefill_and_decode_reuse_same_kv_cache():
    layout = DeepSeekV4CacheLayout(
        ranks=1,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
        block_size=1,
        prefill_ori_max_blocks=1,
        decode_ori_max_blocks=1,
        sliding_window=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
            mtp_prefill=DeepSeekV4L3Callable(compiled=object(), name="mtp_prefill"),
            mtp_decode=DeepSeekV4L3Callable(compiled=object(), name="mtp_decode"),
        )
    )
    weight = torch.arange(2, dtype=torch.float32)
    runner.load_mtp_weights = lambda: weight_loader.DeepSeekV4MtpWeights(tensors={"weight": weight})

    buffers = runner._ensure_mtp_buffers(hidden_size=1)

    assert buffers is not None
    assert buffers.weights["weight"] is weight
    assert not buffers.weights["weight"].is_shared()
    assert buffers.prefill_kv_cache is not None


def test_deepseek_mtp_prefill_outputs_allocate_empty_rank_shards_once():
    layout = DeepSeekV4CacheLayout(
        ranks=2,
        prefill_seq=3,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
        )
    )

    class FakeWorker:
        def __init__(self):
            self.allocations = []

        def alloc_tensor(self, shape, dtype, init=None, *, worker_id=0):
            tensor = DeviceTensor(
                0x1000 + len(self.allocations) * 0x100000,
                tuple(shape),
                dtype,
            )
            self.allocations.append((tuple(shape), dtype, init, worker_id))
            return tensor

        @staticmethod
        def free_tensor(_tensor, *, worker_id=0):
            pass

    worker = FakeWorker()
    runner._l3_worker = worker

    from pypto_serving.model.deepseek.task_args import mtp_prefill_task_args

    ta = mtp_prefill_task_args(runner, hidden=5)
    ta.allocate_host_shared(None)
    ta.allocate_device(worker, None)
    # idempotent: a second allocate_device must not re-allocate.
    ta.allocate_device(worker, None)

    hidden_out = ta.tensors["hidden_out"]
    pre_hc_hidden_out = ta.tensors["pre_hc_hidden_out"]
    logits = ta.tensors["logits"]
    assert hidden_out.full_shape == (2, 3, 5)
    assert pre_hc_hidden_out.full_shape == (2, 3, 4, 5)
    assert logits.full_shape == (2, 8, 129280)
    assert all(shard.dtype == torch.bfloat16 for shard in hidden_out.shards)
    assert all(shard.dtype == torch.float32 for shard in pre_hc_hidden_out.shards)
    assert all(shard.dtype == torch.float32 for shard in logits.shards)
    assert len(worker.allocations) == 6  # 3 device outputs x 2 ranks, no re-allocation
    assert all(init is None for _shape, _dtype, init, _worker_id in worker.allocations)
    assert [worker_id for _shape, _dtype, _init, worker_id in worker.allocations] == [0, 1] * 3


def test_deepseek_mtp_prefill_reads_only_selected_owner_outputs():
    layout = DeepSeekV4CacheLayout(
        ranks=2,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
        )
    )

    class FakeWorker:
        def __init__(self):
            self.allocations = []
            self.copies = []

        def alloc_tensor(self, shape, dtype, init=None, *, worker_id=0):
            tensor = DeviceTensor(
                0x10000000 + len(self.allocations) * 0x1000000,
                tuple(shape),
                dtype,
            )
            self.allocations.append((tensor, init, worker_id))
            return tensor

        @staticmethod
        def free_tensor(_tensor, *, worker_id=0):
            pass

        def copy_from(self, dst, src, nbytes, *, worker_id=0):
            self.copies.append((dst, src, nbytes, worker_id))

    worker = FakeWorker()
    runner._l3_worker = worker
    runner._mtp_buffers = SimpleNamespace(
        prefill_logits=torch.empty(
            (layout.ranks, 8, 129280),
            dtype=torch.float32,
        ).share_memory_(),
        prefill_pre_hc_mirror=torch.empty(
            (layout.ranks, layout.prefill_seq, 4, 5),
            dtype=torch.float32,
        ).share_memory_(),
    )
    from pypto_serving.model.deepseek.task_args import mtp_prefill_task_args

    ta = mtp_prefill_task_args(runner, hidden=5)
    ta.allocate_host_shared(None)
    ta.allocate_device(worker, None)
    runner._mtp_prefill_task_args = ta

    host_row = runner._read_mtp_prefill_logits(owner_rank=1)
    host_pre_hc = runner._read_mtp_prefill_pre_hc(owner_rank=1, row=0)

    assert host_row.data_ptr() == runner._mtp_buffers.prefill_logits[1, 0].data_ptr()
    assert host_pre_hc.shape == (4, 5)
    assert worker.copies == [
        (
            host_row.data_ptr(),
            ta.tensors["logits"].shards[1].data_ptr,
            129280 * torch.float32.itemsize,
            1,
        ),
        (
            runner._mtp_buffers.prefill_pre_hc_mirror[1, 0].data_ptr(),
            ta.tensors["pre_hc_hidden_out"].shards[1].data_ptr,
            4 * 5 * torch.float32.itemsize,
            1,
        ),
    ]


def test_deepseek_mtp_prefill_args_use_device_outputs():
    layout = DeepSeekV4CacheLayout(
        ranks=1,
        prefill_seq=1,
        decode_batch=1,
        decode_seq=1,
        decode_tokens=1,
    )
    runner = DeepSeekV4ModelRunner(
        compiled=DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir="",
            weight_map={},
            weight_store=None,
            compress_ratios=(),
            layer_plan=(),
            kernel_dir="",
            kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
        )
    )

    class FakeWorker:
        def __init__(self):
            self.allocations = 0

        def alloc_tensor(self, shape, dtype, init=None, *, worker_id=0):
            tensor = DeviceTensor(0x1000 + self.allocations * 0x100000, tuple(shape), dtype)
            self.allocations += 1
            return tensor

        @staticmethod
        def free_tensor(_tensor, *, worker_id=0):
            pass

    runner._l3_worker = FakeWorker()
    shared = torch.empty(2, dtype=torch.float32).share_memory_()
    runner._static_freqs_cos_tensor = lambda: shared
    runner._static_freqs_sin_tensor = lambda: shared
    runner._static_lm_head_weight_tensor = lambda: shared
    runner._materialize_mtp_device_kv_cache = lambda: object()
    from collections import defaultdict

    weight = object()
    runner._mtp_device_weights = defaultdict(lambda: weight)
    runner._mtp_device_weights["_seed"] = weight  # truthy for the `or` short-circuit

    from pypto_serving.model.deepseek.task_args import mtp_prefill_task_args

    ta = mtp_prefill_task_args(runner, hidden=1)
    ta.allocate_host_shared(None)
    ta.allocate_device(runner._l3_worker, None)
    runner._mtp_prefill_task_args = ta

    args = runner._mtp_prefill_args()
    names = ta.names

    assert len(args) == len(names)
    assert args[names.index("hidden_out")] is ta.tensors["hidden_out"]
    assert args[names.index("pre_hc_hidden_out")] is ta.tensors["pre_hc_hidden_out"]
    assert args[names.index("logits")] is ta.tensors["logits"]
    assert "num_tokens_per_owner" not in names


def test_deepseek_release_finished_requests_discards_mtp_state():
    runner, _model = _runner_for_prepared_inputs()
    runner._mtp_request_states = {
        "req-a": SimpleNamespace(proposed_tokens=0, tail_rank=1, tail_slot_id=2),
        "req-b": SimpleNamespace(proposed_tokens=0, tail_rank=None, tail_slot_id=None),
    }
    runner._mtp_free_tail_slots = [[] for _ in range(runner._compiled.layout.ranks)]

    runner.release_finished_requests(["req-a"])

    assert runner._mtp_request_states == {
        "req-b": SimpleNamespace(proposed_tokens=0, tail_rank=None, tail_slot_id=None),
    }
    assert runner._mtp_free_tail_slots[1] == [2]


def _write_deepseek_model_dir(tmp_path: Path, *, quant_method: str = "compressed-tensors") -> Path:
    model_dir = tmp_path / "dsv4-flash-w8a8"
    model_dir.mkdir()
    compress_ratios = _deepseek_flash_compress_ratios()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "torch_dtype": "bfloat16",
        "compress_ratios": compress_ratios,
        "quantization_config": {
            "quant_method": quant_method,
            "format": "int-quantized",
            "quantization_status": "compressed",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    weight_names = deepseek_v4_startup_weight_names(
        43,
        n_routed_experts=256,
        compress_ratios=compress_ratios,
        num_hash_layers=3,
    )
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in weight_names}}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return model_dir


def _deepseek_flash_compress_ratios() -> list[int]:
    return [0, 0, *(4 if layer_id % 2 == 0 else 128 for layer_id in range(2, 43)), 0]


def _write_deepseek_kernel_dir(
    tmp_path: Path,
    *,
    lm_head_tp_size: int,
    use_config_constant: bool = False,
    block_size: int = 128,
    hca_state_blocks: int = 2048,
    csa_state_blocks: int = 4096,
    csa_inner_state_blocks: int = 4096,
) -> Path:
    kernel_dir = tmp_path / f"deepseek-v4-kernels-tp{lm_head_tp_size}"
    kernel_dir.mkdir()
    (kernel_dir / "prefill_hca.py").write_text(
        "\n".join(
            [
                "HCA_STATE_BLOCK_NUM = 64",
                f"HCA_STATE_MAX_BLOCKS = {hca_state_blocks}",
                "",
            ]
        )
    )
    (kernel_dir / "prefill_csa.py").write_text(
        "\n".join(
            [
                "CSA_STATE_BLOCK_NUM = 65",
                f"CSA_STATE_MAX_BLOCKS = {csa_state_blocks}",
                "INNER_STATE_BLOCK_NUM = 65",
                f"INNER_STATE_MAX_BLOCKS = {csa_inner_state_blocks}",
                "",
            ]
        )
    )
    (kernel_dir / "prefill_layer.py").write_text("")
    (kernel_dir / "prefill_fwd.py").write_text("")
    (kernel_dir / "prefill_mtp.py").write_text("")
    (kernel_dir / "decode_layer.py").write_text("")
    (kernel_dir / "decode_fwd.py").write_text("")
    (kernel_dir / "decode_fwd_mtp.py").write_text("")
    (kernel_dir / "decode_mtp.py").write_text("")
    (kernel_dir / "serving_contract.py").write_text(
        "\n".join(
            [
                "class _Contract:",
                "    schema_version = '1'",
                "    prefill_tile_tokens = 128",
                "    max_prefill_tokens_per_request = 8192",
                "    max_prefill_requests_per_partition = 1",
                "    requires_homogeneous_prefill_decode = True",
                "",
                "    @staticmethod",
                "    def padded_prefill_tokens(active_tokens):",
                "        if active_tokens <= 0 or active_tokens > 8192:",
                "            raise ValueError('invalid active prefill extent')",
                "        return ((active_tokens + 127) // 128) * 128",
                "",
                "DEEPSEEK_V4_FLASH_SERVING_CONTRACT = _Contract()",
                "",
            ]
        )
    )
    (kernel_dir / "config.py").write_text(
        "\n".join(
            [
                f"BLOCK_SIZE = {block_size}",
                "DECODE_BATCH = 4",
                "DECODE_SEQ = 2",
                "DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ",
                "PREFILL_BATCH = 1",
                "PREFILL_SEQ = 128",
                "KV_ORI_MAX_BLOCKS = 128",
                "KV_ORI_TABLE_MAX_BLOCKS = 128",
                "KV_CMP_MAX_BLOCKS = 32",
                "IDX_CACHE_MAX_BLOCKS = 64",
                "ORI_KV_BLOCK_NUM = 128",
                "HCA_STATE_PHYSICAL_BLOCKS = 64",
                "CSA_STATE_PHYSICAL_BLOCKS = 65",
                "CSA_INNER_STATE_PHYSICAL_BLOCKS = 65",
                "PREFILL_ORI_MAX_BLOCKS = 128",
                "PREFILL_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS",
                "PREFILL_IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS",
                "EP_WORLD_SIZE = 8",
                f"LM_HEAD_TP_SIZE = {lm_head_tp_size}",
                "",
            ]
        )
    )
    if use_config_constant:
        (kernel_dir / "lm_head.py").write_text("TP_SIZE = LM_HEAD_TP_SIZE\n")
    else:
        (kernel_dir / "lm_head.py").write_text(f"TP_SIZE = {lm_head_tp_size}\n")
    return kernel_dir


def _synthetic_layer_raw(*, layer_id: int, n_experts: int) -> dict[str, torch.Tensor]:
    prefix = f"layers.{layer_id}"
    raw = {
        f"{prefix}.hc_attn_fn": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        f"{prefix}.hc_attn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_attn_base": torch.arange(1, dtype=torch.float32),
        f"{prefix}.attn_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        f"{prefix}.attn.wq_a.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.wq_b.weight": torch.arange(12, dtype=torch.int8).reshape(6, 2),
        f"{prefix}.attn.wq_b.scale": torch.arange(6, dtype=torch.float32),
        f"{prefix}.attn.wkv.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        f"{prefix}.attn.q_norm.weight": torch.arange(2, dtype=torch.bfloat16),
        f"{prefix}.attn.kv_norm.weight": torch.arange(3, dtype=torch.bfloat16),
        f"{prefix}.attn.attn_sink": torch.arange(2, dtype=torch.float32),
        f"{prefix}.attn.wo_a.weight": torch.arange(64, dtype=torch.bfloat16).reshape(16, 4),
        f"{prefix}.attn.wo_b.weight": torch.arange(64, dtype=torch.int8).reshape(4, 16),
        f"{prefix}.attn.wo_b.scale": torch.arange(4, dtype=torch.float32),
        f"{prefix}.hc_ffn_fn": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        f"{prefix}.hc_ffn_scale": torch.arange(3, dtype=torch.float32),
        f"{prefix}.hc_ffn_base": torch.arange(1, dtype=torch.float32),
        f"{prefix}.ffn_norm.weight": torch.arange(4, dtype=torch.bfloat16),
        f"{prefix}.ffn.gate.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        f"{prefix}.ffn.gate.bias": torch.arange(4, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w1.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        f"{prefix}.ffn.shared_experts.w1.scale": torch.arange(2, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w2.weight": torch.arange(8, dtype=torch.int8).reshape(4, 2),
        f"{prefix}.ffn.shared_experts.w2.scale": torch.arange(4, dtype=torch.float32),
        f"{prefix}.ffn.shared_experts.w3.weight": torch.arange(8, dtype=torch.int8).reshape(2, 4),
        f"{prefix}.ffn.shared_experts.w3.scale": torch.arange(2, dtype=torch.float32),
        f"{prefix}.attn.compressor.wkv.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.compressor.wgate.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        f"{prefix}.attn.compressor.norm.weight": torch.arange(3, dtype=torch.bfloat16),
        f"{prefix}.attn.indexer.wq_b.weight": torch.arange(12, dtype=torch.int8).reshape(6, 2),
        f"{prefix}.attn.indexer.wq_b.scale": torch.arange(6, dtype=torch.float32),
        f"{prefix}.attn.indexer.weights_proj.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.wkv.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.wgate.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        f"{prefix}.attn.indexer.compressor.ape": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        f"{prefix}.attn.indexer.compressor.norm.weight": torch.arange(2, dtype=torch.bfloat16),
    }
    for expert_id in range(n_experts):
        base = expert_id * 10
        raw.update(
            {
                f"{prefix}.ffn.experts.{expert_id}.w1.weight": torch.full((2, 4), base, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w1.scale": torch.full((2,), base + 1, dtype=torch.float32),
                f"{prefix}.ffn.experts.{expert_id}.w2.weight": torch.full((4, 2), base + 2, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w2.scale": torch.full((4,), base + 3, dtype=torch.float32),
                f"{prefix}.ffn.experts.{expert_id}.w3.weight": torch.full((2, 4), base + 4, dtype=torch.int8),
                f"{prefix}.ffn.experts.{expert_id}.w3.scale": torch.full((2,), base + 5, dtype=torch.float32),
            }
        )
    return raw


class _Tokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = None

    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, token_ids: list[int]) -> str:
        return ""


def _runtime_model_for_embeddings():
    from pypto_serving.config.types import ModelConfig, RuntimeModel

    config = ModelConfig(
        model_id="dsv4",
        architecture="DeepseekV4ForCausalLM",
        vocab_size=6,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        torch_dtype="bfloat16",
    )
    runtime = RuntimeConfig(page_size=128, max_batch_size=1, max_seq_len=260, weight_dtype="int8")
    placeholder = torch.empty(0, config.hidden_size)
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=placeholder,
        final_norm_weight=torch.empty(0),
        lm_head=placeholder,
    )


def _runner_for_prepared_inputs(
    *,
    num_speculative_tokens: int = 0,
) -> tuple[DeepSeekV4ModelRunner, object]:
    model = _runtime_model_for_embeddings()
    compiled = DeepSeekV4CompiledKernels(
        # Exercise the production MTP tile while keeping the tiny host-side model.
        layout=DeepSeekV4CacheLayout(decode_batch=4, decode_seq=2, decode_tokens=8),
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=tuple([0] * 44),
        layer_plan=build_deepseek_v4_layer_plan(
            compress_ratios=tuple([0] * 44),
            num_hidden_layers=43,
            num_hash_layers=3,
        ),
        kernel_dir="",
        kernel_contract=_deepseek_serving_contract(),
        num_speculative_tokens=num_speculative_tokens,
        embedding_weight=torch.arange(128 * 4, dtype=torch.float32)
        .reshape(128, 4)
        .to(torch.bfloat16),
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    runner.init_kv_cache("dsv4", model.config, model.runtime)
    return runner, model


def test_deepseek_static_lm_head_weight_replicates_one_vocab_shard_per_dp_rank():
    layout = DeepSeekV4CacheLayout()
    tp_size = DEEPSEEK_V4_LM_HEAD_TP_SIZE
    # The regression only shows up when the DP world spans more than one TP group.
    assert layout.ranks > tp_size
    compiled = DeepSeekV4CompiledKernels(
        layout=layout,
        model_dir="",
        weight_map={},
        weight_store=None,
        compress_ratios=(),
        layer_plan=(),
        kernel_dir="",
        kernel_contract=_deepseek_serving_contract(prefill_tile_tokens=layout.prefill_seq),
    )
    runner = DeepSeekV4ModelRunner(compiled=compiled)
    vocab_per_rank = 2
    hidden_size = 3
    # Shard s carries the constant s + 1 so every rank's copy is identifiable.
    packed = torch.stack(
        [
            torch.full((vocab_per_rank, hidden_size), float(shard + 1), dtype=torch.bfloat16)
            for shard in range(tp_size)
        ]
    )
    runner._global_weights = weight_loader.DeepSeekV4GlobalWeights(
        embed_weight=torch.empty(0),
        final_norm_weight=torch.empty(0),
        lm_head_weight=packed,
        lm_head_layout=weight_loader.DeepSeekV4LmHeadLayout(
            ranks=tp_size,
            vocab_size=tp_size * vocab_per_rank,
            hidden_size=hidden_size,
            vocab_per_rank=vocab_per_rank,
            padded_vocab_per_rank=vocab_per_rank,
        ),
        hc_head_fn=torch.empty(0),
        hc_head_scale=torch.empty(0),
        hc_head_base=torch.empty(0),
    )

    replicated = runner._static_lm_head_weight_tensor()

    # Every card holds a full shard, not just the first TP group: ranks 4..7 must
    # repeat shards 0..3 instead of reading whatever the kernel maps past rank 3.
    assert replicated.shape == (layout.ranks, vocab_per_rank, hidden_size)
    for rank in range(layout.ranks):
        assert torch.equal(replicated[rank], packed[rank % tp_size]), f"rank {rank} shard mismatch"
    assert [float(replicated[rank, 0, 0]) for rank in range(layout.ranks)] == [
        1.0,
        2.0,
        3.0,
        4.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_map_shared_prepacked_tensors_roundtrips_every_dtype_and_outlives_the_fd(tmp_path):
    """Every sidecar dtype survives the mapping, including the reinterpreted ones.

    BF16/F16 have no numpy dtype, so they are read as same-width integers and
    reinterpreted; a wrong reinterpretation would silently corrupt weights. The
    descriptor is closed before the assertions because the mapping is what keeps
    the pages alive, not the fd.
    """
    from safetensors.torch import save_file

    expected = {
        "bf16": torch.tensor([[1.5, -2.25], [0.0, 7.0]], dtype=torch.bfloat16),
        "f16": torch.tensor([[0.5, -4.0], [1.0, 3.5]], dtype=torch.float16),
        "f32": torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.float32),
        "i32": torch.tensor([[7, -9], [0, 2**30]], dtype=torch.int32),
        "i8": torch.tensor([[3, -4], [127, -128]], dtype=torch.int8),
        "u8": torch.tensor([[200, 5], [0, 255]], dtype=torch.uint8),
    }
    sidecar = tmp_path / "packed.safetensors"
    save_file(expected, str(sidecar))

    fd = os.open(sidecar, os.O_RDONLY)
    try:
        mapped = weight_loader._map_shared_prepacked_tensors(fd, expected.keys())
    finally:
        os.close(fd)

    assert mapped.keys() == expected.keys()
    for name, tensor in expected.items():
        assert mapped[name].dtype == tensor.dtype, name
        assert torch.equal(mapped[name], tensor), name


def test_mapped_prepacked_tensor_outlives_its_siblings_and_intermediates(tmp_path):
    """One surviving tensor keeps its own mapping alive after everything else goes."""
    import gc

    from safetensors.torch import save_file

    sidecar = tmp_path / "packed.safetensors"
    save_file(
        {
            "keep": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "drop": torch.ones((2, 4), dtype=torch.float32),
        },
        str(sidecar),
    )
    fd = os.open(sidecar, os.O_RDONLY)
    try:
        mapped = weight_loader._map_shared_prepacked_tensors(fd, ("keep", "drop"))
    finally:
        os.close(fd)

    keep = mapped["keep"]
    del mapped
    gc.collect()

    assert torch.equal(keep, torch.arange(8, dtype=torch.float32).reshape(2, 4))


def test_prepacked_sidecar_replaced_after_validation_is_not_mapped(tmp_path, monkeypatch):
    """A concurrent publish between validation and mapping must not be mapped.

    The sidecar is published with an atomic ``os.replace``, so the name can point
    at a different inode by the time the payload is read. Validation and mapping
    share one descriptor, so the replacement is invisible to this load.
    """
    from safetensors.torch import save_file

    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    shard_path.write_bytes(b"source-checkpoint")
    store = DeepSeekV4WeightStore(
        model_dir=tmp_path,
        weight_map={"source.weight": shard_path.name},
    )
    params = {
        "ranks": 2,
        "n_routed_experts": 4,
        "compress_ratios": (4,),
        "num_hash_layers": 1,
    }
    metadata = {
        "format": DEEPSEEK_V4_PACKED_FORMAT,
        "source_fingerprint": store.packed_stacked_layer_weights_fingerprint(**params),
    }
    validated = {
        name: torch.zeros((2, 1), dtype=torch.float32)
        for name in weight_loader._DEEPSEEK_V4_PACKED_WEIGHT_NAMES
    }
    packed_path = deepseek_v4_packed_weights_path(tmp_path, ranks=2)
    save_file(validated, str(packed_path), metadata=metadata)

    # Same names, shapes and dtypes, so every structural check would still pass.
    usurper = tmp_path / "usurper.safetensors"
    save_file(
        {name: torch.ones((2, 1), dtype=torch.float32) for name in validated},
        str(usurper),
        metadata=metadata,
    )

    monkeypatch.setattr(weight_loader, "_sample_file_page_cache_residency", lambda fd, path: 1.0)
    real_mapper = weight_loader._map_shared_prepacked_tensors

    def publish_then_map(fd, names):
        os.replace(usurper, packed_path)
        return real_mapper(fd, names)

    monkeypatch.setattr(weight_loader, "_map_shared_prepacked_tensors", publish_then_map)

    packed = store.load_prepacked_stacked_layer_weights(**params)

    assert packed is not None
    for name, tensor in packed.tensors.items():
        assert torch.equal(tensor, validated[name]), f"{name} came from the replacement sidecar"


# ---------------------------------------------------------------------------
# Per-dispatch-class TaskArgs builders (contract: registration order ==
# kernel positional order; each arg declares its kind at registration).
# ---------------------------------------------------------------------------


class _TaskArgsStubRunner:
    """Minimal stand-in exposing every accessor the TaskArgs builders read."""

    def __init__(
        self,
        hidden: int = 4096,
        vocab: int = 129280,
        *,
        layout: DeepSeekV4CacheLayout | None = None,
        prefill_buffer_tokens: int | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        from pypto_serving.model.deepseek.task_args import (
            _PREFILL_CACHE_POOLS,
            _PREFILL_STATIC_WEIGHTS,
            _prefill_slot_specs,
        )

        self._compiled = SimpleNamespace(
            layout=layout or DeepSeekV4CacheLayout(),
            num_speculative_tokens=num_speculative_tokens,
        )
        layout = self._compiled.layout
        self._prefill_buffer_token_count = prefill_buffer_tokens or layout.prefill_seq
        covered = set(_PREFILL_STATIC_WEIGHTS) | set(_PREFILL_CACHE_POOLS)
        covered |= set(_prefill_slot_specs(layout, hidden, vocab))
        self._stacked = {
            n: torch.empty((2, 4), dtype=torch.bfloat16).share_memory_()
            for n in _PREFILL_FWD_TENSOR_ORDER
            if n not in covered
        }
        self._cache = {n: object() for n in _PREFILL_CACHE_POOLS}
        self._embed_handle = object()
        self._pre_hc_handle = object()
        self._decode_task_args = [
            SimpleNamespace(
                tensors={"block_table": torch.empty(layout.ranks, 4, dtype=torch.int32).share_memory_()}
            )
        ]

    def _require_stacked_weights(self):
        return SimpleNamespace(tensors=self._stacked)

    def _prefill_buffer_tokens(self):
        return self._prefill_buffer_token_count

    def _device_cache_values(self):
        return self._cache

    def _materialize_embedding_device_weight(self):
        return self._embed_handle

    def _materialize_main_pre_hc_device(self, hidden):
        return self._pre_hc_handle

    def _static_freqs_cos_tensor(self):
        return torch.empty((2, 8, 4), dtype=torch.bfloat16).share_memory_()

    def _static_freqs_sin_tensor(self):
        return torch.empty((2, 8, 4), dtype=torch.bfloat16).share_memory_()

    def _static_final_norm_weight_tensor(self):
        return torch.empty((2, 4096), dtype=torch.bfloat16).share_memory_()

    def _static_lm_head_weight_tensor(self):
        return torch.empty((2, 129280, 4096), dtype=torch.bfloat16).share_memory_()

    def _hc_head_tensors(self):
        return {
            "hc_head_fn": torch.empty((2, 4, 16384), dtype=torch.float32).share_memory_(),
            "hc_head_scale": torch.empty((2, 4), dtype=torch.float32).share_memory_(),
            "hc_head_base": torch.empty((2, 4), dtype=torch.float32).share_memory_(),
        }


def _task_args_alloc_worker():
    return SimpleNamespace(
        alloc_tensor=lambda shape, dtype, worker_id=0, init=None: DeviceTensor(
            0x1000 * (worker_id + 1), tuple(shape), dtype
        ),
        free_tensor=lambda _tensor, worker_id=0: None,
    )


def test_deepseek_task_args_builders_register_kernel_order():
    standalone_runner = _TaskArgsStubRunner(num_speculative_tokens=0)
    fused_runner = _TaskArgsStubRunner(num_speculative_tokens=1)
    deep_mtp_runner = _TaskArgsStubRunner(num_speculative_tokens=3)

    prefill = prefill_task_args(standalone_runner, 4096, 129280)
    assert isinstance(prefill, DeepSeekPrefillTaskArgs)
    assert prefill.names == _PREFILL_FWD_TENSOR_ORDER
    assert decode_task_args(standalone_runner, 4096, 129280).names == _DECODE_FWD_TENSOR_ORDER
    assert decode_task_args(deep_mtp_runner, 4096, 129280).names == _DECODE_FWD_TENSOR_ORDER
    assert decode_task_args(fused_runner, 4096, 129280).names == _DECODE_FWD_TENSOR_ORDER
    assert mtp_prefill_task_args(deep_mtp_runner, 4096).names == _MTP_PREFILL_TENSOR_ORDER
    assert mtp_decode_task_args(deep_mtp_runner, 4096).names == _MTP_DECODE_TENSOR_ORDER
    assert mtp_decode_task_args(fused_runner, 4096).names == _FUSED_MTP_DECODE_TENSOR_ORDER


def test_deepseek_prefill_task_args_classifies_kinds_and_builds():
    runner = _TaskArgsStubRunner()
    ta = prefill_task_args(runner, 4096, 129280)
    ta.allocate_host_shared(None)

    layout = runner._compiled.layout
    assert tuple(ta.tensors["x_hc"].shape)[1:] == (layout.prefill_seq, layout.hc_mult, 4096)
    assert ta.tensors["logits"].shape[1] == DEEPSEEK_V4_PREFILL_MAX_LOGIT_ROWS
    assert ta.tensors["input_ids"].dtype == torch.int64

    built = ta.build()
    assert len(built) == len(_PREFILL_FWD_TENSOR_ORDER)
    assert built[ta.names.index("x_hc")] is ta.tensors["x_hc"]
    assert isinstance(built[ta.names.index("freqs_cos")], StaticDeviceTensor)
    assert built[ta.names.index("kv_cache")] is runner._cache["kv_cache"]
    assert built[ta.names.index("wq_a")] is runner._stacked["wq_a"]


def test_deepseek_decode_task_args_owns_buffers_and_resolves_handles():
    runner = _TaskArgsStubRunner()
    ta = decode_task_args(runner, 4096, 129280)
    ta.allocate_host_shared(None)
    ta.allocate_device(_task_args_alloc_worker(), None)

    for name, dtype in (
        ("logits", torch.float32),
        ("hidden_out", torch.bfloat16),
        ("sampled_ids", torch.int32),
        ("block_table", torch.int32),
        ("sampling_temperatures", torch.float32),
        ("sampling_top_ks", torch.int32),
        ("sampling_seeds", torch.int32),
        ("sampling_positions", torch.int32),
    ):
        assert ta.tensors[name].dtype == dtype
    built = ta.build()
    assert built[ta.names.index("logits")] is ta.tensors["logits"]
    assert built[ta.names.index("pre_hc_hidden_out")] is ta.tensors["pre_hc_hidden_out"]
    assert isinstance(built[ta.names.index("pre_hc_hidden_out")], StackedDeviceTensor)
    assert isinstance(built[ta.names.index("freqs_cos")], StaticDeviceTensor)
    assert built[ta.names.index("kv_cache")] is runner._cache["kv_cache"]
    assert built[ta.names.index("embed_weight")] is runner._embed_handle


def test_deepseek_mtp_task_args_classify_kinds():
    def _mtp_runner(num_speculative_tokens: int, **overrides):
        weight = object()
        weights = {
            n: weight
            for n in (
                *_MTP_PREFILL_TENSOR_ORDER,
                *_MTP_DECODE_TENSOR_ORDER,
                *_FUSED_MTP_DECODE_TENSOR_ORDER,
            )
        }
        base = dict(
            _compiled=SimpleNamespace(layout=DeepSeekV4CacheLayout()),
            _mtp_device_weights=weights,
            _require_mtp_buffers=lambda: SimpleNamespace(weights={}),
            _materialize_mtp_device_kv_cache=lambda: object(),
            _materialize_mtp_tail_pre_hc_pool=lambda _hidden: object(),
            _materialize_mtp_device_state_tokens=lambda: object(),
            _materialize_mtp_device_state_meta=lambda: object(),
            _materialize_embedding_device_weight=lambda: object(),
            _materialize_main_pre_hc_device=lambda _hidden: object(),
            _static_freqs_cos_tensor=lambda: torch.empty(2, dtype=torch.float32).share_memory_(),
            _static_freqs_sin_tensor=lambda: torch.empty(2, dtype=torch.float32).share_memory_(),
            _static_lm_head_weight_tensor=lambda: torch.empty(2, dtype=torch.float32).share_memory_(),
        )
        base.update(overrides)
        stub = _TaskArgsStubRunner(num_speculative_tokens=num_speculative_tokens)
        for attr in (
            "_require_stacked_weights",
            "_device_cache_values",
            "_materialize_embedding_device_weight",
            "_materialize_main_pre_hc_device",
            "_static_freqs_cos_tensor",
            "_static_freqs_sin_tensor",
            "_static_final_norm_weight_tensor",
            "_static_lm_head_weight_tensor",
            "_hc_head_tensors",
            "_decode_task_args",
        ):
            base[attr] = getattr(stub, attr)
        base["_compiled"] = stub._compiled
        return SimpleNamespace(**base)

    prefill = mtp_prefill_task_args(_mtp_runner(3), 4096)
    prefill.allocate_host_shared(None)
    prefill.allocate_device(_task_args_alloc_worker(), None)
    assert prefill.tensors["input_ids"].dtype == torch.int64
    assert isinstance(prefill.tensors["hidden_out"], StackedDeviceTensor)  # device-resident output

    standalone = mtp_decode_task_args(_mtp_runner(3), 4096)
    standalone.allocate_host_shared(None)
    assert standalone.tensors["hidden_states"].dtype == torch.bfloat16
    assert standalone.tensors["prev_pre_hc_hidden"].dtype == torch.float32
    assert standalone.tensors["swa_slot_mapping"].dtype == torch.int64
    assert standalone.tensors["swa_indices"].shape[-1] == DeepSeekV4CacheLayout().sliding_window
    assert standalone.tensors["sampling_temperatures"].dtype == torch.float32
    assert standalone.tensors["sampling_top_ks"].dtype == torch.int32
    assert "accepted_counts" not in standalone.names
    built = standalone.build()
    assert isinstance(built[standalone.names.index("freqs_cos")], StaticDeviceTensor)

    fused = mtp_decode_task_args(_mtp_runner(1), 4096)
    fused.allocate_host_shared(None)
    assert fused.tensors["accepted_counts"].dtype == torch.int32
    assert fused.names[fused.names.index("state_tokens") - 1] == "state_generations"


def test_deepseek_fused_mtp_write_only_outputs_are_device_resident():
    class _AllocWorker:
        def __init__(self):
            self.next_ptr = 0x1000

        def alloc_tensor(self, shape, dtype, *, worker_id=0, init=None):
            tensor = DeviceTensor(self.next_ptr, tuple(shape), dtype)
            self.next_ptr += 0x100000
            return tensor

        @staticmethod
        def free_tensor(_tensor, *, worker_id=0):
            return None

    runner = _TaskArgsStubRunner()
    runner._compiled.num_speculative_tokens = 1
    worker = _AllocWorker()

    decode = decode_task_args(runner, 4096, 129280)
    decode.allocate_host_shared(None)
    decode.allocate_device(worker, None)
    for name in (
        "sampling_temperatures",
        "sampling_top_ks",
        "sampling_seeds",
        "sampling_positions",
        "hidden_out",
        "logits",
        "pre_hc_hidden_out",
    ):
        assert isinstance(decode.tensors[name], StackedDeviceTensor)
    assert isinstance(decode.tensors["sampled_ids"], torch.Tensor)
    for name in (
        "block_table",
        "hca_cmp_block_table",
        "csa_cmp_block_table",
        "idx_block_table",
        "hca_compress_state_block_table",
        "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "block_counts",
    ):
        assert isinstance(decode.tensors[name], torch.Tensor)

    mtp = mtp_decode_task_args(runner, 4096)
    mtp.allocate_host_shared(None)
    mtp.allocate_device(worker, None)
    for name in (
        "input_ids",
        "position_ids",
        "sampling_temperatures",
        "sampling_top_ks",
        "sampling_seeds",
        "sampling_positions",
        "hidden_out",
        "next_pre_hc_hidden",
        "logits",
    ):
        assert isinstance(mtp.tensors[name], StackedDeviceTensor)
    assert isinstance(mtp.tensors["accepted_counts"], torch.Tensor)
    assert isinstance(mtp.tensors["sampled_ids"], torch.Tensor)

    block_tables = [object(), object()]
    runner._decode_task_args = [
        SimpleNamespace(tensors={"block_table": block_table})
        for block_table in block_tables
    ]
    runner._mtp_device_weights = {
        name: object() for name in _FUSED_MTP_DECODE_TENSOR_ORDER
    }
    runner._materialize_mtp_device_kv_cache = lambda: object()
    runner._materialize_mtp_tail_pre_hc_pool = lambda _hidden: object()
    runner._materialize_mtp_device_state_tokens = lambda: object()
    runner._materialize_mtp_device_state_meta = lambda: object()

    slot_one_mtp = mtp_decode_task_args(runner, 4096, buffer_slot=1)
    slot_one_mtp.allocate_host_shared(None)
    slot_one_mtp.allocate_device(worker, None)
    slot_one_args = slot_one_mtp.build()
    assert (
        slot_one_args[slot_one_mtp.names.index("ori_block_table")]
        is block_tables[1]
    )
