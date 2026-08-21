# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from simpler.task_interface import DataType

from pypto_serving.config.types import (
    DecodeBatch,
    ModelConfig,
    PrefillBatch,
    RuntimeConfig,
    RuntimeModel,
)
from pypto_serving.model.qwen.npu_runner import (
    _CompiledKernels,
    _DecodeKernelInputs,
    _L3Callable,
    Qwen314BModelRunner as ModelRunner,
    QwenLayout,
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.worker.worker import WorkerTensor


ROOT = Path(__file__).resolve().parents[1]
QWEN3_KERNEL_DIR = ROOT / "pypto-lib" / "models" / "qwen3" / "14b"


def _model(
    max_batch_size: int,
    max_seq_len: int = 128,
    page_size: int = 64,
    eos_token_id: int | None = None,
    max_num_batched_tokens: int = 4096,
) -> RuntimeModel:
    config = ModelConfig(
        model_id="test-model",
        architecture="qwen3",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=max_seq_len,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=None,
        eos_token_id=eos_token_id,
        pad_token_id=None,
        torch_dtype="bfloat16",
    )
    runtime = RuntimeConfig(
        page_size=page_size,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device="cpu",
        max_num_batched_tokens=max_num_batched_tokens,
    )
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=torch.zeros(config.vocab_size, config.hidden_size),
        final_norm_weight=torch.ones(config.hidden_size),
        lm_head=torch.zeros(config.vocab_size, config.hidden_size),
    )


def _compiled_kernels(
    model: RuntimeModel,
    *,
    callable_: _L3Callable | None = None,
    decode_weights: dict[str, torch.Tensor] | None = None,
) -> _CompiledKernels:
    kernel_batch = model.runtime.max_batch_size
    sampled_ids_width = 8
    max_seq = model.runtime.max_seq_len
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    head_dim = model.config.head_dim
    kv_hidden = model.config.num_key_value_heads * head_dim
    max_blocks = (max_seq + model.runtime.page_size - 1) // model.runtime.page_size
    if callable_ is None:
        callable_ = _L3Callable(
            compiled=object(),
            name="fake",
            aicpu_thread_num=1,
        )
    if decode_weights is None:
        decode_weights = {
            "decode_input_rms_weight": torch.ones(1, hidden_size),
            "decode_wq": torch.zeros(hidden_size, hidden_size),
            "decode_wk": torch.zeros(hidden_size, kv_hidden),
            "decode_wv": torch.zeros(hidden_size, kv_hidden),
            "decode_q_norm_weight": torch.ones(1, head_dim),
            "decode_k_norm_weight": torch.ones(1, head_dim),
            "decode_wo": torch.zeros(hidden_size, hidden_size),
            "decode_post_rms_weight": torch.ones(1, hidden_size),
            "decode_w_gate": torch.zeros(hidden_size, intermediate_size),
            "decode_w_up": torch.zeros(hidden_size, intermediate_size),
            "decode_w_down": torch.zeros(intermediate_size, hidden_size),
        }
    return _CompiledKernels(
        prefill=callable_,
        decode=callable_,
        topk_select=callable_,
        final_norm_weight=torch.ones(1, hidden_size),
        rope_cos=torch.zeros(max_seq, head_dim),
        rope_sin=torch.zeros(max_seq, head_dim),
        padded_vocab=model.config.vocab_size,
        padded_lm_head_weight=torch.zeros(model.config.vocab_size, hidden_size),
        padded_embed_weight=torch.zeros(model.config.vocab_size, hidden_size),
        decode_weights=decode_weights,
        layout=QwenLayout(
            kernel_batch=kernel_batch,
            max_seq_len=max_seq,
            page_size=model.runtime.page_size,
            max_blocks_per_seq=max_blocks,
            padded_vocab=model.config.vocab_size,
            hidden_size=hidden_size,
            sampled_ids_width=sampled_ids_width,
            topk_width=4,
        ),
    )


def test_kv_cache_capacity_uses_actual_runtime_batch_size():
    model = _model(max_batch_size=1, max_seq_len=128, page_size=64)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)

    k_cache, _ = manager.materialize_single_layer_cache(model.config.model_id, 0)
    assert k_cache.shape[0] == 1 * 2 * model.config.num_key_value_heads * model.runtime.page_size


def test_prefill_inputs_pack_actual_tokens_into_fixed_kernel_buffers():
    model = _model(max_batch_size=15)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    compiled = _compiled_kernels(model)
    runner = ModelRunner(
        compiled=compiled,
    )
    allocations = [
        manager.allocate_for_prompt(model.config.model_id, f"req-{idx}", idx + 1) for idx in range(2)
    ]
    seq_lens = [idx + 1 for idx in range(len(allocations))]
    prepared = runner._prepare_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=[alloc.request_id for alloc in allocations],
            token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            input_embeddings=None,
            seq_lens=seq_lens,
            chunk_lens=[1, 2],
            chunk_offsets=[0, 1],
            chunk_starts=[0, 0],
            kv_allocations=allocations,
        ),
    )

    prefill = runner._prefill_task_args.tensors
    assert prepared.actual_batch == 2
    assert prepared.token_ids.shape == (3,)
    assert prepared.token_ids.tolist() == [1, 2, 3]
    assert prefill["seq_lens"].shape == (model.runtime.max_batch_size,)
    assert prefill["seq_lens"][:2].tolist() == [1, 2]
    assert prefill["seq_lens"][2:].tolist() == [0] * (model.runtime.max_batch_size - 2)
    assert prefill["chunk_lens"][:2].tolist() == [1, 2]
    assert prefill["chunk_lens"][2:].tolist() == [0] * (model.runtime.max_batch_size - 2)
    assert prefill["chunk_offsets"][:2].tolist() == [0, 1]
    assert prefill["chunk_offsets"][2:].tolist() == [0] * (model.runtime.max_batch_size - 2)
    assert prefill["block_table"].shape == (model.runtime.max_batch_size * 2,)
    assert prefill["block_table"][0].item() == allocations[0].page_ids[0]
    assert prefill["block_table"][4:].tolist() == [-1] * (prefill["block_table"].numel() - 4)
    assert prepared.slot_mapping.shape == (3,)
    assert prepared.slot_mapping.data_ptr() == prefill["slot_mapping"].data_ptr()
    assert prepared.slot_mapping[2].item() == manager.slot_mapping_for_request(allocations[1], 1)


def test_prefill_inputs_pack_resumed_chunk_metadata():
    model = _model(max_batch_size=1, max_seq_len=8, page_size=2)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    runner = ModelRunner(
        compiled=_compiled_kernels(model),
    )
    alloc = manager.allocate_for_prompt(model.config.model_id, "req-0", 4)

    prepared = runner._prepare_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=[alloc.request_id],
            token_ids=torch.tensor([5, 6], dtype=torch.long),
            input_embeddings=None,
            seq_lens=[4],
            chunk_lens=[2],
            chunk_offsets=[0],
            chunk_starts=[2],
            kv_allocations=[alloc],
        ),
    )

    prefill = runner._prefill_task_args.tensors
    assert prepared.token_ids.tolist() == [5, 6]
    assert prefill["seq_lens"].tolist() == [4]
    assert prefill["chunk_lens"].tolist() == [2]
    assert prefill["chunk_offsets"].tolist() == [0]
    assert prepared.slot_mapping.tolist() == [
        manager.slot_mapping_for_request(alloc, 2),
        manager.slot_mapping_for_request(alloc, 3),
    ]


def test_write_slot_mapping_rejects_insufficient_pages():
    target = torch.empty(2, dtype=torch.int32).numpy()
    with pytest.raises(ValueError, match="too small"):
        ModelRunner._write_slot_mapping(target, [0], 2, 2, start_pos=1)


def test_write_slot_mapping_fills_target_slice_across_physical_pages():
    target = torch.full((6,), -1, dtype=torch.int32).numpy()

    ModelRunner._write_slot_mapping(target[1:5], [7, 3], 4, 2)

    assert target.tolist() == [-1, 14, 15, 6, 7, -1]


def test_prepare_decode_inputs_writes_compiled_buffers_and_replicates_padding():
    model = _model(max_batch_size=2)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    compiled = _compiled_kernels(model)
    runner = ModelRunner(compiled=compiled)
    alloc = manager.allocate_for_prompt(model.config.model_id, "req-0", 1)
    prepared = runner._prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=[alloc.request_id],
            token_ids=torch.tensor([[7]], dtype=torch.long),
            hidden_states=None,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            kv_allocations=[alloc],
        ),
    )

    decode = runner._decode_task_args.tensors
    assert prepared.actual_batch == 1
    assert prepared.logits is decode["logits"]
    assert decode["token_ids"][:, :1].tolist() == [[7], [7]]
    assert torch.count_nonzero(decode["token_ids"][:, 1:]).item() == 0
    assert decode["seq_lens"].tolist() == [1, 1]
    assert decode["block_table"].reshape(2, 2).tolist() == [
        [alloc.page_ids[0], -1],
        [alloc.page_ids[0], -1],
    ]
    expected_slot = manager.slot_mapping_for_request(alloc)
    assert decode["slot_mapping"].tolist() == [expected_slot, expected_slot]


def test_prepare_decode_inputs_caches_block_table_until_pages_change():
    model = _model(max_batch_size=1)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    runner = ModelRunner(compiled=_compiled_kernels(model))
    alloc = manager.allocate_for_prompt(model.config.model_id, "req-0", 1)

    def prepare(seq_len: int):
        return runner._prepare_decode_inputs(
            model,
            DecodeBatch(
                request_ids=[alloc.request_id],
                token_ids=torch.tensor([[7]], dtype=torch.long),
                hidden_states=None,
                seq_lens=torch.tensor([seq_len], dtype=torch.int32),
                kv_allocations=[alloc],
            ),
        )

    prepared = prepare(1)
    cached_pages = runner._decode_block_table_row_pages[0]
    prepare(2)
    assert runner._decode_block_table_row_pages[0] is cached_pages

    alloc.tokens_used = alloc.tokens_capacity
    manager.ensure_one_more_slot(alloc)
    prepared = prepare(model.runtime.page_size + 1)
    assert runner._decode_block_table_row_pages[0] is not cached_pages
    decode = runner._decode_task_args.tensors
    assert decode["block_table"].tolist() == alloc.page_ids


def test_decode_topk_selects_from_device_resident_logits(monkeypatch):
    model = _model(max_batch_size=1)
    compiled = _compiled_kernels(model)
    compiled.decode = _L3Callable(
        compiled=object(),
        name="decode",
        aicpu_thread_num=1,
    )
    compiled.topk_select = _L3Callable(
        compiled=object(),
        name="topk_select",
        aicpu_thread_num=1,
    )
    runner = ModelRunner(compiled=compiled)
    host_logits = torch.zeros(1, model.config.vocab_size)
    kernel_inputs = _DecodeKernelInputs(
        actual_batch=1,
        logits=host_logits,
    )
    runner._kv_caches = {
        model.config.model_id: SimpleNamespace(
            key_pages=object(),
            value_pages=object(),
        )
    }
    device_logits = object()
    device_next_hidden = object()
    monkeypatch.setattr(
        runner,
        "_prepare_decode_inputs",
        lambda _model, _batch: kernel_inputs,
    )
    monkeypatch.setattr(runner, "_decode_logits_device_arg", lambda: device_logits)
    monkeypatch.setattr(
        runner,
        "_decode_next_hidden_device_arg",
        lambda: device_next_hidden,
    )
    dispatches = []
    monkeypatch.setattr(
        runner,
        "_run_l3",
        lambda callable_spec, *args: dispatches.append((callable_spec, args)),
    )

    result = runner.run_decode(
        model,
        DecodeBatch(
            request_ids=["request"],
            token_ids=torch.tensor([[7]], dtype=torch.long),
            hidden_states=None,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            allow_device_topk_sampling=True,
            block_ids=[[0]],
        ),
    )

    assert result.logits is None
    assert result.sampling_candidates is not None
    assert dispatches[0][0] is compiled.decode
    assert dispatches[0][1][20] is device_logits
    assert dispatches[0][1][-1] is device_next_hidden
    assert dispatches[1][0] is compiled.topk_select
    assert dispatches[1][1][0] is device_logits


def test_decode_kernel_inputs_reject_multi_token_rows():
    model = _model(max_batch_size=2)
    runner = ModelRunner(compiled=_compiled_kernels(model))

    with pytest.raises(ValueError, match="exactly one token per row"):
        runner._prepare_decode_inputs(
            model,
            DecodeBatch(
                request_ids=["req-0"],
                token_ids=torch.tensor([[3, 4]], dtype=torch.int32),
                hidden_states=None,
                seq_lens=torch.tensor([1], dtype=torch.int32),
                block_ids=[[0]],
            ),
        )


def test_compute_kv_cache_pages_takes_max_of_peak_and_simpler_committed(monkeypatch):
    """non_kv = max(peak_non_kv, simpler_committed); kv_budget = total*util - non_kv.

    The driver-visible peak and simpler's committed total overlap (both cover
    weights + arenas), so the larger one must win to avoid over-provisioning.
    ``simpler_committed=0`` (worker/API unknown) must reproduce peak-only sizing.
    """
    model = _model(max_batch_size=1, page_size=64)
    config, runtime = model.config, model.runtime

    total = 10_000_000_000
    free = 6_000_000_000  # peak_non_kv = total - free = 4 GB
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda device: (free, total))

    bytes_per_page = (
        config.num_hidden_layers * 2 * config.num_key_value_heads
        * runtime.page_size * config.head_dim * getattr(torch, runtime.kv_dtype).itemsize
    )
    util = runtime.npu_memory_utilization

    def expected(non_kv):
        return max(int(total * util - non_kv) // bytes_per_page, 1)

    # simpler_committed (5 GB) > peak_non_kv (4 GB): the committed view wins.
    pages_committed_dominates = ModelRunner._compute_kv_cache_pages(
        config, runtime, device_id=0, simpler_committed=5_000_000_000,
    )
    assert pages_committed_dominates == expected(5_000_000_000)

    # simpler_committed (2 GB) < peak_non_kv (4 GB): the driver peak wins.
    pages_peak_dominates = ModelRunner._compute_kv_cache_pages(
        config, runtime, device_id=0, simpler_committed=2_000_000_000,
    )
    assert pages_peak_dominates == expected(4_000_000_000)

    # Unknown committed (0) reproduces the pre-commit peak-only behaviour.
    assert ModelRunner._compute_kv_cache_pages(
        config, runtime, device_id=0, simpler_committed=0,
    ) == pages_peak_dominates

    # A larger non-KV footprint leaves fewer KV pages.
    assert pages_committed_dominates < pages_peak_dominates


@pytest.mark.parametrize("device_id, device_ids, expected_chip", [
    (5, (2, 5, 7), 1),   # device 5 -> chip index 1
    (7, (2, 5, 7), 2),   # device 7 -> chip index 2
    (99, (2, 5, 7), 0),  # device absent from the list -> chip 0
])
def test_query_simpler_committed_queries_current_device_chip(device_id, device_ids, expected_chip):
    committed = {0: 1_000_000_000, 1: 2_000_000_000, 2: 3_000_000_000}
    runner = ModelRunner(compiled=None, device_id=device_id)
    runner._l3_worker = _CommittedFakeWorker(committed, device_ids)

    assert runner._query_simpler_committed() == committed[expected_chip]
    assert runner._l3_worker.queried_chip == expected_chip


def test_query_simpler_committed_returns_zero_when_worker_unavailable():
    runner = ModelRunner(compiled=None, device_id=0)
    runner._l3_worker = None
    assert runner._query_simpler_committed() == 0


def test_query_simpler_committed_returns_zero_when_worker_raises():
    runner = ModelRunner(compiled=None, device_id=0)
    runner._l3_worker = _CommittedFakeWorker({}, device_ids=(0,), raises=True)
    assert runner._query_simpler_committed() == 0


class _FakeWorker:
    _DTYPES = {
        torch.float32: DataType.FLOAT32,
        torch.bfloat16: DataType.BFLOAT16,
        torch.int32: DataType.INT32,
    }

    def __init__(self) -> None:
        self._next_ptr = 1
        self.initialized = True

    def alloc_tensor(self, shape, dtype, init=None):
        nbytes = torch.empty(tuple(shape), dtype=dtype).nbytes
        tensor = WorkerTensor(self._next_ptr, tuple(shape), self._DTYPES[dtype])
        self._next_ptr += nbytes
        return tensor

    def free_tensor(self, tensor):
        return None

    def run(self, compiled, *args, **kwargs):
        return None


class _CommittedFakeWorker:
    """Stand-in for simpler's L3 Worker exposing only committed_device_memory."""

    def __init__(self, committed_by_chip, device_ids, *, raises=False):
        self._committed = committed_by_chip
        self.device_ids = device_ids
        self.raises = raises
        self.queried_chip = None

    def committed_device_memory(self, chip):
        self.queried_chip = chip
        if self.raises:
            raise RuntimeError("worker not ready")
        return self._committed[chip]
