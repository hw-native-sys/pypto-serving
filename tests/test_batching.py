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

from python.core.engine import LLMEngine
from python.core.executor import ModelExecutor
from python.core.kv_cache import KvCacheManager
from python.core.types import (
    DecodeBatch,
    DecodeResult,
    GenerateConfig,
    LayerWeights,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from python.core.scheduler import Request, ScheduledRequest
from python.core.serving_worker import WorkerProcess
from examples.model.qwen3_14b.runner.npu_executor import Qwen314BPyptoExecutor as PyptoExecutor
from examples.model.qwen3_14b.runner.npu_runner import Qwen314BModelRunner as ModelRunner
from examples.model.qwen3_14b.runner.npu_runner import _CompiledKernels
from examples.model.qwen3_14b.runner.npu_runner import _L3Callable
from examples.model.qwen3_14b.runner.npu_runner import _add_run_timing_args
from examples.model.qwen3_14b.runner.npu_runner import _kernel_trace_name
from examples.model.qwen3_14b.runner.npu_runner import _run_timing_us
from python.runtime.worker import WorkerTensor


ROOT = Path(__file__).resolve().parents[1]
QWEN3_DISPATCH = ROOT / "examples" / "model" / "qwen3_14b" / "runner" / "qwen3_l3_dispatch.py"
QWEN3_KERNEL_DIR = ROOT / "pypto-lib" / "models" / "qwen3" / "14b"


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [max(1, len(text))]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _model(
    max_batch_size: int,
    max_seq_len: int = 128,
    page_size: int = 64,
    eos_token_id: int | None = None,
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
    )
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=torch.zeros(config.vocab_size, config.hidden_size),
        final_norm_weight=torch.ones(config.hidden_size),
        lm_head=torch.zeros(config.vocab_size, config.hidden_size),
        layers=[],
    )


def _compiled_kernels(
    model: RuntimeModel,
    *,
    callable_: _L3Callable | None = None,
    decode_weights: dict[str, torch.Tensor] | None = None,
) -> _CompiledKernels:
    kernel_batch = model.runtime.max_batch_size
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
            block_dim=1,
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
        greedy_sample=callable_,
        token_embed=callable_,
        final_norm_weight=torch.ones(1, hidden_size),
        rope_cos=torch.zeros(max_seq, head_dim),
        rope_sin=torch.zeros(max_seq, head_dim),
        padded_vocab=model.config.vocab_size,
        padded_lm_head_weight=torch.zeros(model.config.vocab_size, hidden_size),
        padded_embed_weight=torch.zeros(model.config.vocab_size, hidden_size),
        decode_weights=decode_weights,
        prefill_hidden_buffer=torch.empty(kernel_batch * max_seq, hidden_size, dtype=torch.bfloat16),
        prefill_seq_lens_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_chunk_lens_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_chunk_offsets_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_block_table_buffer=torch.empty(kernel_batch * max_blocks, dtype=torch.int32),
        prefill_slot_mapping_buffer=torch.empty(kernel_batch * max_seq, dtype=torch.int32),
        prefill_logits_buffer=torch.empty(kernel_batch, model.config.vocab_size),
        prefill_sampled_ids_buffer=torch.empty(kernel_batch, 1, dtype=torch.int32),
        prefill_next_hidden_buffer=torch.empty(kernel_batch, hidden_size, dtype=torch.bfloat16),
        decode_hidden_buffer=torch.zeros(kernel_batch, hidden_size, dtype=torch.bfloat16),
        decode_seq_lens_buffer=torch.zeros(kernel_batch, dtype=torch.int32),
        decode_block_table_buffer=torch.zeros(kernel_batch * max_blocks, dtype=torch.int32),
        decode_slot_mapping_buffer=torch.zeros(kernel_batch, dtype=torch.int32),
        decode_logits_buffer=torch.zeros(kernel_batch, model.config.vocab_size),
        decode_token_ids_buffer=torch.empty(kernel_batch, 1, dtype=torch.int32),
        decode_sampled_ids_buffer=torch.empty(kernel_batch, 1, dtype=torch.int32),
        decode_next_hidden_buffer=torch.empty(kernel_batch, hidden_size, dtype=torch.bfloat16),
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
    runner = ModelRunner(
        compiled=_compiled_kernels(model),
    )
    allocations = [
        manager.allocate_for_prompt(model.config.model_id, f"req-{idx}", idx + 1)
        for idx in range(2)
    ]
    seq_lens = torch.tensor(
        [idx + 1 for idx in range(len(allocations))],
        dtype=torch.int32,
    )
    embeddings = torch.ones(
        len(allocations),
        int(seq_lens.max().item()),
        model.config.hidden_size,
    )

    prepared = runner._prepare_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=[alloc.request_id for alloc in allocations],
            token_ids=torch.zeros(
                len(allocations),
                int(seq_lens.max().item()),
                dtype=torch.long,
            ),
            input_embeddings=embeddings,
            seq_lens=seq_lens,
            kv_allocations=allocations,
        ),
    )

    assert prepared.actual_batch == 2
    assert prepared.hidden.shape == (3, model.config.hidden_size)
    assert prepared.seq_lens.shape == (model.runtime.max_batch_size,)
    assert prepared.seq_lens[:2].tolist() == [1, 2]
    assert prepared.seq_lens[2:].tolist() == [0] * (model.runtime.max_batch_size - 2)
    assert prepared.chunk_lens[:2].tolist() == [1, 2]
    assert prepared.chunk_lens[2:].tolist() == [0] * (model.runtime.max_batch_size - 2)
    assert prepared.chunk_offsets[:2].tolist() == [0, 1]
    assert prepared.chunk_offsets[2:].tolist() == [0] * (model.runtime.max_batch_size - 2)
    assert prepared.block_table.shape == (model.runtime.max_batch_size * 2,)
    assert prepared.block_table[0].item() == allocations[0].page_ids[0]
    assert prepared.block_table[4:].tolist() == [-1] * (prepared.block_table.numel() - 4)
    assert prepared.slot_mapping.shape == (3,)
    assert prepared.slot_mapping[2].item() == manager.slot_mapping_for_request(allocations[1], 1)


def test_prefill_inputs_pack_resumed_chunk_positions():
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
            token_ids=torch.zeros(1, 2, dtype=torch.long),
            input_embeddings=torch.ones(1, 2, model.config.hidden_size),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            kv_allocations=[alloc],
            positions=torch.tensor([[2, 3]], dtype=torch.long),
        ),
    )

    assert prepared.hidden.shape == (2, model.config.hidden_size)
    assert prepared.seq_lens.tolist() == [4]
    assert prepared.chunk_lens.tolist() == [2]
    assert prepared.chunk_offsets.tolist() == [0]
    assert prepared.slot_mapping.tolist() == [
        manager.slot_mapping_for_request(alloc, 2),
        manager.slot_mapping_for_request(alloc, 3),
    ]


def test_prefill_inputs_reject_non_contiguous_chunk_positions():
    model = _model(max_batch_size=1, max_seq_len=8, page_size=2)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    runner = ModelRunner(
        compiled=None,  # type: ignore[arg-type]
    )
    alloc = manager.allocate_for_prompt(model.config.model_id, "req-0", 4)

    with pytest.raises(ValueError, match="contiguous chunk"):
        runner._prepare_prefill_inputs(
            model,
            PrefillBatch(
                request_ids=[alloc.request_id],
                token_ids=torch.zeros(1, 3, dtype=torch.long),
                input_embeddings=torch.ones(1, 3, model.config.hidden_size),
                seq_lens=torch.tensor([4], dtype=torch.int32),
                kv_allocations=[alloc],
                positions=torch.tensor([[1, 3, 4]], dtype=torch.long),
            ),
        )


def test_compute_slot_mapping_rejects_insufficient_pages():
    with pytest.raises(ValueError, match="too small"):
        ModelRunner._compute_slot_mapping([0], 2, 2, start_pos=1)


def test_decode_inputs_use_actual_user_batch_without_padding_lanes():
    model = _model(max_batch_size=1)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    runner = ModelRunner(
        compiled=None,  # type: ignore[arg-type]
    )
    alloc = manager.allocate_for_prompt(model.config.model_id, "req-0", 1)
    hidden_states = torch.ones(1, model.config.hidden_size)

    prepared = runner._prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=[alloc.request_id],
            token_ids=torch.zeros(1, 1, dtype=torch.long),
            hidden_states=hidden_states,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            kv_allocations=[alloc],
        ),
    )

    assert prepared.actual_batch == 1
    assert prepared.hidden.shape == (1, model.config.hidden_size)
    assert prepared.seq_lens.tolist() == [1]
    assert prepared.block_table.shape == (2,)
    assert prepared.block_table[0].item() == alloc.page_ids[0]
    assert prepared.slot_mapping.tolist() == [manager.slot_mapping_for_request(alloc)]


def test_decode_kernel_inputs_reject_multi_token_rows():
    model = _model(max_batch_size=2)
    runner = ModelRunner(compiled=_compiled_kernels(model))

    with pytest.raises(ValueError, match="exactly one token per row"):
        runner._pad_decode_inputs(
            model,
            SimpleNamespace(
                actual_batch=1,
                token_ids=torch.tensor([[3, 4]], dtype=torch.int32),
                hidden=torch.ones(1, model.config.hidden_size, dtype=torch.bfloat16),
                seq_lens=torch.tensor([1], dtype=torch.int32),
                block_table=torch.zeros(2, dtype=torch.int32),
                slot_mapping=torch.zeros(1, dtype=torch.int32),
            ),
        )


def test_engine_generate_batch_uses_batched_executor_results():
    model = _model(max_batch_size=2, eos_token_id=0)
    manager = KvCacheManager()
    executor = _ImmediateEosExecutor(manager)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    results = engine.generate_batch(
        model.config.model_id,
        ["a", "abcd"],
        GenerateConfig(max_new_tokens=2, temperature=0.0),
    )

    assert [result.token_ids for result in results] == [[0], [0]]
    assert [result.finish_reason for result in results] == ["eos", "eos"]


def test_engine_uses_device_sampled_prefill_token_when_available():
    model = _model(max_batch_size=1, eos_token_id=0)
    model.embed_tokens = torch.arange(model.config.vocab_size * model.config.hidden_size, dtype=torch.float32).view(
        model.config.vocab_size,
        model.config.hidden_size,
    )
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(manager, first_token=3, second_token=0)
    sampler = _FailingSampler()
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=1, temperature=0.0),
    )[0]

    assert result.token_ids == [3]
    assert executor.prefill_calls == 1
    assert executor.decode_calls == 0
    assert sampler.sample_calls == 0


def test_engine_uses_zero_decode_placeholder_when_executor_embeds_on_device():
    model = _model(max_batch_size=1, eos_token_id=0)
    model.embed_tokens = torch.arange(model.config.vocab_size * model.config.hidden_size, dtype=torch.float32).view(
        model.config.vocab_size,
        model.config.hidden_size,
    )
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(manager, first_token=3, second_token=0)
    sampler = _FailingSampler()
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=2, temperature=0.0),
    )[0]

    assert result.token_ids == [3, 0]
    assert executor.lookup_calls == 1
    assert executor.decode_calls == 1
    assert torch.equal(executor.decode_hidden_seen[0], torch.zeros_like(model.embed_tokens[3]))
    assert sampler.sample_calls == 0


def test_engine_skips_decode_host_embedding_when_executor_embeds_on_device():
    model = _model(max_batch_size=1, eos_token_id=0)
    model.embed_tokens = torch.arange(model.config.vocab_size * model.config.hidden_size, dtype=torch.float32).view(
        model.config.vocab_size,
        model.config.hidden_size,
    )
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(
        manager,
        first_token=3,
        second_token=0,
        return_next_hidden=False,
    )
    sampler = _FailingSampler()
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=2, temperature=0.0),
    )[0]

    assert result.token_ids == [3, 0]
    assert executor.lookup_calls == 1
    assert executor.decode_calls == 1
    assert torch.equal(executor.decode_hidden_seen[0], torch.zeros_like(model.embed_tokens[3]))
    assert sampler.sample_calls == 0


def test_engine_ignores_device_sampled_tokens_for_non_greedy_config():
    model = _model(max_batch_size=1)
    model.embed_tokens = torch.arange(model.config.vocab_size * model.config.hidden_size, dtype=torch.float32).view(
        model.config.vocab_size,
        model.config.hidden_size,
    )
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(manager, first_token=3, second_token=0)
    sampler = _FixedSampler(token_id=7)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=1, temperature=0.8),
    )[0]

    assert result.token_ids == [7]
    assert executor.prefill_calls == 1
    assert executor.decode_calls == 0
    assert sampler.sample_calls == 1


def test_serving_worker_skips_decode_host_embedding_when_executor_embeds_on_device():
    model = _model(max_batch_size=1, eos_token_id=0)
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(
        manager,
        first_token=3,
        second_token=0,
        return_next_hidden=False,
    )

    def fail_lookup(model, token_ids):
        raise AssertionError("serving worker decode should let the device kernel embed token ids")

    executor.lookup_embeddings = fail_lookup
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = executor
    worker.sampler = _FailingSampler()
    worker.model_record = SimpleNamespace(config=model.config)

    request = Request(
        request_id="decode",
        prompt_token_ids=[1],
        max_new_tokens=2,
        temperature=0.0,
    )
    request.output_token_ids.append(3)
    scheduled = ScheduledRequest(
        request=request,
        num_new_tokens=1,
        is_prefill=False,
        block_ids=[0],
    )
    new_tokens: dict[str, int] = {}

    worker._batch_decode([scheduled], model, new_tokens)

    assert new_tokens == {"decode": 0}
    assert executor.decode_calls == 1
    assert torch.equal(executor.decode_hidden_seen[0], torch.zeros(model.config.hidden_size))


def test_pypto_executor_uses_cached_kernel_weights_after_registration(monkeypatch):
    model = _model(max_batch_size=1, page_size=256)
    model.layers = [_layer(model.config.hidden_size, model.config.intermediate_size, model.config.head_dim)]
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    executor = PyptoExecutor(manager)
    cached_layer = executor._kernel_layer_weights(model.layers[0])
    fake_kernel = _CopyKernel()
    fake_callable = _L3Callable(
        compiled=fake_kernel,
        name="fake",
        block_dim=1,
        aicpu_thread_num=1,
    )
    compiled = _compiled_kernels(
        model,
        callable_=fake_callable,
        decode_weights=executor._stack_decode_weights([cached_layer]),
    )
    executor._compiled[model.config.model_id] = compiled
    runner = ModelRunner(
        compiled=compiled,
    )
    monkeypatch.setattr(runner, "_shared_l3_worker", lambda: _FakeWorker())
    monkeypatch.setattr(runner, "_compute_kv_cache_pages", lambda config, runtime: 1)
    monkeypatch.setattr(runner, "_print_memory_breakdown", lambda *a, **kw: None)
    runner.init_kv_cache(model.config.model_id, model.config, model.runtime)
    monkeypatch.setattr(runner, "_static_device_tensor", lambda tensor: tensor)
    monkeypatch.setattr(
        runner,
        "_run_distributed_program",
        lambda callable_spec, *args: callable_spec.compiled(*args),
    )
    executor._runners[model.config.model_id] = runner
    monkeypatch.setattr(
        PyptoExecutor,
        "_kernel_weight",
        staticmethod(lambda weight: (_ for _ in ()).throw(AssertionError("_kernel_weight should be cached"))),
    )

    prefill_alloc = manager.allocate_for_prompt(model.config.model_id, "prefill", 1)
    executor.run_prefill(
        model,
        PrefillBatch(
            request_ids=["prefill"],
            token_ids=torch.zeros(1, 1, dtype=torch.long),
            input_embeddings=torch.ones(1, 1, model.config.hidden_size),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            kv_allocations=[prefill_alloc],
        ),
    )
    manager.free(prefill_alloc)

    decode_alloc = manager.allocate_for_prompt(model.config.model_id, "decode", 1)
    executor.run_decode(
        model,
        DecodeBatch(
            request_ids=["decode"],
            token_ids=torch.zeros(1, 1, dtype=torch.long),
            hidden_states=torch.ones(1, model.config.hidden_size),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            kv_allocations=[decode_alloc],
        ),
    )
    manager.free(decode_alloc)


def test_pypto_executor_preserves_device_group():
    executor = PyptoExecutor(device_ids=[3, 4])

    assert executor._device_ids == (3, 4)
    assert executor._run_config(codegen_only=True).device_id == 3


def test_kernel_profile_helpers_emit_kernel_name_and_runtime_timing():
    args = {"runtime": "tensormap_and_ringbuffer"}
    host_wall_us, device_wall_us = _run_timing_us(
        SimpleNamespace(host_wall_us=1234.5, device_wall_us=678.0)
    )
    _add_run_timing_args(args, SimpleNamespace(host_wall_us=1234.5, device_wall_us=678.0))

    assert _kernel_trace_name("prefill_fwd") == "kernel.prefill_fwd"
    assert _kernel_trace_name("decode_fwd") == "kernel.decode_fwd"
    assert host_wall_us == 1234.5
    assert device_wall_us == 678.0
    assert args["host_wall_us"] == 1234.5
    assert args["host_wall_ms"] == 1.2345
    assert args["device_wall_us"] == 678.0
    assert args["device_wall_ms"] == 0.678


def test_decode_host_inlines_embedding_and_sampling_into_decode_fwd():
    module_source = QWEN3_DISPATCH.read_text(encoding="utf-8")
    start = module_source.index("def qwen3_decode_host")
    end = module_source.index("def qwen3_greedy_sample_host")
    source = module_source[start:end]

    assert source.count("decode_fwd(") == 1
    assert "token_embed_fwd(" not in source
    assert "greedy_sample_fwd(" not in source

    if not QWEN3_KERNEL_DIR.is_dir():
        pytest.skip("pypto-lib submodule is not checked out")
    decode_source = (QWEN3_KERNEL_DIR / "decode_layer.py").read_text(encoding="utf-8")
    assert 'name_hint="token_embed"' in decode_source
    assert 'name_hint="greedy_sample"' in decode_source


def test_prefill_host_keeps_sampling_as_standalone_kernel():
    module_source = QWEN3_DISPATCH.read_text(encoding="utf-8")
    start = module_source.index("def qwen3_prefill_host")
    end = module_source.index("def qwen3_decode_host")
    source = module_source[start:end]

    assert source.count("prefill_fwd(") == 1
    assert "greedy_sample_fwd(" not in source
    assert "token_embed_fwd(" not in source

    if not QWEN3_KERNEL_DIR.is_dir():
        pytest.skip("pypto-lib submodule is not checked out")
    prefill_source = (QWEN3_KERNEL_DIR / "prefill_fwd.py").read_text(encoding="utf-8")
    assert 'name_hint="greedy_sample"' not in prefill_source
    assert 'name_hint="token_embed"' not in prefill_source


def _layer(hidden_size: int, intermediate_size: int, head_dim: int) -> LayerWeights:
    kv_hidden = head_dim
    return LayerWeights(
        input_rms_weight=torch.ones(hidden_size),
        wq=torch.zeros(hidden_size, hidden_size),
        wk=torch.zeros(kv_hidden, hidden_size),
        wv=torch.zeros(kv_hidden, hidden_size),
        q_norm_weight=torch.ones(head_dim),
        k_norm_weight=torch.ones(head_dim),
        wo=torch.zeros(hidden_size, hidden_size),
        post_rms_weight=torch.ones(hidden_size),
        w_gate=torch.zeros(intermediate_size, hidden_size),
        w_up=torch.zeros(intermediate_size, hidden_size),
        w_down=torch.zeros(hidden_size, intermediate_size),
    )


class _CopyKernel:
    def __call__(self, hidden, *args, config=None):
        out = args[-1]
        if out.shape == hidden.shape:
            out.copy_(hidden)
        else:
            out.zero_()


class _ImmediateEosExecutor(ModelExecutor):
    def __init__(self, kv_cache_manager: KvCacheManager) -> None:
        super().__init__(kv_cache_manager)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        logits = torch.full((len(batch.request_ids), model.config.vocab_size), -1.0)
        logits[:, 0] = 1.0
        hidden = torch.zeros(len(batch.request_ids), model.config.hidden_size)
        return PrefillResult(last_hidden=hidden, logits=logits)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        logits = torch.full((len(batch.request_ids), model.config.vocab_size), -1.0)
        logits[:, 0] = 1.0
        hidden = torch.zeros(len(batch.request_ids), model.config.hidden_size)
        return DecodeResult(hidden_states=hidden, logits=logits)


class _NoopKernel:
    def __call__(self, *args, config=None):
        return None


class _FailingSampler:
    def __init__(self) -> None:
        self.sample_calls = 0

    def from_generate_config(self, config):
        return None

    def sample(self, logits, params) -> int:
        self.sample_calls += 1
        raise AssertionError("host sampler should not be used when device sampled ids are available")


class _FixedSampler:
    def __init__(self, token_id: int) -> None:
        self.token_id = token_id
        self.sample_calls = 0

    def from_generate_config(self, config):
        return None

    def sample(self, logits, params) -> int:
        self.sample_calls += 1
        return self.token_id


class _DeviceSamplingExecutor(ModelExecutor):
    def __init__(
        self,
        kv_cache_manager: KvCacheManager,
        *,
        first_token: int,
        second_token: int,
        return_next_hidden: bool = True,
    ) -> None:
        super().__init__(kv_cache_manager)
        self.first_token = first_token
        self.second_token = second_token
        self.return_next_hidden = return_next_hidden
        self.prefill_calls = 0
        self.decode_calls = 0
        self.lookup_calls = 0
        self.decode_hidden_seen: list[torch.Tensor] = []

    @property
    def supports_device_sampling(self) -> bool:
        return True

    @property
    def supports_device_embedding(self) -> bool:
        return True

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        self.lookup_calls += 1
        if self.lookup_calls > 1:
            raise AssertionError("device-embedding decode should use a placeholder instead of host lookup")
        return super().lookup_embeddings(model, token_ids)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        self.prefill_calls += 1
        token = torch.tensor([self.first_token], dtype=torch.int64)
        return PrefillResult(
            last_hidden=None,
            logits=torch.zeros(1, model.config.vocab_size),
            sampled_token_ids=token.to(torch.int32),
            next_hidden_states=model.embed_tokens.index_select(0, token) if self.return_next_hidden else None,
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        self.decode_calls += 1
        self.decode_hidden_seen.append(batch.hidden_states[0].detach().clone())
        token = torch.tensor([self.second_token], dtype=torch.int64)
        return DecodeResult(
            hidden_states=batch.hidden_states,
            logits=torch.zeros(1, model.config.vocab_size),
            sampled_token_ids=token.to(torch.int32),
            next_hidden_states=model.embed_tokens.index_select(0, token) if self.return_next_hidden else None,
        )


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
