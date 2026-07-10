# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import asyncio
import argparse
import json
import struct
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from simpler.task_interface import DataType

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    GenerateConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    LayerWeights,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from pypto_serving.model.common.executor.executor import ModelExecutor
from pypto_serving.model.qwen.a8w8_loader import Qwen3A8W8DirectoryLoader, _SafeTensorIndex
from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor as PyptoExecutor
from pypto_serving.model.qwen.npu_executor_a8w8 import Qwen314BA8W8PyptoExecutor
from pypto_serving.model.qwen.npu_runner import (
    _CompiledKernels,
    _L3Callable,
    Qwen314BModelRunner as ModelRunner,
    _add_run_timing_args,
    _kernel_trace_name,
    _run_timing_us,
)
from pypto_serving.model.qwen.npu_runner_a8w8 import (
    Qwen314BA8W8ModelRunner,
    _KernelLayerWeights,
)
from pypto_serving.serving.engine.async_engine import (
    ReplicaEngineCore,
    TokenOutput,
    _RequestContext,
)
from pypto_serving.serving.engine.engine import LLMEngine
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestOutput,
    RequestStatus,
    ScheduledRequest,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)
from pypto_serving.serving.server.ipc import (
    PLACEHOLDER_TOKEN,
    DecodeRequest,
    NewRequestData,
    PrefillRequest,
    StepCommand,
    StepResult,
    decode_command,
    encode_command,
    encode_result,
)
from pypto_serving.serving.server.serving_worker import WorkerProcess
from pypto_serving.worker.worker import WorkerTensor
from examples.model.qwen3_14b.npu_generate import _validate_generation_args


ROOT = Path(__file__).resolve().parents[1]
QWEN3_DISPATCH = ROOT / "pypto_serving" / "model" / "qwen" / "qwen3_l3_dispatch.py"
QWEN3_KERNEL_DIR = ROOT / "pypto-lib" / "models" / "qwen3" / "14b"


def _write_safetensor(path: Path, name: str, tensor: torch.Tensor) -> None:
    raw = tensor.contiguous().numpy().tobytes()
    dtype_names = {
        torch.int8: "I8",
        torch.float32: "F32",
        torch.bfloat16: "BF16",
    }
    header = {
        name: {
            "dtype": dtype_names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [0, len(raw)],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + raw)


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [max(1, len(text))]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def test_scheduler_speculative_output_counts_only_tokens_retained_before_eos():
    manager = KvCacheManager(num_blocks=4, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(SchedulerConfig(enable_prefix_cache=False), manager)
    request = Request(
        request_id="speculative",
        prompt_token_ids=[1],
        max_new_tokens=4,
        eos_token_id=7,
        num_computed_tokens=1,
        status=RequestStatus.RUNNING,
    )
    scheduler.running.append(request)
    scheduler.requests[request.request_id] = request
    scheduled = SchedulerOutput(
        scheduled_requests=[
            ScheduledRequest(request=request, num_new_tokens=1, is_prefill=False)
        ]
    )

    outputs = scheduler.update_from_output(scheduled, {request.request_id: [7, 8]})

    assert request.output_token_ids == [7]
    assert request.num_computed_tokens == 2
    assert request.status is RequestStatus.FINISHED_EOS
    assert [(output.new_token_id, output.finished) for output in outputs] == [(7, True)]


def _running_decode_request(req_id="r", prompt=(1, 2), first_output=99):
    """A RUNNING request that finished prefill and has one decoded token, i.e.
    ready to schedule its next decode step (num_new_tokens_needed == 1)."""
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=8,
        num_computed_tokens=len(prompt),
        output_token_ids=[first_output],
        status=RequestStatus.RUNNING,
    )


def test_async_advance_after_schedule_reserves_placeholder_and_advances():
    """Optimistic advance: after schedule()+advance_after_schedule(), the step is
    'in flight' — a placeholder stands in for the unsampled token and
    num_computed is advanced, so scheduling can continue before the token
    returns without double-counting the same slot."""
    manager = KvCacheManager(num_blocks=8, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(
        SchedulerConfig(enable_prefix_cache=False, async_scheduling=True), manager
    )
    request = _running_decode_request()  # computed=2, output=[99] -> num_new_needed=1
    scheduler.running.append(request)
    scheduler.requests[request.request_id] = request

    out1 = scheduler.schedule()
    assert len(out1.scheduled_requests) == 1  # decode scheduled
    scheduler.advance_after_schedule(out1)

    # One token in flight: placeholder reserved, computed advanced to cover the
    # token this step covers (prompt 2 + output 1 -> computed 3).
    assert request.num_output_placeholders == 1
    assert request.num_computed_tokens == 3
    # num_tokens = 2 prompt + 1 output + 1 placeholder = 4, so exactly one more
    # slot is schedulable (the NEXT token). The placeholder ensures we advanced
    # rather than re-issuing the same slot; engine-side depth bounds concurrency.
    assert request.num_new_tokens_needed == 1


def test_async_reconciliation_matches_sync_end_state():
    """Driving N decode steps through the async path (schedule -> advance ->
    update_from_output) yields the same request state as the sync path."""
    def run(async_mode: bool):
        manager = KvCacheManager(num_blocks=16, block_size=2, enable_prefix_cache=False)
        scheduler = Scheduler(
            SchedulerConfig(enable_prefix_cache=False, async_scheduling=async_mode),
            manager,
        )
        request = _running_decode_request()
        scheduler.running.append(request)
        scheduler.requests[request.request_id] = request

        collected = []
        for step_token in (10, 11, 12):
            out = scheduler.schedule()
            if not out.scheduled_requests:
                break
            if async_mode:
                scheduler.advance_after_schedule(out)
            outs = scheduler.update_from_output(out, {request.request_id: [step_token]})
            collected.extend(o.new_token_id for o in outs if o.new_token_id is not None)
        return request.output_token_ids, request.num_computed_tokens, collected

    sync_out, sync_comp, sync_tokens = run(async_mode=False)
    async_out, async_comp, async_tokens = run(async_mode=True)

    assert async_out == sync_out == [99, 10, 11, 12]
    assert async_comp == sync_comp
    assert async_tokens == sync_tokens == [10, 11, 12]


def test_async_placeholder_released_and_no_leak_after_reconcile():
    """After the real token is applied, the placeholder is fully released so the
    request can be scheduled again for the next token."""
    manager = KvCacheManager(num_blocks=8, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(
        SchedulerConfig(enable_prefix_cache=False, async_scheduling=True), manager
    )
    request = _running_decode_request()
    scheduler.running.append(request)
    scheduler.requests[request.request_id] = request

    out = scheduler.schedule()
    scheduler.advance_after_schedule(out)
    assert request.num_output_placeholders == 1

    scheduler.update_from_output(out, {request.request_id: [42]})
    assert request.num_output_placeholders == 0
    assert request.output_token_ids == [99, 42]
    # Ready to schedule the next decode.
    assert request.num_new_tokens_needed == 1


def test_async_discards_stale_result_for_preempted_request():
    """A request preempted while its step is in flight must NOT have that step's
    result applied: preemption reset its computed/placeholder state, so appending
    the stale token would corrupt bookkeeping and emit a spurious output."""
    manager = KvCacheManager(num_blocks=8, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(
        SchedulerConfig(enable_prefix_cache=False, async_scheduling=True), manager
    )
    request = _running_decode_request()
    scheduler.running.append(request)
    scheduler.requests[request.request_id] = request

    out = scheduler.schedule()
    scheduler.advance_after_schedule(out)      # step N in flight

    # Preemption (as _preempt_lowest_priority does) resets state and marks the
    # request PREEMPTED before step N's result returns.
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    request.num_output_placeholders = 0

    outputs = scheduler.update_from_output(out, {request.request_id: [42]})

    # Stale token discarded: no output emitted, state untouched by reconcile.
    assert outputs == []
    assert request.output_token_ids == [99]          # unchanged (42 not appended)
    assert request.num_computed_tokens == 0           # reset preserved
    assert request.num_output_placeholders == 0


def test_async_defers_prefix_cache_publish_until_confirmed():
    """Prefix-cache blocks must be published only after the worker confirms the
    step, not optimistically at dispatch — otherwise a failed step leaves hashes
    for uncomputed KV that a later same-prompt request could hit."""
    manager = KvCacheManager(num_blocks=16, block_size=2, enable_prefix_cache=True)
    scheduler = Scheduler(
        SchedulerConfig(enable_prefix_cache=True, async_scheduling=True), manager
    )
    # Fresh prompt long enough to complete >=1 cache block on prefill.
    prompt = [5, 6, 7, 8]
    request = Request(
        request_id="p",
        prompt_token_ids=prompt,
        max_new_tokens=4,
        status=RequestStatus.WAITING,
    )
    scheduler.add_request(request)

    out = scheduler.schedule()
    assert out.scheduled_requests and out.scheduled_requests[0].is_prefill
    scheduler.advance_after_schedule(out)

    # advance_after_schedule advanced computed tokens but must NOT have published
    # any prefix-cache blocks yet.
    assert scheduler.kv_cache_manager.get_computed_blocks(prompt) == []
    assert request.num_blocks_cached == 0

    # After the worker confirms, blocks are published.
    scheduler.update_from_output(out, {request.request_id: [42]})
    assert request.num_blocks_cached >= 1


def test_worker_step_error_queues_finished_ids_for_executor_release():
    aborted: list[str] = []
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.scheduler = SimpleNamespace(abort_request=aborted.append)
    core._pending_free_ids = []
    core._batch_queue = deque()
    core._discard_result_step_ids = set()
    core._request_contexts = {
        "req-a": SimpleNamespace(queue=asyncio.Queue()),
        "req-b": SimpleNamespace(queue=asyncio.Queue()),
    }
    scheduler_output = SimpleNamespace(
        scheduled_requests=[
            SimpleNamespace(request=SimpleNamespace(request_id="req-a")),
            SimpleNamespace(request=SimpleNamespace(request_id="req-b")),
        ]
    )

    # Error path: the failed step's result was already consumed (result_pending
    # False); no in-flight batches, so nothing to drain.
    core._handle_step_error(7, scheduler_output, result_pending=False)

    assert aborted == ["req-a", "req-b"]
    assert core._pending_free_ids == ["req-a", "req-b"]
    for request_id in ("req-a", "req-b"):
        token = core._request_contexts[request_id].queue.get_nowait()
        assert isinstance(token, TokenOutput)
        assert token.finished is True
        assert token.finish_reason == "error"


def test_grouped_cache_preemption_removes_victim_from_running_queue():
    manager = KvCacheManager(block_size=1, enable_prefix_cache=False)
    manager.init_groups(
        (
            KVCacheGroupSpec(
                name="test",
                layer_indices=(0,),
                spec=KVCacheSpec(block_size=1, page_size_bytes=1),
                max_blocks_per_seq=3,
                num_blocks=3,
            ),
        ),
        max_batch_size=2,
    )
    scheduler = Scheduler(
        SchedulerConfig(
            max_num_scheduled_tokens=4,
            enable_prefix_cache=False,
            num_speculative_tokens=1,
        ),
        manager,
    )
    requests = [
        Request(
            request_id=request_id,
            prompt_token_ids=[1],
            max_new_tokens=5,
            arrival_time=arrival_time,
            status=RequestStatus.RUNNING,
            num_computed_tokens=1,
            output_token_ids=[2],
            temperature=0.0,
        )
        for request_id, arrival_time in (("older", 1.0), ("newer", 2.0))
    ]
    for request in requests:
        request.allocated_group_block_ids = manager.ensure_group_blocks(
            request.request_id, 1
        )
        request.cache_partition = 0
        scheduler.requests[request.request_id] = request
    scheduler.running = requests

    output = scheduler.schedule()

    assert [request.request_id for request in output.preempted_requests] == ["newer"]
    assert [request.request_id for request in scheduler.running] == ["older"]
    assert [request.request_id for request in scheduler.waiting] == ["newer"]


def test_step_command_preserves_grouped_cache_metadata_on_preempted_restart():
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._worker_known_req_ids = {"req"}
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2],
        max_new_tokens=1,
        status=RequestStatus.RUNNING,
    )
    scheduled = ScheduledRequest(
        request=request,
        num_new_tokens=2,
        is_prefill=True,
        block_ids_by_group={"ori": [3, 4], "state": [5]},
        cache_partition=2,
    )
    output = SchedulerOutput(scheduled_requests=[scheduled])

    command = core._build_step_command(output, finished_ids=["req"])
    decoded = decode_command(encode_command(command))

    assert [item.request_id for item in decoded.new_requests] == ["req"]
    assert decoded.finished_request_ids == ["req"]
    assert decoded.prefill_requests[0].block_ids_by_group == {
        "ori": [3, 4],
        "state": [5],
    }
    assert decoded.prefill_requests[0].cache_partition == 2


def test_partitioned_prefill_chunks_keep_cache_partitions_unique():
    requests = [
        PrefillRequest(
            request_id=request_id,
            chunk_tokens=[1],
            num_computed_tokens=0,
            block_ids=[],
            cache_partition=partition,
        )
        for request_id, partition in (("a", 0), ("b", 0), ("c", 1))
    ]

    chunks = WorkerProcess._partitioned_prefill_chunks(requests, max_batch=2)

    assert [[request.request_id for request in chunk] for chunk in chunks] == [
        ["a", "c"],
        ["b"],
    ]


def test_worker_releases_preempted_state_before_same_command_reregistration():
    released: list[str] = []
    results: list[bytes] = []
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(release_finished_requests=released.extend)
    worker._req_cache = {
        "req": NewRequestData("req", [0], 0.0, 1.0, None),
    }
    worker._last_tokens = {}
    worker.output_queue = SimpleNamespace(put=results.append)
    worker._execute_step = lambda _cmd: StepResult(new_tokens={})
    replacement = NewRequestData("req", [1, 2], 0.0, 1.0, None)
    command = StepCommand(
        new_requests=[replacement],
        prefill_requests=[],
        decode_requests=[],
        finished_request_ids=["req"],
    )

    worker._handle_step_command(command)

    assert released == ["req"]
    assert worker._req_cache["req"] == replacement
    assert len(results) == 1


def test_abort_request_schedules_worker_cleanup():
    """An aborted request must ride the next StepCommand's finished_request_ids,
    otherwise its worker-side _req_cache entry and device slots leak."""
    aborted: list[str] = []
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.scheduler = SimpleNamespace(abort_request=aborted.append)
    core._pending_free_ids = []
    core._request_contexts = {"req-x": SimpleNamespace(queue=asyncio.Queue())}

    asyncio.run(core.abort_request("req-x"))

    # Scheduler aborted, context removed.
    assert aborted == ["req-x"]
    assert "req-x" not in core._request_contexts
    # The id is queued for worker release exactly once.
    assert core._pending_free_ids == ["req-x"]

    # Idempotent: a second abort (or an abort racing the finish path) must not
    # enqueue a duplicate free id.
    asyncio.run(core.abort_request("req-x"))
    assert core._pending_free_ids == ["req-x"]


def test_abort_request_emits_abort_token_before_scheduling_free():
    """The client-facing queue receives a FINISHED_ABORTED token on abort."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.scheduler = SimpleNamespace(abort_request=lambda _req_id: None)
    core._pending_free_ids = []
    queue: asyncio.Queue = asyncio.Queue()
    core._request_contexts = {"req-y": SimpleNamespace(queue=queue)}

    asyncio.run(core.abort_request("req-y"))

    token = queue.get_nowait()
    assert isinstance(token, TokenOutput)
    assert token.finished is True
    assert token.finish_reason == "FINISHED_ABORTED"
    assert core._pending_free_ids == ["req-y"]


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
        final_norm_weight=torch.ones(1, hidden_size),
        rope_cos=torch.zeros(max_seq, head_dim),
        rope_sin=torch.zeros(max_seq, head_dim),
        padded_vocab=model.config.vocab_size,
        padded_lm_head_weight=torch.zeros(model.config.vocab_size, hidden_size),
        padded_embed_weight=torch.zeros(model.config.vocab_size, hidden_size),
        decode_weights=decode_weights,
        prefill_token_ids_buffer=torch.empty(kernel_batch * max_seq, dtype=torch.int32),
        prefill_seq_lens_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_chunk_lens_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_chunk_offsets_buffer=torch.empty(kernel_batch, dtype=torch.int32),
        prefill_block_table_buffer=torch.empty(kernel_batch * max_blocks, dtype=torch.int32),
        prefill_slot_mapping_buffer=torch.empty(kernel_batch * max_seq, dtype=torch.int32),
        prefill_logits_buffer=torch.empty(kernel_batch, model.config.vocab_size),
        prefill_sampled_ids_buffer=torch.empty(kernel_batch, sampled_ids_width, dtype=torch.int32),
        prefill_next_hidden_buffer=torch.empty(kernel_batch, hidden_size, dtype=torch.bfloat16),
        decode_seq_lens_buffer=torch.zeros(kernel_batch, dtype=torch.int32),
        decode_block_table_buffer=torch.zeros(kernel_batch * max_blocks, dtype=torch.int32),
        decode_slot_mapping_buffer=torch.zeros(kernel_batch, dtype=torch.int32),
        decode_logits_buffer=torch.zeros(kernel_batch, model.config.vocab_size),
        decode_token_ids_buffer=torch.empty(kernel_batch, sampled_ids_width, dtype=torch.int32),
        decode_sampled_ids_buffer=torch.empty(kernel_batch, sampled_ids_width, dtype=torch.int32),
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
    prepared = runner._prepare_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=[alloc.request_id for alloc in allocations],
            token_ids=torch.tensor([[1, 0], [2, 3]], dtype=torch.long),
            input_embeddings=None,
            seq_lens=seq_lens,
            kv_allocations=allocations,
        ),
    )

    assert prepared.actual_batch == 2
    assert prepared.token_ids.shape == (3,)
    assert prepared.token_ids.tolist() == [1, 2, 3]
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
            token_ids=torch.tensor([[5, 6]], dtype=torch.long),
            input_embeddings=None,
            seq_lens=torch.tensor([4], dtype=torch.int32),
            kv_allocations=[alloc],
            positions=torch.tensor([[2, 3]], dtype=torch.long),
        ),
    )

    assert prepared.token_ids.tolist() == [5, 6]
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
                input_embeddings=None,
                seq_lens=torch.tensor([4], dtype=torch.int32),
                kv_allocations=[alloc],
                positions=torch.tensor([[1, 3, 4]], dtype=torch.long),
            ),
        )


def test_compute_slot_mapping_rejects_insufficient_pages():
    with pytest.raises(ValueError, match="too small"):
        ModelRunner._compute_slot_mapping([0], 2, 2, start_pos=1)


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

    assert prepared.actual_batch == 1
    assert prepared.token_ids is compiled.decode_token_ids_buffer
    assert prepared.seq_lens is compiled.decode_seq_lens_buffer
    assert prepared.block_table is compiled.decode_block_table_buffer
    assert prepared.slot_mapping is compiled.decode_slot_mapping_buffer
    assert prepared.logits is compiled.decode_logits_buffer
    assert prepared.token_ids[:, :1].tolist() == [[7], [7]]
    assert torch.count_nonzero(prepared.token_ids[:, 1:]).item() == 0
    assert prepared.seq_lens.tolist() == [1, 1]
    assert prepared.block_table.reshape(2, 2).tolist() == [
        [alloc.page_ids[0], -1],
        [alloc.page_ids[0], -1],
    ]
    expected_slot = manager.slot_mapping_for_request(alloc)
    assert prepared.slot_mapping.tolist() == [expected_slot, expected_slot]


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
    assert prepared.block_table.tolist() == alloc.page_ids


def test_a8w8_decode_inputs_use_actual_user_batch_without_padding_lanes():
    model = _model(max_batch_size=16)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    runner = Qwen314BA8W8ModelRunner(
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


def test_a8w8_init_kv_cache_returns_page_count_for_first_and_repeated_init(monkeypatch):
    model = _model(max_batch_size=16, max_seq_len=128, page_size=64)
    runner = Qwen314BA8W8ModelRunner(
        compiled=None,  # type: ignore[arg-type]
    )
    allocated_shapes = []

    def alloc_kv_cache_tensor(shape, dtype):
        allocated_shapes.append((shape, dtype))
        return WorkerTensor(
            data_ptr=len(allocated_shapes),
            shape=shape,
            dtype=_FakeWorker._DTYPES[dtype],
        )

    monkeypatch.setattr(runner, "_alloc_kv_cache_tensor", alloc_kv_cache_tensor)
    monkeypatch.setattr(runner, "_free_kv_cache_tensor", lambda tensor: None)

    first_pages = runner.init_kv_cache(model.config.model_id, model.config, model.runtime)
    second_pages = runner.init_kv_cache(model.config.model_id, model.config, model.runtime)

    assert first_pages == 32
    assert second_pages == 32
    cache_rows = model.config.num_hidden_layers * 32 * model.config.num_key_value_heads * model.runtime.page_size
    assert allocated_shapes == [
        ((cache_rows, model.config.head_dim), torch.bfloat16),
        ((cache_rows, model.config.head_dim), torch.bfloat16),
        ((cache_rows, 8), torch.float32),
        ((cache_rows, 8), torch.float32),
    ]


def test_a8w8_runtime_contract_rejects_mismatched_page_size():
    model = _model(max_batch_size=16, max_seq_len=128, page_size=64)

    with pytest.raises(ValueError, match="page_size=128"):
        Qwen314BA8W8PyptoExecutor._validate_kernel_runtime_contract(
            model,
            SimpleNamespace(MAX_SEQ=128),
            SimpleNamespace(BLOCK_SIZE=128),
        )


def test_a8w8_runtime_contract_rejects_excessive_max_seq_len():
    model = _model(max_batch_size=16, max_seq_len=256, page_size=128)

    with pytest.raises(ValueError, match="max_seq_len <= 128"):
        Qwen314BA8W8PyptoExecutor._validate_kernel_runtime_contract(
            model,
            SimpleNamespace(MAX_SEQ=128),
            SimpleNamespace(BLOCK_SIZE=128),
        )


def test_a8w8_stack_decode_weights_releases_per_layer_sources():
    def layer(value: float) -> _KernelLayerWeights:
        return _KernelLayerWeights(
            input_rms_weight=torch.full((1, 2), value),
            wq=torch.full((2, 2), value),
            wk=torch.full((2, 1), value),
            wv=torch.full((2, 1), value),
            q_norm_weight=torch.full((1, 1), value),
            k_norm_weight=torch.full((1, 1), value),
            wo=torch.full((2, 2), value),
            post_rms_weight=torch.full((1, 2), value),
            w_gate=torch.full((2, 3), value),
            w_up=torch.full((2, 3), value),
            w_down=torch.full((3, 2), value),
            wq_scale=torch.full((1, 2), value),
            wk_scale=torch.full((1, 1), value),
            wv_scale=torch.full((1, 1), value),
            wo_scale=torch.full((1, 2), value),
        )

    layers = [layer(1.0), layer(2.0)]

    weights = Qwen314BA8W8PyptoExecutor._stack_decode_weights(layers)

    assert weights["decode_wq"].shape == (4, 2)
    assert weights["decode_wq_scale"].shape == (2, 2)
    assert all(layer_weights.wq.numel() == 0 for layer_weights in layers)
    assert all(layer_weights.wq_scale is not None and layer_weights.wq_scale.numel() == 0 for layer_weights in layers)


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


def test_engine_omits_decode_hidden_when_executor_embeds_on_device():
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
    assert executor.lookup_calls == 0
    assert executor.decode_calls == 1
    assert executor.decode_hidden_seen[0] is None
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
    assert executor.lookup_calls == 0
    assert executor.decode_calls == 1
    assert executor.decode_hidden_seen[0] is None
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


def test_a8w8_loader_detects_unindexed_safetensors(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _write_safetensor(tmp_path / "weights.safetensors", "tensor", torch.tensor([[1, 2]], dtype=torch.int8))

    loader = Qwen3A8W8DirectoryLoader()
    index = _SafeTensorIndex(tmp_path)

    assert loader.can_load(tmp_path)
    assert torch.equal(index.load("tensor"), torch.tensor([[1, 2]], dtype=torch.int8))


def test_a8w8_loader_uses_index_shard_names(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _write_safetensor(tmp_path / "custom-shard.safetensors", "tensor", torch.tensor([3], dtype=torch.int8))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "custom-shard.safetensors"}}),
        encoding="utf-8",
    )

    loader = Qwen3A8W8DirectoryLoader()
    index = _SafeTensorIndex(tmp_path)

    assert loader.can_load(tmp_path)
    assert torch.equal(index.load("tensor"), torch.tensor([3], dtype=torch.int8))


def test_a8w8_num_layers_override_fails_fast():
    with pytest.raises(ValueError, match="num-layers-override"):
        _validate_generation_args(
            argparse.Namespace(
                model_format="qwen3-a8w8",
                num_layers_override=1,
                tensor_parallel_size=1,
            )
        )


def test_a8w8_tensor_parallel_fails_fast():
    with pytest.raises(ValueError, match="requires --tp 1"):
        _validate_generation_args(
            argparse.Namespace(
                model_format="qwen3-a8w8",
                num_layers_override=None,
                tensor_parallel_size=2,
            )
        )


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
    worker._req_cache = {
        "decode": NewRequestData(
            request_id="decode",
            prompt_token_ids=[1],
            temperature=0.0,
            top_p=1.0,
            top_k=None,
        )
    }

    # last_token=3 (the one output token), prev_token=prompt_ids[-1]=1, seq_len=2.
    decode_req = DecodeRequest(
        request_id="decode",
        last_token=3,
        prev_token=1,
        seq_len=2,
        block_ids=[0],
    )
    new_tokens: dict[str, list[int]] = {}

    worker._batch_decode([decode_req], model, new_tokens)

    assert new_tokens == {"decode": [0]}
    assert executor.decode_calls == 1
    assert executor.decode_hidden_seen[0] is None


def test_worker_resolves_placeholder_decode_token_from_cache():
    """Under async scheduling the engine sends PLACEHOLDER_TOKEN; the worker must
    substitute the token(s) it last sampled for that request."""
    worker = WorkerProcess.__new__(WorkerProcess)
    worker._last_tokens = {}

    # Record two sampled tokens (simulating two prior decode steps).
    worker._record_last_tokens("r", [11])
    worker._record_last_tokens("r", [22])
    assert worker._last_tokens["r"] == [11, 22]

    placeholder = DecodeRequest(
        request_id="r",
        last_token=PLACEHOLDER_TOKEN,
        prev_token=PLACEHOLDER_TOKEN,
        seq_len=5,
        block_ids=[0],
    )
    # last -> most recent (22); prev -> second-most-recent (11).
    assert worker._resolve_decode_token(placeholder) == 22
    assert worker._resolve_prev_token(placeholder) == 11

    # A real (non-placeholder) token is passed through untouched.
    explicit = DecodeRequest(
        request_id="r", last_token=99, prev_token=88, seq_len=5, block_ids=[0]
    )
    assert worker._resolve_decode_token(explicit) == 99
    assert worker._resolve_prev_token(explicit) == 88

    # Cache keeps only the last 2 tokens (MTP prev context bound).
    worker._record_last_tokens("r", [33])
    assert worker._last_tokens["r"] == [22, 33]

    # Missing cache entry on placeholder is a hard error (never silently wrong).
    orphan = DecodeRequest(
        request_id="missing", last_token=PLACEHOLDER_TOKEN, prev_token=PLACEHOLDER_TOKEN,
        seq_len=1, block_ids=[0],
    )
    with pytest.raises(RuntimeError):
        worker._resolve_decode_token(orphan)


def test_incremental_detok_matches_full_decode_and_hides_partial_chars():
    """Incremental detok must equal a full decode and never stream a partial char.

    Guards the O(N^2) -> O(N) detokenization fix: cumulative text produced step
    by step must match tokenizer.decode(all_ids), and an incomplete multi-token
    character (rendered as U+FFFD) must be withheld until it completes.
    """
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)

    class _MultiByteTokenizer:
        # tokens 6+7 together render '★'; 6 alone is an incomplete char.
        _table = {1: "He", 2: "llo", 3: " wor", 4: "ld", 5: "!"}

        def decode(self, ids):
            out, i = [], 0
            while i < len(ids):
                t = ids[i]
                if t == 6:
                    if i + 1 < len(ids) and ids[i + 1] == 7:
                        out.append("★"); i += 2; continue
                    out.append("�"); i += 1; continue
                if t == 7:
                    out.append("★"); i += 1; continue
                out.append(self._table[t]); i += 1
            return "".join(out)

    core.tokenizer = _MultiByteTokenizer()
    ctx = _RequestContext(request=SimpleNamespace(output_token_ids=[]))

    seq = [1, 2, 3, 4, 5, 6, 7]
    cumulative = ""
    per_step = []
    for k in range(1, len(seq) + 1):
        ctx.request.output_token_ids = seq[:k]
        cumulative = core._detokenize_incrementally(ctx)
        per_step.append(cumulative)

    # No partial char ever leaked, and the final text equals a full decode.
    assert all("�" not in text for text in per_step)
    assert per_step[4] == "Hello world!"       # step 6 (idx 5) withholds partial
    assert per_step[5] == "Hello world!"        # still withheld
    assert cumulative == core.tokenizer.decode(seq) == "Hello world!★"


def test_finalize_detok_flushes_trailing_incomplete_char_at_eos():
    """If generation stops while a multi-token char is incomplete, the finished
    step must flush the authoritative full decode instead of the withheld text.

    Guards the FINAL_ONLY truncation bug: the incremental path withholds a
    trailing U+FFFD forever (no later token completes it once generation ends),
    so _finalize_detokenization must fall back to a full decode.
    """
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)

    class _TrailingByteTokenizer:
        # token 6 alone is an incomplete char (U+FFFD); it is the last token.
        _table = {1: "Hi", 2: "!"}

        def decode(self, ids):
            out = []
            for t in ids:
                out.append("�" if t == 6 else self._table[t])
            return "".join(out)

    core.tokenizer = _TrailingByteTokenizer()
    ctx = _RequestContext(request=SimpleNamespace(output_token_ids=[]))

    # Drive incremental decode up to the trailing incomplete token.
    for k in range(1, 4):
        ctx.request.output_token_ids = [1, 2, 6][:k]
        incremental = core._detokenize_incrementally(ctx)

    # Incremental withholds the trailing U+FFFD (never emits a partial char).
    assert incremental == "Hi!"
    assert "�" not in incremental

    # On finish, the authoritative full decode is flushed (no truncation).
    final = core._finalize_detokenization(ctx)
    assert final == core.tokenizer.decode([1, 2, 6]) == "Hi!�"


class _ScriptedScheduler:
    """Stub scheduler: returns one preset RequestOutput per _process_step_output call."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.finished = []

    def update_from_output(self, scheduler_output, new_tokens):
        return [self._outputs.pop(0)]

    def finish_request(self, request_id, status):
        self.finished.append((request_id, status))


class _WordTokenizer:
    _table = {1: "a", 2: "b", 3: "c", 4: "STOP"}

    def decode(self, ids):
        return "".join(self._table[t] for t in ids)


def _drive(core, ctx, token_ids):
    """Append each token and run one _process_step_output step per token."""
    for t in token_ids:
        ctx.request.output_token_ids.append(t)
        core._process_step_output(SchedulerOutput(scheduled_requests=[]), {})


def test_non_streaming_suppresses_intermediate_outputs():
    """A non-streaming request enqueues exactly one (final) TokenOutput; a
    streaming request enqueues one per token."""
    seq = [1, 2, 3]

    # --- non-streaming: stream=False -> only the final token is published ---
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _WordTokenizer()
    core._pending_free_ids = []
    outputs = [
        RequestOutput(request_id="r", new_token_id=1),
        RequestOutput(request_id="r", new_token_id=2),
        RequestOutput(request_id="r", new_token_id=3, finished=True, finish_reason="FINISHED_LENGTH"),
    ]
    core.scheduler = _ScriptedScheduler(outputs)
    ns_ctx = _RequestContext(
        request=Request(request_id="r", prompt_token_ids=[9], max_new_tokens=3),
        stream=False,
    )
    core._request_contexts = {"r": ns_ctx}

    _drive(core, ns_ctx, seq)

    assert ns_ctx.queue.qsize() == 1
    final = ns_ctx.queue.get_nowait()
    assert final.finished is True
    assert final.text == "abc"                 # full cumulative text on the final output
    assert "r" in core._pending_free_ids

    # --- streaming: stream=True -> one TokenOutput per token ---
    core2 = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core2.tokenizer = _WordTokenizer()
    core2._pending_free_ids = []
    outputs2 = [
        RequestOutput(request_id="r", new_token_id=1),
        RequestOutput(request_id="r", new_token_id=2),
        RequestOutput(request_id="r", new_token_id=3, finished=True, finish_reason="FINISHED_LENGTH"),
    ]
    core2.scheduler = _ScriptedScheduler(outputs2)
    s_ctx = _RequestContext(
        request=Request(request_id="r", prompt_token_ids=[9], max_new_tokens=3),
        stream=True,
    )
    core2._request_contexts = {"r": s_ctx}

    _drive(core2, s_ctx, seq)

    assert s_ctx.queue.qsize() == 3
    texts = [s_ctx.queue.get_nowait().text for _ in range(3)]
    assert texts == ["a", "ab", "abc"]         # cumulative text grows each step


def test_non_streaming_still_detects_stop_string():
    """Stop-string detection must run every step even when outputs are
    suppressed, so a non-streaming request stops mid-generation and publishes
    exactly one finished output."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _WordTokenizer()
    core._pending_free_ids = []
    # Would run 4 tokens, but token 4 decodes to "STOP" which is a stop string.
    outputs = [
        RequestOutput(request_id="r", new_token_id=1),
        RequestOutput(request_id="r", new_token_id=2),
        RequestOutput(request_id="r", new_token_id=4),   # -> text ends with "STOP"
        RequestOutput(request_id="r", new_token_id=3),   # should never be reached
    ]
    scheduler = _ScriptedScheduler(outputs)
    core.scheduler = scheduler
    ctx = _RequestContext(
        request=Request(
            request_id="r",
            prompt_token_ids=[9],
            max_new_tokens=4,
            stop_strings=("STOP",),
        ),
        stream=False,
    )
    core._request_contexts = {"r": ctx}

    _drive(core, ctx, [1, 2, 4])

    # Stop detected at step 3: scheduler.finish_request called, one final output.
    assert scheduler.finished == [("r", RequestStatus.FINISHED_STOP)]
    assert ctx.queue.qsize() == 1
    final = ctx.queue.get_nowait()
    assert final.finished is True
    assert final.finish_reason == "FINISHED_STOP"
    assert final.text == "abSTOP"


def test_non_streaming_final_output_uses_full_decode_on_incomplete_char():
    """FINAL_ONLY must publish the full-decode text on finish, even when the
    last token leaves a multi-token character incomplete (U+FFFD)."""

    class _TrailingByteTokenizer:
        _table = {1: "a", 2: "b"}

        def decode(self, ids):
            return "".join("�" if t == 6 else self._table[t] for t in ids)

    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _TrailingByteTokenizer()
    core._pending_free_ids = []
    outputs = [
        RequestOutput(request_id="r", new_token_id=1),
        RequestOutput(request_id="r", new_token_id=2),
        RequestOutput(request_id="r", new_token_id=6, finished=True, finish_reason="FINISHED_LENGTH"),
    ]
    core.scheduler = _ScriptedScheduler(outputs)
    ctx = _RequestContext(
        request=Request(request_id="r", prompt_token_ids=[9], max_new_tokens=3),
        stream=False,
    )
    core._request_contexts = {"r": ctx}

    _drive(core, ctx, [1, 2, 6])

    assert ctx.queue.qsize() == 1
    final = ctx.queue.get_nowait()
    assert final.finished is True
    # Not the withheld "ab": the trailing incomplete char is flushed.
    assert final.text == "ab�"


def test_process_step_output_schedules_free_once_on_normal_finish():
    """A normally-finished request is scheduled for worker release exactly once
    by _process_step_output (the add_request finally must not re-add it)."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.tokenizer = _WordTokenizer()
    core._pending_free_ids = []
    outputs = [
        RequestOutput(request_id="r", new_token_id=1, finished=True, finish_reason="FINISHED_LENGTH"),
    ]
    core.scheduler = _ScriptedScheduler(outputs)
    ctx = _RequestContext(
        request=Request(request_id="r", prompt_token_ids=[9], max_new_tokens=1),
        stream=False,
    )
    core._request_contexts = {"r": ctx}

    _drive(core, ctx, [1])

    # Scheduled once. The engine loop will drain this into the next StepCommand;
    # add_request's finally must not append it again (double-release guard).
    assert core._pending_free_ids == ["r"]
    # _schedule_worker_free is idempotent while the id is still queued.
    core._schedule_worker_free("r")
    assert core._pending_free_ids == ["r"]


def test_flush_pending_frees_sends_cleanup_only_step_command():
    """Aborting the last active request must not pin it on the worker: when no
    work is schedulable, _flush_pending_frees emits a cleanup-only StepCommand
    carrying the pending ids and drains the worker reply."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.config = SimpleNamespace(executor_cls="PyptoQwen14BExecutor")
    core._worker_known_req_ids = {"aborted"}
    core._pending_free_ids = ["aborted"]
    core._batch_queue = deque()
    core._discard_result_step_ids = set()
    core._step_counter = 0
    core._step_timeout = 300.0

    sent: list[bytes] = []
    core._input_queue = SimpleNamespace(put=sent.append)
    # Worker replies with an empty StepResult for the cleanup-only step.
    core._output_queue = SimpleNamespace(
        get=lambda timeout=None: encode_result(StepResult(new_tokens={}))
    )

    asyncio.run(core._flush_pending_frees())

    # Exactly one cleanup command was sent, carrying the pending id and no work.
    assert len(sent) == 1
    cmd = decode_command(sent[0])
    assert cmd.finished_request_ids == ["aborted"]
    assert cmd.new_requests == []
    assert cmd.prefill_requests == []
    assert cmd.decode_requests == []
    # Pending list drained; known-set no longer tracks the released id.
    assert core._pending_free_ids == []
    assert "aborted" not in core._worker_known_req_ids


def test_flush_pending_frees_noop_when_nothing_pending():
    """No cleanup command is sent when there is nothing to free."""
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._pending_free_ids = []
    sent: list[bytes] = []
    core._input_queue = SimpleNamespace(put=sent.append)

    asyncio.run(core._flush_pending_frees())

    assert sent == []


def _async_pipeline_core():
    """A ReplicaEngineCore wired for async pipelining with a fake in-process
    worker: input_queue records dispatched commands, output_queue synthesises a
    deterministic StepResult (each scheduled decode samples last_token+1)."""
    from pypto_serving.serving.server.ipc import (
        StepResult,
        decode_command,
        encode_result,
    )

    manager = KvCacheManager(num_blocks=32, block_size=2, enable_prefix_cache=False)
    scheduler = Scheduler(
        SchedulerConfig(enable_prefix_cache=False, async_scheduling=True), manager
    )
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.scheduler = scheduler
    core.kv_cache_manager = manager
    core.tokenizer = _Tokenizer()
    core.config = SimpleNamespace(executor_cls="PyptoQwen14BExecutor")
    core._async_scheduling = True
    core._pending_free_ids = []
    core._worker_known_req_ids = set()
    core._request_contexts = {}
    core._batch_queue = deque()
    core._discard_result_step_ids = set()
    core._step_timeout = 300.0
    core._max_in_flight = 2
    core._step_counter = 0

    dispatched: list = []
    results: "deque" = deque()

    def _put_cmd(raw: bytes):
        cmd = decode_command(raw)
        dispatched.append(cmd)
        # Synthesise the worker result: each decode/prefill-completing req yields
        # one token = resolved_last_token + 1 (placeholder resolves from prior).
        new_tokens: dict[str, list[int]] = {}
        for dr in cmd.decode_requests:
            # Mirror the worker's placeholder resolution against our fake cache.
            last = _fake_worker_last.get(dr.request_id)
            base = dr.last_token if dr.last_token != PLACEHOLDER_TOKEN else last
            tok = base + 1
            new_tokens[dr.request_id] = [tok]
            _fake_worker_last[dr.request_id] = tok
        for pr in cmd.prefill_requests:
            # Prefill that completes the prompt samples one token.
            tok = 1000
            new_tokens[pr.request_id] = [tok]
            _fake_worker_last[pr.request_id] = tok
        results.append(encode_result(StepResult(new_tokens=new_tokens, step_id=cmd.step_id)))

    def _get_result(timeout=None):
        return results.popleft()

    _fake_worker_last: dict[str, int] = {}
    core._input_queue = SimpleNamespace(put=_put_cmd)
    core._output_queue = SimpleNamespace(get=_get_result)
    return core, dispatched


def test_async_pipeline_dispatches_two_steps_before_applying_first():
    """Depth-2: the loop dispatches step N+1 while step N is still in flight,
    so two commands reach the worker before the first result is applied."""
    core, dispatched = _async_pipeline_core()
    req = _running_decode_request(prompt=(1, 2), first_output=50)
    core.scheduler.running.append(req)
    core.scheduler.requests[req.request_id] = req

    # Dispatch twice without applying — mirrors the loop filling the queue.
    assert core._try_dispatch_step() is True
    assert core._try_dispatch_step() is True
    assert len(core._batch_queue) == 2       # two steps in flight
    assert len(dispatched) == 2

    # Second dispatched decode carries a PLACEHOLDER (its input token is the
    # not-yet-applied result of the first in-flight step).
    second = dispatched[1]
    assert second.decode_requests[0].last_token == PLACEHOLDER_TOKEN

    # Now drain both in order; output must be the greedy chain 51, 52.
    asyncio.run(core._await_and_apply_oldest())
    asyncio.run(core._await_and_apply_oldest())
    assert req.output_token_ids == [50, 51, 52]
    assert req.num_output_placeholders == 0


def test_async_pipeline_drains_stale_result_after_error():
    """If the oldest in-flight step errors while a later step is queued, the
    later step's result (already in transit from the FIFO worker) must be drained
    and NOT misapplied to a subsequent batch."""
    from pypto_serving.serving.server.ipc import StepResult, decode_result, encode_result

    core, _dispatched = _async_pipeline_core()
    req = _running_decode_request(prompt=(1, 2), first_output=50)
    core.scheduler.running.append(req)
    core.scheduler.requests[req.request_id] = req
    core._request_contexts = {
        "r": SimpleNamespace(queue=asyncio.Queue(), request=req, stream=True)
    }

    # Two steps dispatched: both commands reached the FIFO worker, so both
    # results are already in transit.
    assert core._try_dispatch_step() is True
    assert core._try_dispatch_step() is True
    assert len(core._batch_queue) == 2
    first_step_id, second_step_id = core._batch_queue[0][0], core._batch_queue[1][0]

    # Drive the output queue by hand: step N errors, step N+1's (now stale)
    # result sits behind it, then a fresh live step (id=99) lands.
    outq = deque([
        encode_result(StepResult(new_tokens={}, error="boom", step_id=first_step_id)),
        encode_result(StepResult(new_tokens={"r": [51]}, step_id=second_step_id)),
        encode_result(StepResult(new_tokens={"r": [77]}, step_id=99)),
    ])
    core._output_queue = SimpleNamespace(get=lambda timeout=None: outq.popleft())

    # Apply the oldest: it errors -> both in-flight batches discarded, the
    # second step's id recorded for draining, request aborted with an error.
    assert asyncio.run(core._await_and_apply_oldest()) is False
    assert not core._batch_queue
    assert second_step_id in core._discard_result_step_ids
    tok = core._request_contexts["r"].queue.get_nowait()
    assert tok.finished is True and tok.finish_reason == "error"

    # Next live fetch must SKIP the stale second-step result and return step 99,
    # leaving the discard set empty (no stale result misapplied).
    got = decode_result(asyncio.run(core._get_live_result()))
    assert got.step_id == 99
    assert core._discard_result_step_ids == set()


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
    monkeypatch.setattr(ModelRunner, "_static_device_tensor", staticmethod(lambda tensor: tensor))
    runner = ModelRunner(
        compiled=compiled,
    )
    monkeypatch.setattr(runner, "_shared_l3_worker", lambda: _FakeWorker())
    monkeypatch.setattr(runner, "_compute_kv_cache_pages", lambda config, runtime, device_id=None: 1)
    monkeypatch.setattr(runner, "_print_memory_breakdown", lambda *a, **kw: None)
    runner.init_kv_cache(model.config.model_id, model.config, model.runtime)
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
            input_embeddings=None,
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
    decode_kernel = QWEN3_KERNEL_DIR / "decode_layer.py"
    if not decode_kernel.is_file():
        decode_kernel = QWEN3_KERNEL_DIR / "decode_fwd.py"
    if not decode_kernel.is_file():
        pytest.skip("pypto-lib decode kernel source is not checked out")
    decode_source = decode_kernel.read_text(encoding="utf-8")
    assert 'name_hint="token_embed"' in decode_source
    assert 'name_hint="greedy_sample"' in decode_source


def test_prefill_host_inlines_embedding_and_keeps_sampling_standalone():
    module_source = QWEN3_DISPATCH.read_text(encoding="utf-8")
    start = module_source.index("def qwen3_prefill_host")
    end = module_source.index("def qwen3_decode_host")
    source = module_source[start:end]

    assert source.count("prefill_fwd(") == 1
    assert "greedy_sample_fwd(" not in source
    assert "token_embed_fwd(" not in source
    assert "embed_weight:" in source
    assert "input_ids:" in source

    if not QWEN3_KERNEL_DIR.is_dir():
        pytest.skip("pypto-lib submodule is not checked out")
    prefill_source = (QWEN3_KERNEL_DIR / "prefill_fwd.py").read_text(encoding="utf-8")
    assert 'name_hint="greedy_sample"' not in prefill_source
    assert 'name_hint="token_embed"' in prefill_source


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
    def __call__(self, *args, config=None):
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if len(tensors) < 2:
            return None
        src, out = tensors[0], tensors[-1]
        if out.shape == src.shape:
            out.copy_(src)
        else:
            out.zero_()
        return None


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
        self.decode_hidden_seen: list[torch.Tensor | None] = []

    @property
    def supports_device_sampling(self) -> bool:
        return True

    @property
    def supports_device_embedding(self) -> bool:
        return True

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        self.lookup_calls += 1
        raise AssertionError("device-embedding prefill/decode should not use host lookup")

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        self.prefill_calls += 1
        assert batch.input_embeddings is None
        token = torch.tensor([self.first_token], dtype=torch.int64)
        return PrefillResult(
            last_hidden=None,
            logits=torch.zeros(1, model.config.vocab_size),
            sampled_token_ids=token.to(torch.int32),
            next_hidden_states=model.embed_tokens.index_select(0, token) if self.return_next_hidden else None,
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        self.decode_calls += 1
        hidden = batch.hidden_states
        self.decode_hidden_seen.append(None if hidden is None else hidden[0].detach().clone())
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
