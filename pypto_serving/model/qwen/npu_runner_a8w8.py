# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import torch

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    PrefillBatch,
    PrefillResult,
    RuntimeModel,
)
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.qwen.npu_runner import Qwen314BModelRunner, _add_run_timing_args
from pypto_serving.tools.profile import profile_span
from pypto_serving.worker.worker import Worker as LlmWorker
from pypto_serving.worker.worker import WorkerTensor

_QWEN14B_A8W8_PREFILL_CHUNK_LAYERS = 10
_QWEN14B_LM_HEAD_CHUNK_ROWS = 8192


def _l2_trace_name(kernel_name: str) -> str:
    if "prefill" in kernel_name:
        return "kernel.prefill_fwd"
    if "decode" in kernel_name:
        return "kernel.decode_fwd"
    if "final_rms" in kernel_name:
        return "kernel.final_rms"
    if "lm_head" in kernel_name:
        return "kernel.lm_head"
    return f"kernel.{kernel_name}"


@dataclass
class _KernelLayerWeights:
    """Kernel-ready weights for one transformer layer."""

    input_rms_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    wo: torch.Tensor
    post_rms_weight: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    wq_scale: torch.Tensor | None = None
    wk_scale: torch.Tensor | None = None
    wv_scale: torch.Tensor | None = None
    wo_scale: torch.Tensor | None = None


@dataclass
class _L2Callable:
    """Assembled non-L3 callable and launch metadata."""

    chip_callable: object
    name: str
    runtime_name: str
    block_dim: int
    aicpu_thread_num: int
    param_infos: tuple[object, ...]


@dataclass
class _CompiledKernels:
    """Compiled Qwen3-14B kernels and immutable runtime tensors."""

    prefill: _L2Callable
    decode: _L2Callable
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    decode_weights: dict[str, torch.Tensor]
    decode_logits_buffer: torch.Tensor


@dataclass
class _PrefillInputs:
    """Host tensors passed to the prefill kernel."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    chunk_lens: torch.Tensor
    chunk_offsets: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeInputs:
    """Padded host tensors passed to the decode kernel."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _L2ProgramHandle:
    """L2 callable registration state for one runner process."""

    callable_id: int
    runtime_name: str


class Qwen314BA8W8ModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    _compute_slot_mapping = staticmethod(Qwen314BModelRunner._compute_slot_mapping)
    _write_block_table_row = staticmethod(Qwen314BModelRunner._write_block_table_row)
    _validate_batch_size = staticmethod(Qwen314BModelRunner._validate_batch_size)
    _max_blocks_per_seq = staticmethod(Qwen314BModelRunner._max_blocks_per_seq)

    def __init__(
        self,
        *,
        compiled: _CompiledKernels,
        model_id: str = "test-model",
        platform: str = "a2a3sim",
        device_id: int = 0,
        save_kernels_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._model_id = model_id
        self._compiled = compiled
        self._platform = platform
        self._device_id = device_id
        self._save_kernels_dir = save_kernels_dir
        self._l2_workers: dict[str, LlmWorker] = {}
        self._l2_programs: dict[int, _L2ProgramHandle] = {}
        self._l2_child_allocs: dict[tuple[str, int], tuple[int, int]] = {}
        self._kv_scale_caches: dict[str, tuple[WorkerTensor, WorkerTensor]] = {}

    def init_kv_cache(self, model_id: str, config, runtime) -> int:
        """Create the runner-owned KV cache, plus INT8 scale pages for A8W8."""
        num_pages = super().init_kv_cache(model_id, config, runtime)
        if model_id in self._kv_scale_caches:
            return num_pages
        cache_rows = config.num_hidden_layers * num_pages * config.num_key_value_heads * runtime.page_size
        key_scale = self._alloc_kv_cache_tensor((cache_rows, 8), torch.float32)
        try:
            value_scale = self._alloc_kv_cache_tensor((cache_rows, 8), torch.float32)
        except Exception:
            self._free_kv_cache_tensor(key_scale)
            raise
        self._kv_scale_caches[model_id] = (key_scale, value_scale)
        return num_pages

    def close_kv_cache(self) -> None:
        for key_scale, value_scale in list(self._kv_scale_caches.values()):
            self._free_kv_cache_tensor(key_scale)
            self._free_kv_cache_tensor(value_scale)
        self._kv_scale_caches.clear()
        super().close_kv_cache()

    def _kv_cache_runtime_name(self) -> str:
        if self._compiled.prefill.runtime_name != self._compiled.decode.runtime_name:
            raise ValueError(
                "device-side KV cache requires prefill and decode to use the same L2 runtime: "
                f"{self._compiled.prefill.runtime_name!r} != {self._compiled.decode.runtime_name!r}"
            )
        return self._compiled.prefill.runtime_name

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> WorkerTensor:
        """Allocate one KV cache tensor on the L2 NPU worker."""
        worker = self._worker_for_runtime(self._kv_cache_runtime_name())
        return worker.alloc_tensor(shape, dtype)

    def _free_kv_cache_tensor(self, tensor: WorkerTensor) -> None:
        """Free one KV cache tensor from the L2 NPU worker."""
        worker = self._l2_workers.get(self._kv_cache_runtime_name())
        if worker is not None and worker.initialized:
            worker.free_tensor(tensor)

    @staticmethod
    def _validate_kv_cache_bounds(
        model: RuntimeModel,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache: WorkerTensor,
    ) -> None:
        """Fail on host before an invalid KV page id reaches the NPU kernel."""
        valid_blocks = block_table[block_table >= 0]
        valid_slots = slot_mapping[slot_mapping >= 0]
        if valid_blocks.numel() == 0 and valid_slots.numel() == 0:
            return
        max_block_id = int(valid_blocks.max().item()) if valid_blocks.numel() else -1
        max_slot_block = int(valid_slots.max().item()) // model.runtime.page_size if valid_slots.numel() else -1
        max_page_id = max(max_block_id, max_slot_block)
        rows_per_layer = cache.shape[0] // model.config.num_hidden_layers
        max_pages = rows_per_layer // (model.config.num_key_value_heads * model.runtime.page_size)
        if max_page_id >= max_pages:
            raise RuntimeError(
                "KV cache page id exceeds runner device cache capacity: "
                f"max_page_id={max_page_id}, max_pages={max_pages}, "
                f"cache_shape={cache.shape}, block_table_shape={tuple(block_table.shape)}, "
                f"slot_mapping_shape={tuple(slot_mapping.shape)}"
            )

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run the JIT all-layer prefill kernel and return next-token logits."""
        compiled = self._compiled
        prefill_inputs = self._prepare_prefill_inputs(model, batch)
        dw = compiled.decode_weights

        kv_cache = self._kv_caches.get(model.config.model_id)
        if kv_cache is None:
            raise RuntimeError(f"KV cache for model {model.config.model_id!r} is not initialized")
        k_cache = kv_cache.key_pages
        v_cache = kv_cache.value_pages
        self._validate_kv_cache_bounds(model, prefill_inputs.block_table, prefill_inputs.slot_mapping, k_cache)
        kv_scales = self._kv_scale_caches.get(model.config.model_id)
        logits_padded = torch.zeros(
            (prefill_inputs.actual_batch, compiled.padded_vocab),
            dtype=torch.float32,
        ).share_memory_()

        if kv_scales is None:
            raise RuntimeError(f"missing A8W8 KV scale cache for model {model.config.model_id!r}")
        k_cache_scale, v_cache_scale = kv_scales
        rows_per_layer = k_cache.shape[0] // model.config.num_hidden_layers
        hidden = prefill_inputs.hidden

        def weight_slice(name: str, start: int, layers: int, rows_per_layer_: int = 1) -> WorkerTensor:
            tensor = dw[name][start * rows_per_layer_ : (start + layers) * rows_per_layer_]
            return self._l2_child_tensor(compiled.prefill.runtime_name, tensor)

        for layer_start in range(0, model.config.num_hidden_layers, _QWEN14B_A8W8_PREFILL_CHUNK_LAYERS):
            layer_count = min(
                _QWEN14B_A8W8_PREFILL_CHUNK_LAYERS,
                model.config.num_hidden_layers - layer_start,
            )
            cache_row_start = layer_start * rows_per_layer
            cache_rows = layer_count * rows_per_layer
            hidden_out = torch.empty_like(hidden).share_memory_()
            scratch_logits = torch.empty_like(logits_padded).share_memory_()
            self._run_l2_program(
                compiled.prefill,
                hidden,
                prefill_inputs.seq_lens,
                prefill_inputs.chunk_lens,
                prefill_inputs.chunk_offsets,
                weight_slice("decode_input_rms_weight", layer_start, layer_count),
                weight_slice("decode_wq", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_wk", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_wv", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_wq_scale", layer_start, layer_count),
                weight_slice("decode_wk_scale", layer_start, layer_count),
                weight_slice("decode_wv_scale", layer_start, layer_count),
                weight_slice("decode_q_norm_weight", layer_start, layer_count),
                weight_slice("decode_k_norm_weight", layer_start, layer_count),
                self._l2_child_tensor(compiled.prefill.runtime_name, compiled.rope_cos),
                self._l2_child_tensor(compiled.prefill.runtime_name, compiled.rope_sin),
                prefill_inputs.block_table,
                prefill_inputs.slot_mapping,
                self._worker_tensor_view(
                    k_cache,
                    cache_row_start * model.config.head_dim,
                    (cache_rows, model.config.head_dim),
                    1,
                ),
                self._worker_tensor_view(
                    v_cache,
                    cache_row_start * model.config.head_dim,
                    (cache_rows, model.config.head_dim),
                    1,
                ),
                self._worker_tensor_view(k_cache_scale, cache_row_start * 8, (cache_rows, 8), 4),
                self._worker_tensor_view(v_cache_scale, cache_row_start * 8, (cache_rows, 8), 4),
                weight_slice("decode_wo", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_wo_scale", layer_start, layer_count),
                weight_slice("decode_post_rms_weight", layer_start, layer_count),
                weight_slice("decode_w_gate", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_w_up", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_w_down", layer_start, layer_count, model.config.intermediate_size),
                self._l2_child_tensor(compiled.prefill.runtime_name, compiled.final_norm_weight),
                self._l2_child_tensor(compiled.prefill.runtime_name, compiled.padded_lm_head_weight),
                scratch_logits,
                hidden_out,
            )
            hidden = hidden_out
        logits_padded = self._project_logits_host(model, compiled, prefill_inputs, hidden)

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
        return PrefillResult(
            last_hidden=None,
            logits=logits_padded[:, : model.config.vocab_size],
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the fused all-layer PAGED ``decode_layer.decode_fwd`` and return logits.

        ``decode_fwd`` runs all NUM_LAYERS + the LM head in one dispatch over the
        PAGED KV pool, addressing KV via ``block_table`` + ``slot_mapping`` — the
        SAME device-resident KV pool prefill writes (``self._kv_caches``), so prompt
        KV is already in place with no bridge. KV is keyed by block_table page id, not by
        kernel row, so a request may occupy any row each step (no stable-slot shim).

        The kernel is FIXED-BATCH (it computes all max_batch_size rows and writes
        each row's current-token KV). Pad the active batch up to the kernel batch by
        REPLICATING active row 0's inputs into the padding rows: those rows then
        recompute row 0's K/V and write row 0's own slot with byte-identical values
        (an idempotent, safe write), and their logits are trimmed off below. This
        avoids padded rows clobbering an unrelated request's physical page.
        """
        compiled = self._compiled
        model_id = model.config.model_id
        decode_inputs = self._prepare_decode_inputs(model, batch)
        actual_batch = decode_inputs.actual_batch
        dw = compiled.decode_weights
        rt = compiled.decode.runtime_name
        kernel_batch = model.runtime.max_batch_size
        max_blocks = self._max_blocks_per_seq(model)

        kv_cache = self._kv_caches.get(model_id)
        if kv_cache is None:
            raise RuntimeError(f"KV cache for model {model_id!r} is not initialized")
        k_cache = kv_cache.key_pages
        v_cache = kv_cache.value_pages
        kv_scales = self._kv_scale_caches.get(model_id)

        if kernel_batch > compiled.decode_logits_buffer.shape[0]:
            raise ValueError(
                f"kernel batch {kernel_batch} exceeds logits buffer batch "
                f"{compiled.decode_logits_buffer.shape[0]}"
            )

        # Pad active inputs up to the fixed kernel batch by replicating row 0.
        def _pad_rows(active: torch.Tensor, rows_each: int) -> torch.Tensor:
            view = active.reshape(actual_batch, rows_each)
            padded = view[0:1].expand(kernel_batch - actual_batch, rows_each)
            return torch.cat([view, padded], dim=0).reshape(-1).contiguous()

        hidden = torch.zeros((kernel_batch, model.config.hidden_size), dtype=torch.bfloat16)
        hidden[:actual_batch] = decode_inputs.hidden
        hidden[actual_batch:] = decode_inputs.hidden[0:1]
        hidden = hidden.share_memory_()
        seq_lens = _pad_rows(decode_inputs.seq_lens, 1).to(torch.int32).share_memory_()
        block_table = _pad_rows(decode_inputs.block_table, max_blocks).to(torch.int32).share_memory_()
        slot_mapping = _pad_rows(decode_inputs.slot_mapping, 1).to(torch.int32).share_memory_()

        # Padded block_table / slot_mapping only ever reference row 0's
        # already-valid pages, so bound-check exactly what the kernel will read.
        self._validate_kv_cache_bounds(model, block_table, slot_mapping, k_cache)

        logits_padded = compiled.decode_logits_buffer  # full [kernel_batch, vocab]; trimmed below
        if kv_scales is None:
            raise RuntimeError(f"missing A8W8 KV scale cache for model {model_id!r}")
        k_cache_scale, v_cache_scale = kv_scales
        self._run_l2_program(
            compiled.decode,
            hidden,
            self._l2_child_tensor(rt, dw["decode_input_rms_weight"]),
            self._l2_child_tensor(rt, dw["decode_wq"]),
            self._l2_child_tensor(rt, dw["decode_wk"]),
            self._l2_child_tensor(rt, dw["decode_wv"]),
            self._l2_child_tensor(rt, dw["decode_wq_scale"]),
            self._l2_child_tensor(rt, dw["decode_wk_scale"]),
            self._l2_child_tensor(rt, dw["decode_wv_scale"]),
            self._l2_child_tensor(rt, dw["decode_q_norm_weight"]),
            self._l2_child_tensor(rt, dw["decode_k_norm_weight"]),
            seq_lens,
            block_table,
            slot_mapping,
            self._l2_child_tensor(rt, compiled.rope_cos),
            self._l2_child_tensor(rt, compiled.rope_sin),
            k_cache,
            v_cache,
            k_cache_scale,
            v_cache_scale,
            self._l2_child_tensor(rt, dw["decode_wo"]),
            self._l2_child_tensor(rt, dw["decode_wo_scale"]),
            self._l2_child_tensor(rt, dw["decode_w_gate"]),
            self._l2_child_tensor(rt, dw["decode_w_up"]),
            self._l2_child_tensor(rt, dw["decode_w_down"]),
            self._l2_child_tensor(rt, dw["decode_post_rms_weight"]),
            self._l2_child_tensor(rt, compiled.final_norm_weight),
            self._l2_child_tensor(rt, compiled.padded_lm_head_weight),
            logits_padded,
        )
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        return DecodeResult(
            hidden_states=decode_inputs.hidden.float(),
            logits=logits_padded[:actual_batch, : model.config.vocab_size].to(decode_inputs.hidden.device),
        )

    def _project_logits_host(
        self,
        model: RuntimeModel,
        compiled: _CompiledKernels,
        prefill_inputs: _PrefillInputs,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Project final prefill hidden states on host for A8W8 chunked prefill.

        The A8W8 prefill kernel is currently chunked by transformer layers and
        returns the final hidden states. Keep the final RMSNorm + LM head here as
        an explicit fallback until the prefill kernel owns that projection end to
        end, so the performance tradeoff is visible at the serving boundary.
        """
        final_hidden = torch.zeros((prefill_inputs.actual_batch, model.config.hidden_size), dtype=torch.bfloat16)
        for batch_idx in range(prefill_inputs.actual_batch):
            chunk_offset = int(prefill_inputs.chunk_offsets[batch_idx].item())
            chunk_len = int(prefill_inputs.chunk_lens[batch_idx].item())
            final_hidden[batch_idx] = hidden[chunk_offset + chunk_len - 1]

        gamma = compiled.final_norm_weight.view(1, -1).float()
        x = final_hidden.float()
        normed = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + model.config.rms_norm_eps) * gamma).to(
            torch.bfloat16
        )
        logits = torch.empty((prefill_inputs.actual_batch, compiled.padded_vocab), dtype=torch.float32).share_memory_()
        lm_head = compiled.padded_lm_head_weight
        for row_start in range(0, compiled.padded_vocab, _QWEN14B_LM_HEAD_CHUNK_ROWS):
            row_end = min(row_start + _QWEN14B_LM_HEAD_CHUNK_ROWS, compiled.padded_vocab)
            logits[:, row_start:row_end] = normed.float() @ lm_head[row_start:row_end].float().T
        return logits

    @staticmethod
    def _worker_tensor_view(
        tensor: WorkerTensor,
        element_offset: int,
        shape: tuple[int, ...],
        element_size: int,
    ) -> WorkerTensor:
        """Return a contiguous WorkerTensor view starting at an element offset."""
        return WorkerTensor(
            data_ptr=tensor.data_ptr + element_offset * element_size,
            shape=shape,
            dtype=tensor.dtype,
        )

    def _run_l2_program(self, callable_spec: _L2Callable, *args: Any) -> Any:
        """Run a compiled non-L3 program through the LLM Simpler worker."""
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        span_args = {
            "kernel": callable_spec.name,
            "runtime": callable_spec.runtime_name,
            "block_dim": callable_spec.block_dim,
            "aicpu_thread_num": callable_spec.aicpu_thread_num,
        }
        with profile_span(
            _l2_trace_name(callable_spec.name),
            cat="kernel",
            level="kernel",
            args=span_args,
        ):
            handle = self._ensure_l2_program(callable_spec)
            orch_args = self._build_l2_orch_args(callable_spec, args)

            cfg = CallConfig()
            cfg.block_dim = callable_spec.block_dim
            cfg.aicpu_thread_num = callable_spec.aicpu_thread_num

            worker = self._l2_workers[handle.runtime_name]
            timing = worker.run(handle.callable_id, orch_args, cfg)
            _add_run_timing_args(span_args, timing)
            return timing

    def _worker_for_runtime(self, runtime_name: str) -> LlmWorker:
        """Return an initialized worker for ``runtime_name``."""
        worker = self._l2_workers.get(runtime_name)
        if worker is not None:
            return worker
        worker = LlmWorker(
            level=2,
            platform=self._platform,
            runtime=runtime_name,
            device_id=self._device_id,
            auto_init=True,
        )
        self._l2_workers[runtime_name] = worker
        return worker

    def _ensure_l2_program(self, callable_spec: _L2Callable) -> _L2ProgramHandle:
        """Register and cache one executor-assembled non-L3 callable."""
        key = id(callable_spec)
        cached = self._l2_programs.get(key)
        if cached is not None:
            return cached

        worker = self._worker_for_runtime(callable_spec.runtime_name)

        handle = _L2ProgramHandle(
            callable_id=worker.register(callable_spec.chip_callable),
            runtime_name=callable_spec.runtime_name,
        )
        self._l2_programs[key] = handle
        return handle

    def _l2_child_tensor(
        self,
        runtime_name: str,
        tensor: torch.Tensor,
        *,
        upload: bool = True,
        refresh: bool = False,
    ) -> WorkerTensor:
        """Return a worker-resident view for a CPU tensor's backing storage."""
        from simpler_setup.torch_interop import torch_dtype_to_datatype  # noqa: PLC0415

        if tensor.device.type != "cpu":
            raise ValueError("child-memory tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("child-memory tensor must be contiguous")
        tensor = self._share_cpu_tensor(tensor)
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
        tensor_offset = int(tensor.data_ptr()) - storage_ptr
        if tensor_offset < 0 or tensor_offset + int(tensor.nbytes) > storage_nbytes:
            raise ValueError("tensor view is outside its backing storage")

        key = (runtime_name, storage_ptr)
        alloc = self._l2_child_allocs.get(key)
        if alloc is None:
            worker = self._worker_for_runtime(runtime_name)
            dev_ptr = worker.malloc(storage_nbytes)
            if upload:
                worker.copy_to(dev_ptr, storage_ptr, storage_nbytes)
            alloc = (dev_ptr, storage_nbytes)
            self._l2_child_allocs[key] = alloc
        elif upload and refresh:
            worker = self._worker_for_runtime(runtime_name)
            worker.copy_to(alloc[0], storage_ptr, storage_nbytes)

        dev_base, _ = alloc
        shape = tuple(int(dim) for dim in tensor.shape)
        return WorkerTensor(
            data_ptr=dev_base + tensor_offset,
            shape=shape,
            dtype=torch_dtype_to_datatype(tensor.dtype),
        )

    def _release_l2_child_allocs(self, runtime_name: str) -> None:
        """Free cached child-memory allocations for one L2 runtime."""
        worker = self._l2_workers.get(runtime_name)
        for key, (dev_ptr, _nbytes) in list(self._l2_child_allocs.items()):
            key_runtime, _storage_ptr = key
            if key_runtime != runtime_name:
                continue
            if worker is not None and worker.initialized:
                worker.free(dev_ptr)
            self._l2_child_allocs.pop(key, None)

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor's storage to shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    def close(self) -> None:
        """Release non-L3 child-memory allocations and L2 workers."""
        self.close_kv_cache()
        for (runtime_name, _), (dev_ptr, _nbytes) in list(self._l2_child_allocs.items()):
            worker = self._l2_workers.get(runtime_name)
            if worker is not None and worker.initialized:
                worker.free(dev_ptr)
        self._l2_child_allocs.clear()
        self._l2_programs.clear()
        for worker in self._l2_workers.values():
            worker.close()
        self._l2_workers.clear()

    @staticmethod
    def _build_l2_orch_args(callable_spec: _L2Callable, args: tuple[Any, ...]):
        """Build ``ChipStorageTaskArgs`` for a compiled L2 program call."""
        from simpler.task_interface import ChipStorageTaskArgs, scalar_to_uint64  # noqa: PLC0415
        try:
            from simpler.task_interface import ContinuousTensor  # noqa: PLC0415
        except ImportError:
            from simpler.task_interface import Tensor as ContinuousTensor  # noqa: PLC0415
        from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

        param_infos = callable_spec.param_infos
        if len(args) != len(param_infos):
            names = [p.name for p in param_infos]
            raise TypeError(
                f"compiled program expects {len(param_infos)} arguments, got {len(args)}. Parameters: {names}"
            )

        orch_args = ChipStorageTaskArgs()
        for info, arg in zip(param_infos, args, strict=True):
            if info.shape is None:
                if not isinstance(arg, ctypes._SimpleCData):
                    raise TypeError(f"scalar parameter {info.name!r} must be passed as a ctypes scalar")
                orch_args.add_scalar(scalar_to_uint64(arg))
                continue
            if isinstance(arg, WorkerTensor):
                orch_args.add_tensor(arg.to_continuous_tensor())
                continue
            if isinstance(arg, ContinuousTensor):
                orch_args.add_tensor(arg)
                continue
            if not isinstance(arg, torch.Tensor):
                raise TypeError(f"tensor parameter {info.name!r} expects torch.Tensor, got {type(arg).__name__}")
            if arg.device.type != "cpu":
                raise ValueError(f"tensor parameter {info.name!r} must be on CPU for Simpler L2 dispatch")
            if not arg.is_contiguous():
                raise ValueError(f"tensor parameter {info.name!r} must be contiguous")
            if not arg.is_shared():
                arg.share_memory_()
            orch_args.add_tensor(make_tensor_arg(arg))
        return orch_args

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
    ) -> _PrefillInputs:
        """Pack variable-length prefill requests into kernel input tensors."""
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        max_seq = model.runtime.max_seq_len
        hidden_size = model.config.hidden_size
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)

        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        chunk_lens = torch.empty((actual_batch,), dtype=torch.int32)
        chunk_offsets = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        seq_len_values = [int(batch.seq_lens[idx].item()) for idx in range(actual_batch)]
        chunk_len_values: list[int] = []
        chunk_start_values: list[int] = []
        for batch_idx, seq_len in enumerate(seq_len_values):
            if batch.positions is not None:
                row_positions = batch.positions[batch_idx].detach().cpu()
                valid_positions = row_positions[row_positions >= 0]
                if valid_positions.numel() == 0:
                    raise ValueError("prefill positions must include at least one chunk token")
                chunk_start = int(valid_positions[0].item())
                chunk_len = int(valid_positions.numel())
                expected_positions = torch.arange(
                    chunk_start,
                    chunk_start + chunk_len,
                    dtype=valid_positions.dtype,
                )
                if not torch.equal(valid_positions, expected_positions):
                    raise ValueError(
                        "prefill batch.positions must form one contiguous chunk: "
                        f"chunk_start={chunk_start}, chunk_len={chunk_len}, seq_len={seq_len}"
                    )
            else:
                chunk_len = seq_len
                chunk_start = 0
            if chunk_len <= 0:
                raise ValueError("prefill chunk_lens must be positive")
            if chunk_start + chunk_len != seq_len:
                raise ValueError(
                    "prefill chunk must end at seq_len: "
                    f"chunk_start={chunk_start}, chunk_len={chunk_len}, seq_len={seq_len}"
                )
            chunk_len_values.append(chunk_len)
            chunk_start_values.append(chunk_start)
        total_tokens = sum(chunk_len_values)
        hidden = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16)
        slot_mapping = torch.empty((total_tokens,), dtype=torch.int32)

        token_offset = 0
        for batch_idx in range(actual_batch):
            alloc = batch.kv_allocations[batch_idx] if batch_idx < len(batch.kv_allocations) else None
            seq_len = seq_len_values[batch_idx]
            if seq_len <= 0:
                raise ValueError("prefill seq_lens must be positive")
            if seq_len > max_seq:
                raise ValueError(f"prefill seq_len {seq_len} exceeds max_seq_len {max_seq}")
            seq_lens[batch_idx] = seq_len
            chunk_len = chunk_len_values[batch_idx]
            chunk_start = chunk_start_values[batch_idx]
            chunk_lens[batch_idx] = chunk_len
            chunk_offsets[batch_idx] = token_offset
            embeddings = batch.input_embeddings[batch_idx, :chunk_len, :].to(torch.bfloat16).cpu()
            hidden[token_offset : token_offset + chunk_len, :] = embeddings

            if alloc is not None:
                page_ids = alloc.page_ids
            elif batch_idx < len(batch.block_ids):
                page_ids = batch.block_ids[batch_idx]
            else:
                page_ids = []
            self._write_block_table_row(block_table, batch_idx, max_blocks, page_ids)

            slot_row = self._compute_slot_mapping(page_ids, chunk_len, page_size, start_pos=chunk_start)
            slot_mapping[token_offset : token_offset + chunk_len] = slot_row
            token_offset += chunk_len

        return _PrefillInputs(
            actual_batch=actual_batch,
            hidden=hidden.share_memory_(),
            seq_lens=seq_lens.share_memory_(),
            chunk_lens=chunk_lens.share_memory_(),
            chunk_offsets=chunk_offsets.share_memory_(),
            block_table=block_table.share_memory_(),
            slot_mapping=slot_mapping.share_memory_(),
        )

    def _prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> _DecodeInputs:
        """Pack active decode requests into fused decode-kernel inputs."""
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        hidden_size = model.config.hidden_size
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)

        hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        slot_mapping = torch.empty((actual_batch,), dtype=torch.int32)

        for batch_idx in range(actual_batch):
            alloc = batch.kv_allocations[batch_idx] if batch_idx < len(batch.kv_allocations) else None
            seq_len = int(batch.seq_lens[batch_idx].item())
            if seq_len <= 0:
                raise ValueError("decode seq_lens must be positive")
            if seq_len > model.runtime.max_seq_len:
                raise ValueError(
                    f"decode seq_len {seq_len} exceeds max_seq_len {model.runtime.max_seq_len}"
                )
            hidden[batch_idx, :] = batch.hidden_states[batch_idx].to(torch.bfloat16).cpu()
            seq_lens[batch_idx] = seq_len

            if alloc is not None:
                page_ids = alloc.page_ids
            elif batch_idx < len(batch.block_ids):
                page_ids = batch.block_ids[batch_idx]
            else:
                page_ids = []
            self._write_block_table_row(block_table, batch_idx, max_blocks, page_ids)

            tokens_used = seq_len - 1
            page_idx = tokens_used // page_size
            offset = tokens_used % page_size
            slot_mapping[batch_idx] = page_ids[page_idx] * page_size + offset

        return _DecodeInputs(
            actual_batch=actual_batch,
            hidden=hidden.share_memory_(),
            seq_lens=seq_lens.share_memory_(),
            block_table=block_table.share_memory_(),
            slot_mapping=slot_mapping.share_memory_(),
        )
