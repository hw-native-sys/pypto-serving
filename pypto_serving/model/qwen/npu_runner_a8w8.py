# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
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
from pypto_serving.model.qwen.npu_runner import (
    Qwen314BModelRunner,
    _L3Callable,
    _add_run_timing_args,
)
from pypto_serving.tools.profile import profile_span

_QWEN14B_A8W8_PREFILL_CHUNK_LAYERS = 10
_QWEN14B_LM_HEAD_CHUNK_ROWS = 8192


def _kernel_trace_name(kernel_name: str) -> str:
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
class _CompiledKernels:
    """Compiled Qwen3-14B kernels and immutable runtime tensors."""

    prefill: _L3Callable
    decode: _L3Callable
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    decode_weights: dict[str, torch.Tensor]
    decode_logits_buffer: torch.Tensor
    prefill_hidden_buffer: torch.Tensor
    prefill_next_hidden_buffer: torch.Tensor
    prefill_seq_lens_buffer: torch.Tensor
    prefill_chunk_lens_buffer: torch.Tensor
    prefill_chunk_offsets_buffer: torch.Tensor
    prefill_block_table_buffer: torch.Tensor
    prefill_slot_mapping_buffer: torch.Tensor
    prefill_logits_buffer: torch.Tensor
    decode_hidden_buffer: torch.Tensor
    decode_seq_lens_buffer: torch.Tensor
    decode_block_table_buffer: torch.Tensor
    decode_slot_mapping_buffer: torch.Tensor


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


class Qwen314BA8W8ModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    _compute_slot_mapping = staticmethod(Qwen314BModelRunner._compute_slot_mapping)
    _write_block_table_row = staticmethod(Qwen314BModelRunner._write_block_table_row)
    _validate_batch_size = staticmethod(Qwen314BModelRunner._validate_batch_size)
    _max_blocks_per_seq = staticmethod(Qwen314BModelRunner._max_blocks_per_seq)

    def __init__(
        self,
        *,
        compiled: _CompiledKernels | None,
    ) -> None:
        super().__init__()
        self._compiled = compiled
        self._l3_worker: Any | None = None
        self._l3_static_storage_tensors: dict[int, object] = {}
        self._l3_static_host_tensors: dict[int, torch.Tensor] = {}
        self._kv_scale_caches: dict[str, tuple[Any, Any]] = {}
        if compiled is not None:
            self._register_l3_static_host_tensors()

    def init_kv_cache(self, model_id: str, config, runtime) -> int:
        """Create the runner-owned KV cache, plus INT8 scale pages for A8W8."""
        self._l3_log("init_kv_cache: preparing DistributedWorker")
        self._shared_l3_worker()
        self._l3_log("init_kv_cache: DistributedWorker ready")
        self._l3_log("init_kv_cache: allocating KV cache")
        num_pages = super().init_kv_cache(model_id, config, runtime)
        self._l3_log(f"init_kv_cache: KV cache pages={num_pages}")
        if model_id in self._kv_scale_caches:
            return num_pages
        cache_rows = config.num_hidden_layers * num_pages * config.num_key_value_heads * runtime.page_size
        self._l3_log("init_kv_cache: allocating KV scale cache")
        self._l3_log("init_kv_cache: allocating key scale")
        key_scale = self._alloc_kv_cache_tensor((cache_rows, 8), torch.float32)
        self._l3_log("init_kv_cache: key scale allocated")
        try:
            self._l3_log("init_kv_cache: allocating value scale")
            value_scale = self._alloc_kv_cache_tensor((cache_rows, 8), torch.float32)
            self._l3_log("init_kv_cache: value scale allocated")
        except Exception:
            self._free_kv_cache_tensor(key_scale)
            raise
        self._kv_scale_caches[model_id] = (key_scale, value_scale)
        self._l3_log("init_kv_cache: scale cache ready")
        return num_pages

    def close_kv_cache(self) -> None:
        for key_scale, value_scale in list(self._kv_scale_caches.values()):
            self._free_kv_cache_tensor(key_scale)
            self._free_kv_cache_tensor(value_scale)
        self._kv_scale_caches.clear()
        super().close_kv_cache()

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> Any:
        """Allocate one KV cache tensor on the active NPU worker."""
        return self._shared_l3_worker().alloc_tensor(shape, dtype)

    def _free_kv_cache_tensor(self, tensor: Any) -> None:
        """Free one KV cache tensor from the active NPU worker."""
        worker = self._l3_worker
        if worker is not None:
            worker.free_tensor(tensor)

    @staticmethod
    def _validate_kv_cache_bounds(
        model: RuntimeModel,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache: Any,
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
        self._l3_log("run_prefill: start")
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

        if kv_scales is None:
            raise RuntimeError(f"missing A8W8 KV scale cache for model {model.config.model_id!r}")
        k_cache_scale, v_cache_scale = kv_scales
        rows_per_layer = k_cache.shape[0] // model.config.num_hidden_layers
        hidden = prefill_inputs.hidden

        def weight_slice(name: str, start: int, layers: int, rows_per_layer_: int = 1) -> Any:
            tensor = dw[name][start * rows_per_layer_ : (start + layers) * rows_per_layer_]
            return self._kernel_static_tensor(tensor)

        for layer_start in range(0, model.config.num_hidden_layers, _QWEN14B_A8W8_PREFILL_CHUNK_LAYERS):
            layer_count = min(
                _QWEN14B_A8W8_PREFILL_CHUNK_LAYERS,
                model.config.num_hidden_layers - layer_start,
            )
            cache_row_start = layer_start * rows_per_layer
            cache_rows = layer_count * rows_per_layer
            base_hidden = compiled.prefill_hidden_buffer
            next_hidden = compiled.prefill_next_hidden_buffer
            hidden_out = (
                next_hidden[: hidden.shape[0]]
                if hidden.data_ptr() == base_hidden.data_ptr()
                else base_hidden[: hidden.shape[0]]
            )
            scratch_logits = compiled.prefill_logits_buffer[: prefill_inputs.actual_batch]
            self._l3_log(f"run_prefill: dispatch layers {layer_start}-{layer_start + layer_count - 1}")
            self._run_distributed_program(
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
                self._kernel_static_tensor(compiled.rope_cos),
                self._kernel_static_tensor(compiled.rope_sin),
                prefill_inputs.block_table,
                prefill_inputs.slot_mapping,
                self._device_tensor_view(
                    k_cache,
                    cache_row_start * model.config.head_dim,
                    (cache_rows, model.config.head_dim),
                    1,
                ),
                self._device_tensor_view(
                    v_cache,
                    cache_row_start * model.config.head_dim,
                    (cache_rows, model.config.head_dim),
                    1,
                ),
                self._device_tensor_view(k_cache_scale, cache_row_start * 8, (cache_rows, 8), 4),
                self._device_tensor_view(v_cache_scale, cache_row_start * 8, (cache_rows, 8), 4),
                weight_slice("decode_wo", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_wo_scale", layer_start, layer_count),
                weight_slice("decode_post_rms_weight", layer_start, layer_count),
                weight_slice("decode_w_gate", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_w_up", layer_start, layer_count, model.config.hidden_size),
                weight_slice("decode_w_down", layer_start, layer_count, model.config.intermediate_size),
                self._kernel_static_tensor(compiled.final_norm_weight),
                self._kernel_static_tensor(compiled.padded_lm_head_weight),
                scratch_logits,
                hidden_out,
            )
            self._l3_log(f"run_prefill: layers {layer_start}-{layer_start + layer_count - 1} done")
            hidden = hidden_out
        logits_padded = self._project_logits_host(model, compiled, prefill_inputs, hidden)
        self._l3_log("run_prefill: done")

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
        self._l3_log("run_decode: start")
        compiled = self._compiled
        model_id = model.config.model_id
        decode_inputs = self._prepare_decode_inputs(model, batch)
        actual_batch = decode_inputs.actual_batch
        dw = compiled.decode_weights
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

        hidden = compiled.decode_hidden_buffer
        hidden[:actual_batch].copy_(decode_inputs.hidden)
        hidden[actual_batch:].copy_(decode_inputs.hidden[0:1])
        seq_lens = self._copy_replicated_rows(
            compiled.decode_seq_lens_buffer,
            decode_inputs.seq_lens,
            actual_batch,
            kernel_batch,
            rows_each=1,
        )
        block_table = self._copy_replicated_rows(
            compiled.decode_block_table_buffer,
            decode_inputs.block_table,
            actual_batch,
            kernel_batch,
            rows_each=max_blocks,
        )
        slot_mapping = self._copy_replicated_rows(
            compiled.decode_slot_mapping_buffer,
            decode_inputs.slot_mapping,
            actual_batch,
            kernel_batch,
            rows_each=1,
        )

        # Padded block_table / slot_mapping only ever reference row 0's
        # already-valid pages, so bound-check exactly what the kernel will read.
        self._validate_kv_cache_bounds(model, block_table, slot_mapping, k_cache)

        logits_padded = compiled.decode_logits_buffer  # full [kernel_batch, vocab]; trimmed below
        if kv_scales is None:
            raise RuntimeError(f"missing A8W8 KV scale cache for model {model_id!r}")
        k_cache_scale, v_cache_scale = kv_scales
        self._run_distributed_program(
            compiled.decode,
            hidden,
            self._kernel_static_tensor(dw["decode_input_rms_weight"]),
            self._kernel_static_tensor(dw["decode_wq"]),
            self._kernel_static_tensor(dw["decode_wk"]),
            self._kernel_static_tensor(dw["decode_wv"]),
            self._kernel_static_tensor(dw["decode_wq_scale"]),
            self._kernel_static_tensor(dw["decode_wk_scale"]),
            self._kernel_static_tensor(dw["decode_wv_scale"]),
            self._kernel_static_tensor(dw["decode_q_norm_weight"]),
            self._kernel_static_tensor(dw["decode_k_norm_weight"]),
            seq_lens,
            block_table,
            slot_mapping,
            self._kernel_static_tensor(compiled.rope_cos),
            self._kernel_static_tensor(compiled.rope_sin),
            k_cache,
            v_cache,
            k_cache_scale,
            v_cache_scale,
            self._kernel_static_tensor(dw["decode_wo"]),
            self._kernel_static_tensor(dw["decode_wo_scale"]),
            self._kernel_static_tensor(dw["decode_w_gate"]),
            self._kernel_static_tensor(dw["decode_w_up"]),
            self._kernel_static_tensor(dw["decode_w_down"]),
            self._kernel_static_tensor(dw["decode_post_rms_weight"]),
            self._kernel_static_tensor(compiled.final_norm_weight),
            self._kernel_static_tensor(compiled.padded_lm_head_weight),
            logits_padded,
        )
        self._l3_log("run_decode: done")
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
        logits = compiled.prefill_logits_buffer[: prefill_inputs.actual_batch]
        lm_head = compiled.padded_lm_head_weight
        for row_start in range(0, compiled.padded_vocab, _QWEN14B_LM_HEAD_CHUNK_ROWS):
            row_end = min(row_start + _QWEN14B_LM_HEAD_CHUNK_ROWS, compiled.padded_vocab)
            logits[:, row_start:row_end] = normed.float() @ lm_head[row_start:row_end].float().T
        return logits

    def _device_tensor_view(
        self,
        tensor: Any,
        element_offset: int,
        shape: tuple[int, ...],
        element_size: int,
    ) -> Any:
        """Return a contiguous DeviceTensor view for the L3 worker."""
        from pypto.runtime import DeviceTensor  # noqa: PLC0415

        return DeviceTensor(
            data_ptr=tensor.data_ptr + element_offset * element_size,
            shape=shape,
            dtype=tensor.dtype,
        )

    def _kernel_static_tensor(self, tensor: torch.Tensor) -> Any:
        """Return a backend-resident static tensor argument."""
        return self._l3_static_storage_view(tensor)

    def _run_distributed_program(self, callable_spec: _L3Callable, *args: Any) -> Any:
        """Run a compiled HOST wrapper through the shared PyPTO L3 worker."""
        span_args = {
            "kernel": callable_spec.name,
            "block_dim": callable_spec.block_dim,
            "aicpu_thread_num": callable_spec.aicpu_thread_num,
        }
        with profile_span(
            _kernel_trace_name(callable_spec.name),
            cat="kernel",
            level="kernel",
            args=span_args,
        ):
            worker = self._shared_l3_worker()
            l3_args = callable_spec.dispatch_args + args
            timing = worker.run(callable_spec.compiled, *l3_args)
            _add_run_timing_args(span_args, timing)
            return timing

    def _shared_l3_worker(self) -> Any:
        """Return the L3 worker shared by A8W8 prefill/decode."""
        worker = self._l3_worker
        if worker is None:
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            worker = DistributedWorker([
                self._compiled.prefill.compiled,
                self._compiled.decode.compiled,
            ])
            self._l3_worker = worker
            self._l3_log("DistributedWorker constructed")
        return worker

    def _register_l3_static_host_tensors(self) -> None:
        """Register full host storages that L3 static views can share."""
        compiled = self._compiled
        for tensor in (
            compiled.final_norm_weight,
            compiled.rope_cos,
            compiled.rope_sin,
            compiled.padded_lm_head_weight,
            *compiled.decode_weights.values(),
        ):
            self._register_l3_static_host_tensor(tensor)

    def _register_l3_static_host_tensor(self, tensor: torch.Tensor) -> None:
        """Remember a full shared host tensor by storage pointer."""
        if tensor.device.type != "cpu":
            raise ValueError("L3 static host tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("L3 static host tensor must be contiguous")
        tensor = self._share_cpu_tensor(tensor)
        self._l3_static_host_tensors[int(tensor.untyped_storage().data_ptr())] = tensor

    def _l3_static_storage_view(self, tensor: torch.Tensor) -> Any:
        """Upload a full static storage once and return a DeviceTensor view."""
        from pypto.runtime import DeviceTensor  # noqa: PLC0415

        if tensor.device.type != "cpu":
            raise ValueError("L3 static tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("L3 static tensor view must be contiguous")
        tensor = self._share_cpu_tensor(tensor)
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        host_full = self._l3_static_host_tensors.get(storage_ptr)
        if host_full is None:
            self._register_l3_static_host_tensor(tensor)
            host_full = tensor

        dev_full = self._l3_static_storage_tensors.get(storage_ptr)
        if dev_full is None:
            worker = self._shared_l3_worker()
            dev_full = worker.alloc_tensor(host_full.shape, host_full.dtype, init=host_full)
            self._l3_static_storage_tensors[storage_ptr] = dev_full

        byte_offset = int(tensor.data_ptr()) - storage_ptr
        if byte_offset < 0 or byte_offset + int(tensor.nbytes) > int(storage.nbytes()):
            raise ValueError("L3 static tensor view is outside its backing storage")
        return DeviceTensor(
            data_ptr=dev_full.data_ptr + byte_offset,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=tensor.dtype,
        )

    @staticmethod
    def _l3_log(message: str) -> None:
        """Print L3 diagnostics only when explicitly requested."""
        if os.environ.get("QWEN_A8W8_L3_DEBUG") == "1":
            print(f"[a8w8-l3] {message}", flush=True)

    @staticmethod
    def _copy_replicated_rows(
        dst: torch.Tensor,
        active: torch.Tensor,
        actual_batch: int,
        kernel_batch: int,
        *,
        rows_each: int,
    ) -> torch.Tensor:
        """Copy active rows and fill inactive rows by replicating row 0."""
        active_view = active.reshape(actual_batch, rows_each)
        dst_view = dst.reshape(kernel_batch, rows_each)
        dst_view[:actual_batch].copy_(active_view)
        if actual_batch < kernel_batch:
            dst_view[actual_batch:].copy_(active_view[0:1].expand(kernel_batch - actual_batch, rows_each))
        return dst

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor's storage to shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    def close(self) -> None:
        """Release A8W8 L3 worker resources."""
        self.close_kv_cache()
        worker = self._l3_worker
        if worker is not None:
            worker.close()
        self._l3_worker = None
        self._l3_static_storage_tensors.clear()

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
    ) -> _PrefillInputs:
        """Pack variable-length prefill requests into kernel input tensors."""
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        max_seq = model.runtime.max_seq_len
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)

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
        compiled = self._compiled
        hidden_buffer = compiled.prefill_hidden_buffer
        if total_tokens > hidden_buffer.shape[0]:
            raise ValueError(f"prefill total tokens {total_tokens} exceeds L3 buffer {hidden_buffer.shape[0]}")
        hidden = hidden_buffer[:total_tokens]
        seq_lens = compiled.prefill_seq_lens_buffer[:actual_batch]
        chunk_lens = compiled.prefill_chunk_lens_buffer[:actual_batch]
        chunk_offsets = compiled.prefill_chunk_offsets_buffer[:actual_batch]
        block_table = compiled.prefill_block_table_buffer[: actual_batch * max_blocks]
        slot_mapping = compiled.prefill_slot_mapping_buffer[:total_tokens]
        seq_lens.zero_()
        chunk_lens.zero_()
        chunk_offsets.zero_()
        block_table.fill_(-1)

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
            hidden=hidden,
            seq_lens=seq_lens,
            chunk_lens=chunk_lens,
            chunk_offsets=chunk_offsets,
            block_table=block_table,
            slot_mapping=slot_mapping,
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
