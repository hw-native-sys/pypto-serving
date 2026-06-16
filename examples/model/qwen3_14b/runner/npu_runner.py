# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from pypto.runtime import DeviceTensor

from python.core.model_runner import ModelRunner
from python.core.types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from python.profile import profile_span


def _kernel_trace_name(kernel_name: str) -> str:
    if "prefill" in kernel_name:
        return "kernel.prefill_fwd"
    if "decode" in kernel_name:
        return "kernel.decode_fwd"
    return f"kernel.{kernel_name}"


def _run_timing_us(timing: Any) -> tuple[float | None, float | None]:
    if timing is None:
        return None, None
    host_wall_us = getattr(timing, "host_wall_us", None)
    device_wall_us = getattr(timing, "device_wall_us", None)
    if host_wall_us is not None:
        host_wall_us = float(host_wall_us)
    if device_wall_us is not None:
        device_wall_us = float(device_wall_us)
    return host_wall_us, device_wall_us


def _add_run_timing_args(args: dict[str, Any], timing: Any) -> None:
    host_wall_us, device_wall_us = _run_timing_us(timing)
    if host_wall_us is not None:
        args["host_wall_us"] = host_wall_us
        args["host_wall_ms"] = host_wall_us / 1000.0
    if device_wall_us is not None:
        args["device_wall_us"] = device_wall_us
        args["device_wall_ms"] = device_wall_us / 1000.0


@dataclass
class _L3Callable:
    """HOST-dispatched compiled program and launch metadata."""

    compiled: object
    name: str
    block_dim: int
    aicpu_thread_num: int
    dispatch_args: tuple[Any, ...] = ()


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
    prefill_hidden_buffer: torch.Tensor
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
    decode_logits_buffer: torch.Tensor
    # TurboQuant (TQ) artifacts. Populated only when tq_mode=True.
    tq_mode: bool = False
    rot_matrices: torch.Tensor | None = None  # [num_layers*head_dim, head_dim] BF16
    tq_codebook: torch.Tensor | None = None   # [1, 16] FP32 Lloyd-Max centroids


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
    """Active user rows prepared for decode."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeKernelInputs:
    """Fixed-batch tensors passed to the fused decode kernel."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    logits: torch.Tensor


@dataclass
class _StaticDeviceTensor:
    """A shared host tensor to upload into the shared L3 worker once."""

    tensor: torch.Tensor


@dataclass
class _StaticKernelArgs:
    """Static worker-resident kernel arguments reused across dispatches."""

    final_norm_weight: _StaticDeviceTensor
    rope_cos: _StaticDeviceTensor
    rope_sin: _StaticDeviceTensor
    padded_lm_head_weight: _StaticDeviceTensor
    decode_weights: dict[str, _StaticDeviceTensor]
    # TurboQuant static artifacts (None unless tq_mode).
    rot_matrices: _StaticDeviceTensor | None = None
    tq_codebook: _StaticDeviceTensor | None = None


class Qwen314BModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    def __init__(
        self,
        *,
        compiled: _CompiledKernels,
        tq_mode: bool = False,
    ) -> None:
        super().__init__()
        self._compiled = compiled
        self._tq_mode = tq_mode
        self._l3_worker: Any | None = None
        self._l3_static_tensors: dict[tuple[int, tuple[int, ...], torch.dtype], object] = {}
        self._static_args: _StaticKernelArgs | None = None
        self._pending_kv_cache_specs: dict[str, tuple[ModelConfig, RuntimeConfig]] = {}
        if compiled is not None:
            self._share_static_kernel_tensors()
            self._static_args = self._build_static_kernel_args()

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> None:
        """Create the L3 worker-resident cache before the first request."""
        if model_id in self._kv_caches:
            return
        self._pending_kv_cache_specs[model_id] = (config, runtime)
        with profile_span("Qwen314BModelRunner.prepare_l3_worker", cat="executor"):
            self._shared_l3_worker()
        with profile_span("Qwen314BModelRunner.upload_static_tensors", cat="executor"):
            self._materialize_static_tensors()
        with profile_span("Qwen314BModelRunner.init_kv_cache", cat="executor"):
            ModelRunner.init_kv_cache(self, model_id, config, runtime)

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceTensor:
        """Allocate one worker-resident KV cache tensor shared by prefill/decode."""
        return self._shared_l3_worker().alloc_tensor(shape, dtype)

    def _free_kv_cache_tensor(self, tensor: DeviceTensor) -> None:
        """Release one worker-resident KV cache tensor."""
        worker = self._l3_worker
        if worker is not None:
            worker.free_tensor(tensor)

    def _materialize_kv_cache(self, model: RuntimeModel) -> Any:
        """Return the worker-resident KV cache, allocating only as a fallback."""
        kv_cache = self._kv_caches.get(model.config.model_id)
        if kv_cache is not None:
            return kv_cache
        spec = self._pending_kv_cache_specs.get(model.config.model_id)
        if spec is None:
            spec = (model.config, model.runtime)
            self._pending_kv_cache_specs[model.config.model_id] = spec
        ModelRunner.init_kv_cache(self, model.config.model_id, spec[0], spec[1])
        return self._kv_caches[model.config.model_id]

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

    def _share_static_kernel_tensors(self) -> None:
        """Move static kernel inputs to shared memory before worker creation."""
        for tensor in self._iter_static_host_tensors():
            self._share_cpu_tensor(tensor)

    def _iter_static_host_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return host tensors that must be shared before the worker forks."""
        compiled = self._compiled
        tensors: tuple[torch.Tensor, ...] = (
            compiled.final_norm_weight,
            compiled.rope_cos,
            compiled.rope_sin,
            compiled.padded_lm_head_weight,
            *compiled.decode_weights.values(),
            compiled.prefill_hidden_buffer,
            compiled.prefill_seq_lens_buffer,
            compiled.prefill_chunk_lens_buffer,
            compiled.prefill_chunk_offsets_buffer,
            compiled.prefill_block_table_buffer,
            compiled.prefill_slot_mapping_buffer,
            compiled.prefill_logits_buffer,
            compiled.decode_hidden_buffer,
            compiled.decode_seq_lens_buffer,
            compiled.decode_block_table_buffer,
            compiled.decode_slot_mapping_buffer,
            compiled.decode_logits_buffer,
        )
        # TurboQuant rotation matrices + codebook are also static kernel inputs.
        if compiled.rot_matrices is not None:
            tensors += (compiled.rot_matrices,)
        if compiled.tq_codebook is not None:
            tensors += (compiled.tq_codebook,)
        return tensors

    def _build_static_kernel_args(self) -> _StaticKernelArgs:
        """Create static device-upload markers once per runner."""
        compiled = self._compiled
        return _StaticKernelArgs(
            final_norm_weight=self._static_device_tensor(compiled.final_norm_weight),
            rope_cos=self._static_device_tensor(compiled.rope_cos),
            rope_sin=self._static_device_tensor(compiled.rope_sin),
            padded_lm_head_weight=self._static_device_tensor(compiled.padded_lm_head_weight),
            decode_weights={
                name: self._static_device_tensor(tensor)
                for name, tensor in compiled.decode_weights.items()
            },
            rot_matrices=(
                self._static_device_tensor(compiled.rot_matrices)
                if compiled.rot_matrices is not None
                else None
            ),
            tq_codebook=(
                self._static_device_tensor(compiled.tq_codebook)
                if compiled.tq_codebook is not None
                else None
            ),
        )

    def _require_static_args(self) -> _StaticKernelArgs:
        """Return prebuilt static args for dispatch."""
        if self._static_args is None:
            raise RuntimeError("Qwen314BModelRunner static kernel args are not initialized")
        return self._static_args

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run the JIT all-layer prefill kernel and return next-token logits."""
        compiled = self._compiled
        prefill_inputs = self._prepare_prefill_inputs(model, batch)

        logits_padded = compiled.prefill_logits_buffer

        kv_cache = self._materialize_kv_cache(model)
        if self._tq_mode:
            quant_k = kv_cache.quant_k_pages
            quant_v = kv_cache.quant_v_pages
            k_scales = kv_cache.k_scales_pages
            v_scales = kv_cache.v_scales_pages
            if quant_k is None or quant_v is None or k_scales is None or v_scales is None:
                raise RuntimeError("TurboQuant KV caches are not initialized")
            self._validate_kv_cache_bounds(
                model, prefill_inputs.block_table, prefill_inputs.slot_mapping, quant_k
            )
            self._run_distributed_program(
                compiled.prefill,
                *self._prefill_tq_kernel_args(
                    prefill_inputs, quant_k, quant_v, k_scales, v_scales, logits_padded
                ),
            )
        else:
            k_cache = kv_cache.key_pages
            v_cache = kv_cache.value_pages
            self._validate_kv_cache_bounds(
                model, prefill_inputs.block_table, prefill_inputs.slot_mapping, k_cache
            )
            self._run_distributed_program(
                compiled.prefill,
                *self._prefill_kernel_args(prefill_inputs, k_cache, v_cache, logits_padded),
            )

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
        return PrefillResult(
            last_hidden=None,
            logits=logits_padded[: prefill_inputs.actual_batch, : model.config.vocab_size],
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

        kv_cache = self._kv_caches.get(model_id)
        if kv_cache is None:
            raise RuntimeError(f"KV cache for model {model_id!r} is not initialized")

        kernel_inputs = self._pad_decode_inputs(model, decode_inputs)

        if self._tq_mode:
            quant_k = kv_cache.quant_k_pages
            quant_v = kv_cache.quant_v_pages
            k_scales = kv_cache.k_scales_pages
            v_scales = kv_cache.v_scales_pages
            if quant_k is None or quant_v is None or k_scales is None or v_scales is None:
                raise RuntimeError("TurboQuant KV caches are not initialized")
            # Padded block_table / slot_mapping only ever reference row 0's
            # already-valid pages, so bound-check exactly what the kernel will read.
            self._validate_kv_cache_bounds(
                model, kernel_inputs.block_table, kernel_inputs.slot_mapping, quant_k
            )
            self._run_distributed_program(
                compiled.decode,
                *self._decode_tq_kernel_args(kernel_inputs, quant_k, quant_v, k_scales, v_scales),
            )
        else:
            k_cache = kv_cache.key_pages
            v_cache = kv_cache.value_pages
            # Padded block_table / slot_mapping only ever reference row 0's
            # already-valid pages, so bound-check exactly what the kernel will read.
            self._validate_kv_cache_bounds(
                model, kernel_inputs.block_table, kernel_inputs.slot_mapping, k_cache
            )
            self._run_distributed_program(
                compiled.decode,
                *self._decode_kernel_args(kernel_inputs, k_cache, v_cache),
            )
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        return DecodeResult(
            hidden_states=decode_inputs.hidden.float(),
            logits=kernel_inputs.logits[: kernel_inputs.actual_batch, : model.config.vocab_size].to(
                decode_inputs.hidden.device
            ),
        )

    def _prefill_kernel_args(
        self,
        inputs: _PrefillInputs,
        k_cache: DeviceTensor,
        v_cache: DeviceTensor,
        logits: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_prefill_host`` signature order."""
        static = self._require_static_args()
        weights = static.decode_weights
        return (
            inputs.hidden,
            inputs.seq_lens,
            inputs.chunk_lens,
            inputs.chunk_offsets,
            weights["decode_input_rms_weight"],
            weights["decode_wq"],
            weights["decode_wk"],
            weights["decode_wv"],
            weights["decode_q_norm_weight"],
            weights["decode_k_norm_weight"],
            static.rope_cos,
            static.rope_sin,
            inputs.block_table,
            inputs.slot_mapping,
            k_cache,
            v_cache,
            weights["decode_wo"],
            weights["decode_post_rms_weight"],
            weights["decode_w_gate"],
            weights["decode_w_up"],
            weights["decode_w_down"],
            static.final_norm_weight,
            static.padded_lm_head_weight,
            logits,
        )

    def _decode_kernel_args(
        self,
        inputs: _DecodeKernelInputs,
        k_cache: DeviceTensor,
        v_cache: DeviceTensor,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_decode_host`` signature order."""
        static = self._require_static_args()
        weights = static.decode_weights
        return (
            inputs.hidden,
            weights["decode_input_rms_weight"],
            weights["decode_wq"],
            weights["decode_wk"],
            weights["decode_wv"],
            weights["decode_q_norm_weight"],
            weights["decode_k_norm_weight"],
            inputs.seq_lens,
            inputs.block_table,
            inputs.slot_mapping,
            static.rope_cos,
            static.rope_sin,
            k_cache,
            v_cache,
            weights["decode_wo"],
            weights["decode_w_gate"],
            weights["decode_w_up"],
            weights["decode_w_down"],
            weights["decode_post_rms_weight"],
            static.final_norm_weight,
            static.padded_lm_head_weight,
            inputs.logits,
        )

    def _prefill_tq_kernel_args(
        self,
        inputs: _PrefillInputs,
        quant_k: DeviceTensor,
        quant_v: DeviceTensor,
        k_scales: DeviceTensor,
        v_scales: DeviceTensor,
        logits: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_prefill_tq_host`` signature order.

        TQ prefill drops chunk_lens/chunk_offsets and replaces the BF16 k/v cache
        with UINT8 quantized caches + FP32 scales + rotation matrices + codebook.
        """
        static = self._require_static_args()
        if static.rot_matrices is None or static.tq_codebook is None:
            raise RuntimeError("TurboQuant rotation matrices / codebook are not compiled")
        weights = static.decode_weights
        return (
            inputs.hidden,
            inputs.seq_lens,
            weights["decode_input_rms_weight"],
            weights["decode_wq"],
            weights["decode_wk"],
            weights["decode_wv"],
            weights["decode_q_norm_weight"],
            weights["decode_k_norm_weight"],
            static.rope_cos,
            static.rope_sin,
            inputs.block_table,
            inputs.slot_mapping,
            quant_k,
            quant_v,
            k_scales,
            v_scales,
            static.rot_matrices,
            static.tq_codebook,
            weights["decode_wo"],
            weights["decode_post_rms_weight"],
            weights["decode_w_gate"],
            weights["decode_w_up"],
            weights["decode_w_down"],
            static.final_norm_weight,
            static.padded_lm_head_weight,
            logits,
        )

    def _decode_tq_kernel_args(
        self,
        inputs: _DecodeKernelInputs,
        quant_k: DeviceTensor,
        quant_v: DeviceTensor,
        k_scales: DeviceTensor,
        v_scales: DeviceTensor,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_decode_tq_host`` signature order.

        TQ decode replaces the BF16 k/v cache with the six TQ tensors. Weight
        order is wo, post_rms, w_gate, w_up, w_down (differs from non-TQ decode).
        """
        static = self._require_static_args()
        if static.rot_matrices is None or static.tq_codebook is None:
            raise RuntimeError("TurboQuant rotation matrices / codebook are not compiled")
        weights = static.decode_weights
        return (
            inputs.hidden,
            weights["decode_input_rms_weight"],
            weights["decode_wq"],
            weights["decode_wk"],
            weights["decode_wv"],
            weights["decode_q_norm_weight"],
            weights["decode_k_norm_weight"],
            inputs.seq_lens,
            inputs.block_table,
            inputs.slot_mapping,
            static.rope_cos,
            static.rope_sin,
            quant_k,
            quant_v,
            k_scales,
            v_scales,
            static.rot_matrices,
            static.tq_codebook,
            weights["decode_wo"],
            weights["decode_post_rms_weight"],
            weights["decode_w_gate"],
            weights["decode_w_up"],
            weights["decode_w_down"],
            static.final_norm_weight,
            static.padded_lm_head_weight,
            inputs.logits,
        )

    def _pad_decode_inputs(self, model: RuntimeModel, inputs: _DecodeInputs) -> _DecodeKernelInputs:
        """Pad active decode rows to the fixed kernel batch.

        The fused decode kernel computes all ``max_batch_size`` rows. Inactive
        rows replicate row 0 so their KV writes are idempotent instead of
        targeting unrelated pages.
        """
        compiled = self._compiled
        actual_batch = inputs.actual_batch
        kernel_batch = model.runtime.max_batch_size
        max_blocks = self._max_blocks_per_seq(model)

        if kernel_batch > compiled.decode_logits_buffer.shape[0]:
            raise ValueError(
                f"kernel batch {kernel_batch} exceeds logits buffer batch "
                f"{compiled.decode_logits_buffer.shape[0]}"
            )

        hidden = compiled.decode_hidden_buffer
        hidden[:actual_batch].copy_(inputs.hidden)
        if actual_batch < kernel_batch:
            hidden[actual_batch:].copy_(inputs.hidden[0:1].expand(kernel_batch - actual_batch, -1))

        return _DecodeKernelInputs(
            actual_batch=actual_batch,
            hidden=hidden,
            seq_lens=self._copy_replicated_rows(
                compiled.decode_seq_lens_buffer,
                inputs.seq_lens,
                actual_batch,
                kernel_batch,
                rows_each=1,
            ),
            block_table=self._copy_replicated_rows(
                compiled.decode_block_table_buffer,
                inputs.block_table,
                actual_batch,
                kernel_batch,
                rows_each=max_blocks,
            ),
            slot_mapping=self._copy_replicated_rows(
                compiled.decode_slot_mapping_buffer,
                inputs.slot_mapping,
                actual_batch,
                kernel_batch,
                rows_each=1,
            ),
            logits=compiled.decode_logits_buffer,
        )

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
            l3_args = callable_spec.dispatch_args + tuple(self._coerce_l3_arg(worker, arg) for arg in args)
            worker_run_args = dict(span_args)
            with profile_span(
                f"{_kernel_trace_name(callable_spec.name)}.worker_run",
                cat="kernel",
                level="kernel",
                args=worker_run_args,
            ):
                timing = worker.run(callable_spec.compiled, *l3_args)
                _add_run_timing_args(worker_run_args, timing)
            _add_run_timing_args(span_args, timing)
            return timing

    def _shared_l3_worker(self) -> Any:
        """Return the worker shared by the generation prefill/decode path."""
        worker = self._l3_worker
        if worker is None:
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            worker = DistributedWorker([self._compiled.prefill.compiled, self._compiled.decode.compiled])
            self._l3_worker = worker
        return worker

    def _coerce_l3_arg(self, worker: Any, arg: Any) -> Any:
        """Convert static upload markers to worker-resident tensors."""
        if not isinstance(arg, _StaticDeviceTensor):
            return arg
        tensor = arg.tensor
        key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        cached = self._l3_static_tensors.get(key)
        if cached is not None:
            return cached
        dev = worker.alloc_tensor(tensor.shape, tensor.dtype, init=tensor)
        self._l3_static_tensors[key] = dev
        return dev

    def _materialize_static_tensors(self) -> None:
        """Upload static kernel tensors into the shared L3 worker before serving."""
        worker = self._shared_l3_worker()
        static = self._require_static_args()
        for arg in (
            static.final_norm_weight,
            static.rope_cos,
            static.rope_sin,
            static.padded_lm_head_weight,
            *static.decode_weights.values(),
        ):
            self._coerce_l3_arg(worker, arg)
        # TurboQuant rotation matrices + codebook are also uploaded once.
        if static.rot_matrices is not None:
            self._coerce_l3_arg(worker, static.rot_matrices)
        if static.tq_codebook is not None:
            self._coerce_l3_arg(worker, static.tq_codebook)

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
    def _static_device_tensor(tensor: torch.Tensor) -> _StaticDeviceTensor:
        """Mark a CPU tensor for one-time upload to the shared worker."""
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        return _StaticDeviceTensor(Qwen314BModelRunner._share_cpu_tensor(tensor))

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor's storage to shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    def close(self) -> None:
        """Release shared L3 worker resources and clear static tensor caches."""
        try:
            self.close_kv_cache()
        finally:
            worker = self._l3_worker
            try:
                if worker is not None:
                    worker.close()
            finally:
                self._l3_worker = None
                self._l3_static_tensors.clear()

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
    ) -> _PrefillInputs:
        """Pack variable-length prefill requests into kernel input tensors."""
        compiled = self._compiled
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        max_seq = model.runtime.max_seq_len
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)
        kernel_batch = model.runtime.max_batch_size

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
        max_tokens = kernel_batch * max_seq
        if total_tokens > max_tokens:
            raise ValueError(f"prefill total tokens {total_tokens} exceeds kernel capacity {max_tokens}")

        hidden = compiled.prefill_hidden_buffer[:total_tokens]
        seq_lens = compiled.prefill_seq_lens_buffer
        chunk_lens = compiled.prefill_chunk_lens_buffer
        chunk_offsets = compiled.prefill_chunk_offsets_buffer
        block_table = compiled.prefill_block_table_buffer
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
            hidden=hidden,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    @staticmethod
    def _compute_slot_mapping(
        page_ids: list[int],
        num_tokens: int,
        page_size: int,
        *,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Return physical slot indices for token positions start_pos..start_pos+num_tokens-1."""
        mapping = torch.empty((num_tokens,), dtype=torch.int32)
        if num_tokens > 0:
            max_pos = start_pos + num_tokens - 1
            max_page_idx = max_pos // page_size
            if max_page_idx >= len(page_ids):
                raise ValueError(
                    f"page_ids list length {len(page_ids)} is too small for position {max_pos}; "
                    f"need at least {max_page_idx + 1} pages"
                )
        for offset_idx in range(num_tokens):
            pos = start_pos + offset_idx
            page_idx = pos // page_size
            offset = pos % page_size
            mapping[offset_idx] = page_ids[page_idx] * page_size + offset
        return mapping

    @staticmethod
    def _write_block_table_row(
        block_table: torch.Tensor,
        batch_idx: int,
        max_blocks: int,
        page_ids: list[int],
    ) -> None:
        """Write one request's KV page IDs into a flat block table."""
        row_start = batch_idx * max_blocks
        if page_ids:
            block_table[row_start : row_start + len(page_ids)] = torch.tensor(
                page_ids,
                dtype=torch.int32,
            )

    @staticmethod
    def _validate_batch_size(
        model: RuntimeModel,
        actual_batch: int,
    ) -> int:
        """Validate and return the actual user batch size."""
        if actual_batch <= 0:
            raise ValueError("batch must contain at least one request")
        if actual_batch > model.runtime.max_batch_size:
            max_batch_size = model.runtime.max_batch_size
            raise ValueError(
                f"batch has {actual_batch} requests, but runtime max_batch_size is {max_batch_size}"
            )
        return actual_batch

    @staticmethod
    def _max_blocks_per_seq(model: RuntimeModel) -> int:
        """Return the maximum KV pages one sequence can occupy."""
        return (model.runtime.max_seq_len + model.runtime.page_size - 1) // model.runtime.page_size
