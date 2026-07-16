# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from pypto.runtime import DeviceTensor

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.tools.profile import profile_span


logger = logging.getLogger(__name__)


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
    greedy_sample: _L3Callable
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    padded_embed_weight: torch.Tensor
    decode_weights: dict[str, torch.Tensor]
    prefill_token_ids_buffer: torch.Tensor
    prefill_seq_lens_buffer: torch.Tensor
    prefill_chunk_lens_buffer: torch.Tensor
    prefill_chunk_offsets_buffer: torch.Tensor
    prefill_block_table_buffer: torch.Tensor
    prefill_slot_mapping_buffer: torch.Tensor
    prefill_logits_buffer: torch.Tensor
    prefill_sampled_ids_buffer: torch.Tensor
    prefill_next_hidden_buffer: torch.Tensor
    decode_seq_lens_buffer: torch.Tensor
    decode_block_table_buffer: torch.Tensor
    decode_slot_mapping_buffer: torch.Tensor
    decode_logits_buffer: torch.Tensor
    decode_token_ids_buffer: torch.Tensor
    decode_sampled_ids_buffer: torch.Tensor
    decode_next_hidden_buffer: torch.Tensor


@dataclass
class _PrefillInputs:
    """Host tensors passed to the prefill kernel."""

    actual_batch: int
    token_ids: torch.Tensor
    seq_lens: torch.Tensor
    chunk_lens: torch.Tensor
    chunk_offsets: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeInputs:
    """Active user rows prepared for decode."""

    actual_batch: int
    token_ids: torch.Tensor
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeKernelInputs:
    """Fixed-batch tensors passed to the fused decode kernel."""

    actual_batch: int
    token_ids: torch.Tensor
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
    padded_embed_weight: _StaticDeviceTensor
    decode_weights: dict[str, _StaticDeviceTensor]


class Qwen314BModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    def __init__(
        self,
        *,
        compiled: _CompiledKernels,
        device_id: int = 0,
    ) -> None:
        super().__init__()
        self._compiled = compiled
        self._device_id = device_id
        self._l3_worker: Any | None = None
        self._l3_static_tensors: dict[tuple[int, tuple[int, ...], torch.dtype], object] = {}
        # Device-resident decode output scratch (greedy path): allocated directly on
        # the worker (no host copy) so the per-step memset + D2H copy-back of the
        # max-batch logits/next_hidden vanish.
        self._decode_logits_dev_tensor: DeviceTensor | None = None
        self._decode_next_hidden_dev_tensor: DeviceTensor | None = None
        self._static_args: _StaticKernelArgs | None = None
        self._pending_kv_cache_specs: dict[str, tuple[ModelConfig, RuntimeConfig]] = {}
        if compiled is not None:
            self._share_static_kernel_tensors()
            self._static_args = self._build_static_kernel_args()

    #: Scratch KV pages for the profile pass — slot=-1 means only page 0
    #: is ever touched (reads via block_table=0, writes via slot clamp to 0).
    _PROFILE_PAGES = 1

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> int:
        """Create the L3 worker-resident cache before the first request.

        Order (vLLM-style): run a profile warmup FIRST so the simpler arena
        is allocated before the KV cache competes for HBM.  The profile uses
        ``slot_mapping=-1`` / ``block_table=0`` so only a single dummy page
        is needed.  The KV cache size is then computed by the estimation
        formula and allocated into the remaining space; if allocation fails
        the page count is halved and retried.
        """
        if model_id in self._kv_caches:
            num_pages = self._kv_caches[model_id].key_pages.shape[0] // (
                config.num_hidden_layers * config.num_key_value_heads * runtime.page_size
            )
            return num_pages
        self._pending_kv_cache_specs[model_id] = (config, runtime)

        logger.info("[init_kv_cache] creating L3 worker …")
        with profile_span("Qwen314BModelRunner.prepare_l3_worker", cat="executor"):
            self._shared_l3_worker()

        logger.info("[init_kv_cache] uploading static tensors …")
        with profile_span("Qwen314BModelRunner.upload_static_tensors", cat="executor"):
            self._materialize_static_tensors()
            # Reserve the device-resident decode output scratch (logits + next_hidden)
            # now, before the KV-cache sizing below measures peak_non_kv, so its ~10MB
            # is counted in the memory budget instead of being lazily allocated on the
            # first greedy step and eating into the runtime safety margin.
            self._decode_logits_device_arg()
            self._decode_next_hidden_device_arg()

        # -- phase 1: profile warmup → arena allocated ----------------------
        # Uses slot_mapping=-1 so no real KV cache pages are needed; the
        # 1-page scratch is the dummy target for all reads/writes.
        logger.info(f"[init_kv_cache] profile warmup (scratch {self._PROFILE_PAGES} page) …")
        ModelRunner.init_kv_cache(self, model_id, config, runtime, num_pages=self._PROFILE_PAGES)
        try:
            self._warmup_dispatch(runtime)
        finally:
            self.close_kv_cache()
            self._kv_caches.pop(model_id, None)

        # -- phase 2: real KV cache, halve-and-retry on OOM -----------------
        logger.info("[init_kv_cache] computing KV cache pages …")
        num_pages = self._compute_kv_cache_pages(config, runtime, self._device_id)
        num_pages = self._alloc_kv_cache_with_retry(model_id, config, runtime, num_pages)
        self._print_memory_breakdown("after KV cache alloc", config, runtime, num_pages, self._device_id)
        logger.info("[init_kv_cache] done")
        return num_pages

    def _alloc_kv_cache_with_retry(
        self, model_id: str, config: ModelConfig, runtime: RuntimeConfig, num_pages: int,
    ) -> int:
        """Allocate the KV cache, halving the page count on OOM."""
        floor = max(runtime.max_batch_size, 1)
        requested = num_pages
        num_pages = max(num_pages, floor)  # always try at least the floor
        while num_pages >= floor:
            try:
                logger.info(f"[init_kv_cache] num_pages={num_pages}, allocating …")
                ModelRunner.init_kv_cache(self, model_id, config, runtime, num_pages=num_pages)
                bytes_per_page = (
                    config.num_hidden_layers * 2 * config.num_key_value_heads
                    * runtime.page_size * config.head_dim
                    * getattr(torch, runtime.kv_dtype).itemsize
                )
                logger.info(
                    f"[init_kv_cache] allocated {num_pages} pages "
                    f"(requested {requested}, downgraded after OOM): "
                    f"{num_pages * bytes_per_page / 1e9:.2f} GB KV cache, "
                    f"{num_pages * runtime.page_size} context tokens",
                    
                )
                return num_pages
            except (RuntimeError, MemoryError) as e:
                prev = num_pages
                num_pages //= 2
                if num_pages < floor and prev > floor:
                    num_pages = floor
                logger.info(
                    f"[init_kv_cache] alloc failed ({e}); retrying {prev} -> {num_pages}",
                )
        raise RuntimeError(
            f"KV cache allocation failed even at floor {floor} pages"
        )

    @staticmethod
    def _compute_kv_cache_pages(config: ModelConfig, runtime: RuntimeConfig, device_id: int = 0) -> int:
        """Compute KV cache pages, vLLM-style: total x utilization − peak_non_kv.

        Called AFTER the profile warm-up, so weights, the simpler ring-heap
        arena, compiled buffers and any persistent scratch are already
        allocated — ``peak_non_kv = total − free`` captures all of it. The KV
        budget is ``total x utilization − peak_non_kv``, leaving
        ``total x (1 − utilization)`` as a fixed absolute headroom (more robust
        than ``free x fraction`` whose headroom shrinks when free is small).
        """
        free_bytes, total_bytes = torch.npu.mem_get_info(f"npu:{device_id}")
        dtype_bytes = getattr(torch, runtime.kv_dtype).itemsize
        bytes_per_page = (
            config.num_hidden_layers * 2 * config.num_key_value_heads
            * runtime.page_size * config.head_dim * dtype_bytes
        )
        utilization = getattr(runtime, "npu_memory_utilization", 0.90)
        peak_non_kv = total_bytes - free_bytes
        kv_budget = int(total_bytes * utilization - peak_non_kv)
        num_pages = max(kv_budget // bytes_per_page, 1)
        logger.info(
            "KV cache sizing (vLLM-style): total=%.2f GB, utilization=%.2f, "
            "peak_non_kv=%.2f GB, kv_budget=%.2f GB, requested_pages=%d (%.1f MB/page)",
            total_bytes / 1e9, utilization, peak_non_kv / 1e9,
            kv_budget / 1e9, num_pages, bytes_per_page / 1e6,
        )
        return num_pages

    @staticmethod
    def _print_memory_breakdown(
        label: str, config: ModelConfig, runtime: RuntimeConfig, num_pages: int,
        device_id: int = 0,
    ) -> None:
        """Print a per-component NPU memory breakdown at ``label``.

        ``torch.npu.mem_get_info`` only reports a single total, so each part
        is reconstructed rather than queried: weights (estimated from the
        model config), KV cache (exact = num_pages x bytes_per_page), simpler
        ring-heap arena (from the ``PTO2_RING_HEAP`` env x 4), and the
        residual (compiled buffers + transient activation scratch + overhead).
        """
        free_bytes, total_bytes = torch.npu.mem_get_info(f"npu:{device_id}")
        used_bytes = total_bytes - free_bytes
        dtype_bytes = getattr(torch, runtime.kv_dtype).itemsize

        # Weights — GQA: Q/O are hiddenxhidden, K/V are hiddenxkv_hidden.
        hidden = config.hidden_size
        kv_hidden = config.num_key_value_heads * config.head_dim
        wt_params = (
            config.num_hidden_layers * (
                hidden * hidden * 2
                + hidden * kv_hidden * 2
                + hidden * config.intermediate_size * 3
                + hidden * 4
            )
            + config.vocab_size * hidden
        )
        weight_bytes = int(wt_params * dtype_bytes)

        # KV cache — exact (num_pages already reflects the real allocation).
        bytes_per_page = (
            config.num_hidden_layers * 2 * config.num_key_value_heads
            * runtime.page_size * config.head_dim * dtype_bytes
        )
        kv_bytes = num_pages * bytes_per_page

        # Simpler ring-heap arena — from env (matches _compute_kv_cache_pages).
        ring_heap = int(os.environ.get("PTO2_RING_HEAP", 256 * 1024 * 1024))
        arena_bytes = ring_heap * 4 + 128 * 1024 * 1024

        residual = used_bytes - weight_bytes - kv_bytes - arena_bytes

        logger.info(f"[mem-breakdown] {label}:")
        logger.info(
            f"  total used (measured):      {used_bytes / 1e9:7.2f} GB "
            f"/ {total_bytes / 1e9:.2f} GB (free {free_bytes / 1e9:.2f} GB)",
            
        )
        logger.info(f"  ├─ weights (estimated):     {weight_bytes / 1e9:7.2f} GB")
        kv_tokens = num_pages * runtime.page_size
        max_seq_len = runtime.max_seq_len
        worst_case_demand = runtime.max_batch_size * max_seq_len
        max_len_reqs = kv_tokens // max(max_seq_len, 1)
        logger.info(
            f"  ├─ KV cache ({num_pages} pages):     {kv_bytes / 1e9:7.2f} GB "
            f"({bytes_per_page / 1e6:.1f} MB/page)",
            
        )
        logger.info(
            f"  │     capacity = {kv_tokens} tokens "
            f"≈ {max_len_reqs} x full-len({max_seq_len}) reqs; "
            f"worst-case need {runtime.max_batch_size}x{max_seq_len}="
            f"{worst_case_demand} tokens"
            + ("  [OK]" if kv_tokens >= worst_case_demand else "  [TIGHT]"),
            
        )
        logger.info(f"  ├─ simpler arena (env x 4): {arena_bytes / 1e9:7.2f} GB")
        logger.info(
            f"  └─ residual (buffers/scratch): {residual / 1e9:6.2f} GB "
            f"(compiled buffers + transient activation scratch + overhead)",
            
        )
        logger.info(
            "  note: weights/arena are estimates, KV is exact; total is from "
            "mem_get_info (may under-count simpler's rtMalloc pool).",
            
        )

    def warmup(self, model: RuntimeModel) -> None:
        """Dispatch a dummy prefill + decode through the L3 worker."""
        self._warmup_dispatch(model.runtime)

    def _warmup_dispatch(self, runtime: RuntimeConfig) -> None:
        """Production-scale prefill + decode warm-up with slot_mapping=-1.

        Sizes the prefill to one serving scheduling step — total tokens =
        ``max_num_batched_tokens`` spread across ``max_batch`` requests.
        This deliberately exercises the kernel at the configured capacity so
        that a too-large ``max_num_batched_tokens`` (which would hit the
        single-die attention heap ceiling around seq≈415 in the 40-layer
        fused prefill) fails at startup rather than on the first real
        request.
        """
        kernel_batch = runtime.max_batch_size
        max_seq = runtime.max_seq_len
        batch, per_req, total_tokens = self._warmup_prefill_shape(runtime)
        mnb = runtime.max_num_batched_tokens
        long_prefill_threshold = runtime.long_prefill_token_threshold

        logger.info(
            f"[warmup] starting (prefill_batch={batch}, kernel_batch={kernel_batch}, "
            f"max_num_batched_tokens={mnb}, "
            f"long_prefill_token_threshold={long_prefill_threshold}, "
            f"max_seq={max_seq}, per_req={per_req}, total_tokens={total_tokens}, slot=-1)",
            
        )
        compiled = self._compiled
        kv_cache = list(self._kv_caches.values())[0]

        # -- prefill ---------------------------------------------------------
        compiled.prefill_token_ids_buffer[:total_tokens].zero_()
        compiled.prefill_seq_lens_buffer.zero_()
        compiled.prefill_chunk_lens_buffer.zero_()
        compiled.prefill_chunk_offsets_buffer.zero_()
        compiled.prefill_block_table_buffer.fill_(0)    # all reads from page 0
        compiled.prefill_slot_mapping_buffer.fill_(-1)  # all writes to page 0

        token_offset = 0
        for b in range(batch):
            compiled.prefill_seq_lens_buffer[b] = per_req
            compiled.prefill_chunk_lens_buffer[b] = per_req
            compiled.prefill_chunk_offsets_buffer[b] = token_offset
            token_offset += per_req

        prefill_inputs = _PrefillInputs(
            actual_batch=batch,
            token_ids=compiled.prefill_token_ids_buffer[:total_tokens],
            seq_lens=compiled.prefill_seq_lens_buffer,
            chunk_lens=compiled.prefill_chunk_lens_buffer,
            chunk_offsets=compiled.prefill_chunk_offsets_buffer,
            block_table=compiled.prefill_block_table_buffer,
            slot_mapping=compiled.prefill_slot_mapping_buffer,
        )

        logger.info(f"[warmup] prefill dispatch … (batch={batch}, tokens={total_tokens})")
        t0 = time.perf_counter()
        self._run_distributed_program(
            compiled.prefill,
            *self._prefill_kernel_args(
                prefill_inputs, kv_cache.key_pages, kv_cache.value_pages,
                compiled.prefill_logits_buffer,
            ),
        )
        logger.info(f"[warmup] prefill done ({time.perf_counter() - t0:.2f} s)")

        # -- decode (full fixed batch, minimal seq) -------------------------
        compiled.decode_token_ids_buffer.zero_()
        compiled.decode_seq_lens_buffer.zero_()
        compiled.decode_block_table_buffer.fill_(0)     # all reads from page 0
        compiled.decode_slot_mapping_buffer.fill_(-1)   # all writes to page 0

        for b in range(kernel_batch):
            compiled.decode_seq_lens_buffer[b] = min(per_req + 1, max_seq)

        decode_kernel_inputs = _DecodeKernelInputs(
            actual_batch=kernel_batch,
            token_ids=compiled.decode_token_ids_buffer,
            seq_lens=compiled.decode_seq_lens_buffer,
            block_table=compiled.decode_block_table_buffer,
            slot_mapping=compiled.decode_slot_mapping_buffer,
            logits=compiled.decode_logits_buffer,
        )

        logger.info(f"[warmup] decode dispatch … (batch={kernel_batch}, seq_len={per_req + 1})")
        t0 = time.perf_counter()
        self._run_distributed_program(
            compiled.decode,
            *self._decode_kernel_args(decode_kernel_inputs, kv_cache.key_pages, kv_cache.value_pages),
        )
        logger.info(f"[warmup] decode done ({time.perf_counter() - t0:.2f} s)")

        logger.info("[warmup] complete")

    @staticmethod
    def _warmup_prefill_shape(runtime: RuntimeConfig) -> tuple[int, int, int]:
        """Return active batch, tokens per request, and total warmup tokens."""
        max_batch = runtime.max_batch_size
        effective_budget = runtime.max_num_batched_tokens
        if runtime.long_prefill_token_threshold > 0:
            effective_budget = min(effective_budget, runtime.long_prefill_token_threshold)
        step_tokens = min(effective_budget, max_batch * runtime.max_seq_len)
        if step_tokens <= 0:
            raise ValueError("warmup prefill token budget must be positive")
        active_batch = min(max_batch, step_tokens)
        per_request_tokens = step_tokens // active_batch
        total_tokens = per_request_tokens * active_batch
        return active_batch, per_request_tokens, total_tokens

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
        return (
            compiled.final_norm_weight,
            compiled.rope_cos,
            compiled.rope_sin,
            compiled.padded_lm_head_weight,
            compiled.padded_embed_weight,
            *compiled.decode_weights.values(),
            compiled.prefill_token_ids_buffer,
            compiled.prefill_seq_lens_buffer,
            compiled.prefill_chunk_lens_buffer,
            compiled.prefill_chunk_offsets_buffer,
            compiled.prefill_block_table_buffer,
            compiled.prefill_slot_mapping_buffer,
            compiled.prefill_logits_buffer,
            compiled.prefill_sampled_ids_buffer,
            compiled.prefill_next_hidden_buffer,
            compiled.decode_seq_lens_buffer,
            compiled.decode_block_table_buffer,
            compiled.decode_slot_mapping_buffer,
            compiled.decode_logits_buffer,
            compiled.decode_token_ids_buffer,
            compiled.decode_sampled_ids_buffer,
            compiled.decode_next_hidden_buffer,
        )

    def _build_static_kernel_args(self) -> _StaticKernelArgs:
        """Create static device-upload markers once per runner."""
        compiled = self._compiled
        return _StaticKernelArgs(
            final_norm_weight=self._static_device_tensor(compiled.final_norm_weight),
            rope_cos=self._static_device_tensor(compiled.rope_cos),
            rope_sin=self._static_device_tensor(compiled.rope_sin),
            padded_lm_head_weight=self._static_device_tensor(compiled.padded_lm_head_weight),
            padded_embed_weight=self._static_device_tensor(compiled.padded_embed_weight),
            decode_weights={
                name: self._static_device_tensor(tensor)
                for name, tensor in compiled.decode_weights.items()
            },
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
        k_cache = kv_cache.key_pages
        v_cache = kv_cache.value_pages
        self._validate_kv_cache_bounds(model, prefill_inputs.block_table, prefill_inputs.slot_mapping, k_cache)

        self._run_distributed_program(
            compiled.prefill,
            *self._prefill_kernel_args(prefill_inputs, k_cache, v_cache, logits_padded),
        )

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
        sampled_ids, next_hidden = self._maybe_run_sample_embed(
            logits_padded,
            compiled.prefill_sampled_ids_buffer,
            compiled.prefill_next_hidden_buffer,
            prefill_inputs.actual_batch,
            allow=batch.allow_device_greedy_sampling,
        )
        return PrefillResult(
            last_hidden=None,
            logits=logits_padded[: prefill_inputs.actual_batch, : model.config.vocab_size],
            sampled_token_ids=sampled_ids,
            next_hidden_states=next_hidden,
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the fused all-layer PAGED ``decode_fwd.decode_fwd`` and return logits.

        ``decode_fwd`` runs all NUM_LAYERS + the LM head in one dispatch over the
        PAGED KV pool, addressing KV via ``block_table`` + ``slot_mapping``, the
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
        k_cache = kv_cache.key_pages
        v_cache = kv_cache.value_pages

        kernel_inputs = self._pad_decode_inputs(model, decode_inputs)

        # Padded block_table / slot_mapping only ever reference row 0's
        # already-valid pages, so bound-check exactly what the kernel will read.
        self._validate_kv_cache_bounds(model, kernel_inputs.block_table, kernel_inputs.slot_mapping, k_cache)

        device_greedy = batch.allow_device_greedy_sampling
        self._run_distributed_program(
            compiled.decode,
            *self._decode_kernel_args(kernel_inputs, k_cache, v_cache, device_greedy=device_greedy),
        )
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        sampled_ids, next_hidden = self._integrated_sample_result(
            compiled.decode_sampled_ids_buffer,
            # decode_fwd's next_hidden output is the embedding for sampled_ids_in
            # used by this decode step. The newly sampled token is embedded at the
            # start of the following decode_fwd call, so there is no next-step
            # hidden row to return here.
            None,
            kernel_inputs.actual_batch,
            allow=batch.allow_device_greedy_sampling,
        )
        return DecodeResult(
            hidden_states=decode_inputs.hidden.float(),
            # Device-greedy path: the host consumes sampled_ids, never logits, so we
            # keep the logits buffer device-resident and skip its ~9.7MB D2H copy-back.
            logits=(
                None
                if device_greedy
                else kernel_inputs.logits[: kernel_inputs.actual_batch, : model.config.vocab_size].to(
                    decode_inputs.hidden.device
                )
            ),
            sampled_token_ids=sampled_ids,
            next_hidden_states=next_hidden,
        )

    @staticmethod
    def _integrated_sample_result(
        sampled_ids_buffer: torch.Tensor,
        next_hidden_buffer: torch.Tensor | None,
        actual_batch: int,
        *,
        allow: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Read device sampling output and optional precomputed next hidden rows."""
        if not allow:
            return None, None
        next_hidden = (
            next_hidden_buffer[:actual_batch].clone()
            if next_hidden_buffer is not None
            else None
        )
        return (
            sampled_ids_buffer[:actual_batch, :1].clone(),
            next_hidden,
        )

    def _maybe_run_sample_embed(
        self,
        logits: torch.Tensor,
        sampled_ids_buffer: torch.Tensor,
        next_hidden_buffer: torch.Tensor,
        actual_batch: int,
        *,
        allow: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run device greedy sampling when the request is greedy."""
        if not allow:
            return None, None
        compiled = self._compiled
        self._run_distributed_program(
            compiled.greedy_sample,
            logits,
            sampled_ids_buffer,
        )
        return (
            sampled_ids_buffer[:actual_batch, :1].clone(),
            None,
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
            inputs.token_ids,
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
            weights["decode_w_gate"],
            weights["decode_w_up"],
            weights["decode_w_down"],
            weights["decode_post_rms_weight"],
            static.final_norm_weight,
            static.padded_lm_head_weight,
            static.padded_embed_weight,
            logits,
        )

    def _decode_kernel_args(
        self,
        inputs: _DecodeKernelInputs,
        k_cache: DeviceTensor,
        v_cache: DeviceTensor,
        *,
        device_greedy: bool = False,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_decode_host`` signature order.

        On ``device_greedy`` the logits + next_hidden outputs are passed as
        worker-resident (device) tensors so they are never staged/copied-back
        per step (no memset, no D2H); only the tiny sampled_ids stays host-visible.
        """
        static = self._require_static_args()
        weights = static.decode_weights
        logits_arg = self._decode_logits_device_arg() if device_greedy else inputs.logits
        next_hidden_arg = (
            self._decode_next_hidden_device_arg() if device_greedy else self._compiled.decode_next_hidden_buffer
        )
        return (
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
            logits_arg,
            static.padded_embed_weight,
            inputs.token_ids,
            self._compiled.decode_sampled_ids_buffer,
            next_hidden_arg,
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

        token_ids = compiled.decode_token_ids_buffer
        token_ids.zero_()
        active_token_ids = inputs.token_ids.reshape(actual_batch, -1)
        if active_token_ids.shape[1] != 1:
            raise ValueError(
                "decode token_ids must contain exactly one token per row, "
                f"got shape {tuple(inputs.token_ids.shape)}"
            )
        width = 1
        token_ids[:actual_batch, :width].copy_(active_token_ids[:, :width])
        if actual_batch < kernel_batch:
            token_ids[actual_batch:, :width].copy_(
                active_token_ids[0:1, :width].expand(kernel_batch - actual_batch, width)
            )

        return _DecodeKernelInputs(
            actual_batch=actual_batch,
            token_ids=token_ids,
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

            worker = DistributedWorker([
                self._compiled.prefill.compiled,
                self._compiled.decode.compiled,
                self._compiled.greedy_sample.compiled,
            ])
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

    def _decode_logits_device_arg(self) -> DeviceTensor:
        """Device-resident decode logits scratch (greedy path: never copied back).

        Allocated directly on the worker and left uninitialized — the fused decode
        kernel writes every max_batch row before the on-device sampler reads it — so
        it forwards as a device pointer with no per-step staging/memset/D2H (and no
        ``_coerce_l3_arg`` dict lookup on the hot path).
        """
        dev = self._decode_logits_dev_tensor
        if dev is None:
            buffer = self._compiled.decode_logits_buffer
            dev = self._shared_l3_worker().alloc_tensor(buffer.shape, buffer.dtype)
            self._decode_logits_dev_tensor = dev
        return dev

    def _decode_next_hidden_device_arg(self) -> DeviceTensor:
        """Device-resident decode next_hidden scratch (never read on host)."""
        dev = self._decode_next_hidden_dev_tensor
        if dev is None:
            buffer = self._compiled.decode_next_hidden_buffer
            dev = self._shared_l3_worker().alloc_tensor(buffer.shape, buffer.dtype)
            self._decode_next_hidden_dev_tensor = dev
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
            static.padded_embed_weight,
            *static.decode_weights.values(),
        ):
            self._coerce_l3_arg(worker, arg)

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

        token_ids = compiled.prefill_token_ids_buffer[:total_tokens]
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
            chunk_token_ids = batch.token_ids[batch_idx, :chunk_len].to(torch.int32).cpu()
            token_ids[token_offset : token_offset + chunk_len] = chunk_token_ids

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
            token_ids=token_ids,
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
            token_ids=batch.token_ids.to(torch.int32).cpu(),
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
