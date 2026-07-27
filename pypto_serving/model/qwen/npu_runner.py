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
from pypto_serving.model.common.runner.model_runner import ModelRunner, _KvCachePool
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
    greedy_sample: _L3Callable | None = None
    tq_mode: bool = False
    rot_matrices: torch.Tensor | None = None
    tq_codebook: torch.Tensor | None = None
    tq_prefill_rope_cos: torch.Tensor | None = None
    tq_prefill_rope_sin: torch.Tensor | None = None
    final_norm_weight: torch.Tensor | None = None
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None
    padded_vocab: int = 0
    padded_lm_head_weight: torch.Tensor | None = None
    padded_embed_weight: torch.Tensor | None = None
    decode_weights: dict[str, torch.Tensor] | None = None
    prefill_token_ids_buffer: torch.Tensor | None = None
    prefill_hidden_buffer: torch.Tensor | None = None
    prefill_seq_lens_buffer: torch.Tensor | None = None
    prefill_chunk_lens_buffer: torch.Tensor | None = None
    prefill_chunk_offsets_buffer: torch.Tensor | None = None
    prefill_block_table_buffer: torch.Tensor | None = None
    prefill_slot_mapping_buffer: torch.Tensor | None = None
    prefill_logits_buffer: torch.Tensor | None = None
    prefill_sampled_ids_buffer: torch.Tensor | None = None
    prefill_next_hidden_buffer: torch.Tensor | None = None
    decode_seq_lens_buffer: torch.Tensor | None = None
    decode_block_table_buffer: torch.Tensor | None = None
    decode_slot_mapping_buffer: torch.Tensor | None = None
    decode_logits_buffer: torch.Tensor | None = None
    decode_token_ids_buffer: torch.Tensor | None = None
    decode_sampled_ids_buffer: torch.Tensor | None = None
    decode_next_hidden_buffer: torch.Tensor | None = None
    decode_hidden_buffer: torch.Tensor | None = None


@dataclass
class _PrefillInputs:
    """Host tensors passed to the prefill kernel."""

    actual_batch: int
    seq_lens: torch.Tensor
    chunk_lens: torch.Tensor
    chunk_offsets: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    # FP prefill embeds internally from ``token_ids``; TurboQuant prefill takes
    # the input embedding (hidden_states) directly. The unused one stays None.
    token_ids: torch.Tensor | None = None
    hidden: torch.Tensor | None = None


@dataclass
class _DecodeKernelInputs:
    """Fixed-batch tensors passed to the fused decode kernel."""

    actual_batch: int
    token_ids: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    logits: torch.Tensor
    # TurboQuant decode takes the input embedding (hidden_states) as its first
    # argument; FP decode embeds internally from ``token_ids`` and leaves this None.
    hidden: torch.Tensor | None = None


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
    # TurboQuant-only statics (None on the FP path).
    rot_matrices: _StaticDeviceTensor | None = None
    tq_codebook: _StaticDeviceTensor | None = None
    tq_prefill_rope_cos: _StaticDeviceTensor | None = None
    tq_prefill_rope_sin: _StaticDeviceTensor | None = None


class Qwen314BModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    def __init__(
        self,
        *,
        compiled: _CompiledKernels,
        device_id: int = 0,
        tq_mode: bool = False,
    ) -> None:
        super().__init__()
        self._compiled = compiled
        self._device_id = device_id
        self._tq_mode = tq_mode or getattr(compiled, "tq_mode", False)
        self._l3_worker: Any | None = None
        self._l3_static_tensors: dict[tuple[int, tuple[int, ...], torch.dtype], object] = {}
        # Device-resident decode output scratch (greedy path): allocated directly on
        # the worker (no host copy) so the per-step memset + D2H copy-back of the
        # max-batch logits/next_hidden vanish.
        self._decode_logits_dev_tensor: DeviceTensor | None = None
        self._decode_next_hidden_dev_tensor: DeviceTensor | None = None
        self._static_args: _StaticKernelArgs | None = None
        self._pending_kv_cache_specs: dict[str, tuple[ModelConfig, RuntimeConfig]] = {}
        # Page IDs currently materialized in each row of the persistent decode
        # block-table buffer. A row is rewritten only when its page allocation
        # changes (or when a different row-0 value must be used for padding).
        self._decode_block_table_row_pages: list[list[int] | None] = []
        self._decode_token_padding_initialized = False
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
            num_pages = self._kv_caches[model_id].rows_tensor.shape[0] // (
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
        num_pages = self._compute_kv_cache_pages(config, runtime, self._device_id, tq_mode=self._tq_mode)
        num_pages = self._alloc_kv_cache_with_retry(model_id, config, runtime, num_pages)
        self._print_memory_breakdown(
            "after KV cache alloc", config, runtime, num_pages, self._device_id, tq_mode=self._tq_mode,
        )
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
                bytes_per_page = self._kv_bytes_per_page(config, runtime, tq_mode=self._tq_mode)
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
    def _kv_bytes_per_page(
        config: ModelConfig, runtime: RuntimeConfig, *, tq_mode: bool = False,
    ) -> int:
        """Bytes of device KV cache backing one page (all layers, k + v).

        FP: ``num_layers * 2 * num_kv_heads * page_size * head_dim * dtype``.
        TQ: nibble-packed UINT8 cache (``head_dim // 2`` bytes/row) + one FP32
        scale (4 bytes/row) per k/v row.
        """
        rows_per_page = config.num_hidden_layers * config.num_key_value_heads * runtime.page_size
        if tq_mode:
            return rows_per_page * 2 * (config.head_dim // 2 + 4)
        dtype_bytes = getattr(torch, runtime.kv_dtype).itemsize
        return rows_per_page * 2 * config.head_dim * dtype_bytes

    @staticmethod
    def _compute_kv_cache_pages(
        config: ModelConfig, runtime: RuntimeConfig, device_id: int = 0, *, tq_mode: bool = False,
    ) -> int:
        """Compute KV cache pages, vLLM-style: total x utilization − peak_non_kv.

        Called AFTER the profile warm-up, so weights, the simpler ring-heap
        arena, compiled buffers and any persistent scratch are already
        allocated — ``peak_non_kv = total − free`` captures all of it. The KV
        budget is ``total x utilization − peak_non_kv``, leaving
        ``total x (1 − utilization)`` as a fixed absolute headroom (more robust
        than ``free x fraction`` whose headroom shrinks when free is small).
        """
        free_bytes, total_bytes = torch.npu.mem_get_info(f"npu:{device_id}")
        bytes_per_page = Qwen314BModelRunner._kv_bytes_per_page(config, runtime, tq_mode=tq_mode)
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
        device_id: int = 0, *, tq_mode: bool = False,
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
        bytes_per_page = Qwen314BModelRunner._kv_bytes_per_page(config, runtime, tq_mode=tq_mode)
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
        batch = runtime.max_batch_size
        max_seq = runtime.max_seq_len
        mnb = getattr(runtime, "max_num_batched_tokens", 4096)
        step_tokens = min(mnb, batch * max_seq)
        per_req = max(step_tokens // batch, 1)
        # TQ prefill declares a STATIC rope table (draft MAX_SEQ=128); keep the
        # warmup prefill within it so the rope index stays in bounds.
        if self._tq_mode and self._compiled.tq_prefill_rope_cos is not None:
            per_req = min(per_req, self._compiled.tq_prefill_rope_cos.shape[0])
        total_tokens = per_req * batch

        logger.info(
            f"[warmup] starting (batch={batch}, max_num_batched_tokens={mnb}, "
            f"max_seq={max_seq}, per_req={per_req}, total_tokens={total_tokens}, slot=-1)",
            
        )
        compiled = self._compiled
        kv_cache = list(self._kv_caches.values())[0]

        # -- prefill ---------------------------------------------------------
        # FP prefill embeds from token_ids; TurboQuant prefill takes the input
        # embedding (hidden_states) directly, so warm up the matching buffer.
        prefill_primary = compiled.prefill_hidden_buffer if self._tq_mode else compiled.prefill_token_ids_buffer
        prefill_primary[:total_tokens].zero_()
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
            token_ids=None if self._tq_mode else prefill_primary[:total_tokens],
            hidden=prefill_primary[:total_tokens] if self._tq_mode else None,
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
                prefill_inputs, kv_cache,
                compiled.prefill_logits_buffer,
            ),
        )
        logger.info(f"[warmup] prefill done ({time.perf_counter() - t0:.2f} s)")

        # -- decode (full fixed batch, minimal seq) -------------------------
        compiled.decode_token_ids_buffer.zero_()
        self._decode_token_padding_initialized = True
        compiled.decode_seq_lens_buffer.zero_()
        compiled.decode_block_table_buffer.fill_(0)     # all reads from page 0
        compiled.decode_slot_mapping_buffer.fill_(-1)   # all writes to page 0
        self._decode_block_table_row_pages.clear()

        for b in range(batch):
            compiled.decode_seq_lens_buffer[b] = min(per_req + 1, max_seq)

        # TQ decode takes a hidden_states input (zeros are fine for warmup).
        warmup_hidden = None
        if self._tq_mode and compiled.decode_hidden_buffer is not None:
            compiled.decode_hidden_buffer.zero_()
            warmup_hidden = compiled.decode_hidden_buffer

        decode_kernel_inputs = _DecodeKernelInputs(
            actual_batch=batch,
            token_ids=compiled.decode_token_ids_buffer,
            seq_lens=compiled.decode_seq_lens_buffer,
            block_table=compiled.decode_block_table_buffer,
            slot_mapping=compiled.decode_slot_mapping_buffer,
            logits=compiled.decode_logits_buffer,
            hidden=warmup_hidden,
        )

        logger.info(f"[warmup] decode dispatch … (batch={batch}, seq_len={per_req + 1})")
        t0 = time.perf_counter()
        self._run_distributed_program(
            compiled.decode,
            *self._decode_kernel_args(decode_kernel_inputs, kv_cache),
        )
        logger.info(f"[warmup] decode done ({time.perf_counter() - t0:.2f} s)")

        logger.info("[warmup] complete")

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
        tensors = [
            compiled.final_norm_weight,
            compiled.rope_cos,
            compiled.rope_sin,
            compiled.padded_lm_head_weight,
            compiled.padded_embed_weight,
            *compiled.decode_weights.values(),
            compiled.prefill_hidden_buffer if self._tq_mode else compiled.prefill_token_ids_buffer,
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
        ]
        if self._tq_mode:
            tensors.extend([
                compiled.rot_matrices,
                compiled.tq_codebook,
                compiled.tq_prefill_rope_cos,
                compiled.tq_prefill_rope_sin,
                compiled.decode_hidden_buffer,
            ])
        return tuple(tensors)

    def _build_static_kernel_args(self) -> _StaticKernelArgs:
        """Create static device-upload markers once per runner."""
        compiled = self._compiled
        static = _StaticKernelArgs(
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
        if self._tq_mode:
            static.rot_matrices = self._static_device_tensor(compiled.rot_matrices)
            static.tq_codebook = self._static_device_tensor(compiled.tq_codebook)
            static.tq_prefill_rope_cos = self._static_device_tensor(compiled.tq_prefill_rope_cos)
            static.tq_prefill_rope_sin = self._static_device_tensor(compiled.tq_prefill_rope_sin)
        return static

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
        bounds_cache = kv_cache.rows_tensor
        self._validate_kv_cache_bounds(model, prefill_inputs.block_table, prefill_inputs.slot_mapping, bounds_cache)

        self._run_distributed_program(
            compiled.prefill,
            *self._prefill_kernel_args(prefill_inputs, kv_cache, logits_padded),
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
        kernel_inputs = self._prepare_decode_inputs(model, batch)

        kv_cache = self._kv_caches.get(model_id)
        if kv_cache is None:
            raise RuntimeError(f"KV cache for model {model_id!r} is not initialized")
        bounds_cache = kv_cache.rows_tensor

        # Padded block_table / slot_mapping only ever reference row 0's
        # already-valid pages, so bound-check exactly what the kernel will read.
        self._validate_kv_cache_bounds(model, kernel_inputs.block_table, kernel_inputs.slot_mapping, bounds_cache)

        device_greedy = batch.allow_device_greedy_sampling
        self._run_distributed_program(
            compiled.decode,
            *self._decode_kernel_args(kernel_inputs, kv_cache, device_greedy=device_greedy),
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
            hidden_states=None,
            # Device-greedy path: the host consumes sampled_ids, never logits, so we
            # keep the logits buffer device-resident and skip its ~9.7MB D2H copy-back.
            logits=(
                None
                if device_greedy
                else kernel_inputs.logits[: kernel_inputs.actual_batch, : model.config.vocab_size].cpu()
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
        kv_cache: _KvCachePool,
        logits: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_prefill_host`` (FP) or
        ``qwen3_prefill_tq_host`` (TQ) signature order."""
        static = self._require_static_args()
        weights = static.decode_weights
        if self._tq_mode:
            # 26-param TQ order: NO chunk_lens/chunk_offsets; seq_lens at pos 1;
            # static [tq_prefill_max_seq, head_dim] rope; quant caches + scales;
            # rot/codebook; wo, post_rms, w_gate, w_up, w_down tail.
            return (
                inputs.hidden,                                  # hidden_states
                inputs.seq_lens,                                # seq_lens
                weights["decode_input_rms_weight"],
                weights["decode_wq"],
                weights["decode_wk"],
                weights["decode_wv"],
                weights["decode_q_norm_weight"],
                weights["decode_k_norm_weight"],
                static.tq_prefill_rope_cos,
                static.tq_prefill_rope_sin,
                inputs.block_table,
                inputs.slot_mapping,
                kv_cache.quant_k_pages,
                kv_cache.quant_v_pages,
                kv_cache.k_scales_pages,
                kv_cache.v_scales_pages,
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
            kv_cache.key_pages,
            kv_cache.value_pages,
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
        kv_cache: _KvCachePool,
        *,
        device_greedy: bool = False,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_decode_host`` (FP) or
        ``qwen3_decode_tq_host`` (TQ) signature order.

        On ``device_greedy`` (FP only) the logits + next_hidden outputs are passed
        as worker-resident (device) tensors so they are never staged/copied-back
        per step (no memset, no D2H); only the tiny sampled_ids stays host-visible.
        TQ decode returns raw logits (host sampling), so device_greedy is N/A.
        """
        static = self._require_static_args()
        weights = static.decode_weights
        if self._tq_mode:
            # 26-param TQ order: hidden_states first; quant caches + scales;
            # rot/codebook; wo, post_rms, w_gate, w_up, w_down tail; no
            # embed/sampled_ids/next_hidden.
            return (
                inputs.hidden,                                  # hidden_states
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
                kv_cache.quant_k_pages,
                kv_cache.quant_v_pages,
                kv_cache.k_scales_pages,
                kv_cache.v_scales_pages,
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
            kv_cache.key_pages,
            kv_cache.value_pages,
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

            programs = [
                self._compiled.prefill.compiled,
                self._compiled.decode.compiled,
            ]
            # FP path also registers the device greedy-sampling program; TQ
            # samples on the host (no device sampling program).
            if not self._tq_mode and self._compiled.greedy_sample is not None:
                programs.append(self._compiled.greedy_sample.compiled)
            worker = DistributedWorker(programs)
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
        args = [
            static.final_norm_weight,
            static.rope_cos,
            static.rope_sin,
            static.padded_lm_head_weight,
            static.padded_embed_weight,
            *static.decode_weights.values(),
        ]
        if self._tq_mode:
            args.extend([
                static.rot_matrices,
                static.tq_codebook,
                static.tq_prefill_rope_cos,
                static.tq_prefill_rope_sin,
            ])
        for arg in args:
            self._coerce_l3_arg(worker, arg)

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

        if self._tq_mode:
            hidden = compiled.prefill_hidden_buffer[:total_tokens]
            token_ids = None
        else:
            token_ids = compiled.prefill_token_ids_buffer[:total_tokens]
            hidden = None
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
            if self._tq_mode:
                embeddings = batch.input_embeddings[batch_idx, :chunk_len, :].to(torch.bfloat16).cpu()
                hidden[token_offset : token_offset + chunk_len, :] = embeddings
            else:
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
    ) -> _DecodeKernelInputs:
        """Write active decode metadata directly into persistent kernel buffers.

        The fused kernel has a fixed batch size. Active rows are written first,
        and inactive rows replicate row 0 so their KV writes remain idempotent.
        Block-table rows persist across calls and are rewritten only when their
        page IDs change.
        """
        compiled = self._compiled
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        kernel_batch = model.runtime.max_batch_size
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)

        if kernel_batch > compiled.decode_logits_buffer.shape[0]:
            raise ValueError(
                f"kernel batch {kernel_batch} exceeds logits buffer batch "
                f"{compiled.decode_logits_buffer.shape[0]}"
            )

        token_ids = compiled.decode_token_ids_buffer
        token_rows = token_ids.reshape(kernel_batch, -1)
        active_token_rows = batch.token_ids.reshape(actual_batch, -1)
        if active_token_rows.shape[1] != 1:
            raise ValueError(
                "decode token_ids must contain exactly one token per row, "
                f"got shape {tuple(batch.token_ids.shape)}"
            )
        if token_rows.shape[1] < 1:
            raise ValueError("compiled decode token buffer must have at least one column")
        # The kernel ABI pads token rows to SAMPLED_IDS_PAD columns (8 for
        # Qwen3-14B), but only column 0 carries the next token ID.
        if not self._decode_token_padding_initialized:
            token_rows[:, 1:].zero_()
            self._decode_token_padding_initialized = True
        token_rows[:actual_batch, :1].copy_(active_token_rows)

        seq_lens = compiled.decode_seq_lens_buffer
        seq_lens_flat = seq_lens.reshape(-1)
        active_seq_lens = batch.seq_lens.reshape(-1)
        if active_seq_lens.numel() < actual_batch:
            raise ValueError(
                f"decode seq_lens has {active_seq_lens.numel()} rows, expected {actual_batch}"
            )
        seq_lens_flat[:actual_batch].copy_(active_seq_lens[:actual_batch])
        seq_len_values = seq_lens_flat[:actual_batch].tolist()

        block_table = compiled.decode_block_table_buffer
        block_table_rows = block_table.reshape(kernel_batch, max_blocks)
        slot_mapping = compiled.decode_slot_mapping_buffer
        slot_mapping_flat = slot_mapping.reshape(-1)
        if len(self._decode_block_table_row_pages) != kernel_batch:
            self._decode_block_table_row_pages = [None] * kernel_batch

        first_page_ids: list[int] | None = None

        for batch_idx in range(actual_batch):
            alloc = batch.kv_allocations[batch_idx] if batch_idx < len(batch.kv_allocations) else None
            seq_len = int(seq_len_values[batch_idx])
            if seq_len <= 0:
                raise ValueError("decode seq_lens must be positive")
            if seq_len > model.runtime.max_seq_len:
                raise ValueError(
                    f"decode seq_len {seq_len} exceeds max_seq_len {model.runtime.max_seq_len}"
                )

            if alloc is not None:
                page_ids = alloc.page_ids
            elif batch_idx < len(batch.block_ids):
                page_ids = batch.block_ids[batch_idx]
            else:
                page_ids = []
            self._write_cached_decode_block_table_row(block_table_rows, batch_idx, page_ids)
            if batch_idx == 0:
                first_page_ids = page_ids

            tokens_used = seq_len - 1
            page_idx = tokens_used // page_size
            offset = tokens_used % page_size
            if page_idx >= len(page_ids):
                raise ValueError(
                    f"page_ids list length {len(page_ids)} is too small for decode position {tokens_used}; "
                    f"need at least {page_idx + 1} pages"
                )
            slot_mapping_flat[batch_idx] = page_ids[page_idx] * page_size + offset

        if actual_batch < kernel_batch:
            inactive_rows = kernel_batch - actual_batch
            token_rows[actual_batch:, :1].copy_(token_rows[0:1, :1].expand(inactive_rows, 1))
            seq_lens_flat[actual_batch:].copy_(seq_lens_flat[0:1].expand(inactive_rows))
            slot_mapping_flat[actual_batch:].copy_(slot_mapping_flat[0:1].expand(inactive_rows))
            if first_page_ids is None:
                raise RuntimeError("decode batch is missing row-0 page IDs")
            for batch_idx in range(actual_batch, kernel_batch):
                self._write_cached_decode_block_table_row(block_table_rows, batch_idx, first_page_ids)

        # TurboQuant decode consumes the input embedding (hidden_states) directly
        # instead of embedding inside the kernel; stage the host-provided
        # ``batch.hidden_states`` into the persistent device buffer (padding rows
        # replicate row 0 so their KV writes stay idempotent).
        hidden_arg = None
        if self._tq_mode and batch.hidden_states is not None and compiled.decode_hidden_buffer is not None:
            hidden_arg = compiled.decode_hidden_buffer
            hidden_arg[:actual_batch].copy_(batch.hidden_states)
            if actual_batch < kernel_batch:
                hidden_arg[actual_batch:].copy_(
                    batch.hidden_states[0:1].expand(kernel_batch - actual_batch, -1)
                )

        return _DecodeKernelInputs(
            actual_batch=actual_batch,
            token_ids=token_ids,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            logits=compiled.decode_logits_buffer,
            hidden=hidden_arg,
        )

    def _write_cached_decode_block_table_row(
        self,
        block_table_rows: torch.Tensor,
        batch_idx: int,
        page_ids: list[int],
    ) -> None:
        """Materialize one persistent decode block-table row when it changes."""
        if self._decode_block_table_row_pages[batch_idx] == page_ids:
            return

        row = block_table_rows[batch_idx]
        if len(page_ids) > row.numel():
            raise ValueError(
                f"page_ids list length {len(page_ids)} exceeds block-table width {row.numel()}"
            )

        row.fill_(-1)
        if page_ids:
            row[: len(page_ids)].copy_(torch.tensor(page_ids, dtype=row.dtype))
        self._decode_block_table_row_pages[batch_idx] = list(page_ids)

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
