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
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    SamplingCandidates,
)
from pypto_serving.model.common.runner.buffer_set import StaticDeviceTensor, resolve_l3_arg
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.common.runner.task_args import TaskArgs
from pypto_serving.tools.profile import profile_span


logger = logging.getLogger(__name__)


# HOST-dispatched compiled program + launch metadata. Unified with DeepSeek's
# DeepSeekV4L3Callable in pypto_serving.model.common.compiler.l3_callable.
from pypto_serving.model.common.compiler.l3_callable import L3Callable as _L3Callable


@dataclass
class QwenLayout:
    """Shape-defining constants for the Qwen3-14B host-shared buffers.

    Computed by the executor (which has the model + kernel constants) and read
    by ``task_args.py`` to size the TaskArgs slots. This keeps buffer-shape
    derivation in the executor (model/kernel context) while the allocation
    itself lives with the runner's TaskArgs.
    """

    kernel_batch: int
    max_seq_len: int
    page_size: int
    max_blocks_per_seq: int
    padded_vocab: int
    hidden_size: int
    sampled_ids_width: int
    topk_width: int


@dataclass
class _CompiledKernels:
    """Compiled Qwen3-14B kernels and immutable runtime tensors.

    The per-dispatch I/O host buffers are owned by the runner's TaskArgs
    (allocated via ``TaskArgs.allocate_host_shared``); only the compiled
    programs, the static weights, and the shape layout live here.
    """

    prefill: _L3Callable
    decode: _L3Callable
    topk_select: _L3Callable
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    padded_embed_weight: torch.Tensor
    decode_weights: dict[str, torch.Tensor]
    layout: QwenLayout


@dataclass
class _PrefillInputs:
    """Per-call prefill dispatch inputs spliced into the built TaskArgs tuple.

    The fixed-shape buffers (seq_lens / chunk_lens / block_table / ...) are read
    from the prefill TaskArgs slots by ``build()``; only the length-
    ``total_tokens`` ``input_ids`` / ``slot_mapping`` slices vary per call.
    """

    actual_batch: int
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeKernelInputs:
    """Per-call decode context for host readback (buffers live in the decode TaskArgs)."""

    actual_batch: int
    logits: torch.Tensor


@dataclass
class _StaticKernelArgs:
    """Static worker-resident kernel arguments reused across dispatches."""

    final_norm_weight: StaticDeviceTensor
    rope_cos: StaticDeviceTensor
    rope_sin: StaticDeviceTensor
    padded_lm_head_weight: StaticDeviceTensor
    padded_embed_weight: StaticDeviceTensor
    decode_weights: dict[str, StaticDeviceTensor]


class Qwen314BModelRunner(L3DispatchMixin, ModelRunner):
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
        self._init_l3_dispatch(stacked=False)
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
        self._prefill_metadata_arrays: tuple = ()
        self._prefill_slot_mapping_array: np.ndarray | None = None
        # Per-call length-total_tokens slices over the full prefill input_ids /
        # slot_mapping slots, spliced into the built prefill tuple at dispatch.
        self._active_prefill_token_ids: torch.Tensor | None = None
        self._active_prefill_slot_mapping: torch.Tensor | None = None
        # Per-dispatch TaskArgs owning the host-shared I/O buffers.
        self._prefill_task_args: TaskArgs | None = None
        self._decode_task_args: TaskArgs | None = None
        self._topk_select_task_args: TaskArgs | None = None
        if compiled is not None:
            self._share_static_kernel_tensors()
            self._static_args = self._build_static_kernel_args()
            self._ensure_task_args()

    def _ensure_task_args(self) -> None:
        """Build and allocate the per-dispatch TaskArgs (owns the host buffers).

        Runs at construction -- after the static weights are shared -- so the
        host-shared slots are in shared memory before the L3 worker is created.
        """
        from pypto_serving.model.qwen.task_args import (  # noqa: PLC0415
            decode_task_args,
            prefill_task_args,
            topk_select_task_args,
        )
        self._prefill_task_args = prefill_task_args(self)
        self._decode_task_args = decode_task_args(self)
        self._topk_select_task_args = topk_select_task_args(self)
        for task_args in (
            self._prefill_task_args,
            self._decode_task_args,
            self._topk_select_task_args,
        ):
            task_args.allocate_host_shared(self._compiled.layout)
        # Numpy views over the persistent prefill metadata slots (written via
        # numpy each step in _prepare_prefill_inputs).
        prefill_tensors = self._prefill_task_args.tensors
        self._prefill_metadata_arrays = (
            prefill_tensors["seq_lens"].numpy(),
            prefill_tensors["chunk_lens"].numpy(),
            prefill_tensors["chunk_offsets"].numpy(),
        )
        self._prefill_slot_mapping_array = prefill_tensors["slot_mapping"].numpy()

    @property
    def _active_kv_cache(self) -> Any:
        """The single materialized paged KV cache (read by the TaskArgs lazy sources)."""
        if not self._kv_caches:
            raise RuntimeError("KV cache is not initialized")
        return next(iter(self._kv_caches.values()))

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
        self._configure_l3_rings(runtime)

        logger.info("[init_kv_cache] creating L3 worker …")
        with profile_span("Qwen314BModelRunner.prepare_l3_worker", cat="executor"):
            self._shared_l3_worker()
        # The L3 worker assembles the device binaries into each program's
        # output_dir, which (via the compiler's save_kernels_dir) already is the
        # kernel-cache slot -- so the cache is populated directly, no store step.

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
        simpler_committed = self._query_simpler_committed()
        if simpler_committed:
            logger.info("[init_kv_cache] using simpler committed_device_memory=%.2f GB", simpler_committed / 1e9)
        else:
            logger.info("[init_kv_cache] committed_device_memory unavailable; using driver-only peak_non_kv sizing")
        num_pages = self._compute_kv_cache_pages(
            config, runtime, self._device_id, simpler_committed=simpler_committed,
        )
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

    def _query_simpler_committed(self) -> int:
        """simpler's committed device HBM for the device being sized (self._device_id).

        ``committed_device_memory`` is simpler's authoritative MemoryAllocator
        total (weights + pooled arenas + runtime buffers). KV sizing is per
        device, so query only this device's chip child - not the sum across all
        chips (which would over-subtract on a multi-device worker).
        ``_compute_kv_cache_pages`` takes ``max(peak_non_kv, this)``, so the
        budget never over-provisions whether or not ``aclrtGetMemInfo`` sees
        simpler's ``rtMalloc`` pool. Returns 0 if the worker isn't ready or the
        query fails (falls back to the old sizing + halve-retry).
        """
        worker = self._l3_worker
        if worker is None:
            return 0
        try:
            device_ids = tuple(getattr(worker, "device_ids", ()) or (self._device_id,))
            chip = device_ids.index(self._device_id) if self._device_id in device_ids else 0
            return int(worker.committed_device_memory(chip))
        except Exception as e:  # noqa: BLE001
            logger.warning("committed_device_memory unavailable; KV sizing ignores it: %s", e)
            return 0

    @staticmethod
    def _compute_kv_cache_pages(
        config: ModelConfig, runtime: RuntimeConfig, device_id: int = 0, simpler_committed: int = 0,
    ) -> int:
        """Compute KV cache pages, vLLM-style: total x utilization - peak_non_kv.

        Called AFTER the profile warm-up, so weights, the simpler ring-heap
        arena, compiled buffers and any persistent scratch are already
        allocated -- ``peak_non_kv = total - free`` captures all of it. The KV
        budget is ``total x utilization - max(peak_non_kv, simpler_committed)``,
        leaving ``total x (1 - utilization)`` as a fixed absolute headroom. The
        driver-visible ``peak_non_kv`` and simpler's authoritative
        ``simpler_committed`` (``Worker.committed_device_memory``; 0 if unknown)
        overlap (both include weights + arenas); taking the max never
        over-provisions whether or not the driver sees simpler's ``rtMalloc`` pool.
        """
        free_bytes, total_bytes = torch.npu.mem_get_info(f"npu:{device_id}")
        dtype_bytes = getattr(torch, runtime.kv_dtype).itemsize
        bytes_per_page = (
            config.num_hidden_layers * 2 * config.num_key_value_heads
            * runtime.page_size * config.head_dim * dtype_bytes
        )
        utilization = getattr(runtime, "npu_memory_utilization", 0.90)
        peak_non_kv = total_bytes - free_bytes
        # Two overlapping views of non-KV usage: peak_non_kv (driver-visible via
        # aclrtGetMemInfo) and simpler_committed (simpler's authoritative
        # MemoryAllocator total — weights + arenas + buffers). Take the max so we
        # never over-provision whether or not the driver sees simpler's rtMalloc
        # pool; they usually agree, and max is robust to either undercounting.
        non_kv = max(peak_non_kv, simpler_committed)
        kv_budget = int(total_bytes * utilization - non_kv)
        num_pages = max(kv_budget // bytes_per_page, 1)
        logger.info(
            "KV cache sizing (vLLM-style): total=%.2f GB, utilization=%.2f, "
            "peak_non_kv=%.2f GB, simpler_committed=%.2f GB, kv_budget=%.2f GB, "
            "requested_pages=%d (%.1f MB/page)",
            total_bytes / 1e9, utilization, peak_non_kv / 1e9, simpler_committed / 1e9,
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
        ring-heap arena (from the dispatch ring config x 4), and the
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

        # Simpler ring-heap arena — from the dispatch ring config (matches
        # _configure_l3_rings; replaces the old PTO2_RING_HEAP env read). A
        # scalar broadcasts to the 4 scope-depth rings, a list sizes each;
        # unset sizing falls back to the historical 256 MiB/ring estimate.
        ring_heap = runtime.ring_heap
        if ring_heap is None:
            heap_total = 4 * 256 * 1024 * 1024
        elif isinstance(ring_heap, list):
            heap_total = sum(ring_heap)
        else:
            heap_total = ring_heap * 4
        arena_bytes = heap_total + 128 * 1024 * 1024

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
        logger.info(f"  ├─ simpler arena (rings):    {arena_bytes / 1e9:7.2f} GB")
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
        total_tokens = per_req * batch

        logger.info(
            f"[warmup] starting (batch={batch}, max_num_batched_tokens={mnb}, "
            f"max_seq={max_seq}, per_req={per_req}, total_tokens={total_tokens}, slot=-1)",
        )
        compiled = self._compiled
        # The scratch KV cache is materialized before warmup; the TaskArgs lazy
        # sources read it via _active_kv_cache at build() time.
        prefill = self._prefill_task_args.tensors

        # -- prefill ---------------------------------------------------------
        prefill["input_ids"][:total_tokens].zero_()
        prefill["seq_lens"].zero_()
        prefill["chunk_lens"].zero_()
        prefill["chunk_offsets"].zero_()
        prefill["block_table"].fill_(0)    # all reads from page 0
        prefill["slot_mapping"].fill_(-1)  # all writes to page 0

        token_offset = 0
        for b in range(batch):
            prefill["seq_lens"][b] = per_req
            prefill["chunk_lens"][b] = per_req
            prefill["chunk_offsets"][b] = token_offset
            token_offset += per_req

        prefill_inputs = _PrefillInputs(
            actual_batch=batch,
            token_ids=prefill["input_ids"][:total_tokens],
            slot_mapping=prefill["slot_mapping"][:total_tokens],
        )

        logger.info(f"[warmup] prefill dispatch … (batch={batch}, tokens={total_tokens})")
        t0 = time.perf_counter()
        self._run_l3(compiled.prefill, *self._prefill_kernel_args(prefill_inputs))
        logger.info(f"[warmup] prefill done ({time.perf_counter() - t0:.2f} s)")

        # -- decode (full fixed batch, minimal seq) -------------------------
        decode = self._decode_task_args.tensors
        decode["token_ids"].zero_()
        self._decode_token_padding_initialized = True
        decode["seq_lens"].zero_()
        decode["block_table"].fill_(0)     # all reads from page 0
        decode["slot_mapping"].fill_(-1)   # all writes to page 0
        self._decode_block_table_row_pages.clear()

        for b in range(batch):
            decode["seq_lens"][b] = min(per_req + 1, max_seq)

        logger.info(f"[warmup] decode dispatch … (batch={batch}, seq_len={per_req + 1})")
        t0 = time.perf_counter()
        self._run_l3(
            compiled.decode,
            *self._decode_kernel_args(actual_batch=batch, device_sampling=False),
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

    def _share_static_kernel_tensors(self) -> None:
        """Move static kernel inputs to shared memory before worker creation."""
        for tensor in self._iter_static_host_tensors():
            self._share_cpu_tensor(tensor)

    def _iter_static_host_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return host tensors that must be shared before the worker forks.

        The per-dispatch I/O buffers are owned by the TaskArgs (allocated as
        shared memory via ``shared_empty``), so only the immutable static
        weights need sharing here.
        """
        compiled = self._compiled
        return (
            compiled.final_norm_weight,
            compiled.rope_cos,
            compiled.rope_sin,
            compiled.padded_lm_head_weight,
            compiled.padded_embed_weight,
            *compiled.decode_weights.values(),
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

        logits_padded = self._prefill_task_args.tensors["logits"]

        # Materialize the KV cache so the TaskArgs lazy sources resolve it via
        # _active_kv_cache at build() time.
        self._materialize_kv_cache(model)

        self._run_l3(compiled.prefill, *self._prefill_kernel_args(prefill_inputs))

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = batch.seq_lens[batch_idx]
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
        sampled_ids = self._maybe_run_greedy_sample(
            logits_padded,
            prefill_inputs.actual_batch,
            allow=batch.allow_device_greedy_sampling,
        )
        sampling_candidates = self._device_topk_outputs(
            logits_padded,
            self._topk_select_task_args.tensors["prefill_topk_values"],
            self._topk_select_task_args.tensors["prefill_topk_indices"],
            prefill_inputs.actual_batch,
            allow=batch.allow_device_topk_sampling,
        )
        return PrefillResult(
            last_hidden=None,
            logits=logits_padded[: prefill_inputs.actual_batch, : model.config.vocab_size],
            sampled_token_ids=sampled_ids,
            sampling_candidates=sampling_candidates,
            next_hidden_states=None,
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the fused all-layer PAGED ``decode_fwd.decode_fwd`` and return logits.

        ``decode_fwd`` runs all NUM_LAYERS + the LM head in one dispatch over the
        PAGED KV pool, addressing KV via ``block_table`` + ``slot_mapping``, the
        SAME device-resident KV pool prefill writes (``self._kv_caches``), so prompt
        KV is already in place with no bridge. KV is keyed by block_table page id, not by
        kernel row, so a request may occupy any row each step (no stable-slot shim).

        The compiled ABI has runtime-dynamic batch dimensions. Persistent host and
        device buffers retain max-batch capacity, but each dispatch passes prefix
        views containing only the active rows. In particular, inactive rows must
        not replicate row 0: native paged attention appends K/V in parallel, so
        duplicate rows would race while writing the same physical KV slot.
        """
        compiled = self._compiled
        model_id = model.config.model_id
        kernel_inputs = self._prepare_decode_inputs(model, batch)

        if self._kv_caches.get(model_id) is None:
            raise RuntimeError(f"KV cache for model {model_id!r} is not initialized")

        device_sampling = (
            batch.allow_device_greedy_sampling or batch.allow_device_topk_sampling
        )
        selector_logits = (
            self._decode_logits_device_arg() if device_sampling else kernel_inputs.logits
        )
        self._run_l3(
            compiled.decode,
            *self._decode_kernel_args(
                actual_batch=kernel_inputs.actual_batch,
                device_sampling=device_sampling,
            ),
        )
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        sampled_ids, next_hidden = self._integrated_sample_result(
            self._decode_task_args.tensors["sampled_ids"],
            # decode_fwd's next_hidden output is the embedding for sampled_ids_in
            # used by this decode step. The newly sampled token is embedded at the
            # start of the following decode_fwd call, so there is no next-step
            # hidden row to return here.
            None,
            kernel_inputs.actual_batch,
            allow=batch.allow_device_greedy_sampling,
        )
        sampling_candidates = self._device_topk_outputs(
            selector_logits,
            self._topk_select_task_args.tensors["decode_topk_values"],
            self._topk_select_task_args.tensors["decode_topk_indices"],
            kernel_inputs.actual_batch,
            allow=batch.allow_device_topk_sampling,
        )
        return DecodeResult(
            hidden_states=None,
            # Device sampling consumes only sampled ids or top-k candidates, so
            # full-vocabulary logits remain device-resident.
            logits=(
                None
                if device_sampling
                else kernel_inputs.logits[: kernel_inputs.actual_batch, : model.config.vocab_size].cpu()
            ),
            sampled_token_ids=sampled_ids,
            sampling_candidates=sampling_candidates,
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

    def _maybe_run_greedy_sample(
        self,
        logits: torch.Tensor,
        actual_batch: int,
        *,
        allow: bool,
    ) -> torch.Tensor | None:
        """Run the sampling selector in exact greedy mode for prefill."""
        if not allow:
            return None
        topk = self._topk_select_task_args.tensors
        self._run_sampling_selector(
            logits,
            topk["prefill_topk_values"],
            topk["prefill_topk_indices"],
            actual_batch,
            selection_k=1,
        )
        return topk["prefill_topk_indices"][:actual_batch, :1].clone()

    def _device_topk_outputs(
        self,
        logits: torch.Tensor | DeviceTensor,
        values_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        actual_batch: int,
        *,
        allow: bool,
    ) -> SamplingCandidates | None:
        """Run device top-k candidate selection and return small host tensors."""
        if not allow:
            return None
        self._run_sampling_selector(
            logits,
            values_buffer,
            indices_buffer,
            actual_batch,
            selection_k=indices_buffer.shape[1],
        )
        return SamplingCandidates(
            values=values_buffer[:actual_batch].clone(),
            token_ids=indices_buffer[:actual_batch].clone(),
        )

    def _run_sampling_selector(
        self,
        logits: torch.Tensor | DeviceTensor,
        values_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        actual_batch: int,
        *,
        selection_k: int,
    ) -> None:
        """Run the shared greedy/top-k selector without adding another worker program."""
        control = self._topk_select_task_args.tensors["sampling_control"]
        control[0] = int(actual_batch)
        control[1] = int(selection_k)
        self._run_l3(
            self._compiled.topk_select,
            logits,
            control,
            values_buffer,
            indices_buffer,
        )

    def _prefill_kernel_args(self, inputs: _PrefillInputs) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_prefill_host`` signature order.

        Fixed-shape buffers and weights come from the prefill TaskArgs
        (``build()``); the per-call ``input_ids`` / ``slot_mapping`` are
        length-``total_tokens`` slices over the full slots, spliced in.
        """
        task_args = self._prefill_task_args
        args = list(task_args.build())
        args[task_args.names.index("input_ids")] = inputs.token_ids
        args[task_args.names.index("slot_mapping")] = inputs.slot_mapping
        return tuple(args)

    def _decode_kernel_args(
        self,
        *,
        actual_batch: int,
        device_sampling: bool = False,
    ) -> tuple[Any, ...]:
        """Return arguments in ``qwen3_decode_host`` signature order.

        Buffers and weights come from the decode TaskArgs (``build()``). The
        storage has max-batch capacity, while all batch-shaped arguments are
        narrowed to ``actual_batch`` so the dynamic kernel executes only live
        rows. Under device sampling, ``logits`` and ``next_hidden`` swap from
        the host slots to worker-resident device scratch before those logical
        views are created, so full-vocabulary logits are never copied back.
        """
        kernel_batch = self._compiled.layout.kernel_batch
        if actual_batch <= 0 or actual_batch > kernel_batch:
            raise ValueError(
                f"decode actual batch must be in [1, {kernel_batch}], got {actual_batch}"
            )

        task_args = self._decode_task_args
        args = list(task_args.build())
        if device_sampling:
            args[task_args.names.index("logits")] = self._decode_logits_device_arg()
            args[task_args.names.index("next_hidden")] = self._decode_next_hidden_device_arg()

        for name in (
            "seq_lens",
            "slot_mapping",
            "logits",
            "token_ids",
            "sampled_ids",
            "next_hidden",
        ):
            index = task_args.names.index(name)
            args[index] = self._batch_prefix_view(args[index], actual_batch)

        block_table_index = task_args.names.index("block_table")
        block_table_rows = actual_batch * self._compiled.layout.max_blocks_per_seq
        args[block_table_index] = args[block_table_index][:block_table_rows]
        return tuple(args)

    @staticmethod
    def _batch_prefix_view(tensor: Any, actual_batch: int) -> Any:
        """Return a zero-offset leading-batch view without reallocating storage."""
        if isinstance(tensor, DeviceTensor):
            if tensor.shape[0] < actual_batch:
                raise ValueError(
                    f"device tensor batch capacity {tensor.shape[0]} is smaller than {actual_batch}"
                )
            if tensor.shape[0] == actual_batch:
                return tensor
            return DeviceTensor(
                tensor.data_ptr,
                (actual_batch, *tensor.shape[1:]),
                tensor.dtype,
                buffer=tensor.buffer,
            )
        return tensor[:actual_batch]

    def _shared_l3_worker(self) -> Any:
        """Return the worker shared by the generation prefill/decode path."""
        worker = self._l3_worker
        if worker is None:
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            worker = DistributedWorker([
                self._compiled.prefill.compiled,
                self._compiled.decode.compiled,
                self._compiled.topk_select.compiled,
            ])
            self._l3_worker = worker
        return worker

    def _decode_logits_device_arg(self) -> DeviceTensor:
        """Device-resident decode logits scratch for device sampling.

        Allocated directly on the worker and left uninitialized — the fused decode
        kernel writes every max_batch row before the on-device sampler reads it — so
        it forwards as a device pointer with no per-step staging/memset/D2H (and no
        ``resolve_l3_arg`` dict lookup on the hot path).
        """
        dev = self._decode_logits_dev_tensor
        if dev is None:
            buffer = self._decode_task_args.tensors["logits"]
            dev = self._shared_l3_worker().alloc_tensor(buffer.shape, buffer.dtype)
            self._decode_logits_dev_tensor = dev
        return dev

    def _decode_next_hidden_device_arg(self) -> DeviceTensor:
        """Device-resident decode next_hidden scratch (never read on host)."""
        dev = self._decode_next_hidden_dev_tensor
        if dev is None:
            buffer = self._decode_task_args.tensors["next_hidden"]
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
            resolve_l3_arg(worker, arg, self._l3_static_tensors, stacked=self._l3_stacked)

    @staticmethod
    def _static_device_tensor(tensor: torch.Tensor) -> StaticDeviceTensor:
        """Mark a CPU tensor for one-time upload to the shared worker."""
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        return StaticDeviceTensor(Qwen314BModelRunner._share_cpu_tensor(tensor))

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
                self._reset_l3_dispatch()

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
    ) -> _PrefillInputs:
        """Copy packed prefill inputs into persistent kernel buffers."""
        batch_count = len(batch.request_ids)
        actual_batch = self._validate_batch_size(model, batch_count)

        max_seq = model.runtime.max_seq_len
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)
        kernel_batch = model.runtime.max_batch_size

        seq_len_values = batch.seq_lens
        chunk_len_values = batch.chunk_lens
        chunk_offset_values = batch.chunk_offsets
        chunk_start_values = batch.chunk_starts
        for seq_len in seq_len_values:
            if seq_len > max_seq:
                raise ValueError(f"prefill seq_len {seq_len} exceeds max_seq_len {max_seq}")

        total_tokens = int(batch.token_ids.numel())
        max_tokens = kernel_batch * max_seq
        if total_tokens > max_tokens:
            raise ValueError(f"prefill total tokens {total_tokens} exceeds kernel capacity {max_tokens}")

        buffers = self._prefill_task_args.tensors
        token_ids = buffers["input_ids"][:total_tokens]
        seq_lens = buffers["seq_lens"]
        chunk_lens = buffers["chunk_lens"]
        chunk_offsets = buffers["chunk_offsets"]
        block_table = buffers["block_table"]
        slot_mapping = buffers["slot_mapping"][:total_tokens]
        slot_mapping_array = self._prefill_slot_mapping_array
        if slot_mapping_array is None:
            raise RuntimeError("prefill slot-mapping buffer is not initialized")
        seq_lens.zero_()
        chunk_lens.zero_()
        chunk_offsets.zero_()
        block_table.fill_(-1)

        token_ids.copy_(batch.token_ids)
        for target, values in zip(
            self._prefill_metadata_arrays,
            (seq_len_values, chunk_len_values, chunk_offset_values),
        ):
            target[:actual_batch] = values

        for batch_idx in range(actual_batch):
            alloc = batch.kv_allocations[batch_idx] if batch_idx < len(batch.kv_allocations) else None
            chunk_len = chunk_len_values[batch_idx]
            chunk_start = chunk_start_values[batch_idx]
            chunk_offset = chunk_offset_values[batch_idx]

            if alloc is not None:
                page_ids = alloc.page_ids
            elif batch_idx < len(batch.block_ids):
                page_ids = batch.block_ids[batch_idx]
            else:
                page_ids = []
            self._write_block_table_row(block_table, batch_idx, max_blocks, page_ids)

            self._write_slot_mapping(
                slot_mapping_array[chunk_offset : chunk_offset + chunk_len],
                page_ids,
                chunk_len,
                page_size,
                start_pos=chunk_start,
            )

        return _PrefillInputs(
            actual_batch=actual_batch,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
        )

    def _prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> _DecodeKernelInputs:
        """Write active decode metadata directly into persistent kernel buffers.

        Active rows are written into max-capacity storage; the dispatch path
        exposes only those rows through dynamic prefix views. Block-table rows
        persist across calls and are rewritten only when their page IDs change.
        """
        buffers = self._decode_task_args.tensors
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        kernel_batch = model.runtime.max_batch_size
        page_size = model.runtime.page_size
        max_blocks = self._max_blocks_per_seq(model)

        logits_buffer = buffers["logits"]
        if kernel_batch > logits_buffer.shape[0]:
            raise ValueError(
                f"kernel batch {kernel_batch} exceeds logits buffer batch "
                f"{logits_buffer.shape[0]}"
            )

        token_ids = buffers["token_ids"]
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

        seq_lens = buffers["seq_lens"]
        seq_lens_flat = seq_lens.reshape(-1)
        active_seq_lens = batch.seq_lens.reshape(-1)
        if active_seq_lens.numel() < actual_batch:
            raise ValueError(
                f"decode seq_lens has {active_seq_lens.numel()} rows, expected {actual_batch}"
            )
        seq_lens_flat[:actual_batch].copy_(active_seq_lens[:actual_batch])
        seq_len_values = seq_lens_flat[:actual_batch].tolist()

        block_table = buffers["block_table"]
        block_table_rows = block_table.reshape(kernel_batch, max_blocks)
        slot_mapping = buffers["slot_mapping"]
        slot_mapping_flat = slot_mapping.reshape(-1)
        if len(self._decode_block_table_row_pages) != kernel_batch:
            self._decode_block_table_row_pages = [None] * kernel_batch

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

            tokens_used = seq_len - 1
            page_idx = tokens_used // page_size
            offset = tokens_used % page_size
            if page_idx >= len(page_ids):
                raise ValueError(
                    f"page_ids list length {len(page_ids)} is too small for decode position {tokens_used}; "
                    f"need at least {page_idx + 1} pages"
                )
            slot_mapping_flat[batch_idx] = page_ids[page_idx] * page_size + offset

        return _DecodeKernelInputs(
            actual_batch=actual_batch,
            logits=logits_buffer,
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
    def _write_slot_mapping(
        target: np.ndarray,
        page_ids: list[int],
        num_tokens: int,
        page_size: int,
        *,
        start_pos: int = 0,
    ) -> None:
        """Write physical slots directly into a preallocated kernel-buffer view."""
        if num_tokens > 0:
            max_pos = start_pos + num_tokens - 1
            max_page_idx = max_pos // page_size
            if max_page_idx >= len(page_ids):
                raise ValueError(
                    f"page_ids list length {len(page_ids)} is too small for position {max_pos}; "
                    f"need at least {max_page_idx + 1} pages"
                )

        target_offset = 0
        pos = start_pos
        end_pos = start_pos + num_tokens
        while pos < end_pos:
            page_idx, page_offset = divmod(pos, page_size)
            span = min(end_pos - pos, page_size - page_offset)
            physical_start = page_ids[page_idx] * page_size + page_offset
            target[target_offset : target_offset + span] = np.arange(
                physical_start,
                physical_start + span,
                dtype=target.dtype,
            )
            target_offset += span
            pos += span

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
