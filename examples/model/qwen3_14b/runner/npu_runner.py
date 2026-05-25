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
import time
from dataclasses import dataclass
from typing import Any

import torch

try:
    from python.core._profiling import StageTimer
    from python.core.kv_cache import KvCacheManager
    from python.core.model_runner import ModelRunner
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
        KvAllocation,
        PrefillBatch,
        PrefillResult,
        RuntimeModel,
    )
    from python.runtime.worker import Worker as LlmWorker
    from python.runtime.worker import WorkerTensor
except ImportError:
    from python.core._profiling import StageTimer
    from python.core.kv_cache import KvCacheManager
    from python.core.model_runner import ModelRunner
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
        KvAllocation,
        PrefillBatch,
        PrefillResult,
        RuntimeModel,
    )
    from python.runtime.worker import Worker as LlmWorker
    from python.runtime.worker import WorkerTensor

_TIMING_ENABLED = True
_LOGITS_BATCH_TILE = 16


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


@dataclass
class _L2Callable:
    """Assembled non-L3 callable and launch metadata."""

    chip_callable: object
    runtime_name: str
    block_dim: int
    aicpu_thread_num: int
    param_infos: tuple[object, ...]


@dataclass
class _CompiledKernels:
    """Compiled Qwen3-14B kernels and immutable runtime tensors."""

    prefill: _L2Callable
    decode: _L2Callable
    final_rms: _L2Callable
    lm_head: _L2Callable
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    layers: list[_KernelLayerWeights]
    decode_weights: dict[str, torch.Tensor]
    # L3-wrapped generate artifacts. Populated only when l3_mode=True.
    stacked_weights: dict[str, torch.Tensor] | None = None
    # Compiled L3 generate program (pypto DistributedCompiledProgram). The
    # runner calls .prepare(sub_worker_overrides=...) on it once to obtain a
    # reusable DistributedRuntime; setup (assemble + Worker fork) happens once
    # and every generate request dispatches on the held Worker.
    l3_generate_program: object | None = None


@dataclass
class _PrefillInputs:
    """Padded host tensors passed to the prefill kernel."""

    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
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


@dataclass
class _L3Runtime:
    """Reusable L3 generate handle: prepared once, dispatched per request.

    Holds the pypto ``DistributedRuntime`` (assembled + forked once via
    ``prepare()``) plus all fixed-size shared-memory IO buffers and the
    worker-resident weight/KV ``DeviceTensor`` handles. ``args`` is the
    positional argument tuple for ``rt(*args)`` in kernel-parameter order;
    its host buffers are reused in place across requests and the weight
    ``DeviceTensor`` objects stay device-resident.
    """

    rt: Any  # pypto DistributedRuntime
    args: tuple[Any, ...]
    # Per-request host buffers (filled in place before each dispatch).
    prefill_hidden: torch.Tensor
    prefill_seq_lens: torch.Tensor
    prefill_slot_mapping: torch.Tensor
    block_table: torch.Tensor
    decode_hidden_buf: torch.Tensor
    decode_seq_lens: torch.Tensor
    decode_slot_mapping_buf: torch.Tensor
    decode_out_storage: torch.Tensor  # rms_x; decode_out is the [:batch] view
    # Sub-worker shared state (read in the forked child, reset per request).
    done_flag: torch.Tensor
    generated_ids: torch.Tensor
    token_count: torch.Tensor
    eos_id_buf: torch.Tensor
    max_new_tokens_buf: torch.Tensor
    page_ids_buf: torch.Tensor
    num_pages_buf: torch.Tensor
    # KV cache: device-resident buffers + persistent host pool views for sync.
    kv_k: Any  # DeviceTensor
    kv_v: Any  # DeviceTensor
    kv_k_host: torch.Tensor
    kv_v_host: torch.Tensor


class Qwen314BModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    def __init__(
        self,
        *,
        model_id: str,
        compiled: _CompiledKernels,
        kv_cache_manager: KvCacheManager,
        platform: str,
        device_id: int,
        save_kernels_dir: str | None,
        l3_trace: bool,
    ) -> None:
        self._model_id = model_id
        self._compiled = compiled
        self._kv_cache_manager = kv_cache_manager
        self._platform = platform
        self._device_id = device_id
        self._save_kernels_dir = save_kernels_dir
        self._l3_trace = l3_trace
        self._l2_workers: dict[str, LlmWorker] = {}
        self._l2_programs: dict[int, _L2ProgramHandle] = {}
        self._l2_child_allocs: dict[tuple[str, int], tuple[int, int]] = {}
        self._l2_dirty_kv_models: set[str] = set()
        # One reusable L3 generate handle per model (prepared lazily on first
        # run_generate_l3 call, dispatched per request).
        self._l3_runtimes: dict[str, _L3Runtime] = {}

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run layer-by-layer prompt prefill and project next-token logits."""
        compiled = self._compiled
        prefill_inputs = self._prepare_prefill_inputs(model, batch)
        hidden = prefill_inputs.hidden
        t_prefill_start = time.perf_counter()

        for layer_idx, layer in enumerate(compiled.layers):
            k_cache, v_cache = self._kv_cache_manager.materialize_single_layer_cache(
                model.config.model_id,
                layer_idx,
            )
            out = torch.zeros_like(hidden).share_memory_()
            prefill_args = (
                hidden,
                prefill_inputs.seq_lens,
                layer.input_rms_weight,
                layer.wq,
                layer.wk,
                layer.wv,
                layer.q_norm_weight,
                layer.k_norm_weight,
                compiled.rope_cos,
                compiled.rope_sin,
                prefill_inputs.block_table,
                prefill_inputs.slot_mapping,
                k_cache,
                v_cache,
                layer.wo,
                layer.post_rms_weight,
                layer.w_gate,
                layer.w_up,
                layer.w_down,
                out,
            )
            if isinstance(compiled.prefill, _L2Callable):
                self._run_l2_program(compiled.prefill, *prefill_args)
            else:
                compiled.prefill(*prefill_args, config=None)
            hidden = out
        self._l2_dirty_kv_models.add(model.config.model_id)

        if _TIMING_ENABLED:
            print(
                f"[timing] prefill: {len(model.layers)} layers, "
                f"{(time.perf_counter() - t_prefill_start) * 1000:.2f} ms",
                flush=True,
            )

        last_hidden_rows: list[torch.Tensor] = []
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
            last_hidden_rows.append(hidden[batch_idx, seq_len - 1].float())
        last_hidden = torch.stack(last_hidden_rows)
        logits = self._project_logits(model, last_hidden)
        return PrefillResult(last_hidden=last_hidden, logits=logits)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the fused all-layer decode kernel and project next-token logits."""
        # The fused decode kernel (decode_full.py) processes all
        # layers in one call: weights are pre-stacked into [num_layers * ...]
        # tensors at compile time and the KV cache is the full multi-layer
        # buffer. Argument order mirrors the kernel signature in
        # build_qwen3_decode_program.qwen3_decode.
        compiled = self._compiled
        decode_inputs = self._prepare_decode_inputs(model, batch)
        hidden = decode_inputs.hidden
        dw = compiled.decode_weights

        k_cache, v_cache = self._kv_cache_manager.materialize_full_layer_cache(
            model.config.model_id,
        )
        refresh_kv_cache = model.config.model_id in self._l2_dirty_kv_models
        out = torch.zeros_like(hidden)

        if isinstance(compiled.decode, _L2Callable):
            self._run_l2_program(
                compiled.decode,
                hidden,
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_input_rms_weight"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wq"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wk"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wv"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_q_norm_weight"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_k_norm_weight"]),
                decode_inputs.seq_lens,
                decode_inputs.block_table,
                decode_inputs.slot_mapping,
                self._l2_child_tensor(compiled.decode.runtime_name, compiled.rope_cos),
                self._l2_child_tensor(compiled.decode.runtime_name, compiled.rope_sin),
                self._l2_child_tensor(compiled.decode.runtime_name, k_cache, refresh=refresh_kv_cache),
                self._l2_child_tensor(compiled.decode.runtime_name, v_cache, refresh=refresh_kv_cache),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wo"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_post_rms_weight"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_gate"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_up"]),
                self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_down"]),
                out,
            )
        else:
            compiled.decode(
                hidden,
                dw["decode_input_rms_weight"],
                dw["decode_wq"],
                dw["decode_wk"],
                dw["decode_wv"],
                dw["decode_q_norm_weight"],
                dw["decode_k_norm_weight"],
                decode_inputs.seq_lens,
                decode_inputs.block_table,
                decode_inputs.slot_mapping,
                compiled.rope_cos,
                compiled.rope_sin,
                k_cache,
                v_cache,
                dw["decode_wo"],
                dw["decode_post_rms_weight"],
                dw["decode_w_gate"],
                dw["decode_w_up"],
                dw["decode_w_down"],
                out,
                config=None,
            )
        self._l2_dirty_kv_models.discard(model.config.model_id)

        final_hidden = out.float()

        logits = self._project_logits(model, final_hidden)
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        return DecodeResult(hidden_states=final_hidden, logits=logits)

    def _project_logits(self, model: RuntimeModel, hidden: torch.Tensor) -> torch.Tensor:
        """Run final RMSNorm and LM head kernels for a hidden-state batch."""
        compiled = self._compiled
        hidden_size = model.config.hidden_size
        vocab_size = model.config.vocab_size
        padded_vocab = compiled.padded_vocab

        actual_batch = hidden.shape[0]
        if actual_batch > _LOGITS_BATCH_TILE:
            raise ValueError(
                f"logit batch {actual_batch} exceeds _LOGITS_BATCH_TILE {_LOGITS_BATCH_TILE}"
            )

        x = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16).share_memory_()
        x[:actual_batch] = hidden.to(torch.bfloat16).cpu()
        if not isinstance(compiled.final_rms, _L2Callable) or not isinstance(compiled.lm_head, _L2Callable):
            normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16)
            compiled.final_rms(x, compiled.final_norm_weight, normed, config=None)
            logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32)
            compiled.lm_head(normed, compiled.padded_lm_head_weight, logits_padded, config=None)
            return logits_padded[:actual_batch, :vocab_size].to(hidden.device)

        normed: torch.Tensor | WorkerTensor
        x_arg: torch.Tensor | WorkerTensor = x
        worker: LlmWorker | None = None
        if compiled.final_rms.runtime_name == compiled.lm_head.runtime_name:
            worker = self._worker_for_runtime(compiled.final_rms.runtime_name)
            x_arg = worker.alloc_tensor(x.shape, x.dtype, init=x)
            normed = worker.alloc_tensor(x.shape, x.dtype)
        else:
            normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16).share_memory_()

        try:
            self._run_l2_program(
                compiled.final_rms,
                x_arg,
                self._l2_child_tensor(compiled.final_rms.runtime_name, compiled.final_norm_weight),
                normed,
            )

            logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32).share_memory_()
            self._run_l2_program(
                compiled.lm_head,
                normed,
                self._l2_child_tensor(compiled.lm_head.runtime_name, compiled.padded_lm_head_weight),
                logits_padded,
            )
        finally:
            if isinstance(x_arg, WorkerTensor):
                if worker is None:
                    raise RuntimeError("missing L2 worker for child-memory logits projection")
                worker.free_tensor(x_arg)
            if isinstance(normed, WorkerTensor):
                if worker is None:
                    raise RuntimeError("missing L2 worker for child-memory logits projection")
                worker.free_tensor(normed)
        return logits_padded[:actual_batch, :vocab_size].to(hidden.device)

    def _run_l2_program(self, callable_spec: _L2Callable, *args: Any) -> None:
        """Run a compiled non-L3 program through the LLM Simpler worker."""
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        handle = self._ensure_l2_program(callable_spec)
        orch_args = self._build_l2_orch_args(callable_spec, args)

        cfg = CallConfig()
        cfg.block_dim = callable_spec.block_dim
        cfg.aicpu_thread_num = callable_spec.aicpu_thread_num

        worker = self._l2_workers[handle.runtime_name]
        worker.run(handle.callable_id, orch_args, cfg)

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

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor's storage to shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    def close(self) -> None:
        """Release L3 runtimes, non-L3 child-memory allocations, and L2 workers."""
        # Close reusable L3 generate handles (releases the held Worker + its
        # device-resident weight/KV buffers).
        for lr in self._l3_runtimes.values():
            lr.rt.close()
        self._l3_runtimes.clear()
        for (runtime_name, _), (dev_ptr, _nbytes) in list(self._l2_child_allocs.items()):
            worker = self._l2_workers.get(runtime_name)
            if worker is not None and worker.initialized:
                worker.free(dev_ptr)
        self._l2_child_allocs.clear()
        self._l2_programs.clear()
        self._l2_dirty_kv_models.clear()
        for worker in self._l2_workers.values():
            worker.close()
        self._l2_workers.clear()

    @staticmethod
    def _build_l2_orch_args(callable_spec: _L2Callable, args: tuple[Any, ...]):
        """Build ``ChipStorageTaskArgs`` for a compiled L2 program call."""
        from simpler.task_interface import ChipStorageTaskArgs, ContinuousTensor, scalar_to_uint64  # noqa: PLC0415
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

    # ── L3-wrapped generate: entire prefill + decode loop in one dispatch ──
    #
    # Setup (assemble + Worker fork + static-weight upload) happens once per
    # model via DistributedCompiledProgram.prepare(); each request fills the
    # fixed shared buffers in place and dispatches on the held DistributedRuntime.

    def run_generate_l3(
        self,
        model: RuntimeModel,
        prefill_batch: PrefillBatch,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> tuple[list[int], torch.Tensor]:
        """Run the full generate loop on a reusable L3 DistributedRuntime.

        host_orch drives prefill + final_rms + lm_head + the unrolled decode
        loop in a single dispatch; the sample_and_prepare sub-worker performs
        CPU-side sampling and embedding lookup between decode steps. The worker
        and resident weights are prepared once (see _ensure_l3_runtime) and
        reused; this call only refreshes per-request buffers and dispatches.

        Returns (generated_token_ids, final_hidden).
        """
        compiled = self._compiled
        if compiled.l3_generate_program is None:
            raise RuntimeError("L3 generate program not compiled.")
        if max_new_tokens > model.runtime.max_new_tokens:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} exceeds compiled L3 limit "
                f"{model.runtime.max_new_tokens}"
            )

        timer = StageTimer(
            enabled=self._l3_trace,
            prefix="L3-breakdown",
            title="run_generate_l3 stage timings",
        )

        def _mark(label: str) -> None:
            timer.mark(label)

        lr = self._ensure_l3_runtime(model)
        _mark("ensure_l3_runtime")

        prefill_inputs = self._prepare_prefill_inputs(model, prefill_batch)
        actual_batch = prefill_inputs.actual_batch
        if actual_batch != 1:
            raise ValueError(
                "run_generate_l3 currently supports batch_size=1 only; "
                f"got {actual_batch} requests."
            )
        _mark("prepare_prefill_inputs")

        # Fill the fixed shared buffers in place (allocated once before fork).
        lr.prefill_hidden.copy_(prefill_inputs.hidden)
        lr.prefill_seq_lens.copy_(prefill_inputs.seq_lens)
        lr.prefill_slot_mapping.copy_(prefill_inputs.slot_mapping)
        lr.block_table.copy_(prefill_inputs.block_table)

        # Initial decode input: last prompt token's embedding + its KV slot.
        seq_len0 = int(prefill_inputs.seq_lens[0].item())
        lr.decode_hidden_buf[0] = prefill_inputs.hidden[0, seq_len0 - 1, :]
        lr.decode_seq_lens.copy_(prefill_inputs.seq_lens)
        lr.decode_slot_mapping_buf[0] = int(
            prefill_inputs.slot_mapping[seq_len0 - 1].item()
        )

        # Reset sub-worker control state for this request.
        lr.done_flag.zero_()
        lr.token_count.zero_()
        lr.generated_ids.fill_(-1)
        lr.eos_id_buf[0] = -1 if eos_token_id is None else int(eos_token_id)
        lr.max_new_tokens_buf[0] = int(max_new_tokens)

        # Per-request KV page table used by sample_and_prepare for slot mapping.
        alloc = prefill_batch.kv_allocations[0]
        n_pages = len(alloc.page_ids)
        lr.num_pages_buf[0] = n_pages
        if n_pages > 0:
            lr.page_ids_buf[:n_pages] = torch.tensor(alloc.page_ids, dtype=torch.int32)
        _mark("fill_buffers")

        # Refresh device KV from the host pool, dispatch, then sync back.
        lr.rt.copy_to(lr.kv_k.data_ptr, lr.kv_k_host.data_ptr(), lr.kv_k.nbytes)
        lr.rt.copy_to(lr.kv_v.data_ptr, lr.kv_v_host.data_ptr(), lr.kv_v.nbytes)
        _mark("kv_upload")

        lr.rt(*lr.args)
        _mark("dispatch")

        lr.rt.copy_from(lr.kv_k_host.data_ptr(), lr.kv_k.data_ptr, lr.kv_k.nbytes)
        lr.rt.copy_from(lr.kv_v_host.data_ptr(), lr.kv_v.data_ptr, lr.kv_v.nbytes)
        _mark("kv_sync_back")

        # Update KV allocation usage.
        final_token_count = int(lr.token_count[0].item())
        base_seq = int(prefill_inputs.seq_lens[0].item())
        alloc.tokens_used = max(alloc.tokens_used, base_seq + final_token_count)

        ids = lr.generated_ids[:final_token_count].tolist()
        ret_val = ids, lr.decode_out_storage[:actual_batch].float()
        _mark("post_process")

        timer.report()
        return ret_val

    def _ensure_l3_runtime(self, model: RuntimeModel) -> _L3Runtime:
        """Prepare (once per model) the reusable L3 generate runtime.

        Allocates all fixed-size shared-memory buffers BEFORE the worker fork,
        defines the sample_and_prepare sub-worker closure (which reads only
        shared memory so it serves every request), calls
        DistributedCompiledProgram.prepare() to assemble + fork the worker
        once, and uploads the static weights and KV cache to worker-resident
        DeviceTensor buffers. The result is cached and reused per request.
        """
        model_id = model.config.model_id
        cached = self._l3_runtimes.get(model_id)
        if cached is not None:
            return cached

        compiled = self._compiled
        if compiled.l3_generate_program is None:
            raise RuntimeError("L3 generate program not compiled.")
        hidden_size = model.config.hidden_size
        max_seq = model.runtime.max_seq_len
        vocab_size = model.config.vocab_size
        padded_vocab = compiled.padded_vocab
        page_size = model.runtime.page_size
        mnt_limit = model.runtime.max_new_tokens
        max_blocks = self._max_blocks_per_seq(model)
        batch = 1  # L3 generate fast path is batch_size=1 (see validate_generate_batch)

        # ── Fixed-size shared host buffers. They must exist before the worker
        # fork so the forked sub-worker inherits their shared-memory mappings. ──
        def _shm(*shape: int, dtype: torch.dtype, fill: int | None = None) -> torch.Tensor:
            t = torch.zeros(shape, dtype=dtype) if not fill else torch.full(shape, fill, dtype=dtype)
            return t.share_memory_()

        prefill_hidden = _shm(batch, max_seq, hidden_size, dtype=torch.bfloat16)
        prefill_seq_lens = _shm(batch, dtype=torch.int32)
        prefill_slot_mapping = _shm(batch * max_seq, dtype=torch.int32, fill=-1)
        block_table = _shm(batch * max_blocks, dtype=torch.int32, fill=-1)
        decode_hidden_buf = _shm(batch, hidden_size, dtype=torch.bfloat16)
        decode_seq_lens = _shm(batch, dtype=torch.int32)
        decode_slot_mapping_buf = _shm(batch, dtype=torch.int32)
        # has_prefill is always True: step 0 inside host_orch runs prefill_all
        # then the first decode; subsequent unrolled iterations are decode-only.
        has_prefill = torch.tensor(True, dtype=torch.bool).share_memory_()
        prefill_out = _shm(batch, max_seq, hidden_size, dtype=torch.bfloat16)
        # decode_out and rms_x share one padded buffer (no CPU copy between
        # them). host_orch writes decode_out (first `batch` rows); final_rms
        # reads rms_x (all _LOGITS_BATCH_TILE rows). Padding rows stay zero —
        # the kernel only writes the first `batch` rows — satisfying final_rms.
        decode_out_storage = _shm(_LOGITS_BATCH_TILE, hidden_size, dtype=torch.bfloat16)
        decode_out = decode_out_storage[:batch, :]
        rms_x = decode_out_storage
        rms_normed = _shm(_LOGITS_BATCH_TILE, hidden_size, dtype=torch.bfloat16)
        logits_padded = _shm(_LOGITS_BATCH_TILE, padded_vocab, dtype=torch.float32)

        # Sub-worker shared state (reset per request in run_generate_l3).
        done_flag = _shm(1, dtype=torch.int32)
        generated_ids = _shm(mnt_limit, dtype=torch.int64, fill=-1)
        token_count = _shm(1, dtype=torch.int32)
        eos_id_buf = _shm(1, dtype=torch.int64, fill=-1)
        max_new_tokens_buf = _shm(1, dtype=torch.int64)
        page_ids_buf = _shm(max_blocks, dtype=torch.int32)
        num_pages_buf = _shm(1, dtype=torch.int32)

        # Static model tensors. Each must be a CPU, contiguous, shared-memory
        # tensor allocated BEFORE prepare()'s fork so the upload (which runs in
        # the forked chip worker) reads the right bytes; alloc_tensor enforces
        # this. _static_src guarantees contiguity (alloc_tensor asserts it).
        def _static_src(t: torch.Tensor) -> torch.Tensor:
            if t.device.type != "cpu":
                t = t.cpu()
            if not t.is_contiguous():
                t = t.contiguous()
            return t if t.is_shared() else t.share_memory_()

        embed_tokens = _static_src(model.embed_tokens.to(torch.bfloat16))
        rms_gamma = _static_src(model.final_norm_weight.view(1, hidden_size).float())
        lm_head_weight = _static_src(compiled.padded_lm_head_weight)
        rope_cos = _static_src(compiled.rope_cos)
        rope_sin = _static_src(compiled.rope_sin)
        sm_sw = {k: _static_src(v) for k, v in compiled.stacked_weights.items()}

        # Persistent KV pool views (same backing storage across requests).
        kv_k_host, kv_v_host = self._kv_cache_manager.materialize_full_layer_cache(model_id)
        kv_k_host = self._share_cpu_tensor(kv_k_host)
        kv_v_host = self._share_cpu_tensor(kv_v_host)

        # ── sample_and_prepare sub-worker (runs in the forked child). Reads
        # logits → argmax → writes the next decode inputs. It reads only shared
        # memory, so one closure serves every request. ──
        def sample_and_prepare_fn(task_args):  # noqa: ANN001, ARG001
            if done_flag[0].item():
                return  # request already finished (EOS / length)
            token_id = int(logits_padded[0, :vocab_size].argmax().item())
            step = int(token_count[0].item())
            generated_ids[step] = token_id
            token_count[0] = step + 1
            eos = int(eos_id_buf[0].item())
            if eos >= 0 and token_id == eos:
                done_flag[0] = 1
                return
            if step + 1 >= int(max_new_tokens_buf[0].item()):
                done_flag[0] = 1
                return
            # Embedding lookup for the next decode step.
            decode_hidden_buf[0, :] = embed_tokens[token_id]
            new_seq_len = int(decode_seq_lens[0].item()) + 1
            decode_seq_lens[0] = new_seq_len
            page_idx = (new_seq_len - 1) // page_size
            slot_in_page = (new_seq_len - 1) % page_size
            if page_idx < int(num_pages_buf[0].item()):
                decode_slot_mapping_buf[0] = (
                    int(page_ids_buf[page_idx].item()) * page_size + slot_in_page
                )

        # ── Prepare the worker once (assemble + fork + register override). ──
        rt = compiled.l3_generate_program.prepare(
            sub_worker_overrides={"sample_and_prepare": sample_and_prepare_fn},
        )

        # Upload static weights + KV to worker-resident DeviceTensors (once).
        def _dev(host: torch.Tensor) -> Any:
            return rt.alloc_tensor(tuple(int(s) for s in host.shape), host.dtype, init=host)

        w = {k: _dev(v) for k, v in sm_sw.items()}
        rope_cos_dt = _dev(rope_cos)
        rope_sin_dt = _dev(rope_sin)
        final_norm_dt = _dev(rms_gamma)
        lm_head_dt = _dev(lm_head_weight)
        kv_k = _dev(kv_k_host)
        kv_v = _dev(kv_v_host)

        # Fixed positional args in kernel-parameter order. Host buffers are
        # reused in place; weight/KV DeviceTensors stay device-resident. Order
        # mirrors build_qwen3_14b_l3_generate_program's host_orch signature.
        args = (
            prefill_hidden, prefill_seq_lens, prefill_slot_mapping,
            decode_hidden_buf, decode_seq_lens, decode_slot_mapping_buf,
            w["input_rms_weight"], w["wq"], w["wk"], w["wv"],
            w["q_norm_weight"], w["k_norm_weight"],
            rope_cos_dt, rope_sin_dt,
            block_table, kv_k, kv_v,
            w["wo"], w["post_rms_weight"], w["w_gate"], w["w_up"], w["w_down"],
            has_prefill, prefill_out, decode_out, rms_x,
            final_norm_dt, rms_normed, lm_head_dt, logits_padded,
        )

        lr = _L3Runtime(
            rt=rt,
            args=args,
            prefill_hidden=prefill_hidden,
            prefill_seq_lens=prefill_seq_lens,
            prefill_slot_mapping=prefill_slot_mapping,
            block_table=block_table,
            decode_hidden_buf=decode_hidden_buf,
            decode_seq_lens=decode_seq_lens,
            decode_slot_mapping_buf=decode_slot_mapping_buf,
            decode_out_storage=decode_out_storage,
            done_flag=done_flag,
            generated_ids=generated_ids,
            token_count=token_count,
            eos_id_buf=eos_id_buf,
            max_new_tokens_buf=max_new_tokens_buf,
            page_ids_buf=page_ids_buf,
            num_pages_buf=num_pages_buf,
            kv_k=kv_k,
            kv_v=kv_v,
            kv_k_host=kv_k_host,
            kv_v_host=kv_v_host,
        )
        self._l3_runtimes[model_id] = lr
        return lr

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
    ) -> _PrefillInputs:
        """Pack variable-length prefill requests into kernel input tensors."""
        actual_batch = self._validate_batch_size(model, len(batch.kv_allocations))
        max_seq = model.runtime.max_seq_len
        hidden_size = model.config.hidden_size
        max_blocks = self._max_blocks_per_seq(model)

        has_precomputed = batch.slot_mapping is not None and batch.block_table is not None

        hidden = torch.zeros((actual_batch, max_seq, hidden_size), dtype=torch.bfloat16)
        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        slot_mapping = torch.full((actual_batch * max_seq,), -1, dtype=torch.int32)

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            if seq_len <= 0:
                raise ValueError("prefill seq_lens must be positive")
            if seq_len > max_seq:
                raise ValueError(f"prefill seq_len {seq_len} exceeds max_seq_len {max_seq}")
            seq_lens[batch_idx] = seq_len
            hidden[batch_idx, :seq_len, :] = (
                batch.input_embeddings[batch_idx, :seq_len, :].to(torch.bfloat16).cpu()
            )

            if has_precomputed:
                bt_row = batch.block_table[batch_idx]
                valid_blocks = (bt_row >= 0).sum().item()
                block_table[
                    batch_idx * max_blocks : batch_idx * max_blocks + valid_blocks
                ] = bt_row[:valid_blocks].to(torch.int32).cpu()

                sm_row = batch.slot_mapping[batch_idx]
                valid_slots = (sm_row >= 0).sum().item()
                slot_mapping[
                    batch_idx * max_seq : batch_idx * max_seq + valid_slots
                ] = sm_row[:valid_slots].to(torch.int32).cpu()
            else:
                self._write_block_table_row(block_table, batch_idx, max_blocks, alloc)
                slot_row = self._kv_cache_manager.slot_mapping_for_positions(
                    alloc,
                    seq_len,
                    max_tokens=max_seq,
                )
                slot_mapping[batch_idx * max_seq : (batch_idx + 1) * max_seq] = slot_row

        return _PrefillInputs(
            actual_batch=actual_batch,
            hidden=hidden.share_memory_(),
            seq_lens=seq_lens.share_memory_(),
            block_table=block_table.share_memory_(),
            slot_mapping=slot_mapping.share_memory_(),
        )

    def _prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> _DecodeInputs:
        """Pack active decode requests into fused decode-kernel inputs."""
        actual_batch = self._validate_batch_size(model, len(batch.kv_allocations))
        hidden_size = model.config.hidden_size
        max_blocks = self._max_blocks_per_seq(model)

        hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        slot_mapping = torch.empty((actual_batch,), dtype=torch.int32)

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            if seq_len <= 0:
                raise ValueError("decode seq_lens must be positive")
            if seq_len > model.runtime.max_seq_len:
                raise ValueError(
                    f"decode seq_len {seq_len} exceeds max_seq_len {model.runtime.max_seq_len}"
                )
            hidden[batch_idx, :] = batch.hidden_states[batch_idx].to(torch.bfloat16).cpu()
            seq_lens[batch_idx] = seq_len
            self._write_block_table_row(block_table, batch_idx, max_blocks, alloc)
            slot_mapping[batch_idx] = self._kv_cache_manager.slot_mapping_for_request(alloc)

        return _DecodeInputs(
            actual_batch=actual_batch,
            hidden=hidden.share_memory_(),
            seq_lens=seq_lens.share_memory_(),
            block_table=block_table.share_memory_(),
            slot_mapping=slot_mapping.share_memory_(),
        )

    @staticmethod
    def _write_block_table_row(
        block_table: torch.Tensor,
        batch_idx: int,
        max_blocks: int,
        alloc: KvAllocation,
    ) -> None:
        """Write one request's KV page IDs into a flat block table."""
        row_start = batch_idx * max_blocks
        if alloc.page_ids:
            block_table[row_start : row_start + len(alloc.page_ids)] = torch.tensor(
                alloc.page_ids,
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
