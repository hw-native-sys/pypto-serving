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
import inspect
import time
from dataclasses import dataclass
from typing import Any

import torch

try:
    from python.core._profiling import StageTimer
    from python.core.model_runner import ModelRunner
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
        PrefillBatch,
        PrefillResult,
        RuntimeModel,
    )
    from python.runtime.worker import Worker as LlmWorker
    from python.runtime.worker import WorkerTensor
except ImportError:
    from python.core._profiling import StageTimer
    from python.core.model_runner import ModelRunner
    from python.core.types import (
        DecodeBatch,
        DecodeResult,
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
    final_rms: _L2Callable | None
    lm_head: _L2Callable | None
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    layers: list[_KernelLayerWeights]
    decode_weights: dict[str, torch.Tensor]
    decode_logits_buffer: torch.Tensor
    # L3-wrapped generate artifacts. Populated only when l3_mode=True.
    stacked_weights: dict[str, torch.Tensor] | None = None
    l3_generate_chip_callables: dict[str, object] | None = None
    l3_generate_entry_fn: object | None = None
    l3_generate_sub_worker_fns: dict[str, object] | None = None
    l3_generate_dc: object | None = None  # DistributedConfig
    l3_generate_platform: str | None = None
    l3_generate_runtime_name: str | None = None
    l3_generate_param_infos: object | None = None


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


@dataclass(frozen=True)
class _DeferredChildTensor:
    """CPU tensor that should be materialized as child memory inside an L3 run."""

    runtime_name: str
    tensor: torch.Tensor
    upload: bool = True
    refresh: bool = False
    require_existing: bool = False


@dataclass(frozen=True)
class _PostSubmitCopy:
    """Device-to-host copy to enqueue after an L3 child task has drained."""

    dst: int
    src: int
    nbytes: int


class Qwen314BModelRunner(ModelRunner):
    """Runtime wrapper for one Qwen3-14B model's compiled PyPTO kernels."""

    def __init__(
        self,
        *,
        model_id: str,
        compiled: _CompiledKernels,
        platform: str,
        device_id: int,
        save_kernels_dir: str | None,
        l3_trace: bool,
    ) -> None:
        super().__init__()
        self._model_id = model_id
        self._compiled = compiled
        self._platform = platform
        self._device_id = device_id
        self._save_kernels_dir = save_kernels_dir
        self._l3_trace = l3_trace
        self._l2_workers: dict[str, LlmWorker] = {}
        self._l2_programs: dict[int, _L2ProgramHandle] = {}
        self._l2_child_allocs: dict[tuple[str, int], tuple[int, int]] = {}
        self._l2_dirty_kv_models: set[str] = set()

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run the JIT all-layer prefill kernel and return next-token logits."""
        compiled = self._compiled
        prefill_inputs = self._prepare_prefill_inputs(model, batch)
        dw = compiled.decode_weights
        t_prefill_start = time.perf_counter()

        k_cache, v_cache = self.materialize_full_layer_cache(
            model.config.model_id,
        )
        logits_padded = torch.zeros(
            (prefill_inputs.actual_batch, compiled.padded_vocab),
            dtype=torch.float32,
        ).share_memory_()

        self._run_l2_program(
            compiled.prefill,
            prefill_inputs.hidden,
            prefill_inputs.seq_lens,
            prefill_inputs.chunk_lens,
            prefill_inputs.chunk_offsets,
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_input_rms_weight"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_wq"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_wk"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_wv"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_q_norm_weight"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_k_norm_weight"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, compiled.rope_cos),
            self._l2_child_tensor(compiled.prefill.runtime_name, compiled.rope_sin),
            prefill_inputs.block_table,
            prefill_inputs.slot_mapping,
            k_cache,
            v_cache,
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_wo"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_post_rms_weight"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_w_gate"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_w_up"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, dw["decode_w_down"]),
            self._l2_child_tensor(compiled.prefill.runtime_name, compiled.final_norm_weight),
            self._l2_child_tensor(compiled.prefill.runtime_name, compiled.padded_lm_head_weight),
            logits_padded,
        )
        self._l2_dirty_kv_models.add(model.config.model_id)

        if _TIMING_ENABLED:
            print(
                f"[timing] prefill: fused {len(model.layers)} layers, "
                f"{(time.perf_counter() - t_prefill_start) * 1000:.2f} ms",
                flush=True,
            )

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
        return PrefillResult(
            last_hidden=None,
            logits=logits_padded[:, : model.config.vocab_size],
        )

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run the JIT all-layer decode kernel and return next-token logits."""
        compiled = self._compiled
        decode_inputs = self._prepare_decode_inputs(model, batch)
        hidden = decode_inputs.hidden
        dw = compiled.decode_weights

        k_cache, v_cache = self.materialize_full_layer_cache(
            model.config.model_id,
        )

        if decode_inputs.actual_batch > compiled.decode_logits_buffer.shape[0]:
            raise ValueError(
                f"decode batch {decode_inputs.actual_batch} exceeds logits buffer batch "
                f"{compiled.decode_logits_buffer.shape[0]}"
            )
        logits_padded = compiled.decode_logits_buffer[: decode_inputs.actual_batch]
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
            k_cache,
            v_cache,
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_wo"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_post_rms_weight"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_gate"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_up"]),
            self._l2_child_tensor(compiled.decode.runtime_name, dw["decode_w_down"]),
            self._l2_child_tensor(compiled.decode.runtime_name, compiled.final_norm_weight),
            self._l2_child_tensor(compiled.decode.runtime_name, compiled.padded_lm_head_weight),
            logits_padded,
        )
        self._l2_dirty_kv_models.discard(model.config.model_id)
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        return DecodeResult(
            hidden_states=hidden.float(),
            logits=logits_padded[:, : model.config.vocab_size].to(hidden.device),
        )

    def _project_logits(self, model: RuntimeModel, hidden: torch.Tensor) -> torch.Tensor:
        """Run final RMSNorm and LM head kernels for a hidden-state batch."""
        compiled = self._compiled
        hidden_size = model.config.hidden_size
        vocab_size = model.config.vocab_size
        padded_vocab = compiled.padded_vocab

        if compiled.final_rms is None or compiled.lm_head is None:
            raise RuntimeError("standalone final_rms/lm_head kernels are not compiled")

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

        normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16).share_memory_()
        normed_arg: torch.Tensor | _DeferredChildTensor = normed
        if compiled.final_rms.runtime_name == compiled.lm_head.runtime_name:
            normed_arg = self._l2_child_tensor(compiled.final_rms.runtime_name, normed, upload=False)

        self._run_l2_program(
            compiled.final_rms,
            x,
            self._l2_child_tensor(compiled.final_rms.runtime_name, compiled.final_norm_weight),
            normed_arg,
        )

        lm_head_input: torch.Tensor | _DeferredChildTensor = normed
        if isinstance(normed_arg, _DeferredChildTensor):
            lm_head_input = self._l2_child_tensor(
                compiled.lm_head.runtime_name,
                normed,
                upload=False,
                require_existing=True,
            )

        logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32).share_memory_()
        self._run_l2_program(
            compiled.lm_head,
            lm_head_input,
            self._l2_child_tensor(compiled.lm_head.runtime_name, compiled.padded_lm_head_weight),
            logits_padded,
        )
        return logits_padded[:actual_batch, :vocab_size].to(hidden.device)

    def _run_l2_program(self, callable_spec: _L2Callable, *args: Any) -> None:
        """Run one compiled chip program through a wrapped L3 worker."""
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        self._share_l2_host_args(args)
        handle = self._ensure_l2_program(callable_spec)

        cfg = CallConfig()
        cfg.block_dim = callable_spec.block_dim
        cfg.aicpu_thread_num = callable_spec.aicpu_thread_num

        worker = self._l2_workers[handle.runtime_name]
        post_submit_copies: list[_PostSubmitCopy] = []

        def _orch_fn(orchestrator: Any, _args: Any, _config: Any) -> None:
            orch_args, copies = self._build_l2_orch_args(
                callable_spec,
                args,
                worker=worker,
                orchestrator=orchestrator,
            )
            post_submit_copies.extend(copies)
            worker.submit_next_level(
                handle.callable_id,
                orch_args,
                cfg,
                worker_id=0,
                orchestrator=orchestrator,
            )

        try:
            worker.run(_orch_fn)
            if post_submit_copies:
                def _copy_back_orch_fn(orchestrator: Any, _args: Any, _config: Any) -> None:
                    for copy in post_submit_copies:
                        worker.copy_from(
                            copy.dst,
                            copy.src,
                            copy.nbytes,
                            worker_id=0,
                            orchestrator=orchestrator,
                        )

                worker.run(_copy_back_orch_fn)
        finally:
            self._discard_l2_worker(handle.runtime_name)

    def _worker_for_runtime(self, runtime_name: str) -> LlmWorker:
        """Return a wrapped L3 worker for ``runtime_name``."""
        worker = self._l2_workers.get(runtime_name)
        if worker is not None:
            return worker
        for active_runtime_name in list(self._l2_workers):
            if active_runtime_name != runtime_name:
                self._close_l2_worker(active_runtime_name)
        worker = LlmWorker(
            level=3,
            platform=self._platform,
            runtime=runtime_name,
            device_ids=[self._device_id],
            num_sub_workers=0,
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
        if not worker.initialized:
            handle = self._register_l2_program(callable_spec, worker)
            self._register_known_l2_programs_for_runtime(callable_spec.runtime_name, worker)
            worker.init()
            return handle

        return self._register_l2_program(callable_spec, worker)

    def _register_l2_program(self, callable_spec: _L2Callable, worker: LlmWorker) -> _L2ProgramHandle:
        """Register ``callable_spec`` on ``worker`` once and cache the callable id."""
        key = id(callable_spec)
        cached = self._l2_programs.get(key)
        if cached is not None:
            return cached
        handle = _L2ProgramHandle(
            callable_id=worker.register(callable_spec.chip_callable),
            runtime_name=callable_spec.runtime_name,
        )
        self._l2_programs[key] = handle
        return handle

    def _register_known_l2_programs_for_runtime(self, runtime_name: str, worker: LlmWorker) -> None:
        """Pre-register known callables for a runtime before the L3 worker starts."""
        compiled = self._compiled
        if not isinstance(compiled, _CompiledKernels):
            return
        for callable_spec in (
            compiled.prefill,
            compiled.decode,
            compiled.final_rms,
            compiled.lm_head,
        ):
            if isinstance(callable_spec, _L2Callable) and callable_spec.runtime_name == runtime_name:
                self._register_l2_program(callable_spec, worker)

    def _l2_child_tensor(
        self,
        runtime_name: str,
        tensor: torch.Tensor,
        *,
        upload: bool = True,
        refresh: bool = False,
        require_existing: bool = False,
    ) -> _DeferredChildTensor:
        """Return a deferred worker-resident view for a CPU tensor backing storage."""
        if tensor.device.type != "cpu":
            raise ValueError("child-memory tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("child-memory tensor must be contiguous")
        return _DeferredChildTensor(
            runtime_name=runtime_name,
            tensor=self._share_cpu_tensor(tensor),
            upload=upload,
            refresh=refresh,
            require_existing=require_existing,
        )

    def _resolve_l3_child_tensor(
        self,
        deferred: _DeferredChildTensor,
        *,
        worker: LlmWorker,
        orchestrator: Any,
    ) -> tuple[WorkerTensor, _PostSubmitCopy | None]:
        """Allocate or reuse a child-memory tensor inside an L3 orchestration scope."""
        from simpler_setup.torch_interop import torch_dtype_to_datatype  # noqa: PLC0415

        tensor = deferred.tensor
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
        tensor_offset = int(tensor.data_ptr()) - storage_ptr
        if tensor_offset < 0 or tensor_offset + int(tensor.nbytes) > storage_nbytes:
            raise ValueError("tensor view is outside its backing storage")

        key = (deferred.runtime_name, storage_ptr)
        alloc = self._l2_child_allocs.get(key)
        if alloc is None:
            if deferred.require_existing:
                raise RuntimeError("missing worker-resident tensor allocation")
            dev_ptr = worker.malloc(storage_nbytes, worker_id=0, orchestrator=orchestrator)
            if deferred.upload:
                worker.copy_to(dev_ptr, storage_ptr, storage_nbytes, worker_id=0, orchestrator=orchestrator)
            alloc = (dev_ptr, storage_nbytes)
            self._l2_child_allocs[key] = alloc
        elif deferred.upload and deferred.refresh:
            worker.copy_to(alloc[0], storage_ptr, storage_nbytes, worker_id=0, orchestrator=orchestrator)

        dev_base, _ = alloc
        shape = tuple(int(dim) for dim in tensor.shape)
        worker_tensor = WorkerTensor(
            data_ptr=dev_base + tensor_offset,
            shape=shape,
            dtype=torch_dtype_to_datatype(tensor.dtype),
        )
        return worker_tensor, _PostSubmitCopy(storage_ptr, dev_base, storage_nbytes)

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor's storage to shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    @classmethod
    def _share_l2_host_args(cls, args: tuple[Any, ...]) -> None:
        """Share host tensor mappings before L3 child workers are initialized."""
        for arg in args:
            if isinstance(arg, _DeferredChildTensor):
                cls._share_cpu_tensor(arg.tensor)
            elif isinstance(arg, torch.Tensor):
                cls._share_cpu_tensor(arg)

    def close(self) -> None:
        """Release non-L3 child-memory allocations and L2 workers."""
        for runtime_name in list(self._l2_workers):
            self._close_l2_worker(runtime_name)
        self._l2_dirty_kv_models.clear()

    def _close_l2_worker(self, runtime_name: str) -> None:
        """Close one wrapped L3 worker and forget runtime-local child state."""
        worker = self._l2_workers.pop(runtime_name, None)
        for key, (dev_ptr, _nbytes) in list(self._l2_child_allocs.items()):
            alloc_runtime_name, _storage_ptr = key
            if alloc_runtime_name != runtime_name:
                continue
            self._l2_child_allocs.pop(key, None)
            if worker is not None and worker.initialized and worker.level < 3:
                worker.free(dev_ptr)
        for key, handle in list(self._l2_programs.items()):
            if handle.runtime_name == runtime_name:
                self._l2_programs.pop(key, None)
        if worker is not None:
            worker.close()

    def _discard_l2_worker(self, runtime_name: str) -> None:
        """Discard one-shot L3 worker state after a submitted chip task."""
        worker = self._l2_workers.pop(runtime_name, None)
        for key in list(self._l2_child_allocs):
            alloc_runtime_name, _storage_ptr = key
            if alloc_runtime_name == runtime_name:
                self._l2_child_allocs.pop(key, None)
        for key, handle in list(self._l2_programs.items()):
            if handle.runtime_name == runtime_name:
                self._l2_programs.pop(key, None)
        if worker is not None:
            worker.discard_l3_children()

    def _build_l2_orch_args(
        self,
        callable_spec: _L2Callable,
        args: tuple[Any, ...],
        *,
        worker: LlmWorker | None = None,
        orchestrator: Any = None,
    ):
        """Build ``TaskArgs`` for submitting a compiled child chip program."""
        from simpler.task_interface import (  # noqa: PLC0415
            ContinuousTensor,
            TaskArgs,
            TensorArgType,
            scalar_to_uint64,
        )
        from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

        param_infos = callable_spec.param_infos
        if len(args) != len(param_infos):
            names = [p.name for p in param_infos]
            raise TypeError(
                f"compiled program expects {len(param_infos)} arguments, got {len(args)}. Parameters: {names}"
            )

        orch_args = TaskArgs()
        post_submit_copies: list[_PostSubmitCopy] = []
        for arg_index, (info, arg) in enumerate(zip(param_infos, args, strict=True)):
            if info.shape is None:
                if not isinstance(arg, ctypes._SimpleCData):
                    raise TypeError(f"scalar parameter {info.name!r} must be passed as a ctypes scalar")
                orch_args.add_scalar(scalar_to_uint64(arg))
                continue
            tag = self._l2_tensor_arg_type(info, arg, arg_index, len(args), TensorArgType)
            if isinstance(arg, _DeferredChildTensor):
                if worker is None or orchestrator is None:
                    raise RuntimeError("deferred child tensor requires L3 worker and orchestrator")
                arg, copy = self._resolve_l3_child_tensor(
                    arg,
                    worker=worker,
                    orchestrator=orchestrator,
                )
                if tag == TensorArgType.INOUT and copy is not None:
                    post_submit_copies.append(copy)
            if isinstance(arg, WorkerTensor):
                orch_args.add_tensor(arg.to_continuous_tensor(), tag)
                continue
            if isinstance(arg, ContinuousTensor):
                orch_args.add_tensor(arg, tag)
                continue
            if not isinstance(arg, torch.Tensor):
                raise TypeError(f"tensor parameter {info.name!r} expects torch.Tensor, got {type(arg).__name__}")
            if arg.device.type != "cpu":
                raise ValueError(f"tensor parameter {info.name!r} must be on CPU for Simpler L2 dispatch")
            if not arg.is_contiguous():
                raise ValueError(f"tensor parameter {info.name!r} must be contiguous")
            if not arg.is_shared():
                arg.share_memory_()
            orch_args.add_tensor(make_tensor_arg(arg), tag)
        return orch_args, post_submit_copies

    @staticmethod
    def _l2_tensor_arg_type(
        info: object,
        arg: object,
        arg_index: int,
        arg_count: int,
        tensor_arg_type: object,
    ) -> object:
        """Translate PyPTO parameter metadata into a Simpler tensor argument tag."""
        name = getattr(info, "name", "")
        if name in {"k_cache", "v_cache"}:
            return tensor_arg_type.INOUT

        direction = getattr(info, "direction", None)
        direction_name = getattr(direction, "name", None)
        if direction_name is None and direction is not None:
            direction_name = str(direction).rsplit(".", maxsplit=1)[-1]
        if direction_name == "InOut":
            return tensor_arg_type.INOUT
        if direction_name == "Out":
            return tensor_arg_type.OUTPUT_EXISTING

        if name in {"out", "output"}:
            return tensor_arg_type.OUTPUT_EXISTING
        if isinstance(arg, torch.Tensor) and arg_index == arg_count - 1:
            return tensor_arg_type.OUTPUT_EXISTING
        return tensor_arg_type.INPUT

    # ── L3-wrapped generate: entire prefill + decode loop in one worker.run() ──

    def run_generate_l3(
        self,
        model: RuntimeModel,
        prefill_batch: PrefillBatch,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> tuple[list[int], torch.Tensor]:
        """Run the full generate loop inside a single Worker(level=3).

        Dispatches prefill chunks + final_rms + lm_head + decode loop entirely
        within one worker.run() call, using sub_worker for CPU-side sampling
        and embedding lookup between device dispatches.

        Returns (generated_token_ids, final_hidden).
        """
        from simpler.task_interface import CallConfig  # noqa: PLC0415
        from simpler.worker import Worker  # noqa: PLC0415

        _verbose = self._l3_trace
        timer = StageTimer(
            enabled=_verbose,
            prefix="L3-breakdown",
            title="run_generate_l3 stage timings",
        )

        def _mark(label: str) -> None:
            timer.mark(label)

        compiled = self._compiled
        if compiled.l3_generate_chip_callables is None:
            raise RuntimeError("L3 generate artifacts not compiled.")
        if max_new_tokens > model.runtime.max_new_tokens:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} exceeds compiled L3 limit "
                f"{model.runtime.max_new_tokens}"
            )

        _mark("entry_validate")
        prefill_inputs = self._prepare_prefill_inputs(model, prefill_batch, padded=True)
        _mark("prepare_prefill_inputs")
        actual_batch = prefill_inputs.actual_batch
        if actual_batch != 1:
            raise ValueError(
                "run_generate_l3 currently supports batch_size=1 only; "
                f"got {actual_batch} requests."
            )
        hidden_size = model.config.hidden_size
        max_seq = model.runtime.max_seq_len
        vocab_size = model.config.vocab_size
        padded_vocab = compiled.padded_vocab

        k_cache_all, v_cache_all = self.materialize_full_layer_cache(
            model.config.model_id,
        )
        _mark("kv_cache_materialize")

        # Build initial decode hidden: last prompt token embedding per batch.
        decode_hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        decode_slot_mapping = torch.zeros((actual_batch,), dtype=torch.int32)
        for b in range(actual_batch):
            seq_len_b = int(prefill_inputs.seq_lens[b].item())
            decode_hidden[b] = prefill_inputs.hidden[b, seq_len_b - 1, :]
            decode_slot_mapping[b] = int(
                prefill_inputs.slot_mapping[b * max_seq + seq_len_b - 1].item()
            )

        # Pre-allocate all shared-memory tensors for the full generate loop.
        prefill_out = torch.zeros_like(prefill_inputs.hidden).share_memory_()
        # decode_out and rms_x share one padded buffer so no CPU copy is needed
        # between them.  l3_generate writes to decode_out (the first actual_batch
        # rows); final_rms reads rms_x (all _LOGITS_BATCH_TILE rows).  The padding
        # rows stay zero throughout, satisfying the zero-pad contract of final_rms.
        _decode_out_storage = torch.zeros(
            (_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16
        ).share_memory_()
        decode_out = _decode_out_storage[:actual_batch, :]   # [actual_batch, hidden_size]
        rms_x = _decode_out_storage                          # [_LOGITS_BATCH_TILE, hidden_size]
        # final_rms / lm_head intermediates.
        rms_gamma = model.final_norm_weight.view(1, hidden_size).float().cpu().share_memory_()
        rms_normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16).share_memory_()
        lm_head_weight = compiled.padded_lm_head_weight.share_memory_()
        logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32).share_memory_()
        _mark("alloc_io_buffers")
        # Sub-worker communication tensors.
        embed_tokens = model.embed_tokens.to(torch.bfloat16).cpu().share_memory_()
        _mark("embed_tokens_to_shm")
        done_flag = torch.zeros((1,), dtype=torch.int32).share_memory_()
        generated_ids = torch.full((max_new_tokens,), -1, dtype=torch.int64).share_memory_()
        token_count = torch.zeros((1,), dtype=torch.int32).share_memory_()
        # Mutable decode inputs (updated by sub_worker between steps).
        decode_hidden_buf = decode_hidden.clone().share_memory_()
        decode_seq_lens = prefill_inputs.seq_lens.clone().share_memory_()
        decode_slot_mapping_buf = decode_slot_mapping.clone().share_memory_()

        # Ensure all prefill inputs are in shared memory.
        prefill_hidden = prefill_inputs.hidden.share_memory_()
        prefill_seq_lens = prefill_inputs.seq_lens.share_memory_()
        prefill_slot_mapping = prefill_inputs.slot_mapping.share_memory_()
        block_table = prefill_inputs.block_table.share_memory_()
        rope_cos = compiled.rope_cos.share_memory_()
        rope_sin = compiled.rope_sin.share_memory_()
        k_cache_all = k_cache_all.share_memory_()
        v_cache_all = v_cache_all.share_memory_()
        _mark("kv_and_prefill_to_shm")

        # Ensure stacked weights are in shared memory.
        sm_sw = {}
        for k, v in compiled.stacked_weights.items():
            sm_sw[k] = v.share_memory_() if not v.is_shared() else v
        _mark("stacked_weights_to_shm")

        # SSA-name substrings that identify static weight parameters in the
        # tensors_dict built by _build_full_tensors().  Used in
        # generate_orch_fn to pre-upload these tensors once per generate call
        # (child_memory=True) so all decode dispatches skip H2D re-upload.
        _sw_substrings: frozenset[str] = frozenset(sm_sw.keys()) | {
            "rope_cos", "rope_sin", "final_norm_weight", "lm_head_weight",
        }

        # ── Sub-worker callable ──
        # Runs in a forked child process. Reads logits → argmax → embedding lookup
        # → writes decode_hidden_buf / decode_seq_lens / decode_slot_mapping_buf.
        _eos_id = eos_token_id
        _vocab = vocab_size
        _actual_batch = actual_batch
        _page_size = model.runtime.page_size

        # Pre-capture references for the sub_worker closure.
        # These are shared-memory tensors visible in the forked child.
        _embed_tokens = embed_tokens
        _done_flag = done_flag
        _generated_ids = generated_ids
        _token_count = token_count
        _decode_hidden_buf = decode_hidden_buf
        _decode_seq_lens = decode_seq_lens
        _decode_slot_mapping_buf = decode_slot_mapping_buf
        _logits_padded = logits_padded
        _prefill_batch = prefill_batch

        # L3 tracing: enabled by the executor's l3_trace flag (typically wired
        # to --profile-verbose). When disabled, all per-stage timestamp prints
        # are suppressed.
        _l3_trace_enabled = self._l3_trace

        # Shared-memory anchors so forked sub-worker callbacks can print times
        # relative to a common start (set at prefill submit_start in the parent).
        # Slot 0: start anchor (perf_counter seconds).
        # Slot 1: timestamp of the previous printed event (for Δprev).
        _t_anchors = torch.zeros(2, dtype=torch.float64).share_memory_()

        def _fmt_rel(t_now: float) -> str:
            t0 = float(_t_anchors[0].item())
            t_prev = float(_t_anchors[1].item())
            if t0 <= 0.0:
                return "t=+0.000ms (Δ+0.000ms)"
            rel_ms = (t_now - t0) * 1000.0
            d_ms = (t_now - t_prev) * 1000.0 if t_prev > 0.0 else 0.0
            _t_anchors[1] = t_now
            return f"t=+{rel_ms:.2f}ms (Δ+{d_ms:.2f}ms)"

        # Track when each sample_and_prepare finishes so we can split
        # chip-task time vs sub-worker IPC + work time.
        _sample_done_ts = [0.0]  # timestamp of last sample_and_prepare return

        def sample_and_prepare_fn(task_args):
            """Sub-worker: sample token from logits, prepare next decode inputs."""
            _t_enter = time.perf_counter()
            if _done_flag[0].item():
                return  # EOS already hit, no-op.

            # Read logits (written by lm_head chip task into logits_padded).
            logits = _logits_padded[0, :_vocab]
            token_id = int(logits.argmax().item())

            step = int(_token_count[0].item())
            if _l3_trace_enabled:
                _prev_done = _sample_done_ts[0]
                chip_ms = (_t_enter - _prev_done) * 1000.0 if _prev_done > 0.0 else 0.0
                print(
                    f"[L3-step] step={step:02d} sample_enter {_fmt_rel(_t_enter)}"
                    f"  chip_tasks={chip_ms:.1f}ms",
                    flush=True,
                )
            _generated_ids[step] = token_id
            _token_count[0] = step + 1

            if _eos_id is not None and token_id == _eos_id:
                _done_flag[0] = 1
                _t_exit = time.perf_counter()
                _sample_done_ts[0] = _t_exit
                if _l3_trace_enabled:
                    work_ms = (_t_exit - _t_enter) * 1000.0
                    print(
                        f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                        f"  sample_work={work_ms:.1f}ms",
                        flush=True,
                    )
                return

            if step + 1 >= max_new_tokens:
                _done_flag[0] = 1
                _t_exit = time.perf_counter()
                _sample_done_ts[0] = _t_exit
                if _l3_trace_enabled:
                    work_ms = (_t_exit - _t_enter) * 1000.0
                    print(
                        f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                        f"  sample_work={work_ms:.1f}ms",
                        flush=True,
                    )
                return

            # Embedding lookup.
            _decode_hidden_buf[0, :] = _embed_tokens[token_id]

            # Update seq_lens.
            for b in range(_actual_batch):
                new_seq_len = int(_decode_seq_lens[b].item()) + 1
                _decode_seq_lens[b] = new_seq_len
                # Update slot_mapping for next position.
                alloc = _prefill_batch.kv_allocations[b]
                page_idx = (new_seq_len - 1) // _page_size
                slot_in_page = (new_seq_len - 1) % _page_size
                if page_idx < len(alloc.page_ids):
                    _decode_slot_mapping_buf[b] = alloc.page_ids[page_idx] * _page_size + slot_in_page

            _t_exit = time.perf_counter()
            _sample_done_ts[0] = _t_exit
            if _l3_trace_enabled:
                work_ms = (_t_exit - _t_enter) * 1000.0
                print(
                    f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                    f"  sample_work={work_ms:.1f}ms",
                    flush=True,
                )

        # ── Build the orchestrator function ──

        lg_entry_fn = compiled.l3_generate_entry_fn
        lg_chip_callables = compiled.l3_generate_chip_callables
        lg_dc = compiled.l3_generate_dc
        lg_param_infos = compiled.l3_generate_param_infos

        def _submit_l3_generate(orch, config, tensors_dict, _keep):
            """Submit one l3_generate entry_fn call (dispatches all-layers L2 tasks)."""
            kwargs = {
                "tensors": tensors_dict,
                "callables": lg_callable_ids,
                "sub_ids": sub_ids,
                "_keep": _keep,
            }
            entry_params = inspect.signature(lg_entry_fn).parameters
            if "contexts" in entry_params:
                kwargs["contexts"] = getattr(worker, "chip_contexts", None)
            if "world_size" in entry_params:
                kwargs["world_size"] = 1
            lg_entry_fn(
                orch, None, config,
                **kwargs,
            )

        _has_prefill_tensor = torch.tensor(True, dtype=torch.bool).share_memory_()

        def _build_full_tensors():
            """Build the tensor dict for the single l3_generate dispatch.

            host_orch now owns the full generation loop (prefill step 0 +
            pl.unroll(max_new_tokens) decode steps), so this is called once.
            has_prefill is always True: step 0 inside host_orch runs prefill_all
            then the first decode; subsequent iterations run decode-only.
            """
            td = {}
            for info, val in zip(lg_param_infos, [
                prefill_hidden,           # prefill_hidden
                prefill_seq_lens,         # prefill_seq_lens
                prefill_slot_mapping,     # prefill_slot_mapping
                decode_hidden_buf,        # decode_hidden
                decode_seq_lens,          # decode_seq_lens (mutated by sample_and_prepare sub-worker)
                decode_slot_mapping_buf,  # decode_slot_mapping
                sm_sw["input_rms_weight"],
                sm_sw["wq"],
                sm_sw["wk"],
                sm_sw["wv"],
                sm_sw["q_norm_weight"],
                sm_sw["k_norm_weight"],
                rope_cos,
                rope_sin,
                block_table,
                k_cache_all,
                v_cache_all,
                sm_sw["wo"],
                sm_sw["post_rms_weight"],
                sm_sw["w_gate"],
                sm_sw["w_up"],
                sm_sw["w_down"],
                _has_prefill_tensor,      # has_prefill = True
                prefill_out,              # prefill_out
                decode_out,               # decode_out
                rms_x,                    # rms_x  (shares storage with decode_out)
                rms_gamma,                # final_norm_weight
                rms_normed,               # rms_normed
                lm_head_weight,           # lm_head_weight_t
                logits_padded,            # logits_padded
            ], strict=True):
                if not val.is_shared():
                    val = val.share_memory_()
                td[info.name] = val
            return td

        # KV cache device pointers collected inside generate_orch_fn so the
        # post-run sync-back step can copy updated K/V values back to host.
        # Format: list of (host_ptr: int, dev_ptr: int, nbytes: int).
        _kv_dev_ptrs: list[tuple[int, int, int]] = []

        def generate_orch_fn(orch, _args, _cfg):
            _keep: list = []
            call_config = CallConfig()
            call_config.block_dim = lg_dc.block_dim
            call_config.aicpu_thread_num = lg_dc.aicpu_thread_num

            _t_pf = time.perf_counter()
            _t_anchors[0] = _t_pf
            _t_anchors[1] = _t_pf
            _sample_done_ts[0] = _t_pf  # baseline for step 0 chip_tasks measurement
            if _l3_trace_enabled:
                print(f"[L3-step] host_orch submit_start {_fmt_rel(_t_pf)}", flush=True)

            # Single dispatch: host_orch drives prefill + all decode steps
            # (pl.unroll(max_new_tokens) inside host_orch).
            td = _build_full_tensors()

            from simpler.task_interface import ContinuousTensor as _CT  # noqa: PLC0415
            from simpler_setup.torch_interop import (  # noqa: PLC0415
                torch_dtype_to_datatype as _td2dt,
            )

            # Pre-upload static weight tensors once per generate call.
            # child_memory=True → runtime skips H2D + D2H on every dispatch.
            # ~3 400 ms of init_runtime per step reduced to ~11 ms.
            for _pname, _t in list(td.items()):
                if not isinstance(_t, torch.Tensor):
                    continue
                if not any(_sub in _pname for _sub in _sw_substrings):
                    continue
                _nbytes = int(_t.nbytes)
                _dev_ptr = orch.malloc(worker_id=0, size=_nbytes)
                orch.copy_to(worker_id=0, dst=_dev_ptr, src=_t.data_ptr(), size=_nbytes)
                _shapes = tuple(int(s) for s in _t.shape)
                _dt = _td2dt(_t.dtype)
                td[_pname] = _CT.make(_dev_ptr, _shapes, _dt, child_memory=True)

            # Pre-upload KV cache (k_cache_all / v_cache_all) once per
            # generate call.  The kernel writes updated K/V values in-place on
            # device; child_memory=True skips H2D and D2H on every decode step,
            # saving ~280 ms H2D + ~360 ms D2H per step (~640 ms × 16 steps).
            # After worker.run() drains (all tasks done), a second worker.run()
            # call copies the final device state back to the host KV cache via
            # orch.copy_from so subsequent generate calls see updated values.
            for _pname, _t in list(td.items()):
                if not isinstance(_t, torch.Tensor):
                    continue
                if "k_cache_all" not in _pname and "v_cache_all" not in _pname:
                    continue
                _nbytes = int(_t.nbytes)
                _dev_ptr = orch.malloc(worker_id=0, size=_nbytes)
                orch.copy_to(worker_id=0, dst=_dev_ptr, src=_t.data_ptr(), size=_nbytes)
                _shapes = tuple(int(s) for s in _t.shape)
                _dt = _td2dt(_t.dtype)
                _kv_dev_ptrs.append((_t.data_ptr(), _dev_ptr, _nbytes))
                td[_pname] = _CT.make(_dev_ptr, _shapes, _dt, child_memory=True)

            _submit_l3_generate(orch, call_config, td, _keep)

            if _l3_trace_enabled:
                print(f"[L3-step] all submits done {_fmt_rel(time.perf_counter())}", flush=True)

        # ── Create Worker and execute ──

        lg_sub_fns = dict(compiled.l3_generate_sub_worker_fns or {})
        # Override the placeholder sample_and_prepare sub-worker emitted by the
        # l3_generate compiler with the real closure that reads shared-memory
        # tensors and performs argmax → embedding lookup → slot-map update.
        lg_sub_fns["sample_and_prepare"] = sample_and_prepare_fn

        num_sub = max(lg_dc.num_sub_workers, len(lg_sub_fns))

        _mark("setup_closures_and_buffers")

        worker = Worker(
            level=3,
            device_ids=[self._device_id],
            num_sub_workers=num_sub,
            platform=compiled.l3_generate_platform,
            runtime=compiled.l3_generate_runtime_name,
        )

        _mark("Worker_ctor")

        # Register chip callables and sub-worker callables before worker.init().
        lg_callable_ids: dict[str, int] = {}
        for name, callable_obj in lg_chip_callables.items():
            lg_callable_ids[name] = worker.register(callable_obj)

        sub_ids: dict[str, int] = {}
        for name, fn in lg_sub_fns.items():
            sub_ids[name] = worker.register(fn)

        _mark("worker_register")

        worker.init()
        _mark("worker_init")
        try:
            _t_run_start = time.perf_counter()
            worker.run(generate_orch_fn)
            _mark("worker_run_generate")

            # Sync KV cache back to host.  worker.run() above calls _drain()
            # internally, so all chip tasks (including the last decode step)
            # have completed by the time we reach here.  The child_memory
            # buffers are still live (worker.close() not yet called), so a
            # second worker.run() can copy them back to the host tensors.
            if _kv_dev_ptrs:
                def _kv_sync_orch_fn(orch, _args, _cfg):  # noqa: E306
                    for _host_ptr, _dev_ptr, _nbytes in _kv_dev_ptrs:
                        orch.copy_from(
                            worker_id=0, dst=_host_ptr, src=_dev_ptr, size=_nbytes,
                        )
                worker.run(_kv_sync_orch_fn)
            _mark("worker_run_kv_sync")

            _t_run_end = time.perf_counter()
            print(
                f"[L3-timer] worker.run total wall-clock: "
                f"{(_t_run_end-_t_run_start)*1000:.1f}ms",
                flush=True,
            )
        finally:
            worker.close()
            _mark("worker_close")

        # Update KV allocations.
        final_token_count = int(token_count[0].item())
        for batch_idx, alloc in enumerate(prefill_batch.kv_allocations):
            base_seq = int(prefill_inputs.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, base_seq + final_token_count)

        ids = generated_ids[:final_token_count].tolist()
        ret_val = ids, decode_out[:actual_batch].float()
        _mark("post_process")

        timer.report()
        return ret_val

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
        *,
        padded: bool = False,
    ) -> _PrefillInputs:
        """Pack variable-length prefill requests into kernel input tensors."""
        batch_count = len(batch.kv_allocations) if batch.kv_allocations else int(batch.seq_lens.shape[0])
        actual_batch = self._validate_batch_size(model, batch_count)
        max_seq = model.runtime.max_seq_len
        hidden_size = model.config.hidden_size
        page_size = self._kv_page_sizes.get(model.config.model_id, model.runtime.page_size)
        max_blocks = self._max_blocks_per_seq(model)
        if padded:
            if batch.kv_allocations:
                max_blocks = max(max_blocks, max(len(alloc.page_ids) for alloc in batch.kv_allocations))

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
        if padded:
            hidden = torch.zeros((actual_batch, max_seq, hidden_size), dtype=torch.bfloat16)
            slot_mapping = torch.full((actual_batch * max_seq,), -1, dtype=torch.int32)
        else:
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
            if padded:
                hidden[batch_idx, chunk_start:seq_len, :] = embeddings
            else:
                hidden[token_offset : token_offset + chunk_len, :] = embeddings

            if alloc is not None:
                page_ids = alloc.page_ids
            elif batch_idx < len(batch.block_ids):
                page_ids = batch.block_ids[batch_idx]
            else:
                page_ids = []
            self._write_block_table_row(block_table, batch_idx, max_blocks, page_ids)

            slot_row = self._compute_slot_mapping(page_ids, chunk_len, page_size, start_pos=chunk_start)
            if padded:
                slot_mapping[batch_idx * max_seq + chunk_start : batch_idx * max_seq + seq_len] = slot_row
            else:
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
        page_size = self._kv_page_sizes.get(model.config.model_id, model.runtime.page_size)
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
