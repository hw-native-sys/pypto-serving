# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from examples.model.qwen3_14b.runner.npu_runner import (
    _CompiledKernels,
    _L3Callable,
    Qwen314BModelRunner,
)
from examples.model.qwen3_14b.runner import qwen3_l3_dispatch
from python.core._profiling import StageTimer
from python.core.model_runner import ModelRunner
from python.core.pypto_executor import PyptoExecutor as CorePyptoExecutor
from python.core.types import RuntimeModel
from python.core.utils import rope_tables, round_up


_VOCAB_PAD_MULTIPLE = 512  # must be a multiple of lm_head.VOCAB_CHUNK (64)
_QWEN14B_PAGE_SIZE = 128
_QWEN14B_BLOCK_DIM = 24


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


def _find_pypto_lib_qwen14b_dir() -> Path:
    """Find the Qwen3-14B kernel directory in the pypto-lib submodule."""
    start_dir = Path(__file__).resolve().parent
    for directory in (start_dir, *start_dir.parents):
        pypto_lib_dir = directory / "pypto-lib"
        if pypto_lib_dir.is_dir() or (directory / ".gitmodules").is_file():
            return pypto_lib_dir / "models" / "qwen3" / "14b"
    raise FileNotFoundError(
        "Cannot locate the pypto-lib submodule from npu_executor.py. "
        "Run from a pypto-serving checkout with `git submodule update --init --recursive`."
    )


_PYPTO_LIB_QWEN14B_DIR = _find_pypto_lib_qwen14b_dir()


def _load_pypto_lib_qwen14b_module(module_name: str) -> object:
    """Load a Qwen3-14B kernel module from the pypto-lib submodule."""
    module_path = _PYPTO_LIB_QWEN14B_DIR / f"qwen3_14b_{module_name}.py"
    if not module_path.is_file():
        module_path = _PYPTO_LIB_QWEN14B_DIR / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Missing pypto-lib Qwen3-14B kernel module: {module_path}. "
            "Run `git submodule update --init --recursive`."
        )
    spec = importlib.util.spec_from_file_location(
        f"_pypto_lib_qwen3_14b_{module_name}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pypto-lib kernel module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_PYPTO_LIB_QWEN14B_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(_PYPTO_LIB_QWEN14B_DIR))
        except ValueError:
            pass
    return module


class Qwen314BPyptoExecutor(CorePyptoExecutor):
    """PyPTO executor that compiles and registers the Qwen3-14B kernels."""

    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform: str = "a2a3sim",
        device_id: int = 0,
        device_ids: Sequence[int] | None = None,
        save_kernels_dir: str | None = None,
        l3_trace: bool = False,
    ) -> None:
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_id=device_id,
            device_ids=device_ids,
            save_kernels_dir=save_kernels_dir,
        )
        self._l3_trace = l3_trace

    @property
    def profile_verbose(self) -> bool:
        """Return whether compile and L3 execution timing logs are enabled."""
        return self._l3_trace

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the Qwen3-14B runtime runner for compiled kernels."""
        if not isinstance(compiled, _CompiledKernels):
            raise TypeError("Qwen314BPyptoExecutor requires Qwen3-14B compiled kernels.")
        return Qwen314BModelRunner(
            compiled=compiled,
        )

    def _compile_model(self, model: RuntimeModel) -> _CompiledKernels:
        """Compile Qwen3-14B PyPTO kernels and pack runtime artifacts."""
        timer = StageTimer(
            enabled=self._l3_trace,
            prefix="compile-breakdown",
            title="_compile_model stage timings",
        )

        def _mark(label: str) -> None:
            timer.mark(label)

        qwen3_prefill_fwd = _load_pypto_lib_qwen14b_module("prefill_fwd")
        # The fused all-layer decode lives in decode_layer.decode_fwd (the
        # standalone decode_fwd.py module was removed in pypto-lib). It is now
        # PAGED: it consumes block_table + slot_mapping and reads/writes the SAME
        # device-resident paged KV pool prefill writes (self._kv_caches), so no
        # contiguous bridge / MAX_SEQ env is needed.
        qwen3_decode_layer = _load_pypto_lib_qwen14b_module("decode_layer")
        qwen3_l3_dispatch.prefill_fwd = qwen3_prefill_fwd.prefill_fwd
        qwen3_l3_dispatch.decode_fwd = qwen3_decode_layer.decode_fwd

        _mark("imports")

        self._validate_supported_shape(model)
        kernel_batch = model.runtime.max_batch_size
        if int(qwen3_decode_layer.BATCH) != kernel_batch:
            raise ValueError(
                "decode_layer.decode_fwd is compiled for a fixed kernel BATCH of "
                f"{int(qwen3_decode_layer.BATCH)}, but runtime max_batch_size is "
                f"{kernel_batch}; they must match (decode statically computes and "
                "writes BATCH rows / BATCH logit rows)."
            )
        if int(model.config.num_hidden_layers) != int(qwen3_decode_layer.NUM_LAYERS):
            raise ValueError(
                "decode_layer.decode_fwd fuses a FIXED "
                f"NUM_LAYERS={int(qwen3_decode_layer.NUM_LAYERS)} loop (the layer count "
                "is a kernel constant, not derived from the weight tensors), but the "
                f"model has {model.config.num_hidden_layers} layers. The fused decode "
                "does not support --num-layers-override; run the full model."
            )
        self._validate_total_kv_pages(model, kernel_batch)

        padded_vocab = round_up(model.config.vocab_size, _VOCAB_PAD_MULTIPLE)
        if padded_vocab != int(qwen3_decode_layer.VOCAB):
            raise ValueError(
                f"decode_layer.decode_fwd hard-codes VOCAB={int(qwen3_decode_layer.VOCAB)} "
                f"(config.VOCAB) for its fused LM head, but the runtime padded vocab is "
                f"{padded_vocab} (round_up({model.config.vocab_size}, {_VOCAB_PAD_MULTIPLE})); "
                "they must match for the decode logits buffer / lm_head weight to line up."
            )
        page_size = model.runtime.page_size
        max_blocks_per_seq = (model.runtime.max_seq_len + page_size - 1) // page_size
        prefill = self._compile_prefill_fwd_callable(
            qwen3_l3_dispatch.qwen3_prefill_host,
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_layers=model.config.num_hidden_layers,
            vocab_size=padded_vocab,
            block_table_stride=max_blocks_per_seq,
            page_size=page_size,
        )
        _mark("compile_prefill")
        decode = self._compile_decode_fwd_callable(
            qwen3_l3_dispatch.qwen3_decode_host,
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            block_table_stride=max_blocks_per_seq,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_layers=model.config.num_hidden_layers,
            vocab_size=padded_vocab,
            page_size=page_size,
        )
        _mark("compile_decode")
        rope_cos_raw, rope_sin_raw = rope_tables(
            model.runtime.max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
        )
        rope_cos = self._shared_tensor(rope_cos_raw)
        rope_sin = self._shared_tensor(rope_sin_raw)

        _mark("rope_tables")

        lm_head_weight = model.lm_head
        if padded_vocab != lm_head_weight.shape[0]:
            pad_rows = padded_vocab - lm_head_weight.shape[0]
            padding = torch.zeros(
                (pad_rows, lm_head_weight.shape[1]),
                dtype=lm_head_weight.dtype,
                device=lm_head_weight.device,
            )
            lm_head_weight = torch.cat([lm_head_weight, padding], dim=0)
        padded_lm_head_weight = self._shared_tensor(lm_head_weight.to(torch.bfloat16).contiguous().cpu())
        _mark("pad_lm_head")
        layers = []
        for layer in model.layers:
            layers.append(self._kernel_layer_weights(layer))
            self._release_layer_weights(layer)
        final_norm_weight = self._shared_tensor(model.final_norm_weight.view(1, -1).float().cpu())
        _mark("kernel_layer_weights")

        decode_weights = {
            name: self._shared_tensor(tensor)
            for name, tensor in self._stack_decode_weights(layers).items()
        }
        _mark("stack_decode_weights")
        prefill_hidden_buffer = torch.empty(
            (kernel_batch * model.runtime.max_seq_len, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        prefill_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_chunk_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_chunk_offsets_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        prefill_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,),
            dtype=torch.int32,
        ).share_memory_()
        prefill_slot_mapping_buffer = torch.empty(
            (kernel_batch * model.runtime.max_seq_len,),
            dtype=torch.int32,
        ).share_memory_()
        prefill_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
        _mark("prefill_buffers")
        decode_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
        decode_hidden_buffer = torch.empty(
            (kernel_batch, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        decode_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        decode_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,),
            dtype=torch.int32,
        ).share_memory_()
        decode_slot_mapping_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        _mark("decode_logits_buffer")

        timer.report()

        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            final_norm_weight=final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
            decode_weights=decode_weights,
            prefill_hidden_buffer=prefill_hidden_buffer,
            prefill_seq_lens_buffer=prefill_seq_lens_buffer,
            prefill_chunk_lens_buffer=prefill_chunk_lens_buffer,
            prefill_chunk_offsets_buffer=prefill_chunk_offsets_buffer,
            prefill_block_table_buffer=prefill_block_table_buffer,
            prefill_slot_mapping_buffer=prefill_slot_mapping_buffer,
            prefill_logits_buffer=prefill_logits_buffer,
            decode_hidden_buffer=decode_hidden_buffer,
            decode_seq_lens_buffer=decode_seq_lens_buffer,
            decode_block_table_buffer=decode_block_table_buffer,
            decode_slot_mapping_buffer=decode_slot_mapping_buffer,
            decode_logits_buffer=decode_logits_buffer,
        )

    def _compile_prefill_fwd_callable(
        self,
        jit_fn: object,
        *,
        batch: int,
        max_seq: int,
        block_table_stride: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        vocab_size: int,
        page_size: int,
    ) -> _L3Callable:
        """Compile the prefill HOST wrapper into a distributed program."""
        kv_hidden = num_kv_heads * head_dim
        total_tokens = batch * max_seq
        runtime_cache_blocks = (max_seq + page_size - 1) // page_size
        cache_rows = batch * runtime_cache_blocks * num_layers * num_kv_heads * page_size
        dummy_args = [
            torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16),
            torch.empty((batch,), dtype=torch.int32),
            torch.empty((batch,), dtype=torch.int32),
            torch.empty((batch,), dtype=torch.int32),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),
            torch.empty((num_layers, head_dim), dtype=torch.float32),
            torch.empty((num_layers, head_dim), dtype=torch.float32),
            torch.empty((max_seq, head_dim), dtype=torch.float32),
            torch.empty((max_seq, head_dim), dtype=torch.float32),
            torch.empty((batch * block_table_stride,), dtype=torch.int32),
            torch.empty((total_tokens,), dtype=torch.int32),
            torch.empty((cache_rows, head_dim), dtype=torch.bfloat16),
            torch.empty((cache_rows, head_dim), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((1, hidden_size), dtype=torch.float32),
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((batch, vocab_size), dtype=torch.float32),
        ]
        return self._compile_jit_fwd_callable("prefill_fwd", jit_fn, dummy_args)

    def _compile_decode_fwd_callable(
        self,
        jit_fn: object,
        *,
        batch: int,
        max_seq: int,
        block_table_stride: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        vocab_size: int,
        page_size: int,
    ) -> _L3Callable:
        """Compile the fused all-layer PAGED decode HOST wrapper into a distributed program.

        Signature (22 args; PAGED KV via block_table + slot_mapping, same pool as
        prefill):
          hidden_states, input_rms_weight, wq, wk, wv, q_norm_weight,
          k_norm_weight, seq_lens, block_table, slot_mapping, rope_cos, rope_sin,
          k_cache, v_cache, wo, w_gate, w_up, w_down, post_rms_weight,
          final_norm_weight, lm_head_weight, out.

        k_cache/v_cache are the PAGED pool (rows = num_layers * batch *
        runtime_cache_blocks * num_kv_heads * page_size — identical to prefill);
        the kernel derives the per-layer stride + max_blocks_per_seq from the
        tensor dims. Projection weights are stacked ``[num_layers*HIDDEN, ...]``
        and norm gammas ``[num_layers, dim]`` — exactly what
        ``_stack_decode_weights`` produces.
        """
        kv_hidden = num_kv_heads * head_dim
        runtime_cache_blocks = (max_seq + page_size - 1) // page_size
        cache_rows = num_layers * batch * runtime_cache_blocks * num_kv_heads * page_size
        dummy_args = [
            torch.empty((batch, hidden_size), dtype=torch.bfloat16),                          # hidden_states
            torch.empty((num_layers, hidden_size), dtype=torch.float32),                      # input_rms_weight
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),        # wq
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),          # wk
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),          # wv
            torch.empty((num_layers, head_dim), dtype=torch.float32),                          # q_norm_weight
            torch.empty((num_layers, head_dim), dtype=torch.float32),                          # k_norm_weight
            torch.empty((batch,), dtype=torch.int32),                                          # seq_lens
            torch.empty((batch * block_table_stride,), dtype=torch.int32),                     # block_table
            torch.empty((batch,), dtype=torch.int32),                                          # slot_mapping
            torch.empty((max_seq, head_dim), dtype=torch.float32),                             # rope_cos
            torch.empty((max_seq, head_dim), dtype=torch.float32),                             # rope_sin
            torch.empty((cache_rows, head_dim), dtype=torch.bfloat16),                         # k_cache (paged pool)
            torch.empty((cache_rows, head_dim), dtype=torch.bfloat16),                         # v_cache (paged pool)
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),        # wo
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_gate
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_up
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),  # w_down
            torch.empty((num_layers, hidden_size), dtype=torch.float32),                       # post_rms_weight
            torch.empty((1, hidden_size), dtype=torch.float32),                                # final_norm_weight
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),                      # lm_head_weight
            torch.empty((batch, vocab_size), dtype=torch.float32),                             # out
        ]
        return self._compile_jit_fwd_callable("decode_fwd", jit_fn, dummy_args)

    def _compile_jit_fwd_callable(
        self,
        name: str,
        jit_fn: object,
        dummy_args: list[torch.Tensor],
    ) -> _L3Callable:
        """Compile a HOST wrapper into a PyPTO DistributedCompiledProgram."""
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415
        from pypto.runtime import RunConfig  # noqa: PLC0415

        config = self._run_config(codegen_only=True)
        distributed_config = DistributedConfig(
            device_ids=list(self._device_ids),
            num_sub_workers=0,
            block_dim=_QWEN14B_BLOCK_DIM,
            aicpu_thread_num=4,
        )
        run_config = RunConfig(
            platform=config.platform,
            device_id=config.device_id,
            backend_type=config.backend_type,
            strategy=config.strategy,
            dump_passes=config.dump_passes,
            save_kernels=config.save_kernels,
            save_kernels_dir=config.save_kernels_dir,
            codegen_only=True,
            pto_isa_commit=config.pto_isa_commit,
            diagnostic_phase=config.diagnostic_phase,
            disabled_diagnostics=config.disabled_diagnostics,
            compile_profiling=config.compile_profiling,
            distributed_config=distributed_config,
        )
        compiled = jit_fn.compile(*dummy_args, config=run_config)
        if not isinstance(compiled, DistributedCompiledProgram):
            raise TypeError(
                f"{name} did not compile to DistributedCompiledProgram; got {type(compiled).__name__}"
            )
        return _L3Callable(
            compiled=compiled,
            name=name,
            block_dim=_QWEN14B_BLOCK_DIM,
            aicpu_thread_num=4,
        )

    @staticmethod
    def _load_runtime_config(output_dir: Path) -> dict[str, Any]:
        """Load ``RUNTIME_CONFIG`` from a generated ``kernel_config.py``."""
        config_path = output_dir / "kernel_config.py"
        spec = importlib.util.spec_from_file_location(f"_qwen_l2_kernel_config_{abs(hash(output_dir))}", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load kernel_config.py from {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return dict(getattr(module, "RUNTIME_CONFIG", {}))

    @staticmethod
    def _stack_decode_weights(layers: list[_KernelLayerWeights]) -> dict[str, torch.Tensor]:
        """Stack per-layer weights into fused decode-kernel tensors."""
        # Stack from already-prepared per-layer kernel weights. Each
        # _KernelLayerWeights field is already in the kernel-ready shape/dtype
        # (transposed bf16 cpu for projections, [1, N] float cpu for norms),
        # so a plain cat along dim 0 is all that's left. Reading from the
        # original model.layers here would crash because _release_layer_weights
        # has already replaced those tensors with torch.empty(0).
        def cat(attr: str) -> torch.Tensor:
            return torch.cat([getattr(l, attr) for l in layers], dim=0)

        return {
            "decode_input_rms_weight": cat("input_rms_weight").contiguous(),
            "decode_wq":               cat("wq"),
            "decode_wk":               cat("wk"),
            "decode_wv":               cat("wv"),
            "decode_q_norm_weight":    cat("q_norm_weight").contiguous(),
            "decode_k_norm_weight":    cat("k_norm_weight").contiguous(),
            "decode_wo":               cat("wo"),
            "decode_post_rms_weight":  cat("post_rms_weight").contiguous(),
            "decode_w_gate":           cat("w_gate"),
            "decode_w_up":             cat("w_up"),
            "decode_w_down":           cat("w_down"),
        }

    @classmethod
    def _validate_total_kv_pages(cls, model: RuntimeModel, kernel_batch: int) -> None:
        """Validate that runtime KV page count matches compiled capacity."""
        if model.runtime.total_kv_pages is None:
            return
        expected_pages = kernel_batch * (model.runtime.max_seq_len + model.runtime.page_size - 1) // model.runtime.page_size
        if model.runtime.total_kv_pages != expected_pages:
            raise ValueError(
                "PyPTO Qwen3-14B kernels require total_kv_pages to match the runtime batch capacity: "
                f"{model.runtime.total_kv_pages} provided, {expected_pages} required."
            )

    @staticmethod
    def _kernel_weight(weight: torch.Tensor) -> torch.Tensor:
        """Convert a 2-D model weight into kernel-ready orientation and dtype."""
        return weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu().share_memory_()

    @classmethod
    def _kernel_layer_weights(cls, layer) -> _KernelLayerWeights:
        """Convert one Hugging Face layer into kernel-ready weight tensors."""
        return _KernelLayerWeights(
            input_rms_weight=cls._shared_tensor(layer.input_rms_weight.view(1, -1).float().cpu()),
            wq=cls._kernel_weight(layer.wq),
            wk=cls._kernel_weight(layer.wk),
            wv=cls._kernel_weight(layer.wv),
            q_norm_weight=cls._shared_tensor(layer.q_norm_weight.view(1, -1).float().cpu()),
            k_norm_weight=cls._shared_tensor(layer.k_norm_weight.view(1, -1).float().cpu()),
            wo=cls._kernel_weight(layer.wo),
            post_rms_weight=cls._shared_tensor(layer.post_rms_weight.view(1, -1).float().cpu()),
            w_gate=cls._kernel_weight(layer.w_gate),
            w_up=cls._kernel_weight(layer.w_up),
            w_down=cls._kernel_weight(layer.w_down),
        )

    @staticmethod
    def _shared_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor into shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

    @staticmethod
    def _release_layer_weights(layer) -> None:
        """Drop original layer tensors after kernel-ready copies are built."""
        empty = torch.empty(0)
        layer.input_rms_weight = empty
        layer.wq = empty
        layer.wk = empty
        layer.wv = empty
        layer.q_norm_weight = empty
        layer.k_norm_weight = empty
        layer.wo = empty
        layer.post_rms_weight = empty
        layer.w_gate = empty
        layer.w_up = empty
        layer.w_down = empty

    @staticmethod
    def _validate_supported_shape(model: RuntimeModel) -> None:
        """Ensure the loaded model matches the bundled Qwen3-14B kernels."""
        config = model.config
        expected = {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        actual = {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
        }
        if actual != expected:
            mismatch = ", ".join(f"{k}={actual[k]} (expected {v})" for k, v in expected.items() if actual[k] != v)
            raise ValueError(
                "Bundled kernels under model/ currently support Qwen3-14B layer shapes only: " + mismatch
            )
        if model.runtime.page_size != _QWEN14B_PAGE_SIZE:
            raise ValueError(
                "PyPTO Qwen3-14B kernels require runtime page_size "
                f"{_QWEN14B_PAGE_SIZE}, got {model.runtime.page_size}."
            )
