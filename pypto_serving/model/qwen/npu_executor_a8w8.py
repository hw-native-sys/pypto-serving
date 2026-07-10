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
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from pypto_serving.config.types import RuntimeModel
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor as CorePyptoExecutor
from pypto_serving.model.common.executor.utils import rope_tables, round_up
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.qwen.npu_executor import (
    _QWEN14B_BLOCK_DIM,
    _VOCAB_PAD_MULTIPLE,
    _find_pypto_lib_qwen14b_dir,
    Qwen314BPyptoExecutor,
)
from pypto_serving.model.qwen.npu_runner_a8w8 import (
    _CompiledKernels,
    _KernelLayerWeights,
    _L2Callable,
    Qwen314BA8W8ModelRunner,
)


_QWEN14B_A8W8_PREFILL_CHUNK_LAYERS = 10


class StageTimer:
    """Lightweight stage timer used by verbose A8W8 compilation logs."""

    def __init__(self, *, enabled: bool, prefix: str, title: str) -> None:
        self._enabled = enabled
        self._prefix = prefix
        self._title = title
        self._started_at = time.perf_counter() if enabled else 0.0
        self._stages: list[tuple[str, float]] = []

    def mark(self, label: str) -> None:
        if self._enabled:
            self._stages.append((label, time.perf_counter()))

    def report(self) -> None:
        if not self._enabled or not self._stages:
            return
        previous = self._started_at
        total_ms = (self._stages[-1][1] - self._started_at) * 1000.0
        print(f"[{self._prefix}] {self._title}:", flush=True)
        for label, timestamp in self._stages:
            print(f"[{self._prefix}]   {label:30s} : {(timestamp - previous) * 1000.0:9.1f} ms", flush=True)
            previous = timestamp
        print(f"[{self._prefix}]   {'TOTAL':30s} : {total_ms:9.1f} ms", flush=True)


def _patch_aicore_bitcast_helpers(work_dir: Path) -> None:
    """Mark ptoas-generated bitcast helpers as AICore-callable for A8W8 JIT builds."""
    needle = "static inline To ptoas_bitcast(From from) {"
    replacement = "static __aicore__ inline To ptoas_bitcast(From from) {"
    patched = 0
    for cpp in work_dir.rglob("*.cpp"):
        try:
            text = cpp.read_text()
        except UnicodeDecodeError:
            continue
        if needle not in text:
            continue
        cpp.write_text(text.replace(needle, replacement))
        patched += 1
    if patched:
        print(f"[RUN] patched {patched} ptoas_bitcast helper(s) for aicore compilation", flush=True)


def _ensure_pypto_task_id_alias() -> None:
    """Install the legacy TASK_ID alias required by older pypto-lib kernels."""
    import pypto.language as pl_mod  # noqa: PLC0415

    if not hasattr(pl_mod, "TASK_ID"):
        # Keep this process-wide: kernel functions resolve pl.TASK_ID later when
        # JIT compilation runs, not only while this module is imported.
        pl_mod.TASK_ID = pl_mod.INDEX


def _load_pypto_lib_qwen14b_module(module_name: str, kernel_dir: Path) -> object:
    """Load a Qwen3-14B kernel module from the pypto-lib submodule."""
    module_path = kernel_dir / f"qwen3_14b_{module_name}.py"
    if not module_path.is_file():
        module_path = kernel_dir / f"{module_name}.py"
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
    sys.path.insert(0, str(kernel_dir))
    try:
        _ensure_pypto_task_id_alias()
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(kernel_dir))
        except ValueError:
            pass
    return module


class Qwen314BA8W8PyptoExecutor(CorePyptoExecutor):
    """PyPTO executor that compiles and registers the Qwen3-14B kernels."""

    _shared_tensor = staticmethod(Qwen314BPyptoExecutor._shared_tensor)
    _validate_supported_shape = staticmethod(Qwen314BPyptoExecutor._validate_supported_shape)
    _validate_total_kv_pages = staticmethod(Qwen314BPyptoExecutor._validate_total_kv_pages)

    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform: str = "a2a3sim",
        device_ids: Sequence[int] = (0,),
        save_kernels_dir: str | None = None,
        pto_isa_commit: str | None = None,
        l3_trace: bool = False,
        pypto_root: str | None = None,
    ) -> None:
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=device_ids,
            save_kernels_dir=save_kernels_dir,
            pto_isa_commit=pto_isa_commit,
        )
        self._l3_trace = l3_trace
        self._l2_compile_root: Path | None = None
        self._pypto_root = pypto_root

    @property
    def profile_verbose(self) -> bool:
        """Return whether compile and L3 execution timing logs are enabled."""
        return self._l3_trace

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the Qwen3-14B runtime runner for compiled kernels."""
        if not isinstance(compiled, _CompiledKernels):
            raise TypeError("Qwen314BA8W8PyptoExecutor requires Qwen3-14B compiled kernels.")
        return Qwen314BA8W8ModelRunner(
            model_id=model_id,
            compiled=compiled,
            platform=self._platform,
            device_id=self._device_ids[0],
            save_kernels_dir=self._save_kernels_dir,
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

        quantization = self._model_quantization(model)
        if quantization != "a8w8":
            raise ValueError("Qwen314BA8W8PyptoExecutor only supports qwen3-a8w8 loaded models.")
        kernel_dir = _find_pypto_lib_qwen14b_dir(self._pypto_root)
        qwen3_prefill_fwd = _load_pypto_lib_qwen14b_module("prefill_fwd_a8w8", kernel_dir)
        # The fused all-layer decode lives in decode_layer.decode_fwd (the
        # standalone decode_fwd.py module was removed in pypto-lib). It is now
        # PAGED: it consumes block_table + slot_mapping and reads/writes the SAME
        # device-resident paged KV pool prefill writes (self._kv_caches), so no
        # contiguous bridge / MAX_SEQ env is needed.
        qwen3_decode_layer = _load_pypto_lib_qwen14b_module("decode_layer_a8w8", kernel_dir)
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
        self._validate_kernel_runtime_contract(model, qwen3_prefill_fwd, qwen3_decode_layer)
        self._validate_total_kv_pages(model, kernel_batch)

        padded_vocab = round_up(model.config.vocab_size, _VOCAB_PAD_MULTIPLE)
        kernel_vocab = getattr(qwen3_decode_layer, "VOCAB", padded_vocab)
        if padded_vocab != int(kernel_vocab):
            raise ValueError(
                f"decode_layer.decode_fwd hard-codes VOCAB={int(kernel_vocab)} "
                f"(config.VOCAB) for its fused LM head, but the runtime padded vocab is "
                f"{padded_vocab} (round_up({model.config.vocab_size}, {_VOCAB_PAD_MULTIPLE})); "
                "they must match for the decode logits buffer / lm_head weight to line up."
            )
        page_size = model.runtime.page_size
        max_blocks_per_seq = (model.runtime.max_seq_len + page_size - 1) // page_size
        prefill = self._compile_prefill_fwd_callable_a8w8(
            qwen3_prefill_fwd.prefill_hidden_a8w8,
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_layers=min(model.config.num_hidden_layers, _QWEN14B_A8W8_PREFILL_CHUNK_LAYERS),
            vocab_size=padded_vocab,
            block_table_stride=max_blocks_per_seq,
            page_size=page_size,
        )
        _mark("compile_prefill")
        decode = self._compile_decode_fwd_callable_a8w8(
            qwen3_decode_layer.decode_fwd,
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

        decode_weights = self._stack_decode_weights(layers)
        layers.clear()
        _mark("stack_decode_weights")
        decode_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
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
            decode_logits_buffer=decode_logits_buffer,
        )

    def _compile_prefill_fwd_callable_a8w8(
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
    ) -> _L2Callable:
        """Compile the A8W8 all-layer prefill kernel into an L2 callable."""
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
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.int8),
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.int8),
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.int8),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers, kv_hidden), dtype=torch.float32),
            torch.empty((num_layers, kv_hidden), dtype=torch.float32),
            torch.empty((num_layers, head_dim), dtype=torch.float32),
            torch.empty((num_layers, head_dim), dtype=torch.float32),
            torch.empty((max_seq, head_dim), dtype=torch.float32),
            torch.empty((max_seq, head_dim), dtype=torch.float32),
            torch.empty((batch * block_table_stride,), dtype=torch.int32),
            torch.empty((total_tokens,), dtype=torch.int32),
            torch.empty((cache_rows, head_dim), dtype=torch.int8),
            torch.empty((cache_rows, head_dim), dtype=torch.int8),
            torch.empty((cache_rows, 8), dtype=torch.float32),
            torch.empty((cache_rows, 8), dtype=torch.float32),
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.int8),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((1, hidden_size), dtype=torch.float32),
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((batch, vocab_size), dtype=torch.float32),
            torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16),
        ]
        return self._compile_jit_fwd_callable("prefill_hidden_a8w8", jit_fn, dummy_args)

    def _compile_decode_fwd_callable_a8w8(
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
    ) -> _L2Callable:
        """Compile the A8W8 fused all-layer paged decode kernel."""
        kv_hidden = num_kv_heads * head_dim
        runtime_cache_blocks = (max_seq + page_size - 1) // page_size
        cache_rows = num_layers * batch * runtime_cache_blocks * num_kv_heads * page_size
        dummy_args = [
            torch.empty((batch, hidden_size), dtype=torch.bfloat16),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.int8),
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.int8),
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.int8),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers, kv_hidden), dtype=torch.float32),
            torch.empty((num_layers, kv_hidden), dtype=torch.float32),
            torch.empty((num_layers, head_dim), dtype=torch.float32),
            torch.empty((num_layers, head_dim), dtype=torch.float32),
            torch.empty((batch,), dtype=torch.int32),
            torch.empty((batch * block_table_stride,), dtype=torch.int32),
            torch.empty((batch,), dtype=torch.int32),
            torch.empty((max_seq, head_dim), dtype=torch.float32),
            torch.empty((max_seq, head_dim), dtype=torch.float32),
            torch.empty((cache_rows, head_dim), dtype=torch.int8),
            torch.empty((cache_rows, head_dim), dtype=torch.int8),
            torch.empty((cache_rows, 8), dtype=torch.float32),
            torch.empty((cache_rows, 8), dtype=torch.float32),
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.int8),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((num_layers, hidden_size), dtype=torch.float32),
            torch.empty((1, hidden_size), dtype=torch.float32),
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),
            torch.empty((batch, vocab_size), dtype=torch.float32),
        ]
        return self._compile_jit_fwd_callable("decode_fwd_a8w8", jit_fn, dummy_args)

    def _compile_jit_fwd_callable(
        self,
        name: str,
        jit_fn: object,
        dummy_args: list[torch.Tensor],
    ) -> _L2Callable:
        """Compile a top-level ``@pl.jit`` kernel into an L2 callable."""
        from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415
        from pypto.runtime.runner import _patch_orchestration_headers  # noqa: PLC0415

        config = self._run_config(codegen_only=True)
        compiled = jit_fn.compile(*dummy_args, config=config)
        work_dir = Path(compiled.output_dir)
        _patch_orchestration_headers(work_dir)
        _patch_aicore_bitcast_helpers(work_dir)
        assembled = compile_and_assemble(
            work_dir,
            self._platform,
            pto_isa_commit=config.pto_isa_commit,
        )
        if len(assembled) == 2:
            chip_callable, runtime_name = assembled
            runtime_config = {}
        else:
            chip_callable, runtime_name, runtime_config = assembled
        runtime_config = runtime_config or {}
        param_infos, _, _ = compiled._get_metadata()
        return _L2Callable(
            chip_callable=chip_callable,
            name=name,
            runtime_name=runtime_name,
            block_dim=int(runtime_config.get("block_dim", _QWEN14B_BLOCK_DIM)),
            aicpu_thread_num=int(runtime_config.get("aicpu_thread_num", 4)),
            param_infos=tuple(param_infos),
        )

    def _l2_work_dir(self, name: str) -> Path:
        """Return a dedicated compile directory for one non-L3 program."""
        if self._save_kernels_dir is not None:
            root = Path(self._save_kernels_dir)
        else:
            if self._l2_compile_root is None:
                self._l2_compile_root = Path(tempfile.mkdtemp(prefix="qwen3_14b_l2_"))
            root = self._l2_compile_root
        work_dir = root / name
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    @staticmethod
    def _validate_kernel_runtime_contract(
        model: RuntimeModel,
        qwen3_prefill_fwd: object,
        qwen3_decode_layer: object,
    ) -> None:
        """Validate runtime dimensions baked into the A8W8 L2 kernels."""
        expected_page_size = int(qwen3_decode_layer.BLOCK_SIZE)
        if int(model.runtime.page_size) != expected_page_size:
            raise ValueError(
                f"qwen3-a8w8 requires page_size={expected_page_size} "
                f"to match decode BLOCK_SIZE, got {model.runtime.page_size}"
            )
        max_supported_seq = int(qwen3_prefill_fwd.MAX_SEQ)
        if int(model.runtime.max_seq_len) > max_supported_seq:
            raise ValueError(
                f"qwen3-a8w8 supports max_seq_len <= {max_supported_seq}, "
                f"got {model.runtime.max_seq_len}"
            )

    @classmethod
    def _stack_decode_weights(cls, layers: list[_KernelLayerWeights]) -> dict[str, torch.Tensor]:
        """Stack per-layer weights into fused decode-kernel tensors and release sources."""
        # Stack from already-prepared per-layer kernel weights. Each
        # _KernelLayerWeights field is already in the kernel-ready shape/dtype
        # (transposed bf16 cpu for projections, [1, N] float cpu for norms),
        # so a plain cat along dim 0 is all that's left. Reading from the
        # original model.layers here would crash because _release_layer_weights
        # has already replaced those tensors with torch.empty(0).
        def cat(attr: str) -> torch.Tensor:
            stacked = cls._shared_tensor(torch.cat([getattr(l, attr) for l in layers], dim=0).contiguous())
            for layer in layers:
                setattr(layer, attr, torch.empty(0))
            return stacked

        weights = {
            "decode_input_rms_weight": cat("input_rms_weight"),
            "decode_wq":               cat("wq"),
            "decode_wk":               cat("wk"),
            "decode_wv":               cat("wv"),
            "decode_q_norm_weight":    cat("q_norm_weight"),
            "decode_k_norm_weight":    cat("k_norm_weight"),
            "decode_wo":               cat("wo"),
            "decode_post_rms_weight":  cat("post_rms_weight"),
            "decode_w_gate":           cat("w_gate"),
            "decode_w_up":             cat("w_up"),
            "decode_w_down":           cat("w_down"),
        }
        if layers and layers[0].wq_scale is not None:
            weights.update(
                {
                    "decode_wq_scale": cat("wq_scale"),
                    "decode_wk_scale": cat("wk_scale"),
                    "decode_wv_scale": cat("wv_scale"),
                    "decode_wo_scale": cat("wo_scale"),
                }
            )
        return weights

    @classmethod
    def _kernel_layer_weights(cls, layer) -> _KernelLayerWeights:
        """Convert one Hugging Face layer into kernel-ready weight tensors."""
        return _KernelLayerWeights(
            input_rms_weight=cls._shared_tensor(layer.input_rms_weight.float().cpu()),
            wq=cls._shared_tensor(layer.wq.contiguous().cpu()),
            wk=cls._shared_tensor(layer.wk.contiguous().cpu()),
            wv=cls._shared_tensor(layer.wv.contiguous().cpu()),
            q_norm_weight=cls._shared_tensor(layer.q_norm_weight.float().cpu()),
            k_norm_weight=cls._shared_tensor(layer.k_norm_weight.float().cpu()),
            wo=cls._shared_tensor(layer.wo.contiguous().cpu()),
            post_rms_weight=cls._shared_tensor(layer.post_rms_weight.float().cpu()),
            w_gate=cls._shared_tensor(layer.w_gate.contiguous().cpu()),
            w_up=cls._shared_tensor(layer.w_up.contiguous().cpu()),
            w_down=cls._shared_tensor(layer.w_down.contiguous().cpu()),
            wq_scale=cls._shared_tensor(layer.wq_scale.float().cpu()),
            wk_scale=cls._shared_tensor(layer.wk_scale.float().cpu()),
            wv_scale=cls._shared_tensor(layer.wv_scale.float().cpu()),
            wo_scale=cls._shared_tensor(layer.wo_scale.float().cpu()),
        )

    @staticmethod
    def _release_layer_weights(layer) -> None:
        """Drop original layer tensors after kernel-ready copies are built."""
        Qwen314BPyptoExecutor._release_layer_weights(layer)
        if hasattr(layer, "wq_scale"):
            empty = torch.empty(0)
            layer.wq_scale = empty
            layer.wk_scale = empty
            layer.wv_scale = empty
            layer.wo_scale = empty

    @staticmethod
    def _model_quantization(model: RuntimeModel) -> str:
        """Return the quantization mode declared by loaded layer weights."""
        if not model.layers:
            return "bf16"
        quantization = getattr(model.layers[0], "quantization", None)
        if quantization == "a8w8":
            return str(quantization)
        return "bf16"
