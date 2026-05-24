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
from pathlib import Path
from typing import Any

import torch
from simpler.task_interface import ContinuousTensor

from examples.model.qwen3_14b.runner.npu_runner import (
    _CompiledKernels,
    _KernelLayerWeights,
    _L2Callable,
    Qwen314BModelRunner,
)
from python.core._profiling import StageTimer
from python.core.model_runner import ModelRunner
from python.core.pypto_executor import PyptoExecutor as CorePyptoExecutor
from python.core.types import (
    GenerateConfig,
    GenerateResult,
    ModelRecord,
    PrefillBatch,
    RequestState,
    RuntimeModel,
)
from python.core.utils import rope_tables, round_up
from python.profile import profile_span


_VOCAB_PAD_MULTIPLE = 512  # must be a multiple of lm_head.VOCAB_CHUNK (64)
_QWEN14B_PAGE_SIZE = 128
_QWEN14B_BLOCK_DIM = 24


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


def _patch_orch_make_tensor_arg(module: object) -> None:
    """Allow ``make_tensor_arg`` in a generated orchestration module to pass
    ``ContinuousTensor`` objects through unchanged.

    The generated ``host_orch.py`` always calls ``make_tensor_arg(t)`` for
    every entry in the ``tensors`` dict.  When we pre-upload static weights
    and replace them with ``ContinuousTensor(child_memory=True)`` objects,
    the default ``make_tensor_arg`` (which expects a ``torch.Tensor``) would
    crash.  Patching it here lets child_memory tensors pass through as-is
    so the runtime skips H2D/D2H for those buffers.
    """
    _orig = getattr(module, "make_tensor_arg", None)
    if _orig is None or getattr(_orig, "_child_memory_patched", False):
        return

    def _patched(tensor: object) -> object:
        if isinstance(tensor, ContinuousTensor):
            return tensor
        return _orig(tensor)  # type: ignore[misc]

    _patched._child_memory_patched = True  # type: ignore[attr-defined]
    module.make_tensor_arg = _patched  # type: ignore[attr-defined]


class _StackedLayerView:
    """Adapter exposing HF-format LayerWeights in kernel orientation.

    stack_layer_weights() expects each per-layer weight already in the
    orientation the kernel ingests (transposed BF16 contiguous CPU). This
    view computes that view lazily per attribute access so the stacker can
    iterate ``getattr(layer, attr)`` against the standard LayerWeights.
    """

    _KERNEL_2D_ATTRS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")

    def __init__(self, layer) -> None:
        self._layer = layer

    def __getattr__(self, name: str) -> torch.Tensor:
        weight = getattr(self._layer, name)
        if name in self._KERNEL_2D_ATTRS:
            return weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu()
        # Norm gammas (input_rms_weight, q/k/post norm). The stacker
        # flattens to [dim] then re-stacks; cast to FP32 to match the
        # kernel signature ([num_layers, dim], FP32).
        return weight.view(-1).float().contiguous().cpu()


class Qwen314BPyptoExecutor(CorePyptoExecutor):
    """PyPTO executor that compiles and registers the Qwen3-14B kernels."""

    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform: str = "a2a3sim",
        device_id: int = 0,
        save_kernels_dir: str | None = None,
        l3_mode: bool = False,
        l3_trace: bool = False,
    ) -> None:
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_id=device_id,
            save_kernels_dir=save_kernels_dir,
        )
        self._l3_mode = l3_mode
        self._l3_trace = l3_trace
        self._l2_compile_root: Path | None = None

    @property
    def profile_verbose(self) -> bool:
        """Return whether compile and L3 execution timing logs are enabled."""
        return self._l3_trace

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the Qwen3-14B runtime runner for compiled kernels."""
        if not isinstance(compiled, _CompiledKernels):
            raise TypeError("Qwen314BPyptoExecutor requires Qwen3-14B compiled kernels.")
        return Qwen314BModelRunner(
            model_id=model_id,
            compiled=compiled,
            platform=self._platform,
            device_id=self._device_id,
            save_kernels_dir=self._save_kernels_dir,
            l3_trace=self._l3_trace,
            kv_cache_manager=self._kv_cache_manager,
        )

    def validate_generate_batch(
        self,
        record: ModelRecord,
        batch_size: int,
        config: GenerateConfig,
    ) -> None:
        """Reject generation settings unsupported by the Qwen3-14B L3 path."""
        if not self._l3_mode:
            return
        if batch_size != 1:
            raise NotImplementedError(
                "L3 generate fast path currently supports batch_size=1; "
                f"got {batch_size} prompts."
            )
        if any(config.stop):
            raise NotImplementedError(
                "L3 generate fast path does not support generate_config.stop; "
                "the device-side decode loop cannot break on host-side string matches."
            )

    def prompt_allocation_length(
        self,
        record: ModelRecord,
        prompt_len: int,
        config: GenerateConfig,
    ) -> int:
        """Return prompt KV length, including generated-token capacity for L3."""
        if not self._l3_mode:
            return prompt_len
        max_seq = record.runtime.max_seq_len
        if config.max_new_tokens < 1:
            raise ValueError(
                f"L3 mode requires max_new_tokens >= 1, got {config.max_new_tokens}."
            )
        if prompt_len > max_seq:
            raise ValueError(
                f"L3 mode: prompt length ({prompt_len}) exceeds max_seq_len ({max_seq}). "
                f"Either shorten the prompt or increase --max-seq-len to at least {prompt_len}."
            )
        alloc_len = prompt_len + config.max_new_tokens
        if alloc_len > max_seq:
            allowed_new_tokens = max_seq - prompt_len
            raise ValueError(
                f"L3 mode: prompt length ({prompt_len}) + max_new_tokens "
                f"({config.max_new_tokens}) = {alloc_len} exceeds max_seq_len ({max_seq}). "
                f"Either reduce --max-new-tokens to at most {allowed_new_tokens}, "
                f"or increase --max-seq-len to at least {alloc_len}."
            )
        return alloc_len

    def try_generate_batch(
        self,
        record: ModelRecord,
        requests: list[RequestState],
        prefill_batch: PrefillBatch,
        config: GenerateConfig,
    ) -> list[GenerateResult] | None:
        """Run the single-dispatch L3 generation path when L3 mode is enabled."""
        if not self._l3_mode:
            return None
        generated_ids, _ = self.run_generate_l3(
            record.runtime_model,
            prefill_batch,
            max_new_tokens=config.max_new_tokens,
            eos_token_id=record.config.eos_token_id,
        )
        request = requests[0]
        request.generated_token_ids = list(generated_ids)
        request.output_text = record.tokenizer.decode(generated_ids)
        if (
            generated_ids
            and record.config.eos_token_id is not None
            and generated_ids[-1] == record.config.eos_token_id
        ):
            finish_reason = "eos"
        else:
            finish_reason = "length"
        return [
            GenerateResult(
                text=request.output_text,
                token_ids=list(request.generated_token_ids),
                finish_reason=finish_reason,
            )
        ]

    def run_generate_l3(
        self,
        model: RuntimeModel,
        prefill_batch: PrefillBatch,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> tuple[list[int], torch.Tensor]:
        """Delegate L3 generation to the registered Qwen3-14B runner."""
        runner = self._runners[model.config.model_id]
        if not isinstance(runner, Qwen314BModelRunner):
            raise TypeError("Qwen314BPyptoExecutor requires a Qwen314BModelRunner.")
        with profile_span(
            "executor.run_generate_l3",
            cat="executor",
            args={
                "model_id": model.config.model_id,
                "batch_size": len(prefill_batch.request_ids),
                "max_new_tokens": max_new_tokens,
            },
        ):
            return runner.run_generate_l3(
                model,
                prefill_batch,
                max_new_tokens,
                eos_token_id,
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

        qwen3_l3_generate = _load_pypto_lib_qwen14b_module("l3_generate")
        qwen3_prefill_fwd = _load_pypto_lib_qwen14b_module("prefill_fwd")
        # The fused all-layer decode lives in decode_layer.decode_fwd (the
        # standalone decode_fwd.py module was removed in pypto-lib). It is now
        # PAGED: it consumes block_table + slot_mapping and reads/writes the SAME
        # device-resident paged KV pool prefill writes (self._kv_caches), so no
        # contiguous bridge / MAX_SEQ env is needed.
        qwen3_decode_layer = _load_pypto_lib_qwen14b_module("decode_layer")

        build_qwen3_14b_l3_generate_program = qwen3_l3_generate.build_qwen3_14b_l3_generate_program
        stack_layer_weights_full = qwen3_l3_generate.stack_layer_weights_full
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
            qwen3_prefill_fwd.prefill_fwd,
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
        final_rms = None
        lm_head = None

        # KV quantization kernels (compiled only when kv_quant enabled).
        # TQ fused kernels for prefill and decode.
        kv_quant_config = getattr(model.runtime, "kv_quant_config", None)
        prefill_tq: _L2Callable | None = None
        decode_tq: _L2Callable | None = None
        rot_matrices: torch.Tensor | None = None
        tq_codebook: torch.Tensor | None = None
        # QJL (Algorithm 2): per-layer orthogonal projection + K sign/norm caches.
        qjl_matrices: torch.Tensor | None = None
        quant_k_signs_cache: torch.Tensor | None = None
        qjl_norms_cache: torch.Tensor | None = None
        if kv_quant_config is not None and kv_quant_config.enabled:
            # Prefill TQ kernel: @pl.jit multi-layer fused, writes directly to
            # quantized KV cache (UINT8 + BF16 norms). Compiled via the same
            # _compile_jit_fwd_callable path as regular prefill_fwd.
            qwen3_prefill_tq = _load_pypto_lib_qwen14b_module("prefill_tq")
            prefill_tq = self._compile_prefill_tq_callable(
                qwen3_prefill_tq.prefill_fwd_tq,
                batch=kernel_batch,
                max_seq=model.runtime.max_seq_len,
                block_table_stride=prefill_block_table_stride,
                hidden_size=model.config.hidden_size,
                intermediate_size=model.config.intermediate_size,
                num_heads=model.config.num_attention_heads,
                num_kv_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
                num_layers=model.config.num_hidden_layers,
                vocab_size=padded_vocab,
            )
            _mark("compile_prefill_tq")

            # Decode TQ kernel: @pl.jit multi-layer fused with hybrid
            # resident (BF16) + compressed (UINT8) block attention.
            qwen3_decode_tq = _load_pypto_lib_qwen14b_module("decode_tq")
            decode_tq = self._compile_decode_tq_callable(
                qwen3_decode_tq.decode_fwd_tq,
                batch=kernel_batch,
                max_seq=model.runtime.max_seq_len,
                hidden_size=model.config.hidden_size,
                intermediate_size=model.config.intermediate_size,
                num_heads=model.config.num_attention_heads,
                num_kv_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
                num_layers=model.config.num_hidden_layers,
                vocab_size=padded_vocab,
            )
            _mark("compile_decode_tq")

            # Generate per-layer rotation matrices matching the compressor seeds.
            from python.core.turboquant.compressor import generate_rotation_matrix  # noqa: PLC0415
            num_layers = model.config.num_hidden_layers
            head_dim = model.config.head_dim
            rot_mats = torch.stack([
                generate_rotation_matrix(head_dim, seed=42 + l * 1000)
                for l in range(num_layers)
            ]).bfloat16().contiguous().cpu()
            rot_matrices = self._shared_tensor(rot_mats.reshape(num_layers * head_dim, head_dim))

            # TQ codebook (single row of Lloyd-Max centroids) — used by decode only.
            n_levels = int(qwen3_prefill_tq.N_LEVELS)
            from turboquant_kv import _lm_centroids  # noqa: PLC0415
            _cb_row = _lm_centroids.float().unsqueeze(0)  # [1, N_LEVELS]
            tq_codebook = _cb_row.clone().contiguous().share_memory_()  # [1, N_LEVELS]

            # QJL (Algorithm 2): per-layer orthogonal projection matrices.
            # Independent from rot_matrices (different seed).
            qjl_mats = torch.stack([
                generate_rotation_matrix(head_dim, seed=123 + l * 1000)
                for l in range(num_layers)
            ]).bfloat16().contiguous().cpu()
            qjl_matrices = self._shared_tensor(qjl_mats.reshape(num_layers * head_dim, head_dim))

            # QJL caches: K sign bits and residual norms.
            # Same row layout as quant_k_cache: [num_layers * num_pages * num_kv_heads * page_size, ...]
            runtime_cache_blocks = (model.runtime.max_seq_len + _QWEN14B_PAGE_SIZE - 1) // _QWEN14B_PAGE_SIZE
            cmp_cache_rows = num_layers * kernel_batch * runtime_cache_blocks * model.config.num_key_value_heads * _QWEN14B_PAGE_SIZE
            quant_k_signs_cache = torch.zeros(
                cmp_cache_rows, head_dim, dtype=torch.uint8,
            ).share_memory_()
            qjl_norms_cache = torch.zeros(
                cmp_cache_rows, 1, dtype=torch.float32,
            ).share_memory_()
        _mark("compile_kv_quant")

        rope_cos_raw, rope_sin_raw = rope_tables(
            model.runtime.max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
        )
        rope_cos = self._shared_tensor(rope_cos_raw)
        rope_sin = self._shared_tensor(rope_sin_raw)

        _mark("rope_tables")

        # L3 artefacts (built only when l3_mode=True).
        l3_generate: object | None = None
        stacked_weights: dict[str, torch.Tensor] | None = None
        if self._l3_mode:
            l3_generate_program = build_qwen3_14b_l3_generate_program(
                num_layers=model.config.num_hidden_layers,
                batch=kernel_batch,
                max_seq=model.runtime.max_seq_len,
                hidden_size=model.config.hidden_size,
                intermediate_size=model.config.intermediate_size,
                num_heads=model.config.num_attention_heads,
                num_kv_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
                max_new_tokens=model.runtime.max_new_tokens,
                padded_vocab=padded_vocab,
                page_size=model.runtime.page_size,
            )
            # pypto.runtime.run hard-codes DistributedConfig(block_dim=1), so we call
            # ir.compile directly to override it.
            from pypto import ir  # noqa: PLC0415
            from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

            _rc = self._run_config(codegen_only=True)
            l3_generate = ir.compile(
                l3_generate_program,
                output_dir=_rc.save_kernels_dir,
                strategy=_rc.strategy,
                backend_type=_rc.backend_type,
                dump_passes=_rc.dump_passes,
                diagnostic_phase=_rc.diagnostic_phase,
                disabled_diagnostics=_rc.disabled_diagnostics,
                platform=_rc.platform,
                profiling=_rc.compile_profiling,
                distributed_config=DistributedConfig(
                    device_ids=[self._device_id],
                    block_dim=_QWEN14B_BLOCK_DIM,
                    num_sub_workers=0,
                    aicpu_thread_num=4,
                ),
            )
            _mark("compile_l3_generate")
            # stack_layer_weights_full expects each weight already in
            # kernel orientation ([in_dim, out_dim] for 2D matmul weights).
            # Adapt via a per-layer view that mirrors _kernel_weight().
            stacked_layers = [_StackedLayerView(layer) for layer in model.layers]
            stacked_weights = stack_layer_weights_full(
                stacked_layers,
                hidden=model.config.hidden_size,
                kv_hidden=model.config.num_key_value_heads * model.config.head_dim,
                inter=model.config.intermediate_size,
                head_dim=model.config.head_dim,
            )
            _mark("stack_layer_weights")

        # L3-wrapped generate: pre-extract l3_generate setup artifacts
        # (expensive compile_and_assemble + module loading done once,
        # reused per generate call).
        l3_generate_chip_callables: dict[str, object] | None = None
        l3_generate_entry_fn: object | None = None
        l3_generate_sub_worker_fns: dict[str, object] | None = None
        l3_generate_dc: object | None = None
        l3_generate_platform: str | None = None
        l3_generate_runtime_name: str | None = None
        l3_generate_param_infos: object | None = None
        if self._l3_mode:
            from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415
            from pypto.runtime.distributed_runner import _load_generated_module  # noqa: PLC0415
            from pypto.pypto_core.ir import FunctionType  # noqa: PLC0415

            # Pre-extract l3_generate setup artifacts.
            lg_dc = l3_generate._distributed_config
            lg_output_dir = l3_generate.output_dir
            lg_chip_callables: dict[str, object] = {}
            lg_runtime_name = "tensormap_and_ringbuffer"
            lg_next_levels_dir = lg_output_dir / "next_levels"
            for func in l3_generate._program.functions.values():
                if func.func_type in (FunctionType.Orchestration, FunctionType.Opaque):
                    chip_dir = lg_next_levels_dir / func.name
                    if chip_dir.exists():
                        cc, lg_runtime_name, _ = compile_and_assemble(
                            chip_dir, l3_generate.platform,
                        )
                        lg_chip_callables[func.name] = cc
            lg_orch_path = lg_output_dir / "orchestration" / "host_orch.py"
            lg_orch_module = _load_generated_module(lg_orch_path)
            # Patch make_tensor_arg so pre-uploaded ContinuousTensor objects
            # (child_memory=True) are passed through unchanged instead of
            # triggering a crash when the generated code calls make_tensor_arg
            # on a non-torch.Tensor value.
            _patch_orch_make_tensor_arg(lg_orch_module)
            lg_entry_fn = None
            for attr_name in ("entry", "host_orch"):
                lg_entry_fn = getattr(lg_orch_module, attr_name, None)
                if lg_entry_fn is not None:
                    break
            if lg_entry_fn is None:
                for name in dir(lg_orch_module):
                    obj = getattr(lg_orch_module, name)
                    if callable(obj) and not name.startswith("_"):
                        lg_entry_fn = obj
                        break
            lg_sub_worker_fns: dict[str, object] = {}
            lg_sub_workers_dir = lg_output_dir / "sub_workers"
            if lg_sub_workers_dir.exists():
                for py_file in sorted(lg_sub_workers_dir.glob("*.py")):
                    mod = _load_generated_module(py_file)
                    fn_name = py_file.stem
                    fn = getattr(mod, fn_name, None)
                    if fn is not None:
                        lg_sub_worker_fns[fn_name] = fn
            lg_param_infos, _, _ = l3_generate._get_metadata()

            l3_generate_chip_callables = lg_chip_callables
            l3_generate_entry_fn = lg_entry_fn
            l3_generate_sub_worker_fns = lg_sub_worker_fns
            l3_generate_dc = lg_dc
            l3_generate_platform = l3_generate.platform
            l3_generate_runtime_name = lg_runtime_name
            l3_generate_param_infos = lg_param_infos
            _mark("l3_extract_artifacts")

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
        decode_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
        _mark("decode_logits_buffer")

        timer.report()

        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            final_rms=final_rms,
            lm_head=lm_head,
            final_norm_weight=final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
            layers=layers,
            decode_weights=decode_weights,
            decode_logits_buffer=decode_logits_buffer,
            stacked_weights=stacked_weights,
            l3_generate_chip_callables=l3_generate_chip_callables,
            l3_generate_entry_fn=l3_generate_entry_fn,
            l3_generate_sub_worker_fns=l3_generate_sub_worker_fns,
            l3_generate_dc=l3_generate_dc,
            l3_generate_platform=l3_generate_platform,
            l3_generate_runtime_name=l3_generate_runtime_name,
            l3_generate_param_infos=l3_generate_param_infos,
            prefill_tq=prefill_tq,
            decode_tq=decode_tq,
            rot_matrices=rot_matrices,
            tq_codebook=tq_codebook,
            qjl_matrices=qjl_matrices,
            quant_k_signs_cache=quant_k_signs_cache,
            qjl_norms_cache=qjl_norms_cache,
        )

    def _compile_l2_callable(self, name: str, program: object) -> _L2Callable:
        """Compile one non-L3 program and assemble it into a Simpler callable."""
        from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
        from pypto.runtime import compile_program  # noqa: PLC0415
        from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

        config = self._run_config(codegen_only=True)
        work_dir = self._l2_work_dir(name)
        compile_program(
            program,
            work_dir,
            strategy=config.strategy,
            backend_type=config.backend_type,
            dump_passes=config.dump_passes,
            diagnostic_phase=config.diagnostic_phase,
            disabled_diagnostics=config.disabled_diagnostics,
            profiling=config.compile_profiling,
        )
        chip_callable, runtime_name, runtime_config = compile_and_assemble(
            work_dir,
            self._platform,
            pto_isa_commit=config.pto_isa_commit,
        )
        runtime_config = runtime_config or {}
        compiled_view = CompiledProgram(
            program,
            str(work_dir),
            backend_type=config.backend_type,
            platform=self._platform,
        )
        param_infos, _, _ = compiled_view._get_metadata()
        return _L2Callable(
            chip_callable=chip_callable,
            name=name,
            runtime_name=runtime_name,
            block_dim=int(runtime_config.get("block_dim", _QWEN14B_BLOCK_DIM)),
            aicpu_thread_num=int(runtime_config.get("aicpu_thread_num", 4)),
            param_infos=tuple(param_infos),
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
    ) -> _L2Callable:
        """Compile the top-level ``@pl.jit`` prefill_fwd into an L2 callable."""
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
    ) -> _L2Callable:
        """Compile the fused all-layer PAGED ``decode_layer.decode_fwd`` into an L2 callable.

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

    def _compile_decode_tq_callable(
        self,
        jit_fn: object,
        *,
        batch: int,
        max_seq: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        vocab_size: int,
    ) -> _L2Callable:
        """Compile the top-level ``@pl.jit`` decode_fwd_tq into an L2 callable."""
        kv_hidden = num_kv_heads * head_dim
        runtime_cache_blocks = (max_seq + _QWEN14B_PAGE_SIZE - 1) // _QWEN14B_PAGE_SIZE
        cache_rows = batch * runtime_cache_blocks * num_layers * num_kv_heads * _QWEN14B_PAGE_SIZE
        block_table_stride = int(max_seq + _QWEN14B_PAGE_SIZE - 1) // _QWEN14B_PAGE_SIZE
        # TQ constants (must match qwen3_14b_decode_tq.py).
        cmp_chunk = 32
        n_levels = 16
        dummy_args = [
            torch.empty((batch, hidden_size), dtype=torch.bfloat16),               # hidden_states
            torch.empty((num_layers, hidden_size), dtype=torch.float32),            # input_rms_weight
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),     # wq
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),      # wk
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),      # wv
            torch.empty((num_layers, head_dim), dtype=torch.float32),               # q_norm_weight
            torch.empty((num_layers, head_dim), dtype=torch.float32),               # k_norm_weight
            torch.empty((batch,), dtype=torch.int32),                               # seq_lens
            torch.empty((batch * block_table_stride,), dtype=torch.int32),          # block_table
            torch.empty((batch,), dtype=torch.int32),                               # slot_mapping
            torch.empty((max_seq, head_dim), dtype=torch.float32),                  # rope_cos
            torch.empty((max_seq, head_dim), dtype=torch.float32),                  # rope_sin
            # TQ quantized KV cache.
            torch.empty((cache_rows, head_dim), dtype=torch.uint8),                 # quant_k_cache
            torch.empty((cache_rows, head_dim), dtype=torch.uint8),                 # quant_v_cache
            torch.empty((cache_rows, 1), dtype=torch.float32),                      # quant_k_scales
            torch.empty((cache_rows, 1), dtype=torch.float32),                      # quant_v_scales
            # Compressed block tracking.
            torch.empty((batch * block_table_stride,), dtype=torch.int32),          # cmp_block_table
            torch.empty((batch,), dtype=torch.int32),                               # cmp_num_blocks
            # Per-layer rotation matrices.
            torch.empty((num_layers * head_dim, head_dim), dtype=torch.bfloat16),   # rot_matrices
            # TQ codebook.
            torch.empty((1, n_levels), dtype=torch.float32),                # tq_codebook
            # QJL (Algorithm 2): projection matrices + K sign/norm caches.
            torch.empty((num_layers * head_dim, head_dim), dtype=torch.bfloat16),   # qjl_matrices
            torch.empty((cache_rows, head_dim), dtype=torch.uint8),                 # quant_k_signs_cache
            torch.empty((cache_rows, 1), dtype=torch.float32),                      # qjl_norms_cache
            # Standard layer weights.
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),     # wo
            torch.empty((num_layers, hidden_size), dtype=torch.float32),            # post_rms_weight
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_gate
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_up
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),  # w_down
            # Final norm + LM head.
            torch.empty((1, hidden_size), dtype=torch.float32),                     # final_norm_weight
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),           # lm_head_weight
            # Output (logits).
            torch.empty((batch, vocab_size), dtype=torch.float32),                  # out (logits)
        ]
        return self._compile_jit_fwd_callable("decode_tq", jit_fn, dummy_args)

    def _compile_prefill_tq_callable(
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
    ) -> _L2Callable:
        """Compile the top-level ``@pl.jit`` prefill_fwd_tq into an L2 callable."""
        kv_hidden = num_kv_heads * head_dim
        total_tokens = batch * max_seq
        runtime_cache_blocks = (max_seq + _QWEN14B_PAGE_SIZE - 1) // _QWEN14B_PAGE_SIZE
        cache_rows = batch * runtime_cache_blocks * num_layers * num_kv_heads * _QWEN14B_PAGE_SIZE
        # TQ constants (must match qwen3_14b_prefill_tq.py).
        n_levels = 16    # int4 -> 16 levels
        dummy_args = [
            # Standard prefill inputs (same as regular prefill_fwd).
            torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16),       # hidden_states
            torch.empty((batch,), dtype=torch.int32),                              # seq_lens
            torch.empty((num_layers, hidden_size), dtype=torch.float32),           # input_rms_weight
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),    # wq
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),     # wk
            torch.empty((num_layers * hidden_size, kv_hidden), dtype=torch.bfloat16),     # wv
            torch.empty((num_layers, head_dim), dtype=torch.float32),              # q_norm_weight
            torch.empty((num_layers, head_dim), dtype=torch.float32),              # k_norm_weight
            torch.empty((max_seq, head_dim), dtype=torch.float32),                 # rope_cos
            torch.empty((max_seq, head_dim), dtype=torch.float32),                 # rope_sin
            torch.empty((batch * block_table_stride,), dtype=torch.int32),         # block_table
            torch.empty((total_tokens,), dtype=torch.int32),                       # slot_mapping
            # TQ quantized KV cache (replaces k_cache/v_cache).
            torch.empty((cache_rows, head_dim), dtype=torch.uint8),                # quant_k_cache
            torch.empty((cache_rows, head_dim), dtype=torch.uint8),                # quant_v_cache
            torch.empty((cache_rows, 1), dtype=torch.float32),                     # quant_k_scales
            torch.empty((cache_rows, 1), dtype=torch.float32),                     # quant_v_scales
            # Per-layer rotation matrices [num_layers * head_dim, head_dim].
            torch.empty((num_layers * head_dim, head_dim), dtype=torch.bfloat16),  # rot_matrices
            # QJL (Algorithm 2): per-layer projection + K sign/norm caches.
            torch.empty((num_layers * head_dim, head_dim), dtype=torch.bfloat16),  # qjl_matrices
            torch.empty((cache_rows, head_dim), dtype=torch.uint8),                # quant_k_signs_cache
            torch.empty((cache_rows, 1), dtype=torch.float32),                     # qjl_norms_cache
            # Standard layer weights + final norm + lm_head.
            torch.empty((num_layers * hidden_size, hidden_size), dtype=torch.bfloat16),    # wo
            torch.empty((num_layers, hidden_size), dtype=torch.float32),           # post_rms_weight
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_gate
            torch.empty((num_layers * hidden_size, intermediate_size), dtype=torch.bfloat16),  # w_up
            torch.empty((num_layers * intermediate_size, hidden_size), dtype=torch.bfloat16),  # w_down
            torch.empty((1, hidden_size), dtype=torch.float32),                    # final_norm_weight
            torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16),          # lm_head_weight
            torch.empty((batch, vocab_size), dtype=torch.float32),                 # out (logits)
        ]
        return self._compile_jit_fwd_callable("prefill_tq", jit_fn, dummy_args)

    def _compile_jit_fwd_callable(
        self,
        name: str,
        jit_fn: object,
        dummy_args: list[torch.Tensor],
    ) -> _L2Callable:
        """Compile a top-level ``@pl.jit`` kernel into an L2 callable."""
        import pypto.language as pl_mod  # noqa: PLC0415
        from pypto.jit.cache import make_cache_key  # noqa: PLC0415
        from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415
        from pypto.runtime.runner import _patch_orchestration_headers  # noqa: PLC0415

        param_names, _arguments, tensor_meta, scalar_values, scalar_dtypes, dynamic_dims = jit_fn._bind_args(
            tuple(dummy_args), {}
        )
        key = make_cache_key(
            source_hash=jit_fn._get_source_hash(),
            param_names=param_names,
            tensor_shapes={name: meta.shape for name, meta in tensor_meta.items()},
            tensor_dtypes={name: meta.dtype for name, meta in tensor_meta.items()},
            dynamic_dims=dynamic_dims[id(jit_fn._func)],
            scalar_values=scalar_values,
            platform=self._platform,
        )
        if key not in jit_fn._cache:
            jit_fn._cache[key] = jit_fn._compile(
                tensor_meta,
                scalar_values,
                scalar_dtypes,
                dynamic_dims,
                pl_mod,
                platform=self._platform,
            )
        compiled = jit_fn._cache[key]
        work_dir = Path(compiled.output_dir)
        _patch_orchestration_headers(work_dir)
        chip_callable, runtime_name, runtime_config = compile_and_assemble(
            work_dir,
            self._platform,
            pto_isa_commit=self._run_config(codegen_only=True).pto_isa_commit,
        )
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

    def _compile_l2_callable_from_jit(self, name: str, jit_fn, dummy_args: tuple) -> _L2Callable:
        """Compile a ``@pl.jit`` function into a Simpler callable.

        Uses the JIT compilation path (``fn._compile``) which supports
        ``@pl.jit.inline`` helpers and scalar operations, unlike
        ``compile_program()`` which requires ``@pl.program`` classes.
        """
        from pypto.jit.cache import make_cache_key  # noqa: PLC0415
        from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415
        import pypto.language as pl_mod  # noqa: PLC0415

        config = self._run_config(codegen_only=True)
        work_dir = self._l2_work_dir(name)

        # Bind dummy args to trigger JIT compilation.
        param_names, _arguments, tensor_meta, scalar_values, scalar_dtypes, dynamic_dims = (
            jit_fn._bind_args(dummy_args, {})
        )
        key = make_cache_key(
            source_hash=jit_fn._get_source_hash(),
            param_names=param_names,
            tensor_shapes={n: m.shape for n, m in tensor_meta.items()},
            tensor_dtypes={n: m.dtype for n, m in tensor_meta.items()},
            dynamic_dims=dynamic_dims,
            scalar_values=scalar_values,
            platform=self._platform,
        )
        if key not in jit_fn._cache:
            jit_fn._cache[key] = jit_fn._compile(
                tensor_meta, scalar_values, scalar_dtypes, dynamic_dims, pl_mod,
                platform=self._platform,
            )
        jit_output_dir = Path(jit_fn._cache[key].output_dir)

        # Copy JIT output to our work_dir so compile_and_assemble finds it.
        import shutil  # noqa: PLC0415
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(jit_output_dir, work_dir)

        chip_callable, runtime_name, runtime_config = compile_and_assemble(
            work_dir,
            self._platform,
            pto_isa_commit=config.pto_isa_commit,
        )
        runtime_config = runtime_config or {}

        # Build param_infos from bind_args metadata (JIT functions
        # don't have a .functions dict that CompiledProgram expects).
        class _ParamInfo:
            __slots__ = ("name", "shape", "dtype")
            def __init__(self, name, shape, dtype=None):
                self.name = name
                self.shape = shape
                self.dtype = dtype

        param_infos = []
        for pname in param_names:
            if pname in tensor_meta:
                meta = tensor_meta[pname]
                param_infos.append(_ParamInfo(pname, tuple(meta.shape), meta.dtype))
            else:
                param_infos.append(_ParamInfo(pname, None))
        return _L2Callable(
            chip_callable=chip_callable,
            runtime_name=runtime_name,
            block_dim=int(runtime_config.get("block_dim", 24)),
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
