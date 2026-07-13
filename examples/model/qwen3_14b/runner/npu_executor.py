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
        device_ids: Sequence[int] = (0,),
        save_kernels_dir: str | None = None,
        l3_trace: bool = False,
    ) -> None:
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=device_ids,
            save_kernels_dir=save_kernels_dir,
        )
        self._l3_trace = l3_trace

    @property
    def profile_verbose(self) -> bool:
        """Return whether compile and L3 execution timing logs are enabled."""
        return self._l3_trace

    @property
    def supports_device_sampling(self) -> bool:
        """Qwen3 NPU runner can return greedy sampled token ids."""
        return True

    @property
    def supports_device_embedding(self) -> bool:
        """Qwen3 NPU decode embeds greedy token ids inside the device kernel."""
        return True

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the Qwen3-14B runtime runner for compiled kernels."""
        if not isinstance(compiled, _CompiledKernels):
            raise TypeError("Qwen314BPyptoExecutor requires Qwen3-14B compiled kernels.")
        return Qwen314BModelRunner(
            compiled=compiled,
            device_id=self._device_ids[0],
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
        # The fused all-layer decode lives in decode_fwd.decode_fwd. It is
        # PAGED: it consumes block_table + slot_mapping and reads/writes the SAME
        # device-resident paged KV pool prefill writes (self._kv_caches), so no
        # contiguous bridge / MAX_SEQ env is needed.
        qwen3_decode_fwd = _load_pypto_lib_qwen14b_module("decode_fwd")
        qwen3_greedy_sample = _load_pypto_lib_qwen14b_module("greedy_sample")
        qwen3_l3_dispatch.prefill_fwd = qwen3_prefill_fwd.prefill_fwd
        qwen3_l3_dispatch.decode_fwd = qwen3_decode_fwd.decode_fwd
        qwen3_l3_dispatch.greedy_sample_fwd = qwen3_greedy_sample.greedy_sample_fwd

        _mark("imports")

        self._validate_supported_shape(model)
        kernel_batch = model.runtime.max_batch_size

        kernel_max_seq = int(getattr(qwen3_decode_fwd, "MAX_SEQ", 4096))
        if model.runtime.max_seq_len > kernel_max_seq:
            raise ValueError(
                f"max_model_len {model.runtime.max_seq_len} exceeds the kernel's "
                f"compile-time MAX_SEQ {kernel_max_seq} (config.py). The decode/prefill "
                "kernels precompute MAX_CTX_BLOCKS, NUM_PAGES, and rope table sizes from "
                "MAX_SEQ; a larger runtime value silently produces wrong attention and "
                "out-of-bounds rope reads. Rebuild the kernel with a larger MAX_SEQ."
            )

        if int(qwen3_decode_fwd.BATCH) != kernel_batch:
            raise ValueError(
                "decode_fwd.decode_fwd is compiled for a fixed kernel BATCH of "
                f"{int(qwen3_decode_fwd.BATCH)}, but runtime max_batch_size is "
                f"{kernel_batch}; they must match (decode statically computes and "
                "writes BATCH rows / BATCH logit rows)."
            )
        if int(model.config.num_hidden_layers) != int(qwen3_decode_fwd.NUM_LAYERS):
            raise ValueError(
                "decode_fwd.decode_fwd fuses a FIXED "
                f"NUM_LAYERS={int(qwen3_decode_fwd.NUM_LAYERS)} loop (the layer count "
                "is a kernel constant, not derived from the weight tensors), but the "
                f"model has {model.config.num_hidden_layers} layers. The fused decode "
                "does not support --num-layers-override; run the full model."
            )
        self._validate_total_kv_pages(model, kernel_batch)

        padded_vocab = round_up(model.config.vocab_size, _VOCAB_PAD_MULTIPLE)
        if padded_vocab != int(qwen3_decode_fwd.VOCAB):
            raise ValueError(
                f"decode_fwd.decode_fwd hard-codes VOCAB={int(qwen3_decode_fwd.VOCAB)} "
                f"(config.VOCAB) for its fused LM head, but the runtime padded vocab is "
                f"{padded_vocab} (round_up({model.config.vocab_size}, {_VOCAB_PAD_MULTIPLE})); "
                "they must match for the decode logits buffer / lm_head weight to line up."
            )
        if model.config.vocab_size != int(qwen3_decode_fwd.REAL_VOCAB):
            raise ValueError(
                "decode_fwd.decode_fwd hard-codes REAL_VOCAB for padded-token masking, "
                f"but the runtime model vocab_size is {model.config.vocab_size}; expected "
                f"{int(qwen3_decode_fwd.REAL_VOCAB)}."
            )
        if int(qwen3_greedy_sample.BATCH) != kernel_batch:
            raise ValueError(
                "greedy_sample_fwd is compiled for a fixed kernel BATCH of "
                f"{int(qwen3_greedy_sample.BATCH)}, but runtime max_batch_size is {kernel_batch}."
            )
        if int(qwen3_greedy_sample.VOCAB) != padded_vocab:
            raise ValueError(
                "greedy_sample_fwd VOCAB must match the padded logits vocab: "
                f"{int(qwen3_greedy_sample.VOCAB)} != {padded_vocab}."
            )
        sampled_ids_width = int(
            getattr(qwen3_decode_fwd, "SAMPLED_IDS_PAD", getattr(qwen3_greedy_sample, "SAMPLED_IDS_PAD", 1))
        )
        page_size = model.runtime.page_size
        max_blocks_per_seq = (model.runtime.max_seq_len + page_size - 1) // page_size

        # Populate the HOST wrappers' annotation globals so they can be compiled
        # straight from their signatures (no throwaway torch.empty dummies). The
        # dynamic-dim vars are pulled from the loaded prefill kernel so host and
        # kernel share the SAME pl.dynamic instances (dims unify across the call).
        self._inject_dispatch_shapes(
            qwen3_prefill_fwd,
            hidden_size=model.config.hidden_size,
            kv_hidden=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            intermediate_size=model.config.intermediate_size,
            vocab_size=padded_vocab,
            num_layers=model.config.num_hidden_layers,
            batch=kernel_batch,
            sampled_ids_width=sampled_ids_width,
            rope_seq=model.runtime.max_seq_len,
            decode_block_table_flat=kernel_batch * max_blocks_per_seq,
        )

        prefill = self._compile_jit_fwd_callable("prefill_fwd", qwen3_l3_dispatch.qwen3_prefill_host)
        _mark("compile_prefill")
        decode = self._compile_jit_fwd_callable("decode_fwd", qwen3_l3_dispatch.qwen3_decode_host)
        _mark("compile_decode")
        greedy_sample = self._compile_jit_fwd_callable(
            "greedy_sample_fwd", qwen3_l3_dispatch.qwen3_greedy_sample_host
        )
        _mark("compile_greedy_sample")
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
            padding = lm_head_weight[:1].expand(pad_rows, -1).clone()
            lm_head_weight = torch.cat([lm_head_weight, padding], dim=0)
        padded_lm_head_weight = self._shared_tensor(lm_head_weight.to(torch.bfloat16).contiguous().cpu())
        _mark("pad_lm_head")
        embed_weight = model.embed_tokens
        if padded_vocab != embed_weight.shape[0]:
            pad_rows = padded_vocab - embed_weight.shape[0]
            padding = torch.zeros(
                (pad_rows, embed_weight.shape[1]),
                dtype=embed_weight.dtype,
                device=embed_weight.device,
            )
            embed_weight = torch.cat([embed_weight, padding], dim=0)
        padded_embed_weight = self._shared_tensor(embed_weight.to(torch.bfloat16).contiguous().cpu())
        _mark("pad_embed")
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
        prefill_sampled_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        prefill_next_hidden_buffer = torch.empty(
            (kernel_batch, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        _mark("prefill_buffers")
        decode_logits_buffer = torch.empty(
            (kernel_batch, padded_vocab),
            dtype=torch.float32,
        ).share_memory_()
        decode_seq_lens_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        decode_block_table_buffer = torch.empty(
            (kernel_batch * max_blocks_per_seq,),
            dtype=torch.int32,
        ).share_memory_()
        decode_slot_mapping_buffer = torch.empty((kernel_batch,), dtype=torch.int32).share_memory_()
        decode_token_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        decode_sampled_ids_buffer = torch.empty(
            (kernel_batch, sampled_ids_width),
            dtype=torch.int32,
        ).share_memory_()
        decode_next_hidden_buffer = torch.empty(
            (kernel_batch, model.config.hidden_size),
            dtype=torch.bfloat16,
        ).share_memory_()
        _mark("decode_logits_buffer")

        timer.report()

        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            greedy_sample=greedy_sample,
            final_norm_weight=final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
            padded_embed_weight=padded_embed_weight,
            decode_weights=decode_weights,
            prefill_hidden_buffer=prefill_hidden_buffer,
            prefill_seq_lens_buffer=prefill_seq_lens_buffer,
            prefill_chunk_lens_buffer=prefill_chunk_lens_buffer,
            prefill_chunk_offsets_buffer=prefill_chunk_offsets_buffer,
            prefill_block_table_buffer=prefill_block_table_buffer,
            prefill_slot_mapping_buffer=prefill_slot_mapping_buffer,
            prefill_logits_buffer=prefill_logits_buffer,
            prefill_sampled_ids_buffer=prefill_sampled_ids_buffer,
            prefill_next_hidden_buffer=prefill_next_hidden_buffer,
            decode_seq_lens_buffer=decode_seq_lens_buffer,
            decode_block_table_buffer=decode_block_table_buffer,
            decode_slot_mapping_buffer=decode_slot_mapping_buffer,
            decode_logits_buffer=decode_logits_buffer,
            decode_token_ids_buffer=decode_token_ids_buffer,
            decode_sampled_ids_buffer=decode_sampled_ids_buffer,
            decode_next_hidden_buffer=decode_next_hidden_buffer,
        )

    @staticmethod
    def _inject_dispatch_shapes(
        prefill_module: object,
        *,
        hidden_size: int,
        kv_hidden: int,
        head_dim: int,
        intermediate_size: int,
        vocab_size: int,
        num_layers: int,
        batch: int,
        sampled_ids_width: int,
        rope_seq: int,
        decode_block_table_flat: int,
    ) -> None:
        """Bind the HOST wrappers' annotation globals for signature-mode compile.

        The wrappers in ``qwen3_l3_dispatch`` are compiled with ``compile()`` (no
        sample tensors), so their ``pl.Tensor[[...], dtype]`` annotations — lazy
        strings under ``from __future__ import annotations`` — are resolved from
        that module's globals when compiling. Static extents come from the
        validated model config; the ``pl.dynamic`` vars are re-exported by the
        loaded prefill kernel (``from config import ...``), so binding those exact
        instances lets each host dim unify with the same dim in the callee.
        """
        qwen3_l3_dispatch.HIDDEN = hidden_size
        qwen3_l3_dispatch.KV_HIDDEN = kv_hidden
        qwen3_l3_dispatch.HEAD_DIM = head_dim
        qwen3_l3_dispatch.INTERMEDIATE = intermediate_size
        qwen3_l3_dispatch.VOCAB = vocab_size
        qwen3_l3_dispatch.NUM_LAYERS = num_layers
        qwen3_l3_dispatch.BATCH = batch
        qwen3_l3_dispatch.SAMPLED_IDS_PAD = sampled_ids_width
        qwen3_l3_dispatch.ROPE_SEQ = rope_seq
        qwen3_l3_dispatch.DEC_BLOCK_TABLE_FLAT = decode_block_table_flat
        for dyn_name in (
            "USER_BATCH_DYN",
            "PREFILL_TOKENS_DYN",
            "KV_CACHE_ROWS_DYN",
            "BLOCK_TABLE_FLAT_DYN",
            "LAYER_DYN",
            "LAYER_HIDDEN_ROWS_DYN",
            "LAYER_INTER_ROWS_DYN",
        ):
            setattr(qwen3_l3_dispatch, dyn_name, getattr(prefill_module, dyn_name))

    def _compile_jit_fwd_callable(
        self,
        name: str,
        jit_fn: object,
    ) -> _L3Callable:
        """Compile a HOST wrapper into a PyPTO DistributedCompiledProgram.

        Compiled straight from the wrapper's fully-annotated signature (pypto
        #2014): ``compile()`` reads the shape/dtype contract from the parameter
        annotations, so no throwaway ``torch.empty`` sample tensors are built.
        """
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
        compiled = jit_fn.compile(config=run_config)
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
        """Validate that runtime KV page count covers the batch capacity."""
        if model.runtime.total_kv_pages is None:
            return
        if model.runtime.total_kv_pages < kernel_batch:
            raise ValueError(
                f"total_kv_pages must be at least kernel_batch ({kernel_batch}), "
                f"got {model.runtime.total_kv_pages}"
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
