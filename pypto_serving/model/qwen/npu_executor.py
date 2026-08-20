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
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from pypto_serving.config.types import RuntimeModel
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor as CorePyptoExecutor
from pypto_serving.model.common.executor.utils import (
    build_pypto_run_config,
    rope_tables,
    round_up,
)
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.qwen.npu_runner import (
    _CompiledKernels,
    QwenLayout,
    _L3Callable,
    Qwen314BModelRunner,
)


_VOCAB_PAD_MULTIPLE = 512  # must be a multiple of lm_head.VOCAB_CHUNK (64)
_QWEN14B_PAGE_SIZE = 128
_QWEN14B_TOPK_SELECT_K = 32


def _kernel_batch_pad(kernel_module: object) -> tuple[int, bool]:
    """Return ``(padded row count, module predates the BATCH_PAD rename)``.

    pypto-lib renamed this constant ``BATCH`` -> ``BATCH_PAD`` when decode's
    public batch became dynamic, because the two meanings had been conflated:
    ``BATCH_PAD`` is the padded pipeline width (the M of every matmul), while
    the public batch is a runtime value read from the ``seq_lens`` descriptor.
    Accept either spelling so this file works against a pypto-lib from before
    or after that rename.

    The second element reports only which pypto-lib generation the module came
    from -- NOT that the stage itself takes a dynamic batch. Decode does;
    greedy_sample carries the new spelling but is still fixed-batch. Callers
    must apply their own stage's rule. The distinction matters because a
    pre-rename decode writes exactly ``BATCH`` rows whatever the runtime batch,
    while topk_select remains fixed-batch. Pointing either stage at undersized
    buffers would overrun them.
    """
    value = getattr(kernel_module, "BATCH_PAD", None)
    if value is not None:
        return int(value), True
    value = getattr(kernel_module, "BATCH", None)
    if value is not None:
        return int(value), False
    raise AttributeError(
        f"{getattr(kernel_module, '__name__', kernel_module)!r} exposes neither "
        "BATCH_PAD nor BATCH; cannot determine the kernel's padded batch width"
    )


def _find_pypto_lib_qwen14b_dir(pypto_lib_root: str | None = None) -> Path:
    """Find the Qwen3-14B kernel directory from configuration or a checkout."""
    if pypto_lib_root is None:
        pypto_lib_root = os.environ.get("PYPTO_LIB_ROOT")
    if pypto_lib_root:
        candidate = Path(pypto_lib_root) / "models" / "qwen3_14b"
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Qwen3-14B kernel directory not found under PYPTO_LIB_ROOT={pypto_lib_root!r}")

    start_dir = Path(__file__).resolve().parent
    for directory in (start_dir, *start_dir.parents):
        candidate = directory / "pypto-lib" / "models" / "qwen3_14b"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot locate Qwen3-14B kernels. Run from a checkout with pypto-lib available "
        "or set PYPTO_LIB_ROOT to a pypto-lib checkout."
    )


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
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(kernel_dir))
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
        pypto_build_dir: str = "build_output",
        use_compile_cache: bool = False,
    ) -> None:
        # ``pypto_build_dir`` is the per-worker build dir (set by the serving
        # worker). When ``use_compile_cache`` is set, the compiler writes each
        # kernel straight to ``<pypto_build_dir>/<name>`` and reloads it on a
        # later launch, so it doubles as the on-disk kernel cache. No
        # fingerprinting -- the caller must keep this dir config/kernel-source
        # appropriate. When off, pypto uses its default per-kernel build dirs.
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=device_ids,
            pypto_build_dir=pypto_build_dir,
            use_compile_cache=use_compile_cache,
        )
        self._compiler = KernelCompiler(
            run_config=build_pypto_run_config(
                platform=self._platform,
                device_ids=self._device_ids,
                pypto_build_dir=self._pypto_build_dir,
            ),
            cache_dir=self._pypto_build_dir,
        )

    @property
    def supports_device_sampling(self) -> bool:
        """Qwen3 NPU runner can return greedy sampled token ids."""
        return True

    @property
    def device_topk_sampling_k(self) -> int:
        """Qwen3 NPU runner can return top-k sampling candidates."""
        return _QWEN14B_TOPK_SELECT_K

    @property
    def supports_device_embedding(self) -> bool:
        """Qwen3 NPU prefill and decode embed token ids inside device kernels."""
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
        kernel_dir = _find_pypto_lib_qwen14b_dir()
        qwen3_prefill_fwd = _load_pypto_lib_qwen14b_module("prefill_fwd", kernel_dir)
        # The fused all-layer decode lives in decode_fwd.decode_fwd. It is
        # PAGED: it consumes block_table + slot_mapping and reads/writes the SAME
        # device-resident paged KV pool prefill writes (self._kv_caches), so no
        # contiguous bridge / MAX_SEQ env is needed.
        qwen3_decode_fwd = _load_pypto_lib_qwen14b_module("decode_fwd", kernel_dir)
        qwen3_topk_select = _load_pypto_lib_qwen14b_module("topk_select", kernel_dir)

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

        decode_batch_pad, decode_batch_is_dynamic = _kernel_batch_pad(qwen3_decode_fwd)
        if decode_batch_is_dynamic:
            if kernel_batch > decode_batch_pad:
                raise ValueError(
                    "decode_fwd.decode_fwd pads its pipeline to "
                    f"{decode_batch_pad} rows, but runtime max_batch_size is "
                    f"{kernel_batch}; the runtime batch must not exceed the "
                    "padded width."
                )
        elif decode_batch_pad != kernel_batch:
            raise ValueError(
                "decode_fwd.decode_fwd is compiled for a fixed kernel BATCH of "
                f"{decode_batch_pad}, but runtime max_batch_size is "
                f"{kernel_batch}; they must match (this pypto-lib predates the "
                "dynamic public batch, so decode statically writes BATCH rows / "
                "BATCH logit rows and would overrun smaller buffers)."
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
        # topk_select_fwd remains a fixed-batch stage, so require an exact
        # match while accepting either pypto-lib batch constant spelling.
        topk_select_batch, _ = _kernel_batch_pad(qwen3_topk_select)
        if topk_select_batch != kernel_batch:
            raise ValueError(
                "topk_select_fwd is compiled for a fixed kernel BATCH of "
                f"{topk_select_batch}, but runtime max_batch_size is {kernel_batch}."
            )
        if int(qwen3_topk_select.VOCAB) != padded_vocab:
            raise ValueError(
                "topk_select_fwd VOCAB must match the padded logits vocab: "
                f"{int(qwen3_topk_select.VOCAB)} != {padded_vocab}."
            )
        if model.config.vocab_size != int(qwen3_topk_select.REAL_VOCAB):
            raise ValueError(
                "topk_select_fwd REAL_VOCAB must match model vocab_size: "
                f"{int(qwen3_topk_select.REAL_VOCAB)} != {model.config.vocab_size}."
            )
        topk_width = int(qwen3_topk_select.TOPK)
        if topk_width != _QWEN14B_TOPK_SELECT_K:
            raise ValueError(
                "topk_select_fwd TOPK must match executor capability: "
                f"{topk_width} != {_QWEN14B_TOPK_SELECT_K}."
            )
        sampled_ids_width = int(getattr(qwen3_decode_fwd, "SAMPLED_IDS_PAD", 1))
        page_size = model.runtime.page_size
        max_blocks_per_seq = (model.runtime.max_seq_len + page_size - 1) // page_size
        prefill = self._compile_jit_fwd_callable("prefill_fwd", qwen3_prefill_fwd.qwen3_prefill_host)
        decode = self._compile_jit_fwd_callable("decode_fwd", qwen3_decode_fwd.qwen3_decode_host)
        topk_select = self._compile_jit_fwd_callable(
            "topk_select_fwd", qwen3_topk_select.qwen3_topk_select_host
        )
        rope_cos_raw, rope_sin_raw = rope_tables(
            model.runtime.max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
        )
        rope_cos = self._shared_tensor(rope_cos_raw)
        rope_sin = self._shared_tensor(rope_sin_raw)

        lm_head_weight = model.lm_head
        if padded_vocab != lm_head_weight.shape[0]:
            pad_rows = padded_vocab - lm_head_weight.shape[0]
            padding = lm_head_weight[:1].expand(pad_rows, -1).clone()
            lm_head_weight = torch.cat([lm_head_weight, padding], dim=0)
        padded_lm_head_weight = self._shared_tensor(lm_head_weight.to(torch.bfloat16).contiguous().cpu())
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
        final_norm_weight = self._shared_tensor(model.final_norm_weight.view(1, -1).float().cpu())
        decode_weights = self._stage_stacked_decode_weights(model)
        # The per-dispatch I/O host buffers are owned by the runner's TaskArgs;
        # only the shape-defining constants are passed through here so the
        # TaskArgs builders can size the slots.
        layout = QwenLayout(
            kernel_batch=kernel_batch,
            max_seq_len=model.runtime.max_seq_len,
            page_size=page_size,
            max_blocks_per_seq=max_blocks_per_seq,
            padded_vocab=padded_vocab,
            hidden_size=model.config.hidden_size,
            sampled_ids_width=sampled_ids_width,
            topk_width=topk_width,
        )
        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            topk_select=topk_select,
            final_norm_weight=final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
            padded_embed_weight=padded_embed_weight,
            decode_weights=decode_weights,
            layout=layout,
        )

    def _compile_jit_fwd_callable(
        self,
        name: str,
        jit_fn: object,
    ) -> _L3Callable:
        """Compile a HOST wrapper into a PyPTO DistributedCompiledProgram.

        Signature mode: tensor shapes/dtypes are read from the wrapper's
        annotations, so no positional sample tensors are passed. The on-disk
        cache fast-path and the JIT compile are both handled by the shared
        :class:`KernelCompiler`.
        """
        return self._compiler.compile(name, jit_fn, use_cache=self._use_compile_cache)

    @classmethod
    def _stage_stacked_decode_weights(cls, model: RuntimeModel) -> dict[str, torch.Tensor]:
        """Stage per-layer weights into pre-allocated stacked shm tensors, one layer at a time.

        Each weight kind gets one stacked tensor (num_layers slabs along dim 0): projections are
        transposed to bf16, norm gammas become ``[1, N]`` float32 rows. The layout, the dtypes
        and the checkpoint names it reads all come from ``qwen/weight_spec.py``; what stays here
        is the wiring.

        Peak host memory is ~1x: one layer's raw tensors are read, written into their slice, and
        dropped before the next layer is read. Reading the whole checkpoint up front — as the
        loader used to — kept the state dict, the cast per-layer copies and the destinations all
        alive at once.

        The per-layer copies dominate startup (~90s serially for a 14B), so they run on a thread
        pool: each layer owns a disjoint row-slice of every stacked tensor, and ``copy_`` releases
        the GIL for the memcpy and the dtype cast, so they genuinely overlap.
        """
        from pypto_serving.model.common.weights.packer import pack_layer  # noqa: PLC0415
        from pypto_serving.model.common.weights.spec import LayerContext  # noqa: PLC0415
        from pypto_serving.model.common.weights.stacker import stack_layers  # noqa: PLC0415

        from .weight_spec import (  # noqa: PLC0415
            QWEN_LAYER_RULES,
            QwenWeightStore,
            qwen_layer_prefix,
            qwen_layer_weight_names,
            qwen_policy,
            qwen_stack_groups,
            qwen_staging_policy,
        )

        num_layers = int(model.config.num_hidden_layers)
        store = cls._weight_store(model, QwenWeightStore)
        head_dim = int(model.config.head_dim)

        def context(layer_id: int) -> LayerContext:
            return LayerContext(
                layer_id=layer_id,
                prefix=qwen_layer_prefix(layer_id),
                ranks=1,
                dims={"head_dim": head_dim},
            )

        def read(layer_id: int) -> dict[str, torch.Tensor]:
            # Optional names (the QK norms) may be absent; the rules default them.
            names = [name for name in qwen_layer_weight_names(layer_id) if name in store]
            return store.load_many(names)

        # Layer 0 packed on its own sizes the slabs; the shapes are uniform across a transformer.
        template = pack_layer(QWEN_LAYER_RULES, read(0), context(0), policy=qwen_policy())

        def pack_into(layer_id: int, destinations: dict[str, torch.Tensor]) -> None:
            pack_layer(
                QWEN_LAYER_RULES,
                read(layer_id),
                context(layer_id),
                policy=qwen_policy(),
                destinations=destinations,
            )

        return stack_layers(
            qwen_stack_groups(num_layers),
            template,
            layer_ids=range(num_layers),
            pack_into=pack_into,
            template_layer_id=0,
            stack_axis=0,
            allocate=lambda shape, dtype: torch.empty(shape, dtype=dtype).share_memory_(),
            policy=qwen_staging_policy(num_layers, cls._staging_worker_count(num_layers)),
        )

    @staticmethod
    def _weight_store(model: RuntimeModel, store_cls):
        """Open a lazy checkpoint store from the metadata the loader left in ``extra``."""
        model_dir = model.extra.get("model_dir")
        weight_map = model.extra.get("weight_map")
        if not model_dir or not isinstance(weight_map, dict):
            raise ValueError(
                "Qwen3 staging needs the checkpoint metadata the loader records in "
                "RuntimeModel.extra ('model_dir' and 'weight_map'); this model was loaded "
                "without them."
            )
        return store_cls(model_dir=model_dir, weight_map=weight_map)

    @staticmethod
    def _staging_worker_count(num_layers: int) -> int:
        """Thread count for parallel weight staging (env-tunable)."""
        raw = os.environ.get("PYPTO_STAGING_THREADS")
        if raw:
            try:
                return max(1, min(int(raw), num_layers))
            except ValueError:
                pass
        # Staging is memory-bandwidth bound: it plateaus by ~16-32 threads and
        # regresses beyond that, so cap the default even on many-core hosts.
        return max(1, min(num_layers, os.cpu_count() or 8, 32))

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
    def _shared_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor into shared memory if needed."""
        if tensor.device.type == "cpu" and not tensor.is_shared():
            return tensor.share_memory_()
        return tensor

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
