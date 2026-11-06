# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8 kernel discovery, weight staging and compilation for one NPU."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch

from pypto_serving.config.types import RuntimeModel
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor as CorePyptoExecutor
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.common.weights.packer import pack_globals
from pypto_serving.model.qwen3_8_27b.cache import build_qwen38_cache_group_specs
from pypto_serving.model.qwen3_8_27b.npu_runner import (
    Qwen38CompiledKernels,
    Qwen38Layout,
    QWEN38_HIDDEN_SIZE,
    QWEN38_VOCAB_SIZE,
)
from pypto_serving.model.qwen3_8_27b.weight_spec import (
    QWEN38_GLOBAL_RULES,
    Qwen38WeightStore,
    stage_qwen38_weights,
)


def find_qwen38_kernel_dir(pypto_lib_root: str | None = None) -> Path:
    """Find the Qwen3.8 kernels, allowing an explicit development worktree."""
    root = pypto_lib_root or os.environ.get("PYPTO_LIB_ROOT")
    if root:
        candidate = Path(root) / "models" / "qwen3_8_27b"
        if candidate.is_dir():
            return candidate.resolve()
        raise FileNotFoundError(f"Qwen3.8 kernels not found under PYPTO_LIB_ROOT={root!r}")
    start = Path(__file__).resolve().parent
    for directory in (start, *start.parents):
        candidate = directory / "pypto-lib" / "models" / "qwen3_8_27b"
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot locate Qwen3.8 kernels in the pinned pypto-lib submodule. "
        "Initialize the submodule or set PYPTO_LIB_ROOT to a compatible pypto-lib checkout."
    )


def load_qwen38_serving_contract(pypto_lib_root: str | None = None):
    """Load the kernel-owned capability manifest without importing model kernels."""
    path = find_qwen38_kernel_dir(pypto_lib_root) / "serving_contract.py"
    if not path.is_file():
        raise FileNotFoundError(f"Qwen3.8 serving contract not found at {path}")
    name = f"_pypto_lib_qwen38_serving_contract_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Qwen3.8 serving contract from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    contract = getattr(module, "QWEN3_8_27B_SERVING_CONTRACT", None)
    if getattr(contract, "schema_version", None) != "1":
        raise ValueError("Qwen3.8 requires serving contract schema version 1")
    return contract


def load_qwen38_kernel_module(name: str, kernel_dir: Path):
    """Import one flat PyPTO kernel module without silently reusing another family's file."""
    path = kernel_dir / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Qwen3.8 kernel module: {path}")
    existing = sys.modules.get(name)
    if existing is not None:
        found = getattr(existing, "__file__", None)
        if found is None or Path(found).resolve() != path.resolve():
            raise RuntimeError(f"kernel module {name!r} is already loaded from {found}, not {path}")
        return existing
    sys.path.insert(0, str(kernel_dir))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError(f"kernel module {name!r} resolved to {module.__file__}, not {path}")
    return module


@contextmanager
def _qwen38_import_context(kernel_dir: Path):
    """Isolate the flat pypto-lib imports from other model families."""
    missing = object()
    previous = {path.stem: sys.modules.get(path.stem, missing) for path in kernel_dir.glob("*.py")}
    old_path = list(sys.path)
    for name in previous:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(kernel_dir))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class Qwen38PyptoExecutor(CorePyptoExecutor):
    """Compile the Qwen3.8 text prefill, decoder stack, embedding and head."""

    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform: str = "a2a3sim",
        device_ids: Sequence[int] = (0,),
        pypto_build_dir: str = "build_output",
        use_compile_cache: bool = False,
    ) -> None:
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=device_ids,
            pypto_build_dir=pypto_build_dir,
            use_compile_cache=use_compile_cache,
        )
        if len(self._device_ids) != 1:
            raise ValueError("Qwen3.8 kernels currently run on one NPU per replica")
        self._compiler = KernelCompiler(
            run_config=build_pypto_run_config(
                platform=self._platform,
                device_ids=self._device_ids,
                pypto_build_dir=self._pypto_build_dir,
            ),
            cache_dir=self._pypto_build_dir,
        )

    @property
    def supports_device_embedding(self) -> bool:
        return True

    def _create_runner(self, model_id: str, compiled: object):
        """Attach the model's compiled resources to its single-device runner."""
        from pypto_serving.model.qwen3_8_27b.model_runner import Qwen38ModelRunner

        return Qwen38ModelRunner(model_id=model_id, compiled=compiled, device_id=self._device_ids[0])

    @staticmethod
    def _validate_model(model: RuntimeModel) -> Qwen38Layout:
        config, runtime = model.config, model.runtime
        if model.extra.get("family") != "qwen3_8_27b":
            raise ValueError("Qwen3.8 executor requires the Qwen3.8 W8A8 directory loader")
        if (config.hidden_size, config.vocab_size, config.num_hidden_layers, config.head_dim) != (
            QWEN38_HIDDEN_SIZE, QWEN38_VOCAB_SIZE, 64, 256
        ):
            raise ValueError("Qwen3.8 runtime model disagrees with the fixed kernel dimensions")
        if runtime.page_size != 128 or runtime.kv_dtype not in ("bfloat16", "bf16"):
            raise ValueError("Qwen3.8 cache requires 128-token BF16 attention pages")
        if runtime.kv_cache_groups != build_qwen38_cache_group_specs(runtime.max_seq_len):
            raise ValueError("Qwen3.8 runtime must carry its two declared cache groups")
        tokens = min(runtime.max_prefill_tokens_per_request or runtime.max_num_batched_tokens,
                     runtime.max_seq_len)
        load_qwen38_serving_contract().validate_prefill_tokens(tokens)
        pages_per_seq = (runtime.max_seq_len + 127) // 128
        return Qwen38Layout(
            max_seq_len=runtime.max_seq_len,
            max_batch_size=runtime.max_batch_size,
            prefill_tokens=tokens,
            num_kv_pages=runtime.max_batch_size * pages_per_seq,
        )

    def _compile_model(self, model: RuntimeModel) -> Qwen38CompiledKernels:
        layout = self._validate_model(model)
        kernel_dir = find_qwen38_kernel_dir()
        with _qwen38_import_context(kernel_dir):
            prefill_module = load_qwen38_kernel_module("prefill_text_fwd", kernel_dir)
            decode_module = load_qwen38_kernel_module("decode_fwd", kernel_dir)
            embedding_module = load_qwen38_kernel_module("embedding", kernel_dir)
            head_module = load_qwen38_kernel_module("rms_lm_head", kernel_dir)

            prefill_by_rows = {}
            for rows in range(1, layout.max_prefill_rows + 1):
                prefill_by_rows[rows] = self._compiler.compile(
                    f"qwen38_prefill_text_packed_t{layout.prefill_tokens}_r{rows}"
                    f"_p{layout.max_blocks_per_seq}_k{layout.kv_layer_stride}"
                    f"_s{layout.conv_pool_shape[0]}",
                    prefill_module.build_l3_kernel(
                        t=layout.prefill_tokens, rows=rows,
                        max_pages=layout.max_blocks_per_seq,
                        kv_layer_stride=layout.kv_layer_stride,
                        state_count=layout.conv_pool_shape[0],
                    ),
                    use_cache=self._use_compile_cache,
                )
            decode_fwd_by_rows = {}
            for rows in (layout.decode_batch_pad, *range(1, min(2, layout.max_batch_size) + 1)):
                active_rows = 0 if rows == layout.decode_batch_pad else rows
                suffix = f"_r{rows}_h{decode_module.ACTIVE_HEAD_TILE}" if active_rows else ""
                decode_fwd_by_rows[rows] = self._compiler.compile(
                    f"qwen38_decode_fwd_b{layout.decode_batch_pad}_g16"
                    f"_p{layout.max_blocks_per_seq}_k{layout.kv_pool_shape[0]}"
                    f"_s{layout.conv_pool_shape[0]}{suffix}",
                    decode_module.build_l3_kernel(
                        layout.decode_batch_pad, max_pages=layout.max_blocks_per_seq,
                        kv_rows=layout.kv_pool_shape[0], state_count=layout.conv_pool_shape[0],
                        active_rows=active_rows,
                    ),
                    use_cache=self._use_compile_cache,
                )
            embedding = self._compiler.compile(
                "qwen38_decode_embedding",
                embedding_module.build_l3_kernel(layout.decode_batch_pad),
                use_cache=self._use_compile_cache,
            )
            final_head = self._compiler.compile(
                "qwen38_decode_final_head",
                head_module.build_l3_kernel(layout.decode_batch_pad),
                use_cache=self._use_compile_cache,
            )

        model_dir = model.extra.get("model_dir")
        weight_map = model.extra.get("weight_map")
        if not model_dir or not isinstance(weight_map, dict):
            raise ValueError("Qwen3.8 staging requires model_dir and weight_map metadata")
        store = Qwen38WeightStore(model_dir=model_dir, weight_map=weight_map)
        workers = min(16, os.cpu_count() or 1)
        weights = stage_qwen38_weights(
            store, workers=workers,
            allocate=lambda shape, dtype: torch.empty(shape, dtype=dtype).share_memory_(),
        )
        globals_ = pack_globals(
            QWEN38_GLOBAL_RULES,
            store.load_many(tuple(rule.source for rule in QWEN38_GLOBAL_RULES)),
        )
        globals_ = {name: tensor.share_memory_() for name, tensor in globals_.items()}
        return Qwen38CompiledKernels(
            prefill_by_rows=prefill_by_rows,
            decode_fwd_by_rows=decode_fwd_by_rows,
            decode_embedding=embedding,
            final_head=final_head,
            weights=weights,
            globals=globals_,
            layout=layout,
            kernel_dir=str(kernel_dir),
            prefill_ring_heap=tuple(prefill_module.text_ring_heap(layout.prefill_tokens)),
        )
