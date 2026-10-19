# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile accepted Qwen3.8 whole-network PyPTO entries for one NPU."""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from pypto_serving.config.types import GenerateConfig, ModelRecord, RuntimeModel
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.compiler.l3_callable import L3Callable
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.qwen3_5.weight_spec import (
    MAX_SEQUENCE, VOCAB, WEIGHT_ORDER, load_qwen35_weight_banks,
)


def _find_qwen35_kernel_dir(pypto_lib_root: str | None = None) -> Path:
    """Resolve the pinned serving submodule unless an explicit development root is set."""
    override = pypto_lib_root if pypto_lib_root is not None else os.environ.get("PYPTO_LIB_ROOT")
    base = Path(override) if override else Path(__file__).resolve().parents[3] / "pypto-lib"
    kernel = base if base.name == "qwen3_8_27b" else base / "models/qwen3_8_27b"
    if not (kernel / "prefill_fwd.py").is_file() or not (kernel / "decode_fwd.py").is_file():
        raise FileNotFoundError(f"accepted Qwen3.8 whole-network entries are missing in {kernel}")
    return kernel


def _load_kernel_module(name: str, root: Path):
    path = root / f"{name}.py"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise RuntimeError(f"kernel module {name} came from another pypto-lib checkout")
        return existing
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(root))


def _rope_table() -> tuple[torch.Tensor, torch.Tensor]:
    angles = [[position / (10_000_000 ** (2 * dim / 64))
               for dim in range(32)] for position in range(MAX_SEQUENCE)]
    cos = torch.tensor([[math.cos(angle) for angle in row] * 2 for row in angles], dtype=torch.bfloat16)
    sin = torch.tensor([[math.sin(angle) for angle in row] * 2 for row in angles], dtype=torch.bfloat16)
    return cos.share_memory_(), sin.share_memory_()


@dataclass
class Qwen35Compiled:
    prefill: L3Callable
    decode: L3Callable
    weights: tuple[torch.Tensor, ...]
    cos: torch.Tensor
    sin: torch.Tensor


class Qwen35PyptoExecutor(PyptoExecutor):
    """Dedicated one-card Qwen3.8 executor; no Qwen14B paged-cache reuse."""

    def __init__(
        self, kv_cache_manager=None, *, platform: str = "a2a3",
        device_ids: Sequence[int] = (0,), pypto_build_dir: str | None = None,
    ) -> None:
        super().__init__(kv_cache_manager, platform=platform, device_ids=device_ids,
                         pypto_build_dir=pypto_build_dir)
        if len(self._device_ids) != 1:
            raise ValueError("Qwen3.8 text requires exactly one NPU")
        self._compiler = KernelCompiler(run_config=build_pypto_run_config(
            platform=platform, device_ids=self._device_ids, pypto_build_dir=pypto_build_dir,
        ))

    @property
    def supports_device_embedding(self) -> bool:
        return True

    def validate_generate_batch(self, record: ModelRecord, batch_size: int, config: GenerateConfig) -> None:
        if batch_size != 1 or record.runtime.max_seq_len > MAX_SEQUENCE:
            raise ValueError("Qwen3.8 text supports one request and at most 128 positions")
        if config.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    def _compile_model(self, model: RuntimeModel) -> Qwen35Compiled:
        if (model.extra.get("family") != "qwen3_5" or model.config.num_hidden_layers != 64
                or model.config.hidden_size != 5120 or model.config.head_dim != 256
                or model.config.vocab_size != VOCAB or model.runtime.max_batch_size != 1
                or model.runtime.max_seq_len > MAX_SEQUENCE or model.runtime.num_speculative_tokens):
            raise ValueError("model/runtime differs from the accepted Qwen3.8 whole-network ABI")
        kernel_dir = _find_qwen35_kernel_dir()
        prefill = _load_kernel_module("prefill_fwd", kernel_dir)
        decode = _load_kernel_module("decode_fwd", kernel_dir)
        compiled_prefill = self._compiler.compile("qwen35_prefill_fwd", prefill.prefill_fwd)
        compiled_decode = self._compiler.compile("qwen35_decode_fwd", decode.decode_fwd)
        banks = load_qwen35_weight_banks(
            Path(model.extra["model_dir"]), model.extra["weight_map"], model.extra["config_data"],
        )
        weights = tuple(banks[name].contiguous().share_memory_() for name in WEIGHT_ORDER)
        cos, sin = _rope_table()
        return Qwen35Compiled(compiled_prefill, compiled_decode, weights, cos, sin)

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        from pypto_serving.model.qwen3_5.npu_runner import Qwen35ModelRunner

        if not isinstance(compiled, Qwen35Compiled):
            raise TypeError("Qwen35PyptoExecutor requires Qwen3.8 compiled kernels")
        return Qwen35ModelRunner(compiled=compiled, device_id=self._device_ids[0])

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        for runner in self._runners.values():
            runner.release_finished_requests(request_ids)
