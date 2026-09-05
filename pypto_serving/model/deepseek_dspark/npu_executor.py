# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO executor for the DSpark DeepSeek-V4-Flash target kernels.

Compiles ``l3_prefill_fwd`` and ``l3_decode_fwd`` from
``pypto-lib/models/deepseek_v4_flash_dspark`` with the canonical 16-card
import contract (``--tp 4 --ep 16 --weight-bank-size 43``) and hands the
compiled programs to :class:`DSparkModelRunner`.  The DSpark kernels freeze
their shapes from ``sys.argv`` at import, so every import runs inside
``_dspark_import_context``; the module names collide with the MTP variant's
(``config``, ``moe``, ``lm_head`` ...), so the context evicts any same-named
module that resolves to either kernel directory on entry and restores the
prior state on exit.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from pypto_serving.config.types import RuntimeModel
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor as CorePyptoExecutor
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.deepseek_dspark.npu_runner import (
    DSPARK_FWD_NUM_LAYERS,
    DSparkCacheLayout,
    DSparkCompiledKernels,
    DSparkRopeTables,
    DSparkModelRunner,
    build_dspark_layer_plan,
)
from pypto_serving.model.deepseek_dspark.weight_loader import DSparkWeightStore
from pypto_serving.tools.profile import profile_span

_DSPARK_KERNEL_DIRNAME = "deepseek_v4_flash_dspark"

# Every top-level module in the DSpark kernel directory.  The eviction set is
# the full directory (not just what M1 imports) so a later drafter import can
# never observe a stale MTP module under one of these names.
_DSPARK_IMPORT_MODULES = (
    "config",
    "decode_compressor_ratio128",
    "decode_compressor_ratio4",
    "decode_cp_token_allgather",
    "decode_csa",
    "decode_fwd",
    "decode_hca",
    "decode_indexer",
    "decode_indexer_compressor",
    "decode_layer",
    "decode_metadata",
    "decode_o_proj",
    "decode_sparse_attn_csa",
    "decode_sparse_attn_hca",
    "decode_sparse_attn_swa",
    "decode_swa",
    "dspark_attention",
    "dspark_context_kv",
    "dspark_drafter",
    "dspark_markov",
    "dspark_prefill",
    "dspark_proj",
    "expert_routed",
    "expert_shared",
    "gate",
    "hc_head",
    "hc_post",
    "hc_pre",
    "lm_head",
    "lookup_embedding",
    "markov_head",
    "moe",
    "prefill_compressor_ratio128",
    "prefill_compressor_ratio4",
    "prefill_cp_token_allgather",
    "prefill_csa",
    "prefill_fwd",
    "prefill_hca",
    "prefill_indexer",
    "prefill_indexer_compressor",
    "prefill_layer",
    "prefill_o_proj",
    "prefill_sparse_attn",
    "prefill_swa",
    "qkv_proj_rope",
    "rmsnorm",
    "rope_interleave",
    "utils",
)


def _find_pypto_lib_dspark_dir(pypto_lib_root: str | None = None) -> Path:
    """Find the DSpark kernel directory."""
    if pypto_lib_root is None:
        pypto_lib_root = os.environ.get("PYPTO_LIB_ROOT")
    if pypto_lib_root:
        candidate = Path(pypto_lib_root) / "models" / _DSPARK_KERNEL_DIRNAME
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"DSpark kernel directory not found under PYPTO_LIB_ROOT={pypto_lib_root!r}"
        )
    start_dir = Path(__file__).resolve().parent
    for directory in (start_dir, *start_dir.parents):
        candidate = directory / "pypto-lib" / "models" / _DSPARK_KERNEL_DIRNAME
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot locate DSpark kernels. Run from a checkout with pypto-lib available "
        "or set PYPTO_LIB_ROOT to a pypto-lib checkout."
    )


def _is_dspark_module_file(path: Path, kernel_dir: Path) -> bool:
    """Return whether ``path`` is one of the top-level DSpark kernel modules."""
    resolved = path.resolve()
    if resolved.is_relative_to(kernel_dir):
        return True
    parts = resolved.parts
    return len(parts) >= 3 and parts[-3:-1] == ("models", _DSPARK_KERNEL_DIRNAME)


@contextlib.contextmanager
def _dspark_import_context(
    kernel_dir: Path,
    *,
    pypto_lib_root: Path,
    tp: int,
    ep: int,
    weight_bank_size: int,
):
    """Import DSpark kernels with the canonical 16-card shape arguments."""
    old_argv = list(sys.argv)
    old_path = list(sys.path)
    missing = object()
    old_modules = {
        module_name: sys.modules.get(module_name, missing)
        for module_name in _DSPARK_IMPORT_MODULES
    }
    for module_name in _DSPARK_IMPORT_MODULES:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is not None and _is_dspark_module_file(Path(module_file), kernel_dir):
            sys.modules.pop(module_name, None)
    sys.argv = [
        "pypto-serving-dspark",
        "--tp", str(int(tp)),
        "--ep", str(int(ep)),
        "--weight-bank-size", str(int(weight_bank_size)),
    ]
    sys.path.insert(0, str(kernel_dir))
    sys.path.insert(0, str(pypto_lib_root))
    try:
        yield
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        for module_name, module in old_modules.items():
            if module is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module


class DeepSeekV4DSparkPyptoExecutor(CorePyptoExecutor):
    """PyPTO executor boundary for DSpark target-model serving."""

    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform: str = "a2a3sim",
        device_id: int = 0,
        device_ids: Sequence[int] | None = None,
        pypto_build_dir: str = "build_output",
        use_compile_cache: bool = False,
        compile_kernels: bool = False,
        num_speculative_tokens: int = 0,
    ) -> None:
        worker_device_ids = tuple(device_ids) if device_ids is not None else (int(device_id),)
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=worker_device_ids,
            pypto_build_dir=pypto_build_dir,
            use_compile_cache=use_compile_cache,
        )
        self._kernel_dir = _find_pypto_lib_dspark_dir()
        self._compile_kernels = bool(compile_kernels)
        self._num_speculative_tokens = int(num_speculative_tokens)
        if self._num_speculative_tokens != 0:
            raise ValueError(
                "DSpark serving runs the target model without speculation in this "
                "milestone; the drafter chain is tracked by pypto-lib#1078"
            )
        self._embedding_cache: dict[str, torch.Tensor] = {}
        compile_cache_dir = self._pypto_build_dir if self._use_compile_cache else None
        self._compiler = KernelCompiler(
            run_config=build_pypto_run_config(
                platform=self._platform,
                device_ids=self._device_ids,
                pypto_build_dir=compile_cache_dir,
            ),
            cache_dir=compile_cache_dir,
        )

    @property
    def max_prefill_batch_size(self) -> int:
        """One prefill request per TP group."""
        return DSparkCacheLayout().prefill_batch

    @property
    def supports_device_sampling(self) -> bool:
        """The kernels sample greedily on device for both dispatch classes."""
        return True

    @property
    def supports_device_decode_embedding(self) -> bool:
        """Decode embeds token ids from the resident table on device."""
        return True

    @property
    def supports_async_decode_prepare(self) -> bool:
        """Keep M1 on the synchronous decode path."""
        return False

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        """Prefill embedding lookup from the lazily loaded DSpark table."""
        runner = self._runners.get(model.config.model_id)
        if runner is None:
            raise RuntimeError(f"DSpark model {model.config.model_id!r} is not registered")
        embed_weight = self._compiled[model.config.model_id].embedding_weight
        if embed_weight is None:
            embed_weight = self._embedding_cache.get(model.config.model_id)
        if embed_weight is None:
            embed_weight = (
                self._compiled[model.config.model_id]
                .weight_store.load_tensor("embed.weight")
                .contiguous()
            )
            if embed_weight.ndim != 2:
                raise ValueError(
                    f"embed.weight must be rank-2, got shape={tuple(embed_weight.shape)}"
                )
        self._compiled[model.config.model_id].embedding_weight = embed_weight
        self._embedding_cache[model.config.model_id] = embed_weight
        flat_ids = token_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        embeddings = embed_weight.index_select(0, flat_ids)
        return embeddings.reshape(*token_ids.shape, model.config.hidden_size).to(
            device=token_ids.device
        )

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Forward request release to the runners (a no-op in M1)."""
        for runner in self._runners.values():
            release = getattr(runner, "release_finished_requests", None)
            if callable(release):
                release(request_ids)

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the DSpark runtime runner."""
        if not isinstance(compiled, DSparkCompiledKernels):
            raise TypeError("DeepSeekV4DSparkPyptoExecutor requires DSpark compiled metadata.")
        return DSparkModelRunner(compiled=compiled)

    def _compile_model(self, model: RuntimeModel) -> DSparkCompiledKernels:
        """Validate DSpark metadata, compile the two L3 programs, and package."""
        metadata = model.extra
        if metadata.get("family") != "deepseek_v4":
            raise ValueError("DeepSeekV4DSparkPyptoExecutor received a non-DeepSeekV4 model")
        if metadata.get("checkpoint_format") != "w8a8-compressed-tensors":
            raise ValueError(
                "DeepSeekV4DSparkPyptoExecutor requires the W8A8 compressed-tensors checkpoint"
            )
        layout = DSparkCacheLayout()
        layout.validate_runtime(model.config, model.runtime, self._device_ids)
        compress_ratios = tuple(int(ratio) for ratio in metadata["compress_ratios"])
        if len(compress_ratios) < model.config.num_hidden_layers + 1:
            raise ValueError(
                "DSpark compress_ratios must include one entry per hidden layer plus "
                "the drafter/final entries"
            )
        config_data = metadata.get("config_data", {})
        n_routed_experts = (
            int(config_data.get("n_routed_experts", 256)) if isinstance(config_data, dict) else 256
        )
        num_hash_layers = (
            int(config_data.get("num_hash_layers", 3)) if isinstance(config_data, dict) else 3
        )
        layer_plan = build_dspark_layer_plan(
            compress_ratios=compress_ratios,
            num_hidden_layers=model.config.num_hidden_layers,
            num_hash_layers=num_hash_layers,
        )
        weight_map = dict(metadata["weight_map"])
        weight_store = DSparkWeightStore(
            model_dir=str(metadata["model_dir"]), weight_map=weight_map
        )
        weight_store.validate_startup_contract(
            num_hidden_layers=model.config.num_hidden_layers,
            n_routed_experts=n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=num_hash_layers,
        )

        prefill = None
        decode = None
        rope: DSparkRopeTables | None = None
        if self._compile_kernels:
            modules = self._load_kernel_modules(layout)
            prefill = self._compile_l3_callable("dspark_prefill", modules["prefill_fwd"].l3_prefill_fwd)
            decode = self._compile_l3_callable("dspark_decode", modules["decode_fwd"].l3_decode_fwd)
            rope = self._build_rope_tables(
                modules["utils"],
                modules["config"],
                max_position=int(model.runtime.max_seq_len),
            )

        return DSparkCompiledKernels(
            layout=layout,
            model_dir=str(metadata["model_dir"]),
            weight_map=weight_map,
            weight_store=weight_store,
            compress_ratios=compress_ratios,
            layer_plan=layer_plan,
            kernel_dir=str(self._kernel_dir),
            runtime_model=model,
            prefill=prefill,
            decode=decode,
            rope=rope,
            platform=self._platform,
            device_id=self._device_ids[0],
            device_ids=self._device_ids,
            n_routed_experts=n_routed_experts,
            num_hash_layers=num_hash_layers,
        )

    def _load_kernel_modules(self, layout: DSparkCacheLayout) -> dict[str, object]:
        """Import the DSpark pypto-lib modules with the canonical shapes frozen."""
        pypto_lib_root = self._kernel_dir.parents[1]
        with _dspark_import_context(
            self._kernel_dir,
            pypto_lib_root=pypto_lib_root,
            tp=layout.tp_size,
            ep=layout.ranks,
            weight_bank_size=DSPARK_FWD_NUM_LAYERS,
        ):
            config = importlib.import_module("config")
            utils = importlib.import_module("utils")
            decode_fwd = importlib.import_module("decode_fwd")
            prefill_fwd = importlib.import_module("prefill_fwd")
        return {"config": config, "utils": utils, "decode_fwd": decode_fwd, "prefill_fwd": prefill_fwd}

    def _compile_l3_callable(self, name: str, jit_fn: object):
        """Compile one fully annotated DSpark HOST wrapper."""
        with profile_span(f"DeepSeekV4DSparkPyptoExecutor.compile.{name}", cat="executor"):
            return self._compiler.compile(name, jit_fn, use_cache=self._use_compile_cache)

    def _build_rope_tables(
        self,
        utils_module: object,
        config_module: object,
        *,
        max_position: int,
    ) -> DSparkRopeTables:
        """Build the four position-indexed rope base tables.

        The rows are gathered per step by the runner, so one full table per
        rope profile (uncompressed, ratio-4 YaRN, ratio-128 YaRN, and the
        half-width FP32 HCA compressor variant) covers both dispatch classes.
        """
        model_config = config_module.FLASH
        positions = torch.arange(max_position, dtype=torch.int64)

        def build(ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
            cos, sin = utils_module.token_local_rope(
                model_config,
                ratio,
                positions,
                max_seq_len=max_position,
                dtype=torch.bfloat16,
            )
            return cos.contiguous(), sin.contiguous()

        swa_cos, swa_sin = build(0)
        ratio4_cos, ratio4_sin = build(4)
        ratio128_cos, ratio128_sin = build(128)
        return DSparkRopeTables(
            max_position=max_position,
            swa_cos=swa_cos,
            swa_sin=swa_sin,
            ratio4_cos=ratio4_cos,
            ratio4_sin=ratio4_sin,
            ratio128_cos=ratio128_cos,
            ratio128_sin=ratio128_sin,
            ratio128_half_cos=ratio128_cos[:, :32].float().contiguous(),
            ratio128_half_sin=ratio128_sin[:, :32].float().contiguous(),
        )
