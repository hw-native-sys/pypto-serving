# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from pypto import passes

from pypto_serving.config.types import RuntimeModel
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor as CorePyptoExecutor
from pypto_serving.model.common.executor.utils import (
    build_pypto_run_config,
    rope_tables,
)
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.qwen.npu_runner import (
    _CompiledKernels,
    QwenLayout,
    _L3Callable,
    Qwen314BModelRunner,
)


_REQUIRED_QWEN_CAPABILITIES = frozenset(
    {
        "paged_kv",
        "chunked_prefill",
        "device_greedy_sampling",
        "device_topk_sampling",
        "device_embedding",
    }
)


def _find_pypto_lib_root(pypto_lib_root: str | None = None) -> Path:
    """Find the pypto-lib root containing the external Contract registry."""
    if pypto_lib_root is None:
        pypto_lib_root = os.environ.get("PYPTO_LIB_ROOT")
    if pypto_lib_root:
        candidate = Path(pypto_lib_root)
        if (candidate / "contract" / "registry.py").is_file():
            return candidate
        raise FileNotFoundError(
            f"pypto-lib Contract registry not found under PYPTO_LIB_ROOT={pypto_lib_root!r}"
        )

    start_dir = Path(__file__).resolve().parent
    for directory in (start_dir, *start_dir.parents):
        candidate = directory / "pypto-lib"
        if (candidate / "contract" / "registry.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot locate pypto-lib Contract registry. Run from a checkout with pypto-lib available "
        "or set PYPTO_LIB_ROOT to a pypto-lib checkout."
    )


def _load_pypto_lib_contract_registry(pypto_lib_root: str | None = None) -> object:
    """Import the registry from the selected pypto-lib checkout."""
    root = _find_pypto_lib_root(pypto_lib_root).resolve()
    sys.path.insert(0, str(root))
    try:
        registry = importlib.import_module("contract.registry")
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    actual = Path(registry.__file__).resolve()
    expected = (root / "contract" / "registry.py").resolve()
    if actual != expected:
        raise ImportError(
            f"contract.registry is already loaded from {actual}, "
            f"expected the selected pypto-lib at {expected}"
        )
    return registry


def _load_qwen3_14b_contract(pypto_lib_root: str | None = None) -> object:
    """Resolve the Qwen3-14B Contract through pypto-lib's public registry."""
    return _load_pypto_lib_contract_registry(pypto_lib_root).get_contract("qwen3", "14b")


def _validate_qwen_contract_surface(contract: object) -> None:
    """Fail fast when pypto-lib lacks a serving capability or stage."""
    missing = _REQUIRED_QWEN_CAPABILITIES.difference(contract.capabilities)
    if missing:
        raise ValueError(f"Qwen3-14B Contract lacks required capabilities: {sorted(missing)}")
    for flow in ("prefill", "decode", "topk_select"):
        stages = tuple(contract.execution.get(flow, ()))
        if len(stages) != 1 or stages[0] not in contract.kernels:
            raise ValueError(
                f"Qwen3-14B Contract flow {flow!r} must resolve to exactly one "
                f"registered kernel; got {stages}"
            )


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
        self._contract_registry = _load_pypto_lib_contract_registry()
        self._contract = self._contract_registry.get_contract("qwen3", "14b")
        _validate_qwen_contract_surface(self._contract)

    @property
    def supports_device_sampling(self) -> bool:
        """Qwen3 NPU runner can return greedy sampled token ids."""
        return "device_greedy_sampling" in self._contract.capabilities

    @property
    def device_topk_sampling_k(self) -> int:
        """Qwen3 NPU runner can return top-k sampling candidates."""
        return int(self._contract.limits["topk"])

    @property
    def supports_device_embedding(self) -> bool:
        """Qwen3 NPU prefill and decode embed token ids inside device kernels."""
        return "device_embedding" in self._contract.capabilities

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
        contract = self._contract_registry.find_contract_for_model_config(model.config)
        _validate_qwen_contract_surface(contract)
        self._contract = contract

        kernel_batch = int(contract.limits["batch"])
        runtime_batch = int(model.runtime.max_batch_size)
        if runtime_batch > kernel_batch:
            raise ValueError(
                f"Qwen3-14B Contract supports max_batch_size <= {kernel_batch}, got {runtime_batch}."
            )
        self._validate_total_kv_pages(model, runtime_batch)

        runtime_values = vars(model.runtime).copy()
        runtime_values.update(
            max_batch_size=kernel_batch,
            vocab_pad_multiple=int(contract.limits["vocab_pad_multiple"]),
        )
        contract_runtime = SimpleNamespace(**runtime_values)
        loaded_kernels = contract.load_kernels()
        contract.kernel_binder(**loaded_kernels.functions)
        contract.validate_kernels(
            contract,
            loaded_kernels,
            SimpleNamespace(config=model.config, runtime=contract_runtime),
        )

        def compile_flow(flow: str) -> _L3Callable:
            stage_name = contract.execution[flow][0]
            stage = contract.kernels[stage_name]
            compile_args = stage.compile_args_builder(model.config, contract_runtime)
            return self._compile_jit_fwd_callable(
                f"{stage.name}_fwd",
                stage.host_jit_fn,
                *compile_args,
                memory_planner=(passes.MemoryPlanner.PTOAS if flow == "topk_select" else None),
            )

        prefill = compile_flow("prefill")
        decode = compile_flow("decode")
        topk_select = compile_flow("topk_select")
        padded_vocab = int(contract.limits["vocab"])
        sampled_ids_width = int(contract.limits["sampled_ids_pad"])
        topk_width = int(contract.limits["topk"])
        sampling_control_fields = int(contract.limits["sampling_control_fields"])
        page_size = int(contract.limits["page_size"])
        max_blocks_per_seq = (model.runtime.max_seq_len + page_size - 1) // page_size
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
        # The Contract hook consumes already-materialized ``runtime_model.layers``.
        # Serving intentionally keeps checkpoints lazy and stages one layer at
        # a time to cap startup host memory, while preserving the Contract ABI.
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
            sampling_control_fields=sampling_control_fields,
        )
        return _CompiledKernels(
            contract=contract,
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
        *compile_args: object,
        memory_planner: passes.MemoryPlanner | None = None,
    ) -> _L3Callable:
        """Compile a HOST wrapper into a PyPTO DistributedCompiledProgram.

        Contract stages pass positional sample tensors for generic HOST
        annotations; direct callers may omit them to retain signature mode. The
        shared :class:`KernelCompiler` owns both compilation and cache reload.
        """
        run_config_overrides = {"memory_planner": memory_planner} if memory_planner is not None else None
        return self._compiler.compile(
            name,
            jit_fn,
            *compile_args,
            use_cache=self._use_compile_cache,
            run_config_overrides=run_config_overrides,
        )

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
