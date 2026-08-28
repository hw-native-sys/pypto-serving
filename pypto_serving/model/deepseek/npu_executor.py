# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

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
from pypto_serving.model.deepseek.npu_runner import (
    DEEPSEEK_V4_DSPARK_TP_SIZE,
    DEEPSEEK_V4_FWD_NUM_LAYERS,
    DEEPSEEK_V4_LM_HEAD_TP_SIZE,
    DeepSeekV4CacheLayout,
    DeepSeekV4CompiledKernels,
    DeepSeekV4L3Callable,
    DeepSeekV4ModelRunner,
    build_deepseek_v4_layer_plan,
    deepseek_v4_decode_layout,
    deepseek_v4_dspark_decode_layout,
)
from pypto_serving.model.deepseek.weight_loader import (
    DeepSeekV4StackedLayerWeights,
    DeepSeekV4WeightStore,
)
from pypto_serving.tools.profile import profile_span

_DEEPSEEK_V4_KERNEL_DIRNAMES = {
    None: "deepseek_v4_flash_mtp",
    "mtp": "deepseek_v4_flash_mtp",
    "dspark": "deepseek_v4_flash_dspark",
}
_DEEPSEEK_V4_KERNEL_DIRNAME_SET = frozenset(_DEEPSEEK_V4_KERNEL_DIRNAMES.values())
_DEEPSEEK_V4_IMPORT_MODULES = (
    "config",
    "moe",
    "combine",
    "decode_csa",
    "decode_hca",
    "decode_swa",
    "decode_device_state",
    "decode_fwd",
    "decode_fwd_mtp",
    "decode_input_pack",
    "decode_indexer",
    "decode_indexer_compressor",
    "decode_layer",
    "decode_metadata",
    "decode_mtp",
    "decode_mtp_verify",
    "decode_prepare",
    "lookup_embedding",
    "decode_sparse_attn_csa",
    "decode_sparse_attn_hca",
    "decode_sparse_attn_swa",
    "dspark_attention",
    "dspark_drafter",
    "dspark_layer",
    "dspark_markov",
    "dspark_proj",
    "dispatch",
    "expert_routed",
    "expert_shared",
    "gate",
    "hc_post",
    "hc_pre",
    "lm_head",
    "prefill_csa",
    "prefill_hca",
    "prefill_swa",
    "prefill_indexer_compressor",
    "prefill_layer",
    "prefill_mtp",
    "prefill_fwd",
    "prefill_sparse_attn",
    "qkv_proj_rope",
    "rmsnorm",
    "sample",
    "utils",
)


def _find_pypto_lib_deepseek_v4_dir(
    pypto_lib_root: str | None = None,
    *,
    speculative_method: str | None = None,
) -> Path:
    """Find the DeepSeekV4 kernel directory."""
    try:
        kernel_dirname = _DEEPSEEK_V4_KERNEL_DIRNAMES[speculative_method]
    except KeyError as exc:
        raise ValueError(
            f"unsupported DeepSeekV4 speculative method: {speculative_method!r}"
        ) from exc
    if pypto_lib_root is None:
        pypto_lib_root = os.environ.get("PYPTO_LIB_ROOT")
    if pypto_lib_root:
        root = Path(pypto_lib_root)
        candidate = root / "models" / kernel_dirname
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"DeepSeekV4 kernel directory not found under PYPTO_LIB_ROOT={pypto_lib_root!r}")

    start_dir = Path(__file__).resolve().parent
    for directory in (start_dir, *start_dir.parents):
        pypto_lib_dir = directory / "pypto-lib"
        candidate = pypto_lib_dir / "models" / kernel_dirname
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Cannot locate DeepSeekV4 kernels. Run from a checkout with pypto-lib available "
        "or set PYPTO_LIB_ROOT to a pypto-lib checkout."
    )


def _is_deepseek_v4_module_file(path: Path, kernel_dir: Path) -> bool:
    """Return whether ``path`` belongs to either DeepSeekV4 kernel variant."""
    resolved = path.resolve()
    if resolved.is_relative_to(kernel_dir):
        return True
    return (
        resolved.parent.name in _DEEPSEEK_V4_KERNEL_DIRNAME_SET
        and resolved.parent.parent.name == "models"
    )


@contextlib.contextmanager
def _deepseek_v4_import_context(
    kernel_dir: Path,
    *,
    pypto_lib_root: Path,
    ep: int,
    lm_head_tp: int | None = None,
    moe_shape: str | None = None,
    num_layers: int | None = None,
):
    """Import DeepSeekV4 kernels with fixed EP and LM-head TP arguments."""
    old_argv = list(sys.argv)
    old_path = list(sys.path)
    missing = object()
    old_modules = {
        module_name: sys.modules.get(module_name, missing)
        for module_name in _DEEPSEEK_V4_IMPORT_MODULES
    }
    for module_name in _DEEPSEEK_V4_IMPORT_MODULES:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is not None and _is_deepseek_v4_module_file(Path(module_file), kernel_dir):
            sys.modules.pop(module_name, None)
    sys.argv = ["pypto-serving-deepseek-v4", "--ep", str(int(ep))]
    if lm_head_tp is not None:
        # pypto-lib names this kernel-local LM-head sharding argument ``--tp``;
        # it is independent of pypto-serving's model-level CLI TP setting.
        sys.argv.extend(["--tp", str(int(lm_head_tp))])
    if moe_shape is not None:
        sys.argv.extend(["--moe-shape", moe_shape])
    if num_layers is not None:
        # prefill_fwd freezes its layer-stack span from ``--num-layers`` at import;
        # serving always packs the full 43-layer forward.
        sys.argv.extend(["--num-layers", str(int(num_layers))])
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


class DeepSeekV4PyptoExecutor(CorePyptoExecutor):
    """PyPTO executor boundary for DeepSeekV4 Flash W8A8 serving."""

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
        speculative_method: str | None = None,
    ) -> None:
        worker_device_ids = tuple(device_ids) if device_ids is not None else (int(device_id),)
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_ids=worker_device_ids,
            pypto_build_dir=pypto_build_dir,
            use_compile_cache=use_compile_cache,
        )
        self._compile_kernels = bool(compile_kernels)
        self._num_speculative_tokens = int(num_speculative_tokens)
        if self._num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens must be non-negative")
        if speculative_method not in (None, "mtp", "dspark"):
            raise ValueError(
                f"unsupported DeepSeekV4 speculative method: {speculative_method!r}"
            )
        if speculative_method == "dspark" and self._num_speculative_tokens != 7:
            raise ValueError("DeepSeekV4 DSpark requires exactly seven speculative tokens")
        if speculative_method in (None, "mtp") and self._num_speculative_tokens > 0:
            speculative_method = "mtp"
        self._speculative_method = speculative_method
        self._kernel_dir = _find_pypto_lib_deepseek_v4_dir(
            speculative_method=self._speculative_method
        )
        self._embedding_cache: dict[str, torch.Tensor] = {}
        # Shared JIT-compile core; DeepSeek wraps each compile in a per-kernel
        # profile span (see _compile_l3_callable). With ``use_compile_cache`` the
        # build dir doubles as the on-disk kernel cache (load-or-compile, slotted
        # by kernel name); otherwise pypto uses its default per-kernel build dirs.
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
        """Return the global DP width: one local prefill request per EP rank."""
        return DeepSeekV4CacheLayout().ranks

    @property
    def supports_device_sampling(self) -> bool:
        """Use sampled IDs emitted by the integrated decode sampler."""
        return True

    @property
    def supports_device_stochastic_sampling(self) -> bool:
        """Use temperature/top-k sampling for AR and fused K=1 decode."""
        return self._num_speculative_tokens <= 1

    @property
    def supports_device_decode_embedding(self) -> bool:
        """Use token IDs directly in the packed DeepSeek decode kernels."""
        return True

    @property
    def supports_async_decode_prepare(self) -> bool:
        """Keep arbitrary-depth MTP on its synchronous chunked decode path."""
        return self._num_speculative_tokens <= 1

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup token embeddings from the lazily loaded DeepSeekV4 embedding table."""
        compiled = self._compiled.get(model.config.model_id)
        if not isinstance(compiled, DeepSeekV4CompiledKernels):
            raise RuntimeError(f"DeepSeekV4 model {model.config.model_id!r} is not registered")
        embed_weight = compiled.embedding_weight
        if embed_weight is None:
            embed_weight = self._embedding_cache.get(model.config.model_id)
        if embed_weight is None:
            embed_weight = compiled.weight_store.load_tensor("embed.weight").contiguous()
            if embed_weight.ndim != 2:
                raise ValueError(f"embed.weight must be rank-2, got shape={tuple(embed_weight.shape)}")
            if int(embed_weight.shape[0]) != model.config.vocab_size:
                raise ValueError(
                    f"embed.weight vocab size must be {model.config.vocab_size}, "
                    f"got {int(embed_weight.shape[0])}"
                )
            if int(embed_weight.shape[1]) != model.config.hidden_size:
                raise ValueError(
                    f"embed.weight hidden size must be {model.config.hidden_size}, "
                    f"got {int(embed_weight.shape[1])}"
                )
        compiled.embedding_weight = embed_weight
        self._embedding_cache[model.config.model_id] = embed_weight

        flat_ids = token_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        embeddings = embed_weight.index_select(0, flat_ids)
        return embeddings.reshape(*token_ids.shape, model.config.hidden_size).to(device=token_ids.device)

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Release runner-local DeepSeekV4 cache ownership metadata."""
        for runner in self._runners.values():
            release = getattr(runner, "release_finished_requests", None)
            if callable(release):
                release(request_ids)

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the DeepSeekV4 runtime runner."""
        if not isinstance(compiled, DeepSeekV4CompiledKernels):
            raise TypeError("DeepSeekV4PyptoExecutor requires DeepSeekV4 compiled metadata.")
        return DeepSeekV4ModelRunner(compiled=compiled)

    def _compile_model(self, model: RuntimeModel) -> DeepSeekV4CompiledKernels:
        """Validate DeepSeekV4 W8A8 metadata and return runner artifacts.

        The current pypto-lib DeepSeekV4 programs are single-layer kernels. This
        method intentionally validates and packages the serving contract without
        pretending those kernels are already a full-model generator.
        """
        metadata = model.extra
        if metadata.get("family") != "deepseek_v4":
            raise ValueError("DeepSeekV4PyptoExecutor received a non-DeepSeekV4 model")
        if metadata.get("checkpoint_format") != "w8a8-compressed-tensors":
            raise ValueError("DeepSeekV4PyptoExecutor requires the W8A8 compressed-tensors checkpoint")

        if model.runtime.num_speculative_tokens != self._num_speculative_tokens:
            raise ValueError(
                "DeepSeekV4 executor/runtime MTP depth mismatch: "
                f"executor={self._num_speculative_tokens}, "
                f"runtime={model.runtime.num_speculative_tokens}"
            )
        # Autoregressive decode keeps the established eight-token tile. MTP uses
        # a 16-token specialization and the smallest power-of-two request-local
        # sequence that can cover one target-verification chunk.
        layout = (
            deepseek_v4_dspark_decode_layout()
            if self._speculative_method == "dspark"
            else deepseek_v4_decode_layout(self._num_speculative_tokens)
        )
        layout.validate_runtime(model.config, model.runtime, self._device_ids)
        compress_ratios = tuple(int(ratio) for ratio in metadata["compress_ratios"])
        if len(compress_ratios) != model.config.num_hidden_layers + 1:
            raise ValueError("DeepSeekV4 compress_ratios must include hidden layers plus MTP/final entry")
        config_data = metadata.get("config_data", {})
        n_routed_experts = int(config_data.get("n_routed_experts", 256)) if isinstance(config_data, dict) else 256
        num_hash_layers = int(config_data.get("num_hash_layers", 3)) if isinstance(config_data, dict) else 3
        layer_plan = build_deepseek_v4_layer_plan(
            compress_ratios=compress_ratios,
            num_hidden_layers=model.config.num_hidden_layers,
            num_hash_layers=num_hash_layers,
        )
        weight_map = dict(metadata["weight_map"])
        weight_store = DeepSeekV4WeightStore(model_dir=str(metadata["model_dir"]), weight_map=weight_map)
        weight_store.validate_startup_contract(
            num_hidden_layers=model.config.num_hidden_layers,
            n_routed_experts=n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=num_hash_layers,
        )
        if self._speculative_method == "dspark":
            weight_store.validate_dspark_startup_contract(
                n_routed_experts=n_routed_experts,
            )
        elif self._num_speculative_tokens:
            weight_store.validate_mtp_startup_contract(n_routed_experts=n_routed_experts)

        layer_compress_ratios = tuple(layer.compress_ratio for layer in layer_plan)
        prepacked_layer_weights: DeepSeekV4StackedLayerWeights | None = None
        if self._compile_kernels and self._speculative_method != "dspark":
            prepacked_layer_weights = weight_store.load_prepacked_stacked_layer_weights(
                ranks=layout.ranks,
                n_routed_experts=n_routed_experts,
                compress_ratios=layer_compress_ratios,
                num_hash_layers=num_hash_layers,
            )

        prefill = None
        decode = None
        mtp_decode = None
        mtp_prefill = None
        dspark_drafter = None
        dspark_markov = None
        freqs_cos = freqs_sin = None
        if self._compile_kernels:
            modules = self._load_kernel_modules(layout)
            prefill = self._compile_l3_callable(
                "deepseek_v4_prefill",
                modules["prefill_fwd"].l3_prefill_fwd,
                layout=layout,
            )
            use_fused_mtp = (
                self._speculative_method != "dspark"
                and self._num_speculative_tokens == 1
            )
            decode = self._compile_l3_callable(
                "deepseek_v4_decode_mtp_fused" if use_fused_mtp else "deepseek_v4_decode",
                (
                    modules["decode_fwd_mtp"].l3_decode_fwd_mtp
                    if use_fused_mtp
                    else modules["decode_fwd"].l3_decode_fwd
                ),
                layout=layout,
                runtime_scalar_names=(
                    frozenset({"mtp_num_tokens"}) if use_fused_mtp else None
                ),
            )
            if self._speculative_method == "dspark":
                dspark_drafter = self._compile_l3_callable(
                    "deepseek_v4_dspark_drafter",
                    modules["dspark_drafter"].l3_dspark_drafter,
                    layout=layout,
                )
                dspark_markov = self._compile_l3_callable(
                    "deepseek_v4_dspark_markov",
                    modules["dspark_markov"].l3_distributed_markov_sample,
                    layout=layout,
                )
            elif self._num_speculative_tokens:
                mtp_prefill = self._compile_l3_callable(
                    "deepseek_v4_mtp_prefill",
                    modules["prefill_mtp"].l3_mtp_prefill_fwd,
                    layout=layout,
                    runtime_scalar_names=frozenset({"num_tokens"}),
                )
                if not use_fused_mtp:
                    mtp_decode = self._compile_l3_callable(
                        "deepseek_v4_mtp_decode",
                        modules["decode_mtp"].l3_decode_mtp,
                        layout=layout,
                        runtime_scalar_names=frozenset({"num_tokens"}),
                    )
            freqs_cos, freqs_sin = self._build_rope_tables(
                modules["utils"],
                modules["config"],
            )

        return DeepSeekV4CompiledKernels(
            layout=layout,
            model_dir=str(metadata["model_dir"]),
            weight_map=weight_map,
            weight_store=weight_store,
            prepacked_layer_weights=prepacked_layer_weights,
            compress_ratios=compress_ratios,
            layer_plan=layer_plan,
            kernel_dir=str(self._kernel_dir),
            runtime_model=model,
            prefill=prefill,
            decode=decode,
            mtp_prefill=mtp_prefill,
            mtp_decode=mtp_decode,
            dspark_drafter=dspark_drafter,
            dspark_markov=dspark_markov,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            platform=self._platform,
            device_id=self._device_ids[0],
            device_ids=self._device_ids,
            n_routed_experts=n_routed_experts,
            num_hash_layers=num_hash_layers,
            num_speculative_tokens=self._num_speculative_tokens,
            speculative_method=self._speculative_method,
        )

    def _load_kernel_modules(self, layout: DeepSeekV4CacheLayout) -> dict[str, object]:
        """Import DeepSeekV4 pypto-lib modules with EP fixed to the serving world size."""
        pypto_lib_root = self._kernel_dir.parents[1]
        ranks = layout.ranks
        fwd_layers = DEEPSEEK_V4_FWD_NUM_LAYERS
        with _deepseek_v4_import_context(
            self._kernel_dir,
            pypto_lib_root=pypto_lib_root,
            ep=ranks,
            lm_head_tp=DEEPSEEK_V4_LM_HEAD_TP_SIZE,
            moe_shape="prefill",
            num_layers=fwd_layers,
        ):
            prefill_layer = importlib.import_module("prefill_layer")
            prefill_fwd = importlib.import_module("prefill_fwd")
            prefill_mtp = (
                None
                if self._speculative_method == "dspark"
                else importlib.import_module("prefill_mtp")
            )
        with _deepseek_v4_import_context(
            self._kernel_dir,
            pypto_lib_root=pypto_lib_root,
            ep=ranks,
            lm_head_tp=DEEPSEEK_V4_LM_HEAD_TP_SIZE,
            moe_shape="decode",
        ):
            config = importlib.import_module("config")
            # pypto-lib freezes B/S into module-level shapes at import. Override
            # the deployment preset before importing any decode program while
            # specializing T with the selected layout while preserving physical
            # cache capacities.
            config.DECODE_BATCH = (
                layout.decode_batch * DEEPSEEK_V4_DSPARK_TP_SIZE
                if self._speculative_method == "dspark"
                else layout.decode_batch
            )
            config.DECODE_SEQ = layout.decode_seq
            config.DECODE_TOKENS = (
                config.DECODE_BATCH * config.DECODE_SEQ
                if self._speculative_method == "dspark"
                else layout.decode_tokens
            )
            config.MOE_TOKENS = layout.decode_tokens
            config.DECODE_RECV_MAX = (
                config.DP * config.DECODE_TOKENS
                if self._speculative_method == "dspark"
                else ranks * layout.decode_tokens
            )
            config.RECV_MAX = config.DECODE_RECV_MAX
            modules = {"config": config}
            decode_module_names = ["decode_layer", "decode_fwd", "lm_head", "utils"]
            if self._speculative_method == "dspark":
                decode_module_names.extend(("dspark_drafter", "dspark_markov"))
            elif self._num_speculative_tokens:
                decode_module_names.append("decode_mtp")
            if self._num_speculative_tokens == 1:
                decode_module_names.append("decode_fwd_mtp")
            modules.update(
                {
                    name: importlib.import_module(name)
                    for name in decode_module_names
                }
            )
        modules["prefill_layer"] = prefill_layer
        modules["prefill_fwd"] = prefill_fwd
        if prefill_mtp is not None:
            modules["prefill_mtp"] = prefill_mtp
        return modules

    def _compile_l3_callable(
        self,
        name: str,
        jit_fn: object,
        *,
        layout: DeepSeekV4CacheLayout,
        runtime_scalar_names: frozenset[str] | None = None,
    ) -> DeepSeekV4L3Callable:
        """Compile one fully annotated DeepSeekV4 HOST wrapper.

        The JIT compile + type-check + (optional) cache load-or-compile are
        delegated to the shared :class:`KernelCompiler`; ``runtime_scalar_names``
        feeds the startup-optimisation compile signature. ``layout`` is retained
        on the signature for the call sites but is not needed now that the
        layout-aware fingerprint is gone.
        """
        from pypto.language import RUNTIME  # noqa: PLC0415

        # Runtime scalars (mtp_num_tokens / num_tokens) vary per step, so leave
        # them unspecialized: RUNTIME keeps them out of the compiled artifact and
        # the cache key, like a pl.dynamic extent. Tensor params are read from
        # their annotations, so no sample tensors are required.
        runtime_scalars = (
            {name: RUNTIME for name in runtime_scalar_names}
            if runtime_scalar_names
            else {}
        )
        with profile_span(f"DeepSeekV4PyptoExecutor.compile.{name}", cat="executor"):
            return self._compiler.compile(
                name, jit_fn, use_cache=self._use_compile_cache, **runtime_scalars
            )

    def _build_rope_tables(
        self,
        utils_module: object,
        config_module: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build packed SWA and compressed DeepSeek-V4 RoPE profiles.

        The profile is a property of each attention layer: SWA layers (including
        the MTP draft layer) use ratio zero, while CSA/HCA layers use the YaRN
        compressed profile shared by ratios four and 128.
        """
        swa_cos, swa_sin = utils_module.build_rope_tables(
            config_module.FLASH, 0, dtype=torch.bfloat16
        )
        compressed_cos, compressed_sin = utils_module.build_rope_tables(
            config_module.FLASH, 4, dtype=torch.bfloat16
        )
        return (
            torch.stack((swa_cos, compressed_cos), dim=0).contiguous().cpu(),
            torch.stack((swa_sin, compressed_sin), dim=0).contiguous().cpu(),
        )
