# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared kernel-compilation core for PyPTO model executors."""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .l3_callable import L3Callable

if TYPE_CHECKING:
    from pypto.runtime import RunConfig

logger = logging.getLogger(__name__)

#: Emitted by the pypto compiler; its presence tells us a cache slot holds a
#: complete assembled program safe to reload.
_META_FILE = "distributed_meta.json"


class KernelCompiler:
    """Compile PyPTO kernel modules into L3-callable distributed programs.

    De-duplicates the JIT-compile core shared by the qwen and DeepSeek
    executors: derive the compile ``RunConfig`` from the executor's base config
    (built by :func:`build_pypto_run_config`, always codegen-only), run
    ``jit_fn.compile``, type-check the result, and wrap it in an :class:`L3Callable`.
    Any extra ``RunConfig`` field overrides a model needs are accepted as
    ``extra_configs`` and applied via ``dataclasses.replace``. Per-kernel
    profiling, if wanted, is the caller's concern -- wrap the :meth:`compile`
    call in a ``profile_span``.

    When a ``cache_dir`` is supplied, each kernel compiles straight into
    ``cache_dir/<name>`` (set as the per-kernel ``save_kernels_dir``, so pypto
    writes its IR there and the L3 worker later assembles the device binaries
    into the same dir). A later launch finds that slot populated and reloads it
    via ``DistributedCompiledProgram.from_dir``, skipping the JIT and the
    device-binary assembly. There is no separate copy step and no fingerprinting:
    a slot is keyed only by kernel name, so the caller is responsible for
    pointing ``cache_dir`` at a directory appropriate for the current config and
    kernel sources -- a stale slot would otherwise be reused. Caching is
    best-effort and never fatal.
    """

    def __init__(
        self,
        *,
        run_config: "RunConfig",
        cache_dir: str | os.PathLike[str] | None = None,
        **extra_configs: Any,
    ) -> None:
        """Store the base RunConfig, the optional cache dir, and extra overrides."""
        self._run_config = run_config
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._extra_configs = extra_configs

    def compile(
        self,
        name: str,
        jit_fn: object,
        *compile_args: object,
        use_cache: bool = False,
        run_config_overrides: Mapping[str, Any] | None = None,
        **compile_kwargs: Any,
    ) -> L3Callable:
        """Compile a HOST wrapper into a PyPTO ``DistributedCompiledProgram``.

        With a ``cache_dir``, a populated ``cache_dir/<name>`` slot is reloaded
        (skipping the JIT); otherwise the kernel compiles straight into that slot.

        Positional ``compile_args`` support generic HOST wrappers whose shapes
        and dtypes come from Contract-provided sample tensors. Annotation-driven
        wrappers may omit them. ``run_config_overrides`` applies stage-local
        compiler policy, while ``compile_kwargs`` are forwarded to
        ``jit_fn.compile`` (e.g. ``name=pl.RUNTIME`` for runtime scalars).
        """
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        aicpu_thread_num = self._run_config.distributed_config.aicpu_thread_num
        configs = {**self._extra_configs, "codegen_only": True}
        if run_config_overrides is not None:
            configs.update(run_config_overrides)
        if self._cache_dir is not None:
            slot = self._cache_dir / name
            cached = self._load_cached(slot, self._run_config) if use_cache else None
            if cached is not None:
                logger.info("[kernel-cache] HIT: reused %s from %s (skipped JIT)", name, slot)
                return L3Callable(compiled=cached, name=name, aicpu_thread_num=aicpu_thread_num)
            if use_cache:
                logger.info("[kernel-cache] MISS: compiling %s into %s", name, slot)
            else:
                logger.info(
                    "[kernel-cache] disabled (use_cache=False): recompiling %s, ignoring slot %s",
                    name, slot,
                )
            # pypto uses ``save_kernels_dir`` verbatim (no per-program suffix),
            # so slot by kernel name ourselves to keep each program's build dir
            # distinct and stable across launches.
            configs["save_kernels"] = True
            configs["save_kernels_dir"] = str(slot)
        else:
            logger.info("[kernel-compile] compiling %s (no cache_dir)", name)

        # ``dataclasses.replace`` keeps every base-config field (so the
        # ``DistributedConfig`` and any new pypto RunConfig fields are forwarded
        # automatically) and re-runs ``__post_init__``.
        run_config = dataclasses.replace(self._run_config, **configs)
        compiled = jit_fn.compile(*compile_args, config=run_config, **compile_kwargs)
        if not isinstance(compiled, DistributedCompiledProgram):
            raise TypeError(
                f"{name} did not compile to DistributedCompiledProgram; got {type(compiled).__name__}"
            )
        return L3Callable(compiled=compiled, name=name, aicpu_thread_num=aicpu_thread_num)

    @staticmethod
    def _load_cached(slot: Path, run_config: "RunConfig") -> object | None:
        """Reload a compiled program from ``slot``, or ``None`` on miss/error."""
        if not (slot / _META_FILE).exists():
            return None
        try:
            from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

            return DistributedCompiledProgram.from_dir(
                str(slot),
                platform=run_config.platform,
                distributed_config=run_config.distributed_config,
            )
        except Exception as exc:  # noqa: BLE001 - reuse must never be fatal
            logger.warning(
                "[kernel-cache] reload of %s failed (%s: %s); recompiling",
                slot, type(exc).__name__, exc,
            )
            return None
