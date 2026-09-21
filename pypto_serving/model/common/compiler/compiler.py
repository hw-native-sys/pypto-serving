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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .l3_callable import L3Callable

if TYPE_CHECKING:
    from pypto.runtime import RunConfig

logger = logging.getLogger(__name__)


class KernelCompiler:
    """Compile HOST kernels through PyPTO's validated persistent JIT cache.

    PyPTO owns specialization keys, source/toolchain identity, immutable
    publication and restoration. Ordinary DistributedWorker preparation builds
    and publishes READY binaries; serving never reloads a directory by name.

    ``use_cache=None`` inherits the RunConfig/process/environment policy. An
    explicit bool overrides enablement. ``cache_dir`` is an optional artifact
    root, not an output directory; diagnostic/output controls retain PyPTO's
    bypass rules.
    """

    def __init__(
        self,
        *,
        run_config: "RunConfig",
        cache_dir: str | os.PathLike[str] | None = None,
        **extra_configs: Any,
    ) -> None:
        self._run_config = run_config
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._extra_configs = extra_configs

    def compile(
        self,
        name: str,
        jit_fn: object,
        *,
        use_cache: bool | None = None,
        **compile_kwargs: Any,
    ) -> L3Callable:
        """Compile in signature mode and retain runtime scalar keyword arguments.

        Explicit enablement overrides only ``enabled`` on the effective policy,
        resolved the way PyPTO does: the supplied RunConfig policy, then process
        defaults from ``pypto.configure_cache()``, then the artifact
        root/read-only environment settings. Without an explicit override,
        PyPTO resolves its public configuration untouched.
        """
        from pypto._cache_config import capture_cache_config  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        if use_cache is not None and type(use_cache) is not bool:
            raise TypeError("use_cache must be bool or None")
        configs = {**self._extra_configs, "codegen_only": True}
        if self._run_config.save_kernels_dir is not None and "save_kernels_dir" not in configs:
            configs["save_kernels_dir"] = str(Path(self._run_config.save_kernels_dir) / name)
        policy = configs.get("cache_config", self._run_config.cache_config)
        if use_cache is not None:
            if policy is None:
                # Preserve process/env policy fields (root, read-only, extra
                # source paths, fingerprints) -- only enablement is overridden.
                # The fingerprints take part in the artifact key; dropping them
                # could validate a stale artifact as a hit.
                policy = capture_cache_config(None)
                if not use_cache:
                    policy = dataclasses.replace(policy, readonly=False)
            policy = dataclasses.replace(policy, enabled=use_cache)
        if self._cache_dir is not None:
            if policy is None:
                raise ValueError("cache_dir requires explicit use_cache or RunConfig.cache_config")
            policy = dataclasses.replace(policy, root=self._cache_dir)
        if policy is not None:
            configs["cache_config"] = policy

        # Preserve every compiler/diagnostic field, including future RunConfig
        # additions. In particular, an explicit save_kernels_dir still bypasses
        # reuse; serving's normal build directory is configured separately.
        run_config = dataclasses.replace(self._run_config, **configs)
        logger.info("[kernel-compile] resolving %s through PyPTO JIT", name)
        compiled = jit_fn.compile(config=run_config, **compile_kwargs)
        if not isinstance(compiled, DistributedCompiledProgram):
            raise TypeError(
                f"{name} did not compile to DistributedCompiledProgram; "
                f"got {type(compiled).__name__}"
            )
        return L3Callable(
            compiled=compiled,
            name=name,
            aicpu_thread_num=run_config.distributed_config.aicpu_thread_num,
        )
