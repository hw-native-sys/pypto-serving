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
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from .executor import ModelExecutor
from .kv_cache import KvCacheManager
from .model_runner import ModelRunner
from python.profile import profile_span
from .types import (
    DecodeBatch,
    DecodeResult,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeModel,
)
from .utils import backend_type_for_platform


logger = logging.getLogger(__name__)


class PyptoExecutor(ModelExecutor, ABC):
    """Base executor for PyPTO backends that compile once and delegate runtime."""

    def __init__(
        self,
        kv_cache_manager: KvCacheManager | None = None,
        *,
        platform: str = "a2a3sim",
        device_ids: Sequence[int] = (0,),
        save_kernels_dir: str | None = None,
    ) -> None:
        """Initialize common PyPTO runtime options and model registries."""
        super().__init__(kv_cache_manager)
        self._platform = platform
        self._device_ids = tuple(int(device) for device in device_ids)
        if not self._device_ids:
            raise ValueError("device_ids must contain at least one device id")
        self._save_kernels_dir = save_kernels_dir
        self._runners: dict[str, ModelRunner] = {}
        self._compiled: dict[str, object] = {}

    def register_model(self, model_id: str, record: ModelRecord) -> int:
        """Compile kernels for ``record`` and attach a runner to ``model_id``.

        Returns the number of KV cache pages allocated on the device so the
        caller can synchronise host-side block metadata.
        """
        print("[register_model] compiling kernels …", flush=True)
        with profile_span("PyptoExecutor.register_model", cat="executor", args={"model_id": model_id}):
            compiled = self._compile_model(record.runtime_model)
            runner = self._create_runner(model_id, compiled)
            try:
                num_pages = runner.init_kv_cache(model_id, record.config, record.runtime)
            except Exception:
                close = getattr(runner, "close", None)
                if callable(close):
                    close()
                raise
            self._compiled[model_id] = compiled
            self._runners[model_id] = runner
        return num_pages

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Delegate prefill execution to the registered model runner."""
        with profile_span(
            "PyptoExecutor.run_prefill",
            cat="executor",
            args={"model_id": model.config.model_id, "batch_size": len(batch.request_ids)},
        ):
            return self._runners[model.config.model_id].run_prefill(model, batch)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Delegate decode execution to the registered model runner."""
        with profile_span(
            "PyptoExecutor.run_decode",
            cat="executor",
            args={"model_id": model.config.model_id, "batch_size": len(batch.request_ids)},
        ):
            return self._runners[model.config.model_id].run_decode(model, batch)

    @contextlib.contextmanager
    def session(self):
        """Provide a generation lifecycle hook for PyPTO runtimes."""
        yield

    def close(self) -> None:
        """Release runtime resources held by registered model runners."""
        for model_id, runner in self._runners.items():
            close = getattr(runner, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close PyPTO runner for model %s", model_id)

    def _run_config(self, *, codegen_only: bool):
        """Build a PyPTO ``RunConfig`` for compile or execution calls."""
        from pypto.runtime import RunConfig

        return RunConfig(
            platform=self._platform,
            device_id=self._device_ids[0],
            backend_type=backend_type_for_platform(self._platform),
            codegen_only=codegen_only,
            save_kernels=self._save_kernels_dir is not None,
            save_kernels_dir=self._save_kernels_dir,
        )

    @abstractmethod
    def _compile_model(self, model: RuntimeModel) -> object:
        """Compile model-specific PyPTO kernels and return runtime artifacts."""
        raise NotImplementedError

    @abstractmethod
    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create a model-specific runner from compiled artifacts."""
        raise NotImplementedError
