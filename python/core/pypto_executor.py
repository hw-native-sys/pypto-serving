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
from abc import ABC, abstractmethod

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


class PyptoExecutor(ModelExecutor, ABC):
    """Base executor for PyPTO backends that compile once and delegate runtime."""

    def __init__(
        self,
        kv_cache_manager: KvCacheManager | None = None,
        *,
        platform: str = "a2a3sim",
        device_id: int = 0,
        save_kernels_dir: str | None = None,
    ) -> None:
        """Initialize common PyPTO runtime options and model registries."""
        super().__init__(kv_cache_manager)
        self._platform = platform
        self._device_id = device_id
        self._save_kernels_dir = save_kernels_dir
        self._runners: dict[str, ModelRunner] = {}
        self._compiled: dict[str, object] = {}

    def register_model(self, model_id: str, record: ModelRecord) -> None:
        """Compile kernels for ``record`` and attach a runner to ``model_id``."""
        with profile_span("PyptoExecutor.register_model", cat="executor", args={"model_id": model_id}):
            compiled = self._compile_model(record.runtime_model)
            self._compiled[model_id] = compiled
            runner = self._create_runner(model_id, compiled)
            runner.init_kv_cache(model_id, record.config, record.runtime)
            self._runners[model_id] = runner

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

    def materialize_kv_page_view(self, model_id: str):
        """Return a worker-side canonical KV page view for offload backends."""
        runner = self._runners[model_id]
        materialize = getattr(runner, "materialize_kv_page_view", None)
        if not callable(materialize):
            raise RuntimeError(f"Runner for model {model_id!r} does not expose a KV page view")
        return materialize(model_id)

    @contextlib.contextmanager
    def session(self):
        """Provide a generation lifecycle hook for PyPTO runtimes."""
        yield

    def _run_config(self, *, codegen_only: bool):
        """Build a PyPTO ``RunConfig`` for compile or execution calls."""
        from pypto.runtime import RunConfig

        return RunConfig(
            platform=self._platform,
            device_id=self._device_id,
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
