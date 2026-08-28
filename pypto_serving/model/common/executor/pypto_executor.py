# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from .executor import ModelExecutor
from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeModel,
    SamplingParams,
)
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.tools.profile import profile_span


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PendingRunnerDecode:
    """Route an async reclaim ticket back to the runner that dispatched it."""

    model_id: str
    pending: object


class PyptoExecutor(ModelExecutor, ABC):
    """Base executor for PyPTO backends that compile once and delegate runtime."""

    def __init__(
        self,
        kv_cache_manager: KvCacheManager | None = None,
        *,
        platform: str = "a2a3sim",
        device_ids: Sequence[int] = (0,),
        pypto_build_dir: str = "build_output",
        use_compile_cache: bool = False,
    ) -> None:
        """Initialize common PyPTO runtime options and model registries."""
        super().__init__(kv_cache_manager)
        self._platform = platform
        self._device_ids = tuple(int(device) for device in device_ids)
        if not self._device_ids:
            raise ValueError("device_ids must contain at least one device id")
        self._pypto_build_dir = pypto_build_dir
        self._use_compile_cache = use_compile_cache
        self._runners: dict[str, ModelRunner] = {}
        self._compiled: dict[str, object] = {}

    def register_model(self, model_id: str, record: ModelRecord) -> int:
        """Compile a model, attach its runner, and return scheduler page capacity."""
        import time

        with profile_span("PyptoExecutor.register_model", cat="executor", args={"model_id": model_id}):
            start_t0 = time.perf_counter()
            compiled = self._compile_model(record.runtime_model)
            runner = self._create_runner(model_id, compiled)

            try:
                num_pages = runner.init_kv_cache(model_id, record.config, record.runtime)
            except Exception:
                close = getattr(runner, "close", None)
                if callable(close):
                    close()
                raise

            with profile_span("PyptoExecutor.preflight", cat="executor", args={"model_id": model_id}):
                runner.preflight(record)
            logger.info(
                "PyptoExecutor %s: model loaded (%.1fs total)",
                model_id,
                time.perf_counter() - start_t0,
            )
            self._runners[model_id] = runner
            self._compiled[model_id] = compiled
        return num_pages

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Delegate prefill execution to the registered model runner."""
        with profile_span(
            "PyptoExecutor.run_prefill",
            cat="executor",
            args={"model_id": model.config.model_id, "batch_size": len(batch.request_ids)},
        ):
            return self._runners[model.config.model_id].run_prefill(model, batch)

    def finalize_prefill(
        self,
        model: RuntimeModel,
        request_ids: list[str],
        sampled_token_ids: list[int],
        sampling_params: list[SamplingParams] | None = None,
    ) -> None:
        """Let a runner seed decode state while still in the prefill stage."""
        runner = self._runners[model.config.model_id]
        finalize = getattr(runner, "finalize_prefill", None)
        if callable(finalize):
            with profile_span(
                "PyptoExecutor.finalize_prefill",
                cat="executor",
                args={"model_id": model.config.model_id, "batch_size": len(request_ids)},
            ):
                finalize(request_ids, sampled_token_ids, sampling_params)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Delegate decode execution to the registered model runner."""
        with profile_span(
            "PyptoExecutor.run_decode",
            cat="executor",
            args={"model_id": model.config.model_id, "batch_size": len(batch.request_ids)},
        ):
            return self._runners[model.config.model_id].run_decode(model, batch)

    @property
    def supports_async_decode_prepare(self) -> bool:
        """Return whether every registered runner exposes split decode execution."""
        return bool(self._runners) and all(
            callable(getattr(runner, "prepare_decode", None))
            and callable(getattr(runner, "run_prepared_decode", None))
            for runner in self._runners.values()
        )

    def prepare_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        *,
        buffer_slot: int,
    ) -> object:
        """Build runner-owned metadata for a future decode dispatch."""
        runner = self._runners[model.config.model_id]
        prepare = getattr(runner, "prepare_decode", None)
        if not callable(prepare):
            return super().prepare_decode(model, batch, buffer_slot=buffer_slot)
        with profile_span(
            "PyptoExecutor.prepare_decode",
            cat="executor",
            args={"model_id": model.config.model_id, "batch_size": len(batch.request_ids)},
        ):
            return prepare(model, batch, buffer_slot=buffer_slot)

    def run_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> DecodeResult:
        """Execute one previously prepared runner snapshot."""
        runner = self._runners[model.config.model_id]
        execute = getattr(runner, "run_prepared_decode", None)
        if not callable(execute):
            return super().run_prepared_decode(model, batch, prepared)
        with profile_span(
            "PyptoExecutor.run_prepared_decode",
            cat="executor",
            args={"model_id": model.config.model_id, "batch_size": len(batch.request_ids)},
        ):
            return execute(model, batch, prepared)

    @property
    def supports_async_decode_reclaim(self) -> bool:
        """Return whether every runner exposes independent dispatch/reclaim."""
        return bool(self._runners) and all(
            bool(getattr(runner, "supports_async_decode_reclaim", False))
            and callable(getattr(runner, "dispatch_prepared_decode", None))
            and callable(getattr(runner, "reclaim_prepared_decode", None))
            for runner in self._runners.values()
        )

    def dispatch_prepared_decode(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        prepared: object,
    ) -> object:
        """Submit only the device phase of a prepared decode."""
        runner = self._runners[model.config.model_id]
        dispatch = getattr(runner, "dispatch_prepared_decode", None)
        if not callable(dispatch):
            return super().dispatch_prepared_decode(model, batch, prepared)
        with profile_span(
            "PyptoExecutor.dispatch_prepared_decode",
            cat="executor",
            args={"model_id": model.config.model_id, "batch_size": len(batch.request_ids)},
        ):
            return _PendingRunnerDecode(
                model_id=model.config.model_id,
                pending=dispatch(model, batch, prepared),
            )

    def reclaim_prepared_decode(self, pending: object) -> DecodeResult:
        """Run host output processing for a completed decode dispatch."""
        if not isinstance(pending, _PendingRunnerDecode):
            return super().reclaim_prepared_decode(pending)
        runner = self._runners[pending.model_id]
        reclaim = getattr(runner, "reclaim_prepared_decode", None)
        if not callable(reclaim):
            return super().reclaim_prepared_decode(pending)
        with profile_span(
            "PyptoExecutor.reclaim_prepared_decode",
            cat="executor",
            args={"model_id": pending.model_id},
        ):
            return reclaim(pending.pending)

    def prepared_decode_requires_token(self, prepared: object) -> bool:
        """Delegate cold-token dependency detection to the active runner."""
        for runner in self._runners.values():
            requires = getattr(runner, "prepared_decode_requires_token", None)
            if callable(requires):
                return bool(requires(prepared))
        return super().prepared_decode_requires_token(prepared)

    def close(self) -> None:
        """Release runtime resources held by registered model runners."""
        for model_id, runner in self._runners.items():
            close = getattr(runner, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close PyPTO runner for model %s", model_id)

    @abstractmethod
    def _compile_model(self, model: RuntimeModel) -> object:
        """Compile model-specific PyPTO kernels and return runtime artifacts."""
        raise NotImplementedError

    @abstractmethod
    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create a model-specific runner from compiled artifacts."""
        raise NotImplementedError
