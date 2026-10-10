# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Executor routing for V4.1; devices are opened only after contract validation."""
from pypto_serving.model.common.executor.executor import ModelExecutor

from .composite import BuildOptions, MissingCompositeInterface, load_composite_bindings
from .execution_plan import RankPlacement, V41ExecutionPlan
from .npu_runner import V41ModelRunner


class DeepSeekV41PyptoExecutor(ModelExecutor):
    """Shared worker interface with a synchronous, collective V4.1 runner."""
    def __init__(self, kv_cache_manager=None, *, platform="a5", device_ids=tuple(range(8)),
                 pypto_build_dir="build_output", use_compile_cache=False, bindings=None):
        super().__init__(kv_cache_manager)
        if platform != "a5":
            raise ValueError("V4.1 M0 currently requires the A5 platform")
        self.device_ids = tuple(device_ids)
        self.bindings = bindings
        self.runners = {}
        self._failed_runners = []
        self.platform = platform
        self.pypto_build_dir = pypto_build_dir
        self.use_compile_cache = use_compile_cache

    def register_model(self, model_id, record):
        if model_id in self.runners:
            raise ValueError("model already registered")
        bindings = self.bindings or load_composite_bindings()
        if not bindings.cache_groups:
            raise MissingCompositeInterface("V4.1 requires explicit grouped cache layouts and capacities")
        if record.runtime.kv_cache_groups != bindings.cache_groups:
            raise ValueError("scheduler and composite cache group contracts differ")
        plan = V41ExecutionPlan(record.runtime_model.extra["model_dir"], RankPlacement(0))
        runner = V41ModelRunner(
            plan, bindings, device_ids=self.device_ids, runtime=record.runtime,
            build_options=BuildOptions(self.platform, self.pypto_build_dir, self.use_compile_cache),
        )
        try:
            pages = runner.preflight()
        except Exception:
            if not runner.closed:
                self._failed_runners.append(runner)
            raise
        self.runners[model_id] = runner
        return pages

    def lookup_embeddings(self, model, token_ids):
        return self.runners[model.config.model_id].lookup_embeddings(token_ids)

    def run_prefill(self, model, batch):
        return self.runners[model.config.model_id].run_prefill(model, batch)

    def run_decode(self, model, batch):
        return self.runners[model.config.model_id].run_decode(model, batch)

    def release_finished_requests(self, request_ids):
        for runner in self.runners.values():
            runner.release_finished_requests(request_ids)

    def close(self):
        for runner in (*self.runners.values(), *self._failed_runners):
            runner.close()
        self.runners.clear()
        self._failed_runners.clear()
