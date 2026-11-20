# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving executor for the TP8 batch-one, target-only lowlat variant."""

from dataclasses import replace

from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor
from pypto_serving.model.common.executor.utils import build_pypto_run_config

from .contract import LowlatServingContract, validate_sampling
from .kernel_loader import compile_programs


class DeepSeekV4LowlatPyptoExecutor(PyptoExecutor):
    def __init__(
        self,
        kv_cache_manager=None,
        *,
        platform="a2a3",
        device_id=0,
        device_ids=None,
        pypto_build_dir=None,
        target_tp_size=8,
        lm_head_tp_size=8,
        num_speculative_tokens=5,
        compile_kernels=True,
        cache_initialization="on-demand",
        prefix_cache_namespace="",
    ):
        devices = tuple(device_ids) if device_ids is not None else (device_id,)
        if (
            len(devices) != 8
            or len(set(devices)) != 8
            or (target_tp_size, lm_head_tp_size) != (8, 8)
            or type(num_speculative_tokens) is not int or num_speculative_tokens != 5
        ):
            raise ValueError("lowlat requires TP8, head TP8, eight distinct devices and K=5")
        if not compile_kernels:
            raise ValueError("lowlat requires actual compiled kernels")
        super().__init__(
            kv_cache_manager, platform=platform, device_ids=devices, pypto_build_dir=pypto_build_dir
        )
        self.contract = LowlatServingContract(num_speculative_tokens=5)
        if cache_initialization not in ("on-demand", "request"):
            raise ValueError("unknown lowlat cache initialization policy")
        self.cache_initialization = cache_initialization
        self.prefix_cache_namespace=prefix_cache_namespace

    def _compile_model(self, model):
        self.contract = replace(self.contract, compiled_context=model.runtime.max_seq_len)
        self.contract.validate_checkpoint(model.extra["config_data"])
        if (model.runtime.num_speculative_tokens != 5 or model.runtime.max_batch_size != 1):
            raise ValueError("lowlat requires B1 and K5")
        config = build_pypto_run_config(
            platform=self._platform, device_ids=self._device_ids, pypto_build_dir=self._pypto_build_dir
        )
        programs = compile_programs(model, config)
        programs.prefix_namespace=self.prefix_cache_namespace
        from .weight_loader import LowlatWeightStore

        LowlatWeightStore(model, programs).validate_sources()
        from .draft_weights import NativeDraftWeights
        from .kernel_loader import compile_native_programs

        compile_native_programs(programs, cycles=(1, 2))
        programs.draft_weights = NativeDraftWeights(model, programs.native_metadata[1])
        programs.draft_weights.prepare()
        return programs

    def _create_runner(self, model_id, compiled):
        from .native_pipeline_runner import NativePipelineRunner

        return NativePipelineRunner(
            compiled, self.contract, cache_initialization=self.cache_initialization
        )

    @property
    def uses_request_execution_context(self):
        return True

    def take_cache_publications(self, model):
        runner=self._runners[model.config.model_id]
        proofs=tuple(runner.cache_publications)
        runner.cache_publications.clear()
        return proofs

    def prepare_request_metadata(self, model, registrations):
        from dataclasses import asdict
        from .execution_policy import LowlatRequestContext, decode_request_metadata

        runner=self._runners[model.config.model_id]
        for registration in registrations:
            if registration.model_extension:
                descriptor=decode_request_metadata(registration.model_extension).prefix_resume
                context=runner.request_contexts[registration.request_id]
                runner.request_contexts[registration.request_id]=LowlatRequestContext(**asdict(context),prefix_resume=descriptor)

    def prepare_request_contexts(self, model, contexts):
        self._runners[model.config.model_id].register_contexts(contexts)

    def cache_group_sizes(self, model_id):
        return self._runners[model_id].cache.counts

    def lookup_embeddings(self, model, token_ids):
        return self._runners[model.config.model_id].weights.lookup_embeddings(token_ids)

    def release_finished_requests(self, request_ids):
        for runner in self._runners.values():
            runner.release_finished_requests(request_ids)

    def validate_generate_batch(self, record, batch_size, config):
        validate_sampling(config)

    @property
    def supports_device_decode_embedding(self):
        return True

    @property
    def supports_device_sampling(self):
        return True

    @property
    def max_prefill_batch_size(self):
        return 1

    @property
    def max_prefill_tokens_per_partition(self):
        return self.contract.max_prefill_tokens
