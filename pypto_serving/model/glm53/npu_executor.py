# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile GLM kernels for the serving-owned persistent L3 runner."""

import os
from dataclasses import replace

from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.common.executor.pypto_executor import PyptoExecutor
from pypto_serving.model.glm53.kernel_api import (
    KERNEL_BATCH,
    PREFILL_CHUNK_ALIGNMENT,
    build_glm53_cache_group_specs,
    import_kernel,
    packed_capacity,
    validate_serving_limits,
    validate_topology,
)
from pypto_serving.model.glm53.npu_runner import Glm53ModelRunner
from pypto_serving.model.glm53.task_args import Glm53CompiledKernels
from pypto_serving.model.glm53.weight_loader import load_kernel_weights


def prepare_kernels(
    api,
    device_ids,
    *,
    platform="a2a3",
    pypto_build_dir=None,
    enable_mtp=False,
):
    """Compile compute and cache-maintenance entries through the shared JIT path."""
    compiler = KernelCompiler(
        run_config=build_pypto_run_config(
            platform=platform,
            device_ids=device_ids,
            pypto_build_dir=pypto_build_dir,
        )
    )
    decode_api = import_kernel(
        "decode_fwd", capacity=api.CAPACITY, max_seq_len=api.MAX_SEQ_LEN,
        cache_tokens=api.CACHE_TOKENS, tp_size=api.TP_SIZE, ep_size=api.EP_SIZE, context_limit=None,
    )
    programs = {
        "prefill": compiler.compile("glm53_prefill", api.l3_prefill_fwd),
        "decode": compiler.compile("glm53_decode", decode_api.l3_decode_fwd),
    }
    entries = {"reset": api.l3_reset_request_caches}
    if enable_mtp:
        entries.update(mtp_prepare=api.l3_prefill_mtp_prepare, mtp_head=api.l3_prefill_mtp_head)
        entries.update({key: api.production_program_entry(key) for key in ("dsa_attn", "moe_ffn", "head")})
    programs["_auxiliary"] = {key: compiler.compile(f"glm53_{key}", entry) for key, entry in entries.items()}
    return programs


class Glm53PyptoExecutor(PyptoExecutor):
    @classmethod
    def resolve_runtime_config(cls, runtime, device_ids):
        validate_topology(len(device_ids), len(device_ids), device_ids=device_ids)
        validate_serving_limits(runtime.max_batch_size, runtime.max_seq_len, runtime.max_num_batched_tokens)
        return replace(
            runtime, prefill_chunk_alignment=PREFILL_CHUNK_ALIGNMENT,
            requires_homogeneous_prefill_decode=True,
            kv_cache_groups=build_glm53_cache_group_specs(
                runtime.max_seq_len, runtime.max_num_batched_tokens, tp_size=len(device_ids),
            ),
        )

    @classmethod
    def configure_engine(cls, config):
        runtime = config.resolve_runtime_config()
        devices = config.worker_device_ids()
        parallel = config.parallel_config or ParallelConfig(
            tensor_parallel_size=len(devices), expert_parallel_size=len(devices),
            enable_expert_parallel=True, placement_mode="overlapped", devices=devices,
        )
        validate_topology(parallel.tensor_parallel_size, parallel.expert_parallel_size,
                          parallel.data_parallel_size, devices)
        return replace(
            config, runtime_config=runtime, parallel_config=parallel,
            max_num_running_reqs=min(config.max_num_running_reqs, runtime.max_batch_size),
            max_num_scheduled_tokens=min(config.max_num_scheduled_tokens, runtime.max_num_batched_tokens),
        )

    def cache_group_sizes(self, model_id):
        return dict(self._runners[model_id]._cache_group_num_blocks)

    @property
    def supports_async_decode_prepare_without_embeddings(self):
        return True

    @property
    def max_prefill_batch_size(self):
        return KERNEL_BATCH

    @property
    def max_prefill_requests_per_partition(self):
        return KERNEL_BATCH

    def _compile_model(self, model):
        if model.extra.get("family") != "glm53":
            raise ValueError("Glm53PyptoExecutor requires a GLM-5.3-Flash W8A8 checkpoint")
        # GLM's supported TP and EP axes span the same worker device group.
        ranks = len(self._device_ids)
        validate_topology(ranks, ranks, device_ids=self._device_ids)
        runtime = model.runtime
        validate_serving_limits(runtime.max_batch_size, runtime.max_seq_len, runtime.max_num_batched_tokens)
        options = {
            "capacity": packed_capacity(runtime.max_num_batched_tokens),
            "max_seq_len": runtime.max_seq_len,
            "tp_size": ranks,
            "ep_size": ranks,
        }
        checkpoint_api = import_kernel("utils", **options)
        checkpoint = checkpoint_api.Checkpoint(model.extra["model_dir"])
        api = import_kernel("prefill_fwd", **options)
        # Validate every planned tensor before compilation; preflight stages payloads.
        weights = load_kernel_weights(api, checkpoint, tp_size=ranks, ep_size=ranks)
        programs = prepare_kernels(
            api,
            list(self._device_ids),
            platform=self._platform,
            pypto_build_dir=self._pypto_build_dir,
        )
        return Glm53CompiledKernels(api, programs, checkpoint, weights)

    def _create_runner(self, model_id, compiled):
        if not isinstance(compiled, Glm53CompiledKernels):
            raise TypeError("Glm53PyptoExecutor requires GLM compiled kernels")
        options = {}
        raw_threads = os.environ.get("PYPTO_GLM_HOST_THREADS")
        if raw_threads is not None:
            try:
                threads = int(raw_threads)
            except ValueError as error:
                raise ValueError("PYPTO_GLM_HOST_THREADS must be a positive integer") from error
            if threads <= 0:
                raise ValueError("PYPTO_GLM_HOST_THREADS must be a positive integer")
            options["host_num_threads"] = threads
        return Glm53ModelRunner(
            compiled.api,
            compiled.programs,
            compiled.checkpoint,
            self._device_ids,
            self._platform,
            prepared_weights=compiled.weights,
            **options,
        )

    def lookup_embeddings(self, model, token_ids):
        return self._runners[model.config.model_id].weights.lookup_rows(token_ids)

    def release_finished_requests(self, request_ids):
        for runner in self._runners.values():
            runner.release_finished_requests(request_ids)
