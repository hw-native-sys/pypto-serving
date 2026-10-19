# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resolve CLI or Python inputs into one launch configuration."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from pypto_serving.config.launch_options import LaunchOptions
from pypto_serving.config.parallel import parse_device_ids
from pypto_serving.config.types import GenerateConfig, RuntimeConfig
from pypto_serving.model.integrations import (
    resolve_speculation, select_integration,
)
from pypto_serving.model.metadata import read_model_metadata
from pypto_serving.serving.engine.async_engine import EngineConfig
from pypto_serving.tools.profile import create_profile_config


@dataclass(frozen=True)
class ResolvedLaunchConfig:
    engine: EngineConfig
    generation: GenerateConfig
    serve_host: str
    serve_port: int
    prompts: tuple[str, ...] | None


def resolve_launch_options(options: LaunchOptions) -> ResolvedLaunchConfig:
    if options.device.backend != "npu":
        raise ValueError(f"Only NPU backend is supported, got: {options.device.backend}")
    model_dir = Path(options.model.path).expanduser().resolve()
    metadata = read_model_metadata(model_dir)
    speculation = resolve_speculation(options.features)
    integration = select_integration(metadata, speculation)
    devices = parse_device_ids(options.device.devices, default_device=options.device.default_device)
    plan = integration.resolve(options, metadata, devices, speculation)

    scheduler = options.scheduler
    kv_dtype = options.model.dtype if options.model.kv_cache_dtype == "auto" else options.model.kv_cache_dtype
    runtime_values = {
        "page_size": scheduler.block_size,
        "max_batch_size": scheduler.max_num_seqs,
        "max_seq_len": scheduler.max_model_len,
        "device": "cpu",
        "kv_dtype": kv_dtype,
        "weight_dtype": options.model.dtype,
        "npu_memory_utilization": scheduler.npu_memory_utilization,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "ring_dep_pool": options.runtime.ring_dep_pool,
        "ring_task_window": options.runtime.ring_task_window,
        "ring_heap": options.runtime.ring_heap,
    }
    runtime = RuntimeConfig(**(runtime_values | plan.runtime_values))
    profile = options.profile
    if not profile.enabled and (profile.output is not None or profile.levels is not None):
        raise ValueError("--profile-output and --profile-level require --profile")
    profile_config = create_profile_config(
        enabled=profile.enabled,
        output=Path(profile.output or "./profile_out").expanduser().resolve(),
        levels=profile.levels or "e2e,kernel",
    )
    first_group = plan.parallel.replica_device_groups[0]
    engine = EngineConfig(
        model_id=options.model.served_name or model_dir.name,
        model_dir=str(model_dir),
        integration_id=plan.integration_id,
        capabilities=plan.capabilities,
        platform=options.device.platform,
        device_id=first_group[0],
        device_ids=first_group if plan.parallel.num_replicas == 1 else (),
        parallel_config=plan.parallel,
        executor_kwargs=plan.executor_kwargs,
        runtime_config=runtime,
        profile_config=profile_config,
        max_num_running_reqs=scheduler.max_num_seqs,
        max_num_scheduled_tokens=scheduler.max_num_batched_tokens,
        long_prefill_token_threshold=scheduler.long_prefill_token_threshold,
        enable_prefix_cache=plan.enable_prefix_cache,
        enable_chunk_prefill=scheduler.enable_chunked_prefill,
    )
    generation = resolve_generate_config(options.generate_config)
    if options.generate is not None and generation.stream and len(options.generate.prompts) > 1:
        raise ValueError("--generate-config stream=true requires a single --prompt")
    prompts = options.generate.prompts if options.generate is not None else None
    return ResolvedLaunchConfig(engine, generation, options.serve.host, options.serve.port, prompts)


def resolve_generate_config(data: dict[str, object] | None) -> GenerateConfig:
    if data is None:
        return GenerateConfig()
    if not isinstance(data, dict):
        raise ValueError("--generate-config must be a JSON object")
    if not all(isinstance(name, str) for name in data):
        raise ValueError("--generate-config field names must be strings")
    valid_fields = {field.name for field in dataclasses.fields(GenerateConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(
            f"--generate-config has unknown fields: {sorted(unknown)}; "
            f"valid fields: {sorted(valid_fields)}"
        )

    values = dict(data)
    if "stop" in values:
        if not isinstance(values["stop"], list):
            raise ValueError(f"--generate-config field 'stop' must be an array; got {values['stop']!r}")
        values["stop"] = tuple(values["stop"])
    try:
        return GenerateConfig(**values)
    except ValueError as exc:
        raise ValueError(f"--generate-config: {exc}") from exc
