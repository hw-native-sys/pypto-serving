# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Typed, unresolved launch inputs shared by CLI, resolver, and model integrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOptions:
    path: str
    served_name: str | None
    dtype: str
    kv_cache_dtype: str


@dataclass(frozen=True)
class DeviceOptions:
    backend: str
    platform: str
    default_device: int
    devices: str | None


@dataclass(frozen=True)
class ParallelOptions:
    data_parallel_size: int
    tensor_parallel_size: int
    expert_parallel_size: int
    data_parallel_routing: str


@dataclass(frozen=True)
class SchedulerOptions:
    max_model_len: int
    block_size: int
    npu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    long_prefill_token_threshold: int
    enable_prefix_caching: bool
    enable_chunked_prefill: bool


@dataclass(frozen=True)
class RuntimeOptions:
    ring_dep_pool: int | list[int] | None
    ring_task_window: int | list[int] | None
    ring_heap: int | list[int] | None


@dataclass(frozen=True)
class FeatureOptions:
    speculative_config: dict[str, object] | None
    num_speculative_tokens: int | None


@dataclass(frozen=True)
class ProfileOptions:
    enabled: bool
    output: str | None
    levels: str | None


@dataclass(frozen=True)
class ServeOptions:
    host: str
    port: int


@dataclass(frozen=True)
class PromptOptions:
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class LaunchOptions:
    model: ModelOptions
    device: DeviceOptions
    parallel: ParallelOptions
    scheduler: SchedulerOptions
    runtime: RuntimeOptions
    features: FeatureOptions
    profile: ProfileOptions
    serve: ServeOptions
    generate: PromptOptions | None
    generate_config: dict[str, object] | None
    show_startup_logs: bool
