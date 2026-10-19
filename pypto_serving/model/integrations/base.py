# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Contracts owned by one supported model execution path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pypto_serving.config.launch_options import LaunchOptions
from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.serving.sampling.capabilities import ModelCapabilities


@dataclass(frozen=True)
class Speculation:
    method: str = ""
    num_tokens: int = 0
    explicit_tokens: bool = False


@dataclass(frozen=True)
class IntegrationPlan:
    integration_id: str
    parallel: ParallelConfig
    runtime_values: dict[str, object] = field(default_factory=dict)
    executor_kwargs: dict[str, object] = field(default_factory=dict)
    enable_prefix_cache: bool = True
    capabilities: ModelCapabilities = ModelCapabilities()


class ModelIntegration(Protocol):
    integration_id: str

    def resolve(
        self,
        options: LaunchOptions,
        metadata: dict[str, object],
        devices: tuple[int, ...],
        speculation: Speculation,
    ) -> IntegrationPlan:
        """Validate the model-specific contract and derive its runtime values."""

    def executor_class(self) -> type:
        """Import the executor only in the worker process."""
