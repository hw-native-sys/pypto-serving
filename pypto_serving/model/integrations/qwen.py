# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pypto_serving.config.launch_options import LaunchOptions
from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.model.integrations.base import IntegrationPlan, ModelIntegration, Speculation


class QwenIntegration(ModelIntegration):
    integration_id = "qwen3_14b"

    def resolve(
        self, options: LaunchOptions, metadata: dict[str, object],
        devices: tuple[int, ...], speculation: Speculation,
    ) -> IntegrationPlan:
        del metadata
        if speculation.method or speculation.num_tokens:
            raise ValueError("--speculative-config/--num-speculative-tokens is only supported for DeepSeek V4")
        placement = ParallelConfig(
            data_parallel_size=options.parallel.data_parallel_size,
            tensor_parallel_size=options.parallel.tensor_parallel_size,
            expert_parallel_size=options.parallel.expert_parallel_size,
            enable_expert_parallel=options.parallel.expert_parallel_size > 1,
            devices=devices,
            data_parallel_routing=options.parallel.data_parallel_routing,
            placement_mode="replica",
        )
        return IntegrationPlan(self.integration_id, placement,
                               enable_prefix_cache=options.scheduler.enable_prefix_caching)

    def executor_class(self) -> type:
        from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor

        return Qwen314BPyptoExecutor
